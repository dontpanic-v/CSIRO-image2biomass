"""DINOv3 training loops: per-fold out-of-fold training and full-data retraining."""
import gc

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sklearn.model_selection import GroupKFold

from src.config import cfg, DEVICE, ALL_TARGETS
from src.data import make_cv_groups
from src.metrics import weighted_r2_global
from src.models.borders import compute_target_borders, bin_index
from src.models.losses import WeightedBiomassLoss
from src.models.dataset import BiomassDataset
from src.models.dinov3 import DINOv3Regressor
from src.models.transforms import get_train_transforms, get_validation_transforms


def _head_parameters(network):
    return (
        list(network.fusion.parameters())
        + list(network.regression_heads.parameters())
        + list(network.classification_heads.parameters())
    )


def _cls_labels(targets, borders):
    return {
        target: bin_index(targets[:, index], borders[target])
        for index, target in enumerate(ALL_TARGETS)
    }


def train_dino_oof(dataframe, n_folds=5):
    # grouped by state+date (see src/data/cv.py) so the same plot/date can't
    # leak across train and validation folds
    groups = make_cv_groups(dataframe)
    kfold = GroupKFold(n_splits=n_folds)
    oof_predictions = np.zeros((len(dataframe), len(ALL_TARGETS)), dtype=np.float32)
    epochs = cfg.EPOCHS if not cfg.FAST_DEBUG else 1
    warmup_epochs = min(cfg.WARMUP_EPOCHS, max(0, epochs - 1))

    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataframe, groups=groups)):
        print(f'\nDINOv3 fold {fold + 1}/{n_folds}')

        dataframe_train = dataframe.iloc[train_indices].reset_index(drop=True)
        dataframe_val = dataframe.iloc[val_indices].reset_index(drop=True)

        # bin edges computed from this fold's train split only, never validation
        borders = compute_target_borders(dataframe_train, n_bins=cfg.N_BINS)

        dataset_train = BiomassDataset(dataframe_train, get_train_transforms(cfg.IMG_SIZE))
        dataset_val = BiomassDataset(dataframe_val, get_validation_transforms(cfg.IMG_SIZE))

        loader_train = DataLoader(
            dataset_train, batch_size=cfg.BATCH_SIZE_DINO, shuffle=True,
            num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
        )
        loader_val = DataLoader(
            dataset_val, batch_size=cfg.BATCH_SIZE_DINO, shuffle=False,
            num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False,
        )

        # stage 1: start with the backbone frozen so the
        # randomly initialized heads warm up before backprop reaches DINOv3
        network = DINOv3Regressor(
            cfg.DINO_MODEL_NAME, borders,
            local_only=cfg.DINO_LOCAL_ONLY,
            freeze_backbone=warmup_epochs > 0,
        ).to(DEVICE)

        # slow LR for the pretrained backbone, fast LR for the randomly
        # initialized fusion + regression/classification heads
        backbone_params = list(network.backbone.parameters())
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': _head_parameters(network), 'lr': 2e-4},
        ], weight_decay=1e-2)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * len(loader_train),
        )
        criterion = WeightedBiomassLoss(cls_weight=cfg.CLS_LOSS_WEIGHT)

        for epoch in range(epochs):
            if epoch == warmup_epochs and warmup_epochs > 0:
                for param in network.backbone.parameters():
                    param.requires_grad = True

            network.train()
            running_loss = 0.0
            progress_bar = tqdm(loader_train, desc=f'epoch {epoch + 1}/{epochs}')

            for left, right, targets in progress_bar:
                left = left.to(DEVICE, non_blocking=True)
                right = right.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                predictions, cls_logits = network(left, right)
                loss = criterion(predictions, targets, cls_logits, _cls_labels(targets, borders))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=f'{loss.item():.4f}')

            print(f'  average loss: {running_loss / len(loader_train):.4f}')

        network.eval()
        val_predictions = []
        with torch.no_grad():
            for left, right, _ in tqdm(loader_val, desc='validation'):
                left = left.to(DEVICE, non_blocking=True)
                right = right.to(DEVICE, non_blocking=True)
                predictions, _ = network(left, right)
                val_predictions.append(predictions.float().cpu().numpy())

        oof_predictions[val_indices] = np.maximum(0, np.vstack(val_predictions))

        del network, optimizer, scheduler, loader_train, loader_val
        del dataset_train, dataset_val
        torch.cuda.empty_cache()
        gc.collect()

        fold_r2 = weighted_r2_global(
            dataframe.iloc[val_indices][ALL_TARGETS].values,
            oof_predictions[val_indices],
        )
        print(f'fold {fold + 1} R2: {fold_r2:.4f}')

    return oof_predictions

