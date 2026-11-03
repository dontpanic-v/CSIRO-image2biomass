"""DINOv3 training loops: per-fold out-of-fold training and full-data retraining.

Both entry points share one training recipe (two-stage freeze/unfreeze,
differential LR, cosine annealing, MixUp, AMP, EMA), so the out-of-fold
score measures the same recipe that produces the submission model.
"""
import gc
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel
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

MIXUP_PROB = 0.30
MIXUP_ALPHA = 0.4


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


def _epoch_schedule():
    epochs = cfg.EPOCHS if not cfg.FAST_DEBUG else 1
    # clamped so at least one epoch fine-tunes the unfrozen backbone
    warmup_epochs = min(cfg.WARMUP_EPOCHS, max(0, epochs - 1))
    return epochs, warmup_epochs


def _make_ema(model):
    # exponential moving average of the weights, updated every optimizer step
    decay = cfg.EMA_DECAY

    def ema_avg(averaged_param, current_param, num_averaged):
        return decay * averaged_param + (1.0 - decay) * current_param

    return AveragedModel(model, avg_fn=ema_avg)


def _train_model(dataframe_train, borders, desc):
    # trains one DINOv3Regressor on dataframe_train; returns the live model
    # and its EMA shadow copy
    dataset_train = BiomassDataset(dataframe_train, get_train_transforms(cfg.IMG_SIZE))
    loader_train = DataLoader(
        dataset_train, batch_size=cfg.BATCH_SIZE_DINO, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    if len(loader_train) == 0:
        raise ValueError(
            f'{len(dataset_train)} training samples is fewer than one batch '
            f'(BATCH_SIZE_DINO={cfg.BATCH_SIZE_DINO}); increase --debug-samples'
        )

    epochs, warmup_epochs = _epoch_schedule()

    # stage 1: start with the backbone frozen so the randomly initialized
    # heads warm up before backprop reaches DINOv3
    model = DINOv3Regressor(
        cfg.DINO_MODEL_NAME, borders,
        local_only=cfg.DINO_LOCAL_ONLY,
        freeze_backbone=warmup_epochs > 0,
    ).to(DEVICE)

    # slow LR for the pretrained backbone, fast LR for the randomly
    # initialized fusion + regression/classification heads
    optimizer = optim.AdamW([
        {'params': list(model.backbone.parameters()), 'lr': 1e-5},
        {'params': _head_parameters(model), 'lr': 2e-4},
    ], weight_decay=1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader_train),
    )
    criterion = WeightedBiomassLoss(cls_weight=cfg.CLS_LOSS_WEIGHT)

    use_amp = (DEVICE == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema_model = _make_ema(model)

    for epoch in range(epochs):
        # stage 2: unfreeze the backbone for full fine-tuning
        if epoch == warmup_epochs and warmup_epochs > 0:
            for param in model.backbone.parameters():
                param.requires_grad = True

        model.train()
        running_loss = 0.0
        progress_bar = tqdm(loader_train, desc=f'{desc} epoch {epoch + 1}/{epochs}')

        for left, right, batch_targets in progress_bar:
            left = left.to(DEVICE, non_blocking=True)
            right = right.to(DEVICE, non_blocking=True)
            batch_targets = batch_targets.to(DEVICE, non_blocking=True)

            # MixUp: blend pairs of samples within the batch, before binning
            # so classification labels stay derived from the same values
            # the regression heads are trained against
            if random.random() < MIXUP_PROB and left.size(0) > 1:
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                perm = torch.randperm(left.size(0), device=DEVICE)
                left = lam * left + (1 - lam) * left[perm]
                right = lam * right + (1 - lam) * right[perm]
                batch_targets = lam * batch_targets + (1 - lam) * batch_targets[perm]

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                predictions, cls_logits = model(left, right)
                loss = criterion(predictions, batch_targets, cls_logits, _cls_labels(batch_targets, borders))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema_model.update_parameters(model)

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')

        print(f'  average loss: {running_loss / len(loader_train):.4f}')

    del loader_train, dataset_train, optimizer, scheduler, scaler
    return model, ema_model


def _predict(network, dataframe):
    dataset = BiomassDataset(dataframe, get_validation_transforms(cfg.IMG_SIZE))
    loader = DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE_DINO, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False,
    )
    use_amp = (DEVICE == 'cuda')

    network.eval()
    predictions = []
    with torch.no_grad():
        for left, right, _ in tqdm(loader, desc='validation'):
            left = left.to(DEVICE, non_blocking=True)
            right = right.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                batch_predictions, _ = network(left, right)
            predictions.append(batch_predictions.float().cpu().numpy())
    return np.vstack(predictions)


def train_dino_oof(dataframe, n_folds=5):
    # grouped by state+date (see src/data/cv.py) so the same plot/date can't
    # leak across train and validation folds
    groups = make_cv_groups(dataframe)
    kfold = GroupKFold(n_splits=n_folds)
    oof_predictions = np.zeros((len(dataframe), len(ALL_TARGETS)), dtype=np.float32)

    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataframe, groups=groups)):
        print(f'\nDINOv3 fold {fold + 1}/{n_folds}')

        dataframe_train = dataframe.iloc[train_indices].reset_index(drop=True)
        dataframe_val = dataframe.iloc[val_indices].reset_index(drop=True)

        # bin edges computed from this fold's train split only, never validation
        borders = compute_target_borders(dataframe_train, n_bins=cfg.N_BINS)

        model, ema_model = _train_model(dataframe_train, borders, desc=f'fold {fold + 1}')

        # scored with the EMA weights, the same weights the submission model saves
        oof_predictions[val_indices] = np.maximum(0, _predict(ema_model, dataframe_val))

        del model, ema_model
        torch.cuda.empty_cache()
        gc.collect()

        fold_r2 = weighted_r2_global(
            dataframe_val[ALL_TARGETS].values,
            oof_predictions[val_indices],
        )
        print(f'fold {fold + 1} R2: {fold_r2:.4f}')

    return oof_predictions


def train_dino_full(dataframe):
    # retrain on 100% of the data (no held-out fold) for the submission model.
    # No validation split to protect, so borders are computed from all of it.
    borders = compute_target_borders(dataframe, n_bins=cfg.N_BINS)
    model, ema_model = _train_model(dataframe, borders, desc='full-train')
    return model, ema_model, borders
