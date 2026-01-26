"""
End-to-end training pipeline for the CSIRO image-to-biomass competition.

A single DINOv3 model (cross-view attention fusion, five independent
regression heads, auxiliary interval-classification heads) reimplementing
the competition's 1st place solution — see README.md for the pipeline
diagram and the reasoning behind each stage.

Usage:
    python train.py
    python train.py --fast-debug --debug-samples 50
    python train.py --n-folds 3 --epochs 8 --data-path /content
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import cfg, DEVICE, ALL_TARGETS, R2_WEIGHTS
from src.data import load_dataframes, attach_absolute_paths
from src.metrics import weighted_r2_global, enforce_mass_balance
from src.models import train_dino_oof, train_dino_full


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-path', type=Path, default=None, help='overrides CSIRO_DATA_PATH / cfg.DATA_PATH')
    parser.add_argument('--output-dir', type=Path, default=Path('/content/models'), help='where trained artifacts are saved')
    parser.add_argument('--n-folds', type=int, default=None, help='overrides cfg.N_FOLDS')
    parser.add_argument('--epochs', type=int, default=None, help='overrides cfg.EPOCHS')
    parser.add_argument('--fast-debug', action='store_true', help='overrides cfg.FAST_DEBUG: one epoch, subsampled data')
    parser.add_argument('--debug-samples', type=int, default=None, help='overrides cfg.DEBUG_SAMPLES')
    return parser.parse_args()


def apply_overrides(args):
    if args.data_path is not None:
        cfg.DATA_PATH = args.data_path
    if args.n_folds is not None:
        cfg.N_FOLDS = args.n_folds
    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    if args.fast_debug:
        cfg.FAST_DEBUG = True
    if args.debug_samples is not None:
        cfg.DEBUG_SAMPLES = args.debug_samples


def load_data():
    train_wide, test_long = load_dataframes()
    train_wide = attach_absolute_paths(train_wide)

    if cfg.FAST_DEBUG:
        train_wide = train_wide.sample(
            min(cfg.DEBUG_SAMPLES, len(train_wide)), random_state=cfg.SEED
        ).reset_index(drop=True)

    print(f'train_wide: {train_wide.shape}')
    print(f'test_long : {test_long.shape}')
    return train_wide, test_long


def save_artifacts(output_dir, ema_model, borders):
    output_dir.mkdir(parents=True, exist_ok=True)

    print('[1/2] DINOv3 checkpoint')
    source_model = ema_model.module
    hidden_dimension = getattr(source_model, 'hidden_dim', 768)

    torch.save({
        'model_state_dict': source_model.state_dict(),
        'hidden_dim': hidden_dimension,
        'config': {'model_name': cfg.DINO_MODEL_NAME, 'img_size': cfg.IMG_SIZE},
    }, output_dir / 'dinov3_regressor.pth')

    torch.save({
        'fusion':                source_model.fusion.state_dict(),
        'regression_heads':      source_model.regression_heads.state_dict(),
        'classification_heads':  source_model.classification_heads.state_dict(),
        'hidden_dim': hidden_dimension,
        'borders': {target: values.tolist() for target, values in borders.items()},
    }, output_dir / 'dinov3_heads_only.pth')

    print('[2/2] configuration')
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({
            'ALL_TARGETS': ALL_TARGETS,
            'R2_WEIGHTS': R2_WEIGHTS.tolist(),
            'DINO_MODEL_NAME': cfg.DINO_MODEL_NAME,
            'IMG_SIZE': cfg.IMG_SIZE,
            'SEED': cfg.SEED,
            'N_BINS': cfg.N_BINS,
            'CLS_LOSS_WEIGHT': cfg.CLS_LOSS_WEIGHT,
            'target_borders': {target: values.tolist() for target, values in borders.items()},
        }, f, indent=2)

    print()
    total_megabytes = 0.0
    for entry in sorted(output_dir.iterdir()):
        size_megabytes = entry.stat().st_size / (1024 * 1024)
        total_megabytes += size_megabytes
        print(f'  {entry.name}: {size_megabytes:.2f} MB')
    print(f'  TOTAL: {total_megabytes:.2f} MB')


def main():
    args = parse_args()
    apply_overrides(args)

    print('DEVICE:', DEVICE)
    train_wide, _ = load_data()
    ground_truth = train_wide[ALL_TARGETS].values.astype(np.float32)

    oof_predictions = train_dino_oof(train_wide, n_folds=cfg.N_FOLDS)
    print(f'DINOv3 out-of-fold R2 (raw): {weighted_r2_global(ground_truth, oof_predictions):.4f}')

    # the architecture no longer guarantees mass balance, so check here
    # whether enforcing it helps or hurts the actual score
    balanced_oof = enforce_mass_balance(pd.DataFrame(oof_predictions, columns=ALL_TARGETS))[ALL_TARGETS].values
    print(f'DINOv3 out-of-fold R2 (mass-balanced): {weighted_r2_global(ground_truth, balanced_oof):.4f}')

    _, ema_model, borders = train_dino_full(train_wide)

    save_artifacts(args.output_dir, ema_model, borders)


if __name__ == '__main__':
    main()
