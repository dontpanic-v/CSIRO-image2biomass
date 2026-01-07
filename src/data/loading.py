"""CSV loading, long-to-wide pivot, and image path resolution."""
import os

import pandas as pd

from src.config import cfg, ALL_TARGETS


def load_dataframes():
    # train.csv is long format: one row per (image, target) pair. Pivot to
    # wide so each image is one row with all five targets as columns.
    train_long = pd.read_csv(cfg.DATA_PATH / cfg.TRAIN_CSV)
    test_long = pd.read_csv(cfg.DATA_PATH / cfg.TEST_CSV)

    candidate_meta = [
        'image_path', 'Sampling_Date', 'State',
        'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm',
    ]
    meta_cols = [column for column in candidate_meta if column in train_long.columns]

    # the target value and target name columns vary across competition dataset versions
    target_value_column = 'target' if 'target' in train_long.columns else 'target_value'
    target_name_column = 'target_name' if 'target_name' in train_long.columns else 'component'

    train_wide = train_long.pivot_table(
        index=meta_cols,
        columns=target_name_column,
        values=target_value_column,
        aggfunc='mean',
    ).reset_index()

    for target in ALL_TARGETS:
        if target not in train_wide.columns:
            print(f'WARNING {target} missing from pivot, filling with zeros')
            train_wide[target] = 0.0

    return train_wide, test_long


def resolve_path(relative_path, is_train=True):
    relative_path = str(relative_path)
    if relative_path.startswith('train') or relative_path.startswith('test'):
        return str(cfg.DATA_PATH / relative_path)
    subdirectory = cfg.TRAIN_IMAGES_DIR if is_train else cfg.TEST_IMAGES_DIR
    return str(cfg.DATA_PATH / subdirectory / relative_path)


def attach_absolute_paths(train_wide):
    train_wide['abs_path'] = train_wide['image_path'].apply(
        lambda path: resolve_path(path, is_train=True)
    )

    # some CSV versions omit the train/test prefix resolve_path expects — probe
    # the first row and fall back to a plain DATA_PATH join if it doesn't exist
    probe = train_wide['abs_path'].iloc[0]
    if not os.path.exists(probe):
        print(f'path not found: {probe} — falling back to direct join')
        train_wide['abs_path'] = train_wide['image_path'].apply(
            lambda path: str(cfg.DATA_PATH / path)
        )

    return train_wide
