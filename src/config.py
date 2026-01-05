"""Project-wide configuration: paths, hyperparameters, and constants."""
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class CFG:
    """Paths, hyperparameters, and constants for the DINOv3 training pipeline."""

    # paths — default to Colab's mount point; override with CSIRO_DATA_PATH
    # for local runs or other notebook environments
    DATA_PATH: Path = Path(os.environ.get('CSIRO_DATA_PATH', '/content/'))
    TRAIN_CSV: str = 'train.csv'
    TEST_CSV: str = 'test.csv'
    TRAIN_IMAGES_DIR: str = 'train_images'
    TEST_IMAGES_DIR: str = 'test_images'

    # DINOv3 backbone (Hugging Face hub identifier)
    # vits16 = 22 million parameters
    # vitb16 = 86 million parameters
    # vitl16 = 300 million parameters
    DINO_MODEL_NAME: str = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
    DINO_LOCAL_ONLY: bool = False  # set True for Kaggle offline submissions

    # training
    SEED: int = 42
    N_FOLDS: int = 3  # matches the 1st place solution's fold count
    BATCH_SIZE_DINO: int = 8
    NUM_WORKERS: int = 2
    IMG_SIZE: int = 640  # must be a multiple of 16 for patch16 Vision Transformer
    EPOCHS: int = 5
    FAST_DEBUG: bool = False
    DEBUG_SAMPLES: int = 50

    # two-stage training: epochs with the DINOv3 backbone frozen before
    # unfreezing for fine-tuning; clamped to leave >=1 fine-tune epoch
    WARMUP_EPOCHS: int = 2

    # interval-classification heads: quantile bins per target and the
    # auxiliary classification loss's weight relative to regression
    N_BINS: int = 7
    CLS_LOSS_WEIGHT: float = 0.3


cfg = CFG()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# biomass target columns in the order expected by the competition
ALL_TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

# per-target weights used by the competition R2 metric
# Dry_Total_g carries the most weight (0.5), GDM_g next (0.2), leaf components 0.1 each
R2_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float32)

# ImageNet channel-wise mean and standard deviation for normalizing pretrained Vision Transformer inputs
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed=42):
    """Set random seeds across all libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(cfg.SEED)
