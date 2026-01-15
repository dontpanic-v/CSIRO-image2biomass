"""Albumentations train/validation augmentation pipelines."""
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import IMAGENET_MEAN, IMAGENET_STD
from src.models.camera_distance import SimulateCameraDistance


def get_train_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        SimulateCameraDistance(scale_range=(0.85, 1.0), p=0.2),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_validation_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
