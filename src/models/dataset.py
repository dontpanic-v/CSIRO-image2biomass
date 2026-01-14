"""PyTorch Dataset yielding split left/right image halves and biomass targets."""
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import ALL_TARGETS
from src.data import load_and_preprocess_image, split_left_right


class BiomassDataset(Dataset):
    """Yields (left_half, right_half, target_vector) — the image is split
    vertically so DINOv3Regressor can encode both halves and fuse them.
    """

    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.paths = self.dataframe['abs_path'].values
        self.targets = self.dataframe[ALL_TARGETS].values.astype(np.float32)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image = load_and_preprocess_image(self.paths[index], clean=True)
        if image is None:
            image = np.zeros((900, 2000, 3), dtype=np.uint8)

        left_half, right_half = split_left_right(image)
        left_half = self.transform(image=left_half)['image']
        right_half = self.transform(image=right_half)['image']
        return left_half, right_half, torch.from_numpy(self.targets[index])
