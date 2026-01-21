"""Quantile bin-edge computation for the auxiliary interval-classification heads."""
import numpy as np
import torch

from src.config import ALL_TARGETS


def compute_target_borders(dataframe, n_bins=7):
    # quantile bin edges per target, derived from this split's own labels
    # rather than hardcoded — inspired by the 1st place solution's UEPNet-
    # style interval classification head, but a fixed
    # BORDERS_DICT from someone else's fold would be meaningless on ours
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    return {
        target: np.quantile(dataframe[target].values, quantiles).astype(np.float32)
        for target in ALL_TARGETS
    }


def bin_index(values, borders):
    # right-open bins: values <= borders[0] -> class 0, etc.
    return torch.bucketize(values, torch.as_tensor(borders, device=values.device))
