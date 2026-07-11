"""Competition R2 metric and post-hoc mass-balance enforcement."""
import numpy as np

from src.config import ALL_TARGETS, R2_WEIGHTS


def weighted_r2_global(y_true, y_pred, weights=R2_WEIGHTS):
    # competition metric: R2 computed on the flattened (samples x targets)
    # array with per-target weights tiled across it, not averaged per-target
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    tiled_weights = np.tile(weights, len(y_true) // len(weights))

    weighted_mean = np.sum(tiled_weights * y_true) / (np.sum(tiled_weights) + 1e-10)
    sum_squared_residuals = np.sum(tiled_weights * (y_true - y_pred) ** 2)
    sum_squared_total = np.sum(tiled_weights * (y_true - weighted_mean) ** 2)

    if sum_squared_total > 1e-10:
        return 1.0 - sum_squared_residuals / (sum_squared_total + 1e-10)
    return 0.0


def enforce_mass_balance(dataframe):
    # clip to non-negative, then rederive GDM/Total from their leaf
    # components so GDM = Green + Clover and Total = GDM + Dead always hold
    out = dataframe.copy()
    for column in ALL_TARGETS:
        if column in out.columns:
            out[column] = out[column].clip(lower=0.0)

    out['GDM_g'] = out['Dry_Green_g'] + out['Dry_Clover_g']
    out['Dry_Total_g'] = out['GDM_g'] + out['Dry_Dead_g']
    out['GDM_g'] = out['GDM_g'].clip(lower=0.0)
    out['Dry_Total_g'] = out['Dry_Total_g'].clip(lower=0.0)
    return out
