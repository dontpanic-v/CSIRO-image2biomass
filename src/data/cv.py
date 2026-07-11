"""Group-aware cross-validation fold keys (state + sampling date)."""
import numpy as np


def make_cv_groups(train_wide):
    # groups photos taken in the same state on the same sampling date, so a
    # K-fold split can't leak the same plot/date across train and validation
    # — adapted from the CSIRO-biomass 1st place solution's fold strategy
    if 'State' in train_wide.columns and 'Sampling_Date' in train_wide.columns:
        return (train_wide['State'].astype(str) + '_' + train_wide['Sampling_Date'].astype(str)).values
    return np.arange(len(train_wide))
