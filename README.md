# CSIRO Image-to-Biomass

Predict pasture biomass from field photographs. Reimplements the [1st-place solution](https://www.kaggle.com/competitions/csiro-biomass/writeups/1st-place-solution) for the Kaggle CSIRO Biomass competition.

## Task

Given a photograph of a pasture plot, predict five dry-weight targets (grams):

| Target | Weight in metric |
|--------|-----------------|
| `Dry_Total_g` | 0.5 |
| `GDM_g` (Green Dry Matter) | 0.2 |
| `Dry_Green_g` | 0.1 |
| `Dry_Clover_g` | 0.1 |
| `Dry_Dead_g` | 0.1 |

Evaluation: weighted R-squared across all five targets.

## Pipeline

```
Raw Image
    |
Clean (crop metadata strip, inpaint date-stamps)
    |
Split into left and right halves
    |
Shared DINOv3 backbone (both halves)
    |
Cross-view self-attention fusion
    |
5 regression heads  +  5 auxiliary interval-classification heads
    |
Mass balance enforcement (clip negatives, recompute GDM/Total)
    |
Final predictions
```

## Model

| Component | Detail |
|-----------|--------|
| Backbone | DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`, ~300M params) |
| Fusion | CrossViewAttention — multi-head self-attention over left/right encodings |
| Heads | 5 independent regression (Softplus) + 5 auxiliary bin classifiers |
| Training | Two-stage (frozen warmup -> full fine-tune), differential lr, cosine annealing, MixUp, AMP, EMA |
| CV | Group-aware K-fold (state + date), no site-visit leakage |

## Project Structure

```
csiro-image2biomass/
  train.py                        Entry point: full pipeline end to end
  src/
    config.py                     Paths, hyperparameters, constants, seed
    metrics.py                    Weighted R2 metric, mass balance enforcement
    data/
      loading.py                  CSV loading, pivot, path resolution
      preprocessing.py            Image cleaning, left/right split
      cv.py                       Group-aware cross-validation split keys
    models/
      dataset.py                  BiomassDataset
      transforms.py               Albumentations augmentation pipelines
      camera_distance.py          Camera-distance simulation transform
      attention_fusion.py         CrossViewAttention (left/right fusion)
      borders.py                  Quantile bin edges for interval classification
      losses.py                   Regression + classification combined loss
      dinov3.py                   DINOv3Regressor
      engine.py                   K-fold and full-data training loops
  scripts/
    setup_colab.sh                Colab environment bootstrap
    download_weights.py           Predownload weights for offline Kaggle
```

## Quick Start

Built for Google Colab with GPU. Paths default to `/content/`; override with `CSIRO_DATA_PATH` or `--data-path`.

```bash
bash scripts/setup_colab.sh       # install deps, download competition data
pip install -r requirements.txt
python train.py                   # full run
python train.py --fast-debug --debug-samples 50   # smoke test
```

Run `python train.py --help` for all flags (`--n-folds`, `--epochs`, `--output-dir`, etc.).

## Outputs

Saved to `--output-dir` (default `/content/models`):

| File | Description |
|------|-------------|
| `dinov3_regressor.pth` | Full checkpoint (EMA weights) |
| `dinov3_heads_only.pth` | Fusion + heads only (smaller, requires HF backbone) |
| `config.json` | Metadata including fitted bin edges for inference |

## Key Design Decisions

**Cross-view fusion** — left and right image halves encode through a shared backbone, then interact via self-attention before regression. Captures spatial relationships across the full plot.

**Auxiliary interval classifiers** — each target gets a quantile-bin classifier trained jointly with regression. Complementary signal that regularizes the regression heads (adapted from UEPNet crowd counting).

**Mass balance post-processing** — GDM and Dry_Total are recomputed from components after inference. The architecture doesn't enforce physical consistency, so this is the only place it's guaranteed.

**Group-aware CV** — folds split by state + sampling date so photos from the same site visit can't leak across train and validation.

## Acknowledgments

Reimplements the 1st-place solution:

> Baiph, HZM, TheoQiu, zxc123cc. *1st Place Solution.* Kaggle, 2026.

Interval-classification head adapted from:

> Wang et al. *Uniformity in Heterogeneity: Diving Deep into Count Interval Partition for Crowd Counting.* ICCV, 2021.
