# CSIRO Image-to-Biomass
## Competition Task

Given a photograph of a pasture plot, predict five biomass targets (in grams):

- `Dry_Green_g` - dry weight of green grass
- `Dry_Clover_g` - dry weight of clover
- `Dry_Dead_g` - dry weight of dead material
- `GDM_g` - Green Dry Matter (= Dry_Green + Dry_Clover)
- `Dry_Total_g` - total dry biomass (= GDM + Dry_Dead)

Evaluation metric: **Weighted R-squared** across all five targets, with Dry_Total weighted 0.5, GDM weighted 0.2, and each leaf component weighted 0.1.

## Pipeline

```
Raw Image
    |
    v
Clean (crop bottom strip, inpaint orange date-stamps)
    |
    v
Split into left and right halves
    |
    v
Shared DINOv3 backbone (both halves)
    |
    v
Cross-view self-attention fusion
    |
    v
Five independent regression heads  +  five interval-classification heads
(green, dead, clover, GDM, total)     (auxiliary training signal, discarded at inference)
    |
    v
Mass Balance Enforcement
(clip negatives, recompute GDM/Total from components)
    |
    v
Final Predictions
```

## Repository Structure

```
csiro-image2biomass/
    README.md
    requirements.txt
    train.py                  - entry point: runs the full pipeline end to end
    src/
        __init__.py
        config.py              - paths, hyperparameters, constants, seed
        data/
            loading.py          - CSV loading, pivot, path resolution
            preprocessing.py    - image cleaning, left/right split
            cv.py                 - group-aware cross-validation split keys
        models/
            dataset.py            - BiomassDataset
            transforms.py          - albumentations augmentation pipelines
            camera_distance.py      - camera-distance simulation transform
            attention_fusion.py      - CrossViewAttention (left/right fusion)
            borders.py                - quantile bin edges for interval classification
            losses.py                  - regression + classification combined loss
            dinov3.py                   - DINOv3Regressor
            engine.py                    - DINOv3 two-stage, K-fold and full-data training
        metrics.py             - competition R2 metric, mass balance enforcement
    scripts/
        setup_colab.sh          - one-time Colab environment bootstrap
        download_weights.py     - predownload weights for offline Kaggle submission
```

## Key Design Decisions

- **Cross-view self-attention fusion.** Left and right image halves are encoded separately through a shared DINOv3 backbone, then interact through a single multi-head self-attention layer before the regression heads.
- **Five independent regression heads.** The model predicts all five targets directly, with no architectural constraint forcing GDM/Total consistency during training.
- **Auxiliary interval-classification heads.** Every target also gets a classifier predicting which quantile bin its value falls in, trained jointly with the regression heads as a complementary signal.
- **Two-stage training and expanded augmentation.** The backbone trains frozen for a warmup phase before unfreezing, and the augmentation pipeline covers a wider range of lighting/camera conditions.
- **Group-aware cross-validation.** Folds are split by state + sampling date, not a plain shuffle, so photos from the same site visit can't leak across train and validation.
- **Mass balance post-processing.** Predictions are clipped to non-negative values, then GDM and Dry_Total are recomputed from their constituent parts to guarantee physical consistency — the only place this is enforced, since the architecture doesn't guarantee it.

## Model

| Model | Source | Parameters |
|-------|--------|------------|
| DINOv3 | `facebook/dinov3-vitl16-pretrain-lvd1689m` (Hugging Face) | ~300 million |

## Environment

Built for **Google Colab** with GPU. Configuration paths default to `/content/`; override with the `CSIRO_DATA_PATH` environment variable, or `train.py --data-path`, for local or other notebook environments. For Kaggle offline submissions, set `DINO_LOCAL_ONLY = True` in `src/config.py` and predownload model weights with `scripts/download_weights.py`.

## Quick Start

```bash
# in Colab, first bootstrap the environment (installs deps, downloads competition data)
bash scripts/setup_colab.sh

pip install -r requirements.txt
python train.py
```

Useful flags: `python train.py --fast-debug --debug-samples 50` for a quick end-to-end smoke test, or `--n-folds` / `--epochs` / `--output-dir` to override `src/config.py` defaults without editing it. Run `python train.py --help` for the full list.

## Outputs

`train.py` writes to `--output-dir` (default `/content/models`):

- `dinov3_regressor.pth` - full model checkpoint (Exponential Moving Average weights)
- `dinov3_heads_only.pth` - fusion + regression/classification heads only (smaller file, requires the Hugging Face backbone)
- `config.json` - configuration metadata, including the fitted interval-classification bin edges, for inference

## Acknowledgments

This project reimplements the competition's 1st place solution.

> Baiph, HZM, TheoQiu, zxc123cc. *1st Place Solution.* https://www.kaggle.com/competitions/csiro-biomass/writeups/1st-place-solution. 2026. Kaggle.

Their interval-classification head is itself adapted from crowd-counting research: Wang, C., Song, Q., Zhang, B., Wang, Y., Tai, Y., Hu, X., Wang, C., Li, J., Ma, J., & Wu, Y. (2021). *Uniformity in heterogeneity: Diving deep into count interval partition for crowd counting.* ICCV.