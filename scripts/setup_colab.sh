#!/usr/bin/env bash
# One-time Colab environment setup: installs dependencies, authenticates
# with Hugging Face, and downloads the competition dataset.
#
# transformers is installed from source because DINOv3 support is not yet
# in the stable release. Restart the runtime after the first run, then
# rerun only `huggingface-cli login` + the kaggle download lines if needed.
set -euo pipefail

pip install -q "git+https://github.com/huggingface/transformers.git" accelerate timm
pip install -Uq huggingface_hub
pip install -q albumentations opencv-python-headless scikit-learn tqdm

# HF_TOKEN must be set in the environment first, e.g. in Colab:
#   import os; os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
huggingface-cli login --token "$HF_TOKEN"

kaggle competitions download -c csiro-biomass
unzip -q csiro-biomass.zip
rm csiro-biomass.zip
