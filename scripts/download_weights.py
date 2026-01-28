"""
Predownload and zip a Hugging Face model for offline Kaggle submission.

Kaggle notebooks with internet disabled can't call from_pretrained against
the Hub. Run this once in Colab (which has internet), attach the resulting
zip as a Kaggle Dataset, then set DINO_LOCAL_ONLY = True in src/config.py
so the pipeline loads from the local copy instead.

Usage:
    python scripts/download_weights.py --model facebook/dinov3-vitl16-pretrain-lvd1689m
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def download_and_zip(repo_id, output_dir):
    local_dir = output_dir / repo_id.split('/')[-1]
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    archive_path = shutil.make_archive(str(local_dir), 'zip', root_dir=local_dir)
    print(f'saved: {archive_path}')
    return archive_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='Hugging Face repo id')
    parser.add_argument('--output-dir', type=Path, default=Path('/content'))
    args = parser.parse_args()
    download_and_zip(args.model, args.output_dir)


if __name__ == '__main__':
    main()
