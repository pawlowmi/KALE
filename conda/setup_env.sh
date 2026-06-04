#!/bin/bash
# Recreate the KUEA conda environment from scratch.
# Usage: bash conda/setup_env.sh
set -e

ENV_NAME=myenv
PYTHON_VERSION=3.11

# Accept conda TOS if needed
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Create env
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
source activate $ENV_NAME

# PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Core ML deps
pip install open-clip-torch==2.19.0 timm==0.6.13 einops==0.6.1 einops-exts==0.0.4   ftfy==6.1.1 webdataset==0.2.48 scikit-learn==1.3.2 scipy==1.10.1   nltk==3.8.1 sentencepiece==0.1.98 accelerate==0.24.0 datasets==2.12.0   pycocoevalcap==1.2 pycocotools==2.0.6 geotorch==0.3.0 torchdiffeq==0.2.3 tensorboard

# Pinned transformers (from original requirements.txt)
pip install "transformers @ git+https://github.com/huggingface/transformers@d3cbc997a231098cca81ac27fd3028a5536abe67"

# Fix pkg_resources for older wandb/setuptools compat
pip install "setuptools<70"

# Robustbench (pinned commit, no deps to avoid pandas conflict)
pip install "robustbench @ git+https://github.com/RobustBench/robustbench.git@e67e4225facde47be6a41ed78b576076e8b90cc5" --no-deps

# Autoattack (for CLIP_benchmark eval)
pip install git+https://github.com/fra31/auto-attack

# TensorFlow + VTAB datasets (for evaluation)
pip install tensorflow-cpu tensorflow-datasets task_adaptation importlib_resources

# Fix protobuf (tensorflow needs >=6, open_clip wants <4 but works with 6)
pip install "protobuf>=6.31,<7"

# Wandb (>=0.16 for protobuf 6 compat)
pip install "wandb>=0.16"

# Dataset tools
pip install img2dataset unrar-cffi rarfile

echo "=== Environment $ENV_NAME ready ==='
echo "Activate with: source activate $ENV_NAME"
