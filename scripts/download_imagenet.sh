#!/bin/bash
set -e

# =============================================================================
# Download ImageNet-1K (ILSVRC2012)
# =============================================================================
#
# The KUEA paper uses ImageNet-1K training set (1.28M images) as the alignment
# fine-tuning dataset, and the validation set (50K images) for eval-acc
# (zero-shot classification during training).
#
# PREREQUISITES:
#   1. Accept terms at: https://huggingface.co/datasets/ILSVRC/imagenet-1k
#   2. Log in: huggingface-cli login
#
# The dataset is downloaded in its original format with train/ and val/
# directories containing class subdirectories (ImageFolder format), which
# is what the KUEA training code expects via --imagenet_root.
# =============================================================================

DEST=${1:-/mnt/data/datasets/imagenet}

echo "Verifying HuggingFace authentication..."
if ! huggingface-cli whoami > /dev/null 2>&1; then
  echo "ERROR: Not logged in. Run: huggingface-cli login"
  exit 1
fi
echo "Authenticated as: $(huggingface-cli whoami 2>/dev/null | head -1)"

mkdir -p "$DEST"

echo "Downloading ImageNet-1K to $DEST ..."
echo "This is ~150GB and will take a while."

huggingface-cli download --repo-type dataset ILSVRC/imagenet-1k \
  --local-dir "$DEST"

echo ""
echo "=== Done ==="
echo "Train images: $(find $DEST/train -type f 2>/dev/null | wc -l)"
echo "Val images:   $(find $DEST/val -type f 2>/dev/null | wc -l)"
echo ""
echo "Usage: --imagenet_root $DEST"
