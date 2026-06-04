#!/bin/bash
set -e

METADATA_DIR=/mnt/data/datasets/datacomp-medium/metadata
SHARDS_DIR=/mnt/data/datasets/datacomp-medium/shards

# Ensure dependencies
python -c "import img2dataset" 2>/dev/null || pip install img2dataset
which huggingface-cli >/dev/null 2>&1 || pip install huggingface_hub[cli]

echo "=== Step 1: Download metadata (parquet files, ~30GB) ==="
mkdir -p $METADATA_DIR
huggingface-cli download --repo-type dataset mlfoundations/datacomp_medium \
  --include "*.parquet" \
  --local-dir $METADATA_DIR

echo "=== Step 2: Download images and create webdataset shards ==="
mkdir -p $SHARDS_DIR
img2dataset \
  --url_list $METADATA_DIR \
  --input_format parquet \
  --url_col url \
  --caption_col text \
  --output_format webdataset \
  --output_folder $SHARDS_DIR \
  --processes_count 16 \
  --thread_count 64 \
  --image_size 256 \
  --resize_mode keep_ratio \
  --enable_wandb False

echo "=== Done ==="
echo "Shards: $(ls $SHARDS_DIR/*.tar 2>/dev/null | wc -l) files"
echo "Size: $(du -sh $SHARDS_DIR)"
