#!/bin/bash
set -e

# Usage:
#   # Run all stages
#   ./scripts/run_precompute_cc12m.sh
#
#   # Run only extraction (train + val)
#   STAGE=extract ./scripts/run_precompute_cc12m.sh
#
#   # Run only precompute (after extraction)
#   STAGE=precompute ./scripts/run_precompute_cc12m.sh

STAGE=${STAGE:-all}
CC12M_SHARDS=${CC12M_SHARDS:-/mnt/data/datasets/cc12m/shards}
EXTRACT_DIR=${EXTRACT_DIR:-/mnt/ramdisk/cc12m-6m}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/ramdisk/cc12m-6m/precomputed}
N_SAMPLES=${N_SAMPLES:-6600000}
N_VAL_SAMPLES=${N_VAL_SAMPLES:-50000}
SEED=${SEED:-198}
BATCH_SIZE=${BATCH_SIZE:-2048}
NUM_WORKERS=${NUM_WORKERS:-32}

cd /mnt/data/code/KUEA

# Stage 1: Extract train + val images from CC12M shards
if [ "$STAGE" = "extract" ] || [ "$STAGE" = "all" ]; then
    echo "=== Stage 1: Extract ==="
    /home/ec2-user/miniconda3/envs/myenv/bin/python -u -m precompute.precompute_cc12m \
        --stage extract \
        --cc12m_shards $CC12M_SHARDS \
        --extract_dir $EXTRACT_DIR \
        --n_samples $N_SAMPLES \
        --n_val_samples $N_VAL_SAMPLES \
        --seed $SEED \
        --num_readers 32 \
        --num_preprocessors 128 \
        --num_writers 32 
fi

# Stage 2: Precompute CLIP + DINOv2 embeddings for train + val
if [ "$STAGE" = "precompute" ] || [ "$STAGE" = "all" ]; then
    echo "=== Stage 2: Precompute embeddings ==="
    /home/ec2-user/miniconda3/envs/myenv/bin/python -u -m precompute.precompute_cc12m \
        --stage precompute \
        --extract_dir $EXTRACT_DIR \
        --output_dir $OUTPUT_DIR \
        --clip_model_name ViT-L-14 \
        --vision_model dino \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --prefetch_factor 10
fi

echo "=== Done ==="
