#!/bin/bash
set -e

# Usage:
#   ./scripts/run_align_cc12m.sh
#
# Prerequisites:
#   Run ./scripts/run_precompute_cc12m.sh first to extract images and precompute embeddings.
#
# Override parameters via environment variables:
#   BS=128 PW=10 DEVICES=0,1 EPOCHS=4 ./scripts/run_align_cc12m.sh
#
# Run in tmux:
#   tmux new-session -d -s cc12m_train 'DEVICES=0,1,2,3 BS=128 PW=10 EPOCHS=4 bash /mnt/data/code/KUEA/scripts/run_align_cc12m.sh'

# Configurable parameters with defaults
BS=${BS:-64}
PW=${PW:-0.5}
DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
STEPS=${STEPS:-0}
EPOCHS=${EPOCHS:-4}
WARMUP=${WARMUP:-0}
WARMUP_PCT=${WARMUP_PCT:-10}
CC12M_ROOT=${CC12M_ROOT:-/mnt/ramdisk/cc12m}
PRECOMPUTED_DIR=${PRECOMPUTED_DIR:-/mnt/ramdisk/cc12m/precomputed}

# Build steps/epochs flag (mutually exclusive)
if [ "$STEPS" -gt 0 ]; then
    DURATION_FLAG="--steps $STEPS"
else
    DURATION_FLAG="--epochs $EPOCHS"
fi

cd /mnt/data/code/KUEA

if [ -n "$DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$DEVICES
fi

# Uses --dataset imagenet since extracted CC12M is in ImageFolder format.
# --imagenet_root points to the extracted CC12M directory (has train/class_0/).
# Eval still uses ImageNet val from /mnt/ramdisk/val.
/home/ec2-user/miniconda3/envs/myenv/bin/python -u -m train.align_training_clip \
    --clip_model_name ViT-L-14 \
    --pretrained openai \
    --vision_model dino \
    --dataset imagenet \
    --imagenet_root $CC12M_ROOT \
    --template std \
    --output_normalize False \
    --loss l2 \
    --loss_clean l2 \
    --opt adamw \
    --lr 1e-5 \
    --wd 1e-4 \
    --inner_loss l2 \
    --wandb False \
    --output_dir /mnt/data/experiments/ \
    $DURATION_FLAG \
    --warmup $WARMUP \
    --warmup_pct $WARMUP_PCT \
    --batch_size $BS \
    --clean_weight 1. \
    --penalty_weight $PW \
    --kernel_dino polynomial \
    --kernel_clip polynomial \
    --gamma 0.0032 \
    --coef0 0.191623 \
    --experiment_name cc12m \
    --log_freq 1 \
    --eval_freq 10 \
    --precomputed_dir $PRECOMPUTED_DIR \
    --dataloader_num_workers 4 \
    --prefetch_factor 4 \
    --enable_bs_scaling True \
    --enhanced_metrics True \
    --lam 1e-4
