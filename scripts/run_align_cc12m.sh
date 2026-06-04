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
# Train on extracted CC12M with precomputed embeddings (default, fast):
#   tmux new-session -d -s cc12m_train "DEVICES=0,1,2,3,4,5,6,7 BS=128 PW=100 EPOCHS=4 WARMUP_PCT=5 bash /mnt/data/code/KUEA/scripts/run_align_cc12m.sh"
#
# Train directly on CC12M shards without precompute (slow, runs 3 models on GPU):
#   Note: No precomputed embeddings, no --epochs (use --steps), no eval accuracy.
#   tmux new-session -d -s cc12m_live "DEVICES=0,1,2,3,4,5,6,7 BS=64 PW=0.5 STEPS=40000 MODE=live bash /mnt/data/code/KUEA/scripts/run_align_cc12m.sh"
#
# Eval on ImageNet val for meaningful zero-shot accuracy during training.

# Configurable parameters with defaults
BS=${BS:-64}
PW=${PW:-0.5}
DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
STEPS=${STEPS:-0}
EPOCHS=${EPOCHS:-4}
WARMUP=${WARMUP:-0}
WARMUP_PCT=${WARMUP_PCT:-10}
LR=${LR:-1.6e-4}
CC12M_ROOT=${CC12M_ROOT:-/mnt/ramdisk/cc12m-3m}
PRECOMPUTED_DIR=${PRECOMPUTED_DIR:-/mnt/ramdisk/cc12m-3m/precomputed}
MODE=${MODE:-precomputed}
CC12M_SHARDS=${CC12M_SHARDS:-/mnt/data/datasets/cc12m/shards}
DYNAMIC_PW=${DYNAMIC_PW:-0}
DYNAMIC_PW_TARGET=${DYNAMIC_PW_TARGET:-0.5}
DYNAMIC_PW_COSINE_DECAY=${DYNAMIC_PW_COSINE_DECAY:-False}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/data/experiments/}
DRIFT_FREQ=${DRIFT_FREQ:-100}
BF16=${BF16:-True}

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

# Count GPUs
NUM_GPUS=$(echo "$DEVICES" | tr ',' '\n' | wc -l)

# Mode-specific flags
if [ "$MODE" = "live" ]; then
    MODE_FLAGS="--cc12m_shards $CC12M_SHARDS"
else
    MODE_FLAGS="--imagenet_root $CC12M_ROOT --precomputed_dir $PRECOMPUTED_DIR"
fi

# Train on CC12M, eval on ImageNet val for meaningful zero-shot accuracy.
/home/ec2-user/miniconda3/envs/myenv/bin/torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 -m train.align_training_clip \
    --clip_model_name ViT-L-14 \
    --pretrained openai \
    --vision_model dino \
    --dataset cc12m \
    $MODE_FLAGS \
    --eval_root /mnt/data/datasets/imagenet \
    --template std \
    --output_normalize False \
    --loss l2 \
    --loss_clean l2 \
    --opt adamw \
    --lr $LR \
    --lr_min_pct 10 \
    --wd 1e-4 \
    --inner_loss l2 \
    --wandb False \
    --output_dir $OUTPUT_DIR \
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
    --experiment_name cc12m-3m \
    --log_freq 1 \
    --eval_freq 1000 \
    --dataloader_num_workers 16 \
    --prefetch_factor 8 \
    --enable_bs_scaling True \
    --enhanced_metrics False \
    --lam 1e-4 \
    --dynamic_pw $DYNAMIC_PW \
    --dynamic_pw_target_ratio $DYNAMIC_PW_TARGET \
    --dynamic_pw_cosine_decay $DYNAMIC_PW_COSINE_DECAY \
    --drift_freq $DRIFT_FREQ \
    --bf16 $BF16
