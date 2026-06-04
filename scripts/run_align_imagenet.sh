#!/bin/bash
set -e

# Usage:
#   ./scripts/run_align_imagenet.sh
#
# Override parameters via environment variables:
#   BS=128 PW=10 DEVICES=0,1 EPOCHS=4 ./scripts/run_align_imagenet.sh
#
# Run multiple experiments in parallel tmux sessions:
#   tmux new-session -d -s train_64_05  'DEVICES=0       BS=64  PW=0.5 WARMUP=1500 EPOCHS=2 bash /mnt/data/code/KUEA/scripts/run_align_imagenet.sh'
#   tmux new-session -d -s train_128_10 'DEVICES=1       BS=128 PW=10  WARMUP_PCT=10 EPOCHS=2 bash /mnt/data/code/KUEA/scripts/run_align_imagenet.sh'
#   tmux new-session -d -s train_256_10 'DEVICES=2,3     BS=128 PW=10  WARMUP_PCT=10 EPOCHS=2 bash /mnt/data/code/KUEA/scripts/run_align_imagenet.sh'
#   tmux new-session -d -s train_512_10 'DEVICES=4,5,6,7 BS=128 PW=10  WARMUP_PCT=10 EPOCHS=2 bash /mnt/data/code/KUEA/scripts/run_align_imagenet.sh'
#
# Note: BS is per-GPU batch size. With --enable_bs_scaling True (default),
#       effective batch size = BS x number of GPUs.

# Configurable parameters with defaults
BS=${BS:-64}
PW=${PW:-0.5}
DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
STEPS=${STEPS:-0}
EPOCHS=${EPOCHS:-4}
WARMUP=${WARMUP:-0}
WARMUP_PCT=${WARMUP_PCT:-10}

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

/home/ec2-user/miniconda3/envs/myenv/bin/python -u -m train.align_training_clip \
    --clip_model_name ViT-L-14 \
    --pretrained openai \
    --vision_model dino \
    --dataset imagenet \
    --imagenet_root /mnt/ramdisk \
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
    --experiment_name precomputed_inet \
    --log_freq 1 \
    --eval_freq 10 \
    --precomputed_dir /mnt/ramdisk/precomputed \
    --dataloader_num_workers 4 \
    --prefetch_factor 4 \
    --enable_bs_scaling True
