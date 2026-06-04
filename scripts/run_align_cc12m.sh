#!/bin/bash
set -e

cd /mnt/data/code/KUEA

python -m train.align_training_clip \
    --clip_model_name ViT-L-14 \
    --pretrained openai \
    --dataset cc12m \
    --imagenet_root /mnt/data/datasets/imagenet \
    --cc12m_root /mnt/data/datasets/cc12m/shards \
    --template std \
    --output_normalize False \
    --steps 40000 \
    --warmup 2800 \
    --batch_size 256 \
    --loss l2 \
    --loss_clean l2 \
    --opt adamw \
    --lr 1e-5 \
    --wd 1e-4 \
    --inner_loss l2 \
    --wandb False \
    --output_dir /mnt/data/experiments \
    --clean_weight 1. \
    --penalty_weight 0.5 \
    --kernel_dino polynomial \
    --kernel_clip polynomial \
    --gamma 0.0032 \
    --coef0 0.191623 \
    --experiment_name cc12m_align \
    --log_freq 1 \
    --eval_freq 10
