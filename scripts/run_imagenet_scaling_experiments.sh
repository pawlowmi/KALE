#!/bin/bash
set -e
eval "$(/home/ec2-user/miniconda3/bin/conda shell.bash hook)"
conda activate myenv

# =============================================================================
# ImageNet Scaling Experiments: Batch Size vs Penalty Weight
# =============================================================================
#
# GOAL:
# Find the right penalty_weight when scaling batch size beyond the paper's
# default (bs=64, pw=0.5). Run on ImageNet-1K to match the paper's setup.
#
# BACKGROUND:
# The kernel alignment loss averages over N^2 pairs in the batch. As batch
# size grows, the number of uninformative pairs (where both CLIP and DINOv2
# agree) grows quadratically, diluting the gradient signal. The paper used
# bs=64 with pw=0.5 on 2x RTX 4090.
#
# In our CC12M run (bs=1024, pw=0.5, 8x H100), the kernel alignment loss
# plateaued at ~0.027 and never decreased — confirming the dilution problem.
#
# SCALING LAW:
# To compensate for N^2 dilution, penalty_weight should scale as:
#   pw_new = pw_base × (N_new / N_base)^2
# This gives: bs=128 → pw=2, bs=256 → pw=8
# We test at, below, and between these predicted values.
#
# EXPERIMENT DESIGN (all 6 run in parallel, 3000 steps, 1500 warmup):
#
#   Exp  GPUs   BS   PW    Purpose
#   1    GPU 0  64   0.5   Paper baseline — reference for loss curve shape
#   2    GPU 1  128  0.5   Paper pw at larger batch — expect flat loss
#   3    GPU 2  128  1.0   Midpoint — does moderate scaling help?
#   4    GPU 3  128  2.0   Math-predicted pw for bs=128
#   5    GPU 4,5 256 4.0   Below predicted pw for bs=256
#   6    GPU 6,7 256 8.0   Math-predicted pw for bs=256
#
# SUCCESS CRITERIA:
#   - Kernel alignment loss should decrease over steps (like paper Fig. 4)
#   - eval-acc should stay near baseline 74.9% (text alignment preserved)
#   - Compare loss curves across experiments to find optimal bs/pw combo
#
# MEMORY CONSTRAINTS (discovered empirically):
#   - Single H100 80GB: max bs ~128 (3x ViT-L-14 models + activations)
#   - 2x H100: max bs ~256
#   - bs=512 needs 4+ GPUs, bs=1024 needs 8 GPUs
# =============================================================================

COMMON_ARGS="--clip_model_name ViT-L-14 --pretrained openai \
  --dataset imagenet --imagenet_root /mnt/data/datasets/imagenet \
  --template std --output_normalize False \
  --steps 3000 --warmup 1500 \
  --loss l2 --loss_clean l2 --opt adamw --lr 1e-5 --wd 1e-4 \
  --inner_loss l2 --wandb False \
  --output_dir /mnt/data/experiments \
  --clean_weight 1.0 \
  --kernel_dino polynomial --kernel_clip polynomial \
  --gamma 0.0032 --coef0 0.191623 \
  --log_freq 1 --eval_freq 10"

cd /mnt/data/code/KUEA

echo "=== Starting 6 ImageNet scaling experiments in parallel ==="
echo "=== $(date -Iseconds) ==="

# Exp 1: Paper baseline (bs=64, pw=0.5, single GPU)
nohup python -m train.align_training_clip \
  $COMMON_ARGS \
  --devices 0 --batch_size 64 --penalty_weight 0.5 \
  --experiment_name inet_bs64_pw05 \
  > /mnt/data/experiments/inet_bs64_pw05.log 2>&1 &
echo "Exp1 (bs=64, pw=0.5) started on GPU 0 — PID: $!"

# Exp 2: Larger batch, paper pw (bs=128, pw=0.5, single GPU)
nohup python -m train.align_training_clip \
  $COMMON_ARGS \
  --devices 1 --batch_size 128 --penalty_weight 0.5 \
  --experiment_name inet_bs128_pw05 \
  > /mnt/data/experiments/inet_bs128_pw05.log 2>&1 &
echo "Exp2 (bs=128, pw=0.5) started on GPU 1 — PID: $!"

# Exp 3: Larger batch, midpoint pw (bs=128, pw=1.0, single GPU)
nohup python -m train.align_training_clip \
  $COMMON_ARGS \
  --devices 2 --batch_size 128 --penalty_weight 1.0 \
  --experiment_name inet_bs128_pw1 \
  > /mnt/data/experiments/inet_bs128_pw1.log 2>&1 &
echo "Exp3 (bs=128, pw=1.0) started on GPU 2 — PID: $!"

# Exp 4: Larger batch, math-predicted pw (bs=128, pw=2.0, single GPU)
nohup python -m train.align_training_clip \
  $COMMON_ARGS \
  --devices 3 --batch_size 128 --penalty_weight 2.0 \
  --experiment_name inet_bs128_pw2 \
  > /mnt/data/experiments/inet_bs128_pw2.log 2>&1 &
echo "Exp4 (bs=128, pw=2.0) started on GPU 3 — PID: $!"

# Exp 5: Large batch, below predicted pw (bs=256, pw=4.0, 2 GPUs)
nohup python -m train.align_training_clip \
  $COMMON_ARGS \
  --devices 4,5 --batch_size 256 --penalty_weight 4.0 \
  --experiment_name inet_bs256_pw4 \
  > /mnt/data/experiments/inet_bs256_pw4.log 2>&1 &
echo "Exp5 (bs=256, pw=4.0) started on GPUs 4,5 — PID: $!"

# Exp 6: Large batch, math-predicted pw (bs=256, pw=8.0, 2 GPUs)
nohup python -m train.align_training_clip \
  $COMMON_ARGS \
  --devices 6,7 --batch_size 256 --penalty_weight 8.0 \
  --experiment_name inet_bs256_pw8 \
  > /mnt/data/experiments/inet_bs256_pw8.log 2>&1 &
echo "Exp6 (bs=256, pw=8.0) started on GPUs 6,7 — PID: $!"

echo ""
echo "=== All 6 experiments launched ==="
echo "Monitor:"
echo "  for f in /mnt/data/experiments/inet_*.log; do echo \"--- \$(basename \$f) ---\"; grep '\[loss\]' \$f | tail -3; done"
