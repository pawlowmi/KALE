#!/bin/bash
set -e

# =============================================================================
# Diagnostic Experiments: Kernel Alignment Loss Not Decreasing
# =============================================================================
#
# CONTEXT:
# We are training KUEA (Kernel-based Unsupervised Embedding Alignment) on CC12M
# (~10.7M images), scaling up from the paper's ImageNet-1K (1.28M images).
# The original run used batch_size=1024 on 8x H100 GPUs with penalty_weight=0.5.
#
# PROBLEM:
# The kernel alignment loss plateaued at ~0.027 from step ~80 and never decreased
# over 6,400+ steps. The paper (Appendix C.1, Fig. 4) shows this loss should
# steadily decrease during training.
#
# HYPOTHESIS:
# The kernel alignment loss computes pairwise similarities across all images in a
# batch, producing an NxN kernel matrix. With batch_size=1024, this is a 1024x1024
# matrix with ~500K pairs. Most pairs are between unrelated images where both CLIP
# and DINOv2 agree (similarity ≈ 0), contributing no learning signal. The useful
# signal from the few disagreeing pairs gets diluted when averaged over all pairs.
# The paper used batch_size=64 (64x64 = ~2K pairs) where the signal-to-noise
# ratio is much better.
#
# An alternative/complementary hypothesis: with penalty_weight=0.5 and
# clean_weight=1.0, the regularization term (keeping CLIP close to original)
# may overpower the kernel alignment term, preventing the model from moving
# toward DINOv2's representation structure.
#
# EXPERIMENT DESIGN:
# We run 4 short diagnostic experiments (2,000 steps each) in parallel on 2 GPUs
# each to identify the root cause. We check:
#   - Does reducing batch size fix the loss curve? (exp1, exp2)
#   - Does increasing penalty_weight fix it at batch 128? (exp3)
#   - Does massively increasing penalty_weight compensate for large batch? (exp4)
#     (1024^2 / 64^2 ≈ 256x more pairs, so pw=128 roughly compensates)
#
# SUCCESS CRITERIA:
# After 2,000 steps (1,500 warmup + 500 at full lr), check:
#   1. Is the kernel alignment loss trending downward? (primary signal)
#   2. Is eval-acc (ImageNet zero-shot) staying near baseline 74.9%? (safety check)
# The winning config gets all 8 GPUs for the full CC12M training run.
#
# BASELINE REFERENCE (previous run):
#   batch_size=1024, penalty_weight=0.5, 8x H100
#   Loss: flat at ~0.027 from step 80 to 6,400+
#   Eval-acc: oscillating 71-75% (baseline CLIP ViT-L-14 = 74.9%)
#   Checkpoint saved at step 6,450 for potential resume.
#
# =============================================================================

COMMON_ARGS="--clip_model_name ViT-L-14 --pretrained openai \
  --dataset cc12m --imagenet_root /mnt/data/datasets/imagenet \
  --cc12m_root /mnt/data/datasets/cc12m/shards \
  --template std --output_normalize False \
  --steps 2000 --warmup 1500 \
  --loss l2 --loss_clean l2 --opt adamw --lr 1e-5 --wd 1e-4 \
  --inner_loss l2 --wandb False \
  --output_dir /mnt/data/experiments \
  --clean_weight 1.0 \
  --kernel_dino polynomial --kernel_clip polynomial \
  --gamma 0.0032 --coef0 0.191623 \
  --log_freq 1 --eval_freq 10"

eval "$(/home/ec2-user/miniconda3/bin/conda shell.bash hook)" && conda activate myenv
cd /mnt/data/code/KUEA

echo "=== Starting 4 diagnostic experiments in parallel ==="
echo "=== $(date -Iseconds) ==="

# Exp 1: Paper baseline batch size (64), paper penalty weight (0.5)
# Purpose: Does matching the paper's batch size fix the loss curve?
nohup python -m train.align_training_clip \
  $COMMON_ARGS \
  --devices 0,1 --batch_size 64 --penalty_weight 0.5 \
  --experiment_name diag_bs64_pw05 \
  > /mnt/data/experiments/diag_bs64_pw05.log 2>&1 &
echo "Exp1 (bs=64, pw=0.5) started on GPUs 0,1 — PID: $!"

# Exp 2: Slightly larger batch (128), paper penalty weight (0.5)
# Purpose: Find if 128 still works or if the threshold is below 128.
nohup python -m train.align_training_clip \
  $COMMON_ARGS \
  --devices 2,3 --batch_size 128 --penalty_weight 0.5 \
  --experiment_name diag_bs128_pw05 \
  > /mnt/data/experiments/diag_bs128_pw05.log 2>&1 &
echo "Exp2 (bs=128, pw=0.5) started on GPUs 2,3 — PID: $!"

# Exp 3: Batch 128 with higher penalty weight (2.0)
# Purpose: Does doubling alignment weight help at moderate batch size?
nohup python -m train.align_training_clip \
  $COMMON_ARGS \
  --devices 4,5 --batch_size 128 --penalty_weight 2.0 \
  --experiment_name diag_bs128_pw2 \
  > /mnt/data/experiments/diag_bs128_pw2.log 2>&1 &
echo "Exp3 (bs=128, pw=2.0) started on GPUs 4,5 — PID: $!"

# Exp 4: Large batch (512) with scaled penalty weight (32)
# Purpose: Test if scaling penalty_weight proportionally to batch size
# compensates for pair dilution. 512^2/64^2 = 64x more pairs, pw=32
# roughly compensates. Original exp4 (bs=1024) OOM on 2 GPUs.
nohup python -m train.align_training_clip \
  $COMMON_ARGS \
  --devices 6,7 --batch_size 512 --penalty_weight 32.0 \
  --experiment_name diag_bs512_pw32 \
  > /mnt/data/experiments/diag_bs512_pw32.log 2>&1 &
echo "Exp4 (bs=512, pw=32) started on GPUs 6,7 — PID: \$!"

echo ""
echo "=== All experiments launched ==="
echo "Monitor with:"
echo "  tail -f /mnt/data/experiments/diag_bs64_pw05.log"
echo "  tail -f /mnt/data/experiments/diag_bs128_pw05.log"
echo "  tail -f /mnt/data/experiments/diag_bs128_pw2.log"
echo "  tail -f /mnt/data/experiments/diag_bs1024_pw128.log"
echo ""
echo "Quick loss check:"
echo "  for f in /mnt/data/experiments/diag_*.log; do echo \"--- \$f ---\"; grep '\[loss\]' \$f | tail -5; done"
