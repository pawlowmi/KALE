#!/bin/bash
set -e

# Usage:
#   ./scripts/run_eval.sh
#
# Override parameters via environment variables:
#   CHECKPOINT=/path/to/final.pt DEVICES=0 ./scripts/run_eval.sh
#
# Examples:
#   # Evaluate original CLIP (no checkpoint)
#   DEVICES=0 ./scripts/run_eval.sh
#
#   # Evaluate a fine-tuned model
#   CHECKPOINT=/mnt/data/experiments/ViT-L-14_.../checkpoints/final.pt DEVICES=0 ./scripts/run_eval.sh
#
#   # Run all tasks in parallel on separate GPUs
#   CHECKPOINT=/path/to/final.pt SAVE_DIR=/mnt/data/eval
#   TASKS=zeroshot  DEVICES=0 CHECKPOINT=$CHECKPOINT SAVE_DIR=$SAVE_DIR bash /mnt/data/code/KUEA/scripts/run_eval.sh &
#   TASKS=lp        DEVICES=1 CHECKPOINT=$CHECKPOINT SAVE_DIR=$SAVE_DIR bash /mnt/data/code/KUEA/scripts/run_eval.sh &
#   TASKS=retrieval DEVICES=2 CHECKPOINT=$CHECKPOINT SAVE_DIR=$SAVE_DIR bash /mnt/data/code/KUEA/scripts/run_eval.sh &
#
#   # Run only zero-shot classification
#   CHECKPOINT=/path/to/final.pt TASKS=zeroshot DEVICES=0 ./scripts/run_eval.sh
#
#   # Run only linear probing and retrieval
#   TASKS="lp,retrieval" DEVICES=0 ./scripts/run_eval.sh
#
#   # Run in tmux
#   CHECKPOINT=/mnt/data/experiments/ViT-L-14_openai_imagenet_l2_40000steps_baseline_paper_reproduced_pw0.5_MysNy/checkpoints/final.pt
#   SAVE_DIR=/mnt/data/eval_results/ViT-L-14_openai_imagenet_l2_40000steps_baseline_paper_reproduced_pw0.5_MysNy
#   tmux new-session -d -s eval_zs "TASKS=zeroshot  DEVICES=0 CHECKPOINT=$CHECKPOINT SAVE_DIR=$SAVE_DIR bash /mnt/data/code/KUEA/scripts/run_eval.sh"
#   tmux new-session -d -s eval_lp "TASKS=lp        DEVICES=1 CHECKPOINT=$CHECKPOINT SAVE_DIR=$SAVE_DIR bash /mnt/data/code/KUEA/scripts/run_eval.sh"
#   tmux new-session -d -s eval_rt "TASKS=retrieval DEVICES=2 CHECKPOINT=$CHECKPOINT SAVE_DIR=$SAVE_DIR bash /mnt/data/code/KUEA/scripts/run_eval.sh"

# Configurable parameters with defaults
MODEL=${MODEL:-ViT-L-14}
CHECKPOINT=${CHECKPOINT:-openai}
DEVICES=${DEVICES:-0}
BS=${BS:-64}
TASKS=${TASKS:-zeroshot,lp,retrieval}
SAVE_DIR=${SAVE_DIR:-/mnt/data/eval_results}
DATASET_ROOT=${DATASET_ROOT:-/mnt/data/datasets/eval/wds_{dataset_cleaned}}

if [ -n "$DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$DEVICES
fi

cd /mnt/data/code/KUEA/CLIP_benchmark
export PYTHONPATH="/mnt/data/code/KUEA":"${PYTHONPATH}"

# Generate models.txt with the model to evaluate
echo "${MODEL},${CHECKPOINT}" > benchmark/models.txt
echo "Evaluating: ${MODEL},${CHECKPOINT}"

mkdir -p "$SAVE_DIR"

SECONDS=0

# Zero-shot classification
if echo "$TASKS" | grep -q "zeroshot"; then
    echo ""
    echo "=== Zero-shot Classification ==="
    /home/ec2-user/miniconda3/envs/myenv/bin/python -m clip_benchmark.cli eval \
        --dataset_root "$DATASET_ROOT" \
        --dataset benchmark/datasets.txt \
        --pretrained_model benchmark/models.txt \
        --output "${SAVE_DIR}/zeroshot_{model}_{pretrained}_{dataset}.json" \
        --attack none --eps 1 \
        --batch_size $BS --n_samples -1
fi

# Linear probing
if echo "$TASKS" | grep -q "lp"; then
    echo ""
    echo "=== Linear Probing ==="
    /home/ec2-user/miniconda3/envs/myenv/bin/python -m clip_benchmark.cli eval \
        --dataset_root "$DATASET_ROOT" \
        --dataset benchmark/datasets_lp.txt \
        --task linear_probe \
        --pretrained_model benchmark/models.txt \
        --output "${SAVE_DIR}/lp_{model}_{pretrained}_{dataset}.json" \
        --attack none --eps 1 \
        --batch_size $BS --n_samples -1
fi

# Image-text retrieval
if echo "$TASKS" | grep -q "retrieval"; then
    echo ""
    echo "=== Image-Text Retrieval ==="
    /home/ec2-user/miniconda3/envs/myenv/bin/python -m clip_benchmark.cli eval \
        --dataset_root "$DATASET_ROOT" \
        --dataset benchmark/datasets_rt.txt \
        --task zeroshot_retrieval \
        --recall_k 1 5 10 \
        --pretrained_model benchmark/models.txt \
        --output "${SAVE_DIR}/retrieval_{model}_{pretrained}_{dataset}.json" \
        --attack none --eps 1 \
        --batch_size $BS --n_samples -1
fi

hours=$((SECONDS / 3600))
minutes=$(( (SECONDS % 3600) / 60 ))
echo ""
echo "=== Done. Results saved to ${SAVE_DIR} ==="
echo "[Runtime] ${hours}h ${minutes}min"
