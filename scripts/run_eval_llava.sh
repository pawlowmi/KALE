#!/bin/bash
set -e

# Evaluate a fine-tuned LLaVA model on the paper's benchmarks using lmms-eval.
#
# Usage:
#   ./scripts/run_eval_llava.sh /path/to/lora_checkpoint /path/to/clip_checkpoint.pt
#
#   DEVICES=0,1 TASKS=pope,ai2d ./scripts/run_eval_llava.sh /path/to/lora_checkpoint /path/to/clip_checkpoint.pt
#
# This script:
#   1. Merges LoRA weights into a full model (if not already done)
#   2. Runs lmms-eval on the paper's benchmarks
#
# Paper benchmarks: VQAv2, TextVQA, RefCOCO, RefCOCO+, RefCOCOg, POPE, AI2D
# (VSR and TallyQA are not available in lmms-eval)
#
# Prerequisites:
#   cd /mnt/data/code/KUEA/lmms-eval && pip install -e ".[all]"

LORA_PATH=${1:?Usage: $0 /path/to/lora_checkpoint /path/to/clip_checkpoint.pt}
CLIP_CHECKPOINT=${2:?Usage: $0 /path/to/lora_checkpoint /path/to/clip_checkpoint.pt}

MODEL_BASE=${MODEL_BASE:-liuhaotian/llava-v1.5-7b}
DEVICES=${DEVICES:-0}
BS=${BS:-1}
TASKS=${TASKS:-vqav2_val,textvqa_val,pope,ai2d,refcoco,refcoco+,refcocog}
SAVE_DIR=${SAVE_DIR:-/mnt/data/eval_results/llava}

if [ -n "$DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$DEVICES
fi

LORA_NAME=$(basename "$LORA_PATH")
MERGED_PATH="${LORA_PATH}_merged"

cd /mnt/data/code/KUEA

# ── Step 1: Merge LoRA weights ───────────────────────────────────────────────

if [ -d "$MERGED_PATH" ] && [ -f "$MERGED_PATH/config.json" ]; then
    echo "=== Merged model found at $MERGED_PATH, skipping merge ==="
else
    echo "=== Merging LoRA weights ==="
    echo "  LoRA:  $LORA_PATH"
    echo "  Base:  $MODEL_BASE"
    echo "  CLIP:  $CLIP_CHECKPOINT"
    echo "  Out:   $MERGED_PATH"
    python LLaVA/scripts/merge_lora_weights.py \
        --model-path "$LORA_PATH" \
        --model-base "$MODEL_BASE" \
        --vision_tower "$CLIP_CHECKPOINT" \
        --save-model-path "$MERGED_PATH"
    echo "=== Merge complete ==="
fi

# ── Step 2: Run lmms-eval ────────────────────────────────────────────────────

mkdir -p "$SAVE_DIR"

echo ""
echo "=== Running lmms-eval ==="
echo "  Model: $MERGED_PATH"
echo "  Tasks: $TASKS"
echo "  GPUs:  $DEVICES"
echo "  Output: $SAVE_DIR"
echo ""

python -m lmms_eval \
    --model llava \
    --model_args "pretrained=${MERGED_PATH},conv_template=vicuna_v1" \
    --tasks "$TASKS" \
    --batch_size "$BS" \
    --output_path "$SAVE_DIR/${LORA_NAME}"

echo ""
echo "=== Done. Results saved to $SAVE_DIR/${LORA_NAME} ==="
