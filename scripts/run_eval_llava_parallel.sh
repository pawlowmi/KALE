#!/bin/bash
set -e

# ── Configure experiments here ───────────────────────────────────────────────

LORA_CHECKPOINTS=(
    "llava-v1.5-7b-lora-ViT-L-14_openai_imagenet_l2_40000steps_baseline_paper_reproduced_pw0.5_MysNy"
    "llava-v1.5-7b-lora-ViT-L-14_openai_cc12m_l2_2epochs_bs1024_pw0.5_dpw100_t0.8_lr0.0001_cc12m-3m_ntKBz"
    "llava-v1.5-7b-lora-ViT-L-14_openai_cc12m_l2_2epochs_bs1024_pw0.5_dpw100_t0.8_lr0.0001_cc12m-3m-cap-lr-20_AaApf"
)

CLIP_CHECKPOINTS=(
    "ViT-L-14_openai_imagenet_l2_40000steps_baseline_paper_reproduced_pw0.5_MysNy"
    "ViT-L-14_openai_cc12m_l2_2epochs_bs1024_pw0.5_dpw100_t0.8_lr0.0001_cc12m-3m_ntKBz"
    "ViT-L-14_openai_cc12m_l2_2epochs_bs1024_pw0.5_dpw100_t0.8_lr0.0001_cc12m-3m-cap-lr-20_AaApf"
)

STEPS=(
    "final.pt"
    "step_4025.pt"
    "step_4830.pt"
)

# ── Settings ─────────────────────────────────────────────────────────────────

GPUS=(1 2 3 4 5 6 7)
BS=8
SAVE_DIR=/mnt/data/eval_results/llava
EXPERIMENTS_DIR=/mnt/data/experiments
LORA_BASE=/mnt/data/llava/checkpoints

TASK_GROUPS=(
    "vqav2_val"
    "textvqa_val"
    "pope,ai2d"
    "refcoco"
    "refcoco+"
    "refcocog"
)

# ── Main ─────────────────────────────────────────────────────────────────────

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13

cd /mnt/data/code/KUEA

# Build flat job queue: "lora_name|merged_path|tasks|tag"
JOBS=()
for i in "${!LORA_CHECKPOINTS[@]}"; do
    LORA_NAME="${LORA_CHECKPOINTS[$i]}"
    CLIP_EXP="${CLIP_CHECKPOINTS[$i]}"
    STEP="${STEPS[$i]}"
    LORA_PATH="$LORA_BASE/$LORA_NAME"
    CLIP_PATH="$EXPERIMENTS_DIR/$CLIP_EXP/checkpoints/$STEP"
    MERGED_PATH="${LORA_PATH}_merged"
    TAG=$(echo "$LORA_NAME" | grep -oP '[A-Za-z0-9]{5}$')

    # Merge before queuing (blocking, CPU-only)
    if [ -d "$MERGED_PATH" ] && [ -f "$MERGED_PATH/config.json" ]; then
        echo "[$LORA_NAME] merge: already done"
    else
        echo "[$LORA_NAME] merging..."
        python LLaVA/scripts/merge_lora_weights.py \
            --model-path "$LORA_PATH" \
            --model-base liuhaotian/llava-v1.5-7b \
            --vision_tower "$CLIP_PATH" \
            --save-model-path "$MERGED_PATH"
        echo "[$LORA_NAME] merge done"
    fi

    for TASKS in "${TASK_GROUPS[@]}"; do
        JOBS+=("${LORA_NAME}|${MERGED_PATH}|${TASKS}|${TAG}")
    done
done

echo ""
echo "=== ${#JOBS[@]} jobs across ${#GPUS[@]} GPUs ==="

# GPU pool: track which session is running on each GPU slot
declare -A GPU_SESSION  # GPU index -> session name (empty = free)

dispatch() {
    local GPU=$1 LORA_NAME=$2 MERGED_PATH=$3 TASKS=$4 TAG=$5
    local SESSION="eval-${TAG}-$(echo "$TASKS" | tr ',+' '-')"
    local LOG="/mnt/data/eval_results/eval_${TAG}_$(echo "$TASKS" | tr ',+' '-').log"

    echo "  GPU $GPU | $SESSION | $TASKS"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION"
    tmux send-keys -t "$SESSION" "
conda activate myenv && cd /mnt/data/code/KUEA
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_HOME=$CUDA_HOME CUDA_VISIBLE_DEVICES=$GPU \
python -m lmms_eval --model llava \
    --model_args 'pretrained=${MERGED_PATH},conv_template=vicuna_v1' \
    --tasks '$TASKS' --batch_size $BS \
    --output_path '$SAVE_DIR/$LORA_NAME' \
2>&1 | tee '$LOG' && echo DONE_SENTINEL
" Enter
    GPU_SESSION[$GPU]="$SESSION"
}

is_done() {
    local SESSION=$1
    tmux capture-pane -t "$SESSION" -p 2>/dev/null | grep -q "DONE_SENTINEL"
}

# Fill all GPUs initially
JOB_IDX=0
for GPU in "${GPUS[@]}"; do
    if [ $JOB_IDX -ge ${#JOBS[@]} ]; then break; fi
    IFS='|' read -r LORA_NAME MERGED_PATH TASKS TAG <<< "${JOBS[$JOB_IDX]}"
    dispatch "$GPU" "$LORA_NAME" "$MERGED_PATH" "$TASKS" "$TAG"
    JOB_IDX=$(( JOB_IDX + 1 ))
done

# Scheduler loop: poll for finished GPUs, dispatch next job
while [ $JOB_IDX -lt ${#JOBS[@]} ] || [ ${#GPU_SESSION[@]} -gt 0 ]; do
    for GPU in "${GPUS[@]}"; do
        SESSION="${GPU_SESSION[$GPU]:-}"
        [ -z "$SESSION" ] && continue

        if is_done "$SESSION"; then
            echo "  GPU $GPU | $SESSION: done"
            unset GPU_SESSION[$GPU]

            if [ $JOB_IDX -lt ${#JOBS[@]} ]; then
                IFS='|' read -r LORA_NAME MERGED_PATH TASKS TAG <<< "${JOBS[$JOB_IDX]}"
                dispatch "$GPU" "$LORA_NAME" "$MERGED_PATH" "$TASKS" "$TAG"
                JOB_IDX=$(( JOB_IDX + 1 ))
            fi
        fi
    done
    sleep 30
done

echo ""
echo "=== All ${#JOBS[@]} jobs complete. Results in $SAVE_DIR ==="
