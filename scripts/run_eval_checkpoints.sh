#!/bin/bash
set -e

# Usage:
#   ./scripts/run_eval_checkpoints.sh /mnt/data/experiments/EXPERIMENT_DIR
#
# Runs zeroshot, lp, retrieval evaluations on the top N step checkpoints.
# Each checkpoint gets its own GPU and runs all 3 tasks sequentially.
# N checkpoints run in parallel across N GPUs.
#
# Override parameters:
#   N=4 ./scripts/run_eval_checkpoints.sh /path/to/experiment
#   GPUS=0,1,2,3 N=4 ./scripts/run_eval_checkpoints.sh /path/to/experiment
#
# Examples:
#   # Top 8 step checkpoints, excluding final, on 8 GPUs
#   N=8 ./scripts/run_eval_checkpoints.sh /mnt/data/experiments/ViT-L-14_openai_cc12m_l2_8epochs_bs1024_pw0.5_lr4e-05_cc12m-3m_VuUNz
#
#   # Include final + top 7 step checkpoints on 8 GPUs
#   INCLUDE_FINAL=1 N=8 ./scripts/run_eval_checkpoints.sh /mnt/data/experiments/ViT-L-14_openai_cc12m_l2_8epochs_bs1024_pw0.5_lr4e-05_cc12m-3m_VuUNz
#
#   # Top 4 checkpoints on specific GPUs
#   GPUS=4,5,6,7 N=4 ./scripts/run_eval_checkpoints.sh /mnt/data/experiments/ViT-L-14_openai_cc12m_l2_8epochs_bs1024_pw0.5_lr4e-05_cc12m-3m_VuUNz

EXP_DIR=${1:?Usage: $0 /path/to/experiment}
N=${N:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
INCLUDE_FINAL=${INCLUDE_FINAL:-0}

CKPT_DIR="$EXP_DIR/checkpoints"
EXP_NAME=$(basename "$EXP_DIR")
EVAL_BASE="/mnt/data/eval_results"

# Parse GPU list
IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

# Adjust N if final takes a GPU slot
if [ "$INCLUDE_FINAL" = "1" ] && [ -f "$CKPT_DIR/final.pt" ]; then
    MAX_STEP_CKPTS=$((NUM_GPUS - 1))
    if [ "$N" -gt "$MAX_STEP_CKPTS" ]; then
        N=$MAX_STEP_CKPTS
    fi
fi

# Find top N step checkpoints by step number
CHECKPOINTS=$(ls "$CKPT_DIR"/step_*.pt 2>/dev/null | grep -v '_opt' | \
    sed 's/.*step_\([0-9]*\)\.pt/\1/' | sort -rn | head -n "$N")

# Optionally add final checkpoint
if [ "$INCLUDE_FINAL" = "1" ] && [ -f "$CKPT_DIR/final.pt" ]; then
    CHECKPOINTS="final $CHECKPOINTS"
fi

if [ -z "$CHECKPOINTS" ]; then
    echo "No step checkpoints found in $CKPT_DIR"
    exit 1
fi

# Convert to array
CKPT_ARRAY=($CHECKPOINTS)
ACTUAL_N=${#CKPT_ARRAY[@]}

echo "========================================"
echo "Checkpoint Evaluation"
echo "========================================"
echo "Experiment: $EXP_NAME"
echo "Checkpoint dir: $CKPT_DIR"
echo "Results dir: $EVAL_BASE/${EXP_NAME}/"
echo "Tasks: zeroshot, lp, retrieval"
echo "GPUs: $GPUS ($NUM_GPUS available)"
echo "Checkpoints: $ACTUAL_N (top by step$([ "$INCLUDE_FINAL" = "1" ] && echo ", including final" || echo ", excluding final"))"
echo "----------------------------------------"

for i in $(seq 0 $((ACTUAL_N - 1))); do
    STEP=${CKPT_ARRAY[$i]}
    GPU=${GPU_LIST[$((i % NUM_GPUS))]}
    if [ "$STEP" = "final" ]; then
        CKPT="$CKPT_DIR/final.pt"
        SAVE="$EVAL_BASE/${EXP_NAME}/final"
    else
        CKPT="$CKPT_DIR/step_${STEP}.pt"
        SAVE="$EVAL_BASE/${EXP_NAME}/step_${STEP}"
    fi

    echo "  GPU $GPU -> ${STEP}.pt -> $SAVE/"
done

echo "----------------------------------------"
echo "Starting $ACTUAL_N tmux sessions..."

for i in $(seq 0 $((ACTUAL_N - 1))); do
    STEP=${CKPT_ARRAY[$i]}
    GPU=${GPU_LIST[$((i % NUM_GPUS))]}
    if [ "$STEP" = "final" ]; then
        CKPT="$CKPT_DIR/final.pt"
        SAVE="$EVAL_BASE/${EXP_NAME}/final"
    else
        CKPT="$CKPT_DIR/step_${STEP}.pt"
        SAVE="$EVAL_BASE/${EXP_NAME}/step_${STEP}"
    fi

    echo "  Started: eval_${STEP} (GPU $GPU)"

    tmux new-session -d -s "eval_${STEP}" "\
TASKS=zeroshot DEVICES=$GPU CHECKPOINT=$CKPT SAVE_DIR=$SAVE bash /mnt/data/code/KUEA/scripts/run_eval.sh && \
TASKS=lp DEVICES=$GPU CHECKPOINT=$CKPT SAVE_DIR=$SAVE bash /mnt/data/code/KUEA/scripts/run_eval.sh && \
TASKS=retrieval DEVICES=$GPU CHECKPOINT=$CKPT SAVE_DIR=$SAVE bash /mnt/data/code/KUEA/scripts/run_eval.sh"
done

echo ""
echo "========================================"
echo "All $ACTUAL_N sessions started."
echo "Monitor:  tmux ls"
echo "Attach:   tmux attach -t eval_XXXXX"
echo "Results:  $EVAL_BASE/${EXP_NAME}/"
echo "========================================"
