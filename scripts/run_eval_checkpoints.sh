#!/bin/bash
set -e

# Evaluate ALL checkpoints from an experiment directory.
# Launches up to NUM_GPUS evaluations in parallel, rolls forward when a GPU frees up.
#
# Usage:
#   ./scripts/run_eval_checkpoints.sh /mnt/data/experiments/EXPERIMENT_DIR
#   GPUS=0,1,2,3 ./scripts/run_eval_checkpoints.sh /path/to/experiment
#   CKPT_TYPE=step ./scripts/run_eval_checkpoints.sh /path/to/experiment
#
# CKPT_TYPE: epoch (default), step, or all (epochs first)

EXP_DIR=${1:?Usage: $0 /path/to/experiment}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
CKPT_TYPE=${CKPT_TYPE:-epoch}
EVAL_BASE=${EVAL_BASE:-/mnt/data/eval_results}

CKPT_DIR="$EXP_DIR/checkpoints"

# Derive eval subpath: preserve directory structure relative to experiments/
EXP_DIR_CLEAN=$(echo "$EXP_DIR" | sed 's:/*$::')
if echo "$EXP_DIR_CLEAN" | grep -q '/experiments/'; then
    EVAL_SUBPATH=$(echo "$EXP_DIR_CLEAN" | sed 's:.*/experiments/::')
else
    EVAL_SUBPATH=$(basename "$EXP_DIR_CLEAN")
fi
EXP_NAME=$(basename "$EXP_DIR_CLEAN")

IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

# Collect checkpoints: epoch first when "all", sorted by number descending
collect_ckpts() {
    local type=$1 dir=$2
    ls "$dir"/${type}_*.pt 2>/dev/null | grep -v '_opt' | \
        sed "s/.*\(${type}_[0-9]*\)\.pt/\1/" | sort -t'_' -k2 -rn
}

CHECKPOINTS=()
if [ "$CKPT_TYPE" = "all" ]; then
    while IFS= read -r c; do CHECKPOINTS+=("$c"); done < <(collect_ckpts epoch "$CKPT_DIR")
    while IFS= read -r c; do CHECKPOINTS+=("$c"); done < <(collect_ckpts step "$CKPT_DIR")
else
    while IFS= read -r c; do CHECKPOINTS+=("$c"); done < <(collect_ckpts "$CKPT_TYPE" "$CKPT_DIR")
fi

# Add final if it exists
[ -f "$CKPT_DIR/final.pt" ] && CHECKPOINTS+=("final")

TOTAL=${#CHECKPOINTS[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "No checkpoints found in $CKPT_DIR"
    exit 1
fi

echo "========================================"
echo "Checkpoint Evaluation"
echo "========================================"
echo "Experiment: $EXP_NAME"
echo "Results:    $EVAL_BASE/${EVAL_SUBPATH}/"
echo "GPUs:       $GPUS ($NUM_GPUS)"
echo "Checkpoints: $TOTAL ($CKPT_TYPE)"
echo "----------------------------------------"
for c in "${CHECKPOINTS[@]}"; do echo "  $c"; done
echo "----------------------------------------"

# Track which GPU is running which tmux session
GPU_SESSION=()
for ((g=0; g<NUM_GPUS; g++)); do GPU_SESSION+=(""); done

wait_for_gpu() {
    # Wait until at least one GPU's tmux session has finished
    while true; do
        for ((g=0; g<NUM_GPUS; g++)); do
            if [ -n "${GPU_SESSION[$g]}" ] && ! tmux has-session -t "${GPU_SESSION[$g]}" 2>/dev/null; then
                GPU_SESSION[$g]=""
                echo "$g"
                return
            fi
        done
        sleep 10
    done
}

find_free_gpu() {
    for ((g=0; g<NUM_GPUS; g++)); do
        if [ -z "${GPU_SESSION[$g]}" ]; then
            echo "$g"
            return
        fi
    done
    # All busy, wait
    wait_for_gpu
}

for c in "${CHECKPOINTS[@]}"; do
    if [ "$c" = "final" ]; then
        CKPT="$CKPT_DIR/final.pt"
        SAVE="$EVAL_BASE/${EVAL_SUBPATH}/final"
    else
        CKPT="$CKPT_DIR/${c}.pt"
        SAVE="$EVAL_BASE/${EVAL_SUBPATH}/${c}"
    fi

    GPU_IDX=$(find_free_gpu)
    GPU=${GPU_LIST[$GPU_IDX]}
    SESSION="eval_${c}"

    echo "  GPU $GPU -> ${c} -> $SAVE/"
    GPU_SESSION[$GPU_IDX]="$SESSION"

    tmux new-session -d -s "$SESSION" "\
TASKS=zeroshot DEVICES=$GPU CHECKPOINT=$CKPT SAVE_DIR=$SAVE bash /mnt/data/code/KUEA/scripts/run_eval.sh && \
TASKS=lp DEVICES=$GPU CHECKPOINT=$CKPT SAVE_DIR=$SAVE bash /mnt/data/code/KUEA/scripts/run_eval.sh && \
TASKS=retrieval DEVICES=$GPU CHECKPOINT=$CKPT SAVE_DIR=$SAVE bash /mnt/data/code/KUEA/scripts/run_eval.sh; \
echo '=== DONE (exit code: '\$?') === Press enter to close'; read"
done

echo ""
echo "All $TOTAL evaluations launched."
echo "Waiting for remaining to finish..."

# Wait for all remaining sessions
while true; do
    RUNNING=0
    for ((g=0; g<NUM_GPUS; g++)); do
        if [ -n "${GPU_SESSION[$g]}" ] && tmux has-session -t "${GPU_SESSION[$g]}" 2>/dev/null; then
            RUNNING=$((RUNNING + 1))
        fi
    done
    [ "$RUNNING" -eq 0 ] && break
    echo "  $RUNNING still running..."
    sleep 30
done

echo ""
echo "========================================"
echo "All $TOTAL evaluations complete."
echo "Results: $EVAL_BASE/${EVAL_SUBPATH}/"
echo "========================================"
