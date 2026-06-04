#!/bin/bash
set -e

# Overnight experiment chain.
# Define experiments below, they run sequentially with evaluation after each.
#
# Usage:
#   tmux new-session -d -s overnight "bash /mnt/data/code/KUEA/scripts/experiment.sh"
#
# Monitor:
#   tmux attach -t overnight
#   tmux ls

# ── Configuration ────────────────────────────────────────────────────────────

DEVICES=0,1,2,3,4,5,6,7
TRAIN_SCRIPT=/mnt/data/code/KUEA/scripts/run_align_cc12m.sh
EVAL_SCRIPT=/mnt/data/code/KUEA/scripts/run_eval_checkpoints.sh
EXPERIMENTS_DIR=${EXPERIMENTS_DIR:-/mnt/data/experiments}
EVAL_BASE=${EVAL_BASE:-${EXPERIMENTS_DIR/experiments/eval_results}}
BF16=${BF16:-True}

# Define experiments: "BS PW EPOCHS WARMUP_PCT LR DYNAMIC_PW DYNAMIC_PW_TARGET COSINE_DECAY CW"
# Set DYNAMIC_PW=0 to disable dynamic penalty weight.
# Set COSINE_DECAY=True to enable cosine decay of target ratio.
EXPERIMENTS=(
    "128 0.5 2 8 1e-4 100 0.8 False 1.0"
    "128 0.5 2 8 2e-4 100 0.8 False 1.0"
)

# ── Functions ────────────────────────────────────────────────────────────────

run_training() {
    local bs=$1 pw=$2 epochs=$3 warmup_pct=$4 lr=$5 dynamic_pw=$6 dynamic_pw_target=$7 cosine_decay=${8:-False} cw=${9:-1.0}
    echo ""
    echo "=== Training: BS=$bs PW=$pw EPOCHS=$epochs WARMUP_PCT=$warmup_pct LR=$lr DYNAMIC_PW=$dynamic_pw DYNAMIC_PW_TARGET=$dynamic_pw_target COSINE_DECAY=$cosine_decay CW=$cw ==="
    echo "Started: $(date)"

    DEVICES=$DEVICES BS=$bs PW=$pw EPOCHS=$epochs WARMUP_PCT=$warmup_pct LR=$lr \
        DYNAMIC_PW=$dynamic_pw DYNAMIC_PW_TARGET=$dynamic_pw_target \
        DYNAMIC_PW_COSINE_DECAY=$cosine_decay CW=$cw \
        OUTPUT_DIR=$EXPERIMENTS_DIR BF16=$BF16 \
        bash "$TRAIN_SCRIPT"

    echo "Finished: $(date)"
}

find_latest_experiment() {
    ls -td "$EXPERIMENTS_DIR"/ViT-L-14_* 2>/dev/null | head -1
}

run_evaluation() {
    local exp_dir=$1
    if [ -z "$exp_dir" ]; then
        echo "WARNING: Experiment dir not found, skipping evaluation"
        return
    fi

    echo ""
    echo "=== Evaluating: $(basename $exp_dir) ==="
    echo "Started: $(date)"

    GPUS=$DEVICES CKPT_TYPE=all EVAL_BASE=$EVAL_BASE bash "$EVAL_SCRIPT" "$exp_dir"

    echo "Evaluation done: $(date)"
}

# ── Main ─────────────────────────────────────────────────────────────────────

cd /mnt/data/code/KUEA
export CUDA_VISIBLE_DEVICES=$DEVICES

echo "========================================"
echo "Overnight Experiment Chain"
echo "Started: $(date)"
echo "Experiments: ${#EXPERIMENTS[@]}"
for i in "${!EXPERIMENTS[@]}"; do
    read -r bs pw epochs warmup_pct lr dynamic_pw dynamic_pw_target cosine_decay cw <<< "${EXPERIMENTS[$i]}"
    echo "  $((i+1)). BS=$bs PW=$pw EPOCHS=$epochs WARMUP=$warmup_pct% LR=$lr DPW=$dynamic_pw DPW_T=$dynamic_pw_target COSINE=$cosine_decay CW=$cw"
done
echo "========================================"

COMPLETED_DIRS=()

for i in "${!EXPERIMENTS[@]}"; do
    read -r bs pw epochs warmup_pct lr dynamic_pw dynamic_pw_target cosine_decay cw <<< "${EXPERIMENTS[$i]}"

    echo ""
    echo "──────────────────────────────────────"
    echo "Experiment $((i+1))/${#EXPERIMENTS[@]}"
    echo "──────────────────────────────────────"

    run_training "$bs" "$pw" "$epochs" "$warmup_pct" "$lr" "$dynamic_pw" "$dynamic_pw_target" "$cosine_decay" "$cw"

    exp_dir=$(find_latest_experiment)
    echo "Experiment dir: $exp_dir"
    COMPLETED_DIRS+=("$exp_dir")

    run_evaluation "$exp_dir"
done

echo ""
echo "========================================"
echo "All experiments complete."
echo "Finished: $(date)"
echo ""
echo "Results:"
for dir in "${COMPLETED_DIRS[@]}"; do
    echo "  Train: $dir"
    echo "  Eval:  /mnt/data/eval_results/$(basename $dir)/"
    echo ""
done
echo "========================================"
