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
EXPERIMENTS_DIR=/mnt/data/experiments
EVAL_N=8

# Define experiments: "BS PW EPOCHS WARMUP_PCT LR DYNAMIC_PW DYNAMIC_PW_TARGET"
# Set DYNAMIC_PW=0 to disable dynamic penalty weight.
EXPERIMENTS=(
    "128 1000 4 5 1.6e-4 100 1.0"
    "128 1000 4 5 4e-5   100 1.0"
)

# ── Functions ────────────────────────────────────────────────────────────────

run_training() {
    local bs=$1 pw=$2 epochs=$3 warmup_pct=$4 lr=$5 dynamic_pw=$6 dynamic_pw_target=$7
    echo ""
    echo "=== Training: BS=$bs PW=$pw EPOCHS=$epochs WARMUP_PCT=$warmup_pct LR=$lr DYNAMIC_PW=$dynamic_pw DYNAMIC_PW_TARGET=$dynamic_pw_target ==="
    echo "Started: $(date)"

    DEVICES=$DEVICES BS=$bs PW=$pw EPOCHS=$epochs WARMUP_PCT=$warmup_pct LR=$lr \
        DYNAMIC_PW=$dynamic_pw DYNAMIC_PW_TARGET=$dynamic_pw_target \
        bash "$TRAIN_SCRIPT"

    echo "Finished: $(date)"
}

find_latest_experiment() {
    local pw=$1 epochs=$2 bs=$3 lr=$4
    ls -td "$EXPERIMENTS_DIR"/ViT-L-14_openai_cc12m_l2_${epochs}epochs_bs${bs}_pw${pw}_lr${lr}_cc12m-3m_* 2>/dev/null | head -1
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

    INCLUDE_FINAL=1 N=$EVAL_N bash "$EVAL_SCRIPT" "$exp_dir"

    echo "Waiting for evaluations to finish..."
    while tmux ls 2>/dev/null | grep -q "eval_"; do
        sleep 60
    done

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
    read -r bs pw epochs warmup_pct lr dynamic_pw dynamic_pw_target <<< "${EXPERIMENTS[$i]}"
    echo "  $((i+1)). BS=$bs PW=$pw EPOCHS=$epochs WARMUP=$warmup_pct% LR=$lr DPW=$dynamic_pw DPW_T=$dynamic_pw_target"
done
echo "========================================"

COMPLETED_DIRS=()

for i in "${!EXPERIMENTS[@]}"; do
    read -r bs pw epochs warmup_pct lr dynamic_pw dynamic_pw_target <<< "${EXPERIMENTS[$i]}"

    echo ""
    echo "──────────────────────────────────────"
    echo "Experiment $((i+1))/${#EXPERIMENTS[@]}"
    echo "──────────────────────────────────────"

    run_training "$bs" "$pw" "$epochs" "$warmup_pct" "$lr" "$dynamic_pw" "$dynamic_pw_target"

    exp_dir=$(find_latest_experiment "$pw" "$epochs" "$((bs * 8))" "$lr")
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
