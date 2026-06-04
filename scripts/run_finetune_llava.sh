#!/bin/bash
set -e

# Fine-tune LLaVA v1.5-7b with LoRA using a KUEA-aligned CLIP checkpoint.
#
# Usage:
#   ./scripts/run_finetune_llava.sh /path/to/clip_checkpoint.pt
#
#   DEVICES=0,1,2,3 BS=2 ./scripts/run_finetune_llava.sh /path/to/clip_checkpoint.pt
#
# Prerequisites:
#   - LLaVA training data at LLaVA/playground/data/
#   - DeepSpeed installed

CHECKPOINT=${1:?Usage: $0 /path/to/clip_checkpoint.pt}

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
BS=${BS:-1}
EPOCHS=${EPOCHS:-1}
LR=${LR:-2e-4}
LORA_R=${LORA_R:-128}
LORA_ALPHA=${LORA_ALPHA:-256}
OUTPUT_DIR=${OUTPUT_DIR:-}

if [ -n "$DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$DEVICES
fi

NUM_GPUS=$(echo "$DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')

# Derive output dir from checkpoint name if not set
if [ -z "$OUTPUT_DIR" ]; then
    CKPT_NAME=$(basename "$(dirname "$(dirname "$CHECKPOINT")")")
    OUTPUT_DIR="/mnt/data/llava/checkpoints/llava-v1.5-7b-lora-${CKPT_NAME}"
fi

mkdir -p "$OUTPUT_DIR"
cd /mnt/data/code/KUEA/LLaVA

export CUDA_HOME=${CUDA_HOME:-/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13}
export PATH="$CUDA_HOME/bin:$PATH"

echo "=== LLaVA LoRA Fine-tuning ==="
echo "CLIP checkpoint: $CHECKPOINT"
echo "Output:          $OUTPUT_DIR"
echo "GPUs:            $NUM_GPUS ($DEVICES)"
echo "BS:              $BS"
echo "LoRA:            r=$LORA_R alpha=$LORA_ALPHA"
echo ""

/home/ec2-user/miniconda3/envs/myenv/bin/deepspeed --num_gpus=$NUM_GPUS llava/train/train_mem.py \
    --lora_enable True --lora_r $LORA_R --lora_alpha $LORA_ALPHA --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower "$CHECKPOINT" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

echo ""
echo "=== Done. Model saved to $OUTPUT_DIR ==="
