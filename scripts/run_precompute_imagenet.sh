#!/bin/bash
set -e

cd /mnt/data/code/KUEA

/home/ec2-user/miniconda3/envs/myenv/bin/python -u -m precompute.precompute_imagenet \
    --imagenet_root /mnt/ramdisk \
    --output_dir /mnt/ramdisk/precomputed \
    --clip_model_name ViT-L-14 \
    --vision_model dino \
    --batch_size 2048 \
    --num_workers 32
