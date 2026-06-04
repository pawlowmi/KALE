#!/bin/bash
set -e

# Download LLaVA v1.5 training data (mix665k) to /mnt/data/datasets/llava/
# Sources: COCO, Visual Genome, GQA, OCR-VQA, TextVQA
#
# Usage:
#   bash scripts/download_llava_data.sh
#   OUT_DIR=/custom/path bash scripts/download_llava_data.sh

OUT_DIR=${OUT_DIR:-/mnt/data/datasets/llava}
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo "=== LLaVA v1.5 Data Download ==="
echo "Output: $OUT_DIR"
echo ""

# ── Annotations JSON ─────────────────────────────────────────────────────────
echo "[1/6] Downloading llava_v1_5_mix665k.json..."
wget -q --show-progress -c \
    "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json" \
    -O llava_v1_5_mix665k.json

# ── COCO 2017 train images (~18GB) ───────────────────────────────────────────
echo "[2/6] Downloading COCO train2017..."
mkdir -p coco
wget -q --show-progress -c \
    "http://images.cocodataset.org/zips/train2017.zip" \
    -O coco/train2017.zip
echo "  Extracting COCO..."
unzip -q -n coco/train2017.zip -d coco/
rm coco/train2017.zip

# ── Visual Genome (~15GB) ────────────────────────────────────────────────────
echo "[3/6] Downloading Visual Genome part1 + part2..."
mkdir -p vg
wget -q --show-progress -c \
    "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip" \
    -O vg/images_part1.zip
wget -q --show-progress -c \
    "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip" \
    -O vg/images_part2.zip
echo "  Extracting VG..."
unzip -q -n vg/images_part1.zip -d vg/
unzip -q -n vg/images_part2.zip -d vg/
rm vg/images_part1.zip vg/images_part2.zip
# Merge VG_100K_2 into VG_100K
mv vg/VG_100K_2/* vg/VG_100K/ 2>/dev/null || true
rmdir vg/VG_100K_2 2>/dev/null || true

# ── GQA images (~20GB) ───────────────────────────────────────────────────────
echo "[4/6] Downloading GQA images..."
mkdir -p gqa
wget -q --show-progress -c \
    "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip" \
    -O gqa/images.zip
echo "  Extracting GQA..."
unzip -q -n gqa/images.zip -d gqa/
rm gqa/images.zip

# ── OCR-VQA (~10GB) ──────────────────────────────────────────────────────────
echo "[5/6] Downloading OCR-VQA via HuggingFace..."
mkdir -p ocr_vqa
conda run -n myenv python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='howard-hou/OCR-VQA',
    repo_type='dataset',
    local_dir='ocr_vqa',
    ignore_patterns=['*.parquet'],
)
"

# ── TextVQA (~7GB) ───────────────────────────────────────────────────────────
echo "[6/6] Downloading TextVQA train images..."
mkdir -p textvqa
wget -q --show-progress -c \
    "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" \
    -O textvqa/train_val_images.zip
echo "  Extracting TextVQA..."
unzip -q -n textvqa/train_val_images.zip -d textvqa/
rm textvqa/train_val_images.zip

echo ""
echo "=== Download complete: $OUT_DIR ==="
du -sh "$OUT_DIR"
