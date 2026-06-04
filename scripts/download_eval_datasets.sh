#!/bin/bash
set -e

DEST=${1:-/mnt/data/datasets/eval}

# Verify HF authentication
echo "Verifying HuggingFace authentication..."
if ! hf auth whoami > /dev/null 2>&1; then
  echo "ERROR: Not logged in. Run: huggingface-cli login"
  exit 1
fi
echo "Authenticated as: $(hf auth whoami 2>/dev/null | head -1)"

DATASETS=(
  wds_vtab-cifar10 wds_vtab-cifar100 wds_vtab-caltech101
  wds_fer2013 wds_vtab-pets wds_vtab-dtd wds_vtab-resisc45
  wds_vtab-eurosat wds_vtab-pcam wds_imagenet_sketch wds_imagenet-o
  wds_svhn_cropped wds_gtsrb wds_vtab-clevr_closest_object_distance
  wds_vtab-clevr_count_all wds_mscoco_captions wds_flickr30k
)

FAILED=()
for ds in "${DATASETS[@]}"; do
  if [ -d "$DEST/$ds" ] && [ -f "$DEST/$ds/test/nshards.txt" ]; then
    echo "SKIP $ds (already exists)"
    continue
  fi
  echo "Downloading $ds..."
  if ! hf download --repo-type dataset "clip-benchmark/${ds}" --local-dir "$DEST/$ds" 2>&1; then
    echo "FAILED: $ds (may require accepting terms at https://huggingface.co/datasets/clip-benchmark/${ds})"
    FAILED+=("$ds")
  fi
done

echo ""
echo "=== Done ==="
echo "Downloaded: $(ls -d $DEST/wds_* 2>/dev/null | wc -l) datasets"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Failed (${#FAILED[@]}):"
  printf "  - %s\n" "${FAILED[@]}"
fi
