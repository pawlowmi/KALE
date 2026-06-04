#!/bin/bash
set -e

DEST_DIR=/mnt/data/datasets/cc12m/shards
LOG_FILE=/mnt/data/datasets/download_cc12m.log
REPO=pixparse/cc12m-wds

echo "=== Downloading CC12M webdataset shards ==="
mkdir -p "$DEST_DIR"
python -c "from huggingface_hub import snapshot_download" 2>/dev/null || pip install "huggingface_hub[cli]"

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=\"pixparse/cc12m-wds\",
    repo_type=\"dataset\",
    local_dir=\"/mnt/data/datasets/cc12m/shards\",
    resume_download=True,
    max_workers=64,
)
print(\"=== Done ===\")
" 2>&1 | tee "$LOG_FILE"

echo "Shards: \$(ls \$DEST_DIR/*.tar 2>/dev/null | wc -l) files"
echo "Size: \$(du -sh \$DEST_DIR)"
