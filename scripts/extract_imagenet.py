"""
Extract ImageNet from HuggingFace parquet format to ImageFolder structure.

Input:  /mnt/data/datasets/imagenet/data/train-*.parquet  (294 shards)
        /mnt/data/datasets/imagenet/data/validation-*.parquet (14 shards)
Output: /mnt/data/datasets/imagenet/train/{synset_id}/img_XXXXXX.JPEG
        /mnt/data/datasets/imagenet/val/{synset_id}/img_XXXXXX.JPEG

Each parquet row has: image={'bytes': b'...'}, label=int (0-999)
Label ints map to synset IDs via classes.py IMAGENET2012_CLASSES dict.
"""
import os, sys, glob
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/datasets/imagenet"

# Load label -> synset mapping
exec(open(os.path.join(ROOT, "classes.py")).read())
idx_to_synset = list(IMAGENET2012_CLASSES.keys())

def process_shard(args):
    parquet_path, out_dir = args
    df = pd.read_parquet(parquet_path)
    count = 0
    for idx, row in df.iterrows():
        synset = idx_to_synset[row["label"]]
        class_dir = os.path.join(out_dir, synset)
        os.makedirs(class_dir, exist_ok=True)
        # unique name from shard + row index
        shard_id = Path(parquet_path).stem
        img_path = os.path.join(class_dir, f"{shard_id}_{idx:06d}.JPEG")
        with open(img_path, "wb") as f:
            f.write(row["image"]["bytes"])
        count += 1
    return count

for split, pattern, out_name in [
    ("train", "train-*.parquet", "train"),
    ("val", "validation-*.parquet", "val"),
]:
    out_dir = os.path.join(ROOT, out_name)
    shards = sorted(glob.glob(os.path.join(ROOT, "data", pattern)))
    print(f"Extracting {split}: {len(shards)} shards -> {out_dir}")
    tasks = [(s, out_dir) for s in shards]
    total = 0
    with ProcessPoolExecutor(max_workers=16) as pool:
        for i, count in enumerate(pool.map(process_shard, tasks)):
            total += count
            print(f"  [{i+1}/{len(shards)}] +{count} images (total: {total})")
    print(f"{split} done: {total} images\n")
