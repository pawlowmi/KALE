"""
Precompute CLIP-original and DINOv2 embeddings for CC12M.

Two stages:
  --stage extract    : Sample N images from CC12M webdataset shards, save as ImageFolder
  --stage precompute : Compute embeddings from the extracted ImageFolder

Usage:
  # Stage 1: Extract 1.28M images to ramdisk
  python -m precompute.precompute_cc12m --stage extract \
      --cc12m_shards /mnt/data/datasets/cc12m/shards \
      --extract_dir /mnt/ramdisk/cc12m \
      --n_samples 1281167 --seed 99

  # Stage 2: Precompute embeddings
  python -m precompute.precompute_cc12m --stage precompute \
      --extract_dir /mnt/ramdisk/cc12m \
      --output_dir /mnt/ramdisk/cc12m/precomputed
"""
import argparse
import glob
import io
import os
import random
import sys
import tarfile
import multiprocessing as mp

sys.path.append("open_flamingo")

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

from precompute.common import load_and_wrap_models, extract_embeddings, save_meta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, required=True, choices=['extract', 'precompute'],
                        help='Stage to run: extract or precompute')

    # Extract stage args
    parser.add_argument('--cc12m_shards', type=str, default='', help='CC12M webdataset shards directory')
    parser.add_argument('--extract_dir', type=str, required=True,
                        help='Directory to save extracted images (ImageFolder format)')
    parser.add_argument('--n_samples', type=int, default=0,
                        help='Number of train images to sample (only for extract stage)')
    parser.add_argument('--n_val_samples', type=int, default=50000,
                        help='Number of val images to sample (only for extract stage)')
    parser.add_argument('--seed', type=int, default=99, help='Random seed for shard shuffling')
    parser.add_argument('--num_readers', type=int, default=8, help='Number of reader processes')
    parser.add_argument('--num_preprocessors', type=int, default=16, help='Number of preprocessor processes')
    parser.add_argument('--num_writers', type=int, default=8, help='Number of writer processes')

    # Precompute stage args
    parser.add_argument('--output_dir', type=str, default='', help='Directory to save embeddings')
    parser.add_argument('--clip_model_name', type=str, default='ViT-L-14', help='CLIP architecture')
    parser.add_argument('--vision_model', type=str, default='dino', help='Vision model: dino, mlcd')
    parser.add_argument('--output_normalize', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=32, help='DataLoader workers')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Batches each worker pre-loads ahead')

    return parser.parse_args()


# ── Reader processes ──────────────────────────────────────────────────────────
# Read raw bytes from tar shards, sample a fraction, push to preprocess queue.

def reader_worker(shard_list, sample_rate, seed, raw_queues, reader_id):
    """Read raw image bytes from assigned shards and round-robin to preprocessor queues."""
    rng = random.Random(seed + reader_id)
    n_queues = len(raw_queues)
    idx = 0
    for tar_path in shard_list:
        try:
            with tarfile.open(tar_path, 'r') as tf:
                members = [m for m in tf if m.isfile() and
                           m.name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                rng.shuffle(members)
                n_take = max(1, int(len(members) * sample_rate))
                for member in members[:n_take]:
                    try:
                        raw = tf.extractfile(member).read()
                        raw_queues[idx % n_queues].put(raw)
                        idx += 1
                    except Exception:
                        pass
        except Exception:
            pass
    # Send sentinel to all preprocessor queues
    for q in raw_queues:
        q.put(None)


# ── Preprocessor processes ────────────────────────────────────────────────────
# Decode raw bytes to RGB JPEG bytes, push to writer queue.

def preprocessor_worker(raw_queue, jpg_queues, num_reader_sentinels):
    """Decode raw image bytes to RGB JPEG, round-robin to writer queues."""
    sentinels_seen = 0
    n_queues = len(jpg_queues)
    idx = 0
    while True:
        item = raw_queue.get()
        if item is None:
            sentinels_seen += 1
            if sentinels_seen >= num_reader_sentinels:
                break
            continue
        try:
            img = Image.open(io.BytesIO(item)).convert('RGB')
            buf = io.BytesIO()
            img.save(buf, 'JPEG', quality=95)
            jpg_queues[idx % n_queues].put(buf.getvalue())
            idx += 1
        except (OSError, Exception):
            pass
    # Send sentinel to all writer queues
    for q in jpg_queues:
        q.put(None)


# ── Writer processes ──────────────────────────────────────────────────────────
# Write JPEG bytes to disk.

def writer_worker(jpg_queue, out_dir, counter, lock, n_samples, num_preprocessor_sentinels):
    """Write JPEG bytes to numbered files on disk."""
    sentinels_seen = 0
    while True:
        item = jpg_queue.get()
        if item is None:
            sentinels_seen += 1
            if sentinels_seen >= num_preprocessor_sentinels:
                break
            continue
        with lock:
            idx = counter.value
            if idx >= n_samples:
                continue
            counter.value += 1
        with open(os.path.join(out_dir, f'{idx:08d}.jpg'), 'wb') as f:
            f.write(item)


def extract_from_shards(cc12m_shards, extract_dir, n_samples, seed, split_name,
                        num_readers, num_preprocessors, num_writers):
    """Sample N images from CC12M shards and save as ImageFolder.

    Three-stage pipeline:
      Readers (tar I/O) -> Preprocessors (decode/encode) -> Writers (disk I/O)
    All stages run as separate processes communicating via queues.

    Args:
        cc12m_shards: Directory containing .tar shard files.
        extract_dir: Output directory.
        n_samples: Number of images to sample.
        seed: Random seed for shard shuffling.
        split_name: 'train' or 'val'.
        num_readers: Number of reader processes.
        num_preprocessors: Number of preprocessor processes.
        num_writers: Number of writer processes.
    """
    tar_files = sorted(glob.glob(os.path.join(cc12m_shards, '*.tar')))
    assert len(tar_files) > 0, f'No .tar files found in {cc12m_shards}'
    print(f'\n=== Extracting {split_name} ({n_samples} images) ===', flush=True)
    print(f'Found {len(tar_files)} shards in {cc12m_shards}', flush=True)

    out_dir = os.path.join(extract_dir, split_name, 'class_0')
    os.makedirs(out_dir, exist_ok=True)

    # Shuffle shards and distribute across readers
    rng = random.Random(seed)
    rng.shuffle(tar_files)

    est_total = len(tar_files) * 5000
    sample_rate = min(n_samples / est_total * 1.1, 1.0)  # 10% oversample
    print(f'Pipeline: {num_readers} readers -> {num_preprocessors} preprocessors -> {num_writers} writers',
          flush=True)
    print(f'Sampling ~{sample_rate:.1%} from each shard for {n_samples} target images', flush=True)

    # Split shards across readers
    shard_chunks = [[] for _ in range(num_readers)]
    for i, tf in enumerate(tar_files):
        shard_chunks[i % num_readers].append(tf)

    # Per-preprocessor queues to reduce contention
    raw_queues = [mp.Queue(maxsize=10000) for _ in range(num_preprocessors)]
    jpg_queues = [mp.Queue(maxsize=10000) for _ in range(num_writers)]

    # Shared counter for output file naming
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    # Start writers — each gets its own input queue
    writers = []
    for w in range(num_writers):
        p = mp.Process(target=writer_worker,
                       args=(jpg_queues[w], out_dir, counter, lock, n_samples, num_preprocessors))
        p.start()
        writers.append(p)

    # Start preprocessors — each reads from its own raw queue, round-robins to writer queues
    preprocessors = []
    for pp in range(num_preprocessors):
        p = mp.Process(target=preprocessor_worker,
                       args=(raw_queues[pp], jpg_queues, num_readers))
        p.start()
        preprocessors.append(p)

    # Start readers — round-robin to preprocessor queues
    readers = []
    for i in range(num_readers):
        p = mp.Process(target=reader_worker,
                       args=(shard_chunks[i], sample_rate, seed, raw_queues, i))
        p.start()
        readers.append(p)

    # Monitor progress
    print(f'Extracting...', flush=True)
    while True:
        with lock:
            current = counter.value
        if current >= n_samples:
            break
        print(f'\r  {current}/{n_samples} ({current * 100 // n_samples}%)', end='', flush=True)
        import time
        time.sleep(2)
    print(f'\r  {n_samples}/{n_samples} (100%)', flush=True)

    # Wait for readers to finish
    for p in readers:
        p.join()

    # Send stop signals to preprocessors (readers already sent their sentinels)
    # Preprocessors will stop after seeing all reader sentinels

    for p in preprocessors:
        p.join()

    # Preprocessors sent their sentinels to writers
    for p in writers:
        p.join()

    with lock:
        final = counter.value
    print(f'Extracted {final} images to {out_dir}', flush=True)
    return final


def precompute(args):
    """Run embedding extraction on the extracted ImageFolder."""
    output_dir = args.output_dir or os.path.join(args.extract_dir, 'precomputed')
    os.makedirs(output_dir, exist_ok=True)

    clip_model, dino_model, preprocess = load_and_wrap_models(
        args.clip_model_name, args.vision_model
    )

    train_root = os.path.join(args.extract_dir, 'train')
    assert os.path.isdir(train_root), f'{train_root} not found. Run --stage extract first.'

    clip_dim, dino_dim = extract_embeddings(
        'train', train_root,
        clip_model, dino_model, preprocess,
        args.batch_size, args.num_workers, output_dir, args.output_normalize,
        args.prefetch_factor
    )

    val_root = os.path.join(args.extract_dir, 'val')
    if os.path.isdir(val_root):
        extract_embeddings(
            'val', val_root,
            clip_model, dino_model, preprocess,
            args.batch_size, args.num_workers, output_dir, args.output_normalize,
            args.prefetch_factor
        )

    save_meta(output_dir, args.clip_model_name, args.vision_model,
              clip_dim, dino_dim, args.output_normalize,
              source='cc12m', seed=args.seed)

    print(f'\nDone. Saved to {output_dir}', flush=True)


def main():
    args = parse_args()

    if args.stage == 'extract':
        assert args.cc12m_shards, '--cc12m_shards required for extract stage'
        assert args.n_samples > 0, '--n_samples required for extract stage'
        extract_from_shards(args.cc12m_shards, args.extract_dir, args.n_samples, args.seed,
                            'train', args.num_readers, args.num_preprocessors, args.num_writers)
        if args.n_val_samples > 0:
            extract_from_shards(args.cc12m_shards, args.extract_dir, args.n_val_samples,
                                args.seed + 1, 'val',
                                args.num_readers, args.num_preprocessors, args.num_writers)
    elif args.stage == 'precompute':
        precompute(args)


if __name__ == '__main__':
    main()
