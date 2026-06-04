"""
CC12M dataset loader using webdataset.
Supports random subsampling to a fixed number of samples.
"""
import os
import glob
import random

import torch
import webdataset as wds
from torch.utils.data import IterableDataset


def create_cc12m_dataloader(shards_dir, transform, batch_size, n_samples=None,
                            num_workers=4, seed=0):
    """Create a DataLoader for CC12M webdataset shards.

    Subsamples to n_samples by selecting a proportional number of shards,
    then truncating to the exact count. Deterministic across epochs.

    Args:
        shards_dir: Directory containing .tar shard files.
        transform: Image transform to apply.
        batch_size: Batch size.
        n_samples: Number of samples to use. None = use all shards.
        num_workers: DataLoader workers.
        seed: Random seed for shard selection.

    Returns:
        DataLoader yielding (image_tensor, caption_string) batches.
    """
    tar_files = sorted(glob.glob(os.path.join(shards_dir, '*.tar')))
    assert len(tar_files) > 0, f'No .tar files found in {shards_dir}'
    print(f'Found {len(tar_files)} shards in {shards_dir}', flush=True)

    if n_samples is not None:
        # Estimate ~10K samples per shard (typical for CC12M)
        # Select enough shards to cover n_samples, then truncate
        samples_per_shard = _estimate_shard_size(tar_files[0])
        n_shards_needed = (n_samples // samples_per_shard) + 1
        n_shards_needed = min(n_shards_needed, len(tar_files))

        rng = random.Random(seed)
        tar_files = sorted(rng.sample(tar_files, n_shards_needed))
        print(f'Selected {n_shards_needed} shards (~{samples_per_shard} samples/shard) '
              f'for {n_samples} target samples', flush=True)

    urls = [f'file:{f}' for f in tar_files]

    dataset = (
        wds.WebDataset(urls, shardshuffle=False)
        .decode('pil')
        .to_tuple('jpg;png;webp', 'txt')
        .map_tuple(transform, lambda x: x)
    )

    if n_samples is not None:
        dataset = dataset.with_length(n_samples)

    loader = wds.WebLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                           pin_memory=True)

    if n_samples is not None:
        loader = loader.with_length(n_samples // batch_size)

    return loader


def _estimate_shard_size(tar_path):
    """Count samples in a single shard to estimate shard size."""
    count = 0
    ds = wds.WebDataset(f'file:{tar_path}').decode('pil')
    for _ in ds:
        count += 1
    return count
