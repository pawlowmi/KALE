"""WebDataset loader for DataComp shards.
Returns (image, caption_string) batches. The training loop checks
isinstance(targets, torch.Tensor) — strings make is_classification=False,
so class-label-dependent code paths are skipped.
"""
import os
import torch
import webdataset as wds
from torch.utils.data import DataLoader


def build_datacomp_dataloader(root, transform, batch_size, num_workers=8):
    shards = sorted([
        os.path.join(root, f) for f in os.listdir(root) if f.endswith('.tar')
    ])
    assert len(shards) > 0, f"No .tar shards found in {root}"

    dataset = (
        wds.WebDataset(shards, shardshuffle=True)
        .shuffle(10000)
        .decode("pil")
        .to_tuple("jpg;png;jpeg;webp", "txt", handler=wds.warn_and_continue)
        .map_tuple(transform, lambda x: x)
    )

    loader = DataLoader(
        dataset, batch_size=None, num_workers=num_workers, prefetch_factor=2,
    )
    return _BatchedLoader(loader, batch_size)


class _BatchedLoader:
    """Batches (image, caption) pairs from a webdataset stream."""
    def __init__(self, loader, batch_size):
        self.loader = loader
        self.batch_size = batch_size
        self._len = None

    def __iter__(self):
        batch_imgs, batch_caps = [], []
        for img, cap in self.loader:
            batch_imgs.append(img)
            batch_caps.append(cap)
            if len(batch_imgs) == self.batch_size:
                yield torch.stack(batch_imgs), batch_caps
                batch_imgs, batch_caps = [], []

    def __len__(self):
        if self._len is None:
            self._len = 128_000_000 // self.batch_size
        return self._len
