"""WebDataset loader for CC12M (pixparse/cc12m-wds) shards.
Same interface as datacomp_dataset but with correct dataset size (~11M).
"""
import os
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

CC12M_SIZE = 10_968_539


def build_cc12m_dataloader(root, transform, batch_size, num_workers=8):
    shards = sorted([
        os.path.join(root, f) for f in os.listdir(root) if f.endswith(".tar")
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
    def __init__(self, loader, batch_size):
        self.loader = loader
        self.batch_size = batch_size

    def __iter__(self):
        batch_imgs, batch_caps = [], []
        for img, cap in self.loader:
            batch_imgs.append(img)
            batch_caps.append(cap)
            if len(batch_imgs) == self.batch_size:
                yield torch.stack(batch_imgs), batch_caps
                batch_imgs, batch_caps = [], []

    def __len__(self):
        return CC12M_SIZE // self.batch_size
