"""
Shared utilities for precomputing embeddings.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from train.models import load_clip_orig, load_vision_model, ClipVisionModel, wrap_vision_model


def load_and_wrap_models(clip_model_name, vision_model):
    """Load CLIP and vision models, wrap with normalization, move to GPU."""
    num_gpus = torch.cuda.device_count()
    print(f'Using {num_gpus} GPUs', flush=True)
    print('Loading models...', flush=True)

    model_orig, preprocess, normalize, _ = load_clip_orig(clip_model_name)
    raw_dino = load_vision_model(vision_model)

    clip_model = ClipVisionModel(model=model_orig.visual, normalize=normalize)
    dino_model = wrap_vision_model(raw_dino, clip_model_name, vision_model, normalize)
    del model_orig, raw_dino

    if num_gpus > 1:
        clip_model = nn.DataParallel(clip_model)
        dino_model = nn.DataParallel(dino_model)
    clip_model.cuda().eval()
    dino_model.cuda().eval()

    return clip_model, dino_model, preprocess


def probe_dims(clip_model, dino_model, sample_tensor):
    """Run a single image through both models to get embedding dimensions."""
    with torch.no_grad():
        probe = sample_tensor.unsqueeze(0).cuda()
        clip_dim = clip_model(probe).shape[1]
        dino_dim = dino_model(probe).shape[1]
    return clip_dim, dino_dim


def extract_embeddings(split_name, root, clip_model, dino_model, preprocess,
                       batch_size, num_workers, output_dir, output_normalize=False,
                       prefetch_factor=2):
    """Extract CLIP and DINOv2 embeddings for all images in an ImageFolder."""
    print(f'\n=== {split_name} ===', flush=True)

    dataset = ImageFolder(root=root, transform=preprocess)
    n = len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False,
                        prefetch_factor=prefetch_factor if num_workers > 0 else None)

    sample, _ = dataset[0]
    clip_dim, dino_dim = probe_dims(clip_model, dino_model, sample)

    clip_embs = np.zeros((n, clip_dim), dtype=np.float32)
    dino_embs = np.zeros((n, dino_dim), dtype=np.float32)

    print(f'Running inference on {n} images (clip_dim={clip_dim}, dino_dim={dino_dim}), '
          f'batch_size={batch_size}...', flush=True)

    idx = 0
    with torch.no_grad():
        for batch_imgs, _ in tqdm(loader, desc=f'Inference {split_name}', unit='batch'):
            batch = batch_imgs.cuda(non_blocking=True)
            bs = batch.shape[0]

            clip_out = clip_model(batch, output_normalize=output_normalize)
            dino_out = dino_model(batch, output_normalize=output_normalize)

            clip_embs[idx:idx + bs] = clip_out.cpu().numpy()
            dino_embs[idx:idx + bs] = dino_out.cpu().numpy()
            idx += bs

    labels = np.array([s[1] for s in dataset.samples], dtype=np.int64)
    np.save(os.path.join(output_dir, f'clip_orig_{split_name}.npy'), clip_embs)
    np.save(os.path.join(output_dir, f'dino_{split_name}.npy'), dino_embs)
    np.save(os.path.join(output_dir, f'labels_{split_name}.npy'), labels)
    print(f'Saved: clip={clip_embs.shape}, dino={dino_embs.shape}', flush=True)

    return clip_dim, dino_dim


def save_meta(output_dir, clip_model_name, vision_model, clip_dim, dino_dim,
              output_normalize, **extra):
    """Save metadata JSON."""
    meta = {
        'clip_model': clip_model_name,
        'vision_model': vision_model,
        'clip_dim': clip_dim,
        'dino_dim': dino_dim,
        'output_normalize': output_normalize,
        **extra,
    }
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
