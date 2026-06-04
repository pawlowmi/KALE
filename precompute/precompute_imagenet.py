"""
Precompute CLIP-original and DINOv2 embeddings for ImageNet.
Reuses model creation from train/models.py to guarantee identical embeddings.
"""
import argparse
import json
import os
import sys

sys.path.append("open_flamingo")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from train.models import load_clip_orig, load_vision_model, ClipVisionModel, wrap_vision_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--clip_model_name', type=str, default='ViT-L-14', help='CLIP architecture')
    parser.add_argument('--vision_model', type=str, default='dino', help='Vision model: dino, mlcd')
    parser.add_argument('--output_normalize', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=32, help='DataLoader workers')
    return parser.parse_args()


def extract(split_name, root, clip_model, dino_model, preprocess, args):
    print(f'\n=== {split_name} ===', flush=True)

    dataset = ImageFolder(root=root, transform=preprocess)
    n = len(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Probe dimensions
    with torch.no_grad():
        sample, _ = dataset[0]
        probe = sample.unsqueeze(0).cuda()
        clip_dim = clip_model(probe).shape[1]
        dino_dim = dino_model(probe).shape[1]

    clip_embs = np.zeros((n, clip_dim), dtype=np.float32)
    dino_embs = np.zeros((n, dino_dim), dtype=np.float32)

    print(f'Running inference on {n} images (clip_dim={clip_dim}, dino_dim={dino_dim}), '
          f'batch_size={args.batch_size}...', flush=True)

    idx = 0
    with torch.no_grad():
        for batch_imgs, _ in tqdm(loader, desc=f'Inference {split_name}', unit='batch'):
            batch = batch_imgs.cuda(non_blocking=True)
            bs = batch.shape[0]

            clip_out = clip_model(batch, output_normalize=args.output_normalize)
            dino_out = dino_model(batch, output_normalize=args.output_normalize)

            clip_embs[idx:idx + bs] = clip_out.cpu().numpy()
            dino_embs[idx:idx + bs] = dino_out.cpu().numpy()
            idx += bs

    labels = np.array([s[1] for s in dataset.samples], dtype=np.int64)
    np.save(os.path.join(args.output_dir, f'clip_orig_{split_name}.npy'), clip_embs)
    np.save(os.path.join(args.output_dir, f'dino_{split_name}.npy'), dino_embs)
    np.save(os.path.join(args.output_dir, f'labels_{split_name}.npy'), labels)
    print(f'Saved: clip={clip_embs.shape}, dino={dino_embs.shape}', flush=True)

    return clip_dim, dino_dim


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    print(f'Using {num_gpus} GPUs', flush=True)

    # Load models using the same functions as training
    model_orig, preprocess, normalize = load_clip_orig(args.clip_model_name)
    raw_dino = load_vision_model(args.vision_model)

    # Wrap with normalization — identical to training script
    clip_model = ClipVisionModel(model=model_orig.visual, normalize=normalize)
    dino_model = wrap_vision_model(raw_dino, args.clip_model_name, args.vision_model, normalize)
    del model_orig, raw_dino

    if num_gpus > 1:
        clip_model = nn.DataParallel(clip_model)
        dino_model = nn.DataParallel(dino_model)
    clip_model.cuda().eval()
    dino_model.cuda().eval()

    clip_dim, dino_dim = extract(
        'train', os.path.join(args.imagenet_root, 'train'),
        clip_model, dino_model, preprocess, args
    )
    extract(
        'val', os.path.join(args.imagenet_root, 'val'),
        clip_model, dino_model, preprocess, args
    )

    meta = {
        'clip_model': args.clip_model_name,
        'vision_model': args.vision_model,
        'clip_dim': clip_dim,
        'dino_dim': dino_dim,
        'output_normalize': args.output_normalize,
    }
    with open(os.path.join(args.output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\nDone. Saved to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
