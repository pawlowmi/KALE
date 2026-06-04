"""
Precompute CLIP-original and DINOv2 embeddings for ImageNet.
Uses DataParallel across all available GPUs for maximum throughput.
"""
import argparse, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import open_clip

parser = argparse.ArgumentParser()
parser.add_argument('--imagenet_root', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--output_normalize', action='store_true', default=False)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPUs")

# Load CLIP
print("Loading CLIP...")
clip_model, image_processor = open_clip.create_model_from_pretrained('ViT-L-14', pretrained='openai')
clip_visual = clip_model.visual.eval()
del clip_model
clip_visual = nn.DataParallel(clip_visual).cuda()

# Load DINOv2
print("Loading DINOv2...")
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
dino_model.eval()
dino_model = nn.DataParallel(dino_model).cuda()

# Transforms
preprocess = transforms.Compose(image_processor.transforms[:-1])
normalize = image_processor.transforms[-1]
del image_processor

def extract(split_name, root):
    dataset = ImageFolder(root=root, transform=preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=16, pin_memory=True, drop_last=False)
    n = len(dataset)
    print(f"\n{split_name}: {n} images, batch_size={args.batch_size}")

    clip_embs = np.zeros((n, 768), dtype=np.float32)
    dino_embs = np.zeros((n, 1024), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)

    idx = 0
    with torch.no_grad():
        for i, (imgs, tgts) in enumerate(loader):
            bs = imgs.shape[0]
            imgs_norm = normalize(imgs).cuda(non_blocking=True)

            c_emb = clip_visual(imgs_norm)
            d_emb = dino_model(imgs_norm)

            if args.output_normalize:
                c_emb = F.normalize(c_emb, dim=-1)
                d_emb = F.normalize(d_emb, dim=-1)

            clip_embs[idx:idx+bs] = c_emb.cpu().numpy()
            dino_embs[idx:idx+bs] = d_emb.cpu().numpy()
            labels[idx:idx+bs] = tgts.numpy()
            idx += bs

            if (i+1) % 50 == 0:
                print(f"  [{idx}/{n}] {idx*100//n}%")

    np.save(os.path.join(args.output_dir, f'clip_orig_{split_name}.npy'), clip_embs)
    np.save(os.path.join(args.output_dir, f'dino_{split_name}.npy'), dino_embs)
    np.save(os.path.join(args.output_dir, f'labels_{split_name}.npy'), labels)
    print(f"  Saved: clip={clip_embs.shape}, dino={dino_embs.shape}")

extract('train', os.path.join(args.imagenet_root, 'train'))
extract('val', os.path.join(args.imagenet_root, 'val'))

meta = {
    "clip_model": "ViT-L-14", "clip_pretrained": "openai", "clip_dim": 768,
    "dino_model": "dinov2_vitl14_reg", "dino_dim": 1024,
    "output_normalize": args.output_normalize,
}
with open(os.path.join(args.output_dir, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)
print(f"\nDone. Saved to {args.output_dir}")
