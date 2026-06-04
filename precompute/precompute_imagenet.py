"""
Precompute CLIP-original and DINOv2 embeddings for ImageNet.
"""
import argparse
import os
import sys

sys.path.append("open_flamingo")

from precompute.common import load_and_wrap_models, extract_embeddings, save_meta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--clip_model_name', type=str, default='ViT-L-14', help='CLIP architecture')
    parser.add_argument('--vision_model', type=str, default='dino', help='Vision model: dino, mlcd')
    parser.add_argument('--output_normalize', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=32, help='DataLoader workers')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Batches each worker pre-loads ahead')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    clip_model, dino_model, preprocess = load_and_wrap_models(
        args.clip_model_name, args.vision_model
    )

    clip_dim, dino_dim = extract_embeddings(
        'train', os.path.join(args.imagenet_root, 'train'),
        clip_model, dino_model, preprocess,
        args.batch_size, args.num_workers, args.output_dir, args.output_normalize,
        args.prefetch_factor
    )
    extract_embeddings(
        'val', os.path.join(args.imagenet_root, 'val'),
        clip_model, dino_model, preprocess,
        args.batch_size, args.num_workers, args.output_dir, args.output_normalize,
        args.prefetch_factor
    )

    save_meta(args.output_dir, args.clip_model_name, args.vision_model,
              clip_dim, dino_dim, args.output_normalize)

    print(f'\nDone. Saved to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
