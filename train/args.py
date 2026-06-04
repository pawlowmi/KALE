import argparse
from train.utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--clip_model_name', type=str, default='ViT-L-14', help='ViT-L-14, ViT-B-32')
    parser.add_argument('--pretrained', type=str, default='openai')
    parser.add_argument('--vision_model', type=str, default='dino')
    parser.add_argument('--output_normalize', type=str2bool, default=False, help='Whether the embedding is normalized')

    # Dataset
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--template', type=str, default='std')
    parser.add_argument('--imagenet_root', type=str, default='/mnt/datasets/imagenet', help='Imagenet dataset root directory')
    parser.add_argument('--eval_root', type=str, default='', help='Root for eval dataset (default: imagenet_root)')
    parser.add_argument('--imagenet21k_root', type=str, default='/mnt/datasets/imagenet', help='Imagenet dataset root directory')
    parser.add_argument('--cc12m_shards', type=str, default='', help='CC12M webdataset shards directory')
    parser.add_argument('--n_train_samples', type=int, default=0, help='Subsample training set to N samples (0 = use all)')

    # Training
    parser.add_argument('--start_step', type=int, default=0, help='Start step for training')
    parser.add_argument('--optimizer_state', type=str, default='', help='Optimizer state file path')
    parser.add_argument('--steps', type=int, default=0, help='Number of training steps (mutually exclusive with --epochs)')
    parser.add_argument('--epochs', type=int, default=0, help='Number of training epochs (mutually exclusive with --steps)')
    parser.add_argument('--warmup', type=int, default=0, help='Warmup steps (overrides --warmup_pct if set)')
    parser.add_argument('--warmup_pct', type=float, default=10.0, help='Warmup as percentage of total steps (used when --warmup is 0)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--enable_bs_scaling', type=str2bool, default=False, help='Scale batch size linearly with number of GPUs')
    parser.add_argument('--opt', type=str, default='adamw', help='Optimizer type; sgd, adamw')
    parser.add_argument('--momentum_sgd', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--lr_min_pct', type=float, default=10.0, help='Minimum LR as percentage of peak LR')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--trades', type=str2bool, default=False, help='Use TRADES')

    # Loss
    parser.add_argument('--loss', type=str, default='l2', help='ce, l2')
    parser.add_argument('--loss_clean', type=str, default='none', help='cosine, ce, l2')
    parser.add_argument('--inner_loss', type=str, default='l2', help='Inner loss function for adversarial training')
    parser.add_argument('--penalty_weight', type=float, default=1, help='Weight for penalty loss')
    parser.add_argument('--dynamic_pw', type=int, default=0, help='Update penalty_weight every N steps to match target ratio (0=disabled)')
    parser.add_argument('--dynamic_pw_target_ratio', type=float, default=0.5, help='Target ratio of effective alignment to effective clean loss')
    parser.add_argument('--clean_weight', type=float, default=1, help='Weight for clean loss')

    # Kernel
    parser.add_argument('--band_clip', type=float, default=8.7, help='Bandwidth for calculating kernel matrix for clip')
    parser.add_argument('--band_dino', type=float, default=15.8, help='Bandwidth for calculating kernel matrix for dino')
    parser.add_argument('--gamma', type=float, default=0.0007682, help='kernel function for clip')
    parser.add_argument('--coef0', type=float, default=0.8846858, help='kernel function for clip')
    parser.add_argument('--kernel_dino', type=str, default='gaussian', help='kernel function for dino')
    parser.add_argument('--kernel_clip', type=str, default='gaussian', help='kernel function for clip')

    # Data loading
    parser.add_argument('--dataloader_num_workers', type=int, default=4, help='Number of DataLoader workers per GPU')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of batches each DataLoader worker pre-loads ahead')
    parser.add_argument('--precomputed_dir', type=str, default='', help='Path to precomputed embeddings (skip loading frozen models)')

    # Logging
    parser.add_argument('--wandb', type=str2bool, default=True, help='Use Weights & Biases for logging')
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--log_freq', type=int, default=1, help='Logging frequency')
    parser.add_argument('--eval_freq', type=int, default=50, help='Evaluation frequency')
    parser.add_argument('--enhanced_metrics', type=str2bool, default=False, help='Log kernel sparsity metrics each step')
    parser.add_argument('--lam', type=float, default=1e-4, help='Threshold for kernel sparsity metrics')

    # Output
    parser.add_argument('--output_dir', type=str, default='', help='Output directory')
    parser.add_argument('--overwrite', type=str2bool, default=False, help='Overwrite existing directory')
    parser.add_argument('--resume', type=str, default='', help='Path to experiment dir to resume from (auto-loads latest checkpoint)')
    parser.add_argument('--save_checkpoints', type=str2bool, default=True, help='Save 10 training checkpoints')

    # Device
    parser.add_argument('--devices', type=str, default='', help='Device IDs for CUDA')

    return parser.parse_args()


def print_args(args):
    print(f"Arguments:\n{'-' * 20}")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print(f"{'-' * 20}")
