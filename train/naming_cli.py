"""CLI helper: print experiment glob pattern for use in bash scripts.

Usage:
  python -m train.naming_cli --dataset cc12m --loss l2 --epochs 8 --batch_size 1024 \
      --penalty_weight 0.5 --lr 4e-5 --experiment_name cc12m-3m
"""
import argparse
from train.naming import make_experiment_glob


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clip_model_name', default='ViT-L-14')
    p.add_argument('--pretrained', default='openai')
    p.add_argument('--dataset', required=True)
    p.add_argument('--loss', default='l2')
    p.add_argument('--epochs', type=int, required=True)
    p.add_argument('--batch_size', type=int, required=True)
    p.add_argument('--penalty_weight', required=True)
    p.add_argument('--lr', required=True)
    p.add_argument('--experiment_name', required=True)
    p.add_argument('--dynamic_pw', default='0')
    p.add_argument('--dynamic_pw_target_ratio', default='0.5')
    p.add_argument('--dynamic_pw_cosine_decay', default='False')
    args = p.parse_args()
    print(make_experiment_glob(
        dataset=args.dataset, loss=args.loss, epochs=args.epochs,
        batch_size=args.batch_size, penalty_weight=args.penalty_weight,
        lr=args.lr, experiment_name=args.experiment_name,
        pretrained=args.pretrained, clip_model_name=args.clip_model_name,
        dynamic_pw=args.dynamic_pw, dynamic_pw_target_ratio=args.dynamic_pw_target_ratio,
        dynamic_pw_cosine_decay=args.dynamic_pw_cosine_decay,
    ))


if __name__ == '__main__':
    main()
