#!/usr/bin/env python3
"""
Compute EMA and EMA percentage metrics from existing tensorboard train/loss data.

Reads train/loss from tensorboard event files, computes:
  - train/loss_ema: Exponential moving average (decay=0.99)
  - train/loss_ema_pct: EMA as percentage of initial loss

Writes new metrics back to the same tensorboard directory.

Usage:
  python scripts/compute_ema_from_tb.py /path/to/experiment/tensorboard
  python scripts/compute_ema_from_tb.py /path/to/experiment1/tensorboard /path/to/experiment2/tensorboard
  
  # Process all experiments
  python scripts/compute_ema_from_tb.py /mnt/data/experiments/*/tensorboard

  # Custom decay
  python scripts/compute_ema_from_tb.py --decay 0.995 /path/to/experiment/tensorboard
"""
import argparse
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


def process_experiment(tb_dir, decay, ref_step, force=False):
    print(f'\nProcessing: {tb_dir}', flush=True)

    ea = EventAccumulator(tb_dir, size_guidance={'scalars': 0})
    ea.Reload()

    available_tags = ea.Tags().get('scalars', [])
    if 'train/loss' not in available_tags:
        print(f'  Skipping: train/loss not found. Available: {available_tags}')
        return

    if 'train/loss_ema' in available_tags and not force:
        existing = len(ea.Scalars('train/loss_ema'))
        total = len(ea.Scalars('train/loss'))
        if existing >= total:
            print(f'  Skipping: already processed ({existing}/{total} entries). Use --force to recompute.')
            return
        print(f'  Partial results found ({existing}/{total}), recomputing all...', flush=True)

    events = ea.Scalars('train/loss')
    if not events:
        print(f'  Skipping: no train/loss data')
        return

    print(f'  Found {len(events)} train/loss entries', flush=True)

    # Compute reference loss: average of 50 steps starting from ref_step
    ref_losses = [e.value for e in events if ref_step <= e.step < ref_step + 50]
    if not ref_losses:
        # Fallback: take closest 50 steps after ref_step
        ref_losses = [e.value for e in events if e.step >= ref_step][:50]
    assert ref_losses, f'No events found at or after step {ref_step}'
    loss_initial = sum(ref_losses) / len(ref_losses)

    # Compute EMA
    loss_ema = None
    ema_values = []

    for event in events:
        loss = event.value
        loss_ema = loss if loss_ema is None else decay * loss_ema + (1 - decay) * loss
        loss_ema_pct = loss_ema / loss_initial if loss_initial > 0 else 1.0
        ema_values.append((event.step, event.wall_time, loss_ema, loss_ema_pct))

    print(f'  Reference loss: avg of {len(ref_losses)} steps starting at step {ref_step} = {loss_initial:.6f}', flush=True)

    # Write new metrics
    writer = SummaryWriter(log_dir=tb_dir)
    for step, wall_time, ema, ema_pct in ema_values:
        writer.add_scalar('train/loss_ema', ema, step, walltime=wall_time)
        writer.add_scalar('train/loss_ema_pct', ema_pct, step, walltime=wall_time)
    writer.flush()
    writer.close()

    print(f'  Written {len(ema_values)} entries for train/loss_ema and train/loss_ema_pct')
    print(f'  Initial loss: {loss_initial:.6f}, Final EMA: {loss_ema:.6f}, Final EMA ratio: {loss_ema_pct:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tb_dirs', nargs='+', help='Tensorboard log directories')
    parser.add_argument('--decay', type=float, default=0.99, help='EMA decay factor')
    parser.add_argument('--ref_step', type=int, default=500, help='Step after which to compute reference loss (average of 50 steps)')
    parser.add_argument('--force', action='store_true', help='Recompute even if already processed')
    args = parser.parse_args()

    print(f'EMA decay: {args.decay}')
    for tb_dir in args.tb_dirs:
        try:
            process_experiment(tb_dir, args.decay, args.ref_step, args.force)
        except Exception as e:
            print(f'  Error: {e}', file=sys.stderr)

    print('\nDone.')


if __name__ == '__main__':
    main()
