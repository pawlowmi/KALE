import sys
sys.path.append("open_flamingo")

import os
import shutil
import time
import string
import random
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
from training.scheduler import cosine_lr
from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL
from open_flamingo.eval.models.utils import unwrap_model

from train.args import parse_args, print_args
from train.datasets import ImageNetDataset
from train.indexed_dataset import IndexedImageFolder
from train.cc12m_dataset import create_cc12m_dataloader
from train.models import ClipVisionModel, load_clip_orig, load_vision_model, wrap_vision_model
from train.utils import init_wandb, AverageMeter, compute_text_embeddings
from train.metrics import ClipDriftMetric
from train.naming import make_experiment_name
from CLIP_eval.eval_utils import load_clip_model


# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def cleanup_ddp():
    dist.destroy_process_group()


def is_main():
    return dist.get_rank() == 0


def all_reduce_mean(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


# ── Output dir ────────────────────────────────────────────────────────────────

def setup_output_dir(output_dir, overwrite, args, resume=False):
    if resume:
        assert os.path.isdir(output_dir), f'Resume dir not found: {output_dir}'
        return
    if overwrite:
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=False)
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        f.write(str(args))


def find_latest_checkpoint(checkpoint_dir):
    import re
    best_step, best_model, best_opt = -1, None, None
    for f in os.listdir(checkpoint_dir):
        m = re.match(r'(?:fallback|step)_(\d+)\.pt$', f)
        if m:
            step = int(m.group(1))
            opt_path = os.path.join(checkpoint_dir, f.replace('.pt', '_opt.pt'))
            if step > best_step and os.path.exists(opt_path):
                best_step, best_model, best_opt = step, os.path.join(checkpoint_dir, f), opt_path
    return best_step, best_model, best_opt


# ── Text embeddings ───────────────────────────────────────────────────────────

def build_imagenet_text_embeddings(clip_model, tokenizer, template_name, device):
    if template_name == 'std':
        template = 'This is a photo of a {}'
    elif template_name == 'blurry':
        template = 'This is a blurry photo of a {}'
    else:
        raise ValueError(f'Unknown template: {template_name}')
    print(f'template: {template}')
    texts = [template.format(c) for c in IMAGENET_1K_CLASS_ID_TO_LABEL.values()]
    return compute_text_embeddings(clip_model, texts, tokenizer, device=device)


# ── Loss helpers ──────────────────────────────────────────────────────────────

def compute_loss(loss_str, embedding, targets, embedding_orig, logit_scale,
                 embedding_text_labels_norm=None, reduction='mean'):
    if loss_str == 'cosine':
        return 1 - F.cosine_similarity(embedding, embedding_orig).mean()
    elif loss_str == 'l2':
        return l2(out=embedding, targets=embedding_orig, reduction=reduction)
    elif loss_str == 'ce':
        return ce(out=embedding @ (logit_scale * embedding_text_labels_norm),
                  targets=targets, reduction=reduction)
    else:
        raise ValueError(f'loss {loss_str} not supported')


def l2(out, targets, reduction='none'):
    assert out.shape == targets.shape
    assert out.shape[0] > 1
    squared_error_batch = F.mse_loss(out, targets, reduction='none')
    if reduction == 'mean':
        return torch.mean(squared_error_batch.sum(dim=1))
    return squared_error_batch.sum(dim=1)


def ce(out, targets, reduction='mean'):
    assert out.shape[0] == targets.shape[0]
    assert out.shape[0] > 1
    return F.cross_entropy(out, targets, reduction=reduction)


@torch.no_grad()
def compute_acc(logits, targets):
    preds = logits.max(dim=1)[1]
    return (preds.eq(targets).sum() / targets.shape[0]).item() * 100


def compute_kernel_sparsity(diff_matrix, threshold=1e-4):
    total = diff_matrix.numel()
    mask = diff_matrix > threshold
    n_above = mask.sum().item()
    return {
        'sparsity/mean_above_thresh': diff_matrix[mask].mean().item() if n_above > 0 else 0.0,
        'sparsity/ratio_above_thresh': n_above / total,
        'sparsity/ratio_below_thresh': (total - n_above) / total,
        'sparsity/masked_loss': (diff_matrix * mask).sum().item() / max(n_above, 1),
    }


# ── Dynamic penalty weight ────────────────────────────────────────────────────

class DynamicPenaltyWeight:
    def __init__(self, update_every, target_ratio, initial_pw, ema_decay=0.99,
                 cosine_decay=False, total_steps=None, target_ratio_min=0.3):
        self.update_every = update_every
        self.target_ratio = target_ratio
        self.pw = initial_pw
        self.ema_decay = ema_decay
        self.ema_kernel = None
        self.ema_clean = None
        # Cosine decay of target_ratio: from target_ratio -> target_ratio_min over training
        self.cosine_decay = cosine_decay
        self.total_steps = total_steps
        self.target_ratio_min = target_ratio_min

    def _current_target(self, step_total):
        if not self.cosine_decay or self.total_steps is None:
            return self.target_ratio
        import math
        step_ratio = 0.5 * (1 + math.cos(math.pi * step_total / self.total_steps))
        return self.target_ratio * step_ratio + self.target_ratio_min * (1 - step_ratio)

    def update_ema(self, loss_kernel, loss_clean):
        if self.ema_kernel is None:
            self.ema_kernel, self.ema_clean = loss_kernel, loss_clean
        else:
            self.ema_kernel = self.ema_decay * self.ema_kernel + (1 - self.ema_decay) * loss_kernel
            self.ema_clean = self.ema_decay * self.ema_clean + (1 - self.ema_decay) * loss_clean

    def step(self, step_total, loss_kernel, loss_clean, clean_weight, warmup_steps=0):
        self.update_ema(loss_kernel, loss_clean)
        if step_total > warmup_steps and step_total > 0 and step_total % self.update_every == 0:
            if self.ema_kernel > 0 and self.ema_clean > 0 and self.pw > 0:
                current_ratio = self.pw * self.ema_kernel / (clean_weight * self.ema_clean)
                target = self._current_target(step_total)
                if current_ratio > 0:
                    correction = (target / current_ratio) ** 0.5
                    self.pw *= correction
                    if is_main() if dist.is_initialized() else True:
                        print(f'[dynamic-pw] step={step_total} pw={self.pw:.4f} '
                              f'ratio={current_ratio:.6f} target={target:.6f}',
                              flush=True)
        return self.pw


# ── main ──────────────────────────────────────────────────────────────────────

def main(args, rank, local_rank, world_size):
    device = torch.device(f'cuda:{local_rank}')

    if is_main():
        if args.wandb:
            init_wandb(project_name='clip-finetune', model_name=args.finetuned_model_name,
                       config=vars(args))
        else:
            wandb.init(mode='disabled')
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    else:
        wandb.init(mode='disabled')
        tb_writer = None

    csv_log_path = os.path.join(args.output_dir, 'train_log.csv')
    resuming = bool(args.resume)
    if is_main() and not resuming:
        with open(csv_log_path, 'w') as f:
            header = 'step,lr,loss,loss_total,cos_sim,emb_var,emb_var_global,emb_erank,acc,eval_acc'
            if args.enhanced_metrics:
                header += ',mean_above_thresh,ratio_above_thresh,ratio_below_thresh,masked_loss'
            f.write(header + '\n')

    if is_main():
        print_args(args)
        setup_output_dir(args.output_dir, args.overwrite, args, resume=resuming)
    dist.barrier()

    # Models
    model_orig, preprocessor_without_normalize, normalize, tokenizer = load_clip_orig(args.clip_model_name)
    model, _, _ = load_clip_model(args.clip_model_name, args.pretrained)

    if is_main():
        print(f'[preprocessor_without_normalize] {preprocessor_without_normalize}')

    # DataLoader — each rank gets its own shard via DistributedSampler
    num_workers = args.dataloader_num_workers

    def worker_init(worker_id):
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    if args.dataset == 'cc12m':
        if args.cc12m_shards:
            dataloader = create_cc12m_dataloader(
                shards_dir=args.cc12m_shards, transform=preprocessor_without_normalize,
                batch_size=args.batch_size, num_workers=num_workers, seed=rank)
        else:
            dataset = IndexedImageFolder(root=args.imagenet_root + '/train',
                                         transform=preprocessor_without_normalize)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                    num_workers=num_workers, pin_memory=True, drop_last=True,
                                    persistent_workers=num_workers > 0,
                                    prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
                                    worker_init_fn=worker_init)
    elif args.dataset == 'imagenet':
        if args.precomputed_dir:
            dataset = IndexedImageFolder(root=args.imagenet_root + '/train',
                                         transform=preprocessor_without_normalize)
        else:
            dataset = ImageNetDataset(root=args.imagenet_root + '/train',
                                      transform=preprocessor_without_normalize)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                num_workers=num_workers, pin_memory=True, drop_last=True,
                                persistent_workers=num_workers > 0,
                                prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
                                worker_init_fn=worker_init)
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')

    eval_root = args.eval_root if args.eval_root else args.imagenet_root
    dataset_eval = ImageNetDataset(root=eval_root + '/val', transform=preprocessor_without_normalize)
    eval_sampler = DistributedSampler(dataset_eval, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, sampler=eval_sampler,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 persistent_workers=num_workers > 0,
                                 prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
                                 worker_init_fn=worker_init)

    embedding_text_labels_norm = build_imagenet_text_embeddings(
        model_orig, tokenizer, args.template, device=device)

    # Precomputed embeddings — loaded on CPU, indexed per-batch
    precomputed_orig = precomputed_dino = None
    if args.precomputed_dir:
        del model_orig
        torch.cuda.empty_cache()
        if is_main():
            print(f'Loading precomputed embeddings from {args.precomputed_dir}...', flush=True)
        precomputed_orig = {
            'train': torch.from_numpy(np.load(os.path.join(args.precomputed_dir, 'clip_orig_train.npy'))),
            'val':   torch.from_numpy(np.load(os.path.join(args.precomputed_dir, 'clip_orig_val.npy'))),
        }
        precomputed_dino = {
            'train': torch.from_numpy(np.load(os.path.join(args.precomputed_dir, 'dino_train.npy'))),
            'val':   torch.from_numpy(np.load(os.path.join(args.precomputed_dir, 'dino_val.npy'))),
        }
        if is_main():
            print(f'  clip_orig_train: {precomputed_orig["train"].shape}')
            print(f'  dino_train: {precomputed_dino["train"].shape}')
    else:
        model_orig = ClipVisionModel(model=model_orig.visual, normalize=normalize).to(device)
        model_dino = load_vision_model(args.vision_model)
        model_dino = wrap_vision_model(model_dino, args.clip_model_name, args.vision_model, normalize).to(device)
        model_dino.eval()

    # Trainable model
    model = ClipVisionModel(model=model.visual, normalize=normalize).to(device)
    model = DDP(model, device_ids=[local_rank], bucket_cap_mb=200)

    # CLIP drift metric — rank 0 only, uses a fresh frozen reference model
    drift_metric = None
    if is_main() and args.drift_freq > 0:
        frozen_ref, _, frozen_normalize, _ = load_clip_orig(args.clip_model_name)
        frozen_ref = ClipVisionModel(model=frozen_ref.visual, normalize=frozen_normalize).to(device)
        frozen_ref.eval()
        drift_metric = ClipDriftMetric(
            dataset=dataset_eval, frozen_model=frozen_ref, device=device,
            n_samples=256, seed=0, freq=args.drift_freq)
        if is_main():
            print(f'ClipDriftMetric: 256 fixed samples, every {args.drift_freq} steps', flush=True)

    # Kernel projector (learnable) — initialize directly on device to keep leaf status
    if args.kernel_clip == 'gaussian':
        projector_clip = torch.nn.Parameter(
            torch.tensor(args.band_clip, device=device), requires_grad=True)
    else:
        projector_clip = torch.nn.Parameter(
            torch.tensor([args.gamma, args.coef0], device=device), requires_grad=True)
    band_dino = args.band_dino

    param_groups = [
        {'params': unwrap_model(model).model.parameters(), 'weight_decay': args.wd},
        {'params': [projector_clip], 'weight_decay': 0},
    ]
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum_sgd)
    else:
        raise ValueError(f'Optimizer {args.opt} not supported.')

    # Resume
    if resuming:
        ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
        resume_step, resume_model, resume_opt = find_latest_checkpoint(ckpt_dir)
        assert resume_step >= 0, f'No checkpoint found in {ckpt_dir}'
        if is_main():
            print(f'Resuming from step {resume_step}: {resume_model}', flush=True)
        map_loc = {'cuda:0': f'cuda:{local_rank}'}
        unwrap_model(model).model.load_state_dict(torch.load(resume_model, map_location=map_loc))
        optimizer.load_state_dict(torch.load(resume_opt, map_location=map_loc))
        args.start_step = resume_step

    # Scheduler
    assert (args.steps > 0) != (args.epochs > 0), 'Specify exactly one of --steps or --epochs'
    if args.epochs > 0:
        args.steps = args.epochs * len(dataloader)
        if is_main():
            print(f'Computed {args.steps} steps from {args.epochs} epochs x {len(dataloader)} batches')

    warmup_steps = args.warmup if args.warmup > 0 else int(args.steps * args.warmup_pct / 100)
    lr_min = args.lr * args.lr_min_pct / 100
    _base_scheduler = cosine_lr(optimizer, args.lr, warmup_steps, args.steps)

    def scheduler(step):
        _base_scheduler(step)
        for pg in optimizer.param_groups:
            if pg['lr'] < lr_min:
                pg['lr'] = lr_min

    if is_main():
        print(f'Warmup: {warmup_steps} steps ({warmup_steps * 100 / args.steps:.1f}% of {args.steps})')
        print(f'LR: 0 -> {args.lr} -> {lr_min} (cosine)')
        print(f'train for {args.steps / len(dataloader):.2f} epochs')

    args.total_epochs = args.steps / len(dataloader)

    dynamic_pw = None
    if args.dynamic_pw > 0:
        dynamic_pw = DynamicPenaltyWeight(
            args.dynamic_pw, args.dynamic_pw_target_ratio, args.penalty_weight,
            cosine_decay=args.dynamic_pw_cosine_decay,
            total_steps=args.steps,
            target_ratio_min=args.dynamic_pw_target_ratio_min,
        )
        if resuming:
            dpw_path = os.path.join(args.output_dir, 'checkpoints', 'dynamic_pw.json')
            if os.path.exists(dpw_path):
                import json
                state = json.load(open(dpw_path))
                dynamic_pw.pw = state['pw']
                dynamic_pw.ema_kernel = state.get('ema_kernel')
                dynamic_pw.ema_clean = state.get('ema_clean')
                if is_main():
                    print(f'Restored dynamic_pw: pw={dynamic_pw.pw:.4f}', flush=True)

    step_total = args.start_step
    epoch = 0
    loss_ema, loss_initial, loss_initial_buf = None, None, []
    scaler = torch.cuda.amp.GradScaler(enabled=args.bf16)
    while step_total < args.steps:
        if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        step_total, loss_ema, loss_initial, loss_initial_buf = train_one_epoch(
            step_total, model=model,
            model_orig=model_orig if not args.precomputed_dir else None,
            model_dino=model_dino if not args.precomputed_dir else None,
            projector_clip=projector_clip, band_dino=band_dino,
            dataloader=dataloader, dataloader_eval=dataloader_eval,
            optimizer=optimizer, scheduler=scheduler,
            embedding_text_labels_norm=embedding_text_labels_norm,
            normalize=normalize, args=args, epoch=epoch,
            warmup_steps=warmup_steps, dynamic_pw=dynamic_pw,
            tb_writer=tb_writer, precomputed_orig=precomputed_orig,
            precomputed_dino=precomputed_dino, csv_log_path=csv_log_path,
            device=device, drift_metric=drift_metric,
            loss_ema=loss_ema, loss_initial=loss_initial, loss_initial_buf=loss_initial_buf,
            scaler=scaler,
        )
        if is_main():
            print(f'Epoch {epoch} done.')
            if args.save_epoch_checkpoints:
                torch.save(unwrap_model(model).model.state_dict(),
                           f'{args.output_dir}/checkpoints/epoch_{epoch}.pt')
                torch.save(optimizer.state_dict(),
                           f'{args.output_dir}/checkpoints/epoch_{epoch}_opt.pt')
        epoch += 1

    if is_main():
        if tb_writer:
            tb_writer.flush()
            tb_writer.close()
        torch.save(unwrap_model(model).model.state_dict(),
                   f'{args.output_dir}/checkpoints/final.pt')
        torch.save(optimizer.state_dict(),
                   f'{args.output_dir}/checkpoints/final_opt.pt')
        if args.output_dir.endswith('_temp'):
            os.rename(args.output_dir, args.output_dir[:-5])

    cleanup_ddp()


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
        step_total, model, model_orig, model_dino, projector_clip, band_dino,
        dataloader, dataloader_eval, optimizer, scheduler, normalize,
        embedding_text_labels_norm, args, epoch, warmup_steps,
        dynamic_pw=None, tb_writer=None, precomputed_orig=None,
        precomputed_dino=None, csv_log_path=None, device=None, drift_metric=None,
        loss_ema=None, loss_initial=None, loss_initial_buf=None, scaler=None,
):
    if model_orig is not None:
        model_orig.eval()
    model.train()

    def _cycle_eval():
        while True:
            yield from dataloader_eval
    eval_iter = _cycle_eval()

    loss_meter = AverageMeter('loss')
    cos_sim_meter = AverageMeter('cos-sim')
    acc_meter = AverageMeter('acc')
    if loss_initial_buf is None:
        loss_initial_buf = []
    ema_decay = 0.99
    epoch_start_time = time.time()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', unit='batch', mininterval=1.0,
                disable=not is_main())

    for i, batch in enumerate(pbar):
        if len(batch) == 3:
            data, targets, indices = batch
        else:
            data, targets = batch
            indices = None

        is_classification = isinstance(targets, torch.Tensor)
        data = data.to(device, non_blocking=True)
        n_samples = data.shape[0]
        if is_classification:
            targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            if precomputed_orig is not None and indices is not None:
                # Each rank indexes its own slice — kernel stays local, no gather
                embedding_orig = precomputed_orig['train'][indices].to(device, non_blocking=True)
                embedding_dino = precomputed_dino['train'][indices].to(device, non_blocking=True)
            else:
                embedding_orig = model_orig(vision=data, output_normalize=args.output_normalize)
                embedding_dino = model_dino(vision=data, output_normalize=args.output_normalize)

            # Kernel computed on local rank's batch — no GPU 0 bottleneck
            if args.kernel_dino == 'gaussian':
                k_dino = torch.cdist(embedding_dino, embedding_dino, p=2.0)
                k_dino = torch.exp(-k_dino ** 2 / (2 * band_dino ** 2))
            elif args.kernel_dino == 'polynomial':
                n_feat = embedding_dino.shape[1]
                k_dino = (embedding_dino @ embedding_dino.T / n_feat + 1.) ** 3
                diag = k_dino.diag().clamp(min=1e-8)
                k_dino = k_dino / torch.sqrt(diag.view(-1, 1) * diag.view(1, -1))
            elif args.kernel_dino == 'cosine':
                norm_X = F.normalize(embedding_dino, dim=1)
                k_dino = norm_X @ norm_X.T
            del embedding_dino

        model.train()
        amp_dtype = torch.bfloat16 if args.bf16 else torch.float32
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=args.bf16):
            embedding_clean = model(data, output_normalize=args.output_normalize)

            loss_clean = compute_loss(
                loss_str=args.loss_clean, embedding=embedding_clean, targets=targets,
                embedding_orig=embedding_orig, logit_scale=100.)

            if args.kernel_clip == 'gaussian':
                k_clip = torch.cdist(embedding_clean, embedding_clean, p=2.0)
                k_clip = torch.exp(-k_clip ** 2 / (2 * projector_clip ** 2))
            elif args.kernel_clip == 'polynomial':
                k_clip = ((embedding_clean @ embedding_clean.T) * projector_clip[0] + projector_clip[1]) ** 3
                diag = k_clip.diag().clamp(min=1e-8)
                k_clip = k_clip / torch.sqrt(diag.view(-1, 1) * diag.view(1, -1))
            elif args.kernel_clip == 'cosine':
                norm_X = F.normalize(embedding_clean, dim=1)
                k_clip = norm_X @ norm_X.T

            diff_sq = (k_clip - k_dino) ** 2
            sparsity_metrics = {}
            if args.enhanced_metrics:
                with torch.no_grad():
                    sparsity_metrics = compute_kernel_sparsity(diff_sq, threshold=args.lam)

            loss = torch.mean(diff_sq)

        with torch.no_grad():
            current_pw = args.penalty_weight
            if dynamic_pw is not None:
                current_pw = dynamic_pw.step(
                    step_total, loss.item(), loss_clean.item(), args.clean_weight, warmup_steps)
            eff_clean = args.clean_weight * loss_clean.item()
            loss_ratio = (current_pw * loss.item()) / eff_clean if eff_clean > 0 else 0.0

        loss_total = args.clean_weight * loss_clean + current_pw * loss
        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        step_total += 1
        scheduler(step_total)

        with torch.no_grad():
            cos_sim_clean = F.cosine_similarity(embedding_clean, embedding_orig, dim=1).mean()
            # Variance on rank 0's local batch (128 samples)
            emb_var = embedding_clean.var(dim=0).mean()
            # Effective rank on rank 0's local batch
            sv = torch.linalg.svdvals(embedding_clean.float())
            sv = sv.clamp(min=1e-8)
            p = sv / sv.sum()
            emb_erank = torch.exp(-(p * p.log()).sum())
            # Full-batch variance across all ranks via parallel variance formula
            n = embedding_clean.shape[0]
            local_mean = embedding_clean.mean(dim=0)
            local_var = embedding_clean.var(dim=0, unbiased=False)
            global_mean = local_mean.clone()
            dist.all_reduce(global_mean, op=dist.ReduceOp.SUM)
            global_mean /= world_size
            # E[X^2] - E[X]^2 across ranks: combine via sum of (var + mean^2), then subtract global_mean^2
            local_mean_sq_var = local_var + local_mean ** 2
            dist.all_reduce(local_mean_sq_var, op=dist.ReduceOp.SUM)
            emb_var_global = (local_mean_sq_var / world_size - global_mean ** 2).mean()
            if is_classification:
                embedding_clean_norm = F.normalize(embedding_clean, dim=1)
                logits_clean = embedding_clean_norm @ embedding_text_labels_norm.to(embedding_clean.dtype)
                acc = compute_acc(logits_clean, targets)
                acc_meter.update(acc, n_samples)
            else:
                acc = None

        loss_meter.update(loss.item(), n_samples)
        if loss_initial is None and step_total > warmup_steps:
            loss_initial_buf.append(loss.item())
            if len(loss_initial_buf) >= 50:
                loss_initial = sum(loss_initial_buf) / len(loss_initial_buf)
        loss_ema = loss.item() if loss_ema is None else ema_decay * loss_ema + (1 - ema_decay) * loss.item()
        loss_ema_pct = loss_ema / loss_initial if loss_initial and loss_initial > 0 else 1.0
        cos_sim_meter.update(cos_sim_clean.item(), n_samples)

        # Eval — only rank 0 runs eval (no DistributedSampler needed for quick eval)
        eval_logs = {}
        if is_main() and (step_total - 1) % args.eval_freq == 0:
            model.eval()
            eval_batch = next(eval_iter)
            data_eval, targets_eval = eval_batch[0].to(device), eval_batch[1].to(device)
            with torch.no_grad():
                embedding_eval_norm = model(data_eval, output_normalize=True)
                logits_eval = embedding_eval_norm @ embedding_text_labels_norm.to(embedding_eval_norm.dtype)
                acc_eval = compute_acc(logits_eval, targets_eval)
            print(f'[eval-acc] {acc_eval:.2f}')
            eval_logs['eval/acc'] = acc_eval
            model.train()
            del data_eval, targets_eval, embedding_eval_norm, logits_eval

        # Drift metric — rank 0 only
        if is_main() and drift_metric is not None and drift_metric.should_run(step_total):
            drift_logs = drift_metric.compute(unwrap_model(model))
            eval_logs.update(drift_logs)
            print(f'[drift] l2={drift_logs["drift/l2"]:.4f} angle={drift_logs["drift/angle_deg"]:.2f}°')

        if is_main():
            lr_ = optimizer.param_groups[0].get('lr')
            postfix = {'lr': f'{lr_:.6f}', 'loss': f'{loss.item():.4f}'}
            if acc is not None:
                postfix['acc'] = f'{acc:.1f}'
            if 'eval/acc' in eval_logs:
                postfix['eval_acc'] = f'{eval_logs["eval/acc"]:.1f}'
            pbar.set_postfix(postfix)

            if (step_total - 1) % args.log_freq == 0:
                log_data = {
                    'step': step_total, 'lr': lr_,
                    'loss': loss.item(), 'loss_clean': loss_clean.item(),
                    'loss-total': loss_total.item(), 'loss_ratio': loss_ratio,
                    'penalty_weight': current_pw,
                    'cos-sim-clean': cos_sim_clean.item(),
                    'emb_var': emb_var.item(),
                    'emb_var_global': emb_var_global.item(),
                    'emb_erank': emb_erank.item(),
                    'acc': acc, 'avg/loss': loss_meter.avg, 'avg/acc': acc_meter.avg,
                    'loss_ema': loss_ema, 'loss_ema_pct': loss_ema_pct,
                }
                log_data.update(eval_logs)
                if args.enhanced_metrics:
                    log_data.update(sparsity_metrics)
                if (step_total - 1) % (args.log_freq * 10) == 0:
                    batch_avg_time = (time.time() - epoch_start_time) / (i + 1) / 3600
                    epoch_avg_time = batch_avg_time * len(dataloader)
                    total_remaining = epoch_avg_time * (args.total_epochs - epoch - i / len(dataloader))
                    print(f'[epoch avg time] {epoch_avg_time:.2f}h [total remaining] {total_remaining:.2f}h')
                    log_data.update({
                        'time/total-remaining': total_remaining,
                        'time/epoch-average-time': epoch_avg_time,
                        'other/epoch': epoch + i / len(dataloader),
                    })
                wandb.log(log_data)
                if tb_writer is not None:
                    for k, v in [('train/loss', loss.item()), ('train/loss_clean', loss_clean.item()),
                                 ('train/loss_total', loss_total.item()), ('train/loss_ratio', loss_ratio),
                                 ('train/penalty_weight', current_pw), ('train/lr', lr_),
                                 ('train/cos_sim', cos_sim_clean.item()),
                                 ('train/emb_var', emb_var.item()),
                                 ('train/emb_var_global', emb_var_global.item()),
                                 ('train/emb_erank', emb_erank.item()),
                                 ('train/loss_ema', loss_ema), ('train/loss_ema_pct', loss_ema_pct)]:
                        tb_writer.add_scalar(k, v, step_total)
                    if acc is not None:
                        tb_writer.add_scalar('train/acc', acc, step_total)
                    if args.enhanced_metrics:
                        for k, v in sparsity_metrics.items():
                            tb_writer.add_scalar(k, v, step_total)
                    if 'eval/acc' in eval_logs:
                        tb_writer.add_scalar('eval/acc', eval_logs['eval/acc'], step_total)
                    for k, v in eval_logs.items():
                        if k != 'eval/acc':
                            tb_writer.add_scalar(k, v, step_total)
                    tb_writer.flush()
                with open(csv_log_path, 'a') as f:
                    eval_acc_str = f'{eval_logs["eval/acc"]:.4f}' if 'eval/acc' in eval_logs else ''
                    acc_str = f'{acc:.4f}' if acc is not None else ''
                    row = f'{step_total},{lr_:.8f},{loss.item():.6f},{loss_total.item():.6f},' \
                          f'{cos_sim_clean.item():.6f},{emb_var.item():.6f},{emb_var_global.item():.6f},{emb_erank.item():.4f},{acc_str},{eval_acc_str}'
                    if args.enhanced_metrics and sparsity_metrics:
                        row += f',{sparsity_metrics["sparsity/mean_above_thresh"]:.6f}' \
                               f',{sparsity_metrics["sparsity/ratio_above_thresh"]:.6f}' \
                               f',{sparsity_metrics["sparsity/ratio_below_thresh"]:.6f}' \
                               f',{sparsity_metrics["sparsity/masked_loss"]:.6f}'
                    f.write(row + '\n')

            # Checkpoints (rank 0 only)
            if args.save_checkpoints and step_total % (args.steps // 10) == 0:
                torch.save(unwrap_model(model).model.state_dict(),
                           f'{args.output_dir}/checkpoints/step_{step_total}.pt')
                torch.save(optimizer.state_dict(),
                           f'{args.output_dir}/checkpoints/step_{step_total}_opt.pt')
            if step_total % 200 == 0:
                torch.save(unwrap_model(model).model.state_dict(),
                           f'{args.output_dir}/checkpoints/fallback_{step_total}.pt')
                torch.save(optimizer.state_dict(),
                           f'{args.output_dir}/checkpoints/fallback_{step_total}_opt.pt')
                if dynamic_pw is not None:
                    import json
                    json.dump({'pw': dynamic_pw.pw, 'ema_kernel': dynamic_pw.ema_kernel,
                               'ema_clean': dynamic_pw.ema_clean},
                              open(f'{args.output_dir}/checkpoints/dynamic_pw.json', 'w'))
                for f_name in os.listdir(f'{args.output_dir}/checkpoints'):
                    if f_name.startswith('fallback') and str(step_total) not in f_name:
                        os.remove(f'{args.output_dir}/checkpoints/{f_name}')

        if step_total >= args.steps:
            break

    return step_total, loss_ema, loss_initial, loss_initial_buf


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    rank, local_rank, world_size = setup_ddp()

    args = parse_args()
    assert not any(isinstance(x, str) and x in ['True', 'False'] for x in args.__dict__.values()), \
        f'args contains a string that should be a bool: {args}'
    assert args.eval_freq % args.log_freq == 0, 'eval_freq must be a multiple of log_freq'

    if is_main():
        print(f'DDP: {world_size} GPUs')

    if args.enable_bs_scaling and world_size > 1:
        original_bs = args.batch_size
        args.batch_size = args.batch_size  # per-GPU batch size stays as-is with DDP
        if is_main():
            print(f'Per-GPU batch size: {args.batch_size}, effective: {args.batch_size * world_size}')

    if args.resume:
        args.output_dir = args.resume
        args.finetuned_model_name = os.path.basename(args.resume)
    else:
        eff_bs = args.batch_size * world_size
        args.finetuned_model_name = make_experiment_name(
            clip_model_name=args.clip_model_name, pretrained=args.pretrained,
            dataset=args.dataset, loss=args.loss, steps=args.steps, epochs=args.epochs,
            batch_size=eff_bs, penalty_weight=args.penalty_weight, lr=args.lr,
            experiment_name=args.experiment_name, dynamic_pw=args.dynamic_pw,
            dynamic_pw_target_ratio=args.dynamic_pw_target_ratio,
            dynamic_pw_cosine_decay=args.dynamic_pw_cosine_decay,
            dynamic_pw_target_ratio_min=getattr(args, 'dynamic_pw_target_ratio_min', None),
        )
        args.output_dir = os.path.join(args.output_dir, args.finetuned_model_name)

    main(args, rank, local_rank, world_size)
