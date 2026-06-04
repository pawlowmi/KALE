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
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from training.scheduler import cosine_lr
from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL
from open_flamingo.eval.models.utils import unwrap_model

from train.args import parse_args, print_args
from train.datasets import ImageNetDataset
from train.indexed_dataset import IndexedImageFolder
from train.cc12m_dataset import create_cc12m_dataloader
from train.models import ClipVisionModel, load_clip_orig, load_vision_model, wrap_vision_model
from train.utils import init_wandb, AverageMeter, compute_text_embeddings
from CLIP_eval.eval_utils import load_clip_model


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
    """Find the latest checkpoint in a directory by step number."""
    import re
    best_step = -1
    best_model = None
    best_opt = None
    for f in os.listdir(checkpoint_dir):
        # Match fallback_STEP.pt or step_STEP.pt
        m = re.match(r'(?:fallback|step)_(\d+)\.pt$', f)
        if m:
            step = int(m.group(1))
            opt_path = os.path.join(checkpoint_dir, f.replace('.pt', '_opt.pt'))
            if step > best_step and os.path.exists(opt_path):
                best_step = step
                best_model = os.path.join(checkpoint_dir, f)
                best_opt = opt_path
    return best_step, best_model, best_opt


def build_imagenet_text_embeddings(clip_model, tokenizer, template_name, device=0):
    """Build L2-normalized text embeddings for all 1000 ImageNet classes.

    Args:
        clip_model: Raw open_clip model with encode_text method.
        tokenizer: Tokenizer callable that converts texts to token tensors.
        template_name: Template style ('std' or 'blurry').
        device: Target device for the output tensor.

    Returns:
        Tensor of shape (embedding_dim, 1000) with L2-normalized columns.
    """
    if template_name == 'std':
        template = 'This is a photo of a {}'
    elif template_name == 'blurry':
        template = 'This is a blurry photo of a {}'
    else:
        raise ValueError(f'Unknown template: {template_name}')
    print(f'template: {template}')

    texts = [template.format(c) for c in IMAGENET_1K_CLASS_ID_TO_LABEL.values()]
    return compute_text_embeddings(clip_model, texts, tokenizer, device=device)


def main(args):
    # setup wandb
    if args.wandb:
        init_wandb(
            project_name='clip-finetune',
            model_name=args.finetuned_model_name,
            config=vars(args)
        )
    else:
        wandb.init(mode='disabled')

    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    csv_log_path = os.path.join(args.output_dir, 'train_log.csv')
    resuming = bool(args.resume)
    if not resuming:
        with open(csv_log_path, 'w') as f:
            header = 'step,lr,loss,loss_total,cos_sim,acc,eval_acc'
            if args.enhanced_metrics:
                header += ',mean_above_thresh,ratio_above_thresh,ratio_below_thresh,masked_loss'
            f.write(header + '\n')

    print_args(args)
    setup_output_dir(args.output_dir, args.overwrite, args, resume=resuming)

    # get models
    model_orig, preprocessor_without_normalize, normalize, tokenizer = load_clip_orig(args.clip_model_name)

    model_dino = None
    if not args.precomputed_dir:
        model_dino = load_vision_model(args.vision_model)

    model, _, _ = load_clip_model(args.clip_model_name, args.pretrained)

    print(f'[preprocessor_without_normalize] {preprocessor_without_normalize}')
    print(f'[normalize] {normalize}')

    # get data
    dl_workers = args.dataloader_num_workers * max(num_gpus, 1)
    eval_root = args.eval_root if args.eval_root else args.imagenet_root
    if args.dataset == 'cc12m':
        if args.cc12m_shards:
            assert not args.precomputed_dir, 'Precomputed embeddings not supported with live cc12m'
            n_samples = args.n_train_samples if args.n_train_samples > 0 else None
            dataloader = create_cc12m_dataloader(
                shards_dir=args.cc12m_shards, transform=preprocessor_without_normalize,
                batch_size=args.batch_size, n_samples=n_samples,
                num_workers=dl_workers, seed=0
            )
        elif args.precomputed_dir:
            dataset = IndexedImageFolder(root=args.imagenet_root + '/train', transform=preprocessor_without_normalize)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=dl_workers, drop_last=True, pin_memory=True, persistent_workers=dl_workers > 0, prefetch_factor=args.prefetch_factor if dl_workers > 0 else None)
        else:
            dataset = ImageNetDataset(root=args.imagenet_root + '/train', transform=preprocessor_without_normalize)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=dl_workers, drop_last=True, pin_memory=True, persistent_workers=dl_workers > 0, prefetch_factor=args.prefetch_factor if dl_workers > 0 else None)
    elif args.dataset == 'imagenet':
        if args.precomputed_dir:
            dataset = IndexedImageFolder(root=args.imagenet_root + '/train', transform=preprocessor_without_normalize)
        else:
            dataset = ImageNetDataset(root=args.imagenet_root + '/train', transform=preprocessor_without_normalize)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=dl_workers, drop_last=True, pin_memory=True, persistent_workers=dl_workers > 0, prefetch_factor=args.prefetch_factor if dl_workers > 0 else None)
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')
    dataset_eval = ImageNetDataset(root=eval_root + '/val', transform=preprocessor_without_normalize)
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=True, num_workers=dl_workers, drop_last=True, pin_memory=True, persistent_workers=dl_workers > 0, prefetch_factor=args.prefetch_factor if dl_workers > 0 else None)

    embedding_text_labels_norm = build_imagenet_text_embeddings(
        model_orig, tokenizer, args.template
    )

    if args.precomputed_dir:
        del model_orig
        model_orig = None
        torch.cuda.empty_cache()
        print('Precomputed mode: frozen CLIP and DINOv2 models not loaded to GPU')
    else:
        model_orig = ClipVisionModel(model=model_orig.visual, normalize=normalize)
        if num_gpus > 1:
            model_orig = torch.nn.DataParallel(model_orig)
        model_orig.cuda()

    if args.precomputed_dir:
        model_dino = None
    else:
        model_dino.cpu()
        model_dino = wrap_vision_model(model_dino, args.clip_model_name, args.vision_model, normalize)
        if num_gpus > 1:
            model_dino = torch.nn.DataParallel(model_dino)
        model_dino.cuda()

    model = ClipVisionModel(model=model.visual, normalize=normalize)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()


    # Load precomputed embeddings if provided
    precomputed_orig = None
    precomputed_dino = None
    if args.precomputed_dir:
        import numpy as np
        print(f'Loading precomputed embeddings from {args.precomputed_dir}...')
        precomputed_orig = {
            'train': torch.from_numpy(np.load(os.path.join(args.precomputed_dir, 'clip_orig_train.npy'))),
            'val': torch.from_numpy(np.load(os.path.join(args.precomputed_dir, 'clip_orig_val.npy'))),
        }
        precomputed_dino = {
            'train': torch.from_numpy(np.load(os.path.join(args.precomputed_dir, 'dino_train.npy'))),
            'val': torch.from_numpy(np.load(os.path.join(args.precomputed_dir, 'dino_val.npy'))),
        }
        print(f'  clip_orig_train: {precomputed_orig["train"].shape}')
        print(f'  dino_train: {precomputed_dino["train"].shape}')

    # set optimizer (all params have requires_grad=True)
    #projector_clip = torch.nn.parameter.Parameter(data=torch.eye(768)/np.sqrt(args.band_clip), requires_grad=True)
    if args.kernel_clip == "gaussian":
        projector_clip = torch.nn.parameter.Parameter(data=torch.tensor(args.band_clip), requires_grad=True)
    else:
        projector_clip = torch.nn.parameter.Parameter(data=torch.tensor([args.gamma, args.coef0]), requires_grad=True)

    param_groups = [
        {'params': unwrap_model(model).model.parameters(), 'weight_decay': args.wd},
        {'params': [projector_clip], 'weight_decay': 0}
    ]
    band_dino = args.band_dino
    projector_clip = projector_clip.cuda()

    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=args.lr,
            momentum=args.momentum_sgd
        )
    else:
        raise ValueError(f'Optimizer {args.optimizer} not supported.')

    # Resume from checkpoint
    if resuming:
        ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
        resume_step, resume_model, resume_opt = find_latest_checkpoint(ckpt_dir)
        assert resume_step >= 0, f'No checkpoint found in {ckpt_dir}'
        print(f'Resuming from step {resume_step}: {resume_model}', flush=True)
        unwrap_model(model).model.load_state_dict(torch.load(resume_model, map_location='cpu'))
        model.cuda()
        optimizer.load_state_dict(torch.load(resume_opt, map_location='cpu'))
        args.start_step = resume_step
    elif args.optimizer_state != '':
        optimizer.load_state_dict(torch.load(args.optimizer_state))

    # set scheduler
    assert (args.steps > 0) != (args.epochs > 0), \
        'Specify exactly one of --steps or --epochs'
    if args.epochs > 0:
        args.steps = args.epochs * len(dataloader)
        print(f'Computed {args.steps} steps from {args.epochs} epochs x {len(dataloader)} batches')

    if args.warmup > 0:
        warmup_steps = args.warmup
    else:
        warmup_steps = int(args.steps * args.warmup_pct / 100)
    print(f'Warmup: {warmup_steps} steps ({warmup_steps * 100 / args.steps:.1f}% of {args.steps})')

    lr_min = args.lr * args.lr_min_pct / 100
    _base_scheduler = cosine_lr(optimizer, args.lr, warmup_steps, args.steps)

    def scheduler(step):
        _base_scheduler(step)
        for param_group in optimizer.param_groups:
            if param_group['lr'] < lr_min:
                param_group['lr'] = lr_min

    print(f'LR schedule: 0 -> {args.lr} (warmup) -> {lr_min} (cosine, min {args.lr_min_pct}%)')

    total_epochs = args.steps / len(dataloader)
    print(f'train for {total_epochs:.2f} epochs')
    args.total_epochs = total_epochs

    # finetune
    step_total = args.start_step
    epoch = 0

    dynamic_pw = None
    if args.dynamic_pw > 0:
        dynamic_pw = DynamicPenaltyWeight(
            update_every=args.dynamic_pw,
            target_ratio=args.dynamic_pw_target_ratio,
            initial_pw=args.penalty_weight
        )
    while step_total < args.steps:
        step_total = train_one_epoch(
            step_total,
            model=model,
            model_orig=model_orig,
            model_dino=model_dino,
            projector_clip=projector_clip,
            band_dino=band_dino,
            dataloader=dataloader,
            dataloader_eval=dataloader_eval,
            optimizer=optimizer,
            scheduler=scheduler,
            embedding_text_labels_norm=embedding_text_labels_norm,
            normalize=normalize,
            args=args,
            epoch=epoch,
            warmup_steps=warmup_steps,
            dynamic_pw=dynamic_pw,
            tb_writer=tb_writer,
            precomputed_orig=precomputed_orig,
            precomputed_dino=precomputed_dino,
            csv_log_path=csv_log_path
        )
        print(f'Epoch {epoch} done.')
        epoch += 1

    tb_writer.flush()
    tb_writer.close()

    # save final model
    torch.save(unwrap_model(model).model.state_dict(), f'{args.output_dir}/checkpoints/final.pt')
    torch.save(optimizer.state_dict(), f'{args.output_dir}/checkpoints/final_opt.pt')

    if args.output_dir.endswith('_temp'):
        # rename temp dir to final dir
        os.rename(args.output_dir, args.output_dir[:-5])



def train_one_epoch(
        step_total, model, model_orig, model_dino, projector_clip, band_dino, dataloader, optimizer, scheduler, normalize,
        embedding_text_labels_norm, args, epoch, warmup_steps, dynamic_pw=None, dataloader_eval=None, tb_writer=None,
        precomputed_orig=None, precomputed_dino=None, csv_log_path=None
):
    if model_orig is not None:
        model_orig.eval()
    model.train()
    def _cycle_eval():
        while True:
            yield from dataloader_eval
    eval_iter = _cycle_eval() if dataloader_eval is not None else None

    loss_meter = AverageMeter('loss')
    loss_ema = None
    loss_initial = None
    loss_initial_buf = []
    ema_decay = 0.99
    cos_sim_meter = AverageMeter('cos-sim')
    acc_meter = AverageMeter('acc')

    epoch_start_time = time.time()
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', unit='batch', mininterval=1.0)
    for i, batch in enumerate(pbar):
        if len(batch) == 3:
            data, targets, indices = batch
        else:
            data, targets = batch
            indices = None
        is_classification = isinstance(targets, torch.Tensor)
        data = data.cuda()
        n_samples = data.shape[0]
        if is_classification:
            targets = targets.cuda()

        with torch.no_grad():
            if precomputed_orig is not None and indices is not None:
                embedding_orig = precomputed_orig['train'][indices].cuda(non_blocking=True)
                embedding_dino = precomputed_dino['train'][indices].cuda(non_blocking=True)
            else:
                embedding_orig = model_orig(vision=data, output_normalize=args.output_normalize)
                embedding_dino = model_dino(vision=data, output_normalize=args.output_normalize)
            if args.kernel_dino == "gaussian":
                k_dino = torch.cdist(embedding_dino, embedding_dino, p=2.0)
                k_dino = torch.exp(-k_dino ** 2 / (2 * band_dino ** 2))
            elif args.kernel_dino == "polynomial":
                n_feat = embedding_dino.shape[1]
                k_dino = (embedding_dino @ embedding_dino.T / n_feat + 1.) ** 3
                diag = k_dino.diag()
                k_dino = k_dino / torch.sqrt(diag.view(-1, 1) @ diag.view(1, -1))
            elif args.kernel_dino == "cosine":
                norm_X = embedding_dino / embedding_dino.norm(dim=1, keepdim=True)
                k_dino = norm_X @ norm_X.T
            del embedding_dino

        model.train()
        embedding_clean = model(data, output_normalize=args.output_normalize)

        loss_clean = compute_loss(
            loss_str=args.loss_clean, embedding=embedding_clean, targets=targets,
            embedding_orig=embedding_orig, logit_scale=100., embedding_text_labels_norm=None
            )

        if args.trades:
            embedding_clean_no_grad = embedding_clean.detach().clone()
        if args.kernel_clip == "gaussian":
            k_clip = torch.cdist(embedding_clean, embedding_clean, p=2.0)
            k_clip = torch.exp(-k_clip ** 2 / (2 * projector_clip ** 2))
        elif args.kernel_clip == "polynomial":
            k_clip = ((embedding_clean @ embedding_clean.T) * projector_clip[0] + projector_clip[1]) ** 3
            diag = k_clip.diag()
            k_clip = k_clip / torch.sqrt(diag.view(-1, 1) @ diag.view(1, -1))
        elif args.kernel_clip == "cosine":
            norm_X = embedding_clean / embedding_clean.norm(dim=1, keepdim=True)
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
                    step_total, loss.item(), loss_clean.item(), args.clean_weight, warmup_steps
                )
            eff_align = current_pw * loss.item()
            eff_clean = args.clean_weight * loss_clean.item()
            loss_ratio = eff_align / eff_clean if eff_clean > 0 else 0.0
        loss_total = args.clean_weight * loss_clean + current_pw * loss

        loss_total.backward()
        optimizer.step()
        optimizer.zero_grad()
        step_total += 1
        scheduler(step_total)

        with torch.no_grad():
            cos_sim_clean = F.cosine_similarity(embedding_clean, embedding_orig, dim=1).mean()
            if is_classification:
                embedding_clean_norm = F.normalize(embedding_clean, dim=1)
                logits_clean = embedding_clean_norm @ embedding_text_labels_norm
                acc = compute_acc(logits_clean, targets)
                acc_meter.update(acc, n_samples)
                del embedding_clean_norm
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

        eval_logs = dict()
        if (step_total-1) % args.eval_freq == 0:
            # we compute acc and racc (against supervised apgd) on validation data
            model.eval()
            eval_batch = next(eval_iter)
            data_eval, targets_eval = eval_batch[0], eval_batch[1]
            data_eval, targets_eval = data_eval.cuda(), targets_eval.cuda()

            with torch.no_grad():
                embedding_eval_norm = model(data_eval, output_normalize=True)
                logits_eval = embedding_eval_norm @ embedding_text_labels_norm
                acc_eval = compute_acc(logits_eval, targets_eval)
                # note we compute the cosine sim between clean and adv embedding,
                # not between orig and adv embedding as for training
            print(f'[eval-acc] {acc_eval:.2f}')
            eval_logs['eval/acc'] = acc_eval
            model.train()
            del data_eval, targets_eval, embedding_eval_norm, logits_eval

        lr_ = optimizer.param_groups[0].get('lr')
        postfix = {'lr': f'{lr_:.6f}', 'loss': f'{loss.item():.4f}'}
        if is_classification:
            postfix['acc'] = f'{acc:.1f}'
        if 'eval/acc' in eval_logs:
            postfix['eval_acc'] = f'{eval_logs["eval/acc"]:.1f}'
        if args.enhanced_metrics and sparsity_metrics:
            postfix['outlier'] = f'{sparsity_metrics["sparsity/ratio_above_thresh"]:.3f}'
        pbar.set_postfix(postfix)

        if (step_total-1) % args.log_freq == 0:
            log_data = {
                'step': step_total,
                'lr': lr_,
                'loss': loss.item(),
                'loss_clean': loss_clean.item(),
                'loss-total': loss_total.item(),
                'loss_ratio': loss_ratio,
                'penalty_weight': current_pw,
                'cos-sim-clean': cos_sim_clean.item(),
                'acc': acc,
                'avg/loss': loss_meter.avg,
                'avg/acc': acc_meter.avg,
                'loss_ema': loss_ema,
                'loss_ema_pct': loss_ema_pct,
            }
            log_data.update(eval_logs)
            if args.enhanced_metrics:
                log_data.update(sparsity_metrics)
            if (step_total-1) % (args.log_freq * 10) == 0:
                # compute expected average epoch time in hours
                batch_average_time = (time.time() - epoch_start_time) / (i + 1) / (60**2)
                epoch_average_time = batch_average_time * len(dataloader)
                this_epoch_remaining = epoch_average_time - \
                                       (time.time() - epoch_start_time) / 60**2
                total_remaining = epoch_average_time * (args.total_epochs - epoch - i / len(dataloader))
                print(f'[epoch average time] {epoch_average_time:.2f} [this epoch remaining] '
                      f'{this_epoch_remaining:.2f} [total remaining] {total_remaining:.2f}')

                log_data.update({
                    'time/total-remaining': total_remaining,
                    'time/this-epoch-remaining': this_epoch_remaining,
                    'time/epoch-average-time': epoch_average_time,
                    'time/batch-average-time': batch_average_time,
                    'other/epoch': epoch + i / len(dataloader),
                })
            wandb.log(log_data)
            if tb_writer is not None:
                tb_writer.add_scalar('train/loss', loss.item(), step_total)
                tb_writer.add_scalar('train/loss_clean', loss_clean.item(), step_total)
                tb_writer.add_scalar('train/loss_total', loss_total.item(), step_total)
                tb_writer.add_scalar('train/loss_ratio', loss_ratio, step_total)
                tb_writer.add_scalar('train/penalty_weight', current_pw, step_total)
                tb_writer.add_scalar('train/lr', lr_, step_total)
                if acc is not None:
                    tb_writer.add_scalar('train/acc', acc, step_total)
                tb_writer.add_scalar('train/cos_sim', cos_sim_clean.item(), step_total)
                tb_writer.add_scalar('train/loss_ema', loss_ema, step_total)
                tb_writer.add_scalar('train/loss_ema_pct', loss_ema_pct, step_total)
                if args.enhanced_metrics:
                    for k, v in sparsity_metrics.items():
                        tb_writer.add_scalar(k, v, step_total)
                if 'eval/acc' in eval_logs:
                    tb_writer.add_scalar('eval/acc', eval_logs['eval/acc'], step_total)
                tb_writer.flush()
            with open(csv_log_path, 'a') as f:
                eval_acc_str = f'{eval_logs["eval/acc"]:.4f}' if 'eval/acc' in eval_logs else ''
                acc_str = f'{acc:.4f}' if acc is not None else ''
                f.write(f'{step_total},{lr_:.8f},{loss.item():.6f},{loss_total.item():.6f},{cos_sim_clean.item():.6f},{acc_str},{eval_acc_str}')
                if args.enhanced_metrics and sparsity_metrics:
                    f.write(f',{sparsity_metrics["sparsity/mean_above_thresh"]:.6f},{sparsity_metrics["sparsity/ratio_above_thresh"]:.6f},{sparsity_metrics["sparsity/ratio_below_thresh"]:.6f},{sparsity_metrics["sparsity/masked_loss"]:.6f}')
                f.write('\n')

        # save 10 models over the course of training
        if args.save_checkpoints and (step_total % (args.steps // 10) == 0):
            # save model and optimizer state_dict
            torch.save(unwrap_model(model).model.state_dict(), f'{args.output_dir}/checkpoints/step_{step_total}.pt')
            torch.save(optimizer.state_dict(), f'{args.output_dir}/checkpoints/step_{step_total}_opt.pt')
        # every 200 steps, save a fallback model, which gets overwritten
        if step_total % 200 == 0:
            torch.save(unwrap_model(model).model.state_dict(), f'{args.output_dir}/checkpoints/fallback_{step_total}.pt')
            torch.save(optimizer.state_dict(), f'{args.output_dir}/checkpoints/fallback_{step_total}_opt.pt')
            # remove old fallback models
            for file in os.listdir(f'{args.output_dir}/checkpoints'):
                if file.startswith('fallback') and not str(step_total) in file:
                    os.remove(f'{args.output_dir}/checkpoints/{file}')

        if step_total >= args.steps:
            break

    return step_total


class DynamicPenaltyWeight:
    """Dynamically adjusts penalty_weight to maintain a target ratio
    of effective alignment loss to effective clean loss.

    Uses EMA of both losses for stability, updates every N steps.
    Target ratio = (pw * loss_kernel) / (cw * loss_clean)
    Solving for pw: pw = target_ratio * cw * ema_clean / ema_kernel
    """
    def __init__(self, update_every, target_ratio, initial_pw, ema_decay=0.99):
        self.update_every = update_every
        self.target_ratio = target_ratio
        self.pw = initial_pw
        self.ema_decay = ema_decay
        self.ema_kernel = None
        self.ema_clean = None

    def update_ema(self, loss_kernel, loss_clean):
        if self.ema_kernel is None:
            self.ema_kernel = loss_kernel
            self.ema_clean = loss_clean
        else:
            self.ema_kernel = self.ema_decay * self.ema_kernel + (1 - self.ema_decay) * loss_kernel
            self.ema_clean = self.ema_decay * self.ema_clean + (1 - self.ema_decay) * loss_clean

    def step(self, step_total, loss_kernel, loss_clean, clean_weight, warmup_steps=0):
        """Update EMA and optionally adjust pw. Returns current pw."""
        self.update_ema(loss_kernel, loss_clean)
        if step_total > warmup_steps and step_total > 0 and step_total % self.update_every == 0:
            if self.ema_kernel > 0 and self.ema_clean > 0 and self.pw > 0:
                current_ratio = self.pw * self.ema_kernel / (clean_weight * self.ema_clean)
                if current_ratio > 0:
                    correction = (self.target_ratio / current_ratio) ** 0.5
                    self.pw = self.pw * correction
                    print(f'[dynamic-pw] step={step_total} pw={self.pw:.4f} '
                          f'ratio={current_ratio:.6f} target={self.target_ratio:.6f} '
                          f'correction={correction:.4f} '
                          f'ema_kernel={self.ema_kernel:.6f} ema_clean={self.ema_clean:.6f}',
                          flush=True)
        return self.pw


def compute_kernel_sparsity(diff_matrix, threshold=1e-4):
    """Compute sparsity metrics on the (k_clip - k_dino)² matrix before mean reduction.

    Args:
        diff_matrix: Element-wise squared difference matrix (k_clip - k_dino)².
        threshold: Lambda threshold for classifying elements.

    Returns:
        Dict with sparsity metrics.
    """
    total = diff_matrix.numel()
    mask = diff_matrix > threshold
    n_above = mask.sum().item()
    n_below = total - n_above

    masked_loss = (diff_matrix * mask).sum() / max(n_above, 1)

    return {
        'sparsity/mean_above_thresh': diff_matrix[mask].mean().item() if n_above > 0 else 0.0,
        'sparsity/ratio_above_thresh': n_above / total,
        'sparsity/ratio_below_thresh': n_below / total,
        'sparsity/masked_loss': masked_loss.item(),
    }


@torch.no_grad()
def compute_acc(logits, targets):
    preds_clean = logits.max(dim=1)[1].detach()
    acc = (preds_clean.eq(targets).sum() / targets.shape[0]).item() * 100
    return acc


def compute_loss(loss_str, embedding, targets, embedding_orig, logit_scale,
                 embedding_text_labels_norm=None, reduction='mean'):
    if loss_str == 'cosine':
        loss =  1 - F.cosine_similarity(embedding, embedding_orig).mean()
    elif loss_str == 'l2':
        loss = l2(out=embedding, targets=embedding_orig, reduction=reduction)
    elif loss_str == 'ce':
        loss = ce(
            out=embedding @ (logit_scale * embedding_text_labels_norm),
            targets=targets,
            reduction=reduction
        )
    else:
        raise ValueError(f'loss {loss_str} not supported')
    return loss

def l2(out, targets, reduction='none'):
    # squared l2 - it does not divide by the latent dimension
    # should have shape (batch_size, embedding_size)
    assert out.shape == targets.shape, f'{out.shape} != {targets.shape}'
    assert out.shape[0] > 1
    # Compute the element-wise squared error
    squared_error_batch = F.mse_loss(out, targets, reduction='none')
    if reduction == 'mean':
        #squared_error_batch = torch.mean(torch.sqrt(squared_error_batch.sum(dim=1)))
        squared_error_batch = torch.mean(squared_error_batch.sum(dim=1))
    else:
        squared_error_batch = squared_error_batch.sum(dim=1)
        assert squared_error_batch.shape == (out.shape[0],), f'{squared_error_batch.shape} != {(out.shape[0],)}'
    return squared_error_batch

def ce(out, targets, reduction='mean'):
    # out = logits
    assert out.shape[0] == targets.shape[0], (out.shape, targets.shape)
    assert out.shape[0] > 1

    return F.cross_entropy(out, targets, reduction=reduction)

if __name__ == '__main__':
    # set seeds
    torch.manual_seed(0)
    np.random.seed(0)

    args = parse_args()
    assert not any([isinstance(x, str) and x in ['True', 'False'] for x in args.__dict__.values()]), f'args contains a string that should be a bool: {args}'
    assert args.eval_freq % args.log_freq == 0, 'eval_freq must be a multiple of log_freq'


    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f'Number of GPUs available: {num_gpus}')
    else:
        print('No multiple GPUs available.')

    if args.enable_bs_scaling and num_gpus > 1:
        original_bs = args.batch_size
        args.batch_size = args.batch_size * num_gpus
        print(f'Batch size scaled: {original_bs} x {num_gpus} GPUs = {args.batch_size}')

    # set model name and output dir
    if args.resume:
        args.output_dir = args.resume
        args.finetuned_model_name = os.path.basename(args.resume)
    else:
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
        duration_str = f'{args.steps}steps' if args.steps > 0 else f'{args.epochs}epochs'
        dpw_str = f'_dpw{args.dynamic_pw}' if args.dynamic_pw > 0 else ''
        args.finetuned_model_name = f'{args.clip_model_name}_{args.pretrained}_{args.dataset}_{args.loss}_{duration_str}_bs{args.batch_size}_pw{args.penalty_weight}{dpw_str}_lr{args.lr}_{args.experiment_name}_{random_str}'
        args.finetuned_model_name = args.finetuned_model_name.replace('/', '_')
        args.output_dir = os.path.join(args.output_dir, args.finetuned_model_name)
    # run
    main(args)