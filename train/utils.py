import sys

import torch
import torch.nn.functional as F
import wandb
from time import sleep
import os

def init_wandb(project_name, model_name, config, **wandb_kwargs):
    os.environ['WANDB__SERVICE_WAIT'] = '300'
    while True:
        try:
            wandb_run = wandb.init(
                project=project_name, name=model_name, save_code=True,
                config=config, **wandb_kwargs,
                )
            break
        except Exception as e:
            print('wandb connection error', file=sys.stderr)
            print(f'error: {e}', file=sys.stderr)
            sleep(1)
            print('retrying..', file=sys.stderr)
    return wandb_run


def compute_text_embeddings(clip_model, texts, tokenizer, device=0, batch_size=500):
    """Compute L2-normalized text embeddings for a list of text prompts.

    Encodes texts in batches through the CLIP text encoder to avoid OOM,
    then returns the transposed normalized embedding matrix on the target device.

    Args:
        clip_model: Raw open_clip model (not a vision wrapper) with encode_text method.
        texts: List of text strings to encode (e.g. ImageNet class prompts).
        tokenizer: Tokenizer callable that converts texts to token tensors.
        device: Target device for the output tensor.
        batch_size: Number of texts to encode per batch.

    Returns:
        Tensor of shape (embedding_dim, num_texts) with L2-normalized columns,
        placed on the target device.
    """
    tokens = tokenizer(texts)

    clip_model.to(device)
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i + batch_size].to(device)
            embeddings.append(
                clip_model.encode_text(batch, normalize=True).detach().cpu()
            )
    clip_model.cpu()
    torch.cuda.empty_cache()

    result = torch.cat(embeddings).T.to(device)
    assert torch.allclose(F.normalize(result, dim=0), result), \
        'Text embeddings are not properly normalized'
    return result

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)