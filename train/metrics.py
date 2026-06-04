"""
Extensible training metrics evaluated periodically on a fixed validation batch.

Usage:
    metric = ClipDriftMetric(dataset, frozen_model, device, n_samples=256, seed=0)
    # in training loop:
    if metric.should_run(step):
        results = metric.compute(current_model)
        # results: {'drift/l2': float, 'drift/angle_deg': float}
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import random


class PeriodicMetric:
    """Base class for metrics computed every `freq` steps on a fixed batch."""

    def __init__(self, freq: int = 100):
        self.freq = freq

    def should_run(self, step: int) -> bool:
        return step % self.freq == 0

    def compute(self, model) -> dict:
        raise NotImplementedError


class ClipDriftMetric(PeriodicMetric):
    """Measures how far the trainable CLIP encoder has drifted from the frozen baseline.

    Computes on a fixed set of images sampled once from the training dataset:
      - drift/l2:        mean L2 distance between current and frozen embeddings
      - drift/angle_deg: mean angle (degrees) between current and frozen embeddings
    """

    def __init__(self, dataset, frozen_model, device, n_samples: int = 256,
                 seed: int = 0, freq: int = 100, batch_size: int = 256):
        super().__init__(freq=freq)
        self.device = device
        self.frozen_model = frozen_model  # ClipVisionModel, eval mode, on device

        # Sample a fixed subset once
        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), min(n_samples, len(dataset)))
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

        # Precompute frozen embeddings once
        self.frozen_model.eval()
        frozen_embs = []
        images_list = []
        with torch.no_grad():
            for batch in loader:
                imgs = batch[0].to(device, non_blocking=True)
                frozen_embs.append(self.frozen_model(imgs, output_normalize=False).cpu())
                images_list.append(imgs.cpu())

        self.frozen_embs = torch.cat(frozen_embs)   # [N, D] on CPU
        self.images = torch.cat(images_list)         # [N, C, H, W] on CPU

    @torch.no_grad()
    def compute(self, model) -> dict:
        """Compute drift metrics between model and frozen baseline."""
        model.eval()
        current_embs = []
        for i in range(0, len(self.images), 256):
            imgs = self.images[i:i+256].to(self.device, non_blocking=True)
            current_embs.append(model(imgs, output_normalize=False).cpu())
        model.train()

        current = torch.cat(current_embs)           # [N, D]
        frozen = self.frozen_embs                    # [N, D]

        l2 = (current - frozen).norm(dim=1).mean().item()

        cos = F.cosine_similarity(current, frozen, dim=1).clamp(-1, 1)
        angle_deg = torch.acos(cos).rad2deg().mean().item()

        return {'drift/l2': l2, 'drift/angle_deg': angle_deg}
