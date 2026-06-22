"""Loss functions that matter in medical imaging, implemented and documented.

Each is a small, correct, tested implementation:
  - Focal loss          (class imbalance in detection/classification)
  - Dice loss           (overlap-based, standard for segmentation masks)
  - Tversky loss        (tunable FP/FN trade-off, generalizes Dice)
  - NT-Xent             (contrastive loss used by SimCLR)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss. gamma down-weights easy examples; alpha balances classes."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - p_t) ** self.gamma * ce).mean()


class DiceLoss(nn.Module):
    """Soft Dice loss for binary masks (1 - Dice overlap)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        dims = tuple(range(1, p.ndim))
        inter = (p * target).sum(dims)
        union = p.sum(dims) + target.sum(dims)
        return (1 - (2 * inter + self.smooth) / (union + self.smooth)).mean()


class TverskyLoss(nn.Module):
    """Tversky loss: alpha weights false positives, beta weights false negatives.

    alpha=beta=0.5 recovers Dice. Raise beta to punish missed positives (recall).
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        dims = tuple(range(1, p.ndim))
        tp = (p * target).sum(dims)
        fp = (p * (1 - target)).sum(dims)
        fn = ((1 - p) * target).sum(dims)
        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - ti).mean()


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """SimCLR NT-Xent contrastive loss for a batch of positive pairs (z1[i], z2[i])."""
    n = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n) + n, torch.arange(n)]).to(z.device)
    return F.cross_entropy(sim, targets)
