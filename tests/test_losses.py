"""Property tests: each loss is minimized by correct predictions and behaves
as its parameters intend."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from losses import FocalLoss, DiceLoss, TverskyLoss, nt_xent


def test_focal_lower_when_correct():
    target = torch.tensor([[1.0], [0.0]])
    good = torch.tensor([[8.0], [-8.0]])
    bad = torch.tensor([[-8.0], [8.0]])
    fl = FocalLoss()
    assert fl(good, target) < fl(bad, target)


def test_dice_zero_on_perfect_mask():
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1
    logits = target * 20 - 10
    assert DiceLoss()(logits, target).item() < 0.02


def test_tversky_recovers_dice_at_half():
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1
    logits = torch.randn(1, 1, 16, 16)
    # The Dice<->Tversky identity holds exactly only without smoothing.
    d = DiceLoss(smooth=0.0)(logits, target).item()
    t = TverskyLoss(alpha=0.5, beta=0.5, smooth=0.0)(logits, target).item()
    assert abs(d - t) < 1e-5


def test_tversky_beta_penalizes_false_negatives():
    target = torch.ones(1, 1, 8, 8)
    miss = torch.full((1, 1, 8, 8), -5.0)  # predicts all negative -> all FN
    high_beta = TverskyLoss(alpha=0.1, beta=0.9)(miss, target)
    low_beta = TverskyLoss(alpha=0.9, beta=0.1)(miss, target)
    assert high_beta > low_beta


def test_nt_xent_lower_for_aligned_pairs():
    torch.manual_seed(0)
    base = torch.randn(8, 16)
    aligned = nt_xent(base, base + 0.01 * torch.randn(8, 16))
    random = nt_xent(base, torch.randn(8, 16))
    assert aligned < random


if __name__ == "__main__":
    for fn in [test_focal_lower_when_correct, test_dice_zero_on_perfect_mask,
               test_tversky_recovers_dice_at_half, test_tversky_beta_penalizes_false_negatives,
               test_nt_xent_lower_for_aligned_pairs]:
        fn()
    print("all tests passed")
