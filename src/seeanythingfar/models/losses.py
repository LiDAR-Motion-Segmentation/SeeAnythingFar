import torch
import torch.nn.functional as F


def clip_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.clamp(x.sigmoid(), min=eps, max=1 - eps)


def gaussian_focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, gamma: float = 4.0) -> torch.Tensor:
    """CenterNet focal loss on a Gaussian-splatted heatmap. ``pred`` is already a probability.

    Normalised by the number of peaks (cells where ``target == 1``).
    """
    eps = 1e-12
    pos = target.eq(1).float()
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * (1 - target).pow(gamma) * (1 - pos)
    return (pos_loss + neg_loss).sum() / pos.sum().clamp(min=1)


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Element-wise sigmoid focal loss (no reduction)."""
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t).pow(gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_t * loss
