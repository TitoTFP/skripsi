from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    mask = valid_mask.bool()
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], target.float()[mask])


def masked_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    mask = valid_mask.bool()
    if not bool(mask.any()):
        return logits.sum() * 0.0
    probs = torch.sigmoid(logits)
    probs = probs[mask]
    target = target.float()[mask]
    intersection = (probs * target).sum()
    denominator = probs.sum() + target.sum()
    return 1.0 - ((2.0 * intersection + eps) / (denominator + eps))


def masked_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    return masked_bce_with_logits(logits, target, valid_mask) + masked_dice_loss(logits, target, valid_mask)
