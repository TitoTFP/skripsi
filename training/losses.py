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
