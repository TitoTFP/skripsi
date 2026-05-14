from __future__ import annotations

import torch


def masked_binary_stats(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> dict[str, int]:
    mask = valid_mask.bool()
    if not bool(mask.any()):
        return {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    pred = _to_binary_prediction(prediction, threshold, from_logits)[mask]
    truth = target.bool()[mask]
    return {
        "tp": int((pred & truth).sum().item()),
        "tn": int((~pred & ~truth).sum().item()),
        "fp": int((pred & ~truth).sum().item()),
        "fn": int((~pred & truth).sum().item()),
    }


def masked_iou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> float:
    stats = masked_binary_stats(prediction, target, valid_mask, threshold, from_logits)
    denom = stats["tp"] + stats["fp"] + stats["fn"]
    if denom == 0:
        return 0.0
    return stats["tp"] / denom


def masked_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> float:
    stats = masked_binary_stats(prediction, target, valid_mask, threshold, from_logits)
    denom = (2 * stats["tp"]) + stats["fp"] + stats["fn"]
    if denom == 0:
        return 0.0
    return (2 * stats["tp"]) / denom


def _to_binary_prediction(prediction: torch.Tensor, threshold: float, from_logits: bool) -> torch.Tensor:
    if from_logits and torch.is_floating_point(prediction):
        prediction = torch.sigmoid(prediction)
    return prediction >= threshold
