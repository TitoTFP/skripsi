from __future__ import annotations

from typing import TypeAlias

import torch  # type: ignore[import-not-found]

InputFeatures: TypeAlias = torch.Tensor | dict[str, torch.Tensor]
INPUT_SCENARIOS = ("all", "sentinel1", "sentinel2", "demnas")
SCENARIO_CHANNELS = {
    "all": (0, 1, 2, 3, 4, 5, 6),
    "sentinel1": (0, 1),
    "sentinel2": (2, 3, 4),
    "demnas": (5, 6),
}


def apply_modality_mask(features: InputFeatures, scenario: str) -> InputFeatures:
    """Zero unavailable modalities without changing shape or the source tensor."""
    if scenario not in SCENARIO_CHANNELS:
        raise ValueError(f"input scenario must be one of {INPUT_SCENARIOS}, got {scenario!r}")
    if isinstance(features, dict):
        encoder1 = _mask_tensor(features["encoder1"], scenario)
        encoder2 = features["encoder2"].clone()
        if scenario in ("sentinel2", "demnas"):
            encoder2.zero_()
        return {**features, "encoder1": encoder1, "encoder2": encoder2}
    return _mask_tensor(features, scenario)


def _mask_tensor(features: torch.Tensor, scenario: str) -> torch.Tensor:
    if features.ndim != 4 or features.shape[1] != 7:
        raise ValueError(f"expected features with shape [B, 7, H, W], got {tuple(features.shape)}")
    masked = features.clone()
    if scenario == "all":
        return masked
    keep = set(SCENARIO_CHANNELS[scenario])
    drop = [index for index in range(7) if index not in keep]
    masked[:, drop] = 0
    return masked
