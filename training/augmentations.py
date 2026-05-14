from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class SpatialTransform:
    rot90: int = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False

    def __post_init__(self) -> None:
        if self.rot90 not in (0, 1, 2, 3):
            raise ValueError("rot90 must be 0, 1, 2, or 3")


def random_spatial_transform(rng: np.random.Generator | None = None) -> SpatialTransform:
    rng = rng or np.random.default_rng()
    return SpatialTransform(
        rot90=int(rng.integers(0, 4)),
        flip_horizontal=bool(rng.random() < 0.5),
        flip_vertical=bool(rng.random() < 0.5),
    )


def _transform_tensor(tensor: torch.Tensor, transform: SpatialTransform) -> torch.Tensor:
    out = tensor
    if transform.rot90:
        out = torch.rot90(out, transform.rot90, dims=(-2, -1))
    if transform.flip_horizontal:
        out = torch.flip(out, dims=(-1,))
    if transform.flip_vertical:
        out = torch.flip(out, dims=(-2,))
    return out.contiguous()


def _transform_value(value: Any, transform: SpatialTransform) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim >= 2:
        return _transform_tensor(value, transform)
    if isinstance(value, dict):
        return {key: _transform_value(item, transform) for key, item in value.items()}
    return value


def apply_spatial_transform(sample: dict[str, Any], transform: SpatialTransform) -> dict[str, Any]:
    return {key: _transform_value(value, transform) for key, value in sample.items()}
