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


@dataclass(frozen=True)
class FeatureAugmentConfig:
    noise_std: float = 0.01
    channel_dropout_p: float = 0.03

    def __post_init__(self) -> None:
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if not 0 <= self.channel_dropout_p <= 1:
            raise ValueError("channel_dropout_p must be between 0 and 1")


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


def _augment_feature_tensor(
    tensor: torch.Tensor,
    rng: np.random.Generator,
    config: FeatureAugmentConfig,
) -> torch.Tensor:
    out = tensor.clone()
    if config.noise_std > 0:
        noise = rng.normal(0.0, config.noise_std, size=tuple(out.shape)).astype(np.float32)
        out = out + torch.from_numpy(noise).to(device=out.device, dtype=out.dtype)
        out = torch.clamp(out, 0.0, 1.0)
    if config.channel_dropout_p > 0 and out.ndim >= 3:
        keep = rng.random(out.shape[0]) >= config.channel_dropout_p
        mask = torch.from_numpy(keep.astype(np.float32)).to(device=out.device, dtype=out.dtype)
        out = out * mask.view(-1, *([1] * (out.ndim - 1)))
    return out.contiguous()


def _augment_features(
    features: torch.Tensor | dict[str, torch.Tensor],
    rng: np.random.Generator,
    config: FeatureAugmentConfig,
) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(features, dict):
        if "encoder1" in features and "encoder2" in features:
            encoder1 = _augment_feature_tensor(features["encoder1"], rng, config)
            out = dict(features)
            out["encoder1"] = encoder1
            out["encoder2"] = encoder1[: features["encoder2"].shape[0]].clone()
            return out
        return {key: _augment_feature_tensor(value, rng, config) for key, value in features.items()}
    return _augment_feature_tensor(features, rng, config)


def apply_feature_augmentation(
    sample: dict[str, Any],
    rng: np.random.Generator | None = None,
    config: FeatureAugmentConfig | None = None,
) -> dict[str, Any]:
    if "features" not in sample:
        return sample
    rng = rng or np.random.default_rng()
    config = config or FeatureAugmentConfig()
    out = dict(sample)
    out["features"] = _augment_features(sample["features"], rng, config)
    return out
