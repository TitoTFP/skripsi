from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from training.augmentations import apply_spatial_transform, random_spatial_transform


Architecture = Literal["unet", "procanet"]
Split = Literal["train", "val", "test"]

DEFAULT_TILE_ROOT = Path(__file__).resolve().parents[1] / "dataset" / "tiles"
VALID_SPLITS = {"train", "val", "test"}
VALID_ARCHITECTURES = {"unet", "procanet"}
AUXILIARY_MASK_KEYS = ("water_river_mask", "feature_valid_mask", "s2_valid_mask")


class FloodTileDataset(Dataset):
    def __init__(
        self,
        split: Split | str,
        architecture: Architecture | str,
        root: Path | str | None = None,
        augment: bool | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.split = split
        self.architecture = architecture.lower()
        self.root = Path(root) if root is not None else DEFAULT_TILE_ROOT
        self.rng = rng or np.random.default_rng()
        self.augment = (split == "train") if augment is None else bool(augment and split == "train")

        if self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")
        if self.architecture not in VALID_ARCHITECTURES:
            raise ValueError(f"architecture must be one of {sorted(VALID_ARCHITECTURES)}, got {architecture!r}")

        tile_root = "7ch" if self.architecture == "unet" else "procanet"
        self.tile_dir = self.root / tile_root / self.split
        self.paths = sorted(self.tile_dir.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"no .npz tiles found in {self.tile_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as data:
            sample = {
                "features": self._read_features(data),
                "y": _float_tensor(data["y"]),
                "valid_mask": _bool_tensor(data["valid_mask"]),
                "metadata": self._read_metadata(data, path),
            }
            aux_masks = {
                key: _bool_tensor(data[key])
                for key in AUXILIARY_MASK_KEYS
                if key in data.files
            }
            if aux_masks:
                sample["auxiliary_masks"] = aux_masks

        if self.augment:
            sample = apply_spatial_transform(sample, random_spatial_transform(self.rng))
        return sample

    def _read_features(self, data: np.lib.npyio.NpzFile) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.architecture == "unet":
            return _float_tensor(data["x"])
        return {
            "encoder1": _float_tensor(data["x_encoder1"]),
            "encoder2": _float_tensor(data["x_encoder2"]),
        }

    def _read_metadata(self, data: np.lib.npyio.NpzFile, path: Path) -> dict[str, Any]:
        channels: tuple[str, ...] | dict[str, tuple[str, ...]]
        if self.architecture == "unet":
            channels = _tuple_strings(data["channels"])
        else:
            channels = {
                "encoder1": _tuple_strings(data["encoder1_channels"]),
                "encoder2": _tuple_strings(data["encoder2_channels"]),
            }
        return {
            "region": str(data["region"].item()),
            "row": int(data["row"].item()),
            "col": int(data["col"].item()),
            "path": str(path),
            "channels": channels,
        }


def _float_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32, copy=False)).float()


def _bool_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(bool, copy=False)).bool()


def _tuple_strings(array: np.ndarray) -> tuple[str, ...]:
    return tuple(str(item) for item in array.tolist())
