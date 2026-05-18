from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from scripts.preprocessing_utils import (
    PROCANET_ENCODER1_CHANNELS,
    PROCANET_ENCODER2_CHANNELS,
    split_procanet_encoders,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "dataset/tiles/7ch"
OUT_ROOT = ROOT / "dataset/tiles/procanet"
SOURCE_BY_REGION_ROOT = SOURCE_ROOT / "by_region"
OUT_BY_REGION_ROOT = OUT_ROOT / "by_region"
PASSTHROUGH_KEYS = (
    "y",
    "valid_mask",
    "water_river_mask",
    "feature_valid_mask",
    "s2_valid_mask",
    "region",
    "row",
    "col",
)


def build_procanet_payload(source: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    encoder1, encoder2 = split_procanet_encoders(source["x"].astype(np.float32, copy=False))
    payload = {
        "x_encoder1": encoder1,
        "x_encoder2": encoder2,
        "encoder1_channels": np.array(PROCANET_ENCODER1_CHANNELS),
        "encoder2_channels": np.array(PROCANET_ENCODER2_CHANNELS),
    }
    for key in PASSTHROUGH_KEYS:
        payload[key] = source[key]
    return payload


def clear_output() -> None:
    OUT_BY_REGION_ROOT.mkdir(parents=True, exist_ok=True)
    for region_dir in OUT_BY_REGION_ROOT.glob("*"):
        if region_dir.is_dir():
            for tile in region_dir.glob("*.npz"):
                tile.unlink()


def convert_region(region: str) -> int:
    source_dir = SOURCE_BY_REGION_ROOT / region
    out_dir = OUT_BY_REGION_ROOT / region
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for source_path in sorted(source_dir.glob("*.npz")):
        with np.load(source_path) as source:
            payload = build_procanet_payload(source)
            np.savez_compressed(out_dir / source_path.name, **payload)
        count += 1
    return count


def main() -> None:
    clear_output()
    counts = {region_dir.name: convert_region(region_dir.name) for region_dir in sorted(SOURCE_BY_REGION_ROOT.iterdir()) if region_dir.is_dir()}
    print("procanet_tile_counts", counts)


if __name__ == "__main__":
    main()
