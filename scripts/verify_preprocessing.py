from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from osgeo import gdal

from scripts.preprocessing_utils import CHANNELS_7CH, same_grid


ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = ROOT / "dataset/features_preprocessed"
S1_ROOT = ROOT / "dataset/satelit raw"
LABEL_ROOT = ROOT / "dataset/labels_unosat_rasterized"
TILE_ROOT = ROOT / "dataset/tiles/7ch"
REPORT_PATH = ROOT / "dataset/preprocessing_verification_report.csv"
FEATURE_FILES = [
    "vv_norm.tif",
    "vh_norm.tif",
    "hue.tif",
    "saturation.tif",
    "value.tif",
    "s2_valid_mask.tif",
    "slope_norm.tif",
    "hand_norm.tif",
    "feature_valid_mask.tif",
    "stack_7ch.tif",
]


def reference_for_region(region: str) -> gdal.Dataset:
    folder_name = region.replace("_", " ")
    aliases = {
        "Banda Aceh": "Kota Banda Aceh",
        "Langsa": "Kota Langsa",
    }
    folder_name = aliases.get(folder_name, folder_name)
    s1 = next((S1_ROOT / folder_name).glob("S1_*.tif"))
    return gdal.Open(str(s1), gdal.GA_ReadOnly)


def unique_values(path: Path) -> set[int]:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    values: set[int] = set()
    for yoff in range(0, ds.RasterYSize, 512):
        ysize = min(512, ds.RasterYSize - yoff)
        arr = ds.GetRasterBand(1).ReadAsArray(0, yoff, ds.RasterXSize, ysize)
        values.update(np.unique(arr).astype(int).tolist())
    return values


def min_max_stack(path: Path) -> tuple[float, float, bool]:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    min_value = float("inf")
    max_value = float("-inf")
    all_finite = True
    for band_idx in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(band_idx)
        for yoff in range(0, ds.RasterYSize, 512):
            ysize = min(512, ds.RasterYSize - yoff)
            arr = band.ReadAsArray(0, yoff, ds.RasterXSize, ysize).astype(np.float32)
            finite = np.isfinite(arr)
            all_finite = all_finite and bool(finite.all())
            if finite.any():
                min_value = min(min_value, float(arr[finite].min()))
                max_value = max(max_value, float(arr[finite].max()))
    return min_value, max_value, all_finite


def verify_features() -> list[dict[str, object]]:
    rows = []
    errors = []
    for feature_dir in sorted(p for p in FEATURE_ROOT.iterdir() if p.is_dir()):
        region = feature_dir.name
        ref = reference_for_region(region)
        for filename in FEATURE_FILES:
            path = feature_dir / filename
            if not path.exists():
                errors.append(f"missing {path}")
                continue
            ds = gdal.Open(str(path), gdal.GA_ReadOnly)
            if not same_grid(ref, ds):
                errors.append(f"grid mismatch {path}")
            if filename == "stack_7ch.tif":
                if ds.RasterCount != 7:
                    errors.append(f"stack band count {path}: {ds.RasterCount}")
                descriptions = tuple(ds.GetRasterBand(i).GetDescription() for i in range(1, 8))
                if descriptions != CHANNELS_7CH:
                    errors.append(f"stack descriptions {path}: {descriptions}")
                min_value, max_value, all_finite = min_max_stack(path)
                if min_value < -1e-6 or max_value > 1.000001 or not all_finite:
                    errors.append(f"stack range/finite {path}: {min_value}, {max_value}, finite={all_finite}")
                rows.append({"region": region, "stack_min": min_value, "stack_max": max_value})
            if filename.endswith("_mask.tif"):
                values = unique_values(path)
                if not values.issubset({0, 1}):
                    errors.append(f"mask values {path}: {sorted(values)}")
        for label in ("label_flood_binary.tif", "label_valid_mask.tif", "label_water_river_mask.tif"):
            values = unique_values(LABEL_ROOT / region / label)
            if not values.issubset({0, 1}):
                errors.append(f"label values {region}/{label}: {sorted(values)}")
    if errors:
        raise RuntimeError("\n".join(errors))
    return rows


def verify_tiles() -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for split in counts:
        split_dir = TILE_ROOT / split
        if not split_dir.exists():
            continue
        for tile in split_dir.glob("*.npz"):
            data = np.load(tile)
            if data["x"].shape != (7, 512, 512):
                raise RuntimeError(f"x shape {tile}: {data['x'].shape}")
            for key in ("y", "valid_mask", "water_river_mask", "feature_valid_mask", "s2_valid_mask"):
                if data[key].shape != (1, 512, 512):
                    raise RuntimeError(f"{key} shape {tile}: {data[key].shape}")
                values = set(np.unique(data[key]).astype(int).tolist())
                if not values.issubset({0, 1}):
                    raise RuntimeError(f"{key} values {tile}: {sorted(values)}")
            if not np.isfinite(data["x"]).all():
                raise RuntimeError(f"non-finite x {tile}")
            if float(data["x"].min()) < -1e-6 or float(data["x"].max()) > 1.000001:
                raise RuntimeError(f"x range {tile}: {data['x'].min()}, {data['x'].max()}")
            counts[split] += 1
    return counts


def main() -> None:
    gdal.UseExceptions()
    rows = verify_features()
    tile_counts = verify_tiles()
    with REPORT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["region", "stack_min", "stack_max"])
        writer.writeheader()
        writer.writerows(rows)
    print("feature_regions", len(rows))
    print("tile_counts", tile_counts)


if __name__ == "__main__":
    main()
