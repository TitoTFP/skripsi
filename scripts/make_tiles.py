from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from osgeo import gdal

from scripts.preprocessing_utils import TILE_SIZE, choose_split, read_band, should_keep_tile


ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = ROOT / "dataset/features_preprocessed"
LABEL_ROOT = ROOT / "dataset/labels_unosat_rasterized"
OUT_ROOT = ROOT / "dataset/tiles/7ch"
SUMMARY_PATH = ROOT / "dataset/preprocessing_summary.csv"


def region_dirs() -> list[Path]:
    return sorted(p for p in FEATURE_ROOT.iterdir() if p.is_dir())


def read_window(ds: gdal.Dataset, xoff: int, yoff: int, xsize: int, ysize: int) -> np.ndarray:
    arr = ds.ReadAsArray(xoff, yoff, xsize, ysize)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    return arr


def pad_to_tile(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    channels, height, width = arr.shape
    out = np.full((channels, TILE_SIZE, TILE_SIZE), fill, dtype=arr.dtype)
    out[:, :height, :width] = arr
    return out


def select_background_tiles(candidates: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    return sorted(candidates, key=lambda row: (str(row["region"]), int(row["row"]), int(row["col"])))[:limit]


def collect_tile_candidates() -> tuple[dict[str, list[dict[str, object]]], dict[str, list[dict[str, object]]]]:
    positive_tiles: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    background_tiles: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    for feature_dir in region_dirs():
        region = feature_dir.name
        split = choose_split(region)
        stack_ds = gdal.Open(str(feature_dir / "stack_7ch.tif"), gdal.GA_ReadOnly)
        feature_valid_ds = gdal.Open(str(feature_dir / "feature_valid_mask.tif"), gdal.GA_ReadOnly)
        label_dir = LABEL_ROOT / region
        flood_ds = gdal.Open(str(label_dir / "label_flood_binary.tif"), gdal.GA_ReadOnly)
        valid_ds = gdal.Open(str(label_dir / "label_valid_mask.tif"), gdal.GA_ReadOnly)
        for yoff in range(0, stack_ds.RasterYSize, TILE_SIZE):
            ysize = min(TILE_SIZE, stack_ds.RasterYSize - yoff)
            for xoff in range(0, stack_ds.RasterXSize, TILE_SIZE):
                xsize = min(TILE_SIZE, stack_ds.RasterXSize - xoff)
                flood = read_window(flood_ds, xoff, yoff, xsize, ysize).astype(np.uint8)
                label_valid = read_window(valid_ds, xoff, yoff, xsize, ysize).astype(np.uint8)
                feature_valid = read_window(feature_valid_ds, xoff, yoff, xsize, ysize).astype(np.uint8)
                if not should_keep_tile(label_valid.astype(bool), feature_valid.astype(bool), flood.astype(bool)):
                    continue
                record = {"region": region, "split": split, "row": yoff, "col": xoff, "is_positive": bool(np.any(flood))}
                if record["is_positive"]:
                    positive_tiles[split].append(record)
                else:
                    background_tiles[split].append(record)
    return positive_tiles, background_tiles


def write_tile(record: dict[str, object]) -> dict[str, int | str]:
    region = str(record["region"])
    split = str(record["split"])
    yoff = int(record["row"])
    xoff = int(record["col"])
    feature_dir = FEATURE_ROOT / region
    label_dir = LABEL_ROOT / region
    stack_ds = gdal.Open(str(feature_dir / "stack_7ch.tif"), gdal.GA_ReadOnly)
    feature_valid_ds = gdal.Open(str(feature_dir / "feature_valid_mask.tif"), gdal.GA_ReadOnly)
    s2_valid_ds = gdal.Open(str(feature_dir / "s2_valid_mask.tif"), gdal.GA_ReadOnly)
    flood_ds = gdal.Open(str(label_dir / "label_flood_binary.tif"), gdal.GA_ReadOnly)
    valid_ds = gdal.Open(str(label_dir / "label_valid_mask.tif"), gdal.GA_ReadOnly)
    water_ds = gdal.Open(str(label_dir / "label_water_river_mask.tif"), gdal.GA_ReadOnly)
    xsize = min(TILE_SIZE, stack_ds.RasterXSize - xoff)
    ysize = min(TILE_SIZE, stack_ds.RasterYSize - yoff)
    stack = pad_to_tile(read_window(stack_ds, xoff, yoff, xsize, ysize).astype(np.float32), 0.0)
    flood = pad_to_tile(read_window(flood_ds, xoff, yoff, xsize, ysize).astype(np.uint8), 0)
    label_valid = pad_to_tile(read_window(valid_ds, xoff, yoff, xsize, ysize).astype(np.uint8), 0)
    water = pad_to_tile(read_window(water_ds, xoff, yoff, xsize, ysize).astype(np.uint8), 0)
    feature_valid = pad_to_tile(read_window(feature_valid_ds, xoff, yoff, xsize, ysize).astype(np.uint8), 0)
    s2_valid = pad_to_tile(read_window(s2_valid_ds, xoff, yoff, xsize, ysize).astype(np.uint8), 0)
    split_dir = OUT_ROOT / split
    split_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{region}_r{yoff:06d}_c{xoff:06d}.npz"
    np.savez_compressed(
        split_dir / filename,
        x=stack,
        y=flood,
        valid_mask=label_valid,
        water_river_mask=water,
        feature_valid_mask=feature_valid,
        s2_valid_mask=s2_valid,
        region=np.array(region),
        row=np.array(yoff),
        col=np.array(xoff),
        channels=np.array(["VV", "VH", "Hue", "Saturation", "Value", "Slope", "HAND"]),
    )
    return {
        "region": region,
        "split": split,
        "tile_count": 1,
        "positive_tile_count": int(record["is_positive"]),
        "background_tile_count": int(not record["is_positive"]),
        "flood_pixels": int(np.count_nonzero(flood)),
        "valid_pixels": int(np.count_nonzero(label_valid)),
        "water_river_pixels": int(np.count_nonzero(water)),
        "s2_valid_pixels": int(np.count_nonzero(s2_valid)),
    }


def main() -> None:
    gdal.UseExceptions()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split_dir in OUT_ROOT.glob("*"):
        if split_dir.is_dir():
            for tile in split_dir.glob("*.npz"):
                tile.unlink()
    positive_tiles, background_tiles = collect_tile_candidates()
    region_summaries: dict[tuple[str, str], dict[str, int | str]] = {}

    for feature_dir in region_dirs():
        region = feature_dir.name
        split = choose_split(region)
        region_summaries[(region, split)] = {
            "region": region,
            "split": split,
            "tile_count": 0,
            "positive_tile_count": 0,
            "background_tile_count": 0,
            "flood_pixels": 0,
            "valid_pixels": 0,
            "water_river_pixels": 0,
            "s2_valid_pixels": 0,
        }

    selected = []
    for split in ("train", "val", "test"):
        selected.extend(positive_tiles[split])
        selected.extend(select_background_tiles(background_tiles[split], len(positive_tiles[split])))

    for rec in selected:
        tile_summary = write_tile(rec)
        key = (str(tile_summary["region"]), str(tile_summary["split"]))
        summary = region_summaries[key]
        for field in (
            "tile_count",
            "positive_tile_count",
            "background_tile_count",
            "flood_pixels",
            "valid_pixels",
            "water_river_pixels",
            "s2_valid_pixels",
        ):
            summary[field] = int(summary[field]) + int(tile_summary[field])

    rows = list(region_summaries.values())
    with SUMMARY_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['region']},{row['split']},tiles={row['tile_count']},"
            f"positive={row['positive_tile_count']},background={row['background_tile_count']}"
        )


if __name__ == "__main__":
    main()
