from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from osgeo import gdal

from scripts.preprocessing_utils import TILE_SIZE, TEST_REGION, read_band, should_keep_tile, tile_offsets


ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = ROOT / "dataset/features_preprocessed"
LABEL_ROOT = ROOT / "dataset/labels_unosat_rasterized"
OUT_ROOT = ROOT / "dataset/tiles/7ch"
BY_REGION_ROOT = OUT_ROOT / "by_region"
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
    positive_tiles: dict[str, list[dict[str, object]]] = {}
    background_tiles: dict[str, list[dict[str, object]]] = {}
    for feature_dir in region_dirs():
        region = feature_dir.name
        positive_tiles.setdefault(region, [])
        background_tiles.setdefault(region, [])
        stack_ds = gdal.Open(str(feature_dir / "stack_7ch.tif"), gdal.GA_ReadOnly)
        feature_valid_ds = gdal.Open(str(feature_dir / "feature_valid_mask.tif"), gdal.GA_ReadOnly)
        label_dir = LABEL_ROOT / region
        flood_ds = gdal.Open(str(label_dir / "label_flood_binary.tif"), gdal.GA_ReadOnly)
        valid_ds = gdal.Open(str(label_dir / "label_valid_mask.tif"), gdal.GA_ReadOnly)
        for yoff in tile_offsets(stack_ds.RasterYSize):
            ysize = min(TILE_SIZE, stack_ds.RasterYSize - yoff)
            for xoff in tile_offsets(stack_ds.RasterXSize):
                xsize = min(TILE_SIZE, stack_ds.RasterXSize - xoff)
                flood = read_window(flood_ds, xoff, yoff, xsize, ysize).astype(np.uint8)
                label_valid = read_window(valid_ds, xoff, yoff, xsize, ysize).astype(np.uint8)
                feature_valid = read_window(feature_valid_ds, xoff, yoff, xsize, ysize).astype(np.uint8)
                if not should_keep_tile(label_valid.astype(bool), feature_valid.astype(bool), flood.astype(bool)):
                    continue
                record = {"region": region, "row": yoff, "col": xoff, "is_positive": bool(np.any(flood))}
                if record["is_positive"]:
                    positive_tiles[region].append(record)
                else:
                    background_tiles[region].append(record)
    return positive_tiles, background_tiles


def write_tile(record: dict[str, object]) -> dict[str, int | str]:
    region = str(record["region"])
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
    region_dir = BY_REGION_ROOT / region
    region_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{region}_r{yoff:06d}_c{xoff:06d}.npz"
    np.savez_compressed(
        region_dir / filename,
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
    BY_REGION_ROOT.mkdir(parents=True, exist_ok=True)
    for region_dir in BY_REGION_ROOT.glob("*"):
        if region_dir.is_dir():
            for tile in region_dir.glob("*.npz"):
                tile.unlink()
    positive_tiles, background_tiles = collect_tile_candidates()
    region_summaries: dict[str, dict[str, int | str]] = {}

    for feature_dir in region_dirs():
        region = feature_dir.name
        region_summaries[region] = {
            "region": region,
            "split": "test" if region == TEST_REGION else "cv",
            "tile_count": 0,
            "positive_tile_count": 0,
            "background_tile_count": 0,
            "flood_pixels": 0,
            "valid_pixels": 0,
            "water_river_pixels": 0,
            "s2_valid_pixels": 0,
        }

    selected = []
    for region in sorted(positive_tiles):
        selected.extend(positive_tiles[region])
        selected.extend(select_background_tiles(background_tiles[region], len(positive_tiles[region])))

    for rec in selected:
        tile_summary = write_tile(rec)
        summary = region_summaries[str(tile_summary["region"])]
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
