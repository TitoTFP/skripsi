from __future__ import annotations

from pathlib import Path

import numpy as np
from osgeo import gdal


CHANNELS_7CH = ("VV", "VH", "Hue", "Saturation", "Value", "Slope", "HAND")
PROCANET_ENCODER1_CHANNELS = CHANNELS_7CH
PROCANET_ENCODER2_CHANNELS = ("VV", "VH")
TILE_SIZE = 512
TILE_STRIDE = 256
VALID_COVERAGE_THRESHOLD = 0.70
BAD_S2_REGIONS = {"Aceh_Tamiang", "Agam", "Langsa", "Pasaman_Barat"}
TEST_REGION = "Aceh_Utara"
SPATIAL_CV_FOLDS = (
    ("Pidie", "Pidie_Jaya"),
    ("Aceh_Besar", "Banda_Aceh"),
    ("Aceh_Tamiang", "Aceh_Timur"),
    ("Bireuen", "Langsa"),
    ("Agam", "Pasaman_Barat"),
)
CV_REGIONS = tuple(region for fold in SPATIAL_CV_FOLDS for region in fold)


def region_to_output_name(region: str) -> str:
    if region == "Kota Banda Aceh":
        return "Banda_Aceh"
    if region == "Kota Langsa":
        return "Langsa"
    return region.replace(" ", "_")


def choose_split(region: str) -> str:
    if region == TEST_REGION:
        return "test"
    return "train"


def fold_regions(fold: int) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if fold < 0 or fold >= len(SPATIAL_CV_FOLDS):
        raise ValueError(f"fold must be in 0..{len(SPATIAL_CV_FOLDS) - 1}, got {fold}")
    val_regions = SPATIAL_CV_FOLDS[fold]
    train_regions = tuple(region for region in CV_REGIONS if region not in val_regions)
    return train_regions, val_regions, (TEST_REGION,)


def regions_for_split(split: str, fold: int) -> tuple[str, ...]:
    train_regions, val_regions, test_regions = fold_regions(fold)
    if split == "train":
        return train_regions
    if split == "val":
        return val_regions
    if split == "test":
        return test_regions
    raise ValueError(f"split must be train, val, or test, got {split!r}")


def tile_offsets(length: int, tile_size: int = TILE_SIZE, stride: int = TILE_STRIDE) -> list[int]:
    if length <= 0:
        return []
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    if length <= tile_size:
        return [0]
    offsets = list(range(0, length - tile_size + 1, stride))
    edge_offset = length - tile_size
    if offsets[-1] != edge_offset:
        offsets.append(edge_offset)
    return offsets


def normalize_db(values: np.ndarray, low: float = -30.0, high: float = 0.0) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32), low, high)
    return ((clipped - low) / (high - low)).astype(np.float32)


def normalize_clip(values: np.ndarray, high: float) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32), 0.0, high)
    return (clipped / high).astype(np.float32)


def rgb_to_hsv(rgb: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    r, g, b = rgb[0], rgb[1], rgb[2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    hue = np.zeros_like(maxc, dtype=np.float32)
    nonzero = delta > 0
    red_max = nonzero & (maxc == r)
    green_max = nonzero & (maxc == g)
    blue_max = nonzero & (maxc == b)
    hue[red_max] = ((g[red_max] - b[red_max]) / delta[red_max]) % 6.0
    hue[green_max] = ((b[green_max] - r[green_max]) / delta[green_max]) + 2.0
    hue[blue_max] = ((r[blue_max] - g[blue_max]) / delta[blue_max]) + 4.0
    hue /= 6.0

    saturation = np.zeros_like(maxc, dtype=np.float32)
    saturation[maxc > 0] = delta[maxc > 0] / maxc[maxc > 0]
    value = maxc.astype(np.float32)

    hsv = np.stack([hue, saturation, value]).astype(np.float32)
    hsv[:, ~valid_mask] = 0.0
    return hsv


def should_keep_tile(
    label_valid: np.ndarray,
    feature_valid: np.ndarray,
    flood: np.ndarray,
    threshold: float = VALID_COVERAGE_THRESHOLD,
) -> bool:
    if bool(np.any(flood)):
        return True
    return bool(label_valid.mean() >= threshold and feature_valid.mean() >= threshold)


def split_procanet_encoders(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if stack.shape[0] != len(CHANNELS_7CH):
        raise ValueError(f"expected {len(CHANNELS_7CH)} channels, got {stack.shape[0]}")
    encoder1 = stack
    encoder2 = stack[: len(PROCANET_ENCODER2_CHANNELS)]
    return encoder1, encoder2


def read_band(path: Path | str, band_index: int = 1) -> np.ndarray:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(path)
    arr = ds.GetRasterBand(band_index).ReadAsArray()
    ds = None
    return arr


def create_like(
    reference: gdal.Dataset,
    path: Path,
    band_count: int = 1,
    dtype: int = gdal.GDT_Float32,
    nodata: float | None = None,
) -> gdal.Dataset:
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        str(path),
        reference.RasterXSize,
        reference.RasterYSize,
        band_count,
        dtype,
        options=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    ds.SetGeoTransform(reference.GetGeoTransform())
    ds.SetProjection(reference.GetProjection())
    if nodata is not None:
        for idx in range(1, band_count + 1):
            ds.GetRasterBand(idx).SetNoDataValue(nodata)
    return ds


def write_single_band(reference: gdal.Dataset, path: Path, array: np.ndarray, dtype: int) -> None:
    ds = create_like(reference, path, 1, dtype)
    band = ds.GetRasterBand(1)
    band.WriteArray(array)
    band.FlushCache()
    ds.FlushCache()
    ds = None


def write_stack(reference: gdal.Dataset, path: Path, arrays: list[np.ndarray], descriptions: tuple[str, ...]) -> None:
    ds = create_like(reference, path, len(arrays), gdal.GDT_Float32)
    for idx, (array, desc) in enumerate(zip(arrays, descriptions), start=1):
        band = ds.GetRasterBand(idx)
        band.WriteArray(array.astype(np.float32))
        band.SetDescription(desc)
        band.FlushCache()
    ds.FlushCache()
    ds = None


def same_grid(a: gdal.Dataset, b: gdal.Dataset) -> bool:
    return (
        a.RasterXSize == b.RasterXSize
        and a.RasterYSize == b.RasterYSize
        and a.GetGeoTransform() == b.GetGeoTransform()
        and a.GetProjection() == b.GetProjection()
    )
