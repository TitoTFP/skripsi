from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from osgeo import gdal
from whitebox import WhiteboxTools

from scripts.preprocessing_utils import (
    CHANNELS_7CH,
    normalize_clip,
    normalize_db,
    read_band,
    region_to_output_name,
    rgb_to_hsv,
    create_like,
)


ROOT = Path(__file__).resolve().parents[1]
S1_ROOT = ROOT / "dataset/satelit raw"
DEM_ROOT = ROOT / "dataset/DEMNAS_warped_to_sentinel"
LABEL_ROOT = ROOT / "dataset/labels_unosat_rasterized"
OUT_ROOT = ROOT / "dataset/features_preprocessed"
SUMMARY_PATH = ROOT / "dataset/feature_preprocessing_summary.csv"
DEM_NODATA = -32767
STREAM_THRESHOLD = 1000
BLOCK_SIZE = 512


def find_region_inputs(region_dir: Path) -> tuple[Path, Path, Path, Path]:
    region = region_dir.name
    out_region = region_to_output_name(region)
    s1 = next(region_dir.glob("S1_*.tif"))
    s2 = next(region_dir.glob("S2_*.tif"))
    dem = DEM_ROOT / out_region / f"DEMNAS_{out_region}_warped_to_sentinel.tif"
    labels = LABEL_ROOT / out_region
    if not dem.exists():
        raise FileNotFoundError(dem)
    if not labels.exists():
        raise FileNotFoundError(labels)
    return s1, s2, dem, labels


def run_gdaldem_slope(dem_path: Path, slope_path: Path) -> None:
    slope_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "gdaldem",
            "slope",
            str(dem_path),
            str(slope_path),
            "-of",
            "GTiff",
            "-compute_edges",
            "-co",
            "TILED=YES",
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "BIGTIFF=IF_SAFER",
        ],
        check=True,
    )


def run_whitebox_hand(dem_path: Path, hand_raw_path: Path) -> None:
    hand_raw_path.parent.mkdir(parents=True, exist_ok=True)
    wbt = WhiteboxTools()
    wbt.verbose = False
    with tempfile.TemporaryDirectory(prefix="hand_") as tmp_name:
        tmp = Path(tmp_name)
        breached = tmp / "dem_breached.tif"
        pointer = tmp / "d8_pointer.tif"
        accumulation = tmp / "flow_accum.tif"
        streams = tmp / "streams.tif"
        hand_tmp = tmp / "hand.tif"
        wbt.breach_depressions(str(dem_path), str(breached))
        wbt.d8_pointer(str(breached), str(pointer))
        wbt.d8_flow_accumulation(str(breached), str(accumulation), out_type="cells")
        wbt.extract_streams(str(accumulation), str(streams), threshold=STREAM_THRESHOLD)
        wbt.elevation_above_stream(str(breached), str(streams), str(hand_tmp))
        if not hand_tmp.exists():
            raise RuntimeError(f"Whitebox HAND output missing: {hand_tmp}")
        shutil.copyfile(hand_tmp, hand_raw_path)


def preprocess_region(region_dir: Path) -> dict[str, object]:
    region = region_dir.name
    out_region = region_to_output_name(region)
    out_dir = OUT_ROOT / out_region
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_path, s2_path, dem_path, label_dir = find_region_inputs(region_dir)
    ref = gdal.Open(str(s1_path), gdal.GA_ReadOnly)
    s1_ds = gdal.Open(str(s1_path), gdal.GA_ReadOnly)
    s2_ds = gdal.Open(str(s2_path), gdal.GA_ReadOnly)

    slope_raw = out_dir / "slope_degrees.tif"
    if not slope_raw.exists():
        run_gdaldem_slope(dem_path, slope_raw)

    hand_raw = out_dir / "hand_meters.tif"
    if not hand_raw.exists():
        run_whitebox_hand(dem_path, hand_raw)
    slope_ds = gdal.Open(str(slope_raw), gdal.GA_ReadOnly)
    hand_ds = gdal.Open(str(hand_raw), gdal.GA_ReadOnly)

    output_specs = {
        "vv_norm": create_like(ref, out_dir / "vv_norm.tif", 1, gdal.GDT_Float32),
        "vh_norm": create_like(ref, out_dir / "vh_norm.tif", 1, gdal.GDT_Float32),
        "hue": create_like(ref, out_dir / "hue.tif", 1, gdal.GDT_Float32),
        "saturation": create_like(ref, out_dir / "saturation.tif", 1, gdal.GDT_Float32),
        "value": create_like(ref, out_dir / "value.tif", 1, gdal.GDT_Float32),
        "s2_valid": create_like(ref, out_dir / "s2_valid_mask.tif", 1, gdal.GDT_Byte),
        "slope": create_like(ref, out_dir / "slope_norm.tif", 1, gdal.GDT_Float32),
        "hand": create_like(ref, out_dir / "hand_norm.tif", 1, gdal.GDT_Float32),
        "feature_valid": create_like(ref, out_dir / "feature_valid_mask.tif", 1, gdal.GDT_Byte),
        "stack": create_like(ref, out_dir / "stack_7ch.tif", 7, gdal.GDT_Float32),
    }
    for idx, desc in enumerate(CHANNELS_7CH, start=1):
        output_specs["stack"].GetRasterBand(idx).SetDescription(desc)

    s2_valid_pixels = 0
    feature_valid_pixels = 0
    total_pixels = ref.RasterXSize * ref.RasterYSize
    for yoff in range(0, ref.RasterYSize, BLOCK_SIZE):
        ysize = min(BLOCK_SIZE, ref.RasterYSize - yoff)
        xsize = ref.RasterXSize
        vv = s1_ds.GetRasterBand(1).ReadAsArray(0, yoff, xsize, ysize).astype(np.float32)
        vh = s1_ds.GetRasterBand(2).ReadAsArray(0, yoff, xsize, ysize).astype(np.float32)
        s1_valid = np.isfinite(vv) & np.isfinite(vh)
        vv_norm = normalize_db(vv)
        vh_norm = normalize_db(vh)
        vv_norm[~s1_valid] = 0.0
        vh_norm[~s1_valid] = 0.0

        b4 = s2_ds.GetRasterBand(3).ReadAsArray(0, yoff, xsize, ysize).astype(np.float32)
        b8 = s2_ds.GetRasterBand(4).ReadAsArray(0, yoff, xsize, ysize).astype(np.float32)
        b12 = s2_ds.GetRasterBand(6).ReadAsArray(0, yoff, xsize, ysize).astype(np.float32)
        rgb = np.stack([b12, b8, b4]).astype(np.float32)
        rgb_finite = np.isfinite(rgb).all(axis=0)
        rgb_nonzero = np.any(rgb != 0.0, axis=0)
        s2_valid = rgb_finite & rgb_nonzero
        hsv = rgb_to_hsv(rgb, s2_valid)

        slope = slope_ds.GetRasterBand(1).ReadAsArray(0, yoff, xsize, ysize).astype(np.float32)
        slope_valid = np.isfinite(slope) & (slope != DEM_NODATA)
        slope_norm = normalize_clip(slope, 45.0)
        slope_norm[~slope_valid] = 0.0

        hand = hand_ds.GetRasterBand(1).ReadAsArray(0, yoff, xsize, ysize).astype(np.float32)
        hand_valid = np.isfinite(hand) & (hand != DEM_NODATA) & (hand >= 0)
        hand_norm = normalize_clip(hand, 50.0)
        hand_norm[~hand_valid] = 0.0

        feature_valid = s1_valid & slope_valid & hand_valid
        s2_valid_pixels += int(s2_valid.sum())
        feature_valid_pixels += int(feature_valid.sum())
        outputs = {
            "vv_norm": vv_norm,
            "vh_norm": vh_norm,
            "hue": hsv[0],
            "saturation": hsv[1],
            "value": hsv[2],
            "s2_valid": s2_valid.astype(np.uint8),
            "slope": slope_norm,
            "hand": hand_norm,
            "feature_valid": feature_valid.astype(np.uint8),
        }
        for key, array in outputs.items():
            output_specs[key].GetRasterBand(1).WriteArray(array, 0, yoff)
        stack_arrays = [vv_norm, vh_norm, hsv[0], hsv[1], hsv[2], slope_norm, hand_norm]
        for idx, array in enumerate(stack_arrays, start=1):
            output_specs["stack"].GetRasterBand(idx).WriteArray(array.astype(np.float32), 0, yoff)

    for ds in output_specs.values():
        ds.FlushCache()
    output_specs = {}

    label_valid = read_band(label_dir / "label_valid_mask.tif", 1).astype(bool)
    flood = read_band(label_dir / "label_flood_binary.tif", 1).astype(bool)
    water = read_band(label_dir / "label_water_river_mask.tif", 1).astype(bool)
    ref = None
    return {
        "region": out_region,
        "s2_valid_pct": float(s2_valid_pixels / total_pixels * 100.0),
        "feature_valid_pct": float(feature_valid_pixels / total_pixels * 100.0),
        "label_valid_pct": float(label_valid.mean() * 100.0),
        "flood_pixels": int(flood.sum()),
        "water_river_pixels": int(water.sum()),
    }


def main() -> None:
    gdal.UseExceptions()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [preprocess_region(region_dir) for region_dir in sorted(S1_ROOT.iterdir()) if region_dir.is_dir()]
    with SUMMARY_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['region']}: s2_valid={row['s2_valid_pct']:.4f}% "
            f"feature_valid={row['feature_valid_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
