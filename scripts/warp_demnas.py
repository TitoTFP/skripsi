from __future__ import annotations

import argparse
from pathlib import Path
from osgeo import gdal

from scripts.preprocessing_utils import region_to_output_name

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SENTINEL_ROOT = ROOT / "dataset/satelit raw"
DEFAULT_DEMNAS_EXPORTS_ROOT = ROOT / "dataset/DEMNAS_Exports"
DEFAULT_OUTPUT_ROOT = ROOT / "dataset/DEMNAS_warped_to_sentinel"


def find_s1_reference(region_dir: Path) -> Path:
    s1_files = sorted(
        path
        for path in region_dir.iterdir()
        if path.name.startswith("S1_") and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not s1_files:
        raise FileNotFoundError(f"No S1 GeoTIFF found in {region_dir}")
    if len(s1_files) > 1:
        names = ", ".join(path.name for path in s1_files)
        raise ValueError(f"Multiple S1 GeoTIFF files found in {region_dir}: {names}")
    return s1_files[0]


def warp_dem(dem_raw: Path, ref_s1: Path, dem_out: Path) -> None:
    ref = gdal.Open(str(ref_s1), gdal.GA_ReadOnly)
    if ref is None:
        raise FileNotFoundError(f"Could not open reference image: {ref_s1}")
    proj = ref.GetProjection()
    gt = ref.GetGeoTransform()
    w = ref.RasterXSize
    h = ref.RasterYSize

    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + w * gt[1]
    ymin = ymax + h * gt[5]

    dem_out.parent.mkdir(parents=True, exist_ok=True)

    warp_options = gdal.WarpOptions(
        format="GTiff",
        dstSRS=proj,
        outputBounds=[xmin, ymin, xmax, ymax],
        xRes=gt[1],
        yRes=abs(gt[5]),
        resampleAlg="bilinear",
        creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    gdal.Warp(str(dem_out), str(dem_raw), options=warp_options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warp raw DEMNAS images to match Sentinel-1 reference grids."
    )
    parser.add_argument("--sentinel-root", type=Path, default=DEFAULT_SENTINEL_ROOT)
    parser.add_argument("--demnas-exports-root", type=Path, default=DEFAULT_DEMNAS_EXPORTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--region",
        action="append",
        help="Region directory name (e.g. 'Aceh Besar'). Can be repeated. Default: all.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing warped rasters.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned actions without running warp."
    )
    return parser.parse_args()


def main() -> None:
    gdal.UseExceptions()
    args = parse_args()

    # Discover region directories
    if not args.sentinel_root.exists():
        raise FileNotFoundError(f"Sentinel root directory does not exist: {args.sentinel_root}")

    region_dirs = sorted(path for path in args.sentinel_root.iterdir() if path.is_dir())
    if not region_dirs:
        raise FileNotFoundError(f"No region folders found under {args.sentinel_root}")

    selected_regions = set(args.region) if args.region else {r.name for r in region_dirs}

    print(f"DEMNAS exports root: {args.demnas_exports_root}")
    print(f"Output root: {args.output_root}")
    if args.dry_run:
        print("Dry run mode active. No files will be modified.")

    for r_dir in region_dirs:
        region_name = r_dir.name
        if region_name not in selected_regions:
            continue

        out_region = region_to_output_name(region_name)
        dem_raw = args.demnas_exports_root / out_region / f"DEMNAS_{out_region}.tif"
        if not dem_raw.exists():
            print(f"WARNING: Raw DEM file not found: {dem_raw}. Skipping {region_name}.")
            continue

        try:
            ref_s1 = find_s1_reference(r_dir)
        except Exception as e:
            print(f"ERROR searching reference in {region_name}: {e}. Skipping.")
            continue

        dem_out = args.output_root / out_region / f"DEMNAS_{out_region}_warped_to_sentinel.tif"

        if dem_out.exists() and not args.overwrite:
            print(f"{out_region}: Aligned DEM already exists. Use --overwrite to replace. Skipping.")
            continue

        action = "Would warp" if args.dry_run else "Warping"
        print(f"{out_region}: {action} {dem_raw} -> {dem_out}")

        if not args.dry_run:
            try:
                warp_dem(dem_raw, ref_s1, dem_out)
                print(f"{out_region}: Warp complete.")
            except Exception as e:
                print(f"ERROR warping {out_region}: {e}")


if __name__ == "__main__":
    main()
