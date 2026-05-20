from __future__ import annotations

import argparse
from pathlib import Path
from osgeo import gdal

from scripts.preprocessing_utils import region_to_output_name
from scripts.rasterize_unosat_labels import discover_regions, find_admin_boundary

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SENTINEL_ROOT = ROOT / "dataset/satelit raw"
DEFAULT_INPUT_DEM = (
    ROOT
    / "dataset/indonesia-geospasial.com DEMNAS_sumatera/dem_sumatera_a_1.jp2"
)
DEFAULT_ADMIN_BOUNDARY_ROOT = ROOT / "dataset/batas admin indo"
DEFAULT_OUTPUT_ROOT = ROOT / "dataset/DEMNAS_Exports"


def crop_dem_to_boundary(input_dem: Path, geojson_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    warp_options = gdal.WarpOptions(
        format="GTiff",
        cutlineDSName=str(geojson_path),
        cropToCutline=True,
        creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    gdal.Warp(str(output_path), str(input_dem), options=warp_options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop raw DEM dataset to region boundaries."
    )
    parser.add_argument("--sentinel-root", type=Path, default=DEFAULT_SENTINEL_ROOT)
    parser.add_argument("--input-dem", type=Path, default=DEFAULT_INPUT_DEM)
    parser.add_argument("--admin-boundary-root", type=Path, default=DEFAULT_ADMIN_BOUNDARY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--region",
        action="append",
        help="Region directory name (e.g. 'Aceh Besar'). Can be repeated. Default: all.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing cropped rasters.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned actions without running crop."
    )
    return parser.parse_args()


def main() -> None:
    gdal.UseExceptions()
    args = parse_args()

    if not args.sentinel_root.exists():
        raise FileNotFoundError(f"Sentinel root directory does not exist: {args.sentinel_root}")
    if not args.input_dem.exists():
        raise FileNotFoundError(f"Input DEM file does not exist: {args.input_dem}")
    if not args.admin_boundary_root.exists():
        raise FileNotFoundError(f"Admin boundary directory does not exist: {args.admin_boundary_root}")

    regions = discover_regions(args.sentinel_root)
    selected = set(args.region or regions)
    missing = sorted(selected - set(regions))
    if missing:
        raise ValueError(f"Unknown region(s): {', '.join(missing)}")

    print(f"Input DEM: {args.input_dem}")
    print(f"Admin boundary root: {args.admin_boundary_root}")
    print(f"Output root: {args.output_root}")
    if args.dry_run:
        print("Dry run mode active. No files will be modified.")

    for region in sorted(regions):
        if region not in selected:
            continue

        out_region = region_to_output_name(region)
        try:
            geojson_path = find_admin_boundary(region, args.admin_boundary_root)
        except Exception as e:
            print(f"ERROR finding boundary for {region}: {e}. Skipping.")
            continue

        output_path = args.output_root / out_region / f"DEMNAS_{out_region}.tif"

        if output_path.exists() and not args.overwrite:
            print(f"{out_region}: Cropped DEM already exists. Skipping.")
            continue

        action = "Would crop" if args.dry_run else "Cropping"
        print(f"{out_region}: {action} {args.input_dem} using {geojson_path} -> {output_path}")

        if not args.dry_run:
            try:
                crop_dem_to_boundary(args.input_dem, geojson_path, output_path)
                print(f"{out_region}: Crop complete.")
            except Exception as e:
                print(f"ERROR cropping {out_region}: {e}")


if __name__ == "__main__":
    main()
