from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr

from scripts.preprocessing_utils import create_like, region_to_output_name


gdal.UseExceptions()

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SENTINEL_ROOT = ROOT / "dataset/satelit raw"
DEFAULT_UNOSAT_GDB = ROOT / "dataset/unosat/FL20251126IDN.gdb"
DEFAULT_OUTPUT_ROOT = ROOT / "dataset/labels_unosat_rasterized"
DEFAULT_ADMIN_BOUNDARY_ROOT = ROOT / "dataset/batas admin indo"


@dataclass(frozen=True)
class LabelLayerSets:
    flood: tuple[str, ...]
    valid: tuple[str, ...]
    water_river: tuple[str, ...]


def output_region_name(region: str) -> str:
    return region_to_output_name(region)


def find_s1_reference(files: list[Path]) -> Path:
    s1_files = sorted(path for path in files if path.name.startswith("S1_") and path.suffix.lower() in {".tif", ".tiff"})
    if not s1_files:
        raise FileNotFoundError("No S1 GeoTIFF found for region")
    if len(s1_files) > 1:
        names = ", ".join(path.name for path in s1_files)
        raise ValueError(f"Multiple S1 GeoTIFF files found for region: {names}")
    return s1_files[0]


def admin_boundary_names(region: str) -> tuple[str, ...]:
    base = region.replace("_", " ")
    names = [f"{base}-KAB_KOTA.geojson"]
    if not base.startswith(("Kota ", "Kabupaten ")):
        names.append(f"Kabupaten {base}-KAB_KOTA.geojson")
    return tuple(names)


def find_admin_boundary(region: str, admin_boundary_root: Path) -> Path:
    for name in admin_boundary_names(region):
        path = admin_boundary_root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No admin boundary GeoJSON found for region: {region}")


def apply_roi_mask(valid_mask, roi_mask):
    valid = np.asarray(valid_mask, dtype=np.uint8)
    roi = np.asarray(roi_mask, dtype=np.uint8)
    return ((valid > 0) & (roi > 0)).astype(np.uint8).tolist()


def classify_unosat_layers(layer_names: list[str]) -> LabelLayerSets:
    flood = tuple(name for name in layer_names if "FloodExtent" in name)
    valid = tuple(name for name in layer_names if "AnalysisExtent" in name)
    water_river = tuple(name for name in layer_names if "WaterExtent" in name or "River" in name)
    return LabelLayerSets(flood=flood, valid=valid, water_river=water_river)


def discover_regions(sentinel_root: Path) -> dict[str, Path]:
    regions: dict[str, Path] = {}
    for region_dir in sorted(path for path in sentinel_root.iterdir() if path.is_dir()):
        regions[region_dir.name] = find_s1_reference(list(region_dir.iterdir()))
    if not regions:
        raise FileNotFoundError(f"No Sentinel region folders found under {sentinel_root}")
    return regions


def open_dataset(path: Path | str, access: int = gdal.GA_ReadOnly) -> gdal.Dataset:
    ds = gdal.OpenEx(str(path), access)
    if ds is None:
        raise FileNotFoundError(path)
    return ds


def gdb_layer_names(gdb: gdal.Dataset) -> list[str]:
    return [gdb.GetLayerByIndex(idx).GetName() for idx in range(gdb.GetLayerCount())]


def merge_vector_layers(vector_ds: gdal.Dataset, layer_names: tuple[str, ...], merged_name: str) -> tuple[ogr.DataSource, ogr.Layer]:
    driver = ogr.GetDriverByName("MEM")
    merged_ds = driver.CreateDataSource(f"{merged_name}_ds")
    first_layer = vector_ds.GetLayerByName(layer_names[0])
    if first_layer is None:
        raise ValueError(f"Missing vector layer: {layer_names[0]}")

    merged_layer = merged_ds.CreateLayer(merged_name, srs=first_layer.GetSpatialRef(), geom_type=ogr.wkbUnknown)
    for layer_name in layer_names:
        source_layer = vector_ds.GetLayerByName(layer_name)
        if source_layer is None:
            raise ValueError(f"Missing vector layer: {layer_name}")
        source_layer.ResetReading()
        for source_feature in source_layer:
            geometry = source_feature.GetGeometryRef()
            if geometry is None:
                continue
            merged_feature = ogr.Feature(merged_layer.GetLayerDefn())
            merged_feature.SetGeometry(geometry.Clone())
            merged_layer.CreateFeature(merged_feature)
            merged_feature = None
    merged_layer.ResetReading()
    return merged_ds, merged_layer


def rasterize_to_array(
    reference: gdal.Dataset,
    vector_ds: gdal.Dataset,
    layer_names: tuple[str, ...],
    all_touched: bool = False,
) -> np.ndarray:
    merged_ds, merged_layer = merge_vector_layers(vector_ds, layer_names, "merged_rasterize_input")
    driver = gdal.GetDriverByName("MEM")
    ds = driver.Create("", reference.RasterXSize, reference.RasterYSize, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(reference.GetGeoTransform())
    ds.SetProjection(reference.GetProjection())
    band = ds.GetRasterBand(1)
    band.Fill(0)

    error = gdal.RasterizeLayer(
        ds,
        [1],
        merged_layer,
        burn_values=[1],
        options=[f"ALL_TOUCHED={str(all_touched).upper()}"],
    )
    if error != 0:
        raise RuntimeError(f"Failed to rasterize merged layer: {', '.join(layer_names)}")

    array = band.ReadAsArray().astype(np.uint8)
    ds = None
    merged_ds = None
    return array


def write_byte_mask(reference: gdal.Dataset, output_path: Path, array: np.ndarray, overwrite: bool = False) -> None:
    if output_path.exists():
        if not overwrite:
            return
        output_path.unlink()
    ds = create_like(reference, output_path, band_count=1, dtype=gdal.GDT_Byte, nodata=0)
    band = ds.GetRasterBand(1)
    band.WriteArray(array.astype(np.uint8))
    band.FlushCache()
    ds.FlushCache()
    ds = None


def rasterize_layers(
    reference: gdal.Dataset,
    gdb: gdal.Dataset,
    layer_names: tuple[str, ...],
    output_path: Path,
    overwrite: bool = False,
    all_touched: bool = False,
) -> None:
    if output_path.exists():
        if not overwrite:
            return
        output_path.unlink()

    array = rasterize_to_array(reference, gdb, layer_names, all_touched=all_touched)
    write_byte_mask(reference, output_path, array, overwrite=overwrite)


def rasterize_valid_mask(
    reference: gdal.Dataset,
    gdb: gdal.Dataset,
    layer_names: tuple[str, ...],
    roi_path: Path,
    output_path: Path,
    overwrite: bool = False,
    all_touched: bool = False,
) -> None:
    if output_path.exists():
        if not overwrite:
            return
        output_path.unlink()
    roi_ds = open_dataset(roi_path)
    roi_layers = tuple(gdb_layer_names(roi_ds))
    valid = rasterize_to_array(reference, gdb, layer_names, all_touched=all_touched)
    roi = rasterize_to_array(reference, roi_ds, roi_layers, all_touched=all_touched)
    clipped = ((valid > 0) & (roi > 0)).astype(np.uint8)
    write_byte_mask(reference, output_path, clipped, overwrite=True)
    roi_ds = None


def rasterize_region(
    region: str,
    reference_path: Path,
    gdb: gdal.Dataset,
    layers: LabelLayerSets,
    output_root: Path,
    admin_boundary_root: Path,
    overwrite: bool = False,
    all_touched: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    reference = open_dataset(reference_path)
    region_out = output_root / output_region_name(region)
    roi_path = find_admin_boundary(region, admin_boundary_root)
    outputs = [
        (layers.flood, region_out / "label_flood_binary.tif"),
        (layers.valid, region_out / "label_valid_mask.tif"),
        (layers.water_river, region_out / "label_water_river_mask.tif"),
    ]
    if dry_run:
        reference = None
        return [path for _, path in outputs]

    rasterize_layers(reference, gdb, layers.flood, region_out / "label_flood_binary.tif", overwrite=overwrite, all_touched=all_touched)
    rasterize_valid_mask(
        reference,
        gdb,
        layers.valid,
        roi_path,
        region_out / "label_valid_mask.tif",
        overwrite=overwrite,
        all_touched=all_touched,
    )
    rasterize_layers(
        reference,
        gdb,
        layers.water_river,
        region_out / "label_water_river_mask.tif",
        overwrite=overwrite,
        all_touched=all_touched,
    )
    reference = None
    return [path for _, path in outputs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rasterize UNOSAT flood label layers onto Sentinel-1 grids.")
    parser.add_argument("--sentinel-root", type=Path, default=DEFAULT_SENTINEL_ROOT)
    parser.add_argument("--unosat-gdb", type=Path, default=DEFAULT_UNOSAT_GDB)
    parser.add_argument("--admin-boundary-root", type=Path, default=DEFAULT_ADMIN_BOUNDARY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--region", action="append", help="Region folder name to process. Repeatable. Default: all.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output rasters.")
    parser.add_argument("--all-touched", action="store_true", help="Burn every pixel touched by a polygon.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs without writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regions = discover_regions(args.sentinel_root)
    selected = set(args.region or regions)
    missing = sorted(selected - set(regions))
    if missing:
        raise ValueError(f"Unknown region(s): {', '.join(missing)}")

    gdb = open_dataset(args.unosat_gdb)
    layers = classify_unosat_layers(gdb_layer_names(gdb))
    if not layers.flood or not layers.valid or not layers.water_river:
        raise ValueError(f"Incomplete UNOSAT layer groups: {layers}")

    print(f"UNOSAT GDB: {args.unosat_gdb}")
    print(f"Sentinel root: {args.sentinel_root}")
    print(f"Admin boundary root: {args.admin_boundary_root}")
    print(f"Output root: {args.output_root}")
    print(f"Flood layers: {len(layers.flood)}")
    print(f"Valid layers: {len(layers.valid)}")
    print(f"Water/river layers: {len(layers.water_river)}")

    for region, reference_path in regions.items():
        if region not in selected:
            continue
        outputs = rasterize_region(
            region,
            reference_path,
            gdb,
            layers,
            args.output_root,
            args.admin_boundary_root,
            overwrite=args.overwrite,
            all_touched=args.all_touched,
            dry_run=args.dry_run,
        )
        action = "would write" if args.dry_run else "wrote/skipped"
        print(f"{output_region_name(region)}: {action} {', '.join(str(path.name) for path in outputs)}")

    gdb = None


if __name__ == "__main__":
    main()
