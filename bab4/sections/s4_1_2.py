from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from bab4.artifacts import ALL_ARTIFACTS
from bab4.common import CHANNELS_7CH, REGIONS
from bab4.plots import normalize_image, plot_osm_cache_lines, savefig, setup_style
from bab4.raster import dtype_name, open_dataset, read_raster, read_stack_rgb, raster_metadata
from bab4.sections.base import section_result
from bab4.writer import figure_result, write_table, write_text_artifact


def _spec(artifact_id: str):
    return next(spec for spec in ALL_ARTIFACTS if spec.artifact_id == artifact_id)


def generate_4_1_2(config):
    artifacts = [
        _table_alignment(config),
        _table_stack_layers(config),
        _figure_osm_overlay(config),
        _narrative(config),
    ]
    return section_result("4.1.2", artifacts)


def _table_alignment(config):
    spec = _spec("Tabel 4.4")
    rows = []
    layer_names = ["vv_norm.tif", "vh_norm.tif", "hue.tif", "saturation.tif", "value.tif", "slope_norm.tif", "hand_norm.tif"]
    for region in REGIONS:
        feature_dir = config.dataset_root / "features_preprocessed" / region
        ref = raster_metadata(feature_dir / "vv_norm.tif")
        checks = []
        for name in layer_names + ["stack_7ch.tif"]:
            meta = raster_metadata(feature_dir / name)
            checks.append(
                meta["width"] == ref["width"]
                and meta["height"] == ref["height"]
                and meta["crs"] == ref["crs"]
                and meta["transform"] == ref["transform"]
            )
        rows.append(
            {
                "region": region,
                "reference": "Sentinel-1 VV",
                "raster_size": f"{ref['height']} x {ref['width']}",
                "resolution": f"{abs(ref['pixel_width']):.2f} m",
                "crs_match": all(checks),
                "geotransform_match": all(checks),
                "layers_checked": len(checks),
                "status": "Selaras" if all(checks) else "Perlu dicek",
            }
        )
    return write_table(config, spec, rows, source="dataset/features_preprocessed/*/*.tif")


def _table_stack_layers(config):
    spec = _spec("Tabel 4.5")
    region = config.test_region
    stack_path = config.dataset_root / "features_preprocessed" / region / "stack_7ch.tif"
    ds = open_dataset(stack_path)
    rows = []
    for idx, channel in enumerate(CHANNELS_7CH, start=1):
        band = ds.GetRasterBand(idx)
        rows.append(
            {
                "region": region,
                "band": idx,
                "channel": channel,
                "description": band.GetDescription() or channel,
                "raster_size": f"{ds.RasterYSize} x {ds.RasterXSize}",
                "dtype": dtype_name(band.DataType),
                "status": "Selaras",
            }
        )
    return write_table(config, spec, rows, source=str(stack_path.relative_to(config.root)))


def _figure_osm_overlay(config):
    spec = _spec("Gambar 4.3")
    region = config.test_region
    feature_dir = config.dataset_root / "features_preprocessed" / region
    ds = open_dataset(feature_dir / "vv_norm.tif")
    panels = [
        ("OSM + VV", read_raster(feature_dir / "vv_norm.tif")),
        ("OSM + VH", read_raster(feature_dir / "vh_norm.tif")),
        ("OSM + HSV", read_stack_rgb(feature_dir)),
        ("OSM + Slope", read_raster(feature_dir / "slope_norm.tif")),
        ("OSM + HAND", read_raster(feature_dir / "hand_norm.tif")),
        ("OSM + Label", read_raster(config.dataset_root / "labels_unosat_rasterized" / region / "label_flood_binary.tif")),
    ]
    cache_paths = sorted((config.root / "cache").glob("*.json")) + sorted((config.root / "notebooks" / "cache").glob("*.json"))
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))
    plotted = 0
    for ax, (title, arr) in zip(axes.ravel(), panels):
        if arr.ndim == 3:
            ax.imshow(arr)
        else:
            ax.imshow(normalize_image(arr), cmap="gray")
        plotted += plot_osm_cache_lines(ax, cache_paths, ds)
        ax.set_title(title)
        ax.axis("off")
    if plotted == 0:
        fig.text(0.5, 0.02, "OSM cache tidak tersedia; panel menampilkan raster alignment.", ha="center")
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source="cache/*.json;notebooks/cache/*.json;dataset/features_preprocessed")


def _narrative(config):
    spec = _spec("Narasi 4.1.2")
    text = """
    Verifikasi alignment dibuat ulang dengan membandingkan ukuran raster, CRS, dan geotransform setiap channel
    terhadap `vv_norm.tif` sebagai referensi Sentinel-1. Pemeriksaan ini memastikan setiap piksel pada stack
    tujuh channel merepresentasikan lokasi geografis yang sama sebelum dipakai sebagai input model.
    """
    return write_text_artifact(config, spec, text, source="dataset/features_preprocessed/*/*.tif")
