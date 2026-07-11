from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from bab4.artifacts import ALL_ARTIFACTS
from bab4.common import CHANNELS_7CH, REGIONS
from bab4.plots import hsv_to_display_rgb, normalize_image, plot_osm_cache_lines, savefig, setup_style
from bab4.raster import load_npz, open_dataset, read_stack_rgb, raster_metadata
from bab4.sections.base import section_result
from bab4.writer import figure_result, write_table, write_text_artifact

OSM_TILE = "Aceh_Utara_r001280_c005632"


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
                "wilayah": region.replace("_", " "),
                "ukuran_raster": f"{ref['height']} x {ref['width']}",
                "resolusi_piksel": f"{abs(ref['pixel_width']):.0f} x {abs(ref['pixel_width']):.0f}",
                "crs_proyeksi": _epsg_label(ref["crs"]),
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
                "band": idx,
                "nama_layer": channel,
                "sumber_layer": _source_layer_name(channel),
                "ukuran_raster": f"{ds.RasterYSize} x {ds.RasterXSize}",
                "resolusi_piksel": f"{abs(ds.GetGeoTransform()[1]):.0f} x {abs(ds.GetGeoTransform()[1]):.0f}",
                "crs_proyeksi": _epsg_label(ds.GetProjection()),
            }
        )
    return write_table(config, spec, rows, source=str(stack_path.relative_to(config.root)))


def _epsg_label(crs: str) -> str:
    for code in ("32646", "32647", "32747"):
        if code in crs:
            return f"EPSG:{code}"
    return crs


def _source_layer_name(channel: str) -> str:
    return {
        "VV": "vv_norm.tif",
        "VH": "vh_norm.tif",
        "Hue": "hue.tif",
        "Saturation": "saturation.tif",
        "Value": "value.tif",
        "Slope": "slope_norm.tif",
        "HAND": "hand_norm.tif",
    }[channel]


def _figure_osm_overlay(config):
    spec = _spec("Gambar 4.3")
    region = config.test_region
    tile_path = config.dataset_root / "tiles" / "7ch" / "by_region" / region / f"{OSM_TILE}.npz"
    tile = load_npz(tile_path)
    x = tile["x"]
    row0 = int(tile["row"])
    col0 = int(tile["col"])
    size = int(x.shape[-1])
    raw_ds = _open_raw_reference(config)
    panels = [
        ("OSM + VV", normalize_image(x[0]), "gray"),
        ("OSM + VH", normalize_image(x[1]), "gray"),
        ("OSM + HSV/pseudo-RGB", hsv_to_display_rgb(x[2:5]), None),
        ("OSM + Slope", normalize_image(x[5]), "magma"),
        ("OSM + HAND", normalize_image(x[6]), "viridis"),
        ("OSM + label UNOSAT", tile["y"][0], "Blues"),
    ]
    cache_paths = sorted((config.root / "cache").glob("*.json")) + sorted((config.root / "notebooks" / "cache").glob("*.json"))
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.4))
    extent = [col0, col0 + size, row0 + size, row0]
    plotted = 0
    for idx, (ax, (title, arr, cmap)) in enumerate(zip(axes.ravel(), panels)):
        if arr.ndim == 3:
            ax.imshow(arr, extent=extent)
        else:
            ax.imshow(arr, cmap=cmap, extent=extent, vmin=0 if title.endswith("UNOSAT") else None, vmax=1 if title.endswith("UNOSAT") else None)
        plotted += plot_osm_cache_lines(ax, cache_paths, raw_ds)
        ax.set_xlim(col0, col0 + size)
        ax.set_ylim(row0 + size, row0)
        ax.set_title(title, fontsize=8)
        ax.text(0.5, -0.08, f"({chr(97 + idx)})", transform=ax.transAxes, ha="center", va="top", fontsize=9)
        ax.axis("off")
    if plotted == 0:
        fig.text(0.5, 0.02, "OSM cache tidak tersedia; panel menampilkan crop tile stack.", ha="center")
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source=f"{tile_path.relative_to(config.root)};cache/*.json;dataset/satelit raw/Aceh Utara")


def _open_raw_reference(config):
    raw_dir = config.dataset_root / "satelit raw" / "Aceh Utara"
    candidates = sorted(raw_dir.glob("S1_Aceh_Utara_*.tif"))
    if not candidates:
        return open_dataset(config.dataset_root / "features_preprocessed" / "Aceh_Utara" / "vv_norm.tif")
    return open_dataset(candidates[0])


def _narrative(config):
    spec = _spec("Narasi 4.1.2")
    text = """
    Verifikasi alignment dibuat ulang dengan membandingkan ukuran raster, CRS, dan geotransform setiap channel
    terhadap `vv_norm.tif` sebagai referensi Sentinel-1. Pemeriksaan ini memastikan setiap piksel pada stack
    tujuh channel merepresentasikan lokasi geografis yang sama sebelum dipakai sebagai input model.
    """
    return write_text_artifact(config, spec, text, source="dataset/features_preprocessed/*/*.tif")
