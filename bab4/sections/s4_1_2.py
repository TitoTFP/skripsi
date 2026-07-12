from __future__ import annotations

from bab4.artifacts import ALL_ARTIFACTS
from bab4.common import CHANNELS_7CH, REGIONS
from bab4.raster import open_dataset, raster_metadata
from bab4.sections.base import section_result
from bab4.writer import write_table, write_text_artifact


def _spec(artifact_id: str):
    return next(spec for spec in ALL_ARTIFACTS if spec.artifact_id == artifact_id)


def generate_4_1_2(config):
    artifacts = [
        _table_alignment(config),
        _table_stack_layers(config),
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


def _narrative(config):
    spec = _spec("Narasi 4.1.2")
    text = """
    Verifikasi alignment dibuat ulang dengan membandingkan ukuran raster, CRS, dan geotransform setiap channel
    terhadap `vv_norm.tif` sebagai referensi Sentinel-1. Pemeriksaan ini memastikan setiap piksel pada stack
    tujuh channel merepresentasikan lokasi geografis yang sama sebelum dipakai sebagai input model.
    """
    return write_text_artifact(config, spec, text, source="dataset/features_preprocessed/*/*.tif")
