from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bab4.artifacts import ALL_ARTIFACTS
from bab4.common import REGIONS, fmt_float, read_csv_row_map, region_quality_from_s2_pct, to_float, to_int
from bab4.plots import normalize_image, savefig, setup_style
from bab4.raster import band_stats, load_npz, read_raster, read_stack_rgb
from bab4.sections.base import section_result
from bab4.writer import figure_result, write_table, write_text_artifact


def _spec(artifact_id: str):
    return next(spec for spec in ALL_ARTIFACTS if spec.artifact_id == artifact_id)


def generate_4_1_1(config):
    artifacts = [
        _table_sentinel1(config),
        _table_s2_valid(config),
        _table_demnas(config),
        _figure_s2_valid_vs_empty(config),
        _figure_channel_example(config),
        _narrative(config),
    ]
    return section_result("4.1.1", artifacts)


def _table_sentinel1(config):
    spec = _spec("Tabel 4.1")
    summary = read_csv_row_map(config.dataset_root / "feature_preprocessing_summary.csv")
    rows = []
    for region in REGIONS:
        valid_pct = to_float(summary.get(region, {}).get("feature_valid_pct"))
        for channel, filename in (("VV", "vv_norm.tif"), ("VH", "vh_norm.tif")):
            path = config.dataset_root / "features_preprocessed" / region / filename
            stats = band_stats(path)
            rows.append(
                {
                    "region": region,
                    "channel": channel,
                    "min": fmt_float(stats["min"]),
                    "max": fmt_float(stats["max"]),
                    "mean": fmt_float(stats["mean"]),
                    "std": fmt_float(stats["std"]),
                    "valid_pct": fmt_float(valid_pct),
                    "source": str(path.relative_to(config.root)),
                }
            )
    return write_table(config, spec, rows, source="dataset/features_preprocessed/*/vv_norm.tif;vh_norm.tif")


def _table_s2_valid(config):
    spec = _spec("Tabel 4.2")
    feature = read_csv_row_map(config.dataset_root / "feature_preprocessing_summary.csv")
    tile = read_csv_row_map(config.dataset_root / "preprocessing_summary.csv")
    rows = []
    for region in REGIONS:
        frow = feature.get(region, {})
        trow = tile.get(region, {})
        s2_pct = to_float(frow.get("s2_valid_pct"))
        rows.append(
            {
                "region": region,
                "raster_s2_valid_pct": fmt_float(s2_pct),
                "tile_s2_valid_pixels": to_int(trow.get("s2_valid_pixels")),
                "tile_valid_pixels": to_int(trow.get("valid_pixels")),
                "tile_s2_valid_pct": fmt_float(to_float(trow.get("s2_valid_pixels")) / max(to_float(trow.get("valid_pixels")), 1) * 100),
                "quality_status": region_quality_from_s2_pct(s2_pct),
                "source": "dataset/feature_preprocessing_summary.csv;dataset/preprocessing_summary.csv",
            }
        )
    return write_table(config, spec, rows, source="dataset/feature_preprocessing_summary.csv;dataset/preprocessing_summary.csv")


def _table_demnas(config):
    spec = _spec("Tabel 4.3")
    summary = read_csv_row_map(config.dataset_root / "feature_preprocessing_summary.csv")
    rows = []
    for region in REGIONS:
        valid_pct = to_float(summary.get(region, {}).get("feature_valid_pct"))
        for channel, filename in (("Slope", "slope_norm.tif"), ("HAND", "hand_norm.tif")):
            path = config.dataset_root / "features_preprocessed" / region / filename
            stats = band_stats(path)
            rows.append(
                {
                    "region": region,
                    "channel": channel,
                    "min": fmt_float(stats["min"]),
                    "max": fmt_float(stats["max"]),
                    "mean": fmt_float(stats["mean"]),
                    "std": fmt_float(stats["std"]),
                    "valid_pct": fmt_float(valid_pct),
                    "source": str(path.relative_to(config.root)),
                }
            )
    return write_table(config, spec, rows, source="dataset/features_preprocessed/*/slope_norm.tif;hand_norm.tif")


def _figure_channel_example(config):
    spec = _spec("Gambar 4.2")
    region = config.test_region
    feature_dir = config.dataset_root / "features_preprocessed" / region
    panels = [
        ("VV", read_raster(feature_dir / "vv_norm.tif")),
        ("VH", read_raster(feature_dir / "vh_norm.tif")),
        ("HSV pseudo-RGB", read_stack_rgb(feature_dir)),
        ("Slope", read_raster(feature_dir / "slope_norm.tif")),
        ("HAND", read_raster(feature_dir / "hand_norm.tif")),
    ]
    setup_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 4.2))
    for ax, (title, arr) in zip(axes, panels):
        if arr.ndim == 3:
            ax.imshow(arr)
        else:
            ax.imshow(normalize_image(arr), cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Contoh channel input multisensor - {region}")
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source=f"dataset/features_preprocessed/{region}/")


def _figure_s2_valid_vs_empty(config):
    spec = _spec("Gambar 4.1")
    examples = [config.test_region, "Aceh_Tamiang"]
    setup_style()
    fig, axes = plt.subplots(len(examples), 3, figsize=(10, 6.4))
    for row_idx, region in enumerate(examples):
        tile = _representative_tile(config.dataset_root / "tiles" / "7ch" / "by_region" / region)
        x = np.asarray(tile["x"])
        s2 = np.asarray(tile["s2_valid_mask"])[0]
        rgb = _tile_hsv_to_rgb(x)
        axes[row_idx, 0].imshow(normalize_image(x[0]), cmap="gray")
        axes[row_idx, 0].set_title(f"{region} - VV")
        axes[row_idx, 1].imshow(rgb)
        axes[row_idx, 1].set_title("HSV pseudo-RGB")
        axes[row_idx, 2].imshow(s2, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].set_title("S2 valid mask")
        for ax in axes[row_idx]:
            ax.axis("off")
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source="dataset/tiles/7ch/by_region/*/*.npz")


def _representative_tile(tile_dir: Path) -> dict:
    best_path = None
    best_score = -1.0
    for path in sorted(tile_dir.glob("*.npz"))[:250]:
        tile = load_npz(path)
        score = float(np.asarray(tile["valid_mask"]).mean()) + float(np.asarray(tile["s2_valid_mask"]).mean())
        if score > best_score:
            best_path, best_score = path, score
    return load_npz(best_path or next(tile_dir.glob("*.npz")))


def _tile_hsv_to_rgb(x: np.ndarray) -> np.ndarray:
    from bab4.plots import hsv_to_rgb

    return hsv_to_rgb(x[2:5])


def _narrative(config):
    spec = _spec("Narasi 4.1.1")
    text = """
    Preprocessing menghasilkan tujuh channel input: VV, VH, Hue, Saturation, Value, Slope, dan HAND.
    Tabel 4.1 sampai Tabel 4.3 dibuat ulang dari raster dan ringkasan dataset, sehingga angka pada folder
    `bab4/outputs` dapat dilacak ke `dataset/features_preprocessed`, `dataset/feature_preprocessing_summary.csv`,
    dan `dataset/preprocessing_summary.csv`. Wilayah dengan Sentinel-2 kosong atau hampir kosong tetap dipertahankan
    sebagai kasus ketahanan model karena kanal SAR dan topografi masih tersedia.
    """
    return write_text_artifact(config, spec, text, source="dataset/features_preprocessed;dataset/*.csv")
