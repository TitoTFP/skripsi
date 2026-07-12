from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from bab4.artifacts import ALL_ARTIFACTS
from bab4.common import REGIONS, fmt_float, pct, read_csv_row_map, to_float
from bab4.plots import normalize_image, savefig, setup_style
from bab4.raster import boolean_mask_counts, load_npz, masked_band_stats, read_raster, read_stack_rgb
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
        feature_dir = config.dataset_root / "features_preprocessed" / region
        mask_path = feature_dir / "feature_valid_mask.tif"
        vv_stats = masked_band_stats(feature_dir / "vv_norm.tif", mask_path)
        vh_stats = masked_band_stats(feature_dir / "vh_norm.tif", mask_path)
        rows.append(
            {
                "wilayah": region.replace("_", " "),
                "mean_vv": fmt_float(vv_stats["mean"]),
                "std_vv": fmt_float(vv_stats["std"]),
                "mean_vh": fmt_float(vh_stats["mean"]),
                "std_vh": fmt_float(vh_stats["std"]),
                "piksel_valid_pct": fmt_float(valid_pct),
            }
        )
    return write_table(
        config,
        spec,
        rows,
        source="dataset/features_preprocessed/*/{vv_norm.tif,vh_norm.tif,feature_valid_mask.tif}",
    )


def _table_s2_valid(config):
    spec = _spec("Tabel 4.2")
    rows = []
    for region in REGIONS:
        feature_dir = config.dataset_root / "features_preprocessed" / region
        counts = boolean_mask_counts(feature_dir / "s2_valid_mask.tif", feature_dir / "feature_valid_mask.tif")
        s2_pct = pct(counts["true_count"], counts["total_count"])
        s2_in_feature_pct = pct(counts["intersection_count"], counts["mask_count"])
        feature_pct = pct(counts["mask_count"], counts["total_count"])
        rows.append(
            {
                "wilayah": region.replace("_", " "),
                "s2_valid_terhadap_raster_pct": fmt_float(s2_pct),
                "s2_valid_dalam_feature_valid_pct": fmt_float(s2_in_feature_pct),
                "feature_valid_pct": fmt_float(feature_pct),
            }
        )
    return write_table(config, spec, rows, source="dataset/features_preprocessed/*/{s2_valid_mask.tif,feature_valid_mask.tif}")


def _table_demnas(config):
    spec = _spec("Tabel 4.3")
    summary = read_csv_row_map(config.dataset_root / "feature_preprocessing_summary.csv")
    rows = []
    for region in REGIONS:
        valid_pct = to_float(summary.get(region, {}).get("feature_valid_pct"))
        feature_dir = config.dataset_root / "features_preprocessed" / region
        mask_path = feature_dir / "feature_valid_mask.tif"
        slope_stats = masked_band_stats(feature_dir / "slope_norm.tif", mask_path)
        hand_stats = masked_band_stats(feature_dir / "hand_norm.tif", mask_path)
        rows.append(
            {
                "wilayah": region.replace("_", " "),
                "mean_slope": fmt_float(slope_stats["mean"]),
                "std_slope": fmt_float(slope_stats["std"]),
                "mean_hand": fmt_float(hand_stats["mean"]),
                "std_hand": fmt_float(hand_stats["std"]),
                "piksel_valid_pct": fmt_float(valid_pct),
            }
        )
    return write_table(
        config,
        spec,
        rows,
        source="dataset/features_preprocessed/*/{slope_norm.tif,hand_norm.tif,feature_valid_mask.tif}",
    )


def _figure_channel_example(config):
    spec = _spec("Gambar 4.2")
    region = config.test_region
    feature_dir = config.dataset_root / "features_preprocessed" / region
    panels = [
        ("Channel VV Sentinel-1", read_raster(feature_dir / "vv_norm.tif"), "gray"),
        ("Channel VH Sentinel-1", read_raster(feature_dir / "vh_norm.tif"), "gray"),
        ("Hue Sentinel-2", read_raster(feature_dir / "hue.tif"), "viridis"),
        ("Saturation Sentinel-2", read_raster(feature_dir / "saturation.tif"), "viridis"),
        ("Value Sentinel-2", read_raster(feature_dir / "value.tif"), "viridis"),
        ("Slope DEMNAS", read_raster(feature_dir / "slope_norm.tif"), "viridis"),
        ("HAND", read_raster(feature_dir / "hand_norm.tif"), "viridis"),
        ("Pseudo-RGB HSV Sentinel-2", read_stack_rgb(feature_dir), None),
    ]
    setup_style()
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.0))
    for idx, (ax, (title, arr, cmap)) in enumerate(zip(axes.ravel(), panels)):
        if arr.ndim == 3:
            ax.imshow(arr)
        else:
            ax.imshow(normalize_image(arr), cmap=cmap)
        # ax.set_title(title, fontsize=8)
        _subfigure_label(ax, idx)
        ax.axis("off")
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
        row_panels = [
            (rgb, None, "HSV/Pseudo-RGB"),
            (s2, "gray", "Mask piksel valid S2"),
            (normalize_image(x[0]), "gray", "Channel VV"),
        ]
        for col_idx, (image, cmap, title) in enumerate(row_panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(image, cmap=cmap, vmin=0 if cmap == "gray" else None, vmax=1 if cmap == "gray" else None)
            # ax.set_title(title, fontsize=8)
            _subfigure_label(ax, row_idx * 3 + col_idx)
            ax.axis("off")
            ax.add_patch(
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=ax.transAxes,
                    fill=False,
                    edgecolor="#374151",
                    linewidth=1.4,
                    clip_on=False,
                )
            )
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
    from bab4.plots import hsv_to_display_rgb

    return hsv_to_display_rgb(x[2:5])


def _subfigure_label(ax, idx: int) -> None:
    ax.text(0.5, -0.015, f"({chr(97 + idx)})", transform=ax.transAxes, ha="center", va="top", fontsize=20)


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
