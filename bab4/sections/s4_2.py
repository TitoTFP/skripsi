from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bab4.artifacts import ALL_ARTIFACTS
from bab4.common import REGIONS, fmt_float, pct, read_csv_row_map, to_float, to_int
from bab4.plots import hsv_to_rgb, normalize_image, savefig, setup_style
from bab4.raster import load_npz, read_raster
from bab4.sections.base import section_result
from bab4.writer import figure_result, write_table, write_text_artifact


def _spec(artifact_id: str):
    return next(spec for spec in ALL_ARTIFACTS if spec.artifact_id == artifact_id)


def generate_4_2(config):
    label_rows = _label_rows(config)
    tile_rows = _tile_rows(config)
    artifacts = [
        write_table(config, _spec("Tabel 4.6"), label_rows, source="dataset/preprocessing_summary.csv"),
        write_table(config, _spec("Tabel 4.7"), _label_percentage_rows(label_rows), source="dataset/preprocessing_summary.csv"),
        write_table(config, _spec("Tabel 4.8"), tile_rows, source="dataset/preprocessing_summary.csv;dataset/tiles/7ch/by_region"),
        _figure_mask_panel(config),
        _figure_flood_distribution(config, label_rows),
        _figure_tile_examples(config),
        _narrative(config),
    ]
    return section_result("4.2", artifacts)


def _label_rows(config) -> list[dict[str, object]]:
    summary = read_csv_row_map(config.dataset_root / "preprocessing_summary.csv")
    rows = []
    for region in REGIONS:
        row = summary[region]
        valid = to_int(row.get("valid_pixels"))
        flood = to_int(row.get("flood_pixels"))
        water = to_int(row.get("water_river_pixels"))
        rows.append(
            {
                "region": region,
                "tile_count": to_int(row.get("tile_count")),
                "valid_pixels": valid,
                "flood_pixels": flood,
                "non_flood_pixels": max(valid - flood, 0),
                "water_river_pixels": water,
                "flood_pct_of_valid": fmt_float(pct(flood, valid)),
                "water_river_pct_of_valid": fmt_float(pct(water, valid)),
                "s2_valid_pct_of_valid": fmt_float(pct(to_float(row.get("s2_valid_pixels")), valid)),
                "status": "ok",
            }
        )
    return rows


def _label_percentage_rows(label_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in label_rows:
        rows.append(
            {
                "region": row["region"],
                "valid_pixels": row["valid_pixels"],
                "flood_pixels": row["flood_pixels"],
                "flood_pct_of_valid": row["flood_pct_of_valid"],
                "non_flood_pixels": row["non_flood_pixels"],
                "non_flood_pct_of_valid": fmt_float(100.0 - float(row["flood_pct_of_valid"])),
                "water_river_pixels": row["water_river_pixels"],
                "water_river_pct_of_valid": row["water_river_pct_of_valid"],
                "interpretation": _imbalance_label(float(row["flood_pct_of_valid"])),
            }
        )
    return rows


def _tile_rows(config) -> list[dict[str, object]]:
    summary = read_csv_row_map(config.dataset_root / "preprocessing_summary.csv")
    rows = []
    for region in REGIONS:
        row = summary[region]
        total = to_int(row.get("tile_count"))
        positive = to_int(row.get("positive_tile_count"))
        background = to_int(row.get("background_tile_count"))
        rows.append(
            {
                "region": region,
                "split": row.get("split", "cv"),
                "total_tile": total,
                "tile_positive": positive,
                "tile_background": background,
                "positive_tile_pct": fmt_float(pct(positive, total)),
                "background_tile_pct": fmt_float(pct(background, total)),
                "positive_to_background_ratio": fmt_float(positive / background if background else positive),
            }
        )
    return rows


def _imbalance_label(flood_pct: float) -> str:
    if flood_pct >= 15:
        return "banjir relatif dominan"
    if flood_pct >= 5:
        return "banjir moderat"
    return "banjir minoritas kuat"


def _figure_mask_panel(config):
    spec = _spec("Gambar 4.4")
    region = config.test_region
    label_dir = config.dataset_root / "labels_unosat_rasterized" / region
    panels = [
        ("label_flood_binary", read_raster(label_dir / "label_flood_binary.tif")),
        ("label_valid_mask", read_raster(label_dir / "label_valid_mask.tif")),
        ("label_water_river_mask", read_raster(label_dir / "label_water_river_mask.tif")),
    ]
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (title, arr) in zip(axes, panels):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source=f"dataset/labels_unosat_rasterized/{region}/")


def _figure_flood_distribution(config, rows):
    spec = _spec("Gambar 4.5")
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar([row["region"] for row in rows], [float(row["flood_pct_of_valid"]) for row in rows], color="#2563eb")
    ax.set_ylabel("Flood pixels / valid mask (%)")
    ax.set_title("Class imbalance label banjir per wilayah")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source="dataset/preprocessing_summary.csv")


def _figure_tile_examples(config):
    spec = _spec("Gambar 4.6")
    positive = _find_tile(config.dataset_root / "tiles" / "7ch" / "by_region" / config.test_region, want_positive=True)
    background = _find_tile(config.dataset_root / "tiles" / "7ch" / "by_region" / config.test_region, want_positive=False)
    setup_style()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
    for row_idx, (label, tile_path) in enumerate((("positive", positive), ("background-only", background))):
        tile = load_npz(tile_path)
        x = np.asarray(tile["x"])
        panels = [
            ("VV", normalize_image(x[0])),
            ("HSV pseudo-RGB", hsv_to_rgb(x[2:5])),
            ("HAND", normalize_image(x[6])),
            ("Label", np.asarray(tile["y"])[0]),
        ]
        for ax, (title, arr) in zip(axes[row_idx], panels):
            ax.imshow(arr, cmap=None if arr.ndim == 3 else "gray")
            ax.set_title(f"{label}: {title}")
            ax.axis("off")
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source="dataset/tiles/7ch/by_region/Aceh_Utara/*.npz")


def _find_tile(tile_dir: Path, *, want_positive: bool) -> Path:
    best = None
    best_score = -1.0
    for path in sorted(tile_dir.glob("*.npz")):
        tile = load_npz(path)
        valid = np.asarray(tile["valid_mask"]).astype(bool)
        flood = np.asarray(tile["y"]).astype(bool) & valid
        positives = int(flood.sum())
        if want_positive and positives <= 0:
            continue
        if not want_positive and positives > 0:
            continue
        score = float(valid.sum()) + positives
        if score > best_score:
            best, best_score = path, score
    if best is None:
        return next(tile_dir.glob("*.npz"))
    return best


def _narrative(config):
    spec = _spec("Narasi 4.2")
    text = """
    Label BAB 4 dibuat ulang dari ringkasan tile dan raster label UNOSAT. `label_flood_binary`
    menjadi target utama, `label_valid_mask` membatasi area loss/evaluasi, sedangkan
    `label_water_river_mask` tetap menjadi auxiliary mask. Distribusi tile positif dan background
    menunjukkan class imbalance sehingga IoU/Dice lebih informatif daripada akurasi tunggal.
    """
    return write_text_artifact(config, spec, text, source="dataset/preprocessing_summary.csv;dataset/labels_unosat_rasterized;dataset/tiles")
