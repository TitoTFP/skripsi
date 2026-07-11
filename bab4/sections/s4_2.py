from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bab4.artifacts import ALL_ARTIFACTS
from bab4.common import REGIONS, fmt_float, pct, to_float, to_int
from bab4.plots import hsv_to_display_rgb, normalize_image, savefig, setup_style
from bab4.raster import load_npz, read_raster
from bab4.sections.base import section_result
from bab4.writer import figure_result, write_table, write_text_artifact


def _spec(artifact_id: str):
    return next(spec for spec in ALL_ARTIFACTS if spec.artifact_id == artifact_id)


def generate_4_2(config):
    tile_stats = _tile_stats(config)
    label_rows = _label_rows(tile_stats)
    tile_rows = _tile_rows(tile_stats)
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


def _tile_stats(config) -> dict[str, dict[str, object]]:
    stats = {}
    tile_root = config.dataset_root / "tiles" / "7ch" / "by_region"
    for region in REGIONS:
        total_tile = 0
        valid_tile = 0
        positive_tile = 0
        valid_pixels = 0
        flood_pixels = 0
        water_river_pixels = 0
        s2_valid_pixels = 0
        for path in sorted((tile_root / region).glob("*.npz")):
            total_tile += 1
            tile = load_npz(path)
            valid = np.squeeze(tile["valid_mask"]).astype(bool)
            flood = np.squeeze(tile["y"]).astype(bool) & valid
            water = np.squeeze(tile.get("water_river_mask", np.zeros_like(valid))).astype(bool) & valid
            s2 = np.squeeze(tile.get("s2_valid_mask", np.zeros_like(valid))).astype(bool) & valid
            valid_count = int(valid.sum())
            flood_count = int(flood.sum())
            valid_pixels += valid_count
            flood_pixels += flood_count
            water_river_pixels += int(water.sum())
            s2_valid_pixels += int(s2.sum())
            if valid_count > 0:
                valid_tile += 1
                if flood_count > 0:
                    positive_tile += 1
        stats[region] = {
            "split": "test" if region == config.test_region else "cv",
            "total_tile": total_tile,
            "valid_tile": valid_tile,
            "positive_tile": positive_tile,
            "background_tile": max(valid_tile - positive_tile, 0),
            "valid_pixels": valid_pixels,
            "flood_pixels": flood_pixels,
            "non_flood_pixels": max(valid_pixels - flood_pixels, 0),
            "water_river_pixels": water_river_pixels,
            "s2_valid_pixels": s2_valid_pixels,
        }
    return stats


def _label_rows(tile_stats: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for region in REGIONS:
        row = tile_stats[region]
        valid = to_int(row.get("valid_pixels"))
        flood = to_int(row.get("flood_pixels"))
        water = to_int(row.get("water_river_pixels"))
        rows.append(
            {
                "wilayah": region.replace("_", " "),
                "piksel_valid": valid,
                "piksel_banjir": flood,
                "piksel_non_banjir": max(valid - flood, 0),
                "water_river_pixels": water,
            }
        )
    return rows


def _label_percentage_rows(label_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in label_rows:
        rows.append(
            {
                "wilayah": row["wilayah"],
                "banjir_dalam_valid_pct": fmt_float(pct(to_float(row["piksel_banjir"]), to_float(row["piksel_valid"])), 1),
                "water_river_dalam_valid_pct": fmt_float(pct(to_float(row["water_river_pixels"]), to_float(row["piksel_valid"])), 1),
                "rasio_non_banjir": fmt_float(to_float(row["piksel_non_banjir"]) / max(to_float(row["piksel_banjir"]), 1), 1),
            }
        )
    return rows


def _tile_rows(tile_stats: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for region in REGIONS:
        row = tile_stats[region]
        total = to_int(row.get("total_tile"))
        valid = to_int(row.get("valid_tile"))
        positive = to_int(row.get("positive_tile"))
        background = to_int(row.get("background_tile"))
        rows.append(
            {
                "wilayah": region.replace("_", " "),
                "total_tile": total,
                "tile_valid": valid,
                "tile_positive": positive,
                "tile_background": background,
                "tile_positive_pct": fmt_float(pct(positive, total), 2),
                "tile_background_pct": fmt_float(pct(background, total), 2),
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
    flood = read_raster(label_dir / "label_flood_binary.tif")
    valid = read_raster(label_dir / "label_valid_mask.tif")
    water = read_raster(label_dir / "label_water_river_mask.tif")
    panels = [
        ("label_flood_binary", flood, "Blues"),
        ("label_valid_mask", valid, "Greens"),
        ("label_water_river_mask", water, "Blues"),
        ("area flood within valid", flood.astype(bool) & valid.astype(bool), "Reds"),
    ]
    setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.4))
    for idx, (ax, (title, arr, cmap)) in enumerate(zip(axes, panels)):
        ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
        ax.text(0.5, -0.08, f"({chr(97 + idx)})", transform=ax.transAxes, ha="center", va="top", fontsize=10)
        ax.axis("off")
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source=f"dataset/labels_unosat_rasterized/{region}/")


def _figure_flood_distribution(config, rows):
    spec = _spec("Gambar 4.5")
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(
        [row["wilayah"] for row in rows],
        [pct(to_float(row["piksel_banjir"]), to_float(row["piksel_valid"])) for row in rows],
        color="#d9822b",
    )
    ax.set_ylabel("Flood pixels dalam valid mask (%)")
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
            ("VV", normalize_image(x[0]), "gray"),
            ("HSV pseudo-RGB", hsv_to_display_rgb(x[2:5]), None),
            ("HAND", normalize_image(x[6]), "viridis"),
            ("Label", np.asarray(tile["y"])[0], "gray"),
        ]
        for col_idx, (ax, (title, arr, cmap)) in enumerate(zip(axes[row_idx], panels)):
            ax.imshow(arr, cmap=cmap, vmin=0 if cmap == "gray" else None, vmax=1 if cmap == "gray" else None)
            ax.set_title(f"{label}: {title}", fontsize=8)
            ax.text(0.5, -0.08, f"({chr(97 + row_idx * 4 + col_idx)})", transform=ax.transAxes, ha="center", va="top", fontsize=9)
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
