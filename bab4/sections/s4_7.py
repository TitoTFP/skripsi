from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from matplotlib.patches import Patch

from bab4.artifacts import ALL_ARTIFACTS, ArtifactSpec
from bab4.common import CV_REGIONS, MODEL_KEYS, MODEL_LABELS, fmt_float, pct, read_csv_rows, to_float, to_int
from bab4.plots import hsv_to_display_rgb, normalize_image, savefig, setup_style
from bab4.raster import masked_band_stats
from bab4.sections.base import section_result
from bab4.sections.s4_2 import _tile_stats
from bab4.writer import figure_result, missing_result, write_table, write_text_artifact


SUPPLEMENTARY = (
    ArtifactSpec(
        "Tambahan 4.7",
        "table",
        "4.7",
        "Rincian metrik out-of-fold per wilayah",
        "4_7_oof_extreme_condition_metrics_by_region.csv",
        priority="supplementary",
    ),
)

OOF_SOURCE = "runs/oof_extreme_conditions/{micro_metrics.csv,per_region_metrics.csv,selected_tiles.csv,provenance.json}"


def _spec(artifact_id: str) -> ArtifactSpec:
    for spec in ALL_ARTIFACTS + SUPPLEMENTARY:
        if spec.artifact_id == artifact_id:
            return spec
    raise KeyError(artifact_id)


def generate_4_7(config):
    _remove_obsolete_outputs(config)
    difficult_rows = _difficult_data_rows(config)
    artifacts = [
        write_table(
            config,
            _spec("Tabel 4.16"),
            difficult_rows,
            source="dataset/tiles/7ch/by_region/*/*.npz;dataset/features_preprocessed/*",
        ),
        _figure_difficult_cases(config, difficult_rows),
    ]
    source = config.runs_root / "oof_extreme_conditions"
    required = [source / name for name in ("micro_metrics.csv", "per_region_metrics.csv", "selected_tiles.csv", "provenance.json")]
    if not all(path.exists() for path in required):
        note = "evaluasi OOF belum lengkap; jalankan scripts/evaluate_oof_extreme_conditions.py terlebih dahulu"
        artifacts.extend(
            [
                missing_result(config, _spec("Tabel 4.17"), source=OOF_SOURCE, note=note),
                missing_result(config, _spec("Tabel 4.18"), source=OOF_SOURCE, note=note),
                missing_result(config, _spec("Tambahan 4.7"), source=OOF_SOURCE, note=note),
                *[missing_result(config, _spec(artifact_id), source=OOF_SOURCE, note=note) for artifact_id in ("Gambar 4.16", "Gambar 4.17", "Gambar 4.18")],
                _narrative_4_7(config, [], available=False),
            ]
        )
        return section_result("4.7", artifacts)

    micro_rows = read_csv_rows(source / "micro_metrics.csv")
    per_region_rows = read_csv_rows(source / "per_region_metrics.csv")
    selected_tiles = read_csv_rows(source / "selected_tiles.csv")
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    artifacts.extend(
        [
            write_table(config, _spec("Tabel 4.17"), _selected_tile_rows(selected_tiles), source=OOF_SOURCE),
            write_table(config, _spec("Tabel 4.18"), _micro_metric_rows(micro_rows), source=OOF_SOURCE),
            write_table(config, _spec("Tambahan 4.7"), _per_region_metric_rows(per_region_rows), source=OOF_SOURCE),
            *[
                _figure_extreme_oof(config, artifact_id, selected_tiles, provenance)
                for artifact_id in ("Gambar 4.16", "Gambar 4.17", "Gambar 4.18")
            ],
            _narrative_4_7(config, _micro_metric_rows(micro_rows), available=True),
        ]
    )
    return section_result("4.7", artifacts)


def _remove_obsolete_outputs(config) -> None:
    obsolete = (
        "4_7_unet_vs_procanet_effectiveness_summary.csv",
        "4_7_literature_context_comparison.csv",
        "4_9_bab4_findings_summary.csv",
        "4_7_unet_procanet_effectiveness_discussion.md",
        "4_8_difficult_data_case_studies.csv",
        "4_8_extreme_tile_selection.csv",
        "4_8_difficult_data_case_studies.png",
        "4_8_hsv_zero_tile_panel.png",
        "4_8_topography_radar_shadow_case.png",
        "4_8_permanent_water_case.png",
        "4_8_data_extreme_limitations_interpretation.md",
    )
    for directory in (config.tables_dir, config.figures_dir, config.narratives_dir):
        for filename in obsolete:
            path = directory / filename
            if path.exists():
                path.unlink()


def _difficult_data_rows(config) -> list[dict[str, object]]:
    tile_stats = _tile_stats(config)
    rows = []
    for region in CV_REGIONS:
        summary = tile_stats[region]
        feature_dir = config.dataset_root / "features_preprocessed" / region
        effective_path = feature_dir / "feature_valid_mask.tif"
        slope = masked_band_stats(feature_dir / "slope_degrees.tif", effective_path)
        hand = masked_band_stats(feature_dir / "hand_meters.tif", effective_path)
        valid = to_int(summary.get("valid_pixels"))
        rows.append(
            {
                "wilayah": region.replace("_", " "),
                "piksel_valid_tile": valid,
                "flood_pct_of_valid": fmt_float(pct(to_int(summary.get("flood_pixels")), valid)),
                "water_river_pct_of_valid": fmt_float(pct(to_int(summary.get("water_river_pixels")), valid)),
                "s2_valid_pct_of_valid": fmt_float(pct(to_int(summary.get("s2_valid_pixels")), valid)),
                "mean_slope_degrees": fmt_float(slope["mean"]),
                "mean_hand_meters": fmt_float(hand["mean"]),
            }
        )
    return sorted(rows, key=lambda row: (float(row["s2_valid_pct_of_valid"]), str(row["wilayah"])))


def _selected_tile_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        {
            "kondisi": row["condition_label"],
            "wilayah": row["region"].replace("_", " "),
            "fold_oof": to_int(row["fold"]),
            "tile": row["tile"],
            "piksel_kondisi": to_int(row["condition_pixels"]),
            "piksel_efektif": to_int(row["effective_pixels"]),
            "piksel_banjir": to_int(row["flood_pixels"]),
        }
        for row in rows
    ]


def _metric_value(value: object) -> object:
    if value in (None, "", "None"):
        return "–"
    return fmt_float(to_float(value), 6)


def _micro_metric_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        {
            "kondisi": row["condition_label"],
            "model": row["model"],
            "piksel_dievaluasi": to_int(row["evaluated_pixels"]),
            "tp": to_int(row["tp"]),
            "tn": to_int(row["tn"]),
            "fp": to_int(row["fp"]),
            "fn": to_int(row["fn"]),
            "iou": _metric_value(row.get("iou")),
            "dice": _metric_value(row.get("dice")),
            "precision": _metric_value(row.get("precision")),
            "recall": _metric_value(row.get("recall")),
            "specificity": _metric_value(row.get("specificity")),
            "fpr": _metric_value(row.get("fpr")),
        }
        for row in rows
    ]


def _per_region_metric_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        {
            "kondisi": row["condition_label"],
            "model": row["model"],
            "fold_oof": to_int(row["fold"]),
            "wilayah": row["region"].replace("_", " "),
            "konfigurasi": row["variant"],
            "piksel_kondisi": to_int(row["condition_pixels"]),
            "piksel_dievaluasi": to_int(row["evaluated_pixels"]),
            "tp": to_int(row["tp"]),
            "tn": to_int(row["tn"]),
            "fp": to_int(row["fp"]),
            "fn": to_int(row["fn"]),
            "iou": _metric_value(row.get("iou")),
            "dice": _metric_value(row.get("dice")),
            "precision": _metric_value(row.get("precision")),
            "recall": _metric_value(row.get("recall")),
            "specificity": _metric_value(row.get("specificity")),
            "fpr": _metric_value(row.get("fpr")),
        }
        for row in rows
    ]


def _figure_difficult_cases(config, rows: list[dict[str, object]]):
    spec = _spec("Gambar 4.15")
    setup_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    s2 = np.asarray([float(row["s2_valid_pct_of_valid"]) for row in rows])
    flood = np.asarray([float(row["flood_pct_of_valid"]) for row in rows])
    water = np.asarray([float(row["water_river_pct_of_valid"]) for row in rows])
    colors = np.where(s2 <= 1.0, "#6A3D9A", "#E66101")
    ax.scatter(s2, flood, s=35 + water * 12, c=colors, alpha=0.86, edgecolor="white", linewidth=0.8)
    for row, x_value, y_value in zip(rows, s2, flood):
        ax.text(x_value + 1.0, y_value + 0.15, str(row["wilayah"]).replace(" ", "_"), fontsize=7)
    ax.axvline(1.0, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Piksel Sentinel-2 valid pada tile valid (%)")
    ax.set_ylabel("Piksel label banjir pada tile valid (%)")
    ax.legend(
        handles=[
            Patch(facecolor="#6A3D9A", label="Sentinel-2 valid ≤ 1%"),
            Patch(facecolor="#E66101", label="Sentinel-2 valid > 1%"),
        ],
        title="Kondisi Sentinel-2",
        loc="lower right",
        fontsize=8,
        title_fontsize=8,
    )
    ax.set_xlim(-3, 103)
    ax.grid(alpha=0.25)
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source="dataset/preprocessing_summary.csv;dataset/tiles/7ch/by_region")


def _figure_extreme_oof(config, artifact_id: str, selected_tiles: list[dict[str, str]], provenance: dict[str, object]):
    condition = {
        "Gambar 4.16": "s2_empty",
        "Gambar 4.17": "topography_radar_shadow",
        "Gambar 4.18": "permanent_water",
    }[artifact_id]
    selected = next((row for row in selected_tiles if row.get("condition") == condition), None)
    spec = _spec(artifact_id)
    if selected is None:
        return missing_result(config, spec, source=OOF_SOURCE, note=f"contoh tile {condition} tidak tersedia")
    path = Path(selected["tile_path"])
    if not path.exists():
        return missing_result(config, spec, source=OOF_SOURCE, note=f"tile sumber tidak ditemukan: {path}")
    with np.load(path, allow_pickle=False) as tile:
        x = np.asarray(tile["x"], dtype=np.float32)
        label = np.squeeze(tile["y"]).astype(bool)
        effective = np.squeeze(tile["valid_mask"]).astype(bool) & np.squeeze(tile["feature_valid_mask"]).astype(bool)
        water = np.squeeze(tile["water_river_mask"]).astype(bool)
        s2 = np.squeeze(tile["s2_valid_mask"]).astype(bool)
        row, col = int(tile["row"].item()), int(tile["col"].item())
    thresholds = provenance["radar_percentiles"]
    condition_mask = _tile_condition(condition, x, effective, label, water, s2, thresholds)
    predictions = {
        model: _prediction_crop(config.runs_root / "oof_extreme_conditions" / model / f"fold_{selected['fold']}" / "geotiff" / f"{selected['region']}_prediction.tif", row, col, label.shape)
        for model in MODEL_KEYS
    }
    panels = [
        ("Sentinel-1 VV", normalize_image(x[0]), "gray"),
        ("Sentinel-1 VH", normalize_image(x[1]), "gray"),
        ("Pseudo-RGB HSV", hsv_to_display_rgb(x[2:5]), None),
        ("Slope", normalize_image(x[5]), "magma"),
        ("HAND", normalize_image(x[6]), "viridis"),
        ("Mask kondisi", condition_mask, "Reds"),
        ("Label UNOSAT", label, "Blues"),
        ("Prediksi OOF U-Net", predictions["unet"], "Oranges"),
        ("Prediksi OOF ProCANet", predictions["procanet"], "Purples"),
    ]
    setup_style()
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 8.3))
    for index, (ax, (_, image, cmap)) in enumerate(zip(axes.ravel(), panels)):
        binary = cmap in {"gray", "Reds", "Blues", "Oranges", "Purples"}
        ax.imshow(image, cmap=cmap, vmin=0 if binary else None, vmax=1 if binary else None)
        ax.text(
            0.04,
            0.05,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=20,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
        ax.axis("off")
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.985, hspace=0.025, wspace=0.025)
    out_path = config.figures_dir / spec.filename
    savefig(fig, out_path)
    return figure_result(config, spec, out_path, source=f"{OOF_SOURCE};{path}")


def _tile_condition(condition: str, x: np.ndarray, effective: np.ndarray, label: np.ndarray, water: np.ndarray, s2: np.ndarray, thresholds: dict[str, object]) -> np.ndarray:
    if condition == "s2_empty":
        return effective if effective.any() and float(s2[effective].mean()) <= 0.01 else np.zeros_like(effective)
    if condition == "topography_radar_shadow":
        return effective & (x[5] * 45.0 > 20.0) & (x[6] * 50.0 > 40.0) & ((x[0] <= float(thresholds["p20_vv_norm"])) | (x[1] <= float(thresholds["p20_vh_norm"])))
    return effective & water & ~label


def _prediction_crop(path: Path, row: int, col: int, shape: tuple[int, int]) -> np.ndarray:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(path)
    height, width = shape
    array = np.asarray(ds.GetRasterBand(1).ReadAsArray(col, row, width, height))
    ds = None
    return (array == 1).astype(np.uint8)


def _narrative_4_7(config, rows: list[dict[str, object]], *, available: bool):
    spec = _spec("Narasi 4.7")
    if not available:
        text = """
        Evaluasi kuantitatif kondisi ekstrem belum dapat disajikan karena artefak inferensi out-of-fold belum lengkap.
        Generator sengaja tidak menggunakan prediksi Aceh Utara sebagai pengganti, karena tujuan subbab ini adalah
        membandingkan perilaku model pada wilayah validasi yang tidak digunakan untuk melatih checkpoint fold terkait.
        """
    else:
        water_rows = [row for row in rows if row["kondisi"] == "Badan air permanen/sungai"]
        text = f"""
        Tabel 4.18 menghitung performa dengan prediksi out-of-fold pada sepuluh wilayah cross-validation.
        Setiap wilayah diprediksi hanya oleh checkpoint terbaik fold saat wilayah tersebut menjadi data validasi;
        probabilitas tile yang overlap dirata-ratakan sebelum threshold 0,5 diterapkan. Karena itu, hasil ini bukan
        evaluasi independen setara Aceh Utara pada Subbab 4.5, melainkan bukti pembanding untuk kondisi data sulit.
        Metrik pada badan air permanen/sungai ditampilkan terutama melalui FP, specificity, dan FPR; IoU, Dice,
        precision, serta recall ditandai `–` bila subset tidak memuat label banjir. Rincian per wilayah tersedia
        pada artefak tambahan untuk memeriksa dominasi wilayah tertentu pada agregasi mikro.
        """
        if not water_rows:
            text += "\nTidak ada baris badan air permanen yang dapat diagregasikan."
    return write_text_artifact(config, spec, text, source=OOF_SOURCE)
