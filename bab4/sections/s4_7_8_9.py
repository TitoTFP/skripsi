from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from bab4.artifacts import ALL_ARTIFACTS, ArtifactSpec
from bab4.common import MODEL_KEYS, MODEL_LABELS, REGIONS, fmt_float, pct, read_csv_row_map, read_csv_rows, to_float, to_int
from bab4.plots import hsv_to_display_rgb, normalize_image, savefig, setup_style
from bab4.raster import masked_band_stats
from bab4.sections.s4_2 import _tile_stats
from bab4.sections.base import section_result
from bab4.writer import figure_result, missing_result, write_table, write_text_artifact


SUPPLEMENTARY = (
    ArtifactSpec("Tambahan 4.7", "table", "4.7", "Ringkasan perbandingan U-Net vs ProCANet", "4_7_unet_vs_procanet_effectiveness_summary.csv", priority="supplementary"),
    ArtifactSpec("Tambahan 4.7b", "table", "4.7", "Konteks perbandingan literatur", "4_7_literature_context_comparison.csv", priority="supplementary"),
    ArtifactSpec("Tambahan 4.9", "table", "4.9", "Ringkasan temuan utama BAB 4", "4_9_bab4_findings_summary.csv", priority="supplementary"),
)


def _spec(artifact_id: str):
    for spec in ALL_ARTIFACTS + SUPPLEMENTARY:
        if spec.artifact_id == artifact_id:
            return spec
    raise KeyError(artifact_id)


def generate_4_7_8_9(config):
    artifacts = []
    for result in (generate_4_7(config), generate_4_8(config), generate_4_9(config)):
        artifacts.extend(result.artifacts)
    return section_result("4.7-4.9", artifacts)


def generate_4_7(config):
    effectiveness_rows = _effectiveness_rows(config)
    metrics_source = f"{config.evaluation_source}/metrics.csv"
    artifacts = [
        write_table(
            config,
            _spec("Tambahan 4.7"),
            effectiveness_rows,
            source=f"{metrics_source};training/models/{{unet,procanet}}.py",
        ),
        write_table(
            config,
            _spec("Tambahan 4.7b"),
            _literature_context_rows(effectiveness_rows),
            source=f"{metrics_source};training/models/{{unet,procanet}}.py",
        ),
        _narrative_4_7(config, effectiveness_rows),
    ]
    return section_result("4.7", artifacts)


def generate_4_8(config):
    difficult_rows = _difficult_data_rows(config)
    extreme_rows, selected_tiles = _extreme_tile_rows(config)
    artifacts = [
        write_table(config, _spec("Tabel 4.16"), difficult_rows, source="dataset/tiles/7ch/by_region/*/*.npz;dataset/feature_preprocessing_summary.csv"),
        write_table(config, _spec("Tabel 4.17"), extreme_rows, source="dataset/tiles/7ch/by_region/*/*.npz"),
        _figure_difficult_cases(config, difficult_rows),
        _figure_extreme_tile(config, "Gambar 4.16", selected_tiles.get("hsv_zero"), "Kasus Sentinel-2 kosong/hampir kosong"),
        _figure_extreme_tile(config, "Gambar 4.17", selected_tiles.get("topography"), "Kasus topografi sulit atau kandidat radar shadow"),
        _figure_extreme_tile(config, "Gambar 4.18", selected_tiles.get("permanent_water"), "Kasus badan air permanen"),
        _narrative_4_8(config, extreme_rows),
    ]
    return section_result("4.8", artifacts)


def generate_4_9(config):
    rows = _findings_rows(config)
    artifacts = [
        write_table(
            config,
            _spec("Tambahan 4.9"),
            rows,
            source=f"dataset/preprocessing_summary.csv;{config.evaluation_source}/metrics.csv;bab4 generator functions",
        ),
    ]
    return section_result("4.9", artifacts)


def _effectiveness_rows(config) -> list[dict[str, object]]:
    metrics = _final_metric_rows(config)
    rows = []
    for row in metrics:
        model = str(row["model"])
        fp = int(row["fp"])
        fn = int(row["fn"])
        rows.append(
            {
                "model": model,
                "architecture_character": "baseline encoder-decoder" if model == "U-Net" else "dual encoder with progressive cross-attention",
                "iou": row["iou"],
                "dice": row["dice"],
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "false_positive_pixels": fp,
                "false_negative_pixels": fn,
                "dominant_error": "false positive" if fp > fn else "false negative",
                "interpretation": _effectiveness_interpretation(model, float(row["precision"]), float(row["recall"])),
            }
        )
    if len(rows) == 2:
        delta = float(rows[1]["iou"]) - float(rows[0]["iou"])
        for row in rows:
            row["iou_delta_procanet_minus_unet"] = fmt_float(delta, 6)
    return rows


def _literature_context_rows(effectiveness_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unet = next((row for row in effectiveness_rows if row["model"] == "U-Net"), {})
    procanet = next((row for row in effectiveness_rows if row["model"] == "ProCANet"), {})
    return [
        {
            "comparison_axis": "baseline segmentation",
            "reference_context": "U-Net encoder-decoder dengan skip connection",
            "bab4_evidence": f"IoU={unet.get('iou', '')}, Dice={unet.get('dice', '')}",
            "interpretation": "baseline kuat untuk banjir yang kontras pada radar dan topografi",
        },
        {
            "comparison_axis": "attention fusion",
            "reference_context": "ProCANet dual encoder dengan progressive cross-attention",
            "bab4_evidence": f"IoU={procanet.get('iou', '')}, Dice={procanet.get('dice', '')}",
            "interpretation": "attention fusion menekan FP dan meningkatkan metrik agregat; recall U-Net hanya sedikit lebih tinggi",
        },
        {
            "comparison_axis": "operational trade-off",
            "reference_context": "validasi wilayah held-out Aceh_Utara",
            "bab4_evidence": "precision/recall dihitung dari confusion matrix checkpoint terbaik spatial CV",
            "interpretation": "pilihan model perlu mempertimbangkan biaya FP dan FN, bukan hanya akurasi",
        },
    ]


def _difficult_data_rows(config) -> list[dict[str, object]]:
    tile_stats = _tile_stats(config)
    rows = []
    for region in REGIONS:
        row = tile_stats[region]
        valid = to_int(row.get("valid_pixels"))
        flood = to_int(row.get("flood_pixels"))
        water = to_int(row.get("water_river_pixels"))
        s2_valid = to_int(row.get("s2_valid_pixels"))
        s2_pct = pct(s2_valid, valid)
        feature_dir = config.dataset_root / "features_preprocessed" / region
        mask_path = feature_dir / "feature_valid_mask.tif"
        slope_stats = masked_band_stats(feature_dir / "slope_norm.tif", mask_path)
        hand_stats = masked_band_stats(feature_dir / "hand_norm.tif", mask_path)
        rows.append(
            {
                "wilayah": region.replace("_", " "),
                "piksel_valid": valid,
                "flood_pct_of_valid": fmt_float(pct(flood, valid)),
                "water_river_pct_of_valid": fmt_float(pct(water, valid)),
                "s2_valid_pct_of_valid": fmt_float(s2_pct),
                "mean_slope": fmt_float(slope_stats["mean"]),
                "mean_hand": fmt_float(hand_stats["mean"]),
            }
        )
    order = ["Langsa", "Agam", "Aceh_Tamiang", "Pasaman_Barat", "Aceh_Utara", "Pidie_Jaya", "Pidie"]
    order_labels = [region.replace("_", " ") for region in order]
    by_region = {row["wilayah"]: row for row in rows}
    return [by_region[label] for label in order_labels]


def _extreme_tile_rows(config) -> tuple[list[dict[str, object]], dict[str, Path]]:
    selected = {
        "hsv_zero": _choose_existing_or_scan(
            config.dataset_root / "tiles" / "7ch" / "by_region" / "Aceh_Tamiang",
            "Aceh_Tamiang_r001024_c003840.npz",
            key=lambda payload: -int(np.squeeze(payload.get("s2_valid_mask", np.zeros_like(payload["y"]))).sum()),
        ),
        "topography": _choose_existing_or_scan(
            config.dataset_root / "tiles" / "7ch" / "by_region" / "Aceh_Tamiang",
            "Aceh_Tamiang_r006501_c002304.npz",
            key=lambda payload: float(np.nanmean(np.asarray(payload["x"], dtype=np.float32)[5])),
        ),
        "permanent_water": _choose_existing_or_scan(
            config.dataset_root / "tiles" / "7ch" / "by_region" / config.test_region,
            "",
            key=lambda payload: int(np.squeeze(payload.get("water_river_mask", np.zeros_like(payload["y"]))).sum()),
        ),
    }
    labels = {
        "hsv_zero": ("Sentinel-2 kosong/hampir kosong", "prioritaskan tile Aceh_Tamiang dengan s2_valid_mask minimum"),
        "topography": ("topografi/radar shadow", "prioritaskan tile Aceh_Tamiang ekstrem topografi"),
        "permanent_water": ("badan air permanen", f"pilih tile {config.test_region} dengan water_river_mask maksimum"),
    }
    rows = []
    for case_id, path in selected.items():
        if path is None:
            continue
        payload = _load_npz(path)
        rows.append(_tile_summary_row(path, payload, case_id, labels[case_id][0], labels[case_id][1]))
    return rows, {case_id: path for case_id, path in selected.items() if path is not None}


def _tile_summary_row(path: Path, payload: dict[str, np.ndarray], case_id: str, case_label: str, selection_rule: str) -> dict[str, object]:
    x = np.asarray(payload["x"], dtype=np.float32)
    valid = _valid_mask(payload)
    flood = np.squeeze(payload["y"]).astype(bool) & valid
    water = np.squeeze(payload.get("water_river_mask", np.zeros_like(flood))).astype(bool) & valid
    s2 = np.squeeze(payload.get("s2_valid_mask", np.zeros_like(flood))).astype(bool) & valid
    valid_count = int(valid.sum())
    return {
        "kasus": case_label,
        "tile": path.stem,
        "piksel_valid": valid_count,
        "flood_pct_of_valid": fmt_float(pct(int(flood.sum()), valid_count)),
        "s2_valid_pct_of_valid": fmt_float(pct(int(s2.sum()), valid_count)),
        "water_river_pct_of_valid": fmt_float(pct(int(water.sum()), valid_count)),
        "dark_vv_pct_of_valid": fmt_float(_dark_vv_pct(x[0], valid)),
        "slope_mean_valid": fmt_float(_masked_mean(x[5], valid)),
        "hand_mean_valid": fmt_float(_masked_mean(x[6], valid)),
    }


def _dark_vv_pct(vv: np.ndarray, valid: np.ndarray) -> float:
    values = vv[valid]
    if values.size == 0:
        return 0.0
    threshold = float(np.nanpercentile(values, 20))
    return pct(int(np.count_nonzero(values <= threshold)), int(values.size))


def _figure_difficult_cases(config, rows: list[dict[str, object]]):
    spec = _spec("Gambar 4.15")
    labels = [str(row["wilayah"]) for row in rows]
    flood = np.asarray([float(row["flood_pct_of_valid"]) for row in rows])
    water = np.asarray([float(row["water_river_pct_of_valid"]) for row in rows])
    s2 = np.asarray([float(row["s2_valid_pct_of_valid"]) for row in rows])
    recommended = np.asarray([(value < 1.0) or (flood[idx] > 10.0) for idx, value in enumerate(s2)])
    setup_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    colors = np.where(recommended, "#e45756", "#4c78a8")
    sizes = 35 + water * 12
    ax.scatter(s2, flood, s=sizes, c=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
    for label, x_val, y_val in zip(labels, s2, flood):
        ax.text(x_val + 1.0, y_val + 0.15, label.replace(" ", "_"), fontsize=7)
    ax.set_xlabel("Valid Sentinel-2 dalam valid mask (%)")
    ax.set_ylabel("Piksel label dalam valid mask (%)")
    ax.set_title("Kondisi Data Sulit per Wilayah", fontsize=9)
    yes = Patch(facecolor="#e45756", label="yes")
    optional = Patch(facecolor="#4c78a8", label="optional")
    ax.legend(handles=[yes, optional], title="recommended_use_in_discussion", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    ax.set_xlim(-3, 103)
    ax.grid(alpha=0.25)
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source="dataset/preprocessing_summary.csv;dataset/feature_preprocessing_summary.csv")


def _figure_extreme_tile(config, artifact_id: str, tile_path: Path | None, title: str):
    spec = _spec(artifact_id)
    if tile_path is None:
        return missing_result(config, spec, source="dataset/tiles/7ch/by_region/*/*.npz", note="tile ekstrem tidak ditemukan")
    payload = _load_npz(tile_path)
    x = np.asarray(payload["x"], dtype=np.float32)
    flood = np.squeeze(payload["y"])
    water = np.squeeze(payload.get("water_river_mask", np.zeros_like(flood)))
    s2 = np.squeeze(payload.get("s2_valid_mask", np.zeros_like(flood)))
    panels = _panels_for_extreme_case(config, artifact_id, tile_path, x, flood, water, s2)
    setup_style()
    cols = 4 if artifact_id in {"Gambar 4.16", "Gambar 4.18"} else 3
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(10.5 if cols == 4 else 8.2, 5.4))
    for idx, (ax, (panel_title, image, cmap)) in enumerate(zip(axes.ravel(), panels)):
        binary_cmap = cmap in {"gray", "Reds", "Blues", "Oranges", "Greens"}
        ax.imshow(image, cmap=cmap, vmin=0 if binary_cmap else None, vmax=1 if binary_cmap else None)
        ax.set_title(panel_title, fontsize=8)
        ax.text(0.5, -0.08, f"({chr(97 + idx)})", transform=ax.transAxes, ha="center", va="top", fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{title} - {tile_path.stem}", y=0.98, fontsize=9)
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source=str(tile_path))


def _panels_for_extreme_case(config, artifact_id: str, tile_path: Path, x: np.ndarray, flood: np.ndarray, water: np.ndarray, s2: np.ndarray):
    if artifact_id == "Gambar 4.16":
        return [
            ("Sentinel-1 VV", normalize_image(x[0]), "gray"),
            ("Sentinel-1 VH", normalize_image(x[1]), "gray"),
            ("Pseudo-RGB HSV", hsv_to_display_rgb(x[2:5]), None),
            ("S2 invalid mask", 1 - s2.astype(np.uint8), "Reds"),
            ("HSV=0 pada valid area", np.all(np.isclose(x[2:5], 0.0), axis=0).astype(np.uint8), "Reds"),
            ("Label UNOSAT", flood, "Blues"),
            ("Slope", normalize_image(x[5]), "magma"),
            ("HAND", normalize_image(x[6]), "viridis"),
        ]
    if artifact_id == "Gambar 4.17":
        dark_vv = x[0] <= np.nanpercentile(x[0], 20)
        return [
            ("Sentinel-1 VV", normalize_image(x[0]), "gray"),
            ("Sentinel-1 VH", normalize_image(x[1]), "gray"),
            ("Dark VV candidate", dark_vv.astype(np.uint8), "Reds"),
            ("Slope", normalize_image(x[5]), "magma"),
            ("HAND", normalize_image(x[6]), "viridis"),
            ("Label UNOSAT", flood, "Blues"),
        ]
    unet = _load_prediction_for_tile(config, "unet", tile_path)
    procanet = _load_prediction_for_tile(config, "procanet", tile_path)
    return [
        ("Sentinel-1 VV", normalize_image(x[0]), "gray"),
        ("Sentinel-1 VH", normalize_image(x[1]), "gray"),
        ("Water/river mask", water, "Blues"),
        ("Label UNOSAT", flood, "Blues"),
        ("Slope", normalize_image(x[5]), "magma"),
        ("HAND", normalize_image(x[6]), "viridis"),
        ("Prediksi U-Net", unet, "Oranges"),
        ("Prediksi ProCANet", procanet, "Greens"),
    ]


def _load_prediction_for_tile(config, model: str, tile_path: Path) -> np.ndarray:
    prediction_path = config.evaluation_dir(model) / "predictions" / _region_from_tile(tile_path) / tile_path.name
    if prediction_path.exists():
        return np.squeeze(_load_npz(prediction_path).get("prediction", np.zeros((1, 512, 512), dtype=np.uint8)))
    return np.zeros((512, 512), dtype=np.uint8)


def _findings_rows(config) -> list[dict[str, object]]:
    summary = read_csv_row_map(config.dataset_root / "preprocessing_summary.csv")
    metrics = _final_metric_rows(config)
    best_model = max(metrics, key=lambda row: float(row["iou"])) if metrics else {}
    test = summary.get(config.test_region, {})
    return [
        {
            "finding_id": "F1",
            "topic": "validasi data sumber",
            "evidence": "Tabel 4.1-4.8 dibangkitkan dari raster, CSV preprocessing, dan tile dataset",
            "implication": "output BAB 4 dapat direproduksi tanpa membaca artefak lama",
        },
        {
            "finding_id": "F2",
            "topic": "spatial validation",
            "evidence": "5-fold spatial CV memakai wilayah validasi terpisah dan Aceh_Utara sebagai final test",
            "implication": "evaluasi lebih tahan terhadap spatial leakage",
        },
        {
            "finding_id": "F3",
            "topic": "checkpoint terbaik spatial CV",
            "evidence": f"model terbaik berdasarkan IoU wilayah uji: {best_model.get('model', '')} ({best_model.get('iou', '')})",
            "implication": "pembahasan kinerja harus melihat IoU/Dice serta precision/recall",
        },
        {
            "finding_id": "F4",
            "topic": "kondisi data ekstrem",
            "evidence": f"{config.test_region} memiliki {to_int(test.get('tile_count'))} tile dan {fmt_float(pct(to_float(test.get('s2_valid_pixels')), to_float(test.get('valid_pixels'))))}% piksel S2 valid",
            "implication": "interpretasi visual tetap perlu memeriksa mask validitas dan kelas air permanen",
        },
    ]


def _narrative_4_7(config, rows: list[dict[str, object]]):
    spec = _spec("Narasi 4.7")
    if len(rows) >= 2:
        leader = max(rows, key=lambda row: float(row["iou"]))
        text = f"""
        Pembahasan efektivitas model diturunkan dari metrik checkpoint terbaik spatial CV dan source arsitektur.
        Pada region uji {config.test_region}, {leader['model']} memiliki IoU tertinggi yaitu
        {leader['iou']}. ProCANet unggul pada loss, IoU, Dice, akurasi, precision, dan specificity,
        sedangkan U-Net unggul tipis pada recall dan memiliki FN lebih rendah. Perbandingan tetap
        mempertimbangkan FP dan FN karena kebutuhan operasional tidak selalu identik dengan akurasi global.
        """
    else:
        text = "Pembahasan efektivitas model belum lengkap karena metrics checkpoint terbaik spatial CV tidak ditemukan."
    return write_text_artifact(config, spec, text, source=f"{config.evaluation_source}/metrics.csv;training/models")


def _narrative_4_8(config, rows: list[dict[str, object]]):
    spec = _spec("Narasi 4.8")
    cases = ", ".join(str(row.get("kasus", row.get("case_label", ""))) for row in rows) if rows else "tidak ada tile ekstrem terpilih"
    text = f"""
    Kasus data ekstrem dipilih ulang dari tile `.npz` asli, meliputi {cases}. Statistik pada
    Tabel 4.16 dan 4.17 memperlihatkan bahwa kualitas Sentinel-2, dominasi badan air, dan
    konfigurasi topografi dapat mempengaruhi interpretasi visual. Generator tidak menggunakan
    gambar lama; setiap panel dibentuk langsung dari channel input, label, dan mask tile.
    """
    return write_text_artifact(config, spec, text, source="dataset/preprocessing_summary.csv;dataset/feature_preprocessing_summary.csv;dataset/tiles/7ch")


def _final_metric_rows(config) -> list[dict[str, object]]:
    rows = []
    for model in MODEL_KEYS:
        metrics_path = config.evaluation_dir(model) / "metrics.csv"
        if not metrics_path.exists():
            continue
        raw_rows = read_csv_rows(metrics_path)
        row = next((item for item in raw_rows if item.get("region") == config.test_region), raw_rows[0] if raw_rows else None)
        if row is None:
            continue
        tp = to_int(row.get("tp"))
        fp = to_int(row.get("fp"))
        fn = to_int(row.get("fn"))
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "iou": fmt_float(to_float(row.get("iou")), 6),
                "dice": fmt_float(to_float(row.get("dice")), 6),
                "accuracy": fmt_float(to_float(row.get("accuracy")), 6),
                "precision": fmt_float(tp / (tp + fp) if tp + fp else 0.0, 6),
                "recall": fmt_float(tp / (tp + fn) if tp + fn else 0.0, 6),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return rows


def _effectiveness_interpretation(model: str, precision: float, recall: float) -> str:
    if precision >= recall:
        return f"{model} lebih konservatif terhadap prediksi banjir dibanding cakupan recall"
    return f"{model} lebih agresif menangkap banjir dengan recall relatif lebih tinggi"


def _choose_existing_or_scan(tile_dir: Path, preferred_name: str, key) -> Path | None:
    preferred = tile_dir / preferred_name if preferred_name else None
    if preferred is not None and preferred.exists():
        return preferred
    best_path = None
    best_score = None
    for path in sorted(tile_dir.glob("*.npz")):
        payload = _load_npz(path)
        score = key(payload)
        if best_score is None or score > best_score:
            best_path = path
            best_score = score
    return best_path


def _valid_mask(payload: dict[str, np.ndarray]) -> np.ndarray:
    valid = np.squeeze(payload["valid_mask"]).astype(bool)
    feature = np.squeeze(payload.get("feature_valid_mask", valid)).astype(bool)
    return valid & feature


def _masked_mean(array: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    values = np.asarray(array, dtype=np.float32)[mask]
    return float(np.nanmean(values)) if values.size else 0.0


def _region_from_tile(path: Path) -> str:
    return path.stem.split("_r", 1)[0]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}
