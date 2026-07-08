from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bab4.artifacts import ALL_ARTIFACTS, ArtifactSpec
from bab4.common import MODEL_KEYS, MODEL_LABELS, REGIONS, fmt_float, pct, read_csv_row_map, read_csv_rows, region_quality_from_s2_pct, to_float, to_int
from bab4.plots import hsv_to_rgb, normalize_image, savefig, setup_style
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
    artifacts = [
        write_table(
            config,
            _spec("Tambahan 4.7"),
            effectiveness_rows,
            source="runs/final/{unet,procanet}/eval_test/metrics.csv;training/models/{unet,procanet}.py",
        ),
        write_table(
            config,
            _spec("Tambahan 4.7b"),
            _literature_context_rows(effectiveness_rows),
            source="runs/final/{unet,procanet}/eval_test/metrics.csv;training/models/{unet,procanet}.py",
        ),
        _narrative_4_7(config, effectiveness_rows),
    ]
    return section_result("4.7", artifacts)


def generate_4_8(config):
    difficult_rows = _difficult_data_rows(config)
    extreme_rows, selected_tiles = _extreme_tile_rows(config)
    artifacts = [
        write_table(config, _spec("Tabel 4.16"), difficult_rows, source="dataset/preprocessing_summary.csv;dataset/feature_preprocessing_summary.csv"),
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
            source="dataset/preprocessing_summary.csv;runs/final/{unet,procanet}/eval_test/metrics.csv;bab4 generator functions",
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
            "interpretation": "attention fusion menekan FP tetapi pada hasil final recall lebih rendah",
        },
        {
            "comparison_axis": "operational trade-off",
            "reference_context": "validasi wilayah held-out Aceh_Utara",
            "bab4_evidence": "precision/recall dihitung dari confusion matrix final",
            "interpretation": "pilihan model perlu mempertimbangkan biaya FP dan FN, bukan hanya akurasi",
        },
    ]


def _difficult_data_rows(config) -> list[dict[str, object]]:
    summary = read_csv_row_map(config.dataset_root / "preprocessing_summary.csv")
    feature_summary = read_csv_row_map(config.dataset_root / "feature_preprocessing_summary.csv")
    rows = []
    for region in REGIONS:
        row = summary[region]
        valid = to_int(row.get("valid_pixels"))
        flood = to_int(row.get("flood_pixels"))
        water = to_int(row.get("water_river_pixels"))
        s2_valid = to_int(row.get("s2_valid_pixels"))
        feature_row = feature_summary.get(region, {})
        s2_pct = pct(s2_valid, valid)
        flags = []
        if s2_pct < 0.01:
            flags.append("Sentinel-2 kosong/hampir kosong")
        if pct(water, valid) > 10:
            flags.append("badan air dominan")
        if pct(flood, valid) > 20:
            flags.append("label banjir sangat dominan")
        if to_int(row.get("tile_count")) < 50:
            flags.append("jumlah tile kecil")
        rows.append(
            {
                "region": region,
                "split": row.get("split", "cv"),
                "tile_count": to_int(row.get("tile_count")),
                "valid_pixels": valid,
                "flood_pct_of_valid": fmt_float(pct(flood, valid)),
                "water_river_pct_of_valid": fmt_float(pct(water, valid)),
                "s2_valid_pct_of_valid": fmt_float(s2_pct),
                "s2_valid_pct_feature_report": fmt_float(to_float(feature_row.get("s2_valid_pct"))),
                "feature_valid_pct": fmt_float(to_float(feature_row.get("feature_valid_pct"))),
                "s2_quality": region_quality_from_s2_pct(s2_pct),
                "difficulty_flags": "; ".join(flags) if flags else "tidak dominan",
            }
        )
    return rows


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
        "case_id": case_id,
        "case_label": case_label,
        "region": _region_from_tile(path),
        "tile": path.stem,
        "valid_pixels": valid_count,
        "flood_pixels": int(flood.sum()),
        "water_river_pixels": int(water.sum()),
        "s2_valid_pixels": int(s2.sum()),
        "flood_pct_of_valid": fmt_float(pct(int(flood.sum()), valid_count)),
        "water_river_pct_of_valid": fmt_float(pct(int(water.sum()), valid_count)),
        "s2_valid_pct_of_valid": fmt_float(pct(int(s2.sum()), valid_count)),
        "vv_mean_valid": fmt_float(_masked_mean(x[0], valid)),
        "vh_mean_valid": fmt_float(_masked_mean(x[1], valid)),
        "slope_mean_valid": fmt_float(_masked_mean(x[5], valid)),
        "hand_mean_valid": fmt_float(_masked_mean(x[6], valid)),
        "selection_rule": selection_rule,
        "source_file": str(path),
    }


def _figure_difficult_cases(config, rows: list[dict[str, object]]):
    spec = _spec("Gambar 4.15")
    labels = [str(row["region"]) for row in rows]
    flood = [float(row["flood_pct_of_valid"]) for row in rows]
    water = [float(row["water_river_pct_of_valid"]) for row in rows]
    s2 = [float(row["s2_valid_pct_of_valid"]) for row in rows]
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 5.0))
    x = np.arange(len(rows))
    width = 0.26
    ax.bar(x - width, flood, width=width, label="Flood/valid")
    ax.bar(x, water, width=width, label="Water/valid")
    ax.bar(x + width, s2, width=width, label="S2 valid/valid")
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("Persentase (%)")
    ax.set_title("Kondisi data sulit per wilayah")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
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
    panels = _panels_for_extreme_case(artifact_id, x, flood, water, s2)
    setup_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.4))
    for ax, (panel_title, image, cmap) in zip(axes, panels):
        ax.imshow(image, cmap=cmap, vmin=0 if cmap == "gray" else None, vmax=1 if cmap == "gray" else None)
        ax.set_title(panel_title)
        ax.axis("off")
    fig.suptitle(f"{title}: {tile_path.stem}", y=0.98)
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source=str(tile_path))


def _panels_for_extreme_case(artifact_id: str, x: np.ndarray, flood: np.ndarray, water: np.ndarray, s2: np.ndarray):
    if artifact_id == "Gambar 4.16":
        return [
            ("VV", normalize_image(x[0]), "gray"),
            ("HSV pseudo-RGB", hsv_to_rgb(x[2:5]), None),
            ("S2 valid mask", s2, "gray"),
            ("Label flood", flood, "gray"),
        ]
    if artifact_id == "Gambar 4.17":
        return [
            ("VV", normalize_image(x[0]), "gray"),
            ("VH", normalize_image(x[1]), "gray"),
            ("Slope", normalize_image(x[5]), "viridis"),
            ("HAND", normalize_image(x[6]), "viridis"),
            ("Label flood", flood, "gray"),
        ]
    return [
        ("HSV pseudo-RGB", hsv_to_rgb(x[2:5]), None),
        ("Water/river mask", water, "gray"),
        ("Label flood", flood, "gray"),
        ("VV", normalize_image(x[0]), "gray"),
    ]


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
            "topic": "model final",
            "evidence": f"model terbaik berdasarkan IoU final: {best_model.get('model', '')} ({best_model.get('iou', '')})",
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
        Pembahasan efektivitas model diturunkan dari metrik final dan source arsitektur.
        Pada region uji {config.test_region}, {leader['model']} memiliki IoU tertinggi yaitu
        {leader['iou']}. Perbandingan tetap mempertimbangkan precision, recall, FP, dan FN
        karena kebutuhan operasional segmentasi banjir tidak selalu identik dengan akurasi global.
        """
    else:
        text = "Pembahasan efektivitas model belum lengkap karena metrics final tidak ditemukan."
    return write_text_artifact(config, spec, text, source="runs/final/{unet,procanet}/eval_test/metrics.csv;training/models")


def _narrative_4_8(config, rows: list[dict[str, object]]):
    spec = _spec("Narasi 4.8")
    cases = ", ".join(str(row["case_label"]) for row in rows) if rows else "tidak ada tile ekstrem terpilih"
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
        metrics_path = config.runs_root / "final" / model / "eval_test" / "metrics.csv"
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
