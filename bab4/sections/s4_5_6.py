from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from bab4.artifacts import ALL_ARTIFACTS
from bab4.common import MODEL_KEYS, MODEL_LABELS, fmt_float, read_csv_rows, to_float, to_int
from bab4.plots import hsv_to_rgb, normalize_image, savefig, setup_style
from bab4.sections.base import section_result
from bab4.writer import figure_result, missing_result, write_table, write_text_artifact


TARGET_TILE = "Aceh_Utara_r001280_c005632.npz"


def _spec(artifact_id: str):
    return next(spec for spec in ALL_ARTIFACTS if spec.artifact_id == artifact_id)


def generate_4_5_6(config):
    artifacts = []
    for result in (generate_4_5(config), generate_4_6(config)):
        artifacts.extend(result.artifacts)
    return section_result("4.5-4.6", artifacts)


def generate_4_5(config):
    metric_rows = _metric_rows(config)
    if not metric_rows:
        return section_result(
            "4.5",
            [
                missing_result(
                    config,
                    _spec("Tabel 4.13"),
                    source="runs/final/{unet,procanet}/eval_test/metrics.csv",
                    note="metrics evaluasi final tidak ditemukan",
                ),
                missing_result(
                    config,
                    _spec("Tabel 4.14"),
                    source="runs/final/{unet,procanet}/eval_test/metrics.csv",
                    note="metrics evaluasi final tidak ditemukan",
                ),
                missing_result(
                    config,
                    _spec("Gambar 4.12"),
                    source="runs/final/{unet,procanet}/eval_test/metrics.csv",
                    note="metrics evaluasi final tidak ditemukan",
                ),
                _narrative_4_5(config, metric_rows),
            ],
        )
    artifacts = [
        write_table(config, _spec("Tabel 4.13"), metric_rows, source="runs/final/{unet,procanet}/eval_test/metrics.csv"),
        write_table(config, _spec("Tabel 4.14"), _confusion_rows(metric_rows), source="runs/final/{unet,procanet}/eval_test/metrics.csv"),
        _figure_metric_comparison(config, metric_rows),
        _narrative_4_5(config, metric_rows),
    ]
    return section_result("4.5", artifacts)


def generate_4_6(config):
    tile_path = _select_tile(config)
    if tile_path is None:
        missing_note = "tile test dan/atau prediksi final tidak ditemukan"
        return section_result(
            "4.6",
            [
                missing_result(
                    config,
                    _spec("Tabel 4.15"),
                    source="dataset/tiles/7ch/by_region/Aceh_Utara;runs/final/*/eval_test/predictions/Aceh_Utara",
                    note=missing_note,
                ),
                missing_result(
                    config,
                    _spec("Gambar 4.13"),
                    source="dataset/tiles/7ch/by_region/Aceh_Utara;runs/final/*/eval_test/predictions/Aceh_Utara",
                    note=missing_note,
                ),
                missing_result(
                    config,
                    _spec("Gambar 4.14"),
                    source="dataset/tiles/7ch/by_region/Aceh_Utara;runs/final/*/eval_test/predictions/Aceh_Utara",
                    note=missing_note,
                ),
                _narrative_4_6(config, None),
            ],
        )
    tile = _load_npz(tile_path)
    predictions = {model: _load_npz(_prediction_path(config, model, tile_path.name)) for model in MODEL_KEYS}
    artifacts = [
        write_table(
            config,
            _spec("Tabel 4.15"),
            _error_count_rows(tile_path, tile, predictions),
            source=f"{tile_path.relative_to(config.root)};runs/final/{{unet,procanet}}/eval_test/predictions/{config.test_region}/{tile_path.name}",
        ),
        _figure_segmentation_panel(config, tile_path, tile, predictions),
        _figure_error_map(config, tile_path, tile, predictions),
        _narrative_4_6(config, tile_path),
    ]
    return section_result("4.6", artifacts)


def _metric_rows(config) -> list[dict[str, object]]:
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
        tn = to_int(row.get("tn"))
        fp = to_int(row.get("fp"))
        fn = to_int(row.get("fn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "region": row.get("region", config.test_region),
                "loss": fmt_float(to_float(row.get("loss")), 6),
                "iou": fmt_float(to_float(row.get("iou")), 6),
                "dice": fmt_float(to_float(row.get("dice")), 6),
                "accuracy": fmt_float(to_float(row.get("accuracy")), 6),
                "precision": fmt_float(precision, 6),
                "recall": fmt_float(recall, 6),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "evaluated_batches": to_int(row.get("batches")),
                "source_file": str(metrics_path.relative_to(config.root)),
            }
        )
    return rows


def _confusion_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in metric_rows:
        tp = int(row["tp"])
        tn = int(row["tn"])
        fp = int(row["fp"])
        fn = int(row["fn"])
        total = tp + tn + fp + fn
        rows.append(
            {
                "model": row["model"],
                "region": row["region"],
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "total_evaluated_pixels": total,
                "reference_flood_pixels": tp + fn,
                "predicted_flood_pixels": tp + fp,
                "precision": row["precision"],
                "recall": row["recall"],
                "false_positive_rate": fmt_float(fp / (fp + tn) if fp + tn else 0.0, 6),
                "false_negative_rate": fmt_float(fn / (fn + tp) if fn + tp else 0.0, 6),
            }
        )
    return rows


def _figure_metric_comparison(config, rows: list[dict[str, object]]):
    spec = _spec("Gambar 4.12")
    metrics = ["iou", "dice", "accuracy", "precision", "recall"]
    labels = [str(row["model"]) for row in rows]
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(metrics))
    width = 0.34
    for idx, row in enumerate(rows):
        offset = (idx - (len(rows) - 1) / 2) * width
        ax.bar(x + offset, [float(row[metric]) for metric in metrics], width=width, label=labels[idx])
    ax.set_xticks(x, [metric.upper() if metric in {"iou"} else metric.title() for metric in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Nilai")
    ax.set_title(f"Perbandingan metrik final pada wilayah uji {config.test_region}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source="runs/final/{unet,procanet}/eval_test/metrics.csv")


def _select_tile(config) -> Path | None:
    tile_dir = config.dataset_root / "tiles" / "7ch" / "by_region" / config.test_region
    target = tile_dir / TARGET_TILE
    if target.exists() and all(_prediction_path(config, model, target.name).exists() for model in MODEL_KEYS):
        return target
    best_path = None
    best_score = -1
    for path in sorted(tile_dir.glob("*.npz")):
        if not all(_prediction_path(config, model, path.name).exists() for model in MODEL_KEYS):
            continue
        tile = _load_npz(path)
        valid = _valid_mask(tile)
        flood = np.squeeze(tile["y"]).astype(bool) & valid
        score = int(flood.sum())
        if score > best_score:
            best_path = path
            best_score = score
    return best_path


def _prediction_path(config, model: str, tile_name: str) -> Path:
    return config.runs_root / "final" / model / "eval_test" / "predictions" / config.test_region / tile_name


def _error_count_rows(tile_path: Path, tile: dict[str, np.ndarray], predictions: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    truth = np.squeeze(tile["y"]).astype(bool)
    valid = _valid_mask(tile)
    rows = []
    for model in MODEL_KEYS:
        prediction = np.squeeze(predictions[model]["prediction"]).astype(bool)
        effective = np.squeeze(predictions[model].get("effective_valid_mask", valid)).astype(bool) & valid
        tp = int((truth & prediction & effective).sum())
        tn = int((~truth & ~prediction & effective).sum())
        fp = int((~truth & prediction & effective).sum())
        fn = int((truth & ~prediction & effective).sum())
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        rows.append(
            {
                "tile": tile_path.stem,
                "model": MODEL_LABELS[model],
                "valid_pixels": int(effective.sum()),
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "tile_iou": fmt_float(iou, 6),
                "tile_dice": fmt_float(dice, 6),
            }
        )
    return rows


def _figure_segmentation_panel(config, tile_path: Path, tile: dict[str, np.ndarray], predictions: dict[str, dict[str, np.ndarray]]):
    spec = _spec("Gambar 4.13")
    x = np.asarray(tile["x"], dtype=np.float32)
    truth = np.squeeze(tile["y"])
    panels = [
        ("VV", normalize_image(x[0]), "gray"),
        ("HSV pseudo-RGB", hsv_to_rgb(x[2:5]), None),
        ("HAND", normalize_image(x[6]), "viridis"),
        ("Label UNOSAT", truth, "gray"),
        ("Prediksi U-Net", np.squeeze(predictions["unet"]["prediction"]), "gray"),
        ("Prediksi ProCANet", np.squeeze(predictions["procanet"]["prediction"]), "gray"),
    ]
    setup_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 3.4))
    for ax, (title, arr, cmap) in zip(axes, panels):
        ax.imshow(arr, cmap=cmap, vmin=0 if cmap == "gray" else None, vmax=1 if cmap == "gray" else None)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Panel input, label, dan prediksi: {tile_path.stem}", y=0.98)
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(
        config,
        spec,
        path,
        source=f"{tile_path.relative_to(config.root)};runs/final/{{unet,procanet}}/eval_test/predictions/{config.test_region}/{tile_path.name}",
    )


def _figure_error_map(config, tile_path: Path, tile: dict[str, np.ndarray], predictions: dict[str, dict[str, np.ndarray]]):
    spec = _spec("Gambar 4.14")
    truth = np.squeeze(tile["y"]).astype(bool)
    valid = _valid_mask(tile)
    cmap = ListedColormap(["#d1d5db", "#ffffff", "#16a34a", "#f97316", "#dc2626"])
    labels = [
        Patch(facecolor="#d1d5db", label="Invalid"),
        Patch(facecolor="#ffffff", edgecolor="#9ca3af", label="TN"),
        Patch(facecolor="#16a34a", label="TP"),
        Patch(facecolor="#f97316", label="FP"),
        Patch(facecolor="#dc2626", label="FN"),
    ]
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.0))
    for ax, model in zip(axes, MODEL_KEYS):
        prediction = np.squeeze(predictions[model]["prediction"]).astype(bool)
        effective = np.squeeze(predictions[model].get("effective_valid_mask", valid)).astype(bool) & valid
        error = np.zeros(truth.shape, dtype=np.uint8)
        error[~truth & ~prediction & effective] = 1
        error[truth & prediction & effective] = 2
        error[~truth & prediction & effective] = 3
        error[truth & ~prediction & effective] = 4
        ax.imshow(error, cmap=cmap, vmin=0, vmax=4)
        ax.set_title(MODEL_LABELS[model])
        ax.axis("off")
    fig.legend(handles=labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(f"Error map TP/FP/FN/TN: {tile_path.stem}", y=0.98)
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(
        config,
        spec,
        path,
        source=f"{tile_path.relative_to(config.root)};runs/final/{{unet,procanet}}/eval_test/predictions/{config.test_region}/{tile_path.name}",
    )


def _narrative_4_5(config, rows: list[dict[str, object]]):
    spec = _spec("Narasi 4.5")
    if rows:
        best = max(rows, key=lambda row: float(row["iou"]))
        summary = f"Model dengan IoU tertinggi pada wilayah uji adalah {best['model']} dengan IoU {best['iou']}."
    else:
        summary = "Metrik final belum dapat diringkas karena source metrics.csv tidak ditemukan."
    text = f"""
    Evaluasi akhir dihitung ulang dari `runs/final/*/eval_test/metrics.csv` untuk region
    {config.test_region}. Precision dan recall diturunkan kembali dari TP, FP, dan FN sehingga
    tabel metrik dan confusion matrix memiliki sumber numerik yang sama. {summary}
    """
    return write_text_artifact(config, spec, text, source="runs/final/{unet,procanet}/eval_test/metrics.csv")


def _narrative_4_6(config, tile_path: Path | None):
    spec = _spec("Narasi 4.6")
    tile_text = tile_path.stem if tile_path else TARGET_TILE.removesuffix(".npz")
    text = f"""
    Analisis visual spasial dibuat dari tile sumber `{tile_text}` dan prediksi final pada
    `runs/final/*/eval_test/predictions`. Error map dihitung ulang secara piksel sebagai TN,
    TP, FP, dan FN dengan valid mask tile/prediksi, sehingga angka pada Tabel 4.15 berasal
    dari array prediksi, bukan dari artefak gambar lama.
    """
    return write_text_artifact(
        config,
        spec,
        text,
        source=f"dataset/tiles/7ch/by_region/{config.test_region};runs/final/{{unet,procanet}}/eval_test/predictions/{config.test_region}",
    )


def _valid_mask(tile: dict[str, np.ndarray]) -> np.ndarray:
    valid = np.squeeze(tile["valid_mask"]).astype(bool)
    feature = np.squeeze(tile.get("feature_valid_mask", valid)).astype(bool)
    return valid & feature


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}
