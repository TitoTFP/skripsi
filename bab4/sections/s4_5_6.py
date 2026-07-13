from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from bab4.artifacts import ALL_ARTIFACTS
from bab4.common import MODEL_KEYS, MODEL_LABELS, fmt_float, read_csv_rows, to_float, to_int
from bab4.plots import hsv_to_display_rgb, normalize_image, savefig, setup_style
from bab4.raster import open_dataset
from bab4.sections.base import section_result
from bab4.writer import figure_result, missing_result, write_table, write_text_artifact


TARGET_TILE = "Aceh_Utara_r001280_c005632.npz"
MOSAIC_BLOCK_SIZE = 512


def _spec(artifact_id: str):
    return next(spec for spec in ALL_ARTIFACTS if spec.artifact_id == artifact_id)


def generate_4_5_6(config):
    artifacts = []
    for result in (generate_4_5(config), generate_4_6(config)):
        artifacts.extend(result.artifacts)
    return section_result("4.5-4.6", artifacts)


def generate_4_5(config):
    metrics_source = _metric_source_description(config)
    source_error = ""
    try:
        metric_rows = _metric_rows(config)
    except (FileNotFoundError, ValueError) as exc:
        metric_rows = []
        source_error = str(exc)
    if not metric_rows:
        note = source_error or "sumber metrik mosaik evaluasi checkpoint terbaik spatial CV tidak ditemukan"
        return section_result(
            "4.5",
            [
                missing_result(
                    config,
                    _spec("Tabel 4.13"),
                    source=metrics_source,
                    note=note,
                ),
                missing_result(
                    config,
                    _spec("Tabel 4.14"),
                    source=metrics_source,
                    note=note,
                ),
                missing_result(
                    config,
                    _spec("Gambar 4.12"),
                    source=metrics_source,
                    note=note,
                ),
                _narrative_4_5(config, metric_rows, source_error=source_error),
            ],
        )
    artifacts = [
        write_table(config, _spec("Tabel 4.13"), _metric_table_rows(metric_rows), source=metrics_source),
        write_table(config, _spec("Tabel 4.14"), _confusion_rows(metric_rows), source=metrics_source),
        _figure_metric_comparison(config, metric_rows),
        _narrative_4_5(config, metric_rows),
    ]
    return section_result("4.5", artifacts)


def generate_4_6(config):
    tile_path = _select_tile(config)
    if tile_path is None:
        missing_note = "tile test dan/atau prediksi checkpoint terbaik spatial CV tidak ditemukan"
        prediction_source = f"dataset/tiles/7ch/by_region/{config.test_region};{config.evaluation_source}/predictions/{config.test_region}"
        return section_result(
            "4.6",
            [
                missing_result(
                    config,
                    _spec("Tabel 4.15"),
                    source=prediction_source,
                    note=missing_note,
                ),
                missing_result(
                    config,
                    _spec("Gambar 4.13"),
                    source=prediction_source,
                    note=missing_note,
                ),
                missing_result(
                    config,
                    _spec("Gambar 4.14"),
                    source=prediction_source,
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
            source=f"{tile_path.relative_to(config.root)};{config.evaluation_source}/predictions/{config.test_region}/{tile_path.name}",
        ),
        _figure_segmentation_panel(config, tile_path, tile, predictions),
        _figure_error_map(config, tile_path, tile, predictions),
        _narrative_4_6(config, tile_path),
    ]
    return section_result("4.6", artifacts)


def _metric_rows(config) -> list[dict[str, object]]:
    rows = []
    label_path = config.dataset_root / "labels_unosat_rasterized" / config.test_region / "label_flood_binary.tif"
    for model in MODEL_KEYS:
        metrics_path = config.evaluation_dir(model) / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"sumber loss tile tidak ditemukan: {metrics_path}")
        raw_rows = read_csv_rows(metrics_path)
        row = next((item for item in raw_rows if item.get("region") == config.test_region), raw_rows[0] if raw_rows else None)
        if row is None:
            raise ValueError(f"baris metrik tile kosong: {metrics_path}")
        geotiff_dir = config.evaluation_dir(model) / "geotiff"
        probability_path = geotiff_dir / f"{config.test_region}_probability.tif"
        effective_mask_path = geotiff_dir / f"{config.test_region}_effective_valid_mask.tif"
        stats = _mosaic_binary_stats(
            probability_path,
            effective_mask_path,
            label_path,
            threshold=config.threshold,
        )
        tp = stats["tp"]
        tn = stats["tn"]
        fp = stats["fp"]
        fn = stats["fn"]
        evaluated_unique_pixels = stats["evaluated_unique_pixels"]
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        accuracy = (tp + tn) / evaluated_unique_pixels if evaluated_unique_pixels else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "region": row.get("region", config.test_region),
                "loss_tile_rata_rata_batch": fmt_float(to_float(row.get("loss")), 6),
                "iou": fmt_float(iou, 6),
                "dice": fmt_float(dice, 6),
                "accuracy": fmt_float(accuracy, 6),
                "precision": fmt_float(precision, 6),
                "recall": fmt_float(recall, 6),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "evaluated_unique_pixels": evaluated_unique_pixels,
                "unit_evaluasi": "piksel unik mosaik dalam effective_valid_mask",
                "evaluated_batches": to_int(row.get("batches")),
                "source_file": ";".join(
                    str(path.relative_to(config.root))
                    for path in (metrics_path, probability_path, effective_mask_path, label_path)
                ),
            }
        )
    return rows


def _mosaic_binary_stats(
    probability_path: Path,
    effective_mask_path: Path,
    label_path: Path,
    *,
    threshold: float,
    block_size: int = MOSAIC_BLOCK_SIZE,
) -> dict[str, int]:
    paths = (Path(probability_path), Path(effective_mask_path), Path(label_path))
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"sumber raster mosaik tidak ditemukan: {path}")

    probability_ds, effective_ds, label_ds = (open_dataset(path) for path in paths)
    _validate_same_grid(paths, (probability_ds, effective_ds, label_ds))

    stats = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "evaluated_unique_pixels": 0}
    width = probability_ds.RasterXSize
    height = probability_ds.RasterYSize
    for row in range(0, height, block_size):
        block_height = min(block_size, height - row)
        probability = probability_ds.GetRasterBand(1).ReadAsArray(0, row, width, block_height).astype(np.float32)
        effective = effective_ds.GetRasterBand(1).ReadAsArray(0, row, width, block_height).astype(bool)
        truth = label_ds.GetRasterBand(1).ReadAsArray(0, row, width, block_height).astype(bool)
        invalid_probability = effective & (~np.isfinite(probability) | (probability < 0.0) | (probability > 1.0))
        if invalid_probability.any():
            raise ValueError(f"probabilitas mosaik tidak valid pada effective mask: {probability_path}")
        prediction = probability >= threshold
        stats["tp"] += int((effective & truth & prediction).sum())
        stats["tn"] += int((effective & ~truth & ~prediction).sum())
        stats["fp"] += int((effective & ~truth & prediction).sum())
        stats["fn"] += int((effective & truth & ~prediction).sum())
        stats["evaluated_unique_pixels"] += int(effective.sum())
    return stats


def _validate_same_grid(paths: tuple[Path, ...], datasets: tuple[object, ...]) -> None:
    reference = datasets[0]
    reference_grid = (
        reference.RasterXSize,
        reference.RasterYSize,
        reference.GetGeoTransform(),
        reference.GetProjection(),
    )
    for path, dataset in zip(paths[1:], datasets[1:]):
        grid = (
            dataset.RasterXSize,
            dataset.RasterYSize,
            dataset.GetGeoTransform(),
            dataset.GetProjection(),
        )
        if grid != reference_grid:
            raise ValueError(f"grid raster tidak selaras: {paths[0]} dan {path}")


def _metric_table_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "model": row["model"],
            "loss_tile_rata_rata_batch": _report_float(row["loss_tile_rata_rata_batch"], 6),
            "iou": _report_float(row["iou"], 6),
            "dice_f1": _report_float(row["dice"], 6),
            "accuracy": _report_float(row["accuracy"], 6),
            "precision": _report_float(row["precision"], 6),
            "recall": _report_float(row["recall"], 6),
            "specificity": _report_float(
                int(row["tn"]) / max(int(row["tn"]) + int(row["fp"]), 1),
                6,
            ),
            "evaluated_unique_pixels": row["evaluated_unique_pixels"],
            "unit_evaluasi": row["unit_evaluasi"],
        }
        for row in metric_rows
    ]


def _report_float(value: object, digits: int = 3) -> str:
    quant = Decimal("1").scaleb(-digits)
    adjusted = Decimal(str(float(value) + 0.5 * (10 ** -(digits + 2))))
    return f"{adjusted.quantize(quant, rounding=ROUND_HALF_UP):.{digits}f}"


def _confusion_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in metric_rows:
        tp = int(row["tp"])
        tn = int(row["tn"])
        fp = int(row["fp"])
        fn = int(row["fn"])
        rows.append(
            {
                "model": row["model"],
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "fpr": fmt_float(fp / (fp + tn) if fp + tn else 0.0, 4),
                "fnr": fmt_float(fn / (fn + tp) if fn + tp else 0.0, 4),
                "evaluated_unique_pixels": row["evaluated_unique_pixels"],
                "unit_evaluasi": row["unit_evaluasi"],
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
        bars = ax.bar(x + offset, [float(row[metric]) for metric in metrics], width=width, label=labels[idx])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x, ["iou", "dice_f1", "accuracy", "precision", "recall"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Nilai")
    ax.set_xlabel("Metrik")
    ax.set_title(f"Metrik Checkpoint Terbaik Spatial CV pada {config.test_region.replace('_', ' ')}")
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", alpha=0.25)
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source=_metric_source_description(config))


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
    return config.evaluation_dir(model) / "predictions" / config.test_region / tile_name


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
                "model": MODEL_LABELS[model],
                "valid_pixels": int(effective.sum()),
                "label_positive_pixels": tp + fn,
                "predicted_positive_pixels": tp + fp,
                "tn": tn,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return rows


def _figure_segmentation_panel(config, tile_path: Path, tile: dict[str, np.ndarray], predictions: dict[str, dict[str, np.ndarray]]):
    spec = _spec("Gambar 4.13")
    x = np.asarray(tile["x"], dtype=np.float32)
    truth = np.squeeze(tile["y"])
    panels = [
        ("Kanal Sentinel-1 VV", normalize_image(x[0]), "gray"),
        ("Kanal Sentinel-1 VH", normalize_image(x[1]), "gray"),
        ("Pseudo-RGB HSV Sentinel-2", hsv_to_display_rgb(x[2:5]), None),
        ("Slope", normalize_image(x[5]), "magma"),
        ("HAND", normalize_image(x[6]), "viridis"),
        ("Label UNOSAT", truth, "Blues"),
        ("Prediksi U-Net", np.squeeze(predictions["unet"]["prediction"]), "Oranges"),
        ("Prediksi ProCANet", np.squeeze(predictions["procanet"]["prediction"]), "Greens"),
    ]
    setup_style()
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.6))
    for idx, (ax, (title, arr, cmap)) in enumerate(zip(axes.ravel(), panels)):
        ax.imshow(arr, cmap=cmap, vmin=0 if cmap == "gray" else None, vmax=1 if cmap == "gray" else None)
        # ax.set_title(title, fontsize=8)
        ax.text(0.5, -0.08, f"({chr(97 + idx)})", transform=ax.transAxes, ha="center", va="top", fontsize=20)
        ax.axis("off")
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(
        config,
        spec,
        path,
        source=f"{tile_path.relative_to(config.root)};{config.evaluation_source}/predictions/{config.test_region}/{tile_path.name}",
    )


def _figure_error_map(config, tile_path: Path, tile: dict[str, np.ndarray], predictions: dict[str, dict[str, np.ndarray]]):
    spec = _spec("Gambar 4.14")
    truth = np.squeeze(tile["y"]).astype(bool)
    valid = _valid_mask(tile)
    cmap = ListedColormap(["#ffffff", "#16a34a", "#dc2626", "#2563eb"])
    labels = [
        Patch(facecolor="#16a34a", label="TP"),
        Patch(facecolor="#ffffff", edgecolor="#9ca3af", label="TN"),
        Patch(facecolor="#dc2626", label="FP"),
        Patch(facecolor="#2563eb", label="FN"),
    ]
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    for ax, model in zip(axes, MODEL_KEYS):
        prediction = np.squeeze(predictions[model]["prediction"]).astype(bool)
        effective = np.squeeze(predictions[model].get("effective_valid_mask", valid)).astype(bool) & valid
        error = np.zeros(truth.shape, dtype=np.uint8)
        error[truth & prediction & effective] = 1
        error[~truth & prediction & effective] = 2
        error[truth & ~prediction & effective] = 3
        ax.imshow(error, cmap=cmap, vmin=0, vmax=3)
        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.axis("off")
    fig.legend(handles=labels, loc="lower center", ncol=4, frameon=False)
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(
        config,
        spec,
        path,
        source=f"{tile_path.relative_to(config.root)};{config.evaluation_source}/predictions/{config.test_region}/{tile_path.name}",
    )


def _narrative_4_5(config, rows: list[dict[str, object]], *, source_error: str = ""):
    spec = _spec("Narasi 4.5")
    if rows:
        best = max(rows, key=lambda row: float(row["iou"]))
        summary = f"Model dengan IoU tertinggi pada wilayah uji adalah {best['model']} dengan IoU {best['iou']}."
    else:
        summary = f"Metrik final belum dapat diringkas karena sumber mosaik tidak valid: {source_error or 'tidak ditemukan'}."
    text = dedent(
        f"""
        Evaluasi wilayah uji {config.test_region} menggunakan mosaik piksel unik. Probabilitas dari
        seluruh tile yang bertumpang tindih terlebih dahulu dirata-ratakan pada GeoTIFF, kemudian
        diberi ambang {config.threshold:.1f}. TP, TN, FP, dan FN dihitung hanya satu kali untuk setiap
        piksel dalam `effective_valid_mask`. IoU, Dice/F1, accuracy, precision, recall, dan specificity
        seluruhnya diturunkan dari confusion matrix mosaik yang sama. Kolom
        `loss_tile_rata_rata_batch` tetap berasal dari `metrics.csv` dan dipertahankan sebagai loss
        rata-rata per batch tile, sehingga unitnya berbeda dari metrik klasifikasi piksel unik.
        ProCANet unggul pada IoU, Dice, accuracy, precision, dan specificity, sedangkan U-Net unggul
        tipis pada recall dan memiliki FN lebih rendah. {summary}
        """
    ).strip()
    return write_text_artifact(config, spec, text, source=_metric_source_description(config))


def _metric_source_description(config) -> str:
    return (
        f"{config.evaluation_source}/metrics.csv;"
        f"{config.evaluation_source}/geotiff/{config.test_region}_probability.tif;"
        f"{config.evaluation_source}/geotiff/{config.test_region}_effective_valid_mask.tif;"
        f"dataset/labels_unosat_rasterized/{config.test_region}/label_flood_binary.tif"
    )


def _narrative_4_6(config, tile_path: Path | None):
    spec = _spec("Narasi 4.6")
    tile_text = tile_path.stem if tile_path else TARGET_TILE.removesuffix(".npz")
    text = f"""
    Analisis visual spasial dibuat dari tile sumber `{tile_text}` dan prediksi checkpoint terbaik
    spatial CV pada `{config.evaluation_run}/*/eval_test/predictions`. Error map dihitung ulang secara piksel sebagai TN,
    TP, FP, dan FN dengan valid mask tile/prediksi, sehingga angka pada Tabel 4.15 berasal
    dari array prediksi, bukan dari artefak gambar lama.
    """
    return write_text_artifact(
        config,
        spec,
        text,
        source=f"dataset/tiles/7ch/by_region/{config.test_region};{config.evaluation_source}/predictions/{config.test_region}",
    )


def _valid_mask(tile: dict[str, np.ndarray]) -> np.ndarray:
    valid = np.squeeze(tile["valid_mask"]).astype(bool)
    feature = np.squeeze(tile.get("feature_valid_mask", valid)).astype(bool)
    return valid & feature


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}
