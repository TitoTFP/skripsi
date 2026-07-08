from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from bab4.artifacts import ALL_ARTIFACTS
from bab4.common import MODEL_KEYS, MODEL_LABELS, fmt_float, read_csv_rows, to_float, to_int
from bab4.plots import savefig, setup_style
from bab4.sections.base import section_result
from bab4.writer import figure_result, missing_result, write_table, write_text_artifact


def _spec(artifact_id: str):
    return next(spec for spec in ALL_ARTIFACTS if spec.artifact_id == artifact_id)


def generate_4_4(config):
    artifacts = []
    for result in (generate_4_4_1(config), generate_4_4_2(config), generate_4_4_3(config)):
        artifacts.extend(result.artifacts)
    return section_result("4.4", artifacts)


def generate_4_4_1(config):
    artifacts = [
        write_table(
            config,
            _spec("Tabel 4.10"),
            _model_spec_rows(config),
            source="training/models/unet.py;training/models/procanet.py;training/models/blocks.py;runs/final/*/config.json",
        ),
        write_table(
            config,
            _spec("Tabel 4.11"),
            _forward_contract_rows(config),
            source="training/models/unet.py;training/models/procanet.py;tests/test_models.py",
        ),
        _architecture_figure(config, "unet"),
        _architecture_figure(config, "procanet"),
        _narrative_4_4_1(config),
    ]
    return section_result("4.4.1", artifacts)


def generate_4_4_2(config):
    rows = _grid_summary_rows(config)
    if not rows:
        return section_result(
            "4.4.2",
            [
                missing_result(
                    config,
                    _spec("Tabel 4.12"),
                    source="runs/{unet,procanet}/fold_*/grid_*/metrics.csv",
                    note="grid search metrics.csv tidak ditemukan",
                ),
                missing_result(
                    config,
                    _spec("Gambar 4.10"),
                    source="runs/{unet,procanet}/fold_*/grid_*/metrics.csv",
                    note="grid search metrics.csv tidak ditemukan",
                ),
                _narrative_4_4_2(config, rows),
            ],
        )
    artifacts = [
        write_table(
            config,
            _spec("Tabel 4.12"),
            rows,
            source="runs/{unet,procanet}/fold_*/grid_*/metrics.csv",
        ),
        _hyperparameter_figure(config, rows),
        _narrative_4_4_2(config, rows),
    ]
    return section_result("4.4.2", artifacts)


def generate_4_4_3(config):
    metric_paths = [config.runs_root / "final" / model / "metrics.csv" for model in MODEL_KEYS]
    if not all(path.exists() for path in metric_paths):
        return section_result(
            "4.4.3",
            [
                missing_result(
                    config,
                    _spec("Gambar 4.11"),
                    source="runs/final/{unet,procanet}/metrics.csv",
                    note="final training metrics.csv tidak lengkap",
                ),
                _narrative_4_4_3(config),
            ],
        )
    artifacts = [
        _training_curves_figure(config),
        _narrative_4_4_3(config),
    ]
    return section_result("4.4.3", artifacts)


def _model_spec_rows(config) -> list[dict[str, object]]:
    rows = []
    for model in MODEL_KEYS:
        final_config = _read_json(config.runs_root / "final" / model / "config.json")
        base = to_int(final_config.get("base_channels"), 32)
        channels = [base * (2**idx) for idx in range(4)]
        checkpoint = config.runs_root / "final" / model / "final.pt"
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "source_class": "UNet" if model == "unet" else "ProCANet",
                "input_contract": "7-channel tensor" if model == "unet" else "encoder1=7-channel, encoder2=2-channel auxiliary",
                "encoder_depth": 4,
                "base_channels": base,
                "encoder_channels": "-".join(str(channel) for channel in channels),
                "bottleneck_channels": channels[-1] * 2,
                "decoder": "transpose convolution + skip concatenation",
                "attention": "none" if model == "unet" else "progressive self/cross attention pada skip dan bottleneck",
                "normalization": "GroupNorm",
                "activation": "ReLU",
                "output_channels": 1,
                "final_epochs": final_config.get("epochs", ""),
                "final_lr": final_config.get("lr", ""),
                "final_weight_decay": final_config.get("weight_decay", ""),
                "checkpoint_size_mb": fmt_float(checkpoint.stat().st_size / (1024 * 1024), 2) if checkpoint.exists() else "",
            }
        )
    return rows


def _forward_contract_rows(config) -> list[dict[str, object]]:
    source = config.root / "tests" / "test_models.py"
    test_text = source.read_text(encoding="utf-8") if source.exists() else ""
    checks = [
        (
            "U-Net",
            "test_unet_forward_returns_binary_segmentation_logits",
            "torch.randn(2, 7, 64, 64)",
            "(2, 1, 64, 64)",
        ),
        (
            "ProgressiveCrossAttentionBlock",
            "test_progressive_cross_attention_preserves_shape",
            "dua tensor encoder (2, 16, 32, 32)",
            "(2, 16, 32, 32)",
        ),
        (
            "ProCANet",
            "test_procanet_forward_returns_binary_segmentation_logits",
            "dict encoder1=(2, 7, 64, 64), encoder2=(2, 2, 64, 64)",
            "(2, 1, 64, 64)",
        ),
    ]
    rows = []
    for model, test_name, input_shape, output_shape in checks:
        rows.append(
            {
                "component": model,
                "verification_source": f"tests/test_models.py::{test_name}",
                "input_contract": input_shape,
                "expected_output": output_shape,
                "contract_found_in_source": test_name in test_text,
                "status": "source_contract_verified" if test_name in test_text else "missing_test_contract",
                "note": "verifikasi kontrak forward pass dari test source; tidak menjalankan torch/retraining",
            }
        )
    return rows


def _grid_summary_rows(config) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for model in MODEL_KEYS:
        for metrics_path in sorted((config.runs_root / model).glob("fold_*/grid_*/metrics.csv")):
            match = re.search(r"grid_lr_(.+)_wd_(.+)$", metrics_path.parent.name)
            fold_match = re.search(r"fold_(\d+)$", metrics_path.parent.parent.name)
            if not match or not fold_match:
                continue
            rows = read_csv_rows(metrics_path)
            if not rows:
                continue
            best = max(rows, key=lambda row: to_float(row.get("val_iou")))
            key = (model, match.group(1), match.group(2))
            grouped[key].append(
                {
                    "fold": to_int(fold_match.group(1)),
                    "best_epoch": to_int(best.get("epoch")),
                    "best_val_iou": to_float(best.get("val_iou")),
                    "best_val_dice": to_float(best.get("val_dice")),
                    "best_val_loss": to_float(best.get("val_loss")),
                    "epochs_recorded": len(rows),
                    "source": str(metrics_path.relative_to(config.root)),
                }
            )

    rows = []
    for (model, lr, wd), fold_rows in sorted(grouped.items()):
        ious = [float(row["best_val_iou"]) for row in fold_rows]
        dices = [float(row["best_val_dice"]) for row in fold_rows]
        losses = [float(row["best_val_loss"]) for row in fold_rows]
        best_fold = max(fold_rows, key=lambda row: float(row["best_val_iou"]))
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "learning_rate": lr,
                "weight_decay": wd,
                "folds_completed": len(fold_rows),
                "mean_best_val_iou": fmt_float(_mean(ious)),
                "std_best_val_iou": fmt_float(_std(ious)),
                "mean_best_val_dice": fmt_float(_mean(dices)),
                "mean_best_val_loss": fmt_float(_mean(losses)),
                "best_fold": best_fold["fold"],
                "best_fold_epoch": best_fold["best_epoch"],
                "best_fold_val_iou": fmt_float(float(best_fold["best_val_iou"])),
                "source_files": ";".join(str(row["source"]) for row in sorted(fold_rows, key=lambda item: int(item["fold"]))),
            }
        )
    ranked = []
    for model in MODEL_KEYS:
        label = MODEL_LABELS[model]
        model_rows = [row for row in rows if row["model"] == label]
        model_rows.sort(key=lambda row: float(row["mean_best_val_iou"]), reverse=True)
        for rank, row in enumerate(model_rows, start=1):
            ranked.append({"model_rank": rank, **row})
    return ranked


def _architecture_figure(config, model: str):
    spec = _spec("Gambar 4.8" if model == "unet" else "Gambar 4.9")
    setup_style()
    if model == "unet":
        fig, ax = plt.subplots(figsize=(10, 3.8))
        steps = [
            "Input\n7 channel",
            "Encoder\n32-64-128-256",
            "Bottleneck\n512",
            "Decoder\nskip concat",
            "Logit\n1 channel",
        ]
        _draw_linear_blocks(ax, steps, color="#dbeafe")
        ax.text(0.5, 0.78, "skip connection dari setiap level encoder ke decoder", ha="center", fontsize=9)
    else:
        fig, ax = plt.subplots(figsize=(11, 4.8))
        _draw_box(ax, 0.05, 0.68, 0.16, 0.16, "Encoder 1\n7 channel", "#dcfce7")
        _draw_box(ax, 0.05, 0.28, 0.16, 0.16, "Encoder 2\n2 channel", "#fef3c7")
        for idx, x in enumerate((0.30, 0.47, 0.64)):
            _draw_box(ax, x, 0.50, 0.13, 0.16, f"PCA block\nlevel {idx + 1}", "#e0e7ff")
        _draw_box(ax, 0.77, 0.50, 0.14, 0.16, "Decoder\nfused skip", "#fee2e2")
        _draw_box(ax, 0.77, 0.22, 0.14, 0.13, "Logit\n1 channel", "#f3f4f6")
        for start, end, y in ((0.21, 0.30, 0.76), (0.21, 0.30, 0.36), (0.43, 0.47, 0.58), (0.60, 0.64, 0.58), (0.77, 0.77, 0.50)):
            ax.annotate("", xy=(end, y), xytext=(start, y), arrowprops={"arrowstyle": "->", "lw": 1.1})
        ax.annotate("", xy=(0.84, 0.36), xytext=(0.84, 0.50), arrowprops={"arrowstyle": "->", "lw": 1.1})
        ax.text(0.47, 0.84, "progressive cross-attention menggabungkan dua encoder", ha="center", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source=f"training/models/{model}.py;training/models/blocks.py")


def _hyperparameter_figure(config, rows: list[dict[str, object]]):
    spec = _spec("Gambar 4.10")
    variants = sorted({(str(row["learning_rate"]), str(row["weight_decay"])) for row in rows})
    values = {
        (str(row["model"]), str(row["learning_rate"]), str(row["weight_decay"])): float(row["mean_best_val_iou"])
        for row in rows
    }
    labels = [f"lr={lr}\nwd={wd}" for lr, wd in variants]
    setup_style()
    fig, ax = plt.subplots(figsize=(11, 5.0))
    x = list(range(len(variants)))
    width = 0.34
    for offset, model in zip((-width / 2, width / 2), MODEL_KEYS):
        label = MODEL_LABELS[model]
        series = [values.get((label, lr, wd), 0.0) for lr, wd in variants]
        ax.bar([idx + offset for idx in x], series, width=width, label=label)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean best validation IoU")
    ax.set_title("Perbandingan mean validation IoU per kombinasi hyperparameter")
    ax.set_ylim(0, max([float(row["mean_best_val_iou"]) for row in rows] + [1.0]) * 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source="runs/{unet,procanet}/fold_*/grid_*/metrics.csv")


def _training_curves_figure(config):
    spec = _spec("Gambar 4.11")
    metrics = {model: read_csv_rows(config.runs_root / "final" / model / "metrics.csv") for model in MODEL_KEYS}
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    columns = [("train_loss", "Train loss"), ("train_iou", "Train IoU"), ("train_dice", "Train Dice")]
    for ax, (column, title) in zip(axes, columns):
        for model in MODEL_KEYS:
            rows = metrics[model]
            epochs = [to_int(row.get("epoch")) for row in rows]
            values = [to_float(row.get(column)) for row in rows]
            ax.plot(epochs, values, marker="o", markersize=2.5, linewidth=1.4, label=MODEL_LABELS[model])
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Nilai")
    axes[-1].legend()
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source="runs/final/{unet,procanet}/metrics.csv")


def _draw_linear_blocks(ax, labels: list[str], *, color: str) -> None:
    width = 0.14
    gap = 0.055
    x0 = 0.05
    y = 0.42
    for idx, label in enumerate(labels):
        x = x0 + idx * (width + gap)
        _draw_box(ax, x, y, width, 0.22, label, color)
        if idx < len(labels) - 1:
            ax.annotate("", xy=(x + width + gap * 0.78, y + 0.11), xytext=(x + width, y + 0.11), arrowprops={"arrowstyle": "->", "lw": 1.1})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _draw_box(ax, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    patch = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#374151", linewidth=1.0)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def _narrative_4_4_1(config):
    spec = _spec("Narasi 4.4.1")
    text = """
    Spesifikasi arsitektur dibuat ulang dari source model di `training/models`.
    U-Net menggunakan satu encoder 7-channel, bottleneck, dan decoder dengan skip connection.
    ProCANet mempertahankan encoder utama 7-channel dan encoder auxiliary 2-channel,
    lalu menggabungkan representasi melalui progressive cross-attention sebelum decoder.
    """
    return write_text_artifact(config, spec, text, source="training/models/unet.py;training/models/procanet.py;training/models/blocks.py")


def _narrative_4_4_2(config, rows: list[dict[str, object]]):
    spec = _spec("Narasi 4.4.2")
    if rows:
        best = max(rows, key=lambda row: float(row["mean_best_val_iou"]))
        best_line = (
            f"Kombinasi terbaik pada rekap generator adalah {best['model']} "
            f"lr={best['learning_rate']} wd={best['weight_decay']} "
            f"dengan mean best validation IoU {best['mean_best_val_iou']}."
        )
    else:
        best_line = "Rekap grid search tidak dapat dibuat karena metrics.csv tuning tidak ditemukan."
    text = f"""
    Ringkasan tuning dihitung ulang dari setiap `metrics.csv` pada fold spatial CV.
    Nilai yang dipakai adalah epoch dengan validation IoU tertinggi per fold, lalu dirata-ratakan
    per kombinasi learning rate dan weight decay. {best_line}
    """
    return write_text_artifact(config, spec, text, source="runs/{unet,procanet}/fold_*/grid_*/metrics.csv")


def _narrative_4_4_3(config):
    spec = _spec("Narasi 4.4.3")
    text = """
    Kurva stabilitas training dibuat dari `runs/final/*/metrics.csv`, yaitu log final model
    yang dipakai untuk evaluasi BAB 4. Generator tidak menjalankan training ulang; ia hanya
    membaca loss, IoU, dan Dice per epoch dari artefak final yang sudah tersedia.
    """
    return write_text_artifact(config, spec, text, source="runs/final/{unet,procanet}/metrics.csv")


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
