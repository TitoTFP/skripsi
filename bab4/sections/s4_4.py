from __future__ import annotations

import math
import re
from collections import defaultdict

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
            _grid_table_rows(rows),
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
    return [
        {
            "komponen": "Jenis model",
            "u_net": "Single encoder-decoder",
            "procanet": "Dual encoder single decoder dengan Progressive Cross-Attention",
        },
        {
            "komponen": "Input utama",
            "u_net": "7 channel: VV, VH, Hue, Saturation, Value, Slope, HAND",
            "procanet": "Encoder 1: 7 channel penuh; Encoder 2: VV, VH",
        },
        {"komponen": "Output akhir", "u_net": "1 channel logit", "procanet": "1 channel logit"},
        {"komponen": "Base channels", "u_net": 32, "procanet": 32},
        {"komponen": "Kedalaman encoder", "u_net": "4 level", "procanet": "4 level pada masing-masing encoder"},
        {"komponen": "Channel encoder", "u_net": "32, 64, 128, 256", "procanet": "32, 64, 128, 256 pada kedua encoder"},
        {
            "komponen": "Bottleneck",
            "u_net": "ConvBlock 256 -> 512",
            "procanet": "ConvBlock 256 -> 512 pada tiap encoder, lalu PCAB bottleneck",
        },
        {
            "komponen": "Blok konvolusi",
            "u_net": "Conv2d 3x3 + GroupNorm + ReLU, dua kali",
            "procanet": "Conv2d 3x3 + GroupNorm + ReLU, dua kali",
        },
        {
            "komponen": "Downsampling",
            "u_net": "MaxPool2d 2x2 pada setiap level encoder",
            "procanet": "MaxPool2d 2x2 pada kedua encoder",
        },
        {
            "komponen": "Upsampling/decoder",
            "u_net": "ConvTranspose2d 2x2 + ConvBlock",
            "procanet": "ConvTranspose2d 2x2 + ConvBlock memakai fused skips",
        },
        {
            "komponen": "Mekanisme fusi",
            "u_net": "Fusi langsung 7-channel sejak input dan skip connection encoder-decoder",
            "procanet": "ProgressiveCrossAttentionBlock pada skip features dan bottleneck",
        },
    ]


def _forward_contract_rows(config) -> list[dict[str, object]]:
    return [
        {
            "model": "U-Net",
            "input_uji": "(1, 7, 128, 128)",
            "output": "(1, 1, 128, 128)",
            "tipe_output": "Logit",
            "jumlah_parameter": 7764193,
        },
        {
            "model": "ProCANet",
            "input_uji": "Encoder 1: (1, 7, 128, 128); Encoder 2: (1, 2, 128, 128)",
            "output": "(1, 1, 128, 128)",
            "tipe_output": "Logit",
            "jumlah_parameter": 25052705,
        },
    ]


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


def _grid_table_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_model_variant = {
        (str(row["model"]), str(row["learning_rate"]), str(row["weight_decay"])): row
        for row in rows
    }
    variants = [
        ("1e-4", "1e-4"),
        ("1e-4", "1e-5"),
        ("5e-5", "1e-4"),
        ("5e-5", "1e-5"),
        ("1e-5", "1e-4"),
        ("1e-5", "1e-5"),
    ]
    table_rows = []
    for idx, (lr, wd) in enumerate(variants, start=1):
        unet = by_model_variant.get(("U-Net", lr, wd), {})
        procanet = by_model_variant.get(("ProCANet", lr, wd), {})
        table_rows.append(
            {
                "id_variasi": idx,
                "learning_rate": lr,
                "weight_decay": wd,
                "u_net_mean_loss": _fixed_4(unet.get("mean_best_val_loss")),
                "u_net_mean_iou": _fixed_4(unet.get("mean_best_val_iou")),
                "procanet_mean_loss": _fixed_4(procanet.get("mean_best_val_loss")),
                "procanet_mean_iou": _fixed_4(procanet.get("mean_best_val_iou")),
            }
        )
    return table_rows


def _fixed_4(value: object) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.4f}"


def _architecture_figure(config, model: str):
    spec = _spec("Gambar 4.8" if model == "unet" else "Gambar 4.9")
    setup_style()
    if model == "unet":
        fig, ax = plt.subplots(figsize=(11, 4.2))
        xs = [0.03, 0.18, 0.32, 0.46, 0.60, 0.74, 0.88]
        labels = [
            "Input\n7 x H x W",
            "Encoder\nConvBlock 32\n+ MaxPool",
            "Encoder\nConvBlock 64\n+ MaxPool",
            "Encoder\nConvBlock 128\n+ MaxPool",
            "Encoder\nConvBlock 256\n+ MaxPool",
            "Bottleneck\nConvBlock\n256 -> 512",
            "Decoder\nUpConv + skip\n512 -> 32",
        ]
        for x, label in zip(xs, labels):
            _draw_box(ax, x, 0.38, 0.10, 0.16, label, "#e5f0f3")
        _draw_box(ax, 0.88, 0.16, 0.10, 0.12, "Output\n1 logit\nH x W", "#f6dccb")
        for x0, x1 in zip(xs[:-1], xs[1:]):
            _arrow(ax, x0 + 0.10, 0.46, x1, 0.46)
        _arrow(ax, 0.93, 0.38, 0.93, 0.28)
        for x0, x1, y in ((0.23, 0.83, 0.70), (0.37, 0.79, 0.64), (0.51, 0.75, 0.58), (0.65, 0.74, 0.54)):
            ax.annotate("", xy=(x1, y), xytext=(x0, 0.56), arrowprops={"arrowstyle": "-", "lw": 0.8, "linestyle": "--", "color": "#5f7f7f"})
        ax.text(0.5, 0.86, "Diagram Implementasi U-Net Aktual", ha="center", fontsize=10)
        ax.text(0.5, 0.76, "Skip connection dari 4 level encoder ke decoder", ha="center", fontsize=8, color="#496b6b")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(11, 5.0))
        ax.text(0.5, 0.92, "Diagram Implementasi ProCANet Aktual", ha="center", fontsize=10)
        ax.text(0.5, 0.83, "Encoder 1 menerima 7 channel penuh; Encoder 2 menerima SAR VV/VH", ha="center", fontsize=8)
        xcols = [0.04, 0.18, 0.31, 0.44, 0.57]
        y1, y2 = 0.66, 0.28
        _draw_box(ax, xcols[0], y1, 0.11, 0.12, "Input Encoder 1\n7ch multisensor", "#e5f0f3")
        _draw_box(ax, xcols[0], y2, 0.11, 0.12, "Input Encoder 2\n2ch SAR VV/VH", "#e5f0f3")
        for idx, x in enumerate(xcols[1:], start=1):
            ch = 32 * (2 ** (idx - 1))
            _draw_box(ax, x, y1, 0.10, 0.12, f"Enc1\nConvBlock {ch}\n+ pool", "#dceee3")
            _draw_box(ax, x, y2, 0.10, 0.12, f"Enc2\nConvBlock {ch}\n+ pool", "#dceee3")
            _draw_box(ax, x + 0.02, 0.47, 0.07, 0.09, f"PCAB\nskip {ch}", "#c8ead7")
        _draw_box(ax, 0.72, y1, 0.10, 0.12, "Bottleneck 1\n256 -> 512", "#f4d6c8")
        _draw_box(ax, 0.72, y2, 0.10, 0.12, "Bottleneck 2\n256 -> 512", "#f4d6c8")
        _draw_box(ax, 0.84, 0.46, 0.08, 0.11, "PCAB\nbottleneck\n512", "#c8ead7")
        _draw_box(ax, 0.94, 0.46, 0.07, 0.11, "Decoder\n+ fused skips\n1 logit", "#f4d6c8")
        for row_y in (y1, y2):
            for x0, x1 in zip(xcols[:-1], xcols[1:]):
                _arrow(ax, x0 + 0.11, row_y + 0.06, x1, row_y + 0.06)
            _arrow(ax, xcols[-1] + 0.10, row_y + 0.06, 0.72, row_y + 0.06)
        _arrow(ax, 0.82, y1 + 0.06, 0.84, 0.52)
        _arrow(ax, 0.82, y2 + 0.06, 0.84, 0.52)
        _arrow(ax, 0.92, 0.52, 0.94, 0.52)
        ax.text(0.5, 0.12, "ProgressiveCrossAttentionBlock melakukan fusi skip features dan bottleneck sebelum decoder", ha="center", fontsize=8, color="#496b6b")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source=f"training/models/{model}.py;training/models/blocks.py")


def _hyperparameter_figure(config, rows: list[dict[str, object]]):
    spec = _spec("Gambar 4.10")
    variants = [
        ("1e-4", "1e-4"),
        ("1e-4", "1e-5"),
        ("5e-5", "1e-5"),
        ("5e-5", "1e-4"),
        ("1e-5", "1e-4"),
        ("1e-5", "1e-5"),
    ]
    values = {
        (str(row["model"]), str(row["learning_rate"]), str(row["weight_decay"])): float(row["mean_best_val_iou"])
        for row in rows
    }
    labels = [f"{lr} / {wd}" for lr, wd in variants]
    setup_style()
    fig, ax = plt.subplots(figsize=(11, 5.0))
    x = list(range(len(variants)))
    width = 0.34
    for offset, model in zip((-width / 2, width / 2), ("procanet", "unet")):
        label = MODEL_LABELS[model]
        series = [values.get((label, lr, wd), 0.0) for lr, wd in variants]
        ax.bar([idx + offset for idx in x], series, width=width, label=label)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Learning rate / weight decay")
    ax.set_ylabel("Mean validation IoU")
    ax.set_ylim(0, 0.7)
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.25)
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source="runs/{unet,procanet}/fold_*/grid_*/metrics.csv")


def _training_curves_figure(config):
    spec = _spec("Gambar 4.11")
    metrics = {model: read_csv_rows(config.runs_root / "final" / model / "metrics.csv") for model in MODEL_KEYS}
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    panels = [
        ("loss", "Loss", "train_loss", "val_loss"),
        ("iou", "IoU", "train_iou", "val_iou"),
        ("dice", "Dice/F1", "train_dice", "val_dice"),
    ]
    for idx, (ax, (_, title, train_col, val_col)) in enumerate(zip(axes.ravel()[:3], panels)):
        for model in MODEL_KEYS:
            rows = metrics[model]
            epochs = [to_int(row.get("epoch")) for row in rows]
            ax.plot(epochs, [to_float(row.get(train_col)) for row in rows], linewidth=1.2, label=f"{MODEL_LABELS[model]} train")
            if rows and val_col in rows[0]:
                ax.plot(epochs, [to_float(row.get(val_col)) for row in rows], linestyle="--", linewidth=1.2, label=f"{MODEL_LABELS[model]} validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
        ax.text(0.5, -0.18, f"({chr(97 + idx)})", transform=ax.transAxes, ha="center", va="top", fontsize=9)
    ax_lr = axes.ravel()[3]
    for model in MODEL_KEYS:
        rows = metrics[model]
        epochs = [to_int(row.get("epoch")) for row in rows]
        ax_lr.plot(epochs, [to_float(row.get("lr")) for row in rows], linewidth=1.2, label=MODEL_LABELS[model])
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning rate")
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_lr.grid(alpha=0.25)
    ax_lr.text(0.5, -0.18, "(d)", transform=ax_lr.transAxes, ha="center", va="top", fontsize=9)
    axes.ravel()[0].legend(loc="upper right", fontsize=7)
    ax_lr.legend(loc="upper right", fontsize=7)
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


def _arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 0.9})


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
