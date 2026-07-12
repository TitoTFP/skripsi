from __future__ import annotations

import math
import json
import re
from collections import defaultdict

import matplotlib.pyplot as plt

from bab4.artifacts import ALL_ARTIFACTS
from bab4.common import SPATIAL_CV_FOLDS, MODEL_KEYS, MODEL_LABELS, fmt_float, read_csv_rows, to_float, to_int
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
    selected_source = _selected_training_source(config, "config.json")
    artifacts = [
        write_table(
            config,
            _spec("Tabel 4.10"),
            _model_spec_rows(config),
            source=f"training/models/unet.py;training/models/procanet.py;training/models/blocks.py;{selected_source}",
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
    source = "runs/{unet,procanet}/fold_*/grid_*/metrics.csv"
    try:
        selected_variants, metric_paths = _selected_cv_metric_paths(config)
        source = _metric_paths_source(config, metric_paths)
        artifacts = [
            _training_curves_figure(config, metric_paths, source),
            _narrative_4_4_3(config, selected_variants, source),
        ]
    except (OSError, ValueError) as exc:
        return section_result(
            "4.4.3",
            [
                missing_result(
                    config,
                    _spec("Gambar 4.11"),
                    source=source,
                    note=str(exc),
                ),
                _narrative_4_4_3(config, None, source, unavailable_note=str(exc)),
            ],
        )
    return section_result("4.4.3", artifacts)


def _model_spec_rows(config) -> list[dict[str, object]]:
    configs = {}
    for model in MODEL_KEYS:
        with (config.selected_training_run(model) / "config.json").open(encoding="utf-8") as handle:
            configs[model] = json.load(handle)
    rows = [
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
    rows.extend(
        [
            {
                "komponen": "Checkpoint evaluasi",
                "u_net": str(config.selected_training_run("unet").relative_to(config.root) / "best.pt"),
                "procanet": str(config.selected_training_run("procanet").relative_to(config.root) / "best.pt"),
            },
            {
                "komponen": "Learning rate checkpoint",
                "u_net": configs["unet"].get("lr"),
                "procanet": configs["procanet"].get("lr"),
            },
            {
                "komponen": "Batch size / gradient accumulation",
                "u_net": f"{configs['unet'].get('batch_size')} / {configs['unet'].get('gradient_accumulation_steps')}",
                "procanet": f"{configs['procanet'].get('batch_size')} / {configs['procanet'].get('gradient_accumulation_steps')}",
            },
        ]
    )
    return rows


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
                "mean_best_val_iou_raw": _mean(ious),
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


def _training_curves_figure(config, metric_paths: dict[str, list], source: str):
    spec = _spec("Gambar 4.11")
    fold_rows = {
        model: [read_csv_rows(path) for path in metric_paths[model]]
        for model in MODEL_KEYS
    }
    aggregates = {
        model: {
            metric: _aggregate_fold_metric(fold_rows[model], metric)
            for metric in (
                "train_loss",
                "val_loss",
                "train_iou",
                "val_iou",
                "train_dice",
                "val_dice",
                "lr",
            )
        }
        for model in MODEL_KEYS
    }
    # Purple and orange remain distinct for common colour-vision deficiencies
    # and preserve contrast in the translucent standard-deviation bands.
    colors = {"unet": "#6A3D9A", "procanet": "#E66101"}

    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8))
    # Panel labels sit below each axes, so they need dedicated space between
    # rows rather than intruding into the plots beneath them.
    fig.subplots_adjust(hspace=0.48, wspace=0.34)
    panels = [
        ("Loss", "train_loss", "val_loss"),
        ("IoU", "train_iou", "val_iou"),
        ("Dice/F1", "train_dice", "val_dice"),
    ]
    for idx, (ax, (title, train_metric, val_metric)) in enumerate(zip(axes.ravel()[:3], panels)):
        for model in MODEL_KEYS:
            color = colors[model]
            _plot_aggregate_series(
                ax,
                aggregates[model][train_metric],
                color=color,
                linestyle="-",
                label=f"{MODEL_LABELS[model]} train",
            )
            _plot_aggregate_series(
                ax,
                aggregates[model][val_metric],
                color=color,
                linestyle="--",
                label=f"{MODEL_LABELS[model]} validation",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
        ax.text(0.5, -0.18, f"({chr(97 + idx)})", transform=ax.transAxes, ha="center", va="top", fontsize=20)

    ax_lr = axes.ravel()[3]
    ax_active = ax_lr.twinx()
    lr_handles = []
    active_handles = []
    for model in MODEL_KEYS:
        color = colors[model]
        lr_handle = _plot_aggregate_series(
            ax_lr,
            aggregates[model]["lr"],
            color=color,
            linestyle="-",
            label=f"{MODEL_LABELS[model]} LR",
        )
        if lr_handle is not None:
            lr_handles.append(lr_handle)
        active_points = aggregates[model]["lr"]
        active_handle = ax_active.step(
            [int(point["epoch"]) for point in active_points],
            [int(point["active_folds"]) for point in active_points],
            where="post",
            color=color,
            linestyle=":",
            linewidth=1.3,
            alpha=0.9,
            label=f"{MODEL_LABELS[model]} n",
        )[0]
        active_handles.append(active_handle)

    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning rate")
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_lr.grid(alpha=0.25)
    ax_active.set_ylabel("Fold aktif (n)")
    ax_active.set_ylim(0.5, len(SPATIAL_CV_FOLDS) + 0.5)
    ax_active.set_yticks(range(1, len(SPATIAL_CV_FOLDS) + 1))
    ax_active.grid(False)
    ax_lr.text(0.5, -0.18, "(d)", transform=ax_lr.transAxes, ha="center", va="top", fontsize=20)
    axes.ravel()[0].legend(loc="upper right", fontsize=7)
    ax_lr.legend(handles=lr_handles + active_handles, loc="upper right", fontsize=6.5, ncol=2)
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source=source)


def _plot_aggregate_series(ax, points: list[dict[str, object]], *, color: str, linestyle: str, label: str):
    visible = [point for point in points if point["mean"] is not None]
    if not visible:
        return None
    epochs = [int(point["epoch"]) for point in visible]
    means = [float(point["mean"]) for point in visible]
    stds = [float(point["std"]) for point in visible]
    line = ax.plot(epochs, means, color=color, linestyle=linestyle, linewidth=1.5, label=label)[0]
    ax.fill_between(
        epochs,
        [mean - std for mean, std in zip(means, stds)],
        [mean + std for mean, std in zip(means, stds)],
        color=color,
        alpha=0.10,
        linewidth=0,
    )
    return line


def _aggregate_fold_metric(fold_rows: list[list[dict[str, str]]], metric: str) -> list[dict[str, object]]:
    """Summarize one metric without extending a stopped fold into later epochs."""
    values_by_epoch: dict[int, list[float]] = defaultdict(list)
    for rows in fold_rows:
        observed_epochs: set[int] = set()
        for row in rows:
            epoch = to_int(row.get("epoch"))
            if epoch <= 0:
                continue
            if epoch in observed_epochs:
                raise ValueError(f"epoch duplikat pada metrics.csv: {epoch}")
            observed_epochs.add(epoch)
            value = row.get(metric)
            if value in (None, ""):
                raise ValueError(f"kolom {metric} tidak lengkap pada epoch {epoch}")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"nilai {metric} tidak valid pada epoch {epoch}") from exc
            if not math.isfinite(numeric_value):
                raise ValueError(f"nilai {metric} tidak valid pada epoch {epoch}")
            values_by_epoch[epoch].append(numeric_value)

    summaries = []
    for epoch, values in sorted(values_by_epoch.items()):
        active_folds = len(values)
        mean = _mean(values) if active_folds >= 2 else None
        std = _std(values) if active_folds >= 2 else None
        summaries.append(
            {
                "epoch": epoch,
                "active_folds": active_folds,
                "mean": mean,
                "std": std,
            }
        )
    return summaries


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
    text = f"""
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


def _narrative_4_4_3(config, selected_variants, source: str, *, unavailable_note: str | None = None):
    spec = _spec("Narasi 4.4.3")
    if unavailable_note is not None:
        text = f"""
        Kurva stabilitas training 5-fold spatial cross-validation belum dapat dibuat karena
        {unavailable_note}. Generator tidak menggantikan fold yang tidak tersedia dengan
        nilai epoch terakhir agar tidak menghasilkan kesan stabilitas yang keliru.
        """
        return write_text_artifact(config, spec, text, source=source, note=unavailable_note)

    assert selected_variants is not None
    variant_description = "; ".join(
        f"{MODEL_LABELS[model]} memakai lr={selected_variants[model]['learning_rate']} "
        f"dan wd={selected_variants[model]['weight_decay']}"
        for model in MODEL_KEYS
    )
    text = f"""
    Kurva stabilitas training dibuat dari lima `metrics.csv` spatial cross-validation pada
    konfigurasi hyperparameter dengan mean best validation IoU tertinggi untuk masing-masing
    model. {variant_description}. Pemilihan dan kurva ini tidak memakai metrik evaluasi Aceh
    Utara; wilayah tersebut tetap menjadi data uji independen pada tahap evaluasi akhir.

    Pada setiap epoch, loss, IoU, Dice, dan learning rate diringkas sebagai mean serta pita
    plus/minus satu sample standard deviation dari fold yang masih aktif. Early stopping
    menyebabkan jumlah fold aktif dapat berkurang; garis putus-putus pada sumbu kanan panel
    learning rate menunjukkan nilai n tersebut. Tidak ada nilai yang diteruskan setelah suatu
    fold berhenti. Mean dan pita standard deviation dihentikan saat n kurang dari dua, sehingga
    bagian akhir tidak diklaim sebagai agregasi lima fold.
    """
    return write_text_artifact(config, spec, text, source=source)


def _selected_cv_metric_paths(config) -> tuple[dict[str, dict[str, object]], dict[str, list]]:
    selected_variants = _best_complete_cv_variants(_grid_summary_rows(config))
    metric_paths: dict[str, list] = {}
    missing_paths = []
    for model in MODEL_KEYS:
        variant = selected_variants[model]
        variant_dir = f"grid_lr_{variant['learning_rate']}_wd_{variant['weight_decay']}"
        paths = [
            config.runs_root / model / f"fold_{fold}" / variant_dir / "metrics.csv"
            for fold in range(len(SPATIAL_CV_FOLDS))
        ]
        metric_paths[model] = paths
        missing_paths.extend(path for path in paths if not path.exists())
    if missing_paths:
        missing = ", ".join(str(path.relative_to(config.root)) for path in missing_paths)
        raise ValueError(f"metrics.csv konfigurasi terbaik tidak lengkap untuk 5-fold CV: {missing}")
    return selected_variants, metric_paths


def _best_complete_cv_variants(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    selected = {}
    expected_fold_count = len(SPATIAL_CV_FOLDS)
    for model in MODEL_KEYS:
        candidates = [
            row
            for row in rows
            if row["model"] == MODEL_LABELS[model]
            and to_int(row.get("folds_completed")) == expected_fold_count
        ]
        if not candidates:
            raise ValueError(f"tidak ada konfigurasi lengkap {expected_fold_count}-fold untuk {MODEL_LABELS[model]}")
        selected[model] = max(
            candidates,
            key=lambda row: float(row.get("mean_best_val_iou_raw", row["mean_best_val_iou"])),
        )
    return selected


def _metric_paths_source(config, metric_paths: dict[str, list]) -> str:
    return ";".join(
        str(path.relative_to(config.root))
        for model in MODEL_KEYS
        for path in metric_paths[model]
    )


def _selected_training_source(config, filename: str) -> str:
    return ";".join(
        str((config.selected_training_run(model) / filename).relative_to(config.root))
        for model in MODEL_KEYS
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
