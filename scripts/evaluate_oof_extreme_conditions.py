"""Evaluate difficult flood-segmentation conditions with out-of-fold predictions.

The command deliberately keeps model inference separate from the BAB 4 report
generator.  It creates reusable OOF mosaics and machine-readable summaries;
the report only consumes these completed artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from osgeo import gdal

from scripts.preprocessing_utils import CV_REGIONS, SPATIAL_CV_FOLDS


ROOT = Path(__file__).resolve().parents[1]
MODEL_KEYS = ("unet", "procanet")
MODEL_LABELS = {"unet": "U-Net", "procanet": "ProCANet"}
CONDITIONS = (
    ("s2_empty", "Sentinel-2 kosong/hampir kosong"),
    ("topography_radar_shadow", "Topografi sulit/kandidat radar shadow"),
    ("permanent_water", "Badan air permanen/sungai"),
)
HISTOGRAM_BINS = 65536
BLOCK_ROWS = 512


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "oof_extreme_conditions")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true", help="Run inference again even when a complete OOF mosaic exists.")
    parser.add_argument("--skip-inference", action="store_true", help="Only aggregate already-created OOF mosaics.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    selected = select_best_variants(args.runs_root)
    if not args.skip_inference:
        run_oof_inference(args, selected)
    validate_oof_mosaics(args.output_root)
    thresholds = global_radar_thresholds(args.dataset_root)
    per_region = evaluate_regions(args.dataset_root, args.output_root, selected, thresholds)
    micro = micro_aggregate(per_region)
    selected_tiles = select_example_tiles(args.dataset_root, thresholds)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_root / "per_region_metrics.csv", per_region)
    write_rows(args.output_root / "micro_metrics.csv", micro)
    write_rows(args.output_root / "selected_tiles.csv", selected_tiles)
    provenance = {
        "threshold": args.threshold,
        "models": selected,
        "folds": {str(index): list(regions) for index, regions in enumerate(SPATIAL_CV_FOLDS)},
        "conditions": {
            "s2_empty": "union of effective pixels from tiles with s2_valid_ratio <= 0.01",
            "topography_radar_shadow": "effective & slope_degrees > 20 & hand_meters > 40 & (vv_norm <= p20_vv_norm | vh_norm <= p20_vh_norm)",
            "permanent_water": "effective & water_river_mask & ~label_flood_binary",
        },
        "radar_percentiles": thresholds,
        "radar_percentile_method": f"streaming histogram ({HISTOGRAM_BINS} bins) across all CV effective pixels",
        "selected_tiles": selected_tiles,
    }
    (args.output_root / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"OOF extreme-condition results written to {args.output_root}")


def select_best_variants(runs_root: Path) -> dict[str, dict[str, object]]:
    """Select the complete five-fold grid variant with the highest raw mean IoU."""
    selected: dict[str, dict[str, object]] = {}
    pattern = re.compile(r"grid_lr_(.+)_wd_(.+)$")
    for model in MODEL_KEYS:
        grouped: dict[str, dict[int, float]] = defaultdict(dict)
        for metrics_path in (runs_root / model).glob("fold_*/grid_*/metrics.csv"):
            match = pattern.fullmatch(metrics_path.parent.name)
            fold_match = re.fullmatch(r"fold_(\d+)", metrics_path.parent.parent.name)
            if match is None or fold_match is None:
                continue
            values = _best_iou_values(metrics_path)
            if values:
                grouped[metrics_path.parent.name][int(fold_match.group(1))] = max(values)
        candidates = []
        for variant, folds in grouped.items():
            if set(folds) != set(range(len(SPATIAL_CV_FOLDS))):
                continue
            if not all((runs_root / model / f"fold_{fold}" / variant / "best.pt").exists() for fold in folds):
                continue
            candidates.append((sum(folds.values()) / len(folds), variant, folds))
        if not candidates:
            raise FileNotFoundError(f"no complete five-fold best checkpoint series for {model}")
        mean_iou, variant, folds = max(candidates, key=lambda item: (item[0], item[1]))
        match = pattern.fullmatch(variant)
        assert match is not None
        selected[model] = {
            "variant": variant,
            "lr": match.group(1),
            "weight_decay": match.group(2),
            "mean_best_val_iou": mean_iou,
            "fold_best_val_iou": {str(fold): folds[fold] for fold in sorted(folds)},
        }
    return selected


def _best_iou_values(metrics_path: Path) -> list[float]:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = []
    for row in rows:
        value = row.get("best_val_iou") or row.get("val_iou")
        try:
            numeric = float(value) if value is not None else math.nan
        except ValueError:
            numeric = math.nan
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def run_oof_inference(args: argparse.Namespace, selected: dict[str, dict[str, object]]) -> None:
    for model in MODEL_KEYS:
        variant = str(selected[model]["variant"])
        for fold, regions in enumerate(SPATIAL_CV_FOLDS):
            output_dir = args.output_root / model / f"fold_{fold}"
            expected = [output_dir / "geotiff" / f"{region}_prediction.tif" for region in regions]
            if not args.overwrite and all(path.exists() for path in expected):
                continue
            checkpoint = args.runs_root / model / f"fold_{fold}" / variant / "best.pt"
            command = [
                sys.executable,
                "-m",
                "scripts.infer_segmentation",
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--threshold",
                str(args.threshold),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                str(args.device),
                "--write-geotiff",
                "--no-save-predictions",
            ]
            for region in regions:
                command.extend(("--region", region))
            print(f"infer {MODEL_LABELS[model]} fold {fold}: {', '.join(regions)}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)


def validate_oof_mosaics(output_root: Path) -> None:
    missing = []
    for fold, regions in enumerate(SPATIAL_CV_FOLDS):
        for model in MODEL_KEYS:
            for region in regions:
                path = output_root / model / f"fold_{fold}" / "geotiff" / f"{region}_prediction.tif"
                if not path.exists():
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError("missing OOF mosaics:\n" + "\n".join(missing))


def global_radar_thresholds(dataset_root: Path) -> dict[str, float | int]:
    histograms = {"vv_norm": np.zeros(HISTOGRAM_BINS, dtype=np.int64), "vh_norm": np.zeros(HISTOGRAM_BINS, dtype=np.int64)}
    total = 0
    for region in CV_REGIONS:
        feature_dir = dataset_root / "features_preprocessed" / region
        label_dir = dataset_root / "labels_unosat_rasterized" / region
        label_ds = open_raster(label_dir / "label_valid_mask.tif")
        feature_ds = open_raster(feature_dir / "feature_valid_mask.tif")
        vv_ds = open_raster(feature_dir / "vv_norm.tif")
        vh_ds = open_raster(feature_dir / "vh_norm.tif")
        for yoff, ysize in block_windows(label_ds):
            effective = read_window(label_ds, yoff, ysize).astype(bool) & read_window(feature_ds, yoff, ysize).astype(bool)
            count = int(effective.sum())
            if not count:
                continue
            total += count
            for key, ds in (("vv_norm", vv_ds), ("vh_norm", vh_ds)):
                values = np.clip(read_window(ds, yoff, ysize)[effective], 0.0, 1.0)
                bins = np.minimum((values * HISTOGRAM_BINS).astype(np.int64), HISTOGRAM_BINS - 1)
                histograms[key] += np.bincount(bins, minlength=HISTOGRAM_BINS)
        close_datasets(label_ds, feature_ds, vv_ds, vh_ds)
    if not total:
        raise ValueError("no effective CV pixels for radar percentile")
    return {
        "p20_vv_norm": histogram_quantile(histograms["vv_norm"], 0.20),
        "p20_vh_norm": histogram_quantile(histograms["vh_norm"], 0.20),
        "effective_pixels": total,
    }


def histogram_quantile(histogram: np.ndarray, quantile: float) -> float:
    target = max(0, int(math.ceil(int(histogram.sum()) * quantile)) - 1)
    index = int(np.searchsorted(np.cumsum(histogram), target, side="right"))
    return (index + 1) / HISTOGRAM_BINS


def evaluate_regions(
    dataset_root: Path,
    output_root: Path,
    selected: dict[str, dict[str, object]],
    thresholds: dict[str, float | int],
) -> list[dict[str, object]]:
    rows = []
    fold_for_region = {region: fold for fold, regions in enumerate(SPATIAL_CV_FOLDS) for region in regions}
    for region in CV_REGIONS:
        fold = fold_for_region[region]
        s2_empty = s2_empty_union(dataset_root, region)
        for model in MODEL_KEYS:
            stats = condition_stats_for_region(dataset_root, output_root, region, fold, model, thresholds, s2_empty)
            for condition, label in CONDITIONS:
                summary = metric_summary(stats[condition])
                rows.append(
                    {
                        "condition": condition,
                        "condition_label": label,
                        "model": MODEL_LABELS[model],
                        "model_key": model,
                        "fold": fold,
                        "region": region,
                        "variant": selected[model]["variant"],
                        **summary,
                    }
                )
        del s2_empty
    return rows


def condition_stats_for_region(
    dataset_root: Path,
    output_root: Path,
    region: str,
    fold: int,
    model: str,
    thresholds: dict[str, float | int],
    s2_empty: np.ndarray,
) -> dict[str, dict[str, int]]:
    feature_dir = dataset_root / "features_preprocessed" / region
    label_dir = dataset_root / "labels_unosat_rasterized" / region
    datasets = {
        "label": open_raster(label_dir / "label_flood_binary.tif"),
        "label_valid": open_raster(label_dir / "label_valid_mask.tif"),
        "water": open_raster(label_dir / "label_water_river_mask.tif"),
        "feature_valid": open_raster(feature_dir / "feature_valid_mask.tif"),
        "slope": open_raster(feature_dir / "slope_degrees.tif"),
        "hand": open_raster(feature_dir / "hand_meters.tif"),
        "vv": open_raster(feature_dir / "vv_norm.tif"),
        "vh": open_raster(feature_dir / "vh_norm.tif"),
        "prediction": open_raster(output_root / model / f"fold_{fold}" / "geotiff" / f"{region}_prediction.tif"),
    }
    result = {condition: new_stats() for condition, _ in CONDITIONS}
    try:
        for yoff, ysize in block_windows(datasets["label"]):
            effective = read_window(datasets["label_valid"], yoff, ysize).astype(bool) & read_window(datasets["feature_valid"], yoff, ysize).astype(bool)
            label = read_window(datasets["label"], yoff, ysize).astype(bool)
            water = read_window(datasets["water"], yoff, ysize).astype(bool)
            prediction = read_window(datasets["prediction"], yoff, ysize)
            predicted = (prediction == 0) | (prediction == 1)
            masks = {
                "s2_empty": effective & s2_empty[yoff : yoff + ysize],
                "topography_radar_shadow": effective
                & (read_window(datasets["slope"], yoff, ysize) > 20.0)
                & (read_window(datasets["hand"], yoff, ysize) > 40.0)
                & ((read_window(datasets["vv"], yoff, ysize) <= float(thresholds["p20_vv_norm"])) | (read_window(datasets["vh"], yoff, ysize) <= float(thresholds["p20_vh_norm"]))),
                "permanent_water": effective & water & ~label,
            }
            for condition, mask in masks.items():
                update_stats(result[condition], mask, label, prediction, predicted)
    finally:
        close_datasets(*datasets.values())
    return result


def new_stats() -> dict[str, int]:
    return {"condition_pixels": 0, "evaluated_pixels": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}


def update_stats(stats: dict[str, int], condition: np.ndarray, label: np.ndarray, prediction: np.ndarray, predicted: np.ndarray) -> None:
    stats["condition_pixels"] += int(condition.sum())
    mask = condition & predicted
    stats["evaluated_pixels"] += int(mask.sum())
    truth = label[mask]
    pred = prediction[mask].astype(bool)
    stats["tp"] += int(np.count_nonzero(pred & truth))
    stats["tn"] += int(np.count_nonzero(~pred & ~truth))
    stats["fp"] += int(np.count_nonzero(pred & ~truth))
    stats["fn"] += int(np.count_nonzero(~pred & truth))


def metric_summary(stats: dict[str, int]) -> dict[str, object]:
    tp, tn, fp, fn = (stats[key] for key in ("tp", "tn", "fp", "fn"))
    def safe(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None
    has_positive_label = (tp + fn) > 0
    return {
        **stats,
        # A negative-only condition answers a false-positive question.  Report
        # positive-class metrics as not applicable rather than implying a
        # measured zero score.
        "iou": safe(tp, tp + fp + fn) if has_positive_label else None,
        "dice": safe(2 * tp, 2 * tp + fp + fn) if has_positive_label else None,
        "precision": safe(tp, tp + fp) if has_positive_label else None,
        "recall": safe(tp, tp + fn) if has_positive_label else None,
        "specificity": safe(tn, tn + fp),
        "fpr": safe(fp, fp + tn),
    }


def micro_aggregate(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, int]] = {}
    for row in rows:
        key = (str(row["condition"]), str(row["condition_label"]), str(row["model"]))
        stats = grouped.setdefault(key, new_stats())
        for name in new_stats():
            stats[name] += int(row[name])
    result = []
    for (condition, label, model), stats in grouped.items():
        result.append({"condition": condition, "condition_label": label, "model": model, **metric_summary(stats)})
    return result


def s2_empty_union(dataset_root: Path, region: str) -> np.ndarray:
    stack = open_raster(dataset_root / "features_preprocessed" / region / "stack_7ch.tif")
    mask = np.zeros((stack.RasterYSize, stack.RasterXSize), dtype=bool)
    stack = None
    for tile_path in sorted((dataset_root / "tiles" / "7ch" / "by_region" / region).glob("*.npz")):
        with np.load(tile_path, allow_pickle=False) as tile:
            effective = np.squeeze(tile["valid_mask"]).astype(bool) & np.squeeze(tile["feature_valid_mask"]).astype(bool)
            if not effective.any() or float(np.squeeze(tile["s2_valid_mask"])[effective].mean()) > 0.01:
                continue
            row, col = int(tile["row"].item()), int(tile["col"].item())
            height, width = effective.shape
            mask[row : row + height, col : col + width] |= effective
    return mask


def select_example_tiles(dataset_root: Path, thresholds: dict[str, float | int]) -> list[dict[str, object]]:
    choices: dict[str, tuple[tuple[int, int, str], dict[str, object]]] = {}
    fold_for_region = {region: fold for fold, regions in enumerate(SPATIAL_CV_FOLDS) for region in regions}
    for region in CV_REGIONS:
        for tile_path in sorted((dataset_root / "tiles" / "7ch" / "by_region" / region).glob("*.npz")):
            with np.load(tile_path, allow_pickle=False) as tile:
                x = np.asarray(tile["x"], dtype=np.float32)
                effective = np.squeeze(tile["valid_mask"]).astype(bool) & np.squeeze(tile["feature_valid_mask"]).astype(bool)
                label = np.squeeze(tile["y"]).astype(bool)
                water = np.squeeze(tile["water_river_mask"]).astype(bool)
                s2 = np.squeeze(tile["s2_valid_mask"]).astype(bool)
                masks = {
                    "s2_empty": effective if effective.any() and float(s2[effective].mean()) <= 0.01 else np.zeros_like(effective),
                    "topography_radar_shadow": effective & (x[5] * 45.0 > 20.0) & (x[6] * 50.0 > 40.0) & ((x[0] <= float(thresholds["p20_vv_norm"])) | (x[1] <= float(thresholds["p20_vh_norm"]))),
                    "permanent_water": effective & water & ~label,
                }
                for condition, mask in masks.items():
                    count = int(mask.sum())
                    if not count:
                        continue
                    candidate = {
                        "condition": condition,
                        "condition_label": dict(CONDITIONS)[condition],
                        "region": region,
                        "fold": fold_for_region[region],
                        "tile": tile_path.stem,
                        "tile_path": str(tile_path),
                        "row": int(tile["row"].item()),
                        "col": int(tile["col"].item()),
                        "condition_pixels": count,
                        "effective_pixels": int(effective.sum()),
                        "flood_pixels": int((label & effective).sum()),
                    }
                    rank = (count, int((label & mask).sum()), str(tile_path))
                    if condition not in choices or rank > choices[condition][0]:
                        choices[condition] = (rank, candidate)
    return [choices[condition][1] for condition, _ in CONDITIONS if condition in choices]


def open_raster(path: Path):
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(path)
    return ds


def read_window(ds, yoff: int, ysize: int) -> np.ndarray:
    return np.asarray(ds.GetRasterBand(1).ReadAsArray(0, yoff, ds.RasterXSize, ysize))


def block_windows(ds) -> Iterable[tuple[int, int]]:
    for yoff in range(0, ds.RasterYSize, BLOCK_ROWS):
        yield yoff, min(BLOCK_ROWS, ds.RasterYSize - yoff)


def close_datasets(*datasets) -> None:
    for ds in datasets:
        ds = None


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
