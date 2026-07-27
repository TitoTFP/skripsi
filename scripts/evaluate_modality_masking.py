"""Run modality-masking sensitivity analysis on the independent test region."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal

from bab4.sections.s4_5_6 import _mosaic_binary_stats, _validate_same_grid

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("unet", "procanet")
SCENARIOS = ("all", "sentinel1", "sentinel2", "demnas")
SUMMARY_FIELDS = (
    "model", "input_scenario", "iou", "dice_f1", "accuracy", "precision", "recall",
    "specificity", "fpr", "fnr", "tp", "tn", "fp", "fn", "delta_iou", "delta_dice",
    "checkpoint", "threshold",
)


gdal.UseExceptions()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unet-checkpoint", type=Path, default=_default_checkpoint("unet"))
    parser.add_argument("--procanet-checkpoint", type=Path, default=_default_checkpoint("procanet"))
    parser.add_argument("--test-region", default="Aceh_Utara")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "bab4" / "evaluation" / "modality_masking")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None, help="Smoke-test only; partial runs are not reportable.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_batches is not None and not args.overwrite:
        raise ValueError("--max-batches requires --overwrite and produces a non-reportable smoke-test run")
    checkpoints = {"unet": args.unet_checkpoint, "procanet": args.procanet_checkpoint}
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for scenario in SCENARIOS:
            output_dir = args.output_dir / model / scenario / "eval_test"
            metrics_path = output_dir / "metrics.json"
            if args.overwrite or not metrics_path.exists():
                _run_inference(model, scenario, checkpoints[model], output_dir, args)
            rows.append(
                _summary_row(
                    model,
                    scenario,
                    checkpoints[model],
                    metrics_path,
                    output_dir,
                    args.test_region,
                    args.threshold,
                )
            )
            if scenario == "sentinel2":
                _write_s2_valid_only_metrics(model, output_dir, args.test_region, args.threshold)

    for model in MODELS:
        baseline = next(row for row in rows if row["model"] == model and row["input_scenario"] == "all")
        for row in rows:
            if row["model"] == model:
                try:
                    row["delta_iou"] = float(baseline["iou"]) - float(row["iou"])
                    row["delta_dice"] = float(baseline["dice_f1"]) - float(row["dice_f1"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid summary metrics for {model}") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "modality_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in SUMMARY_FIELDS})
    (args.output_dir / "modality_metrics.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "test_region": args.test_region,
        "threshold": args.threshold,
        "input_scenarios": list(SCENARIOS),
        "max_batches": args.max_batches,
        "evaluation_unit": "unique mosaic pixels in effective_valid_mask",
        "method": "modality masking at inference; no retraining",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def _tile_directory(model: str, test_region: str) -> Path:
    tile_kind = "7ch" if model == "unet" else "procanet"
    return ROOT / "dataset" / "tiles" / tile_kind / "by_region" / test_region


def _default_checkpoint(model: str) -> Path:
    metadata = ROOT / "runs" / "cv_best_checkpoint_eval" / model / "eval_test" / "metrics.json"
    if not metadata.exists():
        return ROOT / "runs" / model / "best.pt"
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        checkpoint = Path(str(payload["checkpoint"]))
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise ValueError(f"invalid evaluation metadata: {metadata}") from exc
    return checkpoint if checkpoint.is_absolute() else ROOT / checkpoint


def _run_inference(
    model: str,
    scenario: str,
    checkpoint: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    command = [
        sys.executable, "-m", "scripts.infer_segmentation",
        "--architecture", model,
        "--checkpoint", str(checkpoint),
        "--region", args.test_region,
        "--input-scenario", scenario,
        "--threshold", str(args.threshold),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--output-dir", str(output_dir),
        "--write-geotiff",
    ]
    if args.max_batches is not None:
        command.extend(("--max-batches", str(args.max_batches)))
    subprocess.run(command, cwd=ROOT, check=True)


def _summary_row(
    model: str,
    scenario: str,
    checkpoint: Path,
    metrics_path: Path,
    output_dir: Path,
    test_region: str,
    threshold: float,
) -> dict[str, Any]:
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        next(row for row in payload["metrics"] if row["region"] == "aggregate")
        expected_tiles = len(list(_tile_directory(model, test_region).glob("*.npz")))
        output_tiles = len(list((output_dir / "predictions" / test_region).glob("*.npz")))
        required_geotiffs = all(
            (output_dir / "geotiff" / f"{test_region}_{name}.tif").exists()
            for name in ("probability", "prediction", "effective_valid_mask")
        )
        metadata_matches = (
            payload["architecture"] == model
            and payload["input_scenario"] == scenario
            and payload["regions"] == [test_region]
            and float(payload["threshold"]) == threshold
            and Path(str(payload["checkpoint"])).resolve() == checkpoint.resolve()
            and payload.get("max_batches") is None
            and bool(payload.get("complete", False))
            and int(payload.get("tiles_processed_by_region", {}).get(test_region, -1)) == expected_tiles
            and output_tiles == expected_tiles
            and required_geotiffs
        )
    except (OSError, json.JSONDecodeError, KeyError, StopIteration, TypeError, ValueError) as exc:
        raise ValueError(f"invalid inference metrics: {metrics_path}") from exc
    if not metadata_matches:
        raise ValueError(f"stale or incomplete inference metadata: {metrics_path}; rerun with --overwrite")
    geotiff_dir = output_dir / "geotiff"
    stats = _mosaic_binary_stats(
        geotiff_dir / f"{test_region}_probability.tif",
        geotiff_dir / f"{test_region}_effective_valid_mask.tif",
        ROOT / "dataset" / "labels_unosat_rasterized" / test_region / "label_flood_binary.tif",
        threshold=threshold,
    )
    return {
        "model": model,
        "input_scenario": scenario,
        **_metrics_from_counts(stats),
        "delta_iou": 0.0,
        "delta_dice": 0.0,
        "checkpoint": str(checkpoint),
        "threshold": threshold,
    }


def _metrics_from_counts(stats: dict[str, int]) -> dict[str, float | int]:
    tp, tn, fp, fn = (stats[key] for key in ("tp", "tn", "fp", "fn"))
    total = tp + tn + fp + fn
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "dice_f1": (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "fpr": fp / (tn + fp) if tn + fp else 0.0,
        "fnr": fn / (tp + fn) if tp + fn else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _write_s2_valid_only_metrics(model: str, output_dir: Path, test_region: str, threshold: float) -> None:
    geotiff_dir = output_dir / "geotiff"
    paths = (
        geotiff_dir / f"{test_region}_probability.tif",
        geotiff_dir / f"{test_region}_effective_valid_mask.tif",
        ROOT / "dataset" / "labels_unosat_rasterized" / test_region / "label_flood_binary.tif",
        ROOT / "dataset" / "features_preprocessed" / test_region / "s2_valid_mask.tif",
    )
    datasets = tuple(gdal.Open(str(path), gdal.GA_ReadOnly) for path in paths)
    _validate_same_grid(paths, datasets)
    counts = {key: 0 for key in ("tp", "tn", "fp", "fn")}
    width, height = datasets[0].RasterXSize, datasets[0].RasterYSize
    for row in range(0, height, 512):
        block_height = min(512, height - row)
        probability, effective, truth, s2_valid = (
            np.asarray(dataset.GetRasterBand(1).ReadAsArray(0, row, width, block_height)) for dataset in datasets
        )
        mask = effective.astype(bool) & s2_valid.astype(bool)
        prediction = probability >= threshold
        truth = truth.astype(bool)
        try:
            counts["tp"] += int((mask & truth & prediction).sum())
            counts["tn"] += int((mask & ~truth & ~prediction).sum())
            counts["fp"] += int((mask & ~truth & prediction).sum())
            counts["fn"] += int((mask & truth & ~prediction).sum())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid Sentinel-2 valid-only raster values for {model}") from exc
    metrics: dict[str, object] = {
        "model": model,
        "population": "effective_valid_mask & s2_valid_mask",
        "evaluated_unique_pixels": sum(counts.values()),
        **_metrics_from_counts(counts),
    }
    csv_path = output_dir / "metrics_s2_valid_only.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    payload = {
        "model": model,
        "input_scenario": "sentinel2",
        "region": test_region,
        "population": "effective_valid_mask & s2_valid_mask; unique mosaic pixels",
        "threshold": threshold,
        "metrics": metrics,
    }
    (output_dir / "metrics_s2_valid_only.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
