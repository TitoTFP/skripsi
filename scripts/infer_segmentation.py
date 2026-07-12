from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from osgeo import gdal
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from scripts.train_segmentation import build_model, resolve_device
from training.datasets import FloodTileDataset
from training.losses import masked_bce_dice_loss
from training.metrics import masked_binary_stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TILE_ROOT = ROOT / "dataset" / "tiles"
FEATURE_ROOT = ROOT / "dataset" / "features_preprocessed"
PROBABILITY_NODATA = -9999.0
PREDICTION_NODATA = 255

gdal.UseExceptions()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run flood segmentation inference from a checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--region", dest="regions", action="append", nargs="?", const="all", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=("unet", "procanet"), default=None)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--water-river", "--water_river", dest="water_river_as_flood", action="store_true")
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--write-geotiff", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    architecture, base_channels = resolve_checkpoint_settings(
        checkpoint,
        architecture_fallback=args.architecture,
        base_channels_fallback=args.base_channels,
    )
    model = build_model(architecture, base_channels).to(device)
    state_dict = checkpoint["model_state_dict"]
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    regions = args.regions
    if not regions or "all" in regions:
        from scripts.preprocessing_utils import CV_REGIONS, TEST_REGION
        regions = [TEST_REGION] + list(CV_REGIONS)

    region_summaries = []
    total_stats = InferenceStats()
    for region in regions:
        region_summary = infer_region(
            model=model,
            architecture=architecture,
            region=region,
            device=device,
            args=args,
        )
        region_summaries.append(region_summary)
        total_stats.merge(region_summary.pop("_stats"))

    aggregate = {"region": "aggregate", **total_stats.summary()}
    rows = region_summaries + [aggregate]
    write_metrics(output_dir, rows, args, architecture, base_channels)


def resolve_checkpoint_settings(
    checkpoint: dict[str, Any],
    architecture_fallback: str | None,
    base_channels_fallback: int,
) -> tuple[str, int]:
    config = checkpoint.get("config") or {}
    architecture = checkpoint.get("architecture") or config.get("architecture") or architecture_fallback
    if architecture is None:
        raise ValueError("checkpoint has no architecture metadata; pass --architecture")
    base_channels = int(config.get("base_channels") or base_channels_fallback)
    return str(architecture), base_channels


@dataclass
class InferenceStats:
    total_loss: float = 0.0
    batches: int = 0
    stats: dict[str, int] = field(default_factory=lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})

    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        effective_valid_mask: torch.Tensor,
        loss: float,
        threshold: float,
    ) -> None:
        self.total_loss += loss
        self.batches += 1
        update = masked_binary_stats(logits, target, effective_valid_mask, threshold=threshold, from_logits=True)
        for key in self.stats:
            self.stats[key] += update[key]

    def merge(self, other: "InferenceStats") -> None:
        self.total_loss += other.total_loss
        self.batches += other.batches
        for key in self.stats:
            self.stats[key] += other.stats[key]

    def summary(self) -> dict[str, float | int]:
        tp = self.stats["tp"]
        tn = self.stats["tn"]
        fp = self.stats["fp"]
        fn = self.stats["fn"]
        iou_den = tp + fp + fn
        dice_den = (2 * tp) + fp + fn
        total = tp + tn + fp + fn
        return {
            "loss": self.total_loss / max(self.batches, 1),
            "iou": tp / iou_den if iou_den else 0.0,
            "dice": (2 * tp) / dice_den if dice_den else 0.0,
            "accuracy": (tp + tn) / total if total else 0.0,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "batches": self.batches,
        }


@torch.no_grad()
def infer_region(
    model: torch.nn.Module,
    architecture: str,
    region: str,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dataset = FloodTileDataset(
        "test",
        architecture=architecture,
        root=args.tile_root,
        augment=False,
        water_river_as_flood=args.water_river_as_flood,
        regions=[region],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    iterator: Iterable[dict[str, Any]] = loader
    total = len(loader)
    if args.max_batches is not None:
        iterator = islice(loader, args.max_batches)
        total = min(total, args.max_batches)

    stats = InferenceStats()
    mosaic = GeoTiffMosaic(FEATURE_ROOT / region / "stack_7ch.tif", threshold=args.threshold) if args.write_geotiff else None
    prediction_dir = args.output_dir / "predictions" / region
    if not args.no_save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(iterator, total=total, desc=f"infer {region}", dynamic_ncols=True, leave=True):
        features = _to_device(batch["features"], device)
        y = batch["y"].to(device)
        valid_mask = _effective_valid_mask(batch, device)
        logits = model(features)
        loss = masked_bce_dice_loss(logits, y, valid_mask)
        stats.update(logits, y, valid_mask, float(loss.detach().cpu()), threshold=args.threshold)

        probability = torch.sigmoid(logits).detach().cpu().numpy()
        prediction = (probability >= args.threshold).astype(np.uint8)
        y_np = y.detach().cpu().numpy()
        label_valid_np = batch["valid_mask"].detach().cpu().numpy().astype(bool)
        effective_valid_np = valid_mask.detach().cpu().numpy().astype(bool)
        rows = _metadata_sequence(batch["metadata"], "row")
        cols = _metadata_sequence(batch["metadata"], "col")
        paths = _metadata_sequence(batch["metadata"], "path")
        regions = _metadata_sequence(batch["metadata"], "region")

        for idx, source_path in enumerate(paths):
            tile_name = Path(str(source_path)).name
            if not args.no_save_predictions:
                np.savez_compressed(
                    prediction_dir / tile_name,
                    probability=probability[idx],
                    prediction=prediction[idx],
                    y=y_np[idx],
                    valid_mask=label_valid_np[idx],
                    effective_valid_mask=effective_valid_np[idx],
                    row=np.array(int(rows[idx])),
                    col=np.array(int(cols[idx])),
                    region=np.array(str(regions[idx])),
                )
            if mosaic is not None:
                mosaic.add_tile(
                    row=int(rows[idx]),
                    col=int(cols[idx]),
                    probability=probability[idx],
                    effective_valid_mask=effective_valid_np[idx],
                )

    if mosaic is not None:
        probability, prediction, effective_valid = mosaic.finalize()
        geotiff_dir = args.output_dir / "geotiff"
        geotiff_dir.mkdir(parents=True, exist_ok=True)
        reference_path = FEATURE_ROOT / region / "stack_7ch.tif"
        write_geotiff(reference_path, geotiff_dir / f"{region}_probability.tif", probability, gdal.GDT_Float32, PROBABILITY_NODATA)
        write_geotiff(reference_path, geotiff_dir / f"{region}_prediction.tif", prediction, gdal.GDT_Byte, PREDICTION_NODATA)
        write_geotiff(reference_path, geotiff_dir / f"{region}_effective_valid_mask.tif", effective_valid, gdal.GDT_Byte, 0)

    return {"region": region, **stats.summary(), "_stats": stats}


class GeoTiffMosaic:
    def __init__(self, reference_path: Path | str, threshold: float) -> None:
        self.reference_path = Path(reference_path)
        ds = gdal.Open(str(self.reference_path), gdal.GA_ReadOnly)
        if ds is None:
            raise FileNotFoundError(self.reference_path)
        self.height = ds.RasterYSize
        self.width = ds.RasterXSize
        self.threshold = threshold
        self.probability_sum = np.zeros((self.height, self.width), dtype=np.float32)
        self.count = np.zeros((self.height, self.width), dtype=np.uint16)
        self.effective_valid = np.zeros((self.height, self.width), dtype=np.uint8)
        ds = None

    def add_tile(
        self,
        row: int,
        col: int,
        probability: np.ndarray,
        effective_valid_mask: np.ndarray,
    ) -> None:
        prob = np.asarray(probability, dtype=np.float32)
        valid = np.asarray(effective_valid_mask, dtype=bool)
        if prob.ndim == 3:
            prob = prob[0]
        if valid.ndim == 3:
            valid = valid[0]
        height = min(prob.shape[0], self.height - row)
        width = min(prob.shape[1], self.width - col)
        if height <= 0 or width <= 0:
            return
        target = np.s_[row : row + height, col : col + width]
        tile_prob = prob[:height, :width]
        tile_valid = valid[:height, :width]
        # Only observations that are valid for the loss/evaluation are allowed
        # to contribute to an overlap average.  Otherwise a zero-filled invalid
        # tile can dilute a valid neighbouring prediction.
        self.probability_sum[target] += tile_prob * tile_valid
        self.count[target] += tile_valid.astype(np.uint16)
        self.effective_valid[target] = np.maximum(self.effective_valid[target], tile_valid.astype(np.uint8))

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        covered = self.count > 0
        probability = np.full((self.height, self.width), PROBABILITY_NODATA, dtype=np.float32)
        probability[covered] = self.probability_sum[covered] / self.count[covered].astype(np.float32)
        prediction = np.full((self.height, self.width), PREDICTION_NODATA, dtype=np.uint8)
        prediction[covered] = (probability[covered] >= self.threshold).astype(np.uint8)
        return probability, prediction, self.effective_valid


def write_geotiff(
    reference_path: Path | str,
    output_path: Path | str,
    array: np.ndarray,
    gdal_dtype: int,
    nodata: float | int,
) -> None:
    reference = gdal.Open(str(reference_path), gdal.GA_ReadOnly)
    if reference is None:
        raise FileNotFoundError(reference_path)
    arr = np.asarray(array)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        str(output_path),
        arr.shape[1],
        arr.shape[0],
        1,
        gdal_dtype,
        options=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    ds.SetGeoTransform(reference.GetGeoTransform())
    ds.SetProjection(reference.GetProjection())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(float(nodata))
    band.WriteArray(arr)
    band.FlushCache()
    ds.FlushCache()
    ds = None
    reference = None


def write_metrics(
    output_dir: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    architecture: str,
    base_channels: int,
) -> None:
    fieldnames = ["region", "loss", "iou", "dice", "accuracy", "tp", "tn", "fp", "fn", "batches"]
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    payload = {
        "checkpoint": str(args.checkpoint),
        "architecture": architecture,
        "base_channels": base_channels,
        "regions": args.regions,
        "threshold": args.threshold,
        "write_geotiff": args.write_geotiff,
        "save_predictions": not args.no_save_predictions,
        "water_river_as_flood": args.water_river_as_flood,
        "metrics": [{key: row[key] for key in fieldnames} for row in rows],
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _effective_valid_mask(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    label_valid_mask = batch["valid_mask"].to(device).bool()
    auxiliary_masks = batch.get("auxiliary_masks") or {}
    feature_valid_mask = auxiliary_masks.get("feature_valid_mask")
    if feature_valid_mask is None:
        return label_valid_mask
    return label_valid_mask & feature_valid_mask.to(device).bool()


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _metadata_sequence(metadata: dict[str, Any], key: str) -> list[Any]:
    value = metadata[key]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return list(value)


if __name__ == "__main__":
    main()
