from __future__ import annotations

import json
from itertools import islice
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from training.losses import masked_bce_dice_loss
from training.metrics import masked_binary_stats


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best_score: float = float("-inf")
    bad_epochs: int = 0

    def step(self, score: float) -> bool:
        if self.patience <= 0:
            return False
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
    progress_desc: str | None = None,
    gradient_accumulation_steps: int = 1,
    amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    batches = 0
    stats = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    accumulation_steps = max(1, gradient_accumulation_steps)
    effective_amp = bool(amp_enabled and device.type == "cuda")
    scaler = scaler or torch.amp.GradScaler("cuda", enabled=effective_amp)
    optimizer.zero_grad(set_to_none=True)
    for batch in _batch_iterator(loader, max_batches=max_batches, progress_desc=progress_desc):
        features = _to_device(batch["features"], device)
        y = batch["y"].to(device)
        valid_mask = _effective_valid_mask(batch, device)

        with torch.amp.autocast(device_type="cuda", enabled=effective_amp):
            logits = model(features)
            loss = masked_bce_dice_loss(logits, y, valid_mask)
            backward_loss = loss / accumulation_steps
        scaler.scale(backward_loss).backward()
        if (batches + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().cpu())
        _accumulate(stats, masked_binary_stats(logits.detach(), y, valid_mask))
        batches += 1
    if batches and batches % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return _summarize(total_loss, batches, stats)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    progress_desc: str | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    batches = 0
    stats = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for batch in _batch_iterator(loader, max_batches=max_batches, progress_desc=progress_desc):
        features = _to_device(batch["features"], device)
        y = batch["y"].to(device)
        valid_mask = _effective_valid_mask(batch, device)
        logits = model(features)
        loss = masked_bce_dice_loss(logits, y, valid_mask)
        total_loss += float(loss.detach().cpu())
        _accumulate(stats, masked_binary_stats(logits, y, valid_mask))
        batches += 1
    return _summarize(total_loss, batches, stats)


def save_checkpoint_if_best(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path | str,
    epoch: int,
    val_iou: float,
    best_val_iou: float,
    architecture: str,
    config: dict[str, Any],
) -> tuple[float, bool]:
    if val_iou <= best_val_iou:
        return best_val_iou, False

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_iou": val_iou,
        "architecture": architecture,
        "config": config,
    }
    torch.save(checkpoint, output_dir / "best.pt")
    return val_iou, True


def write_training_config(output_dir: Path | str, config: dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.json"
    path.write_text(json.dumps(_jsonable(config), indent=2, sort_keys=True) + "\n")
    return path


def _effective_valid_mask(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Return pixels valid for both label and feature tensors.

    ``valid_mask`` comes from ``label_valid_mask`` in the tile maker. When
    ``feature_valid_mask`` is available in auxiliary masks, loss and metrics
    should only use the intersection of both masks so invalid/no-data features
    never contribute to optimization or evaluation.
    """
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


def _accumulate(total: dict[str, int], update: dict[str, int]) -> None:
    for key in total:
        total[key] += update[key]


def _batch_iterator(
    loader: DataLoader,
    max_batches: int | None,
    progress_desc: str | None,
) -> Any:
    if progress_desc is None:
        return islice(loader, max_batches) if max_batches is not None else loader
    total = len(loader)
    if max_batches is not None:
        total = min(total, max_batches)
    iterator = islice(loader, max_batches) if max_batches is not None else loader
    return tqdm(iterator, total=total, desc=progress_desc, dynamic_ncols=True, leave=True)


def _summarize(total_loss: float, batches: int, stats: dict[str, int]) -> dict[str, float]:
    tp = stats["tp"]
    tn = stats["tn"]
    fp = stats["fp"]
    fn = stats["fn"]
    iou_den = tp + fp + fn
    dice_den = (2 * tp) + fp + fn
    total = tp + tn + fp + fn
    return {
        "loss": total_loss / max(batches, 1),
        "iou": tp / iou_den if iou_den else 0.0,
        "dice": (2 * tp) / dice_den if dice_den else 0.0,
        "accuracy": (tp + tn) / total if total else 0.0,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
