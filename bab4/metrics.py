from __future__ import annotations

import csv
from pathlib import Path


def load_evaluation_metrics(evaluation_root: Path, model: str, region: str | None = None) -> dict[str, object]:
    path = evaluation_root / model / "eval_test" / "metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if region is None:
        metrics = next(row for row in rows if row.get("region") != "aggregate")
    else:
        metrics = next(row for row in rows if row.get("region") == region)
    tp = int(float(metrics.get("tp", 0)))
    fp = int(float(metrics.get("fp", 0)))
    fn = int(float(metrics.get("fn", 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {**metrics, "model": model, "precision": precision, "recall": recall}


def load_final_metrics(runs_root: Path, model: str, region: str | None = None) -> dict[str, object]:
    """Backward-compatible loader for legacy callers."""
    return load_evaluation_metrics(runs_root / "final", model, region)
