from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


CHANNELS_7CH = ("VV", "VH", "Hue", "Saturation", "Value", "Slope", "HAND")
BAD_S2_REGIONS = {"Aceh_Tamiang", "Agam", "Langsa", "Pasaman_Barat"}
TEST_REGION = "Aceh_Utara"
SPATIAL_CV_FOLDS = (
    ("Pidie", "Pidie_Jaya"),
    ("Aceh_Besar", "Banda_Aceh"),
    ("Aceh_Tamiang", "Aceh_Timur"),
    ("Bireuen", "Langsa"),
    ("Agam", "Pasaman_Barat"),
)
CV_REGIONS = tuple(region for fold in SPATIAL_CV_FOLDS for region in fold)
REGIONS = tuple(sorted((*CV_REGIONS, TEST_REGION)))
MODEL_KEYS = ("unet", "procanet")
MODEL_LABELS = {"unet": "U-Net", "procanet": "ProCANet"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_csv_row_map(path: Path, key: str = "region") -> dict[str, dict[str, str]]:
    return {row[key]: row for row in read_csv_rows(path)}


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    return int(round(to_float(value, float(default))))


def pct(part: float, total: float) -> float:
    return part / total * 100.0 if total else 0.0


def fmt_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def region_quality_from_s2_pct(s2_pct: float) -> str:
    if s2_pct == 0:
        return "kosong"
    if s2_pct < 0.01:
        return "hampir kosong"
    if s2_pct < 10:
        return "rendah"
    return "baik"


def ensure_list(value: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return list(value)
