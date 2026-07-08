from __future__ import annotations

import csv
import struct
from pathlib import Path

from bab4.artifacts import ArtifactResult, ArtifactSpec


def csv_shape(path: Path) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return rows, len(header)


def png_size(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as handle:
        signature = handle.read(24)
    if len(signature) >= 24 and signature.startswith(b"\x89PNG\r\n\x1a\n"):
        return struct.unpack(">II", signature[16:24])
    return None, None


def summarize_file(path: Path, kind: str) -> dict[str, int | None]:
    size = path.stat().st_size if path.exists() else 0
    rows = columns = width = height = None
    if path.exists() and kind == "table" and path.suffix.lower() == ".csv":
        rows, columns = csv_shape(path)
    elif path.exists() and kind == "figure" and path.suffix.lower() == ".png":
        width, height = png_size(path)
    return {"size_bytes": size, "rows": rows, "columns": columns, "width": width, "height": height}


def result_for_path(spec: ArtifactSpec, path: Path, status: str, source: str, note: str = "") -> ArtifactResult:
    summary = summarize_file(path, spec.kind) if path.exists() else {}
    return ArtifactResult(
        spec=spec,
        path=path,
        status=status,
        source=source,
        note=note,
        size_bytes=int(summary.get("size_bytes") or 0),
        rows=summary.get("rows"),
        columns=summary.get("columns"),
        width=summary.get("width"),
        height=summary.get("height"),
    )


def write_dict_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
