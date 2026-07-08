from __future__ import annotations

from pathlib import Path

from bab4.artifacts import ArtifactResult, ArtifactSpec
from bab4.io import result_for_path, write_dict_rows


def write_table(
    config,
    spec: ArtifactSpec,
    rows: list[dict[str, object]],
    *,
    source: str,
    note: str = "",
    fieldnames: list[str] | None = None,
) -> ArtifactResult:
    path = config.tables_dir / spec.filename
    if not rows:
        fieldnames = fieldnames or []
    else:
        fieldnames = fieldnames or list(rows[0])
    write_dict_rows(path, rows, fieldnames)
    return result_for_path(spec, path, "exists", source, note)


def write_text_artifact(config, spec: ArtifactSpec, text: str, *, source: str, note: str = "") -> ArtifactResult:
    path = config.narratives_dir / spec.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return result_for_path(spec, path, "exists", source, note)


def figure_result(config, spec: ArtifactSpec, path: Path, *, source: str, note: str = "") -> ArtifactResult:
    return result_for_path(spec, path, "exists", source, note)


def missing_result(config, spec: ArtifactSpec, *, source: str, status: str = "missing_source", note: str = "") -> ArtifactResult:
    return result_for_path(spec, config.output_dir_for_kind(spec.kind) / spec.filename, status, source, note)
