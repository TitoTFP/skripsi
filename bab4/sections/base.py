from __future__ import annotations

from pathlib import Path

from bab4.artifacts import SectionResult


def section_result(name: str, artifacts) -> SectionResult:
    return SectionResult(name=name, artifacts=list(artifacts))


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
