from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


def resolve_repo_root(root: Path | str | None = None) -> Path:
    """Resolve a path inside the repository to the repository root."""
    candidate = Path.cwd() if root is None else Path(root)
    candidate = candidate.expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for path in (candidate, *candidate.parents):
        if (path / "pyproject.toml").exists() and (path / "dataset").exists():
            return path
    raise FileNotFoundError(f"could not resolve skripsi repo root from {candidate}")


@dataclass(frozen=True)
class Bab4Config:
    root: Path
    output_root: Path
    legacy_output_root: Path
    dataset_root: Path
    runs_root: Path
    threshold: float = 0.5
    test_region: str = "Aceh_Utara"
    offline: bool = True
    no_retrain: bool = True
    clean_outputs: bool = True

    @classmethod
    def from_repo(
        cls,
        root: Path | str | None = None,
        *,
        output_root: Path | str | None = None,
        threshold: float = 0.5,
        test_region: str = "Aceh_Utara",
        offline: bool = True,
        no_retrain: bool = True,
        clean_outputs: bool = True,
    ) -> "Bab4Config":
        repo_root = resolve_repo_root(root)
        fresh_output = Path(output_root).expanduser().resolve() if output_root else repo_root / "bab4" / "outputs"
        return cls(
            root=repo_root,
            output_root=fresh_output,
            legacy_output_root=repo_root / "outputs" / "bab4",
            dataset_root=repo_root / "dataset",
            runs_root=repo_root / "runs",
            threshold=threshold,
            test_region=test_region,
            offline=offline,
            no_retrain=no_retrain,
            clean_outputs=clean_outputs,
        )

    @property
    def tables_dir(self) -> Path:
        return self.output_root / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / "figures"

    @property
    def narratives_dir(self) -> Path:
        return self.output_root / "narratives"

    def output_dir_for_kind(self, kind: str) -> Path:
        if kind == "table":
            return self.tables_dir
        if kind == "figure":
            return self.figures_dir
        if kind == "narrative":
            return self.narratives_dir
        raise ValueError(f"unknown artifact kind: {kind}")

    def legacy_dir_for_kind(self, kind: str) -> Path:
        if kind == "table":
            return self.legacy_output_root / "tables"
        if kind == "figure":
            return self.legacy_output_root / "figures"
        if kind == "narrative":
            return self.legacy_output_root / "narratives"
        raise ValueError(f"unknown artifact kind: {kind}")

    def ensure_output_dirs(self) -> None:
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.narratives_dir.mkdir(parents=True, exist_ok=True)

    def reset_output_dirs(self) -> None:
        for path in (self.tables_dir, self.figures_dir, self.narratives_dir):
            if path.exists():
                shutil.rmtree(path)
        self.ensure_output_dirs()
