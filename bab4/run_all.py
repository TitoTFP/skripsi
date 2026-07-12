from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from bab4.artifacts import RunResult, SectionResult, validate_manifest
from bab4.config import Bab4Config
from bab4.sections.s4_1_1 import generate_4_1_1
from bab4.sections.s4_1_2 import generate_4_1_2
from bab4.sections.s4_2 import generate_4_2
from bab4.sections.s4_3 import generate_4_3
from bab4.sections.s4_4 import generate_4_4_1, generate_4_4_2, generate_4_4_3
from bab4.sections.s4_5_6 import generate_4_5, generate_4_6
from bab4.sections.s4_7 import generate_4_7

SECTION_GENERATORS = {
    "4.1.1": generate_4_1_1,
    "4.1.2": generate_4_1_2,
    "4.2": generate_4_2,
    "4.3": generate_4_3,
    "4.4.1": generate_4_4_1,
    "4.4.2": generate_4_4_2,
    "4.4.3": generate_4_4_3,
    "4.5": generate_4_5,
    "4.6": generate_4_6,
    "4.7": generate_4_7,
}

SECTION_GROUPS = {
    "4.1": ("4.1.1", "4.1.2"),
    "4.4": ("4.4.1", "4.4.2", "4.4.3"),
    "4.5-4.6": ("4.5", "4.6"),
    "4.7": ("4.7",),
}


def run_all(config: Bab4Config, sections: str | Iterable[str] | None = None) -> RunResult:
    if not config.no_retrain:
        raise ValueError("BAB 4 canonical pipeline does not support retraining")
    if config.clean_outputs:
        config.reset_output_dirs()
    else:
        config.ensure_output_dirs()
    selected = _resolve_sections(sections)
    section_results: list[SectionResult] = []
    for section in selected:
        section_results.append(SECTION_GENERATORS[section](config))
    manifest = validate_manifest(config, section_results)
    _write_manifest(config, manifest)
    _write_validation_report(config, manifest)
    return RunResult(section_results=section_results, manifest=manifest)


def _resolve_sections(sections: str | Iterable[str] | None) -> list[str]:
    if sections is None:
        return list(SECTION_GENERATORS)
    if isinstance(sections, str):
        raw = [part.strip() for part in sections.split(",") if part.strip()]
    else:
        raw = [str(part).strip() for part in sections if str(part).strip()]
    if not raw or raw == ["all"]:
        return list(SECTION_GENERATORS)
    resolved: list[str] = []
    for name in raw:
        sections_for_name = SECTION_GROUPS.get(name)
        if sections_for_name is None and name in SECTION_GENERATORS:
            sections_for_name = (name,)
        if sections_for_name is None:
            raise ValueError(f"unknown BAB 4 section {name!r}")
        for section in sections_for_name:
            if section not in resolved:
                resolved.append(section)
    return resolved


def _write_manifest(config: Bab4Config, manifest) -> Path:
    path = config.tables_dir / "bab4_output_manifest.csv"
    manifest.to_csv(path, index=False)
    return path


def _write_validation_report(config: Bab4Config, manifest) -> Path:
    path = config.narratives_dir / "bab4_validation_report.md"
    status_counts = manifest.status_counts()
    missing = manifest.filter_status_not("exists")
    lines = [
        "# BAB 4 Validation Report",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Output root: `{config.output_root}`",
        f"- Offline mode: `{config.offline}`",
        f"- No retraining: `{config.no_retrain}`",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in status_counts.items():
        lines.append(f"- `{status}`: {count}")
    lines.extend(["", "## Non-Existing Artifacts", ""])
    if missing.empty:
        lines.append("All requested artifacts exist in the fresh BAB 4 output folder.")
    else:
        for row in missing:
            lines.append(f"- {row['artifact_id']}: `{row['status']}` - {row['note']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate canonical BAB 4 validation outputs.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root or path inside it.")
    parser.add_argument("--sections", default="all", help="Comma-separated section selector, e.g. all, 4.1, 4.5")
    parser.add_argument("--threshold", type=float, default=0.5, help="Inference threshold used by report metadata.")
    parser.add_argument("--allow-online", action="store_true", help="Reserved flag; online OSM fetch is disabled by default.")
    parser.add_argument("--no-retrain", action="store_true", help="Keep validation-only behavior; retained for explicit CLI documentation.")
    parser.add_argument("--keep-outputs", action="store_true", help="Do not clear bab4/outputs before generation.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = Bab4Config.from_repo(
        args.repo_root,
        threshold=args.threshold,
        offline=not args.allow_online,
        no_retrain=True,
        clean_outputs=not args.keep_outputs,
    )
    result = run_all(config, sections=args.sections)
    counts = result.manifest.status_counts()
    print(f"wrote {len(result.manifest)} manifest rows to {config.output_root}")
    print(f"status_counts={counts}")


if __name__ == "__main__":
    main()
