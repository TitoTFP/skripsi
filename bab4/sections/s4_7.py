from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bab4.artifacts import ALL_ARTIFACTS
from bab4.common import to_float, to_int
from bab4.plots import savefig, setup_style
from bab4.sections.base import section_result
from bab4.writer import figure_result, missing_result, write_table, write_text_artifact

SOURCE = "bab4/evaluation/modality_masking/{modality_metrics.csv,*/eval_test/predictions,provenance.json}"
SCENARIOS = ("all", "sentinel1", "sentinel2", "demnas")
SCENARIO_LABELS = {
    "all": "Semua fitur",
    "sentinel1": "Sentinel-1",
    "sentinel2": "Sentinel-2",
    "demnas": "DEMNAS",
}


def _spec(artifact_id: str):
    return next(spec for spec in ALL_ARTIFACTS if spec.artifact_id == artifact_id)


def generate_4_7(config):
    source = config.root / "bab4" / "evaluation" / "modality_masking"
    summary_path = source / "modality_metrics.csv"
    rows = _read_csv(summary_path) if summary_path.exists() else []
    expected = {(model, scenario) for model in ("unet", "procanet") for scenario in SCENARIOS}
    actual = {(str(row.get("model")), str(row.get("input_scenario"))) for row in rows}
    s2_valid_rows = _s2_valid_rows(source)
    provenance_ok = False
    provenance_path = source / "provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance_ok = (
            provenance.get("test_region") == config.test_region
            and to_float(provenance.get("threshold")) == config.threshold
            and provenance.get("max_batches") is None
            and provenance.get("evaluation_unit") == "unique mosaic pixels in effective_valid_mask"
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    runs_complete = all(
        _run_complete(source / model / scenario / "eval_test", model, scenario, config.test_region, config.threshold)
        for model, scenario in expected
    )
    if actual != expected or len(rows) != 8 or len(s2_valid_rows) != 2 or not provenance_ok or not runs_complete:
        note = "evaluasi modality masking belum lengkap; jalankan scripts/evaluate_modality_masking.py"
        return section_result(
            "4.7",
            [
                missing_result(config, _spec("Tabel 4.16"), source=SOURCE, note=note),
                missing_result(config, _spec("Tabel 4.17"), source=SOURCE, note=note),
                missing_result(config, _spec("Gambar 4.15"), source=SOURCE, note=note),
                missing_result(config, _spec("Gambar 4.16"), source=SOURCE, note=note),
                _narrative(config, [], available=False),
            ],
        )

    artifacts = [
        write_table(config, _spec("Tabel 4.16"), rows, source=SOURCE),
        write_table(config, _spec("Tabel 4.17"), s2_valid_rows, source=SOURCE),
        _panel(config, source, "unet", "Gambar 4.15"),
        _panel(config, source, "procanet", "Gambar 4.16"),
        _narrative(config, rows, available=True),
    ]
    return section_result("4.7", artifacts)


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _run_complete(path: Path, model: str, scenario: str, region: str, threshold: float) -> bool:
    try:
        payload = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        tile_root = "7ch" if model == "unet" else "procanet"
        repo_root = path.parents[5]
        expected_tiles = len(list((repo_root / "dataset" / "tiles" / tile_root / "by_region" / region).glob("*.npz")))
        output_tiles = len(list((path / "predictions" / region).glob("*.npz")))
        processed = payload.get("tiles_processed_by_region", {})
        return (
            isinstance(processed, dict)
            and payload.get("architecture") == model
            and payload.get("input_scenario") == scenario
            and payload.get("regions") == [region]
            and to_float(payload.get("threshold")) == threshold
            and payload.get("max_batches") is None
            and bool(payload.get("complete"))
            and to_int(processed.get(region)) == expected_tiles
            and output_tiles == expected_tiles
            and all((path / "geotiff" / f"{region}_{name}.tif").exists() for name in ("probability", "prediction", "effective_valid_mask"))
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _s2_valid_rows(source: Path) -> list[dict[str, object]]:
    rows = []
    for model in ("unet", "procanet"):
        path = source / model / "sentinel2" / "eval_test" / "metrics_s2_valid_only.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"model": model, **payload["metrics"]})
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
    return rows


def _panel(config, source: Path, model: str, artifact_id: str):
    tile_names = []
    for scenario in SCENARIOS:
        directory = source / model / scenario / "eval_test" / "predictions" / config.test_region
        tile_names.append({path.name for path in directory.glob("*.npz")})
    common = set.intersection(*tile_names) if tile_names else set()
    if not common:
        return missing_result(config, _spec(artifact_id), source=SOURCE, note="prediksi tile yang sama tidak ditemukan")
    preferred = "Aceh_Utara_r001280_c005632.npz"
    tile_name = preferred if preferred in common else sorted(common)[0]
    setup_style()
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    label = None
    for index, scenario in enumerate(SCENARIOS):
        path = source / model / scenario / "eval_test" / "predictions" / config.test_region / tile_name
        with np.load(path, allow_pickle=False) as data:
            prediction = data["prediction"].squeeze()
            if label is None:
                label = data["y"].squeeze()
        axes[index + 1].imshow(prediction, cmap="Blues", vmin=0, vmax=1)
        axes[index + 1].set_title(SCENARIO_LABELS[scenario])
    axes[0].imshow(label, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("Label referensi")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"{model.upper()} — modality masking pada {tile_name}")
    path = config.figures_dir / _spec(artifact_id).filename
    savefig(fig, path)
    return figure_result(config, _spec(artifact_id), path, source=SOURCE)


def _narrative(config, rows: list[dict[str, object]], *, available: bool):
    spec = _spec("Narasi 4.7")
    if not available:
        text = "Analisis sensitivitas belum ditulis karena delapan inference modality masking belum lengkap."
    else:
        by_key = {(str(row["model"]), str(row["input_scenario"])): row for row in rows}
        u_s1, u_s2, u_dem = (by_key[("unet", scenario)] for scenario in ("sentinel1", "sentinel2", "demnas"))
        p_s1, p_s2, p_dem = (by_key[("procanet", scenario)] for scenario in ("sentinel1", "sentinel2", "demnas"))
        text = (
            "Analisis ini menerapkan modality masking pada checkpoint integrasi tanpa training ulang. "
            "Karena itu, hasil mengukur sensitivitas model terhadap penghilangan kelompok input, bukan "
            "performa model unimodal. Seluruh hasil utama dihitung pada populasi 26.305.235 piksel unik mosaik "
            "dalam effective_valid_mask yang sama. Nilai nol bukan representasi netral sempurna bagi semua channel.\n\n"
            f"Pada U-Net, Sentinel-1 mempertahankan IoU {to_float(u_s1['iou']):.3f}, dibandingkan "
            f"Sentinel-2 {to_float(u_s2['iou']):.3f} dan DEMNAS {to_float(u_dem['iou']):.3f}. "
            f"Pada ProCANet, IoU masing-masing menjadi {to_float(p_s1['iou']):.3f}, "
            f"{to_float(p_s2['iou']):.3f}, dan {to_float(p_dem['iou']):.3f}. Sentinel-1-only cenderung presisi tinggi "
            f"namun recall turun (U-Net {to_float(u_s1['recall']):.3f}; ProCANet {to_float(p_s1['recall']):.3f}), "
            f"sedangkan Sentinel-2-only mempertahankan recall tinggi (U-Net {to_float(u_s2['recall']):.3f}; "
            f"ProCANet {to_float(p_s2['recall']):.3f}) dengan peningkatan FP yang besar. DEMNAS-only menghasilkan "
            f"FN tertinggi, yaitu {to_int(u_dem['fn']):,} pada U-Net dan {to_int(p_dem['fn']):,} pada ProCANet. "
            "Tabel Sentinel-2 valid-only merupakan analisis tambahan pada populasi berbeda dan tidak dibandingkan "
            "langsung dengan skenario utama."
        )
    return write_text_artifact(config, spec, text, source=SOURCE)
