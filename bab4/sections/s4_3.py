from __future__ import annotations

import matplotlib.pyplot as plt

from bab4.artifacts import ALL_ARTIFACTS
from bab4.common import CV_REGIONS, SPATIAL_CV_FOLDS, TEST_REGION, fmt_float, read_csv_row_map, to_int
from bab4.plots import savefig, setup_style
from bab4.sections.base import section_result
from bab4.writer import figure_result, write_table, write_text_artifact


def _spec(artifact_id: str):
    return next(spec for spec in ALL_ARTIFACTS if spec.artifact_id == artifact_id)


def generate_4_3(config):
    fold_rows = _fold_rows(config)
    artifacts = [
        write_table(config, _spec("Tabel 4.9"), fold_rows, source="scripts/preprocessing_utils.py;dataset/preprocessing_summary.csv"),
        _figure_fold_counts(config, fold_rows),
        _narrative(config),
    ]
    return section_result("4.3", artifacts)


def _fold_rows(config) -> list[dict[str, object]]:
    summary = read_csv_row_map(config.dataset_root / "preprocessing_summary.csv")
    test_tiles = to_int(summary[TEST_REGION].get("tile_count"))
    rows = []
    for idx, val_regions in enumerate(SPATIAL_CV_FOLDS):
        train_regions = tuple(region for region in CV_REGIONS if region not in val_regions)
        train_tiles = sum(to_int(summary[region].get("tile_count")) for region in train_regions)
        val_tiles = sum(to_int(summary[region].get("tile_count")) for region in val_regions)
        rows.append(
            {
                "fold": idx,
                "validation_regions": ", ".join(val_regions),
                "training_regions": ", ".join(train_regions),
                "held_out_final_test": TEST_REGION,
                "train_tile_count": train_tiles,
                "validation_tile_count": val_tiles,
                "final_test_tile_count": test_tiles,
                "validation_pct_of_cv_tiles": fmt_float(val_tiles / max(train_tiles + val_tiles, 1) * 100),
            }
        )
    return rows


def _figure_fold_counts(config, rows):
    spec = _spec("Gambar 4.7")
    labels = [f"Fold {row['fold']}" for row in rows]
    train = [int(row["train_tile_count"]) for row in rows]
    val = [int(row["validation_tile_count"]) for row in rows]
    test = [int(row["final_test_tile_count"]) for row in rows]
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(rows))
    ax.bar([i - 0.25 for i in x], train, width=0.25, label="Train", color="#2563eb")
    ax.bar(x, val, width=0.25, label="Validation", color="#f97316")
    ax.bar([i + 0.25 for i in x], test, width=0.25, label="Final test", color="#16a34a")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Jumlah tile")
    ax.set_title("Jumlah tile train, validation, dan final test per fold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = config.figures_dir / spec.filename
    savefig(fig, path)
    return figure_result(config, spec, path, source="scripts/preprocessing_utils.py;dataset/preprocessing_summary.csv")


def _narrative(config):
    spec = _spec("Narasi 4.3")
    text = """
    Pembagian eksperimen dibuat ulang dari definisi fold di `scripts.preprocessing_utils`.
    Aceh_Utara dikunci sebagai final test region, sedangkan sepuluh wilayah lain membentuk
    5-fold spatial cross-validation. Skema ini menghindari spatial leakage yang dapat terjadi
    jika tile dari wilayah yang sama dibagi acak ke train dan test.
    """
    return write_text_artifact(config, spec, text, source="scripts/preprocessing_utils.py;dataset/preprocessing_summary.csv")
