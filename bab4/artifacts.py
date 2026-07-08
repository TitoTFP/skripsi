from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    kind: str
    section: str
    title: str
    filename: str
    legacy_filename: str | None = None
    priority: str = "required"
    source_note: str = ""

    @property
    def expected_filename(self) -> str:
        return self.legacy_filename or self.filename


@dataclass
class ArtifactResult:
    spec: ArtifactSpec
    path: Path
    status: str
    source: str
    note: str = ""
    size_bytes: int = 0
    rows: int | None = None
    columns: int | None = None
    width: int | None = None
    height: int | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "artifact_id": self.spec.artifact_id,
            "kind": self.spec.kind,
            "section": self.spec.section,
            "priority": self.spec.priority,
            "title": self.spec.title,
            "path": str(self.path),
            "status": self.status,
            "source": self.source,
            "note": self.note,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "columns": self.columns,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class SectionResult:
    name: str
    artifacts: list[ArtifactResult] = field(default_factory=list)


@dataclass
class RunResult:
    section_results: list[SectionResult]
    manifest: "ManifestTable"


class ManifestTable:
    def __init__(self, records: Iterable[dict[str, object]] = ()) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    @property
    def empty(self) -> bool:
        return not self.records

    def select(self, columns: Iterable[str]) -> list[dict[str, object]]:
        wanted = list(columns)
        return [{key: record.get(key) for key in wanted} for record in self.records]

    def filter_sections(self, sections: Iterable[str]) -> "ManifestTable":
        wanted = set(sections)
        return ManifestTable(record for record in self.records if record.get("section") in wanted)

    def filter_status_not(self, status: str) -> "ManifestTable":
        return ManifestTable(record for record in self.records if record.get("status") != status)

    def all_status(self, status: str) -> bool:
        return all(record.get("status") == status for record in self.records)

    def paths_startwith(self, prefix: str) -> bool:
        return all(str(record.get("path", "")).startswith(prefix) for record in self.records)

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            status = str(record.get("status", ""))
            counts[status] = counts.get(status, 0) + 1
        return dict(sorted(counts.items()))

    def to_csv(self, path: Path, index: bool = False) -> None:
        import csv

        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.records[0]) if self.records else []
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(self.records)

    def to_pandas(self):
        import pandas as pd

        return pd.DataFrame.from_records(self.records)


REPORT_TABLES: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("Tabel 4.1", "table", "4.1.1", "Statistik Sentinel-1 VV dan VH hasil normalisasi per wilayah", "4_1_1_sentinel1_vv_vh_stats.csv"),
    ArtifactSpec("Tabel 4.2", "table", "4.1.1", "Validitas Sentinel-2 per wilayah", "4_1_1_s2_valid_mask_by_region.csv"),
    ArtifactSpec("Tabel 4.3", "table", "4.1.1", "Statistik Slope dan HAND hasil normalisasi per wilayah", "4_1_1_demnas_slope_hand_stats.csv"),
    ArtifactSpec("Tabel 4.4", "table", "4.1.2", "Hasil verifikasi alignment raster multisensor per wilayah", "4_1_2_alignment_verification.csv"),
    ArtifactSpec("Tabel 4.5", "table", "4.1.2", "Verifikasi layer dalam stack_7ch.tif pada wilayah Aceh Utara", "4_1_2_stack_7ch_layer_verification.csv"),
    ArtifactSpec("Tabel 4.6", "table", "4.2", "Statistik label UNOSAT dan mask per wilayah", "4_2_label_mask_tile_stats.csv"),
    ArtifactSpec("Tabel 4.7", "table", "4.2", "Statistik label UNOSAT dibandingkan valid mask", "4_2_label_valid_mask_percentage.csv", source_note="derived from 4_2_label_mask_tile_stats.csv"),
    ArtifactSpec("Tabel 4.8", "table", "4.2", "Distribusi tile positif dan background per wilayah", "4_2_tile_distribution_by_split_region.csv"),
    ArtifactSpec("Tabel 4.9", "table", "4.3", "Pembagian 5-fold spatial cross-validation", "4_3_five_fold_spatial_cv.csv"),
    ArtifactSpec("Tabel 4.10", "table", "4.4.1", "Spesifikasi implementasi arsitektur U-Net dan ProCANet", "4_4_1_model_architecture_specs.csv"),
    ArtifactSpec("Tabel 4.11", "table", "4.4.1", "Hasil verifikasi forward pass model", "4_4_1_forward_pass_verification.csv"),
    ArtifactSpec("Tabel 4.12", "table", "4.4.2", "Hasil grid search hyperparameter U-Net dan ProCANet", "4_4_2_hyperparameter_tuning_summary.csv"),
    ArtifactSpec("Tabel 4.13", "table", "4.5", "Metrik final U-Net dan ProCANet pada wilayah uji Aceh Utara", "4_5_final_metrics.csv"),
    ArtifactSpec("Tabel 4.14", "table", "4.5", "Confusion matrix piksel pada wilayah uji Aceh Utara", "4_5_confusion_matrix_pixels.csv"),
    ArtifactSpec("Tabel 4.15", "table", "4.6", "Jumlah piksel error map pada tile Aceh_Utara_r001280_c005632", "4_6_error_map_tile_counts.csv"),
    ArtifactSpec("Tabel 4.16", "table", "4.8", "Kondisi data sulit per wilayah", "4_8_difficult_data_case_studies.csv"),
    ArtifactSpec("Tabel 4.17", "table", "4.8", "Contoh tile kondisi ekstrem", "4_8_extreme_tile_selection.csv"),
)


REPORT_FIGURES: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("Gambar 4.1", "figure", "4.1.1", "Perbandingan tile Sentinel-2 valid dan kosong/hampir kosong", "4_1_1_s2_valid_vs_empty_comparison.png"),
    ArtifactSpec("Gambar 4.2", "figure", "4.1.1", "Visualisasi contoh channel input pada wilayah Aceh Utara", "4_1_1_channel_example_aceh_utara.png"),
    ArtifactSpec("Gambar 4.3", "figure", "4.1.2", "Overlay OpenStreetMap terhadap stack preprocessing pada tile Aceh Utara", "4_1_2_osm_overlay_stack_aceh_utara.png"),
    ArtifactSpec("Gambar 4.4", "figure", "4.2", "Visualisasi mask UNOSAT pada wilayah Aceh Utara", "4_2_unosat_mask_panel_aceh_utara.png"),
    ArtifactSpec("Gambar 4.5", "figure", "4.2", "Class imbalance label banjir per wilayah", "4_2_flood_label_distribution.png"),
    ArtifactSpec("Gambar 4.6", "figure", "4.2", "Contoh tile positif dan background-only", "4_2_positive_background_tile_examples.png"),
    ArtifactSpec("Gambar 4.7", "figure", "4.3", "Jumlah tile train, validation, dan final test per fold", "4_3_fold_tile_counts.png"),
    ArtifactSpec("Gambar 4.8", "figure", "4.4.1", "Diagram implementasi U-Net aktual", "4_4_1_unet_architecture_diagram.png"),
    ArtifactSpec("Gambar 4.9", "figure", "4.4.1", "Diagram implementasi ProCANet aktual", "4_4_1_procanet_architecture_diagram.png"),
    ArtifactSpec("Gambar 4.10", "figure", "4.4.2", "Perbandingan mean validation IoU per kombinasi hyperparameter", "4_4_2_hyperparameter_mean_iou_comparison.png"),
    ArtifactSpec("Gambar 4.11", "figure", "4.4.3", "Kurva stabilitas pelatihan pada konfigurasi terbaik", "4_4_3_training_curves.png"),
    ArtifactSpec("Gambar 4.12", "figure", "4.5", "Perbandingan metrik final U-Net dan ProCANet pada wilayah uji Aceh Utara", "4_5_final_metrics_comparison.png"),
    ArtifactSpec("Gambar 4.13", "figure", "4.6", "Panel input, label, dan prediksi segmentasi pada tile Aceh Utara", "4_6_segmentation_panel_aceh_utara.png"),
    ArtifactSpec("Gambar 4.14", "figure", "4.6", "Error map TP/FP/FN/TN pada tile Aceh Utara", "4_6_error_map_aceh_utara.png"),
    ArtifactSpec("Gambar 4.15", "figure", "4.8", "Kondisi data sulit per wilayah", "4_8_difficult_data_case_studies.png"),
    ArtifactSpec("Gambar 4.16", "figure", "4.8", "Kasus Sentinel-2 kosong/hampir kosong pada tile Aceh Tamiang", "4_8_hsv_zero_tile_panel.png"),
    ArtifactSpec("Gambar 4.17", "figure", "4.8", "Kasus topografi sulit atau kandidat radar shadow pada tile Aceh Tamiang", "4_8_topography_radar_shadow_case.png"),
    ArtifactSpec("Gambar 4.18", "figure", "4.8", "Kasus badan air permanen pada tile Aceh Utara", "4_8_permanent_water_case.png"),
)


NARRATIVES: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("Narasi 4.1.1", "narrative", "4.1.1", "Interpretasi karakter input", "4_1_1_input_character_interpretation.md"),
    ArtifactSpec("Narasi 4.1.2", "narrative", "4.1.2", "Interpretasi alignment", "4_1_2_alignment_interpretation.md"),
    ArtifactSpec("Narasi 4.2", "narrative", "4.2", "Interpretasi label dan tile", "4_2_label_tile_interpretation.md"),
    ArtifactSpec("Narasi 4.3", "narrative", "4.3", "Interpretasi spatial split", "4_3_spatial_split_interpretation.md"),
    ArtifactSpec("Narasi 4.4.1", "narrative", "4.4.1", "Interpretasi arsitektur", "4_4_1_architecture_interpretation.md"),
    ArtifactSpec("Narasi 4.4.2", "narrative", "4.4.2", "Interpretasi tuning hyperparameter", "4_4_2_hyperparameter_tuning_interpretation.md"),
    ArtifactSpec("Narasi 4.4.3", "narrative", "4.4.3", "Interpretasi stabilitas training", "4_4_3_training_stability_interpretation.md"),
    ArtifactSpec("Narasi 4.5", "narrative", "4.5", "Interpretasi evaluasi final", "4_5_final_evaluation_interpretation.md"),
    ArtifactSpec("Narasi 4.6", "narrative", "4.6", "Interpretasi visual spasial", "4_6_visual_spatial_interpretation.md"),
    ArtifactSpec("Narasi 4.7", "narrative", "4.7", "Pembahasan efektivitas model", "4_7_unet_procanet_effectiveness_discussion.md"),
    ArtifactSpec("Narasi 4.8", "narrative", "4.8", "Interpretasi data ekstrem dan keterbatasan", "4_8_data_extreme_limitations_interpretation.md"),
)


ALL_ARTIFACTS: tuple[ArtifactSpec, ...] = REPORT_TABLES + REPORT_FIGURES + NARRATIVES


def specs_for_sections(sections: Iterable[str] | None = None, *, include_narratives: bool = True) -> list[ArtifactSpec]:
    specs = list(ALL_ARTIFACTS if include_narratives else REPORT_TABLES + REPORT_FIGURES)
    if sections is None:
        return specs
    wanted = set(sections)
    return [spec for spec in specs if spec.section in wanted or spec.section.split(".")[0] in wanted]


def validate_manifest(config, results: Iterable[SectionResult]) -> ManifestTable:
    records = []
    for section in results:
        records.extend(result.to_record() for result in section.artifacts)
    records.sort(key=lambda row: (str(row["kind"]), str(row["section"]), str(row["artifact_id"])))
    return ManifestTable(records)
