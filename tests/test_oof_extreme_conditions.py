import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.evaluate_oof_extreme_conditions import (
    histogram_quantile,
    metric_summary,
    micro_aggregate,
    new_stats,
    select_best_variants,
    update_stats,
)
from bab4.config import Bab4Config, resolve_repo_root
from bab4.sections.s4_8 import generate_4_8


class OofExtremeConditionTests(unittest.TestCase):
    def test_select_best_variants_requires_all_folds_and_uses_raw_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            for model, winning, losing in (("unet", "grid_lr_5e-5_wd_1e-4", "grid_lr_1e-4_wd_1e-4"), ("procanet", "grid_lr_1e-4_wd_1e-4", "grid_lr_5e-5_wd_1e-4")):
                for fold in range(5):
                    for variant, score in ((winning, 0.70 + fold / 1000), (losing, 0.60)):
                        folder = runs / model / f"fold_{fold}" / variant
                        folder.mkdir(parents=True)
                        (folder / "best.pt").write_bytes(b"checkpoint")
                        with (folder / "metrics.csv").open("w", newline="") as handle:
                            writer = csv.DictWriter(handle, fieldnames=["best_val_iou"])
                            writer.writeheader()
                            writer.writerow({"best_val_iou": score})
                incomplete = runs / model / "fold_0" / "grid_lr_9e-4_wd_1e-4"
                incomplete.mkdir(parents=True)
                (incomplete / "best.pt").write_bytes(b"checkpoint")
                (incomplete / "metrics.csv").write_text("best_val_iou\n0.99\n")

            selected = select_best_variants(runs)

            self.assertEqual(selected["unet"]["variant"], "grid_lr_5e-5_wd_1e-4")
            self.assertEqual(selected["procanet"]["variant"], "grid_lr_1e-4_wd_1e-4")

    def test_condition_stats_micro_aggregation_and_undefined_positive_metrics(self):
        stats = new_stats()
        condition = np.array([[True, True], [True, False]])
        label = np.array([[False, False], [False, True]])
        prediction = np.array([[1, 0], [1, 1]], dtype=np.uint8)
        predicted = np.ones((2, 2), dtype=bool)
        update_stats(stats, condition, label, prediction, predicted)
        summary = metric_summary(stats)

        self.assertEqual(summary["condition_pixels"], 3)
        self.assertEqual(summary["fp"], 2)
        self.assertEqual(summary["tn"], 1)
        self.assertIsNone(summary["iou"])
        self.assertIsNone(summary["dice"])
        self.assertIsNone(summary["precision"])
        self.assertIsNone(summary["recall"])
        self.assertEqual(summary["specificity"], 1 / 3)
        self.assertEqual(summary["fpr"], 2 / 3)

        micro = micro_aggregate(
            [
                {"condition": "permanent_water", "condition_label": "Air", "model": "U-Net", **summary},
                {"condition": "permanent_water", "condition_label": "Air", "model": "U-Net", **summary},
            ]
        )
        self.assertEqual(micro[0]["fp"], 4)
        self.assertEqual(micro[0]["tn"], 2)
        self.assertEqual(micro[0]["fpr"], 2 / 3)

    def test_histogram_quantile_is_deterministic(self):
        histogram = np.array([2, 3, 5, 0], dtype=np.int64)
        self.assertEqual(histogram_quantile(histogram, 0.20), 1 / 65536)
        self.assertEqual(histogram_quantile(histogram, 0.50), 2 / 65536)

    def test_bab4_generator_marks_oof_artifacts_missing_without_results(self):
        root = resolve_repo_root(Path(__file__).parents[1])
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(
                Bab4Config.from_repo(root, output_root=Path(tmp) / "bab4_outputs"),
                runs_root=Path(tmp) / "empty_runs",
            )
            config.reset_output_dirs()
            result = generate_4_8(config)
            artifacts = {artifact.spec.artifact_id: artifact for artifact in result.artifacts}

            self.assertEqual(artifacts["Tabel 4.18"].status, "exists")
            self.assertEqual(artifacts["Tabel 4.20"].status, "missing_source")
            self.assertEqual(artifacts["Gambar 4.18"].status, "missing_source")
            self.assertIn("evaluasi OOF", artifacts["Tabel 4.20"].note)


if __name__ == "__main__":
    unittest.main()
