import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bab4.config import Bab4Config, resolve_repo_root
from bab4.sections.s4_4 import _aggregate_fold_metric, _best_complete_cv_variants, generate_4_4_3


class Bab4TrainingCurvesTests(unittest.TestCase):
    def test_best_variant_requires_complete_five_fold_result(self):
        rows = [
            {"model": "U-Net", "folds_completed": 5, "mean_best_val_iou": 0.64, "mean_best_val_iou_raw": 0.6401, "learning_rate": "5e-5", "weight_decay": "1e-4"},
            {"model": "U-Net", "folds_completed": 5, "mean_best_val_iou": 0.64, "mean_best_val_iou_raw": 0.6402, "learning_rate": "1e-5", "weight_decay": "1e-4"},
            {"model": "U-Net", "folds_completed": 4, "mean_best_val_iou": 0.99, "learning_rate": "1e-4", "weight_decay": "1e-4"},
            {"model": "ProCANet", "folds_completed": 5, "mean_best_val_iou": 0.65, "learning_rate": "1e-4", "weight_decay": "1e-4"},
            {"model": "ProCANet", "folds_completed": 5, "mean_best_val_iou": 0.61, "learning_rate": "5e-5", "weight_decay": "1e-4"},
        ]

        selected = _best_complete_cv_variants(rows)

        self.assertEqual(selected["unet"]["learning_rate"], "1e-5")
        self.assertEqual(selected["unet"]["weight_decay"], "1e-4")
        self.assertEqual(selected["procanet"]["learning_rate"], "1e-4")
        self.assertEqual(selected["procanet"]["weight_decay"], "1e-4")

    def test_aggregation_uses_active_folds_without_carry_forward(self):
        fold_rows = [
            [{"epoch": "1", "train_loss": "1"}, {"epoch": "2", "train_loss": "2"}, {"epoch": "3", "train_loss": "3"}, {"epoch": "4", "train_loss": "4"}],
            [{"epoch": "1", "train_loss": "2"}, {"epoch": "2", "train_loss": "4"}, {"epoch": "3", "train_loss": "6"}],
            [{"epoch": "1", "train_loss": "3"}, {"epoch": "2", "train_loss": "6"}, {"epoch": "3", "train_loss": "9"}],
            [{"epoch": "1", "train_loss": "4"}, {"epoch": "2", "train_loss": "8"}],
            [{"epoch": "1", "train_loss": "5"}],
        ]

        points = _aggregate_fold_metric(fold_rows, "train_loss")

        self.assertEqual([point["active_folds"] for point in points], [5, 4, 3, 1])
        self.assertEqual([point["mean"] for point in points[:3]], [3.0, 5.0, 6.0])
        self.assertAlmostEqual(float(points[0]["std"]), math.sqrt(2.5))
        self.assertAlmostEqual(float(points[1]["std"]), math.sqrt(20 / 3))
        self.assertAlmostEqual(float(points[2]["std"]), 3.0)
        self.assertIsNone(points[3]["mean"])
        self.assertIsNone(points[3]["std"])
        with self.assertRaisesRegex(ValueError, "nilai train_loss tidak valid"):
            _aggregate_fold_metric([[{"epoch": "1", "train_loss": "nan"}]], "train_loss")

    def test_generator_uses_cv_metrics_and_writes_fresh_outputs(self):
        root = resolve_repo_root(Path(__file__).parents[1])
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "bab4_outputs"
            config = Bab4Config.from_repo(root, output_root=output_root)
            result = generate_4_4_3(config)
            artifacts = {artifact.spec.artifact_id: artifact for artifact in result.artifacts}

            figure = artifacts["Gambar 4.11"]
            narrative = artifacts["Narasi 4.4.3"]
            self.assertEqual(figure.status, "exists")
            self.assertTrue(figure.path.exists())
            self.assertGreater(figure.path.stat().st_size, 0)
            self.assertEqual(figure.source.count("metrics.csv"), 10)
            self.assertNotIn("cv_best_checkpoint_eval", figure.source)
            self.assertEqual(narrative.status, "exists")
            self.assertIn("lima `metrics.csv` spatial cross-validation", narrative.path.read_text(encoding="utf-8"))
            self.assertIn("n kurang dari dua", narrative.path.read_text(encoding="utf-8"))

    def test_generator_marks_figure_missing_when_five_fold_input_is_incomplete(self):
        root = resolve_repo_root(Path(__file__).parents[1])
        with tempfile.TemporaryDirectory() as tmp:
            config = Bab4Config.from_repo(root, output_root=Path(tmp) / "bab4_outputs")
            with patch(
                "bab4.sections.s4_4._selected_cv_metric_paths",
                side_effect=ValueError("metrics.csv konfigurasi terbaik tidak lengkap untuk 5-fold CV"),
            ):
                result = generate_4_4_3(config)

            artifacts = {artifact.spec.artifact_id: artifact for artifact in result.artifacts}
            figure = artifacts["Gambar 4.11"]
            self.assertEqual(figure.status, "missing_source")
            self.assertIn("5-fold CV", figure.note)
            self.assertFalse(figure.path.exists())

    def test_generator_marks_figure_missing_when_metric_validation_fails(self):
        root = resolve_repo_root(Path(__file__).parents[1])
        with tempfile.TemporaryDirectory() as tmp:
            config = Bab4Config.from_repo(root, output_root=Path(tmp) / "bab4_outputs")
            with patch(
                "bab4.sections.s4_4._training_curves_figure",
                side_effect=ValueError("kolom lr tidak lengkap pada epoch 3"),
            ):
                result = generate_4_4_3(config)

            artifacts = {artifact.spec.artifact_id: artifact for artifact in result.artifacts}
            figure = artifacts["Gambar 4.11"]
            self.assertEqual(figure.status, "missing_source")
            self.assertIn("kolom lr", figure.note)
            self.assertEqual(figure.source.count("metrics.csv"), 10)


if __name__ == "__main__":
    unittest.main()
