import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_modality_masking import _summary_row, main


class EvaluateModalityMaskingTests(unittest.TestCase):
    def test_partial_batch_requires_overwrite(self):
        with self.assertRaisesRegex(ValueError, "--max-batches requires --overwrite"):
            main(["--max-batches", "1"])

    def test_smoke_run_with_overwrite_is_non_reportable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.evaluate_modality_masking._run_inference"):
                with self.assertRaisesRegex(ValueError, "invalid inference metrics"):
                    main(["--max-batches", "1", "--overwrite", "--output-dir", tmp])

    def test_summary_rejects_stale_or_incomplete_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "eval_test"
            output_dir.mkdir()
            checkpoint = root / "best.pt"
            checkpoint.touch()
            metrics_path = output_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "architecture": "unet",
                        "checkpoint": str(checkpoint),
                        "input_scenario": "all",
                        "regions": ["Aceh_Utara"],
                        "threshold": 0.5,
                        "max_batches": 1,
                        "complete": False,
                        "tiles_processed_by_region": {"Aceh_Utara": 2},
                        "metrics": [{"region": "aggregate"}],
                    }
                )
            )
            tile_dir = root / "tiles"
            tile_dir.mkdir()
            (tile_dir / "a.npz").touch()
            (tile_dir / "b.npz").touch()

            with patch("scripts.evaluate_modality_masking._tile_directory", return_value=tile_dir):
                with self.assertRaisesRegex(ValueError, "stale or incomplete"):
                    _summary_row("unet", "all", checkpoint, metrics_path, output_dir, "Aceh_Utara", 0.5)


if __name__ == "__main__":
    unittest.main()
