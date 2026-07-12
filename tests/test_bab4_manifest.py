import unittest
from pathlib import Path

from bab4.artifacts import ALL_ARTIFACTS, REPORT_FIGURES, REPORT_TABLES


class Bab4ManifestTests(unittest.TestCase):
    def test_report_artifact_counts_match_bab4_pdf_numbering(self):
        self.assertEqual(len(REPORT_TABLES), 18)
        self.assertEqual(len(REPORT_FIGURES), 17)

    def test_artifact_ids_are_unique(self):
        ids = [spec.artifact_id for spec in ALL_ARTIFACTS]

        self.assertEqual(len(ids), len(set(ids)))

    def test_report_artifacts_have_fresh_output_filenames(self):
        for spec in REPORT_TABLES + REPORT_FIGURES:
            self.assertTrue(spec.filename)
            self.assertNotIn("outputs/bab4", spec.filename)

    def test_main_generators_do_not_call_legacy_copy(self):
        root = Path(__file__).parents[1]
        generator_files = [root / "bab4" / "run_all.py", *sorted((root / "bab4" / "sections").glob("s4_*.py"))]

        for path in generator_files:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("copy_from_legacy", text, msg=str(path))
            self.assertNotIn("materialize_section", text, msg=str(path))

    def test_table_4_1_generator_uses_feature_valid_mask(self):
        root = Path(__file__).parents[1]
        text = (root / "bab4" / "sections" / "s4_1_1.py").read_text(encoding="utf-8")

        self.assertIn("masked_band_stats", text)
        self.assertIn("feature_valid_mask.tif", text)

    def test_sections_4_4_through_4_9_do_not_hardcode_final_run(self):
        root = Path(__file__).parents[1]
        target_files = [
            root / "bab4" / "sections" / "s4_4.py",
            root / "bab4" / "sections" / "s4_5_6.py",
            root / "bab4" / "sections" / "s4_7.py",
        ]

        for path in target_files:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("runs/final", text, msg=str(path))
            self.assertNotIn('runs_root / "final"', text, msg=str(path))


if __name__ == "__main__":
    unittest.main()
