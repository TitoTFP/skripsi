import tempfile
import unittest
from pathlib import Path

from bab4.config import Bab4Config, resolve_repo_root


class Bab4ConfigTests(unittest.TestCase):
    def test_from_repo_uses_fresh_bab4_output_root_by_default(self):
        root = resolve_repo_root(Path(__file__).parents[1])

        config = Bab4Config.from_repo(root)

        self.assertEqual(config.output_root, root / "bab4" / "outputs")
        self.assertEqual(config.tables_dir, root / "bab4" / "outputs" / "tables")
        self.assertEqual(config.figures_dir, root / "bab4" / "outputs" / "figures")
        self.assertEqual(config.narratives_dir, root / "bab4" / "outputs" / "narratives")
        self.assertEqual(config.legacy_output_root, root / "outputs" / "bab4")
        self.assertTrue(config.offline)
        self.assertTrue(config.no_retrain)
        self.assertTrue(config.clean_outputs)

    def test_from_repo_accepts_custom_output_root(self):
        root = resolve_repo_root(Path(__file__).parents[1])
        with tempfile.TemporaryDirectory() as tmp:
            config = Bab4Config.from_repo(root, output_root=Path(tmp) / "fresh")

            self.assertEqual(config.output_root, Path(tmp) / "fresh")


if __name__ == "__main__":
    unittest.main()
