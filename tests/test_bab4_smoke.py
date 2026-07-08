import tempfile
import unittest
from pathlib import Path

from bab4.artifacts import validate_manifest
from bab4.config import Bab4Config, resolve_repo_root


def _smoke_dependencies_available() -> tuple[bool, str]:
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local validation env
        return False, f"BAB 4 smoke generation requires scientific plotting dependencies: {exc}"
    return True, ""


try:
    _HAS_SMOKE_DEPS, _SMOKE_SKIP_REASON = _smoke_dependencies_available()
except Exception as exc:  # pragma: no cover - defensive for unusual envs
    _HAS_SMOKE_DEPS, _SMOKE_SKIP_REASON = False, str(exc)


class Bab4SmokeTests(unittest.TestCase):
    @unittest.skipUnless(_HAS_SMOKE_DEPS, _SMOKE_SKIP_REASON)
    def test_lightweight_generators_write_fresh_artifacts_to_custom_output(self):
        from bab4.sections.s4_3 import generate_4_3
        from bab4.sections.s4_5_6 import generate_4_5

        root = resolve_repo_root(Path(__file__).parents[1])
        with tempfile.TemporaryDirectory() as tmp:
            config = Bab4Config.from_repo(root, output_root=Path(tmp) / "bab4_outputs")
            config.reset_output_dirs()

            results = [generate_4_3(config), generate_4_5(config)]
            manifest = validate_manifest(config, results)

            self.assertFalse(manifest.empty)
            self.assertTrue(manifest.all_status("exists"))
            self.assertTrue(manifest.paths_startwith(str(config.output_root)))
            for row in manifest:
                self.assertNotIn("/outputs/bab4/", str(row["source"]))


if __name__ == "__main__":
    unittest.main()
