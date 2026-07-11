import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    _HAS_RASTER_DEPS = True
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local validation env
    _HAS_RASTER_DEPS = False
    _RASTER_SKIP_REASON = f"raster helper tests require rasterio and numpy: {exc}"
else:
    _RASTER_SKIP_REASON = ""


class Bab4RasterTests(unittest.TestCase):
    @unittest.skipUnless(_HAS_RASTER_DEPS, _RASTER_SKIP_REASON)
    def test_masked_band_stats_uses_only_masked_finite_pixels(self):
        from bab4.raster import masked_band_stats

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raster_path = tmp_path / "values.tif"
            mask_path = tmp_path / "mask.tif"
            transform = from_origin(100, 200, 1, 1)
            values = np.array([[1.0, 2.0, np.nan], [4.0, 8.0, 16.0]], dtype=np.float32)
            mask = np.array([[1, 0, 1], [1, 1, 0]], dtype=np.uint8)

            for path, data, dtype in ((raster_path, values, "float32"), (mask_path, mask, "uint8")):
                with rasterio.open(
                    path,
                    "w",
                    driver="GTiff",
                    height=data.shape[0],
                    width=data.shape[1],
                    count=1,
                    dtype=dtype,
                    transform=transform,
                ) as dataset:
                    dataset.write(data, 1)

            stats = masked_band_stats(raster_path, mask_path)

            self.assertEqual(stats["count"], 3)
            self.assertEqual(stats["total_count"], 6)
            self.assertEqual(stats["min"], 1.0)
            self.assertEqual(stats["max"], 8.0)
            self.assertAlmostEqual(stats["mean"], 13.0 / 3.0)
            self.assertAlmostEqual(stats["std"], float(np.std(np.array([1.0, 4.0, 8.0]))))


if __name__ == "__main__":
    unittest.main()
