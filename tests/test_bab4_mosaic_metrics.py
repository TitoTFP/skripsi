import tempfile
import unittest
from pathlib import Path

import numpy as np
from osgeo import gdal

from bab4.sections.s4_5_6 import _mosaic_binary_stats


def write_raster(path: Path, array: np.ndarray, *, x_origin: float = 100.0) -> None:
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), array.shape[1], array.shape[0], 1, gdal.GDT_Float32)
    dataset.SetGeoTransform((x_origin, 10.0, 0.0, 200.0, 0.0, -10.0))
    dataset.SetProjection('LOCAL_CS["unit-test"]')
    dataset.GetRasterBand(1).WriteArray(array.astype(np.float32))
    dataset.FlushCache()
    dataset = None


class Bab4MosaicMetricTests(unittest.TestCase):
    def test_threshold_mask_and_unique_pixel_confusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probability = root / "probability.tif"
            effective = root / "effective.tif"
            label = root / "label.tif"
            write_raster(probability, np.array([[0.50, 0.49, 0.80], [0.10, 0.90, 0.70]]))
            write_raster(effective, np.array([[1, 1, 1], [1, 0, 0]]))
            write_raster(label, np.array([[1, 1, 0], [0, 1, 0]]))

            stats = _mosaic_binary_stats(probability, effective, label, threshold=0.5, block_size=1)

            self.assertEqual(
                stats,
                {"tp": 1, "tn": 1, "fp": 1, "fn": 1, "evaluated_unique_pixels": 4},
            )

    def test_rejects_misaligned_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probability = root / "probability.tif"
            effective = root / "effective.tif"
            label = root / "label.tif"
            write_raster(probability, np.full((2, 2), 0.5))
            write_raster(effective, np.ones((2, 2)), x_origin=110.0)
            write_raster(label, np.ones((2, 2)))

            with self.assertRaisesRegex(ValueError, "grid raster tidak selaras"):
                _mosaic_binary_stats(probability, effective, label, threshold=0.5)

    def test_rejects_incomplete_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probability = root / "probability.tif"
            effective = root / "effective.tif"
            missing_label = root / "missing-label.tif"
            write_raster(probability, np.full((2, 2), 0.5))
            write_raster(effective, np.ones((2, 2)))

            with self.assertRaisesRegex(FileNotFoundError, "sumber raster mosaik tidak ditemukan"):
                _mosaic_binary_stats(probability, effective, missing_label, threshold=0.5)


if __name__ == "__main__":
    unittest.main()
