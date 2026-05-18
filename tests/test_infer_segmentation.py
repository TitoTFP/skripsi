import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from osgeo import gdal

from scripts.infer_segmentation import (
    GeoTiffMosaic,
    InferenceStats,
    parse_args,
    resolve_checkpoint_settings,
    write_geotiff,
)


def create_reference(path: Path, width: int = 4, height: int = 4) -> None:
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), width, height, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((100.0, 10.0, 0.0, 200.0, 0.0, -10.0))
    ds.SetProjection('LOCAL_CS["unit-test"]')
    ds.GetRasterBand(1).WriteArray(np.zeros((height, width), dtype=np.float32))
    ds.FlushCache()
    ds = None


class InferSegmentationTests(unittest.TestCase):
    def test_parse_args_accepts_geotiff_and_repeated_regions(self):
        args = parse_args(
            [
                "--checkpoint",
                "runs/unet/fold_0/best.pt",
                "--region",
                "Aceh_Utara",
                "--region",
                "Pidie",
                "--output-dir",
                "runs/inference",
                "--write-geotiff",
                "--threshold",
                "0.6",
            ]
        )

        self.assertEqual(args.regions, ["Aceh_Utara", "Pidie"])
        self.assertTrue(args.write_geotiff)
        self.assertEqual(args.threshold, 0.6)

    def test_resolve_checkpoint_settings_uses_checkpoint_config_before_fallbacks(self):
        checkpoint = {
            "architecture": "unet",
            "config": {"base_channels": 16},
        }

        architecture, base_channels = resolve_checkpoint_settings(
            checkpoint,
            architecture_fallback="procanet",
            base_channels_fallback=32,
        )

        self.assertEqual(architecture, "unet")
        self.assertEqual(base_channels, 16)

    def test_inference_stats_ignore_invalid_pixels(self):
        stats = InferenceStats()
        logits = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]])
        target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        effective_valid = torch.tensor([[[[True, False], [False, True]]]])

        stats.update(logits, target, effective_valid, loss=0.25, threshold=0.5)
        summary = stats.summary()

        self.assertEqual(summary["loss"], 0.25)
        self.assertEqual(summary["iou"], 1.0)
        self.assertEqual(summary["dice"], 1.0)
        self.assertEqual(summary["accuracy"], 1.0)

    def test_mosaic_averages_overlapping_probabilities_and_marks_uncovered_nodata(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / "reference.tif"
            create_reference(reference)
            mosaic = GeoTiffMosaic(reference, threshold=0.5)

            mosaic.add_tile(
                row=0,
                col=0,
                probability=np.full((1, 3, 3), 0.25, dtype=np.float32),
                effective_valid_mask=np.ones((1, 3, 3), dtype=bool),
            )
            mosaic.add_tile(
                row=1,
                col=1,
                probability=np.full((1, 3, 3), 0.75, dtype=np.float32),
                effective_valid_mask=np.ones((1, 3, 3), dtype=bool),
            )
            probability, prediction, valid = mosaic.finalize()

            self.assertEqual(probability[1, 1], 0.5)
            self.assertEqual(prediction[1, 1], 1)
            self.assertEqual(probability[3, 0], -9999.0)
            self.assertEqual(prediction[3, 0], 255)
            self.assertEqual(valid[3, 0], 0)

    def test_write_geotiff_copies_reference_grid_and_nodata(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / "reference.tif"
            output = Path(tmp) / "out.tif"
            create_reference(reference)
            write_geotiff(
                reference_path=reference,
                output_path=output,
                array=np.array([[0.1, -9999.0], [0.8, 0.2]], dtype=np.float32),
                gdal_dtype=gdal.GDT_Float32,
                nodata=-9999.0,
            )

            ref_ds = gdal.Open(str(reference), gdal.GA_ReadOnly)
            out_ds = gdal.Open(str(output), gdal.GA_ReadOnly)
            self.assertEqual(out_ds.GetGeoTransform(), ref_ds.GetGeoTransform())
            self.assertEqual(out_ds.GetProjection(), ref_ds.GetProjection())
            self.assertEqual(out_ds.GetRasterBand(1).GetNoDataValue(), -9999.0)


if __name__ == "__main__":
    unittest.main()
