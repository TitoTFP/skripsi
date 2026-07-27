import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch  # type: ignore[import-not-found]
from osgeo import gdal

from scripts.infer_segmentation import (
    GeoTiffMosaic,
    InferenceStats,
    parse_args,
    resolve_checkpoint_settings,
    write_geotiff,
)
from training.modality_masking import apply_modality_mask  # type: ignore[import-not-found]


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
        self.assertEqual(args.input_scenario, "all")

    def test_parse_args_omitted_region(self):
        args = parse_args(
            [
                "--checkpoint",
                "runs/unet/fold_0/best.pt",
                "--output-dir",
                "runs/inference",
            ]
        )
        self.assertIsNone(args.regions)

    def test_parse_args_empty_region_flag(self):
        args = parse_args(
            [
                "--checkpoint",
                "runs/unet/fold_0/best.pt",
                "--region",
                "--output-dir",
                "runs/inference",
            ]
        )
        self.assertEqual(args.regions, ["all"])

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

    def test_parse_args_accepts_input_scenario(self):
        args = parse_args(
            [
                "--checkpoint",
                "runs/unet/fold_0/best.pt",
                "--output-dir",
                "runs/inference",
                "--input-scenario",
                "sentinel2",
            ]
        )

        self.assertEqual(args.input_scenario, "sentinel2")

    def test_modality_masking_preserves_shape_values_dtype_and_source(self):
        source = torch.arange(7 * 2 * 3, dtype=torch.float32).reshape(1, 7, 2, 3)
        original = source.clone()

        sentinel1 = apply_modality_mask(source, "sentinel1")
        sentinel2 = apply_modality_mask(source, "sentinel2")
        demnas = apply_modality_mask(source, "demnas")
        all_features = apply_modality_mask(source, "all")
        assert isinstance(sentinel1, torch.Tensor)
        assert isinstance(sentinel2, torch.Tensor)
        assert isinstance(demnas, torch.Tensor)
        assert isinstance(all_features, torch.Tensor)

        self.assertEqual(sentinel1.shape, source.shape)
        self.assertEqual(sentinel1.dtype, source.dtype)
        self.assertEqual(sentinel1.device, source.device)
        self.assertTrue(torch.equal(sentinel1[:, :2], source[:, :2]))
        self.assertEqual(int(torch.count_nonzero(sentinel1[:, 2:])), 0)
        self.assertEqual(int(torch.count_nonzero(sentinel2[:, :2])), 0)
        self.assertTrue(torch.equal(sentinel2[:, 2:5], source[:, 2:5]))
        self.assertEqual(int(torch.count_nonzero(sentinel2[:, 5:])), 0)
        self.assertEqual(int(torch.count_nonzero(demnas[:, :5])), 0)
        self.assertTrue(torch.equal(demnas[:, 5:], source[:, 5:]))
        self.assertTrue(torch.equal(all_features, source))
        self.assertTrue(torch.equal(source, original))
        self.assertIsNot(all_features, source)

    def test_modality_masking_handles_procanet_encoder_inputs(self):
        features = {
            "encoder1": torch.ones(1, 7, 2, 2),
            "encoder2": torch.ones(1, 2, 2, 2),
        }

        sentinel1 = apply_modality_mask(features, "sentinel1")
        sentinel2 = apply_modality_mask(features, "sentinel2")
        demnas = apply_modality_mask(features, "demnas")
        assert isinstance(sentinel1, dict)
        assert isinstance(sentinel2, dict)
        assert isinstance(demnas, dict)

        self.assertTrue(torch.equal(sentinel1["encoder2"], features["encoder2"]))
        self.assertEqual(int(torch.count_nonzero(sentinel2["encoder2"])), 0)
        self.assertEqual(int(torch.count_nonzero(demnas["encoder2"])), 0)
        self.assertEqual(int(torch.count_nonzero(sentinel2["encoder1"][:, :2])), 0)
        self.assertEqual(int(torch.count_nonzero(demnas["encoder1"][:, :5])), 0)

    def test_modality_masking_rejects_invalid_scenario_and_shape(self):
        with self.assertRaises(ValueError):
            apply_modality_mask(torch.ones(1, 7, 2, 2), "invalid")
        with self.assertRaises(ValueError):
            apply_modality_mask(torch.ones(1, 3, 2, 2), "sentinel2")

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
        self.assertEqual(summary["precision"], 1.0)
        self.assertEqual(summary["recall"], 1.0)
        self.assertEqual(summary["specificity"], 1.0)
        self.assertEqual(summary["fpr"], 0.0)
        self.assertEqual(summary["fnr"], 0.0)

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

    def test_mosaic_excludes_invalid_observation_from_overlap_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / "reference.tif"
            create_reference(reference, width=2, height=2)
            mosaic = GeoTiffMosaic(reference, threshold=0.5)
            mosaic.add_tile(
                row=0,
                col=0,
                probability=np.array([[[0.9, 0.1], [0.1, 0.1]]], dtype=np.float32),
                effective_valid_mask=np.array([[[True, False], [False, False]]]),
            )
            mosaic.add_tile(
                row=0,
                col=0,
                probability=np.full((1, 2, 2), 0.1, dtype=np.float32),
                effective_valid_mask=np.ones((1, 2, 2), dtype=bool),
            )

            probability, prediction, valid = mosaic.finalize()

            self.assertAlmostEqual(float(probability[0, 0]), 0.5)
            self.assertEqual(prediction[0, 0], 1)
            self.assertAlmostEqual(float(probability[0, 1]), 0.1)
            self.assertEqual(valid[0, 1], 1)

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
