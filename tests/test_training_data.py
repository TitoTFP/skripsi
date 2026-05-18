import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.augmentations import FeatureAugmentConfig, SpatialTransform, apply_feature_augmentation, apply_spatial_transform
from training.datasets import FloodTileDataset
from training.losses import masked_bce_with_logits
from training.metrics import masked_binary_stats, masked_dice, masked_iou


def write_unet_tile(root: Path, split: str = "train", name: str = "tile.npz") -> Path:
    tile_dir = root / "7ch" / split
    tile_dir.mkdir(parents=True, exist_ok=True)
    path = tile_dir / name
    np.savez_compressed(
        path,
        x=np.arange(7 * 3 * 4, dtype=np.float32).reshape(7, 3, 4),
        y=np.array([[[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]]], dtype=np.uint8),
        valid_mask=np.array([[[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]]], dtype=np.uint8),
        water_river_mask=np.array([[[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]], dtype=np.uint8),
        feature_valid_mask=np.ones((1, 3, 4), dtype=np.uint8),
        s2_valid_mask=np.ones((1, 3, 4), dtype=np.uint8),
        region=np.array("Aceh_Timur"),
        row=np.array(512),
        col=np.array(1024),
        channels=np.array(["VV", "VH", "Hue", "Saturation", "Value", "Slope", "HAND"]),
    )
    return path


def write_unet_region_tile(root: Path, region: str, name: str = "tile.npz") -> Path:
    tile_dir = root / "7ch" / "by_region" / region
    tile_dir.mkdir(parents=True, exist_ok=True)
    path = tile_dir / name
    np.savez_compressed(
        path,
        x=np.ones((7, 3, 4), dtype=np.float32),
        y=np.ones((1, 3, 4), dtype=np.uint8),
        valid_mask=np.ones((1, 3, 4), dtype=np.uint8),
        water_river_mask=np.zeros((1, 3, 4), dtype=np.uint8),
        feature_valid_mask=np.ones((1, 3, 4), dtype=np.uint8),
        s2_valid_mask=np.ones((1, 3, 4), dtype=np.uint8),
        region=np.array(region),
        row=np.array(0),
        col=np.array(0),
        channels=np.array(["VV", "VH", "Hue", "Saturation", "Value", "Slope", "HAND"]),
    )
    return path


def write_procanet_tile(root: Path, split: str = "train", name: str = "tile.npz") -> Path:
    tile_dir = root / "procanet" / split
    tile_dir.mkdir(parents=True, exist_ok=True)
    path = tile_dir / name
    encoder1 = np.arange(7 * 3 * 4, dtype=np.float32).reshape(7, 3, 4)
    np.savez_compressed(
        path,
        x_encoder1=encoder1,
        x_encoder2=encoder1[:2],
        y=np.ones((1, 3, 4), dtype=np.uint8),
        valid_mask=np.ones((1, 3, 4), dtype=np.uint8),
        water_river_mask=np.zeros((1, 3, 4), dtype=np.uint8),
        feature_valid_mask=np.ones((1, 3, 4), dtype=np.uint8),
        s2_valid_mask=np.ones((1, 3, 4), dtype=np.uint8),
        region=np.array("Aceh_Timur"),
        row=np.array(512),
        col=np.array(1024),
        encoder1_channels=np.array(["VV", "VH", "Hue", "Saturation", "Value", "Slope", "HAND"]),
        encoder2_channels=np.array(["VV", "VH"]),
    )
    return path


class TrainingDataTests(unittest.TestCase):
    def test_unet_dataset_reads_tile_features_masks_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tile_path = write_unet_tile(root)

            dataset = FloodTileDataset("train", architecture="unet", root=root, augment=False)
            sample = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertTrue(torch.equal(sample["features"], torch.arange(7 * 3 * 4).reshape(7, 3, 4).float()))
            self.assertEqual(sample["y"].dtype, torch.float32)
            self.assertEqual(sample["valid_mask"].dtype, torch.bool)
            self.assertEqual(sample["metadata"]["region"], "Aceh_Timur")
            self.assertEqual(sample["metadata"]["row"], 512)
            self.assertEqual(sample["metadata"]["col"], 1024)
            self.assertEqual(sample["metadata"]["path"], str(tile_path))
            self.assertEqual(sample["metadata"]["channels"], ("VV", "VH", "Hue", "Saturation", "Value", "Slope", "HAND"))

    def test_unet_dataset_keeps_default_flood_only_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_unet_tile(root)

            sample = FloodTileDataset("train", architecture="unet", root=root, augment=False)[0]

            expected = torch.tensor([[[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]]], dtype=torch.float32)
            self.assertTrue(torch.equal(sample["y"], expected))

    def test_unet_dataset_can_union_water_river_mask_into_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_unet_tile(root)

            sample = FloodTileDataset(
                "train",
                architecture="unet",
                root=root,
                augment=False,
                water_river_as_flood=True,
            )[0]

            expected = torch.tensor([[[1, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 1]]], dtype=torch.float32)
            self.assertTrue(torch.equal(sample["y"], expected))
            self.assertIn("water_river_mask", sample["auxiliary_masks"])

    def test_procanet_dataset_reads_two_encoder_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_procanet_tile(root)

            sample = FloodTileDataset("train", architecture="procanet", root=root, augment=False)[0]

            self.assertEqual(set(sample["features"].keys()), {"encoder1", "encoder2"})
            self.assertEqual(sample["features"]["encoder1"].shape, (7, 3, 4))
            self.assertEqual(sample["features"]["encoder2"].shape, (2, 3, 4))
            self.assertTrue(torch.equal(sample["features"]["encoder2"], sample["features"]["encoder1"][:2]))
            self.assertEqual(sample["metadata"]["channels"]["encoder1"], ("VV", "VH", "Hue", "Saturation", "Value", "Slope", "HAND"))
            self.assertEqual(sample["metadata"]["channels"]["encoder2"], ("VV", "VH"))

    def test_apply_spatial_transform_keeps_features_labels_and_masks_aligned(self):
        features = torch.arange(1 * 2 * 3, dtype=torch.float32).reshape(1, 2, 3)
        y = torch.tensor([[[0, 1, 0], [1, 0, 1]]], dtype=torch.float32)
        valid_mask = torch.tensor([[[1, 0, 1], [0, 1, 1]]], dtype=torch.bool)

        out = apply_spatial_transform(
            {"features": features, "y": y, "valid_mask": valid_mask},
            SpatialTransform(rot90=1, flip_horizontal=True, flip_vertical=False),
        )

        expected_features = torch.flip(torch.rot90(features, 1, dims=(-2, -1)), dims=(-1,))
        expected_y = torch.flip(torch.rot90(y, 1, dims=(-2, -1)), dims=(-1,))
        expected_mask = torch.flip(torch.rot90(valid_mask, 1, dims=(-2, -1)), dims=(-1,))
        self.assertTrue(torch.equal(out["features"], expected_features))
        self.assertTrue(torch.equal(out["y"], expected_y))
        self.assertTrue(torch.equal(out["valid_mask"], expected_mask))

    def test_augmentation_requested_for_non_train_split_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_unet_tile(root, split="val")

            dataset = FloodTileDataset(
                "val",
                architecture="unet",
                root=root,
                augment=True,
                rng=np.random.default_rng(7),
            )
            sample = dataset[0]

            self.assertTrue(torch.equal(sample["features"], torch.arange(7 * 3 * 4).reshape(7, 3, 4).float()))

    def test_fold_dataset_selects_region_level_train_val_and_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for region in (
                "Aceh_Besar",
                "Aceh_Tamiang",
                "Aceh_Timur",
                "Aceh_Utara",
                "Agam",
                "Banda_Aceh",
                "Bireuen",
                "Langsa",
                "Pasaman_Barat",
                "Pidie",
                "Pidie_Jaya",
            ):
                write_unet_region_tile(root, region, name=f"{region}.npz")

            train_regions = {
                FloodTileDataset("train", architecture="unet", root=root, fold=0, augment=False)[idx]["metadata"]["region"]
                for idx in range(8)
            }
            val_regions = {
                FloodTileDataset("val", architecture="unet", root=root, fold=0, augment=False)[idx]["metadata"]["region"]
                for idx in range(2)
            }
            test_dataset = FloodTileDataset("test", architecture="unet", root=root, fold=0, augment=False)

            self.assertEqual(val_regions, {"Pidie", "Pidie_Jaya"})
            self.assertEqual(test_dataset[0]["metadata"]["region"], "Aceh_Utara")
            self.assertNotIn("Aceh_Utara", train_regions)
            self.assertTrue(train_regions.isdisjoint(val_regions))

    def test_feature_augmentation_never_changes_labels_or_masks(self):
        features = torch.ones((7, 3, 4), dtype=torch.float32)
        y = torch.ones((1, 3, 4), dtype=torch.float32)
        valid_mask = torch.ones((1, 3, 4), dtype=torch.bool)
        feature_valid_mask = torch.ones((1, 3, 4), dtype=torch.bool)
        sample = {
            "features": features,
            "y": y.clone(),
            "valid_mask": valid_mask.clone(),
            "auxiliary_masks": {"feature_valid_mask": feature_valid_mask.clone()},
        }

        out = apply_feature_augmentation(
            sample,
            rng=np.random.default_rng(1),
            config=FeatureAugmentConfig(noise_std=0.05, channel_dropout_p=0.0),
        )

        self.assertFalse(torch.equal(out["features"], features))
        self.assertTrue(torch.equal(out["y"], y))
        self.assertTrue(torch.equal(out["valid_mask"], valid_mask))
        self.assertTrue(torch.equal(out["auxiliary_masks"]["feature_valid_mask"], feature_valid_mask))

    def test_feature_augmentation_keeps_procanet_shared_sar_channels_consistent(self):
        encoder1 = torch.ones((7, 3, 4), dtype=torch.float32)
        encoder2 = encoder1[:2].clone()
        sample = {"features": {"encoder1": encoder1, "encoder2": encoder2}}

        out = apply_feature_augmentation(
            sample,
            rng=np.random.default_rng(1),
            config=FeatureAugmentConfig(noise_std=0.05, channel_dropout_p=0.0),
        )

        self.assertTrue(torch.equal(out["features"]["encoder2"], out["features"]["encoder1"][:2]))

    def test_masked_loss_and_metrics_ignore_invalid_pixels(self):
        logits = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]])
        y = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        valid_mask = torch.tensor([[[[True, False], [False, True]]]])

        loss = masked_bce_with_logits(logits, y, valid_mask)
        expected = F.binary_cross_entropy_with_logits(logits[valid_mask], y[valid_mask])
        stats = masked_binary_stats(logits, y, valid_mask)

        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(stats["tp"], 1)
        self.assertEqual(stats["tn"], 1)
        self.assertEqual(stats["fp"], 0)
        self.assertEqual(stats["fn"], 0)
        self.assertEqual(masked_iou(logits, y, valid_mask), 1.0)
        self.assertEqual(masked_dice(logits, y, valid_mask), 1.0)

    def test_masked_loss_returns_differentiable_zero_when_no_valid_pixels(self):
        logits = torch.randn((1, 1, 2, 2), requires_grad=True)
        y = torch.zeros((1, 1, 2, 2))
        valid_mask = torch.zeros((1, 1, 2, 2), dtype=torch.bool)

        loss = masked_bce_with_logits(logits, y, valid_mask)
        loss.backward()

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))


if __name__ == "__main__":
    unittest.main()
