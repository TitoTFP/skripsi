import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from training.losses import masked_bce_dice_loss, masked_dice_loss
from training.train import EarlyStopping, evaluate, save_checkpoint_if_best, train_one_epoch, write_training_config


class SyntheticFloodDataset(Dataset):
    def __init__(self, architecture: str = "unet", length: int = 2):
        self.architecture = architecture
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        y = torch.zeros(1, 16, 16)
        y[:, 4:12, 4:12] = 1.0
        sample = {
            "y": y,
            "valid_mask": torch.ones(1, 16, 16, dtype=torch.bool),
            "metadata": {"index": index},
        }
        if self.architecture == "unet":
            sample["features"] = torch.randn(7, 16, 16)
        else:
            sample["features"] = {
                "encoder1": torch.randn(7, 16, 16),
                "encoder2": torch.randn(2, 16, 16),
            }
        return sample


class TinySegmentationModel(torch.nn.Module):
    def __init__(self, architecture: str = "unet"):
        super().__init__()
        self.architecture = architecture
        in_channels = 7 if architecture == "unet" else 9
        self.conv = torch.nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, features):
        if isinstance(features, dict):
            features = torch.cat([features["encoder1"], features["encoder2"]], dim=1)
        return self.conv(features)


class TrainingLoopTests(unittest.TestCase):
    def test_masked_dice_loss_ignores_invalid_pixels(self):
        logits = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]])
        y = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        valid_mask = torch.tensor([[[[True, False], [False, True]]]])

        loss = masked_dice_loss(logits, y, valid_mask)
        combined = masked_bce_dice_loss(logits, y, valid_mask)

        self.assertLess(loss.item(), 0.001)
        self.assertLess(combined.item(), 0.001)

    def test_train_and_evaluate_one_epoch_with_synthetic_unet_batch(self):
        model = TinySegmentationModel("unet")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = DataLoader(SyntheticFloodDataset("unet"), batch_size=2)

        train_metrics = train_one_epoch(model, loader, optimizer, torch.device("cpu"), max_batches=1)
        val_metrics = evaluate(model, loader, torch.device("cpu"), max_batches=1)

        self.assertIn("loss", train_metrics)
        self.assertIn("iou", val_metrics)
        self.assertIn("dice", val_metrics)

    def test_train_and_evaluate_one_epoch_with_synthetic_procanet_batch(self):
        model = TinySegmentationModel("procanet")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = DataLoader(SyntheticFloodDataset("procanet"), batch_size=2)

        train_metrics = train_one_epoch(model, loader, optimizer, torch.device("cpu"), max_batches=1)
        val_metrics = evaluate(model, loader, torch.device("cpu"), max_batches=1)

        self.assertIn("loss", train_metrics)
        self.assertIn("iou", val_metrics)

    def test_checkpoint_only_updates_when_val_iou_improves(self):
        model = TinySegmentationModel("unet")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            best, saved = save_checkpoint_if_best(
                model,
                optimizer,
                output_dir,
                epoch=1,
                val_iou=0.4,
                best_val_iou=0.3,
                architecture="unet",
                config={"batch_size": 2},
            )
            stale_best, stale_saved = save_checkpoint_if_best(
                model,
                optimizer,
                output_dir,
                epoch=2,
                val_iou=0.2,
                best_val_iou=best,
                architecture="unet",
                config={"batch_size": 2},
            )

            checkpoint = torch.load(output_dir / "best.pt", map_location="cpu")
            self.assertTrue(saved)
            self.assertFalse(stale_saved)
            self.assertEqual(best, 0.4)
            self.assertEqual(stale_best, 0.4)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["best_val_iou"], 0.4)
            self.assertEqual(checkpoint["architecture"], "unet")

    def test_early_stopping_respects_patience_and_min_delta(self):
        stopper = EarlyStopping(patience=2, min_delta=0.01)

        self.assertFalse(stopper.step(0.50))
        self.assertFalse(stopper.step(0.505))
        self.assertTrue(stopper.step(0.506))
        self.assertEqual(stopper.best_score, 0.50)
        self.assertEqual(stopper.bad_epochs, 2)

        self.assertFalse(stopper.step(0.52))
        self.assertEqual(stopper.best_score, 0.52)
        self.assertEqual(stopper.bad_epochs, 0)

    def test_write_training_config_serializes_core_training_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            config_path = write_training_config(
                output_dir,
                {
                    "architecture": "unet",
                    "optimizer": "AdamW",
                    "lr": 1e-4,
                    "batch_size": 2,
                    "epochs": 25,
                    "weight_decay": 1e-4,
                    "early_stopping_patience": 5,
                },
            )

            content = config_path.read_text()
            self.assertIn('"optimizer": "AdamW"', content)
            self.assertIn('"early_stopping_patience": 5', content)


if __name__ == "__main__":
    unittest.main()
