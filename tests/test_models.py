import unittest

import torch

from training.models.procanet import ProCANet, ProgressiveCrossAttentionBlock
from training.models.unet import UNet


class ModelTests(unittest.TestCase):
    def test_unet_forward_returns_binary_segmentation_logits(self):
        model = UNet(in_channels=7, out_channels=1, base_channels=8)
        x = torch.randn(2, 7, 64, 64)

        logits = model(x)

        self.assertEqual(logits.shape, (2, 1, 64, 64))

    def test_progressive_cross_attention_preserves_shape(self):
        block = ProgressiveCrossAttentionBlock(channels=16)
        encoder1 = torch.randn(2, 16, 32, 32)
        encoder2 = torch.randn(2, 16, 32, 32)

        fused = block(encoder1, encoder2)

        self.assertEqual(fused.shape, encoder1.shape)

    def test_procanet_forward_returns_binary_segmentation_logits(self):
        model = ProCANet(encoder1_channels=7, encoder2_channels=2, out_channels=1, base_channels=8)
        features = {
            "encoder1": torch.randn(2, 7, 64, 64),
            "encoder2": torch.randn(2, 2, 64, 64),
        }

        logits = model(features)

        self.assertEqual(logits.shape, (2, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
