from __future__ import annotations

import torch
from torch import nn

from training.models.blocks import ConvBlock, Decoder, Encoder


class UNet(nn.Module):
    def __init__(self, in_channels: int = 7, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels=base_channels, depth=4)
        bottleneck_channels = self.encoder.out_channels[-1] * 2
        self.bottleneck = ConvBlock(self.encoder.out_channels[-1], bottleneck_channels)
        self.decoder = Decoder(self.encoder.out_channels, bottleneck_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips, pooled = self.encoder(x)
        bottleneck = self.bottleneck(pooled)
        return self.decoder(bottleneck, skips)
