from __future__ import annotations

import torch
from torch import nn

from training.models.blocks import ConvBlock, Decoder, Encoder


class ProgressiveCrossAttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.self_encoder1 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.Sigmoid())
        self.self_encoder2 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.Sigmoid())
        self.cross_encoder1 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.Sigmoid())
        self.cross_encoder2 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.Sigmoid())

    def forward(self, encoder1: torch.Tensor, encoder2: torch.Tensor) -> torch.Tensor:
        attended1 = encoder1 * self.self_encoder1(encoder1)
        attended2 = encoder2 * self.self_encoder2(encoder2)
        cross1 = attended1 * self.cross_encoder1(attended2)
        cross2 = attended2 * self.cross_encoder2(attended1)
        return cross1 + cross2


class ProCANet(nn.Module):
    def __init__(
        self,
        encoder1_channels: int = 7,
        encoder2_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.encoder1 = Encoder(encoder1_channels, base_channels=base_channels, depth=4)
        self.encoder2 = Encoder(encoder2_channels, base_channels=base_channels, depth=4)
        channels = self.encoder1.out_channels
        self.attention_blocks = nn.ModuleList(ProgressiveCrossAttentionBlock(channel) for channel in channels)
        self.bottleneck1 = ConvBlock(channels[-1], channels[-1] * 2)
        self.bottleneck2 = ConvBlock(channels[-1], channels[-1] * 2)
        self.bottleneck_attention = ProgressiveCrossAttentionBlock(channels[-1] * 2)
        self.decoder = Decoder(channels, channels[-1] * 2, out_channels)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        skips1, pooled1 = self.encoder1(features["encoder1"])
        skips2, pooled2 = self.encoder2(features["encoder2"])
        fused_skips = [
            attention(skip1, skip2)
            for attention, skip1, skip2 in zip(self.attention_blocks, skips1, skips2)
        ]
        bottleneck = self.bottleneck_attention(self.bottleneck1(pooled1), self.bottleneck2(pooled2))
        return self.decoder(bottleneck, fused_skips)
