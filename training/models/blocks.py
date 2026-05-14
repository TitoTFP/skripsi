from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32, depth: int = 4) -> None:
        super().__init__()
        channels = [base_channels * (2**idx) for idx in range(depth)]
        blocks = []
        current = in_channels
        for channel in channels:
            blocks.append(ConvBlock(current, channel))
            current = channel
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        skips = []
        for block in self.blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        return skips, x


class Decoder(nn.Module):
    def __init__(self, skip_channels: list[int], bottleneck_channels: int, out_channels: int) -> None:
        super().__init__()
        upconvs = []
        blocks = []
        current = bottleneck_channels
        for skip_channel in reversed(skip_channels):
            upconvs.append(nn.ConvTranspose2d(current, skip_channel, kernel_size=2, stride=2))
            blocks.append(ConvBlock(skip_channel * 2, skip_channel))
            current = skip_channel
        self.upconvs = nn.ModuleList(upconvs)
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Conv2d(current, out_channels, kernel_size=1)

    def forward(self, bottleneck: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        x = bottleneck
        for upconv, block, skip in zip(self.upconvs, self.blocks, reversed(skips)):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return self.final(x)


def _norm(channels: int) -> nn.Module:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)
