"""Stable dendritic detector head inspired by the supplied DNM base models."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBNAct(nn.Sequential):
    def __init__(self, c1: int, c2: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(c1, c2, 3, stride, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )


class DendriticFeature(nn.Module):
    """Synapse gate -> grouped dendrite -> membrane sum.

    The geometric mean is evaluated in log space. It is the stable, local
    equivalent of the direct multiplicative dendritic operation in DNM_Linear2.
    """
    def __init__(self, channels: int, branches: int = 4, eps: float = 1e-6) -> None:
        super().__init__()
        if channels % branches:
            raise ValueError("channels must be divisible by branches")
        self.channels, self.branches, self.eps = channels, branches, eps
        self.synapse = nn.Conv2d(channels, channels, 1, bias=True)
        self.threshold = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gain = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.branch_scale = nn.Parameter(torch.ones(1, branches, 1, 1))
        self.norm = nn.BatchNorm2d(branches)
        self.soma = nn.Sequential(nn.Conv2d(branches, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        gated = torch.sigmoid(self.gain * (self.synapse(x) - self.threshold))
        b, _, h, w = gated.shape
        gated = gated.reshape(b, self.branches, self.channels // self.branches, h, w)
        dendrites = torch.exp(torch.log(gated.clamp_min(self.eps)).mean(dim=2))
        membrane = self.norm(dendrites * self.branch_scale)
        return self.soma(membrane)


class TinyBackbone(nn.Module):
    """Stride-8 backbone intentionally small enough for GTX 1060 / CPU."""
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNAct(3, width, 2), ConvBNAct(width, width),
            ConvBNAct(width, width * 2, 2), ConvBNAct(width * 2, width * 2),
            ConvBNAct(width * 2, width * 4, 2), ConvBNAct(width * 4, width * 4),
        )
        self.out_channels = width * 4

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DendriticDetector(nn.Module):
    """Anchor-free single-scale detector; output [B, 5 + num_classes, H/8, W/8]."""
    stride = 8

    def __init__(self, num_classes: int, width: int = 32, branches: int = 4) -> None:
        super().__init__()
        self.backbone = TinyBackbone(width)
        channels = self.backbone.out_channels
        self.dendrite = DendriticFeature(channels, branches)
        self.stem = ConvBNAct(channels, channels)
        self.objectness = nn.Conv2d(channels, 1, 1)
        self.box = nn.Conv2d(channels, 4, 1)
        self.classes = nn.Conv2d(channels, num_classes, 1)
        nn.init.constant_(self.objectness.bias, -4.0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.stem(self.dendrite(self.backbone(x)))
        # box: sigmoid [center offset in a cell, normalized width, normalized height].
        return torch.cat((self.objectness(features), self.box(features), self.classes(features)), dim=1)
