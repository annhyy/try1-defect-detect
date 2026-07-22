"""树突整合启发的轻量目标检测模型。

本模块实现 Dit-CNN 的核心二次神经元：普通卷积负责线性局部建模，
``x^T A x`` 负责同一空间位置上的通道相关性建模。二次项不使用 sigmoid，
这与 Dit-CNN 论文及其官方实现一致，也不同于乘性 DNM 的突触门控结构。
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBNAct(nn.Sequential):
    """3x3 卷积、批归一化和 SiLU 激活组成的轻量卷积单元。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ChannelLayerNorm(nn.Module):
    """在每个空间位置上单独归一化通道，输入输出均为 NCHW。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ChannelQuadratic(nn.Module):
    """计算每个像素处的通道二次积分 ``x^T A x``。

    参数矩阵形状为 ``[输出通道, 输入通道, 输入通道]``。矩阵不强制对称，
    以保持和官方代码相同的参数化方式；其反对称部分在数学上会自然抵消。
    二次参数从零开始，因此训练初始时不会扰动普通卷积分支。
    """

    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.matrix = nn.Parameter(torch.zeros(out_channels, in_channels, in_channels))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"二次积分期望 [B,{self.in_channels},H,W]，实际收到 {tuple(x.shape)}"
            )
        # oij、bihw、bjhw 对应 A_o、x_i、x_j，结果在空间位置之间互不混合。
        return torch.einsum("oij,bihw,bjhw->bohw", self.matrix, x, x)


class DitConvBlock(nn.Module):
    """普通卷积与通道二次积分并联的 Dit 卷积层。

    默认只在较低维的通道子空间内计算二次项，以控制显存和计算量。若
    ``quadratic_channels == channels``，投影层会被移除，此时就是完整的
    ``A in R^(C x C x C)`` 通道二次积分。
    """

    def __init__(self, channels: int, quadratic_channels: int = 16) -> None:
        super().__init__()
        if not 1 <= quadratic_channels <= channels:
            raise ValueError("quadratic_channels 必须处于 [1, channels] 区间")
        self.channels = channels
        self.quadratic_channels = quadratic_channels
        self.input_norm = ChannelLayerNorm(channels)
        self.linear_conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)

        if quadratic_channels == channels:
            self.quadratic_reduce = nn.Identity()
            self.quadratic_expand = nn.Identity()
        else:
            self.quadratic_reduce = nn.Conv2d(channels, quadratic_channels, 1, bias=False)
            self.quadratic_expand = nn.Conv2d(quadratic_channels, channels, 1, bias=False)
        self.quadratic = ChannelQuadratic(quadratic_channels)
        self.output_norm = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.input_norm(x)
        linear = self.linear_conv(normalized)
        reduced = self.quadratic_reduce(normalized)
        quadratic = self.quadratic_expand(self.quadratic(reduced))
        return self.activation(self.output_norm(linear + quadratic))


class DitBackbone(nn.Module):
    """只将最后一个 stride=1 卷积替换为 Dit 卷积的轻量骨干。"""

    def __init__(self, width: int = 32, quadratic_channels: int = 16) -> None:
        super().__init__()
        channels = width * 4
        self.layers = nn.Sequential(
            ConvBNAct(3, width, 2),
            ConvBNAct(width, width),
            ConvBNAct(width, width * 2, 2),
            ConvBNAct(width * 2, width * 2),
            ConvBNAct(width * 2, channels, 2),
            DitConvBlock(channels, quadratic_channels),
        )
        self.out_channels = channels

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DitDetector(nn.Module):
    """采用 Dit-CNN 骨干的单尺度无锚框 APSPC 缺陷检测器。"""

    stride = 8

    def __init__(self, num_classes: int, width: int = 32, quadratic_channels: int = 16) -> None:
        super().__init__()
        self.backbone = DitBackbone(width, quadratic_channels)
        channels = self.backbone.out_channels
        self.stem = ConvBNAct(channels, channels)
        self.objectness = nn.Conv2d(channels, 1, 1)
        self.box = nn.Conv2d(channels, 4, 1)
        self.classes = nn.Conv2d(channels, num_classes, 1)
        nn.init.constant_(self.objectness.bias, -4.0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.stem(self.backbone(x))
        # 输出依次为目标置信度、中心点及宽高、各类别原始 logits。
        return torch.cat(
            (self.objectness(features), self.box(features), self.classes(features)), dim=1
        )
