"""以 2x2 空间树突运算替换普通卷积的轻量检测模型。

与现有 DNM 融合头不同，本模块让树突层直接承担局部空间特征提取：每个树突
分支读取一个 2x2 邻域，四个非线性突触响应相乘后映射为当前位置的一个数。
该操作固定使用 stride=1，不改变特征图尺寸。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _inverse_softplus(value: float) -> float:
    """返回满足 ``softplus(x)=value`` 的 x。"""
    return math.log(math.expm1(value))


class ConvBNAct(nn.Sequential):
    """未被替换位置使用的普通 3x3 卷积单元。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class SpatialSynapse(nn.Module):
    """对每个输出通道、分支和 2x2 位置计算非线性突触响应。"""

    patch_items = 4

    def __init__(self, out_channels: int, branches: int) -> None:
        super().__init__()
        shape = (out_channels, branches, self.patch_items)
        self.raw_weight = nn.Parameter(torch.empty(shape))
        self.raw_threshold = nn.Parameter(torch.empty(shape))
        self.raw_distance = nn.Parameter(
            torch.full(shape, _inverse_softplus(1.0))
        )
        nn.init.uniform_(self.raw_weight, -0.8, 0.8)
        nn.init.uniform_(self.raw_threshold, -0.35, 0.35)

    def biological_parameters(self) -> tuple[Tensor, Tensor, Tensor]:
        """返回有界权重、阈值和严格为正的突触距离。"""
        weight = torch.tanh(self.raw_weight)
        threshold = 1.5 * torch.tanh(self.raw_threshold)
        distance = F.softplus(self.raw_distance) + 1e-4
        return weight, threshold, distance

    def forward(self, positions: tuple[Tensor, Tensor, Tensor, Tensor]) -> tuple[Tensor, ...]:
        if len(positions) != self.patch_items:
            raise ValueError("2x2 空间树突必须接收四个局部位置")
        weight, threshold, distance = self.biological_parameters()
        gates = []
        for index, position in enumerate(positions):
            view = (1, weight.shape[0], weight.shape[1], 1, 1)
            gates.append(
                torch.sigmoid(
                    (position * weight[:, :, index].view(view)
                     - threshold[:, :, index].view(view))
                    / distance[:, :, index].view(view)
                )
            )
        return tuple(gates)


class Dendritic(nn.Module):
    """将同一分支的四个空间突触响应相乘。"""

    def forward(self, gates: tuple[Tensor, ...]) -> Tensor:
        branch = gates[0]
        for gate in gates[1:]:
            branch = branch * gate
        return branch


class Membrane(nn.Module):
    """对一个输出通道的所有树突分支求和。"""

    def forward(self, branches: Tensor) -> Tensor:
        return branches.sum(dim=2)


class Soma(nn.Module):
    """使用可学习正斜率和阈值生成胞体响应。"""

    def __init__(self, channels: int, initial_threshold: float) -> None:
        super().__init__()
        self.raw_slope = nn.Parameter(
            torch.full((channels,), _inverse_softplus(1.0))
        )
        self.threshold = nn.Parameter(
            torch.full((channels,), float(initial_threshold))
        )

    def forward(self, membrane: Tensor) -> Tensor:
        slope = F.softplus(self.raw_slope).view(1, -1, 1, 1) + 1e-4
        threshold = self.threshold.view(1, -1, 1, 1)
        return torch.sigmoid(slope * (membrane - threshold))


class SpatialDendriticConv2d(nn.Module):
    """保持尺寸的 2x2、stride=1 空间树突卷积。

    1x1 投影只负责混合输入通道并为各树突分支生成输入，不聚合空间邻域；
    2x2 范围内唯一的空间聚合由树突乘积完成。右边和下边采用复制填充，使输出
    高宽和输入完全一致。
    """

    kernel_size = 2
    stride = 1

    def __init__(self, in_channels: int, out_channels: int, branches: int = 4) -> None:
        super().__init__()
        if branches < 1:
            raise ValueError("branches 必须大于等于 1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.branches = branches
        self.channel_projection = nn.Conv2d(
            in_channels, out_channels * branches, 1, bias=False
        )
        self.synapse = SpatialSynapse(out_channels, branches)
        self.dendritic = Dendritic()
        self.membrane = Membrane()
        expected_membrane = branches * (0.5 ** SpatialSynapse.patch_items)
        self.soma = Soma(out_channels, expected_membrane)
        self.output_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU(inplace=True)

    def _local_positions(self, projected: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, _, height, width = projected.shape
        padded = F.pad(projected, (0, 1, 0, 1), mode="replicate")
        offsets = ((0, 0), (0, 1), (1, 0), (1, 1))
        return tuple(
            padded[:, :, dy:dy + height, dx:dx + width].reshape(
                batch, self.out_channels, self.branches, height, width
            )
            for dy, dx in offsets
        )

    def forward(self, x: Tensor) -> Tensor:
        projected = self.channel_projection(x)
        positions = self._local_positions(projected)
        synapses = self.synapse(positions)
        branches = self.dendritic(synapses)
        membrane = self.membrane(branches)
        soma = self.soma(membrane)
        return self.activation(self.output_norm(soma))


class DendriticConvBackbone(nn.Module):
    """将骨干末端一个或两个 stride=1 卷积替换为空间树突卷积。"""

    def __init__(self, width: int = 32, branches: int = 4, replace_layers: int = 1) -> None:
        super().__init__()
        if replace_layers not in {1, 2}:
            raise ValueError("replace_layers 只支持 1 或 2")
        middle = width * 2
        output = width * 4
        middle_refine: nn.Module
        if replace_layers == 2:
            middle_refine = SpatialDendriticConv2d(middle, middle, branches)
        else:
            middle_refine = ConvBNAct(middle, middle)
        self.layers = nn.Sequential(
            ConvBNAct(3, width, 2),
            ConvBNAct(width, width),
            ConvBNAct(width, middle, 2),
            middle_refine,
            ConvBNAct(middle, output, 2),
            SpatialDendriticConv2d(output, output, branches),
        )
        self.out_channels = output

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class PlainConvBackbone(nn.Module):
    """与树突替换实验深度和通道完全一致的普通卷积骨干。"""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        middle = width * 2
        output = width * 4
        self.layers = nn.Sequential(
            ConvBNAct(3, width, 2),
            ConvBNAct(width, width),
            ConvBNAct(width, middle, 2),
            ConvBNAct(middle, middle),
            ConvBNAct(middle, output, 2),
            ConvBNAct(output, output),
        )
        self.out_channels = output

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class _DetectorBase(nn.Module):
    """普通卷积和树突替换实验共用的检测头。"""

    stride = 8

    def __init__(self, num_classes: int, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        channels = self.backbone.out_channels
        self.stem = ConvBNAct(channels, channels)
        self.objectness = nn.Conv2d(channels, 1, 1)
        self.box = nn.Conv2d(channels, 4, 1)
        self.classes = nn.Conv2d(channels, num_classes, 1)
        nn.init.constant_(self.objectness.bias, -4.0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.stem(self.backbone(x))
        return torch.cat(
            (self.objectness(features), self.box(features), self.classes(features)), dim=1
        )


class DendriticConvDetector(_DetectorBase):
    """以空间树突卷积为骨干局部算子的单尺度无锚框检测器。"""

    def __init__(
        self,
        num_classes: int,
        width: int = 32,
        branches: int = 4,
        replace_layers: int = 1,
    ) -> None:
        backbone = DendriticConvBackbone(width, branches, replace_layers)
        super().__init__(num_classes, backbone)


class PlainConvDetector(_DetectorBase):
    """除被替换层恢复为普通 3x3 卷积外，其余完全相同的对照模型。"""

    def __init__(self, num_classes: int, width: int = 32) -> None:
        super().__init__(num_classes, PlainConvBackbone(width))
