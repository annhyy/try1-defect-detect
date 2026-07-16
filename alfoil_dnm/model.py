"""用于缺陷检测的乘性树突神经元模块。

该实现遵循 TNSE_Code/DNM_models/DNM_models.py 的四层顺序：
Synapse（突触）→ Dendritic（树突）→ Membrane（膜层）→ Soma（胞体）。
与原始全连接 DNM 的区别仅在于：卷积特征先被投影到少量局部特征，避免在
数百个通道上直接连乘而导致数值下溢。
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBNAct(nn.Sequential):
    """3×3 卷积、批归一化和 SiLU 激活组成的轻量卷积单元。"""

    def __init__(self, c1: int, c2: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(c1, c2, 3, stride, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )


class Synapse(nn.Module):
    """突触层：对每个输出通道、每个树突分支计算非线性突触响应。

    输入为 ``[B, F, H, W]``，输出为 ``[B, H, W, O, M, F]``：
    B 为批大小，F 为每个分支的局部输入数，O 为输出通道，M 为分支数。
    计算公式与 TNSE 中的 Synapse 一致：sigmoid(-k * (w*x - q))。
    """

    def __init__(self, features: int, out_channels: int, branches: int, k: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, branches, features))
        self.threshold = nn.Parameter(torch.empty(out_channels, branches, features))
        self.k = nn.Parameter(torch.tensor(float(k)))
        nn.init.xavier_uniform_(self.weight)
        nn.init.uniform_(self.threshold, -0.25, 0.25)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1).unsqueeze(3).unsqueeze(4)
        # 广播后每个空间位置都拥有 O×M 个分支，分支内拥有 F 个突触。
        return torch.sigmoid(-self.k * (x * self.weight - self.threshold))


class Dendritic(nn.Module):
    """树突层：对同一分支内的 F 个突触响应执行乘性整合。"""

    def forward(self, x: Tensor) -> Tensor:
        # 与 TNSE 的 torch.prod(x, dim=2) 完全同构；这里只是特征维在最后一维。
        return torch.prod(x, dim=-1)


class Membrane(nn.Module):
    """膜层：对一个胞体下的 M 个树突分支输出求和。"""

    def forward(self, x: Tensor) -> Tensor:
        return torch.sum(x, dim=-1)


class Soma(nn.Module):
    """胞体层：对膜电位做可学习阈值的 S 形激活。"""

    def __init__(self, out_channels: int, k: float = 1.0, qs: float = 1.0) -> None:
        super().__init__()
        self.k = nn.Parameter(torch.full((1, 1, 1, out_channels), float(k)))
        self.qs = nn.Parameter(torch.full((1, 1, 1, out_channels), float(qs)))

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.k * (x - self.qs))


class DNMConv(nn.Module):
    """将经典四层 DNM 映射到二维特征图的卷积树突块。

    ``feature_count`` 控制单个分支内的连乘项数。默认值 4 使直接乘性树突
    仍具有可解释性，同时显著降低高维连乘的梯度消失风险。
    """

    def __init__(self, in_channels: int, out_channels: int, branches: int = 4, feature_count: int = 4) -> None:
        super().__init__()
        self.local_projection = nn.Conv2d(in_channels, feature_count, 1, bias=False)
        self.synapse = Synapse(feature_count, out_channels, branches)
        self.dendritic = Dendritic()
        self.membrane = Membrane()
        self.soma = Soma(out_channels)
        # 残差支路不改变 DNM 四层计算，只用于保留卷积骨干的连续特征。
        self.residual = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        local_features = self.local_projection(x)
        synapse_output = self.synapse(local_features)
        dendritic_output = self.dendritic(synapse_output)
        membrane_output = self.membrane(dendritic_output)
        soma_output = self.soma(membrane_output).permute(0, 3, 1, 2)
        return self.fuse(torch.cat((soma_output, self.residual(x)), dim=1))


class TinyBackbone(nn.Module):
    """步长为 8 的轻量卷积骨干，输出保留足够空间分辨率用于小缺陷定位。"""

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
    """以经典四层 DNM 为特征融合核心的单尺度无锚框检测器。"""

    stride = 8

    def __init__(self, num_classes: int, width: int = 32, branches: int = 4, feature_count: int = 4) -> None:
        super().__init__()
        self.backbone = TinyBackbone(width)
        channels = self.backbone.out_channels
        self.dnm = DNMConv(channels, channels, branches, feature_count)
        self.stem = ConvBNAct(channels, channels)
        self.objectness = nn.Conv2d(channels, 1, 1)
        self.box = nn.Conv2d(channels, 4, 1)
        self.classes = nn.Conv2d(channels, num_classes, 1)
        # 初始阶段降低目标置信度，缓解背景网格远多于缺陷网格的类别不平衡。
        nn.init.constant_(self.objectness.bias, -4.0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.stem(self.dnm(self.backbone(x)))
        # 输出通道依次为：目标置信度、框中心偏移与宽高、各类别 logits。
        return torch.cat((self.objectness(features), self.box(features), self.classes(features)), dim=1)
