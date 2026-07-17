"""DNM-V2a、DNM-V2b 与普通卷积消融模型。

三种模型共用相同的卷积骨干、单尺度检测头和训练损失，只替换位于二者之间的
特征融合块。这样可以把指标变化尽量归因于树突计算，而不是额外的检测技巧。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from .model import ConvBNAct, DendriticDetector, TinyBackbone
except ImportError:  # 兼容在 alfoil_dnm 目录内直接导入。
    from model import ConvBNAct, DendriticDetector, TinyBackbone


MODEL_VARIANTS = ("v1", "v2a", "v2b", "conv")


def _inverse_softplus(value: float) -> float:
    """返回 ``softplus(x)=value`` 的 x，用于初始化严格为正的生物参数。"""
    return math.log(math.expm1(value))


class PaperSynapse(nn.Module):
    """带独立距离参数的论文突触层。

    论文公式为 ``1 / (1 + exp(-(w*x-theta)/d))``。根据 sigmoid 的定义，
    PyTorch 等价式是 ``sigmoid((w*x-theta)/d)``：论文中的负号位于指数内，
    不能再放到 ``torch.sigmoid`` 的输入前面。

    输入形状为 ``[B,F,H,W]``，输出为 ``[B,H,W,O,M,F]``。同一套参数在所有
    空间位置共享，相当于每个位置并行放置 O 个、每个具有 M 个分支的 DNM。
    """

    def __init__(self, features: int, out_channels: int, branches: int) -> None:
        super().__init__()
        shape = (out_channels, branches, features)
        self.raw_weight = nn.Parameter(torch.empty(shape))
        self.raw_threshold = nn.Parameter(torch.empty(shape))
        self.raw_distance = nn.Parameter(torch.full(shape, _inverse_softplus(1.0)))
        # 原始 DNM 使用分散的随机突触参数。若采用卷积常见的极小 Xavier 值，
        # 所有门都会挤在 0.5 附近，乘积支路再次退化为常数。
        nn.init.uniform_(self.raw_weight, -0.8, 0.8)
        nn.init.uniform_(self.raw_threshold, -0.35, 0.35)

    def biological_parameters(self) -> tuple[Tensor, Tensor, Tensor]:
        """返回有界权重、阈值和严格为正的突触距离。"""
        weight = torch.tanh(self.raw_weight)              # 论文中 w 位于 [-1, 1]
        threshold = 1.5 * torch.tanh(self.raw_threshold) # 论文中 theta 位于 [-1.5, 1.5]
        distance = F.softplus(self.raw_distance) + 1e-4  # d 必须大于 0
        return weight, threshold, distance

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1).unsqueeze(3).unsqueeze(4)
        weight, threshold, distance = self.biological_parameters()
        return torch.sigmoid((x * weight - threshold) / distance)


class DendriticAggregation(nn.Module):
    """分支内乘性整合；V2a 与 V2b 的前向公式只在这一处不同。"""

    def __init__(self, mode: str, epsilon: float = 1e-6) -> None:
        super().__init__()
        if mode not in {"product", "geometric_mean"}:
            raise ValueError(f"未知树突聚合方式：{mode}")
        self.mode = mode
        self.epsilon = epsilon

    def forward(self, gates: Tensor) -> Tensor:
        log_gate = torch.log(gates.clamp_min(self.epsilon))
        if self.mode == "product":
            # V2a：与论文乘积完全等价，只在 log 域计算以避免浮点下溢。
            return torch.exp(log_gate.sum(dim=-1))
        # V2b：归一化乘积。它改变了函数尺度，因此必须作为独立消融模型报告。
        return torch.exp(log_gate.mean(dim=-1))


class WeightedMembrane(nn.Module):
    """按照论文 ``u=sum(v_j*b_j)`` 学习每个树突分支的正强度 v_j。"""

    def __init__(self, out_channels: int, branches: int) -> None:
        super().__init__()
        self.raw_strength = nn.Parameter(
            torch.full((out_channels, branches), _inverse_softplus(1.0))
        )

    def forward(self, branches: Tensor) -> Tensor:
        strength = F.softplus(self.raw_strength) + 1e-4
        return torch.sum(branches * strength, dim=-1)


class StableSoma(nn.Module):
    """带正斜率和可学习阈值的胞体层。"""

    def __init__(self, out_channels: int, initial_threshold: float) -> None:
        super().__init__()
        self.raw_slope = nn.Parameter(torch.full((out_channels,), _inverse_softplus(1.0)))
        self.threshold = nn.Parameter(torch.full((out_channels,), float(initial_threshold)))

    def forward(self, membrane: Tensor) -> Tensor:
        slope = F.softplus(self.raw_slope) + 1e-4
        return torch.sigmoid(slope * (membrane - self.threshold))


class DNMFusionV2(nn.Module):
    """论文参数补全后的二维 DNM 融合块。

    128 通道视觉特征先投影为 F 个连续输入。归一化和 sigmoid 将输入限制在
    0--1，使兴奋性、抑制性及常量连接的解释更接近论文的二值输入分析。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        branches: int = 4,
        feature_count: int = 8,
        aggregation: str = "product",
    ) -> None:
        super().__init__()
        groups = math.gcd(feature_count, 4)
        self.local_projection = nn.Conv2d(in_channels, feature_count, 1, bias=False)
        self.input_norm = nn.GroupNorm(groups, feature_count)
        self.synapse = PaperSynapse(feature_count, out_channels, branches)
        self.dendritic = DendriticAggregation(aggregation)
        self.membrane = WeightedMembrane(out_channels, branches)

        # 两种公式的天然输出尺度不同；分别把胞体阈值放在各自的理论初始均值，
        # 使两组都从约 0.5 的胞体响应起步，避免初始偏置掩盖聚合公式的影响。
        expected_branch = 0.5 ** feature_count if aggregation == "product" else 0.5
        self.soma = StableSoma(out_channels, branches * expected_branch)

        # 保留与 V1 相同的残差和融合外壳，使消融只改变树突内部公式。
        self.residual = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        local_input = torch.sigmoid(self.input_norm(self.local_projection(x)))
        synapse = self.synapse(local_input)
        branches = self.dendritic(synapse)
        membrane = self.membrane(branches)
        soma = self.soma(membrane).permute(0, 3, 1, 2)
        return self.fuse(torch.cat((soma, self.residual(x)), dim=1))


def _matched_control_hidden(channels: int, branches: int, feature_count: int) -> int:
    """计算使普通卷积分支参数量最接近 V2 树突变换的瓶颈宽度。"""
    dnm_transform = (
        channels * feature_count          # 输入投影
        + 2 * feature_count               # GroupNorm
        + 3 * channels * branches * feature_count  # w、theta、d
        + channels * branches             # 分支强度 v
        + 2 * channels                    # 胞体斜率和阈值
    )
    # 普通分支：1x1 C->H、深度 3x3、1x1 H->C，以及三处归一化。
    return max(8, round((dnm_transform - 2 * channels) / (2 * channels + 13)))


class ConvControlFusion(nn.Module):
    """与 V2 DNM 参数量近似匹配的普通卷积融合块。"""

    def __init__(self, in_channels: int, out_channels: int, branches: int, feature_count: int) -> None:
        super().__init__()
        hidden = _matched_control_hidden(out_channels, branches, feature_count)
        self.hidden_channels = hidden
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fuse(torch.cat((self.transform(x), self.residual(x)), dim=1))


class _DetectorBase(nn.Module):
    """三种消融模型共用的骨干、检测前卷积和单尺度检测头。"""

    stride = 8

    def __init__(self, num_classes: int, width: int, fusion: nn.Module) -> None:
        super().__init__()
        self.backbone = TinyBackbone(width)
        channels = self.backbone.out_channels
        self.fusion = fusion
        self.stem = ConvBNAct(channels, channels)
        self.objectness = nn.Conv2d(channels, 1, 1)
        self.box = nn.Conv2d(channels, 4, 1)
        self.classes = nn.Conv2d(channels, num_classes, 1)
        nn.init.constant_(self.objectness.bias, -4.0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.stem(self.fusion(self.backbone(x)))
        return torch.cat((self.objectness(features), self.box(features), self.classes(features)), dim=1)


class DendriticDetectorV2(_DetectorBase):
    """V2a/V2b 检测器；二者仅使用不同的树突分支聚合公式。"""

    def __init__(
        self,
        num_classes: int,
        width: int = 32,
        branches: int = 4,
        feature_count: int = 8,
        aggregation: str = "product",
    ) -> None:
        channels = width * 4
        fusion = DNMFusionV2(channels, channels, branches, feature_count, aggregation)
        super().__init__(num_classes, width, fusion)


class ConvControlDetector(_DetectorBase):
    """不含树突运算、其余接口与 V2 完全一致的参数匹配对照。"""

    def __init__(self, num_classes: int, width: int = 32, branches: int = 4, feature_count: int = 8) -> None:
        channels = width * 4
        fusion = ConvControlFusion(channels, channels, branches, feature_count)
        super().__init__(num_classes, width, fusion)


def build_detector(
    variant: str,
    num_classes: int,
    width: int = 32,
    branches: int = 4,
    feature_count: int = 8,
) -> nn.Module:
    """按 checkpoint/命令行中的名称构建检测器。"""
    if variant == "v1":
        return DendriticDetector(num_classes, width, branches, feature_count)
    if variant == "v2a":
        return DendriticDetectorV2(num_classes, width, branches, feature_count, "product")
    if variant == "v2b":
        return DendriticDetectorV2(num_classes, width, branches, feature_count, "geometric_mean")
    if variant == "conv":
        return ConvControlDetector(num_classes, width, branches, feature_count)
    raise ValueError(f"未知模型 variant={variant!r}，可选值：{', '.join(MODEL_VARIANTS)}")
