"""用于钢材表面缺陷分类的树突模型与参数匹配普通卷积对照。"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from alfoil_dnm.model import TinyBackbone


MODEL_VARIANTS = ("v1", "v2a", "v2b", "conv")


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class LegacyDNMHead(nn.Module):
    """保留旧版符号和直接连乘的 DNM-V1 分类头。"""

    def __init__(self, in_features: int, classes: int, branches: int, features: int) -> None:
        super().__init__()
        self.projection = nn.Linear(in_features, features, bias=False)
        self.weight = nn.Parameter(torch.empty(classes, branches, features))
        self.threshold = nn.Parameter(torch.empty(classes, branches, features))
        self.synapse_slope = nn.Parameter(torch.tensor(1.0))
        self.soma_slope = nn.Parameter(torch.ones(classes))
        self.soma_threshold = nn.Parameter(torch.ones(classes))
        nn.init.xavier_uniform_(self.weight)
        nn.init.uniform_(self.threshold, -0.25, 0.25)

    def forward(self, x: Tensor) -> Tensor:
        local = self.projection(x).unsqueeze(1).unsqueeze(1)
        gates = torch.sigmoid(-self.synapse_slope * (local * self.weight - self.threshold))
        membrane = gates.prod(dim=-1).sum(dim=-1)
        # 直接返回胞体 sigmoid 之前的值作为多分类 logits；argmax 与施加单调
        # sigmoid 后一致，但交叉熵能得到更稳定的梯度。
        return self.soma_slope * (membrane - self.soma_threshold)


class PaperDNMHead(nn.Module):
    """包含 w、theta、d、v 的论文公式分类头。

    V2a 使用 log 域精确乘积；V2b 使用 log 域几何平均。输入是整张图像经
    卷积骨干和全局池化后的向量，因此计算结构与经典全连接 DNM 同构。
    """

    def __init__(
        self,
        in_features: int,
        classes: int,
        branches: int,
        features: int,
        aggregation: str,
    ) -> None:
        super().__init__()
        if aggregation not in {"product", "geometric_mean"}:
            raise ValueError(f"未知树突聚合：{aggregation}")
        self.aggregation = aggregation
        self.features = features
        self.projection = nn.Linear(in_features, features, bias=False)
        self.input_norm = nn.LayerNorm(features)
        shape = (classes, branches, features)
        self.raw_weight = nn.Parameter(torch.empty(shape))
        self.raw_threshold = nn.Parameter(torch.empty(shape))
        self.raw_distance = nn.Parameter(torch.full(shape, _inverse_softplus(1.0)))
        self.raw_strength = nn.Parameter(torch.full((classes, branches), _inverse_softplus(1.0)))
        self.raw_soma_slope = nn.Parameter(torch.full((classes,), _inverse_softplus(1.0)))
        expected_branch = 0.5**features if aggregation == "product" else 0.5
        self.soma_threshold = nn.Parameter(torch.full((classes,), branches * expected_branch))
        nn.init.uniform_(self.raw_weight, -0.8, 0.8)
        nn.init.uniform_(self.raw_threshold, -0.35, 0.35)

    def forward(self, x: Tensor) -> Tensor:
        local = torch.sigmoid(self.input_norm(self.projection(x))).unsqueeze(1).unsqueeze(1)
        weight = torch.tanh(self.raw_weight)
        threshold = 1.5 * torch.tanh(self.raw_threshold)
        distance = F.softplus(self.raw_distance) + 1e-4
        # 论文 1/(1+exp(-(w*x-theta)/d)) 等价于下面的 sigmoid 正输入。
        gates = torch.sigmoid((local * weight - threshold) / distance)
        log_gates = torch.log(gates.clamp_min(1e-6))
        if self.aggregation == "product":
            branches = torch.exp(log_gates.sum(dim=-1))
        else:
            branches = torch.exp(log_gates.mean(dim=-1))
        strength = F.softplus(self.raw_strength) + 1e-4
        membrane = torch.sum(branches * strength, dim=-1)
        slope = F.softplus(self.raw_soma_slope) + 1e-4
        return slope * (membrane - self.soma_threshold)


def _matched_hidden(in_features: int, classes: int, branches: int, features: int) -> int:
    """计算与 V2 树突分类头参数量最接近的两层 MLP 隐藏宽度。"""
    dnm_parameters = (
        in_features * features + 2 * features
        + 3 * classes * branches * features
        + classes * branches + 2 * classes
    )
    return max(4, round((dnm_parameters - classes) / (in_features + classes + 1)))


class ConvControlHead(nn.Module):
    """不含树突运算的参数匹配普通神经网络分类头。"""

    def __init__(self, in_features: int, classes: int, branches: int, features: int) -> None:
        super().__init__()
        hidden = _matched_hidden(in_features, classes, branches, features)
        self.hidden_features = hidden
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class SurfaceClassifier(nn.Module):
    """共享轻量卷积骨干、仅替换末端分类头的受控模型。"""

    def __init__(
        self,
        variant: str,
        num_classes: int,
        width: int,
        branches: int,
        feature_count: int,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.backbone = TinyBackbone(width)
        channels = self.backbone.out_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        if variant == "v1":
            self.head = LegacyDNMHead(channels, num_classes, branches, feature_count)
        elif variant == "v2a":
            self.head = PaperDNMHead(channels, num_classes, branches, feature_count, "product")
        elif variant == "v2b":
            self.head = PaperDNMHead(channels, num_classes, branches, feature_count, "geometric_mean")
        elif variant == "conv":
            self.head = ConvControlHead(channels, num_classes, branches, feature_count)
        else:
            raise ValueError(f"未知分类模型：{variant}")

    def forward(self, x: Tensor) -> Tensor:
        features = self.pool(self.backbone(x)).flatten(1)
        return self.head(features)


def build_classifier(
    variant: str,
    num_classes: int,
    width: int = 32,
    branches: int = 4,
    feature_count: int = 8,
) -> nn.Module:
    """按实验名称构建分类模型。"""
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"未知模型 {variant!r}；可选：{', '.join(MODEL_VARIANTS)}")
    return SurfaceClassifier(variant, num_classes, width, branches, feature_count)
