"""Independent model definitions for the next X-SDD DNM experiments."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from alfoil_dnm.model import TinyBackbone


MODEL_VARIANTS = ("v2a_f4", "v2b_f4", "v1_tuned")
BRANCH_FEATURES = 4


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class PaperF4DNMHead(nn.Module):
    """V2 paper-formula head with exactly four synapses per branch.

    This intentionally retains the V2 input sigmoid, positive branch strength,
    positive soma slope, and coupled soma threshold. The only structural change
    from V2a/V2b is reducing the branch feature count from eight to four.
    """

    def __init__(
        self,
        in_features: int,
        classes: int,
        branches: int,
        aggregation: str,
    ) -> None:
        super().__init__()
        if aggregation not in {"product", "geometric_mean"}:
            raise ValueError(f"Unknown dendritic aggregation: {aggregation}")
        self.aggregation = aggregation
        self.features = BRANCH_FEATURES
        self.projection = nn.Linear(in_features, self.features, bias=False)
        self.input_norm = nn.LayerNorm(self.features)
        shape = (classes, branches, self.features)
        self.raw_weight = nn.Parameter(torch.empty(shape))
        self.raw_threshold = nn.Parameter(torch.empty(shape))
        self.raw_distance = nn.Parameter(torch.full(shape, _inverse_softplus(1.0)))
        self.raw_strength = nn.Parameter(
            torch.full((classes, branches), _inverse_softplus(1.0))
        )
        self.raw_soma_slope = nn.Parameter(
            torch.full((classes,), _inverse_softplus(1.0))
        )
        expected_branch = 0.5**self.features if aggregation == "product" else 0.5
        self.soma_threshold = nn.Parameter(
            torch.full((classes,), branches * expected_branch)
        )
        nn.init.uniform_(self.raw_weight, -0.8, 0.8)
        nn.init.uniform_(self.raw_threshold, -0.35, 0.35)

    def forward(self, x: Tensor) -> Tensor:
        local = torch.sigmoid(self.input_norm(self.projection(x)))
        local = local.unsqueeze(1).unsqueeze(1)
        weight = torch.tanh(self.raw_weight)
        threshold = 1.5 * torch.tanh(self.raw_threshold)
        distance = F.softplus(self.raw_distance) + 1e-4
        gates = torch.sigmoid((local * weight - threshold) / distance)
        log_gates = torch.log(gates.clamp_min(1e-6))
        if self.aggregation == "product":
            branch_outputs = torch.exp(log_gates.sum(dim=-1))
        else:
            branch_outputs = torch.exp(log_gates.mean(dim=-1))
        strength = F.softplus(self.raw_strength) + 1e-4
        membrane = torch.sum(branch_outputs * strength, dim=-1)
        slope = F.softplus(self.raw_soma_slope) + 1e-4
        return slope * (membrane - self.soma_threshold)


class TunedDNMHead(nn.Module):
    """V1-derived head with branch-specific features and linear class logits."""

    def __init__(
        self,
        in_features: int,
        classes: int,
        branches: int,
    ) -> None:
        super().__init__()
        self.branches = branches
        self.features = BRANCH_FEATURES
        projected_features = branches * self.features
        self.projection = nn.Linear(in_features, projected_features, bias=False)
        self.input_norm = nn.LayerNorm(projected_features)
        shape = (classes, branches, self.features)
        self.weight = nn.Parameter(torch.empty(shape))
        self.threshold = nn.Parameter(torch.empty(shape))
        self.synapse_slope = nn.Parameter(torch.tensor(1.0))
        self.branch_strength = nn.Parameter(torch.empty(classes, branches))
        self.class_bias = nn.Parameter(torch.zeros(classes))
        nn.init.xavier_uniform_(self.weight)
        nn.init.uniform_(self.threshold, -0.25, 0.25)
        nn.init.xavier_uniform_(self.branch_strength)

    def forward(self, x: Tensor) -> Tensor:
        local = self.input_norm(self.projection(x))
        local = local.reshape(x.shape[0], self.branches, self.features).unsqueeze(1)
        gates = torch.sigmoid(
            -self.synapse_slope * (local * self.weight - self.threshold)
        )
        branch_outputs = torch.exp(
            torch.log(gates.clamp_min(1e-6)).sum(dim=-1)
        )
        return torch.sum(branch_outputs * self.branch_strength, dim=-1) + self.class_bias


class SurfaceClassifier(nn.Module):
    """Controlled classifier that changes only the head after a shared backbone."""

    def __init__(
        self,
        variant: str,
        num_classes: int,
        width: int,
        branches: int,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.backbone = TinyBackbone(width)
        channels = self.backbone.out_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        if variant == "v2a_f4":
            self.head = PaperF4DNMHead(channels, num_classes, branches, "product")
        elif variant == "v2b_f4":
            self.head = PaperF4DNMHead(
                channels, num_classes, branches, "geometric_mean"
            )
        elif variant == "v1_tuned":
            self.head = TunedDNMHead(channels, num_classes, branches)
        else:
            raise ValueError(f"Unknown classification model: {variant}")

    def forward(self, x: Tensor) -> Tensor:
        features = self.pool(self.backbone(x)).flatten(1)
        return self.head(features)


def build_classifier(
    variant: str,
    num_classes: int,
    width: int = 32,
    branches: int = 4,
) -> SurfaceClassifier:
    """Build one of the isolated F4 or tuned classification experiments."""
    if variant not in MODEL_VARIANTS:
        raise ValueError(
            f"Unknown model {variant!r}; choose from: {', '.join(MODEL_VARIANTS)}"
        )
    if branches < 1:
        raise ValueError("branches must be at least 1")
    return SurfaceClassifier(variant, num_classes, width, branches)
