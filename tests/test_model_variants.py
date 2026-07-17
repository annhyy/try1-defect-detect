"""DNM 消融模型的公式、形状和参数匹配回归测试。"""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alfoil_dnm.model_variants import DendriticAggregation, PaperSynapse, build_detector
from alfoil_dnm.loss import detector_loss


class ModelVariantTests(unittest.TestCase):
    def test_log_product_matches_direct_product(self) -> None:
        gates = torch.rand(2, 3, 4, 5).clamp_min(1e-3)
        actual = DendriticAggregation("product")(gates)
        self.assertTrue(torch.allclose(actual, gates.prod(dim=-1), atol=1e-6, rtol=1e-5))

    def test_geometric_mean_matches_definition(self) -> None:
        gates = torch.rand(2, 3, 4, 5).clamp_min(1e-3)
        actual = DendriticAggregation("geometric_mean")(gates)
        expected = gates.prod(dim=-1).pow(1 / gates.shape[-1])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))

    def test_synapse_matches_negative_exponent_paper_formula(self) -> None:
        synapse = PaperSynapse(features=1, out_channels=1, branches=1)
        x = torch.tensor([[[[0.8]]]])
        weight, threshold, distance = synapse.biological_parameters()
        expected = 1 / (1 + torch.exp(-(x.item() * weight - threshold) / distance))
        actual = synapse(x).reshape_as(expected)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-6))

    def test_shapes_and_parameter_matching(self) -> None:
        torch.manual_seed(42)
        image = torch.rand(1, 3, 64, 64)
        models = {name: build_detector(name, 10, 32, 4, 8).eval() for name in ("v2a", "v2b", "conv")}
        with torch.no_grad():
            for model in models.values():
                self.assertEqual(tuple(model(image).shape), (1, 15, 8, 8))
        counts = {name: sum(parameter.numel() for parameter in model.parameters()) for name, model in models.items()}
        self.assertEqual(counts["v2a"], counts["v2b"])
        self.assertLess(abs(counts["conv"] - counts["v2a"]) / counts["v2a"], 0.001)

    def test_detection_backward_is_finite(self) -> None:
        image = torch.rand(2, 3, 64, 64)
        targets = [
            torch.tensor([[1.0, 0.4, 0.5, 0.2, 0.1]]),
            torch.tensor([[6.0, 0.6, 0.4, 0.1, 0.2]]),
        ]
        for variant in ("v2a", "v2b", "conv"):
            model = build_detector(variant, 10, 32, 4, 8).train()
            loss, _ = detector_loss(model(image), targets, 10)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ))


if __name__ == "__main__":
    unittest.main()
