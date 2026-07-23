"""空间树突卷积结构和梯度回归测试。"""
from __future__ import annotations

import unittest

import torch
from torch import nn

from alfoil_dendritic_conv.model import (
    Dendritic,
    DendriticConvDetector,
    PlainConvDetector,
    SpatialDendriticConv2d,
    SpatialSynapse,
)
from alfoil_dnm.loss import detector_loss


class SpatialDendriticConvTests(unittest.TestCase):
    def test_four_half_gates_produce_one_sixteenth(self) -> None:
        synapse = SpatialSynapse(out_channels=2, branches=3)
        with torch.no_grad():
            synapse.raw_weight.zero_()
            synapse.raw_threshold.zero_()
        positions = tuple(torch.randn(1, 2, 3, 4, 5) for _ in range(4))
        gates = synapse(positions)
        self.assertTrue(all(torch.equal(gate, torch.full_like(gate, 0.5)) for gate in gates))
        branches = Dendritic()(gates)
        self.assertTrue(torch.allclose(branches, torch.full_like(branches, 0.5 ** 4)))

    def test_stride_one_preserves_spatial_shape_and_gradient(self) -> None:
        layer = SpatialDendriticConv2d(4, 6, branches=2)
        inputs = torch.randn(2, 4, 11, 13, requires_grad=True)
        output = layer(inputs)
        self.assertEqual(output.shape, (2, 6, 11, 13))
        output.mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_replacement_layer_has_no_spatial_convolution(self) -> None:
        layer = SpatialDendriticConv2d(8, 8, branches=2)
        convolutions = [module for module in layer.modules() if isinstance(module, nn.Conv2d)]
        self.assertEqual(len(convolutions), 1)
        self.assertEqual(convolutions[0].kernel_size, (1, 1))

    def test_one_and_two_layer_models_have_expected_output(self) -> None:
        for replacement_count in (1, 2):
            model = DendriticConvDetector(
                num_classes=3, width=8, branches=2, replace_layers=replacement_count
            )
            dendritic_layers = sum(
                isinstance(module, SpatialDendriticConv2d) for module in model.modules()
            )
            self.assertEqual(dendritic_layers, replacement_count)
            prediction = model(torch.randn(2, 3, 64, 64))
            self.assertEqual(prediction.shape, (2, 8, 8, 8))
            targets = [
                torch.tensor([[0.0, 0.5, 0.5, 0.2, 0.2]]),
                torch.tensor([[2.0, 0.25, 0.25, 0.1, 0.1]]),
            ]
            loss, _ = detector_loss(prediction, targets, num_classes=3)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            first_dendritic_layer = next(
                module for module in model.modules()
                if isinstance(module, SpatialDendriticConv2d)
            )
            gradient = first_dendritic_layer.synapse.raw_weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_plain_control_has_same_output_without_dendritic_layer(self) -> None:
        model = PlainConvDetector(num_classes=3, width=8)
        dendritic_layers = sum(
            isinstance(module, SpatialDendriticConv2d) for module in model.modules()
        )
        self.assertEqual(dendritic_layers, 0)
        self.assertEqual(model(torch.randn(2, 3, 64, 64)).shape, (2, 8, 8, 8))


if __name__ == "__main__":
    unittest.main()
