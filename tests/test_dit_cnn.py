"""Dit-CNN 二次积分和检测输出的回归测试。"""
from __future__ import annotations

import unittest

import torch

from alfoil_dit_cnn.model import ChannelQuadratic, DitConvBlock, DitDetector
from alfoil_dnm.loss import detector_loss


class DitCnnTests(unittest.TestCase):
    def test_quadratic_matrix_starts_from_zero(self) -> None:
        layer = ChannelQuadratic(3, 2)
        inputs = torch.randn(2, 3, 4, 5)
        self.assertTrue(torch.equal(layer(inputs), torch.zeros(2, 2, 4, 5)))

    def test_quadratic_matches_manual_channel_interaction(self) -> None:
        layer = ChannelQuadratic(2, 1)
        with torch.no_grad():
            layer.matrix[0, 0, 1] = 2.0
            layer.matrix[0, 1, 1] = -0.5
        inputs = torch.tensor([[[[3.0]], [[4.0]]]])
        expected = 2.0 * 3.0 * 4.0 - 0.5 * 4.0 * 4.0
        actual = float(layer(inputs)[0, 0, 0, 0].detach())
        self.assertAlmostEqual(actual, expected)

    def test_full_channel_mode_removes_projections(self) -> None:
        block = DitConvBlock(channels=8, quadratic_channels=8)
        self.assertIsInstance(block.quadratic_reduce, torch.nn.Identity)
        self.assertIsInstance(block.quadratic_expand, torch.nn.Identity)

    def test_detector_output_and_backward(self) -> None:
        model = DitDetector(num_classes=3, width=8, quadratic_channels=4)
        prediction = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(prediction.shape, (2, 8, 8, 8))
        targets = [
            torch.tensor([[0.0, 0.5, 0.5, 0.2, 0.2]]),
            torch.tensor([[2.0, 0.25, 0.25, 0.1, 0.1]]),
        ]
        loss, _ = detector_loss(prediction, targets, num_classes=3)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.backbone.layers[-1].quadratic.matrix.grad)


if __name__ == "__main__":
    unittest.main()
