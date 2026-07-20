"""七类表面缺陷模型和统一指标的轻量回归测试。"""
from __future__ import annotations

import unittest

import torch

from classification.metrics import extract_logits, metrics_from_predictions
from classification.models import MODEL_VARIANTS, build_classifier


class ClassificationTests(unittest.TestCase):
    def test_all_variants_forward_and_backward(self) -> None:
        images = torch.randn(2, 3, 64, 64)
        labels = torch.tensor([0, 6])
        for variant in MODEL_VARIANTS:
            model = build_classifier(variant, num_classes=7)
            logits = model(images)
            self.assertEqual(tuple(logits.shape), (2, 7))
            self.assertTrue(torch.isfinite(logits).all())
            torch.nn.functional.cross_entropy(logits, labels).backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
            self.assertTrue(any(
                gradient is not None and torch.isfinite(gradient).all()
                for gradient in gradients
            ))

    def test_extract_logits_accepts_ultralytics_eval_tuple(self) -> None:
        probabilities = torch.softmax(torch.randn(2, 7), dim=1)
        logits = torch.randn(2, 7)
        self.assertIs(extract_logits((probabilities, logits)), logits)

    def test_macro_metrics_and_confusion_matrix(self) -> None:
        metrics = metrics_from_predictions(
            targets=[0, 0, 1, 1],
            predictions=[0, 1, 1, 1],
            class_names=["a", "b"],
        )
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["macro_recall"], 0.75)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])


if __name__ == "__main__":
    unittest.main()
