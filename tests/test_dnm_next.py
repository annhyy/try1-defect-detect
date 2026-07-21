"""Regression tests for the isolated next-stage DNM classification heads."""
from __future__ import annotations

import unittest

import torch

from classification.models import PaperDNMHead
from alfoil_dnm_next.models import BRANCH_FEATURES, TunedDNMHead, build_classifier
from alfoil_dnm_next.train import build_optimizer_and_scheduler


class DNMNextTests(unittest.TestCase):
    def test_f4_heads_match_existing_v2_formula_after_state_copy(self) -> None:
        from alfoil_dnm_next.models import PaperF4DNMHead

        for aggregation in ("product", "geometric_mean"):
            torch.manual_seed(123)
            old = PaperDNMHead(128, 7, 4, 4, aggregation).eval()
            torch.manual_seed(456)
            new = PaperF4DNMHead(128, 7, 4, aggregation).eval()
            new.load_state_dict(old.state_dict())
            features = torch.randn(3, 128)
            with torch.no_grad():
                self.assertTrue(torch.allclose(old(features), new(features)))

    def test_tuned_projection_is_branch_specific_and_has_signed_output(self) -> None:
        head = TunedDNMHead(128, 7, 4)
        self.assertEqual(head.projection.out_features, 4 * BRANCH_FEATURES)
        self.assertEqual(tuple(head.branch_strength.shape), (7, 4))
        with torch.no_grad():
            head.weight.zero_()
            head.threshold.zero_()
            head.branch_strength.fill_(1.0)
            head.branch_strength[0, 0] = -1.0
            head.class_bias[0] = 0.25
        actual = head(torch.randn(2, 128))
        expected_class_zero = -0.0625 * 1 + 0.0625 * 3 + 0.25
        self.assertTrue(torch.allclose(actual[:, 0], torch.full((2,), expected_class_zero)))

    def test_all_new_models_have_finite_classification_gradients(self) -> None:
        images = torch.randn(2, 3, 64, 64)
        labels = torch.tensor([0, 6])
        for variant in ("v2a_f4", "v2b_f4", "v1_tuned"):
            model = build_classifier(variant, 7)
            logits = model(images)
            self.assertEqual(tuple(logits.shape), (2, 7))
            torch.nn.functional.cross_entropy(logits, labels).backward()
            self.assertTrue(
                all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                )
            )

    def test_tuned_scheduler_has_separate_learning_rates_and_one_percent_floor(self) -> None:
        model = build_classifier("v1_tuned", 7)
        optimizer, scheduler = build_optimizer_and_scheduler(
            model, 150, 1e-3, 3e-3, 0.01, 1e-4
        )
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        self.assertEqual(optimizer.param_groups[1]["lr"], 3e-3)
        self.assertAlmostEqual(scheduler.lr_lambdas[0](150), 0.01)


if __name__ == "__main__":
    unittest.main()
