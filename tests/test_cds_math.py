"""Standalone unit tests for the final CDS integration; run with unittest."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))
from cds_loss import ConfusionDirectedSqueezeLoss


class CDSTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.loss = ConfusionDirectedSqueezeLoss()
        init = torch.randn(12, 256)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        self.loss.initialize_from_train_features(init, labels)

    def test_forward_finite_and_embedding_gradient(self):
        x = torch.randn(12, 256, requires_grad=True)
        y = torch.arange(6).repeat_interleave(2)
        logits = torch.randn(12, 6, requires_grad=True)
        value = self.loss(x, y, logits)
        gx, gl = torch.autograd.grad(value, (x, logits), allow_unused=True)
        self.assertTrue(torch.isfinite(value))
        self.assertIsNotNone(gx)
        self.assertGreater(float(gx.norm()), 0.0)
        self.assertIsNone(gl)  # confidence weights are stop-gradient

    def test_count_adjusted_ema_and_missing_class(self):
        before = self.loss.centers.clone()
        batch = torch.randn(3, 256)
        labels = torch.tensor([2, 2, 2])
        zmean = torch.nn.functional.normalize(batch, dim=1).mean(0)
        rho = 1.0 - (1.0 - self.loss.rho0) ** 3
        expected = before[2] * (1.0-rho) + zmean * rho
        self.loss.update_centers(batch, labels)
        self.assertTrue(torch.allclose(self.loss.centers[2], expected, atol=1e-6))
        self.assertTrue(torch.equal(self.loss.centers[0], before[0]))
        self.assertEqual(int(self.loss.class_update_counts[2]), 1)
        self.assertTrue(torch.isfinite(self.loss.last_update_norm))

    def test_degenerate_prototype_pair_is_finite(self):
        self.loss.centers[0].zero_(); self.loss.centers[0, 0] = 1
        self.loss.centers[1].copy_(self.loss.centers[0]); self.loss.seen.fill_(True)
        x = torch.randn(2, 256, requires_grad=True)
        y = torch.tensor([0, 1])
        logits = torch.randn(2, 6)
        value = self.loss(x, y, logits)
        grad = torch.autograd.grad(value, x)[0]
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(grad).all())

    def test_invalid_formal_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            ConfusionDirectedSqueezeLoss(feature_dim=128)


if __name__ == "__main__":
    unittest.main(verbosity=2)
