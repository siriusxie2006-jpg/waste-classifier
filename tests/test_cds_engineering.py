"""Regression checks for the Focal/CDS training and inference interface."""
import copy
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from integration.e0_branches import E0Branches
from integration.training_branches import CDSBranch
from train_e1 import FocalLoss, train_epoch
from train_e3 import forward_head, train_arm


class EngineeringTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True),
                                  nn.Dropout(.4), nn.Linear(256, 6))
        self.features = torch.randn(12, 512)
        self.labels = torch.arange(6).repeat_interleave(2)
        self.head.eval()
        with torch.no_grad():
            self.initial, _ = forward_head(self.head, self.features)

    def test_legacy_ce_and_focal_composition(self):
        embedding, logits = forward_head(self.head, self.features)
        for criterion in (nn.CrossEntropyLoss(), FocalLoss(1)):
            branch = CDSBranch(classification_loss=criterion)
            branch.loss.initialize_from_train_features(self.initial, self.labels)
            result = branch(embedding, logits, self.labels, .176)
            torch.testing.assert_close(result['classification'], criterion(logits, self.labels))
            torch.testing.assert_close(result['total'], criterion(logits, self.labels) + .176 * result['cds'])
        legacy = CDSBranch()
        legacy.loss.initialize_from_train_features(self.initial, self.labels)
        result = legacy(embedding, logits, self.labels, .176)
        torch.testing.assert_close(result['total'], result['ce'] + .176 * result['cds'])

    def test_cds_gradient_reaches_first_head_layer(self):
        embedding, logits = forward_head(self.head, self.features)
        branch = CDSBranch(theta=.01)
        branch.loss.initialize_from_train_features(self.initial, self.labels)
        value = branch.loss(embedding, self.labels, logits)
        gradients = torch.autograd.grad(value, (self.head[0].weight, self.head[3].weight), allow_unused=True)
        self.assertGreater(float(gradients[0].abs().sum()), 0)
        self.assertIsNone(gradients[1])
        self.assertFalse(branch.loss.centers.requires_grad)

    def test_zero_cds_replays_existing_training(self):
        batches = [(self.features, self.labels)]
        for criterion in (nn.CrossEntropyLoss(), FocalLoss(1)):
            legacy, combined = copy.deepcopy(self.head), copy.deepcopy(self.head)
            opt_old = torch.optim.Adam(legacy.parameters(), lr=.001)
            opt_new = torch.optim.Adam(combined.parameters(), lr=.001)
            branch = CDSBranch(classification_loss=criterion)
            branch.loss.initialize_from_train_features(self.initial, self.labels)
            train_epoch(legacy, batches, opt_old, criterion, 42, 'cpu')
            train_arm(combined, batches, opt_new, criterion, branch, 0., 42, 'cpu')
            for key, value in legacy.state_dict().items():
                torch.testing.assert_close(value, combined.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(int(branch.loss.total_updates), 1)

    def test_standard_train_mode_keeps_backbone_bn_frozen(self):
        model = E0Branches().train()
        self.assertTrue(model.model.fc.training)
        self.assertTrue(all(p.requires_grad == name.startswith('fc.')
                            for name, p in model.model.named_parameters()))
        for name, module in model.model.named_modules():
            if name and not name.startswith('fc'):
                self.assertFalse(module.training, name)

    def test_lambda_rejects_invalid_inputs(self):
        branch = CDSBranch()
        branch.loss.initialize_from_train_features(self.initial, self.labels)
        embedding, logits = forward_head(self.head, self.features)
        for coefficient in (None, -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                branch(embedding, logits, self.labels, coefficient)


if __name__ == '__main__':
    unittest.main(verbosity=2)
