"""Verify real optimizer updates, trainable-block coverage, and invalid settings."""
import copy
import unittest
import torch
from model_utils import build_model, set_training_phase, build_optimizer


class OptimizerSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        torch.set_num_threads(4)

    def test_default_steps_match_historical_adamw(self):
        for scope in ['head', 'layer4', 'layer3_layer4']:
            with self.subTest(scope=scope):
                model = build_model()
                set_training_phase(model, scope)
                reference = copy.deepcopy(model)
                current = build_optimizer(model)
                if scope == 'head':
                    historical = torch.optim.AdamW(reference.fc.parameters(), lr=.001, weight_decay=.0001)
                else:
                    groups = [{'params': reference.fc.parameters(), 'lr': .001}]
                    for block in ['layer3', 'layer4']:
                        parameters = [parameter for parameter in getattr(reference, block).parameters() if parameter.requires_grad]
                        if parameters:
                            groups.append({'params': parameters, 'lr': .0001})
                    historical = torch.optim.AdamW(groups, weight_decay=.0001)
                inputs = torch.randn(2, 3, 64, 64)
                for instance, optimizer in [(model, current), (reference, historical)]:
                    optimizer.zero_grad(set_to_none=True)
                    instance(inputs).square().mean().backward()
                    optimizer.step()
                self.assertTrue(all(torch.equal(value, reference.state_dict()[key]) for key, value in model.state_dict().items()))

    def test_custom_rates_and_decay_cover_only_active_parameters(self):
        model = build_model()
        for scope, count in [('head', 1), ('layer4', 2), ('layer3_layer4', 3)]:
            set_training_phase(model, scope)
            optimizer = build_optimizer(model, head_lr=.0003, backbone_lr=.00003, weight_decay=.001)
            self.assertEqual(len(optimizer.param_groups), count)
            self.assertEqual([group['lr'] for group in optimizer.param_groups], [.0003]+[.00003]*(count-1))
            self.assertTrue(all(group['weight_decay'] == .001 for group in optimizer.param_groups))
            selected = [parameter for group in optimizer.param_groups for parameter in group['params']]
            self.assertEqual({id(parameter) for parameter in selected}, {id(parameter) for parameter in model.parameters() if parameter.requires_grad})

    def test_invalid_settings_and_inconsistent_freezing_are_rejected(self):
        model = build_model()
        set_training_phase(model, 'head')
        for settings in [dict(head_lr=0), dict(backbone_lr=-1), dict(head_lr=float('nan')),
                         dict(weight_decay=-1), dict(weight_decay=float('inf'))]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                build_optimizer(model, **settings)
        model.layer1[0].conv1.weight.requires_grad_(True)
        with self.assertRaises(ValueError):
            build_optimizer(model)


if __name__ == '__main__':
    unittest.main(verbosity=2)
