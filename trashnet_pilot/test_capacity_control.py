"""Capacity count, matched initial state, true updates, and strict reconstruction."""
import hashlib
import tempfile
import unittest
from pathlib import Path
import torch
from PIL import Image
from train_baseline import ROOT, CLASSES, set_seed
from model_utils import build_model, set_training_phase, build_optimizer, load_checkpoint
from inference import gradcam_overlay


class CapacityControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(42)

    def test_capacity_is_matched_and_common_initial_state_is_shared(self):
        torch.manual_seed(42)
        capacity = build_model(head='residual_capacity').eval()
        torch.manual_seed(42)
        fusion = build_model(architecture='multiscale_fixed').eval()
        capacity_count = sum(parameter.numel() for parameter in capacity.parameters())
        fusion_count = sum(parameter.numel() for parameter in fusion.parameters())
        self.assertEqual(capacity_count, 11311302)
        self.assertEqual(fusion_count, 11311174)
        self.assertEqual(capacity_count-fusion_count, 128)
        self.assertTrue(torch.equal(capacity.fc.classifier.weight, fusion.fc.classifier.weight))
        self.assertTrue(torch.equal(capacity.layer3[0].conv1.weight, fusion.layer3[0].conv1.weight))
        self.assertFalse(any(isinstance(module, torch.nn.Dropout) for module in capacity.fc.modules()))

    def test_real_head_update_and_freezing(self):
        model = build_model(head='residual_capacity')
        set_training_phase(model, 'head')
        optimizer = build_optimizer(model)
        before = model.fc.residual[0].weight.detach().clone()
        loss = torch.nn.functional.cross_entropy(model(torch.randn(4, 3, 64, 64)), torch.tensor([0, 1, 4, 5]))
        loss.backward()
        self.assertIsNotNone(model.fc.residual[2].weight.grad)
        self.assertIsNone(model.layer4[0].conv1.weight.grad)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.fc.residual[0].weight))
        set_training_phase(model, 'layer3_layer4')
        self.assertEqual(len(build_optimizer(model).param_groups), 3)
        self.assertFalse(model.layer1.training)

    def test_roundtrip_and_semantic_layer_gradcam(self):
        model = build_model(head='residual_capacity').eval()
        config = dict(head='residual_capacity', architecture='resnet18',
                      split_manifest_sha256=hashlib.sha256((ROOT/'data'/'split_manifest.csv').read_bytes()).hexdigest())
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as directory:
            path = Path(directory)/'model.pt'
            torch.save(dict(classes=CLASSES, config=config, model_state_dict=model.state_dict()), path)
            restored, _ = load_checkpoint(path, manifest_path=ROOT/'data'/'split_manifest.csv')
            inputs = torch.randn(1, 3, 64, 64)
            with torch.no_grad():
                self.assertTrue(torch.equal(model(inputs), restored(inputs)))
            for parameter in restored.parameters():
                parameter.requires_grad_(False)
            hooks_before = len(restored.layer4._forward_hooks)
            overlay = gradcam_overlay(restored, Image.new('RGB', (224, 224), (120, 110, 100)))
            self.assertEqual(overlay.size, (224, 224))
            self.assertEqual(len(restored.layer4._forward_hooks), hooks_before)
            self.assertTrue(all(parameter.grad is None for parameter in restored.parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
