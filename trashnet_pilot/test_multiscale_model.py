"""Check fusion, real gradients, checkpoint identity, and complete attribution paths."""
import hashlib
import tempfile
import unittest
from pathlib import Path
import torch
from PIL import Image
from train_baseline import ROOT, CLASSES, set_seed
from model_utils import build_model, set_training_phase, build_optimizer, load_checkpoint
from inference import gradcam_overlay


class MultiScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(42)

    def test_shared_initialization_fixed_mixture_and_initial_adaptive_equivalence(self):
        torch.manual_seed(42)
        original = build_model().eval()
        torch.manual_seed(42)
        fixed = build_model(architecture='multiscale_fixed').eval()
        torch.manual_seed(42)
        adaptive = build_model(architecture='multiscale_gated').eval()
        self.assertTrue(torch.equal(original.fc.weight, adaptive.fc.classifier.weight))
        self.assertTrue(torch.equal(original.layer4[0].conv1.weight, adaptive.layer4[0].conv1.weight))
        inputs = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            fixed_output, fixed_gate = fixed.forward_with_gate(inputs)
            output, gate = adaptive.forward_with_gate(inputs)
        self.assertEqual(output.shape, (2, 6))
        self.assertTrue(torch.equal(fixed_gate, torch.full((2, 1), .5)))
        self.assertTrue(torch.equal(gate, fixed_gate))
        self.assertTrue(torch.equal(output, fixed_output))

    def test_new_head_receives_gradients_and_frozen_backbone_stays_frozen(self):
        model = build_model(architecture='multiscale_gated')
        set_training_phase(model, 'head')
        optimizer = build_optimizer(model)
        before = model.fc.project_detail.weight.detach().clone()
        inputs = torch.randn(4, 3, 64, 64)
        loss = torch.nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1, 4, 5]))
        loss.backward()
        self.assertIsNotNone(model.fc.gate[-1].weight.grad)
        self.assertGreater(model.fc.gate[-1].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.layer3[0].conv1.weight.grad)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.fc.project_detail.weight))
        with torch.no_grad():
            _, gate = model.forward_with_gate(inputs)
        self.assertTrue(torch.isfinite(gate).all())
        self.assertTrue(((gate > 0) & (gate < 1)).all())
        set_training_phase(model, 'layer3_layer4')
        optimizer = build_optimizer(model)
        self.assertEqual(len(optimizer.param_groups), 3)
        self.assertFalse(model.layer1.training)
        self.assertTrue(model.layer3.training)

    def test_roundtrip_and_gradcam_cover_both_fusion_paths(self):
        for architecture in ['multiscale_fixed', 'multiscale_gated']:
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory(dir=ROOT/'cache') as directory:
                model = build_model(architecture=architecture).eval()
                config = dict(head='linear', architecture=architecture,
                              split_manifest_sha256=hashlib.sha256((ROOT/'data'/'split_manifest.csv').read_bytes()).hexdigest())
                path = Path(directory)/'model.pt'
                torch.save(dict(classes=CLASSES, config=config, model_state_dict=model.state_dict()), path)
                restored, _ = load_checkpoint(path, manifest_path=ROOT/'data'/'split_manifest.csv')
                inputs = torch.randn(1, 3, 64, 64)
                with torch.no_grad():
                    self.assertTrue(torch.equal(model(inputs), restored(inputs)))
                for parameter in restored.parameters():
                    parameter.requires_grad_(False)
                hooks_before = len(restored.layer3._forward_hooks)
                overlay = gradcam_overlay(restored, Image.new('RGB', (224, 224), (120, 110, 100)))
                self.assertEqual(overlay.size, (224, 224))
                self.assertEqual(len(restored.layer3._forward_hooks), hooks_before)
                self.assertTrue(all(parameter.grad is None for parameter in restored.parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
