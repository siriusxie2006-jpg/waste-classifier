"""Regression checks for checkpoint compatibility and experiment invariants."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import torch
from train_baseline import ROOT, CLASSES, loader, evaluate, set_seed
from model_utils import build_model, set_training_phase, training_class_weights, load_checkpoint
from inference import prepare_image, gradcam_overlay
from PIL import Image


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(42)

    def test_class_weights_ignore_validation_and_test(self):
        records = [dict(split='train', label=str(label)) for label in range(6)]
        records += [dict(split=split, label='0') for split in ['val', 'test'] for _ in range(20)]
        self.assertTrue(torch.equal(training_class_weights(records, 'inverse'), torch.ones(6)))

    def test_frozen_batchnorm_in_deeper_finetuning(self):
        model = build_model('mlp')
        set_training_phase(model, 'layer3_layer4')
        frozen = model.layer1[0].bn1.running_mean.clone()
        trained = model.layer3[0].bn1.running_mean.clone()
        output = model(torch.randn(2, 3, 64, 64))
        output.sum().backward()
        self.assertEqual(output.shape, (2, 6))
        self.assertTrue(torch.equal(frozen, model.layer1[0].bn1.running_mean))
        self.assertFalse(torch.equal(trained, model.layer3[0].bn1.running_mean))
        self.assertIsNone(model.layer1[0].conv1.weight.grad)
        self.assertIsNotNone(model.layer3[0].conv1.weight.grad)

    def test_mlp_checkpoint_and_class_mapping_validation(self):
        model = build_model('mlp').eval()
        config = dict(head='mlp', split_manifest_sha256=hashlib.sha256((ROOT/'data'/'split_manifest.csv').read_bytes()).hexdigest())
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as directory:
            path = Path(directory)/'model.pt'
            checkpoint = dict(classes=CLASSES, config=config, model_state_dict=model.state_dict())
            torch.save(checkpoint, path)
            restored, _ = load_checkpoint(path, manifest_path=ROOT/'data'/'split_manifest.csv')
            image = torch.randn(1, 3, 64, 64)
            with torch.no_grad():
                self.assertTrue(torch.equal(model(image), restored(image)))
            checkpoint['classes'] = list(reversed(CLASSES))
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_legacy_predictions_and_gradcam_hook_cleanup(self):
        out = ROOT/'outputs'/'baseline_resnet18_seed42'
        model, _ = load_checkpoint(out/'best_model.pt', manifest_path=ROOT/'data'/'split_manifest.csv')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        metrics, _ = evaluate(model, loader('val', 32, 0, 42), device)
        original = json.loads((out/'val_metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(metrics['confusion_matrix'], original['confusion_matrix'])
        record = loader('val', 32, 0, 42).dataset.records[0]
        with Image.open(ROOT/record['path']) as image:
            prepared = prepare_image(image)
        for param in model.parameters():
            param.requires_grad_(False)
        hooks_before = len(model.layer4._forward_hooks)
        for _ in range(2):
            overlay = gradcam_overlay(model, prepared, device)
            self.assertEqual(overlay.size, (224, 224))
            self.assertEqual(len(model.layer4._forward_hooks), hooks_before)
        self.assertTrue(all(param.grad is None for param in model.parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
