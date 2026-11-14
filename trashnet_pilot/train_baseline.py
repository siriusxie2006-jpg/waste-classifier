"""A small reproducible ResNet18 transfer-learning pilot; test is held out."""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('TORCH_HOME', str(ROOT / 'cache' / 'torch'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache' / 'matplotlib'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from model_utils import build_model, load_checkpoint, set_training_phase, training_class_weights, build_optimizer
from image_framing import fit_image

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_transform(train=False, augmentation='baseline', framing='center'):
    """Baseline excludes brightness and blur; extension hooks are not run here."""
    if train:
        steps = ([T.Lambda(fit_image)] if framing == 'fit' else [T.RandomResizedCrop(224, scale=(.8, 1.0))]) + [T.RandomHorizontalFlip()]
        if augmentation in ['light', 'both']:
            steps += [T.RandomApply([T.ColorJitter(brightness=(.4, 1.3), contrast=(.8, 1.2))], p=.5)]
        if augmentation in ['blur', 'both']:
            steps += [T.RandomApply([T.GaussianBlur(5, sigma=(.1, 2.0))], p=.5)]
    else:
        steps = [T.Lambda(fit_image)] if framing == 'fit' else [T.Resize(256), T.CenterCrop(224)]
    return T.Compose(steps + [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


class TrashDataset(Dataset):
    def __init__(self, split, transform):
        with (ROOT / 'data' / 'split_manifest.csv').open(encoding='utf-8', newline='') as f:
            self.records = [r for r in csv.DictReader(f) if r['split'] == split]
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        with Image.open(ROOT / r['path']) as im:
            x = self.transform(im.convert('RGB'))
        return x, int(r['label']), r['path']


def loader(split, batch_size, workers, seed, augmentation='baseline', framing='center'):
    dataset = TrashDataset(split, make_transform(split == 'train', augmentation, framing))
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == 'train',
                      num_workers=workers, persistent_workers=workers > 0,
                      pin_memory=torch.cuda.is_available(), generator=torch.Generator().manual_seed(seed))


@torch.inference_mode()
def evaluate(model, data, device):
    model.eval()
    truth, pred, paths, probs = [], [], [], []
    loss_sum = 0.0
    for x, y, p in data:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction='sum').item()
        prob = logits.softmax(1)
        truth.extend(y.cpu().tolist())
        pred.extend(logits.argmax(1).cpu().tolist())
        probs.extend(prob.max(1).values.cpu().tolist())
        paths.extend(p)
    result = dict(loss=loss_sum / len(truth), accuracy=accuracy_score(truth, pred),
                  macro_f1=f1_score(truth, pred, labels=list(range(6)), average='macro', zero_division=0),
                  count=len(truth), classification_report=classification_report(truth, pred, labels=list(range(6)), target_names=CLASSES, output_dict=True, zero_division=0),
                  confusion_matrix=confusion_matrix(truth, pred, labels=list(range(6))).tolist())
    rows = [dict(path=p, true_class=CLASSES[y], predicted_class=CLASSES[z], confidence=float(q)) for p,y,z,q in zip(paths, truth, pred, probs)]
    return result, rows


def save_evaluation(model, data, device, out, split):
    result, rows = evaluate(model, data, device)
    (out / f'{split}_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (out / f'{split}_predictions.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    cm = np.array(result['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.imshow(cm, cmap='Blues')
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
    ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predicted class', ylabel='True class', title=f'ResNet18 - {split} (model selection data)' if split == 'val' else f'ResNet18 - {split}')
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    fig.tight_layout(); fig.savefig(out / f'{split}_confusion_matrix.png', dpi=160); plt.close(fig)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train', 'evaluate'], default='train')
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--finetune-epochs', type=int, default=7)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--augmentation', choices=['baseline', 'light', 'blur', 'both'], default='baseline')
    p.add_argument('--run-name', default='baseline_resnet18_seed42')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--head', choices=['linear', 'mlp', 'residual_capacity'], default='linear')
    p.add_argument('--architecture', choices=['resnet18', 'multiscale_fixed', 'multiscale_gated'], default='resnet18')
    p.add_argument('--finetune-scope', choices=['head', 'layer4', 'layer3_layer4'], default='layer4')
    p.add_argument('--class-weighting', choices=['none', 'inverse'], default='none')
    p.add_argument('--framing', choices=['center', 'fit'], default='center')
    p.add_argument('--head-lr', type=float, default=.001)
    p.add_argument('--backbone-lr', type=float, default=.0001)
    p.add_argument('--weight-decay', type=float, default=.0001)
    args = p.parse_args()
    if args.warmup_epochs < 1 or args.finetune_epochs < 0:
        p.error('Need at least one warmup epoch and nonnegative fine-tuning epochs')
    if args.architecture != 'resnet18' and args.head != 'linear':
        p.error('Multi-scale variants require --head linear')
    if any(not math.isfinite(value) or value <= 0 for value in (args.head_lr, args.backbone_lr)):
        p.error('Learning rates must be finite and positive')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error('Weight decay must be finite and nonnegative')
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        p.error('--run-name must be a single directory name')
    set_seed(args.seed)
    out = ROOT / 'outputs' / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'best_model.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    environment = dict(python=__import__('sys').version, torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                       cuda_runtime=torch.version.cuda, device=str(device),
                       gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                       split_manifest_sha256=hashlib.sha256((ROOT/'data'/'split_manifest.csv').read_bytes()).hexdigest())
    print(json.dumps(environment, indent=2), flush=True)
    if args.mode == 'evaluate':
        model, state = load_checkpoint(checkpoint, device, ROOT/'data'/'split_manifest.csv')
        result = save_evaluation(model.to(device), loader(args.split, args.batch_size, args.workers, args.seed, framing=state['config'].get('framing', 'center')), device, out, args.split)
        print(json.dumps(result, indent=2), flush=True)
        return
    if checkpoint.exists():
        raise FileExistsError(f'{checkpoint} exists: use a new --run-name, do not silently overwrite')
    config = dict(vars(args), **environment, pretrained_weights='ResNet18_Weights.IMAGENET1K_V1',
                  selection_metric='validation macro-F1', test_evaluated=False,
                  learning_rates={'head': args.head_lr, 'trainable_backbone': args.backbone_lr}, weight_decay=args.weight_decay,
                  frozen_batchnorm='Only selected trainable blocks and the head enter train mode',
                  evaluation_loss='Unweighted cross-entropy for comparability',
                  implementation_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                         for name in ['train_baseline.py', 'model_utils.py', 'image_framing.py', 'multiscale_model.py', 'capacity_control.py']})
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    model = build_model(args.head, pretrained=True, architecture=args.architecture)
    set_training_phase(model, 'head')
    model.to(device)
    train = loader('train', args.batch_size, args.workers, args.seed, args.augmentation, args.framing)
    val = loader('val', args.batch_size, args.workers, args.seed, framing=args.framing)
    class_weights = training_class_weights(train.dataset.records, args.class_weighting)
    config['class_weights'] = None if class_weights is None else class_weights.tolist()
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    if class_weights is not None:
        class_weights = class_weights.to(device)
    started = time.perf_counter()
    history, best_f1, best_epoch = [], -1.0, 0
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    phases = [('head', args.warmup_epochs), ('layer4', args.finetune_epochs)]
    epoch = 0
    for phase, epochs in phases:
        if phase == 'layer4':
            # Start fine-tuning from the best head checkpoint, not an arbitrary last epoch.
            if checkpoint.exists():
                model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['model_state_dict'])
            set_training_phase(model, args.finetune_scope)
        optimizer = build_optimizer(model, args.head_lr, args.backbone_lr, args.weight_decay)
        for _ in range(epochs):
            epoch += 1
            t0 = time.perf_counter()
            set_training_phase(model, args.finetune_scope if phase == 'layer4' else 'head')
            total, correct, loss_sum, loss_denominator = 0, 0, 0.0, 0.0
            for x, y, _paths in train:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, y, weight=class_weights)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                total += len(y); correct += (logits.argmax(1) == y).sum().item()
                denominator = len(y) if class_weights is None else class_weights[y].sum().item()
                loss_sum += loss.item() * denominator
                loss_denominator += denominator
            result, _ = evaluate(model, val, device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            row = dict(epoch=epoch, phase=phase if phase == 'head' else f'finetune_{args.finetune_scope}', train_loss=loss_sum/loss_denominator, train_accuracy=correct/total,
                       val_loss=result['loss'], val_accuracy=result['accuracy'], val_macro_f1=result['macro_f1'], seconds=time.perf_counter()-t0)
            history.append(row)
            print(json.dumps(row), flush=True)
            with (out / 'history.csv').open('w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(history)
            if result['macro_f1'] > best_f1:
                best_f1, best_epoch = result['macro_f1'], epoch
                torch.save(dict(model_state_dict=model.state_dict(), classes=CLASSES, epoch=epoch,
                                validation_macro_f1=best_f1, config=config), checkpoint)
    elapsed = time.perf_counter()-started
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['model_state_dict'])
    metrics = save_evaluation(model, val, device, out, 'val')
    summary = dict(best_epoch=best_epoch, validation_accuracy=metrics['accuracy'], validation_macro_f1=metrics['macro_f1'],
                   total_training_validation_seconds=elapsed, epochs_completed=len(history),
                   mean_epoch_seconds=float(np.mean([h['seconds'] for h in history])),
                   peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2 if device.type == 'cuda' else None,
                   test_evaluated=False, note='Pilot validation results used for model selection, not an unbiased final test estimate.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    axes[0].plot([r['epoch'] for r in history], [r['train_loss'] for r in history], label='train')
    axes[0].plot([r['epoch'] for r in history], [r['val_loss'] for r in history], label='validation')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss'); axes[0].legend()
    axes[1].plot([r['epoch'] for r in history], [r['val_accuracy'] for r in history], label='val accuracy')
    axes[1].plot([r['epoch'] for r in history], [r['val_macro_f1'] for r in history], label='val macro-F1')
    axes[1].set(xlabel='Epoch', ylabel='Score', ylim=(0,1)); axes[1].legend()
    fig.tight_layout(); fig.savefig(out/'learning_curves.png', dpi=160); plt.close(fig)
    print('FINAL ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
