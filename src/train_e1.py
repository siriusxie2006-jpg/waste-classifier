"""Compare class weights with a matched unweighted continuation on validation."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from train_transfer import WasteDataset, set_reproducibility, train_transform, val_test_transform


ROOT = Path(__file__).resolve().parent.parent
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
BETA = .995
BATCH_SIZE = 32
LEARNING_RATE = .001


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(seed):
    manifest = ROOT / 'data/manifests/split_manifest.csv'
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    loaders = {}
    for split, transform in [('train', train_transform), ('val', val_test_transform)]:
        dataset = WasteDataset(ROOT / 'data' / split, transform)
        expected = {(row['relative_path'], row['class']) for row in rows if row['split'] == split}
        found = {(Path(path).relative_to(dataset.root_dir).as_posix(), dataset.classes[label])
                 for path, label in dataset.samples}
        if found != expected or dataset.classes != CLASSES:
            raise ValueError(f'{split} images or labels differ from the fixed manifest')
        loaders[split] = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=split == 'train', num_workers=0,
            generator=torch.Generator().manual_seed(seed + (split == 'val')),
        )
    return loaders, file_hash(manifest)


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['classes'] != CLASSES or checkpoint['class_to_idx'] != dict(zip(CLASSES, range(6))):
        raise ValueError('Checkpoint class mapping differs from the experiment')
    model = resnet18(weights=None)
    model.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(.4), nn.Linear(256, 6))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    head = model.fc.to(device)
    model.fc = nn.Identity()
    model.eval().requires_grad_(False).to(device)
    return model, head


@torch.no_grad()
def frozen_batches(backbone, loader, device):
    return [(backbone(images.to(device)), labels.to(device)) for images, labels in loader]


def class_weights(dataset, device, trash_multipliers=()):
    counts = torch.bincount(torch.tensor([label for _, label in dataset.samples]), minlength=len(CLASSES))
    if (counts <= 0).any():
        raise ValueError('Each class needs at least one training example')
    weights = {'ce': torch.ones(len(CLASSES))}
    if trash_multipliers:
        for multiplier in trash_multipliers:
            vector = torch.ones_like(weights['ce'])
            vector[CLASSES.index('trash')] = multiplier
            weights[f'trash_{multiplier}'] = vector
    else:
        weights['inverse'] = counts.float().reciprocal()
        weights['effective'] = (1 - BETA) / (1 - BETA ** counts.float())
    return counts.tolist(), {name: (value / value.mean()).to(device) for name, value in weights.items()}


def train_epoch(head, batches, optimizer, criterion, seed, device):
    head.train()
    cuda_devices = [torch.cuda.current_device()] if device == 'cuda' else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        for features, target in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features), target)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate(head, batches):
    head.eval()
    logits = torch.cat([head(features) for features, _ in batches])
    labels = torch.cat([target for _, target in batches])
    loss = nn.functional.cross_entropy(logits, labels).item()
    truth, predictions = labels.tolist(), logits.argmax(dim=1).tolist()
    report = classification_report(truth, predictions, labels=range(6), target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(truth, predictions, labels=range(6))
    metrics = dict(accuracy=report['accuracy'], macro_f1=report['macro avg']['f1-score'], loss=loss,
                   trash_tp=int(cm[5, 5]), trash_fn=int(cm[5, :5].sum()), trash_fp=int(cm[:5, 5].sum()),
                   report=report, confusion_matrix=cm.tolist())
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--trash-multipliers', type=float, nargs='+', default=(),
                        help='Compare ordinary CE to these Trash weights with all other classes equal.')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    if any(not math.isfinite(value) or value <= 1 for value in args.trash_multipliers):
        parser.error('--trash-multipliers must be finite and greater than 1')

    set_reproducibility(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaders, manifest_hash = load_data(args.seed)
    backbone, initial_head = load_model(args.checkpoint, device)
    counts, weights = class_weights(loaders['train'].dataset, device, args.trash_multipliers)
    args.output.mkdir(parents=True, exist_ok=False)

    settings = dict(checkpoint_sha256=file_hash(args.checkpoint), manifest_sha256=manifest_hash,
                    seed=args.seed, epochs=args.epochs, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                    class_counts=dict(zip(CLASSES, counts)), beta=BETA,
                    weights={name: dict(zip(CLASSES, value.tolist())) for name, value in weights.items()},
                    checkpoint_selection='highest validation accuracy, earliest on tie',
                    device=device, torch_version=str(torch.__version__))
    (args.output / 'config.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')

    heads = {name: copy.deepcopy(initial_head) for name in weights}
    criteria = {name: nn.CrossEntropyLoss(weight=value) for name, value in weights.items()}
    optimizers = {name: torch.optim.Adam(head.parameters(), lr=LEARNING_RATE) for name, head in heads.items()}
    val = frozen_batches(backbone, loaders['val'], device)
    baseline, baseline_pred = evaluate(initial_head, val)
    history = []
    best = {}
    for epoch in range(1, args.epochs + 1):
        # Share augmented inputs. The frozen backbone is in eval mode for every scheme.
        train = frozen_batches(backbone, loaders['train'], device)
        for name, head in heads.items():
            train_epoch(head, train, optimizers[name], criteria[name], args.seed + epoch, device)
            metrics, predictions = evaluate(head, val)
            history.append(dict(epoch=epoch, scheme=name, **metrics))
            if name not in best or metrics['accuracy'] > best[name]['metrics']['accuracy']:
                best[name] = dict(epoch=epoch, metrics=metrics, predictions=predictions,
                                  head={key: value.detach().cpu().clone()
                                        for key, value in head.state_dict().items()})
            print(f"Epoch {epoch:2d}  {name:9s}  acc {metrics['accuracy']:.2%}  "
                  f"macro F1 {metrics['macro_f1']:.4f}  Trash TP/FN/FP "
                  f"{metrics['trash_tp']}/{metrics['trash_fn']}/{metrics['trash_fp']}", flush=True)

    state = {key: value.cpu() for key, value in backbone.state_dict().items()}
    for name, result in best.items():
        head_state = {f'fc.{key}': value for key, value in result.pop('head').items()}
        torch.save(dict(model_state_dict={**state, **head_state}, classes=CLASSES,
                        class_to_idx=dict(zip(CLASSES, range(6))), num_classes=6,
                        freeze_backbone=True, best_val_acc=result['metrics']['accuracy']),
                   args.output / f'{name}.pth')
    prediction_sets = {'local_start': baseline_pred, **{name: result.pop('predictions') for name, result in best.items()}}
    with (args.output / 'validation_predictions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image', 'truth', *prediction_sets])
        for index, (path, label) in enumerate(loaders['val'].dataset.samples):
            writer.writerow([Path(path).relative_to(ROOT).as_posix(), CLASSES[label],
                             *(CLASSES[predictions[index]] for predictions in prediction_sets.values())])
    result = dict(local_start=baseline, best=best, history=history)
    (args.output / 'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
