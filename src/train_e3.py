"""Run matched CE, Focal (E1), CDS (E2), and Focal+CDS (E3) continuations."""
import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integration.training_branches import CDSBranch
from train_e1 import (BATCH_SIZE, CLASSES, LEARNING_RATE, FocalLoss, evaluate,
                      file_hash, frozen_batches, load_data, load_model)
from train_transfer import WasteDataset, set_reproducibility, val_test_transform

SCHEMES = ('ce', 'e1_focal', 'e2_cds', 'e3_focal_cds')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def forward_head(head, features):
    embedding = head[1](head[0](features))
    logits = head[3](head[2](embedding))
    return embedding, logits


def snapshot(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def assert_frozen(backbone, before):
    if any(module.training for module in backbone.modules()):
        raise RuntimeError('Frozen backbone or BatchNorm entered training mode')
    if any(p.requires_grad or p.grad is not None for p in backbone.parameters()):
        raise RuntimeError('A backbone parameter received gradients')
    for key, value in backbone.state_dict().items():
        if not torch.equal(value.detach().cpu(), before[key]):
            raise RuntimeError(f'Frozen backbone parameter or buffer changed: {key}')


def train_arm(head, batches, optimizer, criterion, branch, coefficient, seed, device):
    head.train()
    components = dict(classification=0., cds=0., total=0.)
    images = 0
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == 'cuda' else []):
        torch.manual_seed(seed)
        for features, labels in batches:
            optimizer.zero_grad(set_to_none=True)
            embedding, logits = forward_head(head, features)
            if branch is None:
                classification = criterion(logits, labels)
                cds = classification.new_zeros(())
                loss = classification
            else:
                if not embedding.requires_grad:
                    raise RuntimeError('CDS requires a live classification-head embedding')
                parts = branch(embedding, logits, labels, coefficient)
                classification, cds, loss = parts['classification'], parts['cds'], parts['total']
            if not torch.isfinite(loss):
                raise FloatingPointError('Training loss is not finite')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise FloatingPointError('Head gradients are missing or non-finite')
            optimizer.step()
            if branch is not None:
                # EMA changes targets only after backward has consumed this batch's graph.
                branch.update_centers_after_batch(embedding, labels)
            for key, value in [('classification', classification), ('cds', cds), ('total', loss)]:
                components[key] += float(value.detach()) * len(labels)
            images += len(labels)
    return {key: value / images for key, value in components.items()}


def make_checkpoint(backbone, head_state, metrics):
    state = snapshot(backbone)
    state.update({f'fc.{key}': value for key, value in head_state.items()})
    return dict(model_state_dict=state, classes=CLASSES,
                class_to_idx=dict(zip(CLASSES, range(6))), num_classes=6,
                freeze_backbone=True, best_val_acc=metrics['accuracy'])


def run_seed(args, seed, config, device, output):
    set_reproducibility(seed)
    loaders, manifest_hash = load_data(seed)
    backbone, initial = load_model(args.checkpoint, device)
    frozen = snapshot(backbone)
    # Center initialization sees only deterministic Train views from the shared E0.
    center_dataset = WasteDataset(ROOT / 'data/train', val_test_transform)
    if center_dataset.samples != loaders['train'].dataset.samples:
        raise ValueError('Center-initialization samples differ from the audited Train roster')
    center_features = torch.cat([x for x, _ in frozen_batches(
        backbone, DataLoader(center_dataset, batch_size=BATCH_SIZE), device)])
    center_labels = torch.tensor([label for _, label in center_dataset.samples], device=device)
    initial.eval()
    with torch.no_grad():
        initial_embeddings, _ = forward_head(initial, center_features)

    heads = {name: copy.deepcopy(initial) for name in args.schemes}
    criteria = {name: (FocalLoss(args.gamma) if 'focal' in name else nn.CrossEntropyLoss())
                for name in args.schemes}
    branches = {}
    for name in args.schemes:
        branch = None
        if 'cds' in name:
            branch = CDSBranch(theta=config['theta'], rho0=config['rho0'],
                               classification_loss=criteria[name]).to(device)
            branch.loss.initialize_from_train_features(initial_embeddings, center_labels)
        branches[name] = branch
    optimizers = {name: torch.optim.Adam(head.parameters(), lr=LEARNING_RATE) for name, head in heads.items()}
    for name, optimizer in optimizers.items():
        actual = {id(p) for group in optimizer.param_groups for p in group['params']}
        if actual != {id(p) for p in heads[name].parameters()}:
            raise RuntimeError('Optimizer contains parameters outside the active classification head')
    val = frozen_batches(backbone, loaders['val'], device)
    e0_metrics, e0_predictions = evaluate(initial, val)
    write_json(output / 'config.json', dict(seed=seed, epochs=args.epochs, focal_gamma=args.gamma,
                                            lambda_cds=config['lambda_cds'], theta=config['theta'], rho0=config['rho0'],
                                            checkpoint_sha256=file_hash(args.checkpoint), manifest_sha256=manifest_hash,
                                            source_cds_sha256=file_hash(ROOT / 'integration/cds_loss.py'),
                                            schemes=args.schemes, device=device, optimizer='Adam', learning_rate=LEARNING_RATE,
                                            batch_size=BATCH_SIZE, center_initialization_images=len(center_dataset),
                                            checkpoint_selection='highest validation accuracy, earliest on tie'))
    history, best, last = [], {}, {}
    for epoch in range(1, args.epochs + 1):
        # Only the frozen 512-D encoder output is cached. The 256-D head remains live.
        batches = frozen_batches(backbone, loaders['train'], device)
        for name, head in heads.items():
            components = train_arm(head, batches, optimizers[name], criteria[name], branches[name],
                                   config['lambda_cds'], seed + epoch, device)
            metrics, predictions = evaluate(head, val)
            history.append(dict(epoch=epoch, scheme=name, train=components, validation=metrics))
            record = dict(epoch=epoch, metrics=metrics, predictions=predictions)
            last[name] = record
            if name not in best or metrics['accuracy'] > best[name]['metrics']['accuracy']:
                best[name] = record
                checkpoint = make_checkpoint(backbone, snapshot(head), metrics)
                if branches[name] is not None:
                    checkpoint['cds_state_dict'] = snapshot(branches[name])
                checkpoint['experiment'] = dict(scheme=name, seed=seed, epoch=epoch, gamma=args.gamma,
                                                lambda_cds=config['lambda_cds'])
                torch.save(checkpoint, output / f'{name}.pth')
        assert_frozen(backbone, frozen)
        print(f'seed {seed} epoch {epoch}/{args.epochs} finished', flush=True)

    replay = {}
    for name in args.schemes:
        loaded_backbone, saved_head = load_model(output / f'{name}.pth', device)
        assert_frozen(loaded_backbone, frozen)
        replay_metrics, prediction = evaluate(saved_head, val)
        if prediction != best[name]['predictions'] or replay_metrics != best[name]['metrics']:
            raise RuntimeError(f'Saved {name} checkpoint did not replay its validation result')
        replay[name] = len(prediction)
    torch.save({name: snapshot(head) for name, head in heads.items()}, output / 'epoch_last_heads.pth')
    columns = {'e0': e0_predictions}
    columns.update({name: best[name]['predictions'] for name in args.schemes})
    columns.update({f'{name}_last': last[name]['predictions'] for name in args.schemes})
    with (output / 'validation_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image', 'truth', *columns])
        for index, (path, target) in enumerate(loaders['val'].dataset.samples):
            writer.writerow([Path(path).relative_to(ROOT).as_posix(), CLASSES[target],
                             *(CLASSES[pred[index]] for pred in columns.values())])
    result = dict(seed=seed, e0=e0_metrics, best={name: {key: value for key, value in best[name].items()
                                                     if key != 'predictions'} for name in args.schemes},
                  fixed_last={name: last[name]['metrics'] for name in args.schemes}, history=history,
                  backbone_and_bn_unchanged=True, checkpoint_replay=replay,
                  center_updates={name: int(branch.loss.total_updates) for name, branch in branches.items()
                                  if branch is not None})
    write_json(output / 'metrics.json', result)
    return result


def mean_metrics(metrics):
    return dict(accuracy=sum(m['accuracy'] for m in metrics) / len(metrics),
                macro_f1=sum(m['macro_f1'] for m in metrics) / len(metrics),
                classes={label: {key: sum(m['report'][label][key] for m in metrics) / len(metrics)
                                 for key in ('precision', 'recall', 'f1-score', 'support')} for label in CLASSES})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--cds-config', type=Path, default=ROOT / 'integration/cds_config.json')
    parser.add_argument('--lambda-cds', type=float, default=None)
    parser.add_argument('--schemes', choices=SCHEMES, nargs='+', default=list(SCHEMES))
    args = parser.parse_args()
    if args.epochs < 1 or not math.isfinite(args.gamma) or args.gamma < 0:
        parser.error('epochs must be positive and gamma finite and nonnegative')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.schemes)) != len(args.schemes):
        parser.error('seeds and schemes must not contain duplicates')
    config = json.loads(args.cds_config.read_text(encoding='utf-8-sig'))
    if args.lambda_cds is not None:
        config['lambda_cds'] = args.lambda_cds
    if not math.isfinite(config['lambda_cds']) or config['lambda_cds'] < 0:
        parser.error('lambda_cds must be finite and nonnegative')
    CDSBranch(theta=config['theta'], rho0=config['rho0'])
    args.output.mkdir(parents=True, exist_ok=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []
    for seed in args.seeds:
        folder = args.output / f'seed{seed}'
        folder.mkdir()
        results.append(run_seed(args, seed, config, device, folder))
    summary = dict(e0=mean_metrics([result['e0'] for result in results]),
                   best={name: mean_metrics([r['best'][name]['metrics'] for r in results]) for name in args.schemes},
                   fixed_last={name: mean_metrics([r['fixed_last'][name] for r in results]) for name in args.schemes},
                   seeds=args.seeds, epochs=args.epochs, focal_gamma=args.gamma, cds=config,
                   backbone_and_bn_unchanged=True, test_used=False)
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary['best'], indent=2), flush=True)


if __name__ == '__main__':
    main()
