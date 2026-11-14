"""Resume the pilot after relocation; run only train/validation experiments."""
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    from train_baseline import (
        CLASSES, ROOT, evaluate, loader, nn, resnet18, set_seed, torch,
    )

    manifest = ROOT / 'data' / 'split_manifest.csv'
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    baseline = ROOT / 'outputs' / 'baseline_resnet18_seed42'
    config = json.loads((baseline / 'config.json').read_text(encoding='utf-8'))
    if manifest_hash != config['split_manifest_sha256']:
        raise RuntimeError('Fixed split manifest changed; refusing to continue.')
    # Check paths only for train/validation. Held-out image contents are not opened.
    with manifest.open(encoding='utf-8', newline='') as stream:
        records = [r for r in csv.DictReader(stream) if r['split'] in ('train', 'val')]
    missing = [r['path'] for r in records if not (ROOT / r['path']).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing train/validation files: {missing[:5]}')
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(baseline / 'best_model.pt', map_location='cpu', weights_only=True)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(state['model_state_dict'])
    metrics, _ = evaluate(model.to(device), loader('val', 32, 0, 42), device)
    original = json.loads((baseline / 'val_metrics.json').read_text(encoding='utf-8'))
    if metrics['confusion_matrix'] != original['confusion_matrix']:
        raise RuntimeError('Relocated baseline validation predictions changed.')
    check = dict(
        checked_at=datetime.now().astimezone().isoformat(),
        executable=sys.executable, project_root=str(ROOT),
        torch=str(torch.__version__), device=str(device),
        gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        split_manifest_sha256=manifest_hash,
        train_validation_paths_checked=len(records),
        baseline_validation_accuracy=metrics['accuracy'],
        baseline_validation_macro_f1=metrics['macro_f1'],
        baseline_confusion_matrix_matches=True, test_evaluated=False,
    )
    (ROOT / 'outputs' / 'relocation_check.json').write_text(
        json.dumps(check, indent=2), encoding='utf-8')
    print('RELOCATION CHECK ' + json.dumps(check), flush=True)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    plan_path = ROOT / 'outputs' / 'validation_experiment_plan.json'
    plan = dict(
        seed=42, augmentations=['baseline', 'light', 'blur', 'both'],
        warmup_epochs=3, finetune_epochs=7, batch_size=32, workers=2,
        split_manifest_sha256=manifest_hash,
        checkpoint_selection='clean validation macro-F1',
        conditions=['clean', 'brightness_0.7', 'brightness_0.4',
                    'brightness_0.2', 'blur_1', 'blur_2', 'blur_4'],
        purpose='Single-seed exploratory comparison; no final model selection claim.',
        test_evaluated=False,
    )
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise RuntimeError('Existing experiment plan differs.')
    else:
        plan_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    runs = [('baseline', 'baseline_resnet18_seed42')] + [
        (aug, f'resnet18_{aug}_seed42') for aug in ['light', 'blur', 'both']]
    rows = []
    for augmentation, name in runs:
        out = ROOT / 'outputs' / name
        if augmentation != 'baseline' and not (out / 'summary.json').exists():
            subprocess.run([sys.executable, str(ROOT / 'train_baseline.py'),
                            '--augmentation', augmentation, '--run-name', name], check=True)
        run_config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
        for key in ('seed', 'warmup_epochs', 'finetune_epochs', 'batch_size', 'workers',
                    'split_manifest_sha256'):
            if run_config[key] != plan[key]:
                raise RuntimeError(f'Inconsistent comparison setting: {name} {key}')
        if run_config['augmentation'] != augmentation:
            raise RuntimeError(f'Incorrect augmentation for {name}')
        preview = out / 'validation_robustness_preview.csv'
        if not preview.exists():
            subprocess.run([sys.executable, str(ROOT / 'preview_robustness.py'),
                            '--run-name', name], check=True)
        with preview.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                rows.append(dict(augmentation=augmentation, run_name=name, **row))
    with (ROOT / 'outputs' / 'validation_comparison.csv').open(
            'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = ['# Validation augmentation comparison', '',
              'Seed 42; fixed split; 3 head epochs + 7 layer4 epochs for every run.',
              'Checkpoints selected by clean validation Macro-F1. Test not evaluated.',
              'Single-seed exploratory results; repeat across seeds before firm conclusions.', '',
              '| Condition | Baseline | Light | Blur | Both |',
              '|---|---:|---:|---:|---:|']
    for condition in plan['conditions']:
        scores = [float(next(r['macro_f1'] for r in rows
                             if r['augmentation'] == aug and r['condition'] == condition))
                  for aug, _ in runs]
        report.append('| ' + condition + ' | ' + ' | '.join(f'{s:.4f}' for s in scores) + ' |')
    report += ['', 'Scores above are Macro-F1 on the same 375 validation images.',
               'Brightness and blur are separate synthetic conditions applied after resize/crop.',
               'Training blur and evaluation blur use different implementations, as in the existing preview.',
               'These results do not estimate performance in all real-world environments.', '']
    (ROOT / 'VALIDATION_COMPARISON.md').write_text('\n'.join(report), encoding='utf-8')
    from train_baseline import plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, conditions, title in [
        (axes[0], plan['conditions'][:4], 'Reduced brightness'),
        (axes[1], ['clean'] + plan['conditions'][4:], 'Gaussian blur'),
    ]:
        for augmentation, _ in runs:
            scores = [float(next(r['macro_f1'] for r in rows
                                 if r['augmentation'] == augmentation and r['condition'] == condition))
                      for condition in conditions]
            axis.plot(range(len(conditions)), scores, marker='o', label=augmentation)
        axis.set(xticks=range(len(conditions)), xticklabels=conditions,
                 ylabel='Validation Macro-F1', ylim=(0, 1), title=title)
        axis.tick_params(axis='x', rotation=25)
        axis.grid(alpha=.2)
        axis.legend()
    fig.suptitle('Seed 42 exploratory comparison - validation only')
    fig.tight_layout()
    fig.savefig(ROOT / 'outputs' / 'validation_comparison.png', dpi=160)
    plt.close(fig)
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
