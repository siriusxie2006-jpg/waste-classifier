"""Run isolated integration ablations; leave historical outputs and test images alone."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = [
    dict(run_name='merged_control_seed42', augmentation='baseline', class_weighting='none', head='linear', finetune_scope='layer4'),
    dict(run_name='merged_weighted_seed42', augmentation='baseline', class_weighting='inverse', head='linear', finetune_scope='layer4'),
    dict(run_name='merged_both_weighted_seed42', augmentation='both', class_weighting='inverse', head='linear', finetune_scope='layer4'),
    dict(run_name='merged_mlp_seed42', augmentation='baseline', class_weighting='none', head='mlp', finetune_scope='layer4'),
    dict(run_name='merged_deeper_seed42', augmentation='baseline', class_weighting='none', head='linear', finetune_scope='layer3_layer4'),
]


def main():
    manifest_hash = hashlib.sha256((ROOT/'data'/'split_manifest.csv').read_bytes()).hexdigest()
    plan = dict(seed=42, warmup_epochs=3, finetune_epochs=7, batch_size=32, workers=2,
                split_manifest_sha256=manifest_hash, runs=RUNS,
                selection='clean validation Macro-F1', test_evaluated=False,
                conditions=['clean', 'brightness_0.7', 'brightness_0.4', 'brightness_0.2',
                            'blur_1', 'blur_2', 'blur_4', 'brightness_1.3', 'brightness_1.7'],
                note='Exploratory single-seed ablations; no optimizer or training-budget changes')
    plan_path = ROOT/'outputs'/'merged_experiment_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise RuntimeError('Existing integration plan differs')
    else:
        plan_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    for run in RUNS:
        out = ROOT/'outputs'/run['run_name']
        if not (out/'summary.json').exists():
            command = [sys.executable, str(ROOT/'train_baseline.py')]
            for key, value in run.items():
                command += ['--'+key.replace('_', '-'), value]
            subprocess.run(command, check=True)
        config = json.loads((out/'config.json').read_text(encoding='utf-8'))
        for key in ['seed', 'warmup_epochs', 'finetune_epochs', 'batch_size', 'workers', 'split_manifest_sha256']:
            if config[key] != plan[key]:
                raise RuntimeError(f'Incompatible setting: {run["run_name"]}: {key}')
        if any(config[key] != value for key, value in run.items()):
            raise RuntimeError('Run config does not match the experiment plan')
    names = ['baseline_resnet18_seed42', 'resnet18_both_seed42'] + [r['run_name'] for r in RUNS]
    all_rows, report = [], ['# Integration validation results', '',
        'Seed 42 exploratory results; fixed split, 3 warmup + 7 fine-tuning epochs; test not evaluated.',
        'All checkpoints selected using clean validation Macro-F1. Bright conditions are newly added validation probes.', '',
        '| Run | Clean accuracy | Clean Macro-F1 | Trash recall | Dark 0.2 F1 | Blur 4 F1 | Bright 1.7 F1 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        out = ROOT/'outputs'/name
        extended = out/'validation_robustness_extended.csv'
        if not extended.exists():
            subprocess.run([sys.executable, str(ROOT/'preview_robustness.py'), '--run-name', name, '--include-bright'], check=True)
        with extended.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if [row['condition'] for row in rows] != plan['conditions'] or any(int(row['count']) != 375 or row['split'] != 'validation' for row in rows):
            raise RuntimeError(f'Incomplete robustness evaluation: {name}')
        metrics = json.loads((out/'val_metrics.json').read_text(encoding='utf-8'))
        scores = {row['condition']:float(row['macro_f1']) for row in rows}
        if abs(scores['clean'] - metrics['macro_f1']) > 1e-12:
            raise RuntimeError('Clean robustness and model metrics differ')
        report.append(f"| {name} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | {metrics['classification_report']['trash']['recall']:.4f} | {scores['brightness_0.2']:.4f} | {scores['blur_4']:.4f} | {scores['brightness_1.7']:.4f} |")
        all_rows.extend(dict(run_name=name, **row) for row in rows)
    with (ROOT/'outputs'/'merged_validation_comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    # Check the new default training path against the historical control.
    old = json.loads((ROOT/'outputs'/'baseline_resnet18_seed42'/'val_metrics.json').read_text(encoding='utf-8'))
    new = json.loads((ROOT/'outputs'/'merged_control_seed42'/'val_metrics.json').read_text(encoding='utf-8'))
    report += ['', f"Refactored control confusion matrix matches historical baseline: {old['confusion_matrix'] == new['confusion_matrix']}.",
               'Use the new control for integration ablations. If controls differ, do not attribute that difference to a new method.',
               'Do not infer that combining every available feature improves performance. Repeat across seeds before final selection.', '']
    (ROOT/'MERGED_RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
    for name in ['merged_control_seed42', 'merged_both_weighted_seed42']:
        subprocess.run([sys.executable, str(ROOT/'analyze_errors.py'), '--run-name', name, '--max-errors', '4'], check=True)
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
