"""Describe all six classes, paired mistakes, and curves from completed records."""
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'cache'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    result_path = ROOT/'outputs'/'fixed_fusion_confirmation_results.json'
    result_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
    result = json.loads(result_path.read_text(encoding='utf-8'))
    recipes = ['control', 'fixed', 'legacy']
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    labels = {'control': '匹配增强强对照', 'fixed': '固定融合', 'legacy': '历史 layer4 增强'}
    by_class = []
    for recipe in recipes:
        metrics = [json.loads((ROOT/'outputs'/row['run_name']/'val_metrics.json').read_text(encoding='utf-8'))
                   for row in result['rows'] if row['recipe'] == recipe]
        if len(metrics) != 3:
            raise RuntimeError('Three completed seeds required')
        for class_name in classes:
            support = {int(item['classification_report'][class_name]['support']) for item in metrics}
            if len(support) != 1:
                raise RuntimeError('Per-class evaluation sets differ')
            recalls = [item['classification_report'][class_name]['recall'] for item in metrics]
            f1s = [item['classification_report'][class_name]['f1-score'] for item in metrics]
            by_class.append(dict(recipe=recipe, class_name=class_name, support=support.pop(),
                                 recall_mean=statistics.mean(recalls), recall_seed_sd=statistics.stdev(recalls),
                                 f1_mean=statistics.mean(f1s), f1_seed_sd=statistics.stdev(f1s)))
    paired = []
    for seed in [42, 43, 44]:
        predictions = {}
        for recipe in ['control', 'fixed']:
            record = next(row for row in result['rows'] if row['seed'] == seed and row['recipe'] == recipe)
            with (ROOT/'outputs'/record['run_name']/'val_predictions.csv').open(encoding='utf-8', newline='') as stream:
                predictions[recipe] = {row['path']: row for row in csv.DictReader(stream)}
        control, fixed = predictions['control'], predictions['fixed']
        if control.keys() != fixed.keys() or len(control) != 375:
            raise RuntimeError('Paired prediction paths differ')
        for class_name in ['all']+classes:
            counts = dict(both_correct=0, both_wrong=0, fixed_only_correct=0, control_only_correct=0)
            for path, before in control.items():
                after = fixed[path]
                if before['true_class'] != after['true_class']:
                    raise RuntimeError('Paired labels differ')
                if class_name != 'all' and before['true_class'] != class_name:
                    continue
                old_ok = before['true_class'] == before['predicted_class']
                new_ok = after['true_class'] == after['predicted_class']
                key = 'both_correct' if old_ok and new_ok else 'both_wrong' if not old_ok and not new_ok else 'fixed_only_correct' if new_ok else 'control_only_correct'
                counts[key] += 1
            paired.append(dict(seed=seed, class_name=class_name, **counts))
    for filename, records in [('fixed_fusion_per_class.csv', by_class), ('fixed_fusion_paired_errors.csv', paired)]:
        with (ROOT/'outputs'/filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    with (ROOT/'outputs'/'fixed_fusion_confirmation_conditions.csv').open(encoding='utf-8', newline='') as stream:
        curves = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout='constrained')
    conditions = [['brightness_0.2', 'brightness_0.4', 'brightness_0.7', 'clean', 'brightness_1.3', 'brightness_1.7'],
                  ['clean', 'blur_1', 'blur_2', 'blur_4']]
    for recipe, label, style in [('control', 'Matched strong control', '-'), ('fixed', 'Fixed fusion', '-'), ('legacy', 'Historical layer4 augmentation', '--')]:
        for axis, names, positions in zip(axes, conditions, [[.2, .4, .7, 1., 1.3, 1.7], [0, 1, 2, 4]]):
            values = [[float(row['macro_f1']) for row in curves if row['recipe'] == recipe and row['condition'] == name] for name in names]
            if any(len(group) != 3 for group in values):
                raise RuntimeError('Three seeds required for every condition')
            means = np.array([statistics.mean(group) for group in values])
            deviations = np.array([statistics.stdev(group) for group in values])
            line, = axis.plot(positions, means, marker='o', linestyle=style, label=label)
            axis.fill_between(positions, np.clip(means-deviations, 0, 1), np.clip(means+deviations, 0, 1), color=line.get_color(), alpha=.12)
    for axis, title, xlabel in zip(axes, ['Brightness changes', 'Gaussian blur'], ['Brightness multiplier', 'Blur radius']):
        axis.set(title=title, xlabel=xlabel, ylabel='Mean validation Macro-F1', ylim=(0, 1))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('Three training seeds on the same 375 validation images\nBands: +/- one seed SD, not confidence intervals for new images', fontsize=11)
    fig.savefig(ROOT/'outputs'/'fixed_fusion_confirmation_curves.png', dpi=160)
    plt.close(fig)
    report = ['# 固定融合复核：逐类与同图错误分析', '',
              '| 类别 | 图片数 | 匹配对照召回 | 固定融合召回 | 历史增强召回 | 匹配对照 F1 | 固定融合 F1 | 历史增强 F1 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for class_name in classes:
        selected = [next(row for row in by_class if row['recipe'] == recipe and row['class_name'] == class_name) for recipe in recipes]
        report.append('| '+class_name+' | '+str(selected[0]['support'])+' | '+' | '.join(f"{row['recall_mean']:.2%}" for row in selected)+' | '+' | '.join(f"{row['f1_mean']:.4f}" for row in selected)+' |')
    report += ['', '以上是三次训练的均值，图片数仍是每次相同验证集中的图片数。逐类种子标准差保存在 CSV。', '',
               '| Seed | 两者都对 | 两者都错 | 仅固定融合对 | 仅匹配对照对 |', '|---|---:|---:|---:|---:|']
    for row in paired:
        if row['class_name'] == 'all':
            report.append(f"| {row['seed']} | {row['both_correct']} | {row['both_wrong']} | {row['fixed_only_correct']} | {row['control_only_correct']} |")
    report += ['', '同图配对变化说明增加正确和新增错误同时存在，不能只看净增加。',
               '历史增强的微调范围不同，只作性能背景；同一验证集反复选模不构成独立最终评估。', '']
    (ROOT/'FIXED_FUSION_CLASS_ANALYSIS.md').write_text('\n'.join(report), encoding='utf-8')
    if hashlib.sha256(result_path.read_bytes()).hexdigest() != result_hash:
        raise RuntimeError('Confirmation decisions changed during descriptive analysis')
    print('Saved six-class analysis, paired errors and three-seed curves', flush=True)


if __name__ == '__main__':
    main()
