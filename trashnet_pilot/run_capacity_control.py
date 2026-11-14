"""Three-seed same-stage capacity control; no tuning or test-set evaluation."""
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from run_baseline_check import dataset_audit
from run_fixed_fusion_confirmation import record_run, CONDITIONS

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]
SOURCES = ['train_baseline.py', 'model_utils.py', 'image_framing.py', 'multiscale_model.py', 'capacity_control.py']
LABELS = {'control': '原网络匹配强对照', 'fixed': '跨层固定融合', 'legacy': '历史 layer4 增强', 'capacity': '同层等容量残差头'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capacity_row(seed, common, plan):
    name = f'capacity_residual_seed{seed}'
    out = ROOT/'outputs'/name
    config = read_json(out/'config.json')
    expected = dict(common, head='residual_capacity', architecture='resnet18', seed=seed, run_name=name,
                    split_manifest_sha256=plan['split_manifest_sha256'])
    if any(config.get(key) != value for key, value in expected.items()) or config['implementation_sha256'] != plan['implementation_sha256']:
        raise RuntimeError('Capacity run differs from fixed plan: '+name)
    path = out/'validation_robustness_extended.csv'
    if not path.exists():
        subprocess.run([sys.executable, str(ROOT/'preview_robustness.py'), '--run-name', name, '--include-bright'], check=True)
    with path.open(encoding='utf-8', newline='') as stream:
        probes = list(csv.DictReader(stream))
    if [probe['condition'] for probe in probes] != CONDITIONS or any(probe['split'] != 'validation' or int(probe['count']) != 375 for probe in probes):
        raise RuntimeError('Wrong robustness protocol: '+name)
    scores = {probe['condition']: float(probe['macro_f1']) for probe in probes}
    metrics, summary = read_json(out/'val_metrics.json'), read_json(out/'summary.json')
    if metrics['count'] != 375 or summary['test_evaluated'] is not False or abs(scores['clean']-metrics['macro_f1']) > 1e-12:
        raise RuntimeError('Invalid validation record: '+name)
    bright = statistics.mean(value for key, value in scores.items() if key.startswith('brightness_'))
    blur = statistics.mean(value for key, value in scores.items() if key.startswith('blur_'))
    cm = metrics['confusion_matrix']
    row = dict(recipe='capacity', seed=seed, run_name=name, best_epoch=summary['best_epoch'],
               accuracy=metrics['accuracy'], macro_f1=metrics['macro_f1'],
               correct=sum(cm[index][index] for index in range(6)), robustness_score=(bright+blur)/2,
               brightness_mean_f1=bright, blur_mean_f1=blur,
               plastic_recall=metrics['classification_report']['plastic']['recall'],
               glass_recall=metrics['classification_report']['glass']['recall'],
               plastic_as_glass=cm[4][1], glass_as_plastic=cm[1][4],
               dark_02_f1=scores['brightness_0.2'], blur_4_f1=scores['blur_4'], bright_17_f1=scores['brightness_1.7'],
               training_validation_seconds=summary['total_training_validation_seconds'],
               peak_gpu_allocated_mb=summary['peak_gpu_allocated_mb'])
    return row, [dict(recipe='capacity', seed=seed, run_name=name, **probe) for probe in probes]


def main():
    reference_path = ROOT/'outputs'/'fixed_fusion_confirmation_results.json'
    reference = read_json(reference_path)
    artifact_path = ROOT/'outputs'/'fixed_fusion_confirmation_artifacts.json'
    artifacts = read_json(artifact_path)
    if digest(reference_path) != artifacts['results_sha256']:
        raise RuntimeError('Reference result differs from its saved artifact manifest')
    for relative, fingerprint in artifacts['model_artifact_sha256'].items():
        if digest(ROOT/relative) != fingerprint:
            raise RuntimeError('Reference artifact changed: '+relative)
    common = dict(reference['plan']['common'])
    manifest_hash = digest(ROOT/'data'/'split_manifest.csv')
    if manifest_hash != reference['plan']['split_manifest_sha256']:
        raise RuntimeError('Fixed split changed')
    plan = dict(common=common, new_head='residual_capacity', architecture='resnet18', hidden_width=128,
                width_selection='Nearest integer parameter budget: 1025*h+512 versus fusion 131584 added parameters; no score-based width search',
                seeds=SEEDS, maximum_new_training_runs=3, conditions=CONDITIONS,
                parameter_counts=dict(linear_network=11179590, fixed_fusion=11311174, capacity_control=11311302, difference=128),
                split_manifest_sha256=manifest_hash, implementation_sha256={name: digest(ROOT/name) for name in SOURCES},
                reference_results_sha256=digest(reference_path), reference_artifact_manifest_sha256=digest(artifact_path),
                reference_training_source=reference['plan']['implementation_sha256'],
                checkpoint_selection=reference['plan']['checkpoint_selection'], primary_metric=reference['plan']['primary_metric'],
                added_value_rule=dict(min_mean_robustness_gain_over_capacity=.005, minimum_positive_paired_seeds=2,
                                      max_mean_clean_f1_drop=.01, max_mean_plastic_recall_drop=.02),
                test_evaluated=False, user_feedback_photo_used_for_selection=False,
                limitation='Same parameter count does not equal identical topology or optimization. This control cannot prove a unique causal mechanism.')
    path = ROOT/'outputs'/'capacity_control_plan.json'
    if path.exists():
        if read_json(path) != plan:
            raise RuntimeError('Existing capacity control plan differs')
    else:
        path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    (ROOT/'outputs'/'capacity_control_audit.json').write_text(json.dumps(dataset_audit(), indent=2), encoding='utf-8')
    snapshot = ROOT/'sources'/'training_versions'/'capacity_control_20261110'
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        destination = snapshot/name
        if destination.exists() and destination.read_bytes() != (ROOT/name).read_bytes():
            raise RuntimeError('Immutable source archive differs: '+name)
        destination.write_bytes((ROOT/name).read_bytes())
    (snapshot/'fingerprints.json').write_text(json.dumps(plan['implementation_sha256'], indent=2), encoding='utf-8')
    for seed in SEEDS:
        name = f'capacity_residual_seed{seed}'
        out = ROOT/'outputs'/name
        print('CAPACITY CONTROL: '+name, flush=True)
        if not (out/'summary.json').exists():
            if (out/'best_model.pt').exists():
                raise RuntimeError('Inspect incomplete run before resuming: '+name)
            expected = dict(common, head='residual_capacity', architecture='resnet18', seed=seed, run_name=name)
            command = [sys.executable, str(ROOT/'train_baseline.py')]
            for key, value in expected.items():
                command += ['--'+key.replace('_', '-'), str(value)]
            subprocess.run(command, check=True)
    rows, condition_rows = [], []
    for seed in SEEDS:
        for recipe in ['control', 'fixed', 'legacy']:
            row, probes = record_run(recipe, seed, common, reference['plan'])
            rows.append(row)
            condition_rows.extend(probes)
        row, probes = capacity_row(seed, common, plan)
        rows.append(row)
        condition_rows.extend(probes)
    recipes = ['control', 'fixed', 'capacity', 'legacy']
    keys = ['accuracy', 'macro_f1', 'robustness_score', 'plastic_recall', 'glass_recall', 'dark_02_f1', 'blur_4_f1', 'bright_17_f1']
    aggregate = {}
    for recipe in recipes:
        subset = [row for row in rows if row['recipe'] == recipe]
        aggregate[recipe] = {key: dict(mean=statistics.mean(row[key] for row in subset), sample_sd=statistics.stdev(row[key] for row in subset)) for key in keys}
    paired = []
    for seed in SEEDS:
        fixed = next(row for row in rows if row['recipe'] == 'fixed' and row['seed'] == seed)
        capacity = next(row for row in rows if row['recipe'] == 'capacity' and row['seed'] == seed)
        paired.append(dict(seed=seed, robustness_gain=fixed['robustness_score']-capacity['robustness_score'],
                           clean_f1_gain=fixed['macro_f1']-capacity['macro_f1'], plastic_recall_gain=fixed['plastic_recall']-capacity['plastic_recall']))
    rule = plan['added_value_rule']
    checks = dict(mean_robustness_gain=aggregate['fixed']['robustness_score']['mean'] >= aggregate['capacity']['robustness_score']['mean']+rule['min_mean_robustness_gain_over_capacity'],
                  positive_seeds=sum(row['robustness_gain'] > 0 for row in paired) >= rule['minimum_positive_paired_seeds'],
                  clean_preserved=aggregate['fixed']['macro_f1']['mean'] >= aggregate['capacity']['macro_f1']['mean']-rule['max_mean_clean_f1_drop'],
                  plastic_preserved=aggregate['fixed']['plastic_recall']['mean'] >= aggregate['capacity']['plastic_recall']['mean']-rule['max_mean_plastic_recall_drop'])
    result = dict(plan=plan, rows=rows, aggregate=aggregate, paired=paired, added_value_checks=checks,
                  supports_added_value_over_this_capacity_control=all(checks.values()),
                  test_evaluated=False, deployment_changed=False, real_world_accuracy_unknown=True,
                  note='Comparison with one near-equal-capacity same-stage residual head; not proof that all gains have a unique cross-stage cause.')
    (ROOT/'outputs'/'capacity_control_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for filename, records in [('capacity_control_runs.csv', rows), ('capacity_control_conditions.csv', condition_rows), ('capacity_control_paired.csv', paired)]:
        with (ROOT/'outputs'/filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    for relative, fingerprint in artifacts['model_artifact_sha256'].items():
        if digest(ROOT/relative) != fingerprint:
            raise RuntimeError('Reference artifact changed during study: '+relative)
    if digest(reference_path) != plan['reference_results_sha256'] or digest(ROOT/'data'/'split_manifest.csv') != manifest_hash:
        raise RuntimeError('Reference decisions or split changed')
    report = ['# 参数容量对照：三个种子结果', '',
              '新对照只使用 layer4 描述，经 512→128→512 的残差分类头输出六类；不加入 layer3 描述。与固定融合总参数只差 128，无额外 Dropout。',
              '固定训练设置、三种子、完整图片输入与干扰条件；每次按干净验证 Macro-F1 选权重。没有针对结果调宽度或超参数。', '',
              '| 模型 | Seed | 正确 / 375 | 干净准确率 | 干净 F1 | 均衡干扰 F1 | 塑料召回 | 强模糊 F1 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        report.append(f"| {LABELS[row['recipe']]} | {row['seed']} | {row['correct']} | {row['accuracy']:.2%} | {row['macro_f1']:.4f} | {row['robustness_score']:.4f} | {row['plastic_recall']:.2%} | {row['blur_4_f1']:.4f} |")
    report += ['', '## 三种子均值 ± 样本标准差', '', '| 指标 | 原网络 | 固定融合 | 同层容量对照 | 历史增强 |', '|---|---:|---:|---:|---:|']
    for key in keys:
        report.append('| '+key+' | '+' | '.join(f"{aggregate[recipe][key]['mean']:.4f} ± {aggregate[recipe][key]['sample_sd']:.4f}" for recipe in recipes)+' |')
    report += ['', '表中指标为 0–1，标准差只描述训练随机性；三次训练仍使用同一验证图片。', '',
               '## 固定融合相对容量对照的差值', '', '| Seed | 均衡干扰差值 | 干净 F1 差值 | 塑料召回差值 |', '|---|---:|---:|---:|']
    for row in paired:
        report.append(f"| {row['seed']} | {row['robustness_gain']:+.4f} | {row['clean_f1_gain']:+.4f} | {row['plastic_recall_gain']:+.4f} |")
    report += ['', f"预定检查：`{json.dumps(checks)}`。", '',
               '结论：'+('在这一个近等容量对照下支持跨层方案有附加价值，仍不能证明唯一机制或实拍可靠性。' if all(checks.values()) else '没有达到全部预定附加价值条件，不能把既有收益单独归因于跨层信息。'), '',
               '## 方法边界', '',
               '- 隐藏宽度 128 按参数预算选择；同层头新增 131712 参数，固定融合新增 131584 参数。',
               '- 使用相同六分类器初始化和 ImageNet 主干；新头需要不同的额外层，不能保证函数空间或优化难度相同。',
               '- 本轮仅加入一种容量对照，不对结果追加搜索；所有代码、原权重和原始结果保留。',
               '- 旧实现与新增头选项有不同源码指纹；默认网络/优化器和原权重重载经过回归检查，训练核心新版本单独归档。',
               '- 所有成绩是参与选模/研究的验证成绩；反馈水瓶不参与选择，测试集未评估，演示默认未变。', '']
    (ROOT/'CAPACITY_CONTROL_RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
