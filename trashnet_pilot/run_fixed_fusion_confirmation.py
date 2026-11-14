"""Four-run confirmation of fixed fusion; preserve the failed adaptive pilot."""
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from run_baseline_check import dataset_audit

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]
ARCHITECTURES = {'control': 'resnet18', 'fixed': 'multiscale_fixed'}
LABELS = {'control': '匹配增强强对照', 'fixed': '固定双层融合', 'legacy': '历史 layer4 增强方案'}
CONDITIONS = ['clean', 'brightness_0.7', 'brightness_0.4', 'brightness_0.2',
              'blur_1', 'blur_2', 'blur_4', 'brightness_1.3', 'brightness_1.7']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def name_for(recipe, seed):
    if recipe == 'legacy':
        return f'full_image_both_weighted_seed{seed}'
    return f'multiscale_pilot_{recipe}_seed42' if seed == 42 else f'multiscale_confirm_{recipe}_seed{seed}'


def verify_run(recipe, seed, common, plan):
    name = name_for(recipe, seed)
    out = ROOT/'outputs'/name
    config = read_json(out/'config.json')
    expected = dict(common, seed=seed, run_name=name)
    if recipe == 'legacy':
        expected['finetune_scope'] = 'layer4'
    else:
        expected['architecture'] = ARCHITECTURES[recipe]
    effective = dict(config, architecture=config.get('architecture', 'resnet18'),
                     head_lr=config.get('head_lr', config['learning_rates']['head']),
                     backbone_lr=config.get('backbone_lr', config['learning_rates']['trainable_backbone']))
    for key, value in dict(expected, split_manifest_sha256=plan['split_manifest_sha256']).items():
        if effective.get(key) != value:
            raise RuntimeError(f'{name}: incompatible {key}')
    if recipe != 'legacy' and config['implementation_sha256'] != plan['implementation_sha256']:
        raise RuntimeError('Structural training source changed: '+name)
    metrics, summary = read_json(out/'val_metrics.json'), read_json(out/'summary.json')
    if metrics['count'] != 375 or summary['test_evaluated'] is not False or abs(metrics['macro_f1']-summary['validation_macro_f1']) > 1e-12:
        raise RuntimeError('Invalid validation record: '+name)
    return config, metrics, summary


def record_run(recipe, seed, common, plan):
    name = name_for(recipe, seed)
    out = ROOT/'outputs'/name
    _, metrics, summary = verify_run(recipe, seed, common, plan)
    path = out/'validation_robustness_extended.csv'
    if not path.exists():
        if recipe == 'legacy' or seed == 42:
            raise RuntimeError('Historical robustness result missing; do not silently reevaluate: '+name)
        subprocess.run([sys.executable, str(ROOT/'preview_robustness.py'), '--run-name', name, '--include-bright'], check=True)
    with path.open(encoding='utf-8', newline='') as stream:
        probes = list(csv.DictReader(stream))
    if [probe['condition'] for probe in probes] != CONDITIONS or any(probe['split'] != 'validation' or int(probe['count']) != 375 for probe in probes):
        raise RuntimeError('Inconsistent robustness protocol: '+name)
    scores = {probe['condition']: float(probe['macro_f1']) for probe in probes}
    if abs(scores['clean']-metrics['macro_f1']) > 1e-12:
        raise RuntimeError('Clean and corruption input protocols differ: '+name)
    bright = statistics.mean(value for key, value in scores.items() if key.startswith('brightness_'))
    blur = statistics.mean(value for key, value in scores.items() if key.startswith('blur_'))
    cm = metrics['confusion_matrix']
    row = dict(recipe=recipe, seed=seed, run_name=name, best_epoch=summary['best_epoch'],
               accuracy=metrics['accuracy'], macro_f1=metrics['macro_f1'],
               correct=sum(cm[index][index] for index in range(6)), robustness_score=(bright+blur)/2,
               brightness_mean_f1=bright, blur_mean_f1=blur,
               plastic_recall=metrics['classification_report']['plastic']['recall'],
               glass_recall=metrics['classification_report']['glass']['recall'],
               plastic_as_glass=cm[4][1], glass_as_plastic=cm[1][4],
               dark_02_f1=scores['brightness_0.2'], blur_4_f1=scores['blur_4'], bright_17_f1=scores['brightness_1.7'],
               training_validation_seconds=summary['total_training_validation_seconds'],
               peak_gpu_allocated_mb=summary['peak_gpu_allocated_mb'])
    return row, [dict(recipe=recipe, seed=seed, run_name=name, **probe) for probe in probes]


def main():
    pilot = read_json(ROOT/'outputs'/'multiscale_pilot_plan.json')
    common = {key: value for key, value in pilot['common'].items() if key != 'seed'}
    if digest(ROOT/'data'/'split_manifest.csv') != pilot['split_manifest_sha256']:
        raise RuntimeError('The fixed split changed')
    source_changed = any(digest(ROOT/name) != fingerprint for name, fingerprint in pilot['implementation_sha256'].items())
    if source_changed:
        complete = all((ROOT/'outputs'/name_for(recipe, seed)/filename).exists()
                       for seed in SEEDS for recipe in ['control', 'fixed', 'legacy']
                       for filename in ['summary.json', 'validation_robustness_extended.csv'])
        if not complete:
            raise RuntimeError('Source changed; incomplete historical confirmation cannot resume with new code')
        print('Regenerating complete confirmation from saved records and original source fingerprints; no new training or evaluation.', flush=True)
    references = [name_for(recipe, seed) for recipe, seeds in [('control', [42]), ('fixed', [42]), ('legacy', SEEDS)] for seed in seeds]
    protected = {str(Path(name)/filename): digest(ROOT/'outputs'/name/filename)
                 for name in references for filename in ['best_model.pt', 'val_metrics.json', 'val_predictions.csv', 'validation_robustness_extended.csv']}
    plan = dict(common=common, architectures=ARCHITECTURES, seeds=SEEDS, new_seeds=[43, 44],
                maximum_new_training_runs=4, conditions=CONDITIONS,
                split_manifest_sha256=pilot['split_manifest_sha256'], implementation_sha256=pilot['implementation_sha256'],
                pilot_plan_sha256=digest(ROOT/'outputs'/'multiscale_pilot_plan.json'),
                pilot_results_sha256=digest(ROOT/'outputs'/'multiscale_pilot_results.json'),
                protected_reference_files=protected,
                revised_hypothesis='The fixed fusion, selected after the seed42 pilot, warrants a separate confirmation; adaptive gating remains rejected',
                primary_metric=pilot['primary_metric'], checkpoint_selection=pilot['checkpoint_selection'],
                confirmation_rule=dict(both_new_seeds_positive=True, min_mean_robustness_gain=.01,
                                       max_mean_clean_f1_drop=.01, max_mean_plastic_recall_drop=.02),
                legacy_context='Same input and augmentation but different fine-tuning scope; report performance without treating differences as a single-module ablation',
                test_evaluated=False, user_feedback_photo_used_for_selection=False,
                note='Same validation images; extra training seeds do not provide an independent new-scene test')
    path = ROOT/'outputs'/'fixed_fusion_confirmation_plan.json'
    if path.exists():
        if read_json(path) != plan:
            raise RuntimeError('Confirmation plan differs; preserve this study')
    else:
        path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    (ROOT/'outputs'/'fixed_fusion_confirmation_audit.json').write_text(json.dumps(dataset_audit(), indent=2), encoding='utf-8')
    for seed in [43, 44]:
        for recipe, architecture in ARCHITECTURES.items():
            name = name_for(recipe, seed)
            out = ROOT/'outputs'/name
            print('FIXED FUSION CONFIRMATION: '+name, flush=True)
            if not (out/'summary.json').exists():
                if (out/'best_model.pt').exists():
                    raise RuntimeError('Inspect incomplete run before resuming: '+name)
                expected = dict(common, architecture=architecture, seed=seed, run_name=name)
                command = [sys.executable, str(ROOT/'train_baseline.py')]
                for key, value in expected.items():
                    command += ['--'+key.replace('_', '-'), str(value)]
                subprocess.run(command, check=True)
            verify_run(recipe, seed, common, plan)
    rows, condition_rows = [], []
    for seed in SEEDS:
        for recipe in ['control', 'fixed', 'legacy']:
            row, probes = record_run(recipe, seed, common, plan)
            rows.append(row)
            condition_rows.extend(probes)
    keys = ['accuracy', 'macro_f1', 'robustness_score', 'plastic_recall', 'glass_recall', 'dark_02_f1', 'blur_4_f1', 'bright_17_f1']
    aggregate = {}
    for recipe in ['control', 'fixed', 'legacy']:
        subset = [row for row in rows if row['recipe'] == recipe]
        aggregate[recipe] = {key: dict(mean=statistics.mean(row[key] for row in subset),
                                     sample_sd=statistics.stdev(row[key] for row in subset)) for key in keys}
    paired = []
    for seed in SEEDS:
        control = next(row for row in rows if row['recipe'] == 'control' and row['seed'] == seed)
        fixed = next(row for row in rows if row['recipe'] == 'fixed' and row['seed'] == seed)
        legacy = next(row for row in rows if row['recipe'] == 'legacy' and row['seed'] == seed)
        paired.append(dict(seed=seed, robustness_gain=fixed['robustness_score']-control['robustness_score'],
                           clean_f1_gain=fixed['macro_f1']-control['macro_f1'], plastic_recall_gain=fixed['plastic_recall']-control['plastic_recall'],
                           historical_robustness_difference=fixed['robustness_score']-legacy['robustness_score']))
    new_only = {recipe: {key: statistics.mean(row[key] for row in rows if row['recipe'] == recipe and row['seed'] in [43, 44]) for key in keys}
                for recipe in ['control', 'fixed']}
    rule = plan['confirmation_rule']
    mean_gain = aggregate['fixed']['robustness_score']['mean']-aggregate['control']['robustness_score']['mean']
    checks = dict(both_additional_seeds_positive=all(row['robustness_gain'] > 0 for row in paired if row['seed'] in [43, 44]),
                  mean_gain_sufficient=mean_gain >= rule['min_mean_robustness_gain'],
                  clean_preserved=aggregate['fixed']['macro_f1']['mean'] >= aggregate['control']['macro_f1']['mean']-rule['max_mean_clean_f1_drop'],
                  plastic_preserved=aggregate['fixed']['plastic_recall']['mean'] >= aggregate['control']['plastic_recall']['mean']-rule['max_mean_plastic_recall_drop'])
    result = dict(plan=plan, rows=rows, aggregate=aggregate, paired=paired, additional_seeds_only_means=new_only,
                  confirmation_checks=checks, meets_confirmation_criteria=all(checks.values()),
                  mean_primary_gain_vs_matched_control=mean_gain,
                  mean_primary_difference_vs_historical=aggregate['fixed']['robustness_score']['mean']-aggregate['legacy']['robustness_score']['mean'],
                  test_evaluated=False, deployment_changed=False, real_world_accuracy_unknown=True,
                  note='Fixed fusion was selected on seed42; report additional seeds separately. Seed SD is not uncertainty over independent new images.')
    (ROOT/'outputs'/'fixed_fusion_confirmation_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for filename, records in [('fixed_fusion_confirmation_runs.csv', rows), ('fixed_fusion_confirmation_conditions.csv', condition_rows), ('fixed_fusion_confirmation_paired.csv', paired)]:
        with (ROOT/'outputs'/filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    for relative, fingerprint in protected.items():
        if digest(ROOT/'outputs'/relative) != fingerprint:
            raise RuntimeError('Protected historical record changed: '+relative)
    if digest(ROOT/'outputs'/'multiscale_pilot_results.json') != plan['pilot_results_sha256'] or digest(ROOT/'data'/'split_manifest.csv') != plan['split_manifest_sha256']:
        raise RuntimeError('Pilot results or fixed split changed')
    report = ['# 固定双层融合：三个种子复核', '',
              '保留 seed42 初试结果，新补 seed43/44 的匹配增强强对照与固定融合，共新增四次训练。结构和超参数不变，门控路线停止。',
              '同一 375 张验证图重复用于选模和研究，不能把三次训练当作新的独立实拍测试。', '',
              '| 模型 | Seed | 正确 / 375 | 干净准确率 | 干净 F1 | 均衡干扰 F1 | 塑料召回 | 强模糊 F1 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        report.append(f"| {LABELS[row['recipe']]} | {row['seed']} | {row['correct']} | {row['accuracy']:.2%} | {row['macro_f1']:.4f} | {row['robustness_score']:.4f} | {row['plastic_recall']:.2%} | {row['blur_4_f1']:.4f} |")
    report += ['', '## 三种子均值 ± 样本标准差', '', '| 指标 | 匹配强对照 | 固定融合 | 历史 layer4 增强 |', '|---|---:|---:|---:|']
    for key in keys:
        report.append('| '+key+' | '+' | '.join(f"{aggregate[recipe][key]['mean']:.4f} ± {aggregate[recipe][key]['sample_sd']:.4f}" for recipe in ['control', 'fixed', 'legacy'])+' |')
    report += ['', '表中指标为 0–1。标准差仅描述训练随机性，不是新场景置信区间。历史方案的微调范围不同，不能将其差值全部归因于融合模块。', '',
               '## 配对结果与预定检查', '', '| Seed | 干扰得分差值 | 干净 F1 差值 | 塑料召回差值 | 对历史方案干扰差值 |', '|---|---:|---:|---:|---:|']
    for row in paired:
        report.append(f"| {row['seed']} | {row['robustness_gain']:+.4f} | {row['clean_f1_gain']:+.4f} | {row['plastic_recall_gain']:+.4f} | {row['historical_robustness_difference']:+.4f} |")
    report += ['', f"确认检查：`{json.dumps(checks)}`。", '',
               '本轮结论：'+('达到预定复核条件，可继续容量对照及方法分析；还不是最终测试或部署结论。' if all(checks.values()) else '没有达到全部预定复核条件。保留结果，不能声称稳定提升，不追加种子或参数搜索。'), '',
               '## 单独看新增的 seed43/44', '', '| 模型 | 平均干净 F1 | 平均均衡干扰 F1 | 平均塑料召回 |', '|---|---:|---:|---:|']
    for recipe in ['control', 'fixed']:
        value = new_only[recipe]
        report.append(f"| {LABELS[recipe]} | {value['macro_f1']:.4f} | {value['robustness_score']:.4f} | {value['plastic_recall']:.2%} |")
    report += ['', 'seed42 曾用于挑选固定融合，因此额外列出两个新增种子；这仍没有消除重复使用同一验证集的选择偏差。', '',
               '## 边界与复现', '',
               '- 不修改最初门控方案的失败结论；本轮是明确另立的固定融合确认。',
               '- 无新结构、超参数、额外增强或数据变更；权重与历史预测文件指纹受保护。',
               '- 塑料/玻璃和每个干扰条件须报告取舍，不能只展示平均收益。',
               '- 参数量接近的分类头对照及独立评估仍需进一步完成，当前不能证明融合信息是唯一收益原因。',
               '- 反馈水瓶未用于选择模型，测试集未评估，演示默认未变。',
               '- 本组使用已有多尺度思想的具体变体，不声称首创多尺度或注意力。', '']
    (ROOT/'FIXED_FUSION_CONFIRMATION_RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
