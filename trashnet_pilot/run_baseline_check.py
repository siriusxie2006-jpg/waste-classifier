"""Paired three-seed baseline check; fixed split, existing recipe, validation only."""
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]
RECIPES = {
    'baseline': dict(augmentation='baseline', class_weighting='none', prefix='full_image_control'),
    'candidate': dict(augmentation='both', class_weighting='inverse', prefix='full_image_both_weighted'),
}
CONDITIONS = ['clean', 'brightness_0.7', 'brightness_0.4', 'brightness_0.2',
              'blur_1', 'blur_2', 'blur_4', 'brightness_1.3', 'brightness_1.7']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_config(name, expected, manifest_hash):
    out = ROOT/'outputs'/name
    config = read_json(out/'config.json')
    for key, value in {**expected, 'split_manifest_sha256': manifest_hash}.items():
        if config.get(key) != value:
            raise RuntimeError(f'{name}: {key} differs from fixed plan')
    if read_json(out/'summary.json').get('test_evaluated') is not False:
        raise RuntimeError(f'{name}: summary must identify a validation-only run')


def dataset_audit():
    with (ROOT/'data'/'split_manifest.csv').open(encoding='utf-8', newline='') as stream:
        records = list(csv.DictReader(stream))
    group_splits, pixel_splits = defaultdict(set), defaultdict(set)
    for record in records:
        group_splits[record['group_id']].add(record['split'])
        pixel_splits[record['pixel_sha256']].add(record['split'])
        if record['split'] in ('train', 'val') and not (ROOT/record['path']).is_file():
            raise FileNotFoundError(record['path'])
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise RuntimeError('A recorded similarity group crosses splits')
    if any(len(splits) > 1 for splits in pixel_splits.values()):
        raise RuntimeError('An exact pixel duplicate crosses splits')
    counts = dict(Counter(record['split'] for record in records))
    if counts != {'train': 1766, 'val': 375, 'test': 380}:
        raise RuntimeError(f'Unexpected fixed split sizes: {counts}')
    return dict(split_counts=counts,
                recorded_similarity_groups_cross_splits=False, exact_pixel_duplicates_cross_splits=False,
                train_image_sizes=dict(Counter(record['width']+'x'+record['height'] for record in records if record['split'] == 'train')),
                test_image_contents_opened=False,
                limitation='Recorded groups cannot guarantee discovery of every near duplicate or physical object identity')


def main():
    manifest_hash = hashlib.sha256((ROOT/'data'/'split_manifest.csv').read_bytes()).hexdigest()
    common = dict(warmup_epochs=3, finetune_epochs=7, batch_size=32, workers=2,
                  framing='fit', head='linear', finetune_scope='layer4')
    plan = dict(seeds=SEEDS, common=common, recipes=RECIPES, conditions=CONDITIONS,
                split_manifest_sha256=manifest_hash,
                selection='Best clean validation Macro-F1 checkpoint within each run; compare paired seeds descriptively',
                purpose='Check variation of the existing baseline and candidate before introducing another model change',
                pretrained_initialization='ImageNet ResNet18; independent initialization per seed/recipe',
                validation_used_for_selection=True, test_evaluated=False,
                user_feedback_photo_used_for_selection=False,
                implementation_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                       for name in ['train_baseline.py', 'model_utils.py', 'image_framing.py']})
    plan_path = ROOT/'outputs'/'baseline_check_plan.json'
    if plan_path.exists():
        recorded = read_json(plan_path)
        if recorded != plan:
            current_protocol = {key: value for key, value in plan.items() if key != 'implementation_sha256'}
            recorded_protocol = {key: value for key, value in recorded.items() if key != 'implementation_sha256'}
            if current_protocol != recorded_protocol:
                raise RuntimeError('Existing baseline check protocol differs; use a separate experiment')
            completed = all((ROOT/'outputs'/f'{recipe["prefix"]}_seed{seed}'/filename).exists()
                            for seed in SEEDS for recipe in RECIPES.values()
                            for filename in ['summary.json', 'validation_robustness_extended.csv'])
            if not completed:
                raise RuntimeError('Training implementation changed; incomplete historical study cannot resume with new code')
            plan = recorded
            print('Regenerating completed study from saved results and archived source fingerprints; no new training or evaluation.', flush=True)
    else:
        plan_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    audit = dataset_audit()
    (ROOT/'outputs'/'baseline_check_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    rows, condition_rows = [], []
    for seed in SEEDS:
        for recipe_name, recipe in RECIPES.items():
            name = f'{recipe["prefix"]}_seed{seed}'
            expected = dict(common, seed=seed, run_name=name,
                            augmentation=recipe['augmentation'], class_weighting=recipe['class_weighting'])
            out = ROOT/'outputs'/name
            print(f'BASELINE CHECK: {name}', flush=True)
            if not (out/'summary.json').exists():
                if (out/'best_model.pt').exists():
                    raise RuntimeError(f'Incomplete prior run needs inspection: {name}')
                command = [sys.executable, str(ROOT/'train_baseline.py')]
                for key, value in expected.items():
                    command += ['--'+key.replace('_', '-'), str(value)]
                subprocess.run(command, check=True)
            verify_config(name, expected, manifest_hash)
            metrics = read_json(out/'val_metrics.json')
            summary = read_json(out/'summary.json')
            if metrics['count'] != 375 or abs(metrics['macro_f1']-summary['validation_macro_f1']) > 1e-12:
                raise RuntimeError(f'Incomplete validation records: {name}')
            extended = out/'validation_robustness_extended.csv'
            if not extended.exists():
                subprocess.run([sys.executable, str(ROOT/'preview_robustness.py'), '--run-name', name, '--include-bright'], check=True)
            with extended.open(encoding='utf-8', newline='') as stream:
                probes = list(csv.DictReader(stream))
            if [probe['condition'] for probe in probes] != CONDITIONS or any(probe['split'] != 'validation' or int(probe['count']) != 375 for probe in probes):
                raise RuntimeError(f'Incorrect robustness protocol: {name}')
            scores = {probe['condition']: float(probe['macro_f1']) for probe in probes}
            if abs(scores['clean']-metrics['macro_f1']) > 1e-12:
                raise RuntimeError(f'Robustness input protocol differs: {name}')
            cm = metrics['confusion_matrix']
            row = dict(recipe=recipe_name, seed=seed, run_name=name, best_epoch=summary['best_epoch'],
                       accuracy=metrics['accuracy'], macro_f1=metrics['macro_f1'],
                       correct=sum(cm[index][index] for index in range(6)), count=metrics['count'],
                       plastic_recall=metrics['classification_report']['plastic']['recall'],
                       glass_recall=metrics['classification_report']['glass']['recall'],
                       plastic_as_glass=cm[4][1], glass_as_plastic=cm[1][4],
                       dark_02_f1=scores['brightness_0.2'], blur_4_f1=scores['blur_4'], bright_17_f1=scores['brightness_1.7'])
            rows.append(row)
            condition_rows.extend(dict(recipe=recipe_name, seed=seed, run_name=name, **probe) for probe in probes)
    paired = []
    for seed in SEEDS:
        predictions = {}
        for recipe_name, recipe in RECIPES.items():
            with (ROOT/'outputs'/f'{recipe["prefix"]}_seed{seed}'/'val_predictions.csv').open(encoding='utf-8', newline='') as stream:
                predictions[recipe_name] = {row['path']: row for row in csv.DictReader(stream)}
        baseline, candidate = predictions['baseline'], predictions['candidate']
        if baseline.keys() != candidate.keys() or len(baseline) != 375:
            raise RuntimeError('Paired evaluation image sets differ')
        wins = losses = 0
        for path, before in baseline.items():
            after = candidate[path]
            if before['true_class'] != after['true_class']:
                raise RuntimeError('Paired ground-truth labels differ')
            old_ok = before['true_class'] == before['predicted_class']
            new_ok = after['true_class'] == after['predicted_class']
            wins += int(new_ok and not old_ok)
            losses += int(old_ok and not new_ok)
        paired.append(dict(seed=seed, candidate_only_correct=wins, baseline_only_correct=losses, net_extra_correct=wins-losses))
    for filename, records in [('baseline_check_runs.csv', rows), ('baseline_check_robustness.csv', condition_rows), ('baseline_check_paired.csv', paired)]:
        with (ROOT/'outputs'/filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    aggregate = {}
    keys = ['accuracy', 'macro_f1', 'plastic_recall', 'glass_recall', 'dark_02_f1', 'blur_4_f1', 'bright_17_f1']
    for recipe_name in RECIPES:
        subset = [row for row in rows if row['recipe'] == recipe_name]
        aggregate[recipe_name] = {key: dict(mean=statistics.mean(row[key] for row in subset),
                                           sample_sd=statistics.stdev(row[key] for row in subset)) for key in keys}
    result = dict(plan=plan, audit=audit, runs=rows, paired=paired, aggregate=aggregate,
                  test_evaluated=False, real_world_accuracy_unknown=True,
                  note='Three training seeds on one fixed validation set; seed SD is not uncertainty over new scenes or an unbiased final accuracy estimate')
    (ROOT/'outputs'/'baseline_check_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report = ['# 基线稳定性检查结果', '',
              '同一固定划分；seed42/43/44；每组 3 轮分类头＋7 轮 layer4；完整图片等比例缩放留边。',
              '基线为单层头、普通交叉熵；候选只额外使用光照/模糊增强与逆频率类别加权。没有新增网络结构。', '',
              '| 方案 | Seed | 答对 / 375 | 准确率 | Macro-F1 | 塑料召回 | 塑料→玻璃 | 玻璃→塑料 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        label = '基线' if row['recipe'] == 'baseline' else '现有候选'
        report.append(f"| {label} | {row['seed']} | {row['correct']} | {row['accuracy']:.2%} | {row['macro_f1']:.4f} | {row['plastic_recall']:.2%} | {row['plastic_as_glass']} | {row['glass_as_plastic']} |")
    report += ['', '## 三个训练种子的均值 ± 样本标准差', '',
               '| 指标 | 基线 | 现有候选 |', '|---|---:|---:|']
    labels = {'accuracy': '干净准确率', 'macro_f1': '干净 Macro-F1', 'plastic_recall': '塑料召回',
              'glass_recall': '玻璃召回', 'dark_02_f1': '亮度 0.2 Macro-F1', 'blur_4_f1': '模糊 4 Macro-F1', 'bright_17_f1': '亮度 1.7 Macro-F1'}
    for key in keys:
        values = [aggregate[recipe][key] for recipe in RECIPES]
        report.append('| '+labels[key]+' | '+' | '.join(f"{value['mean']:.4f} ± {value['sample_sd']:.4f}" for value in values)+' |')
    report += ['', '此表指标以 0–1 表示；种子标准差只衡量本次训练随机性的变化，不代表新图片、新拍摄场景上的置信区间。', '',
               '## 同图配对变化', '', '| Seed | 仅候选答对 | 仅基线答对 | 候选净增加答对 |', '|---|---:|---:|---:|']
    for row in paired:
        report.append(f"| {row['seed']} | {row['candidate_only_correct']} | {row['baseline_only_correct']} | {row['net_extra_correct']} |")
    baseline_mean = aggregate['baseline']
    candidate_mean = aggregate['candidate']
    indexed = {(row['recipe'], row['seed']): row for row in rows}
    clean_wins = sum(indexed['candidate', seed]['macro_f1'] > indexed['baseline', seed]['macro_f1'] for seed in SEEDS)
    report += ['', '## 本轮判断与下一步', '',
               f"- 干净验证准确率均值从 {baseline_mean['accuracy']['mean']:.2%} 到 {candidate_mean['accuracy']['mean']:.2%}；候选的干净 Macro-F1 在本次 {clean_wins}/3 个种子中更高，提升幅度较小。",
               f"- 强模糊 Macro-F1 均值从 {baseline_mean['blur_4_f1']['mean']:.4f} 到 {candidate_mean['blur_4_f1']['mean']:.4f}；强暗光均值从 {baseline_mean['dark_02_f1']['mean']:.4f} 到 {candidate_mean['dark_02_f1']['mean']:.4f}。已有增强方案对本轮合成干扰更有效。",
               f"- 塑料召回均值从 {baseline_mean['plastic_recall']['mean']:.2%} 到 {candidate_mean['plastic_recall']['mean']:.2%}，玻璃召回从 {baseline_mean['glass_recall']['mean']:.2%} 到 {candidate_mean['glass_recall']['mean']:.2%}；没有观察到这两个易混类别的平均收益。",
               '- 现有基线可作为后续受控实验的起点。下一步梳理相关研究，提出针对干扰失效或易混类别的单一模型改动，并与普通基线、已有增强强对照一起比较及消融。',
               '- 这轮只复核已有训练方法，不计作新的模型创新；实拍泛化仍需单独评估。', '',
               '## 数据与结论边界', '',
               '- 已核对固定清单、训练/验证文件存在、已记录相似组与像素相同图片不跨划分；无法保证发现所有近重复。',
               '- 训练图片全部为 512×384；手机竖图等输入条件的覆盖不足仍是待验证的问题。',
               '- 验证集参与每次选模，结果是探索性验证成绩，不是独立最终测试成绩。',
               '- 没有打开测试图片或用户水瓶照片进行本轮评估，没有把用户反馈照片加入数据划分。',
               '- 没有根据这轮结果自动替换演示默认模型；实拍准确率仍未知。',
               '- 后续先结合干扰条件与易混类别的取舍确定研究目标，再设计模型改动和消融。', '']
    (ROOT/'BASELINE_CHECK_RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
