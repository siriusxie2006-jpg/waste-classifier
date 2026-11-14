"""One-seed structural pilot with a matched strong control and fixed-fusion ablation."""
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from run_baseline_check import dataset_audit

ROOT = Path(__file__).resolve().parent
RECIPES = {'control': 'resnet18', 'fixed': 'multiscale_fixed', 'gated': 'multiscale_gated'}
CONDITIONS = ['clean', 'brightness_0.7', 'brightness_0.4', 'brightness_0.2',
              'blur_1', 'blur_2', 'blur_4', 'brightness_1.3', 'brightness_1.7']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = read_json(ROOT/'outputs'/'baseline_recipe.json')
    common = dict(frozen['training_settings'], augmentation='both', class_weighting='inverse', seed=42)
    manifest_hash = digest(ROOT/'data'/'split_manifest.csv')
    sources = ['train_baseline.py', 'model_utils.py', 'image_framing.py', 'multiscale_model.py']
    plan = dict(common=common, recipes=RECIPES, conditions=CONDITIONS,
                split_manifest_sha256=manifest_hash, baseline_recipe_sha256=digest(ROOT/'outputs'/'baseline_recipe.json'),
                maximum_training_runs=3, checkpoint_selection='Clean validation Macro-F1, unchanged for all three models',
                primary_metric='Mean of brightness-family mean F1 (5 conditions) and blur-family mean F1 (3 conditions)',
                continuation_rule=dict(min_gain_vs_control=.01, max_clean_f1_drop=.01,
                                       max_plastic_recall_drop=.02, min_gain_vs_fixed=.005),
                single_seed_pilot=True, test_evaluated=False, user_feedback_photo_used_for_selection=False,
                implementation_sha256={name: digest(ROOT/name) for name in sources},
                note='Actual structural change, based on existing multi-scale and gating ideas; not a claim of a novel attention mechanism')
    plan_path = ROOT/'outputs'/'multiscale_pilot_plan.json'
    historical_only = False
    if plan_path.exists():
        recorded = read_json(plan_path)
        if recorded != plan:
            current_protocol = {key: value for key, value in plan.items() if key != 'implementation_sha256'}
            old_protocol = {key: value for key, value in recorded.items() if key != 'implementation_sha256'}
            complete = all((ROOT/'outputs'/f'multiscale_pilot_{recipe}_seed42'/filename).exists()
                           for recipe in RECIPES for filename in ['summary.json', 'validation_robustness_extended.csv'])
            if current_protocol != old_protocol or not complete:
                raise RuntimeError('Pilot source changed and records are incomplete, or the protocol differs')
            plan = recorded
            historical_only = True
            print('Regenerating complete pilot from saved records and archived source; no new training or evaluation.', flush=True)
    else:
        plan_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    (ROOT/'outputs'/'multiscale_pilot_audit.json').write_text(json.dumps(dataset_audit(), indent=2), encoding='utf-8')
    snapshot = ROOT/'sources'/'training_versions'/'multiscale_pilot_20261110'
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in sources:
        destination = snapshot/name
        if historical_only:
            if not destination.exists() or digest(destination) != plan['implementation_sha256'][name]:
                raise RuntimeError('Historical source snapshot missing or invalid: '+name)
        else:
            if destination.exists() and destination.read_bytes() != (ROOT/name).read_bytes():
                raise RuntimeError('Immutable source snapshot differs: '+name)
            destination.write_bytes((ROOT/name).read_bytes())
    (snapshot/'fingerprints.json').write_text(json.dumps(plan['implementation_sha256'], indent=2), encoding='utf-8')
    # Complete all training before inspecting any corruption comparison.
    for recipe, architecture in RECIPES.items():
        name = f'multiscale_pilot_{recipe}_seed42'
        out = ROOT/'outputs'/name
        expected = dict(common, architecture=architecture, run_name=name)
        print('STRUCTURAL PILOT: '+name, flush=True)
        if not (out/'summary.json').exists():
            if (out/'best_model.pt').exists():
                raise RuntimeError('Inspect incomplete run before resuming: '+name)
            command = [sys.executable, str(ROOT/'train_baseline.py')]
            for key, value in expected.items():
                command += ['--'+key.replace('_', '-'), str(value)]
            subprocess.run(command, check=True)
        config = read_json(out/'config.json')
        for key, value in dict(expected, split_manifest_sha256=manifest_hash).items():
            if config.get(key) != value:
                raise RuntimeError(f'{name}: incompatible {key}')
        if config['implementation_sha256'] != plan['implementation_sha256']:
            raise RuntimeError('Training source differs from pilot plan: '+name)
    rows, all_conditions = [], []
    for recipe in RECIPES:
        name = f'multiscale_pilot_{recipe}_seed42'
        out = ROOT/'outputs'/name
        path = out/'validation_robustness_extended.csv'
        if not path.exists():
            subprocess.run([sys.executable, str(ROOT/'preview_robustness.py'), '--run-name', name, '--include-bright'], check=True)
        with path.open(encoding='utf-8', newline='') as stream:
            probes = list(csv.DictReader(stream))
        if [probe['condition'] for probe in probes] != CONDITIONS or any(probe['split'] != 'validation' or int(probe['count']) != 375 for probe in probes):
            raise RuntimeError('Incorrect corruption protocol: '+name)
        scores = {probe['condition']: float(probe['macro_f1']) for probe in probes}
        metrics = read_json(out/'val_metrics.json')
        summary = read_json(out/'summary.json')
        if summary['test_evaluated'] is not False or metrics['count'] != 375 or abs(scores['clean']-metrics['macro_f1']) > 1e-12:
            raise RuntimeError('Validation record mismatch: '+name)
        brightness = statistics.mean(value for key, value in scores.items() if key.startswith('brightness_'))
        blur = statistics.mean(value for key, value in scores.items() if key.startswith('blur_'))
        cm = metrics['confusion_matrix']
        rows.append(dict(recipe=recipe, run_name=name, accuracy=metrics['accuracy'], macro_f1=metrics['macro_f1'],
                         robustness_score=(brightness+blur)/2, brightness_mean_f1=brightness, blur_mean_f1=blur,
                         plastic_recall=metrics['classification_report']['plastic']['recall'],
                         glass_recall=metrics['classification_report']['glass']['recall'],
                         plastic_as_glass=cm[4][1], glass_as_plastic=cm[1][4],
                         dark_02_f1=scores['brightness_0.2'], blur_4_f1=scores['blur_4'],
                         training_validation_seconds=summary['total_training_validation_seconds'],
                         peak_gpu_allocated_mb=summary['peak_gpu_allocated_mb']))
        all_conditions.extend(dict(recipe=recipe, run_name=name, **probe) for probe in probes)
    indexed = {row['recipe']: row for row in rows}
    control, fixed, gated = (indexed[name] for name in ['control', 'fixed', 'gated'])
    rule = plan['continuation_rule']
    checks = dict(robustness_vs_control=gated['robustness_score'] >= control['robustness_score']+rule['min_gain_vs_control'],
                  clean_preserved=gated['macro_f1'] >= control['macro_f1']-rule['max_clean_f1_drop'],
                  plastic_preserved=gated['plastic_recall'] >= control['plastic_recall']-rule['max_plastic_recall_drop'],
                  gate_vs_fixed=gated['robustness_score'] >= fixed['robustness_score']+rule['min_gain_vs_fixed'])
    result = dict(plan=plan, rows=rows, continuation_checks=checks, proceed_to_multiseed_confirmation=all(checks.values()),
                  test_evaluated=False, real_world_accuracy_unknown=True,
                  note='Exploratory seed42 pilot on the validation set; not a final model or novelty proof')
    (ROOT/'outputs'/'multiscale_pilot_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for filename, records in [('multiscale_pilot_runs.csv', rows), ('multiscale_pilot_conditions.csv', all_conditions)]:
        with (ROOT/'outputs'/filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    labels = {'control': '匹配增强强对照', 'fixed': '固定比例双层融合', 'gated': '自适应双层融合'}
    report = ['# 双层特征融合：单种子结构试验', '',
              '三组统一使用固定数据划分、完整图片输入、相同光照/模糊增强与类别加权、相同微调范围和 3+7 轮预算。',
              '该试验改变网络结构，固定融合与自适应融合分别用于检验跨层信息和门控的作用。相关思想已有研究基础，不能直接宣称首创。', '',
              '| 模型 | 干净准确率 | 干净 Macro-F1 | 均衡干扰得分 | 塑料召回 | 玻璃召回 | 强模糊 F1 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        report.append(f"| {labels[row['recipe']]} | {row['accuracy']:.2%} | {row['macro_f1']:.4f} | {row['robustness_score']:.4f} | {row['plastic_recall']:.2%} | {row['glass_recall']:.2%} | {row['blur_4_f1']:.4f} |")
    report += ['', '均衡干扰得分为亮度五条件平均 F1 与模糊三条件平均 F1 的均值，权重在训练前固定；不是通用实拍准确率。', '',
               '## 预定继续条件', '',
               '- 自适应版均衡干扰得分至少比强对照高 0.01。',
               '- 干净 Macro-F1 比强对照下降不超过 0.01。',
               '- 塑料召回比强对照下降不超过 0.02。',
               '- 自适应版均衡干扰得分至少比固定融合高 0.005。', '',
               f"各项检查：`{json.dumps(checks, ensure_ascii=False)}`。", '',
               '结论：'+('满足单种子继续条件，后续需要多随机种子确认，当前尚不能宣称稳定提升。' if all(checks.values()) else '未满足全部继续条件。保留原型与负面结果，不自动部署，也不根据结果追加参数搜索。'), '',
               '## 解释与边界', '',
               '- 多尺度与门控参考已有研究，具体结构、适配与消融由本地实现；未做全面新颖性检索，不能称全新注意力机制。',
               '- 所有训练样本、验证样本、预算与选模指标一致；额外参数和计算量应单独记录。',
               '- 门控权重只是特征混合系数，不是材质概率、物体面积或可靠性保证。',
               '- 多尺度版 Grad-CAM 在 layer3 计算，覆盖两条分支的反向路径；不能与原 layer4 热图作精细位置的直接对比。',
               '- 本轮未使用保留测试集或反馈水瓶挑模型；没有更改当前演示默认。', '']
    (ROOT/'MULTISCALE_PILOT_RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
    if digest(ROOT/'data'/'split_manifest.csv') != manifest_hash:
        raise RuntimeError('Fixed split changed')
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
