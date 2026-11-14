"""Bounded two-stage training-setting study; no architecture or split changes."""
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from run_baseline_check import dataset_audit

ROOT = Path(__file__).resolve().parent
BASE = dict(head_lr=.001, backbone_lr=.0001, weight_decay=.0001, finetune_scope='layer4')
CANDIDATES = {
    'lower_lr': dict(BASE, head_lr=.0003, backbone_lr=.00003),
    'weight_decay': dict(BASE, weight_decay=.001),
    'deeper': dict(BASE, finetune_scope='layer3_layer4'),
}
LABELS = {'original': '原训练设置', 'lower_lr': '较低学习率', 'weight_decay': '较强权重衰减', 'deeper': '更深微调范围'}
COMMON = dict(warmup_epochs=3, finetune_epochs=7, batch_size=32, workers=2,
              augmentation='baseline', class_weighting='none', framing='fit', head='linear')
SEEDS = [42, 43, 44]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_name(recipe, seed):
    return f'full_image_control_seed{seed}' if recipe == 'original' else f'baseline_tune_{recipe}_seed{seed}'


def verify_run(name, recipe, seed, manifest_hash):
    out = ROOT/'outputs'/name
    config = read_json(out/'config.json')
    settings = BASE if recipe == 'original' else CANDIDATES[recipe]
    effective = dict(config, head_lr=config.get('head_lr', config['learning_rates']['head']),
                     backbone_lr=config.get('backbone_lr', config['learning_rates']['trainable_backbone']))
    expected = dict(COMMON, **settings, seed=seed, run_name=name, split_manifest_sha256=manifest_hash)
    for key, value in expected.items():
        if effective.get(key) != value:
            raise RuntimeError(f'{name}: {key} differs from the tuning plan')
    summary = read_json(out/'summary.json')
    metrics = read_json(out/'val_metrics.json')
    if summary['test_evaluated'] is not False or metrics['count'] != 375:
        raise RuntimeError(f'{name}: invalid validation record')
    if abs(summary['validation_macro_f1']-metrics['macro_f1']) > 1e-12:
        raise RuntimeError(f'{name}: saved metrics differ')
    return metrics, summary


def train(recipe, seed, manifest_hash):
    name = run_name(recipe, seed)
    out = ROOT/'outputs'/name
    print(f'TUNING TRAIN: {name}', flush=True)
    if not (out/'summary.json').exists():
        if (out/'best_model.pt').exists():
            raise RuntimeError(f'Inspect incomplete run before resuming: {name}')
        args = dict(COMMON, **CANDIDATES[recipe], seed=seed, run_name=name)
        command = [sys.executable, str(ROOT/'train_baseline.py')]
        for key, value in args.items():
            command += ['--'+key.replace('_', '-'), str(value)]
        subprocess.run(command, check=True)
    return verify_run(name, recipe, seed, manifest_hash)


def metrics_row(recipe, seed, manifest_hash):
    name = run_name(recipe, seed)
    metrics, summary = verify_run(name, recipe, seed, manifest_hash)
    out = ROOT/'outputs'/name
    robustness = out/'validation_robustness_extended.csv'
    if not robustness.exists():
        subprocess.run([sys.executable, str(ROOT/'preview_robustness.py'), '--run-name', name, '--include-bright'], check=True)
    with robustness.open(encoding='utf-8', newline='') as stream:
        probes = list(csv.DictReader(stream))
    expected_conditions = ['clean', 'brightness_0.7', 'brightness_0.4', 'brightness_0.2',
                           'blur_1', 'blur_2', 'blur_4', 'brightness_1.3', 'brightness_1.7']
    if [probe['condition'] for probe in probes] != expected_conditions or any(probe['split'] != 'validation' or int(probe['count']) != 375 for probe in probes):
        raise RuntimeError(f'{name}: robustness protocol differs')
    scores = {probe['condition']: float(probe['macro_f1']) for probe in probes}
    if abs(scores['clean']-metrics['macro_f1']) > 1e-12:
        raise RuntimeError(f'{name}: input protocol differs')
    cm = metrics['confusion_matrix']
    return dict(recipe=recipe, seed=seed, run_name=name, best_epoch=summary['best_epoch'],
                accuracy=metrics['accuracy'], macro_f1=metrics['macro_f1'],
                training_validation_seconds=summary['total_training_validation_seconds'],
                peak_gpu_allocated_mb=summary['peak_gpu_allocated_mb'],
                correct=sum(cm[index][index] for index in range(6)),
                plastic_recall=metrics['classification_report']['plastic']['recall'],
                glass_recall=metrics['classification_report']['glass']['recall'],
                plastic_as_glass=cm[4][1], glass_as_plastic=cm[1][4],
                dark_02_f1=scores['brightness_0.2'], blur_4_f1=scores['blur_4'], bright_17_f1=scores['brightness_1.7'])


def main():
    manifest_hash = sha256(ROOT/'data'/'split_manifest.csv')
    reference_hashes = {run_name('original', seed): sha256(ROOT/'outputs'/run_name('original', seed)/'best_model.pt') for seed in SEEDS}
    plan = dict(common=COMMON, original=BASE, candidates=CANDIDATES, screening_seed=42,
                confirmation_seeds=[43, 44], maximum_new_training_runs=5,
                split_manifest_sha256=manifest_hash, reference_checkpoint_sha256=reference_hashes,
                screening_selection='Clean validation Macro-F1 only; retain original if no improvement; exact ties follow candidate insertion order',
                confirmation_decision='Recommend screened winner only if mean clean Macro-F1 increases and at least 2/3 paired seeds improve',
                class_and_robustness_metrics='Report tradeoffs after selection; not used to expand the search',
                test_evaluated=False, user_feedback_photo_used_for_selection=False,
                implementation_sha256={name: sha256(ROOT/name) for name in ['train_baseline.py', 'model_utils.py', 'image_framing.py']},
                compatibility='Default optimizer updates verified against historical AdamW for head/layer4/layer3_layer4')
    plan_path = ROOT/'outputs'/'baseline_tuning_plan.json'
    if plan_path.exists():
        recorded = read_json(plan_path)
        if recorded != plan:
            current_protocol = {key: value for key, value in plan.items() if key != 'implementation_sha256'}
            old_protocol = {key: value for key, value in recorded.items() if key != 'implementation_sha256'}
            selection_path = ROOT/'outputs'/'baseline_tuning_screening.json'
            if current_protocol != old_protocol or not selection_path.exists():
                raise RuntimeError('Existing tuning protocol differs or screening is incomplete')
            old_winner = read_json(selection_path)['screened_winner']
            required = [('original', seed) for seed in SEEDS] + [(recipe, 42) for recipe in CANDIDATES]
            if old_winner != 'original':
                required += [(old_winner, seed) for seed in [43, 44]]
            completed = all((ROOT/'outputs'/run_name(recipe, seed)/filename).exists()
                            for recipe, seed in required
                            for filename in ['summary.json', 'validation_robustness_extended.csv'])
            if not completed:
                raise RuntimeError('Training source changed; cannot resume an incomplete historical tuning study')
            plan = recorded
            print('Regenerating completed tuning from saved results and archived source fingerprints; no new training or evaluation.', flush=True)
    else:
        plan_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    (ROOT/'outputs'/'baseline_tuning_audit.json').write_text(json.dumps(dataset_audit(), indent=2), encoding='utf-8')
    reference_metrics, _ = verify_run(run_name('original', 42), 'original', 42, manifest_hash)
    screening = {'original': reference_metrics['macro_f1']}
    for recipe in CANDIDATES:
        metrics, _ = train(recipe, 42, manifest_hash)
        screening[recipe] = metrics['macro_f1']
    winner = max(screening, key=screening.get)
    selection = dict(screening_macro_f1=screening, screened_winner=winner,
                     selection_metric='clean validation Macro-F1', search_expanded_after_results=False)
    (ROOT/'outputs'/'baseline_tuning_screening.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print('TUNING SCREEN WINNER: '+winner, flush=True)
    if winner != 'original':
        for seed in [43, 44]:
            train(winner, seed, manifest_hash)
    identities = [('original', seed) for seed in SEEDS] + [(recipe, 42) for recipe in CANDIDATES]
    if winner != 'original':
        identities += [(winner, seed) for seed in [43, 44]]
    rows = [metrics_row(recipe, seed, manifest_hash) for recipe, seed in identities]
    keys = ['accuracy', 'macro_f1', 'plastic_recall', 'glass_recall', 'dark_02_f1', 'blur_4_f1', 'bright_17_f1']
    aggregate = {}
    for recipe in ['original'] + ([winner] if winner != 'original' else []):
        subset = [row for row in rows if row['recipe'] == recipe]
        aggregate[recipe] = {key: dict(mean=statistics.mean(row[key] for row in subset),
                                     sample_sd=statistics.stdev(row[key] for row in subset)) for key in keys}
    better_seeds = 0
    if winner != 'original':
        for seed in SEEDS:
            candidate = next(row for row in rows if row['recipe'] == winner and row['seed'] == seed)
            original = next(row for row in rows if row['recipe'] == 'original' and row['seed'] == seed)
            better_seeds += int(candidate['macro_f1'] > original['macro_f1']+1e-12)
    recommend = winner if winner != 'original' and better_seeds >= 2 and aggregate[winner]['macro_f1']['mean'] > aggregate['original']['macro_f1']['mean']+1e-12 else 'original'
    result = dict(plan=plan, screening=selection, rows=rows, aggregate=aggregate,
                  screened_winner_better_seeds=better_seeds, recommended_recipe=recommend,
                  recommendation_scope='Clean validation objective experimental reference only; not deployment promotion',
                  test_evaluated=False, real_world_accuracy_unknown=True,
                  note='Same validation set used for tuning and checkpoint selection; seeds do not make this an independent final evaluation')
    (ROOT/'outputs'/'baseline_tuning_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (ROOT/'outputs'/'baseline_tuning_runs.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    frozen = dict(recipe=recommend, representative_run=run_name(recommend, 42),
                  runs=[run_name(recommend, seed) for seed in SEEDS], seed_policy='42 is fixed representative, not the best seed',
                  training_settings=dict(COMMON, **(BASE if recommend == 'original' else CANDIDATES[recommend])),
                  split_manifest_sha256=manifest_hash, validation_macro_f1=aggregate[recommend]['macro_f1'],
                  role='Additional clean-validation reference; retain original baseline and existing augmentation controls',
                  original_reference_runs=[run_name('original', seed) for seed in SEEDS],
                  deployment_promoted=False,
                  frozen_for_next_model_comparison=True, test_evaluated=False,
                  note='Training-setting baseline, not a novel architecture; same validation data used for selection')
    (ROOT/'outputs'/'baseline_recipe.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    for name, digest in reference_hashes.items():
        if sha256(ROOT/'outputs'/name/'best_model.pt') != digest:
            raise RuntimeError(f'Original reference checkpoint changed: {name}')
    if sha256(ROOT/'data'/'split_manifest.csv') != manifest_hash:
        raise RuntimeError('Fixed split changed during study')
    report = ['# 基线训练设置调优结果', '',
              '固定 TrashNet 划分、完整图片输入、ResNet18 单层分类头、3+7 轮。只调整训练设置。',
              '先在 seed42 筛选三个单因素配置，再按预定规则对筛选胜出者补 seed43/44。验证集同时用于选模与调参，结果不是独立最终测试成绩。', '',
              '| 配置 | Seed | 正确 / 375 | 准确率 | Macro-F1 | 塑料召回 | 玻璃召回 | 塑料→玻璃 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        report.append(f"| {LABELS[row['recipe']]} | {row['seed']} | {row['correct']} | {row['accuracy']:.2%} | {row['macro_f1']:.4f} | {row['plastic_recall']:.2%} | {row['glass_recall']:.2%} | {row['plastic_as_glass']} |")
    report += ['', f"seed42 筛选胜出：{LABELS[winner]}。", f"按三种子确认规则推荐的干净验证对照：{LABELS[recommend]}。",
               '此推荐仅针对预定的干净验证目标；原基线及已有增强对照继续保留，不代表部署升级。', '',
               '## 原设置与筛选胜出者的均值 ± 样本标准差', '',
               '| 指标 | 原设置 | 筛选胜出者 |', '|---|---:|---:|']
    for key in keys:
        recipes = ['original', winner]
        report.append('| '+key+' | '+' | '.join(f"{aggregate[recipe][key]['mean']:.4f} ± {aggregate[recipe][key]['sample_sd']:.4f}" for recipe in recipes)+' |')
    report += ['', '表中指标为 0–1；标准差只反映三个训练种子的变化，不是新拍摄场景上的置信区间。', '',
               '## 训练开销记录', '',
               '| 配置 | Seed | 训练与验证秒数 | 峰值 GPU 分配 MB |', '|---|---:|---:|---:|']
    for row in rows:
        report.append(f"| {LABELS[row['recipe']]} | {row['seed']} | {row['training_validation_seconds']:.1f} | {row['peak_gpu_allocated_mb']:.1f} |")
    report += ['', '轮数相同不代表计算量相同；耗时也受冷启动与硬件状态影响，仅作本次运行记录。', '',
               '## 下一轮固定的基线', '',
               f"训练设置：`{json.dumps(frozen['training_settings'], ensure_ascii=False)}`。", '',
               f"代表权重：`outputs/{frozen['representative_run']}/best_model.pt`。固定用 seed42 代表，不挑最高分种子。", '',
               '机器可读设置为 `outputs/baseline_recipe.json`；未自动替换演示模型。', '',
               '## 结论边界', '',
               '- 该研究属于基线训练调优，没有提出新的模型结构。',
               '- 只比较预先限定的三个配置；根据结果没有追加搜索。',
               '- 旧训练器默认优化器更新已用真实反向传播步骤核对一致，原权重和固定清单指纹未变。',
               '- 学习率与权重衰减变化各单独比较；更深微调保持网络结构，但可训练参数和计算量增加。',
               '- 光照/模糊、逐类指标用于报告取舍；没有根据它们扩大搜索。',
               '- 用户矿泉水瓶未用于本轮调参或选模型。实拍准确率仍未知。',
               '- 没有评估保留测试集，没有把已有方法称为模型创新。', '']
    (ROOT/'BASELINE_TUNING_RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
