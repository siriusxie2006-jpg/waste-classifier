"""Describe capacity-control class tradeoffs, curves, and inference costs."""
import csv
import hashlib
import json
import statistics
from train_baseline import ROOT, torch, set_seed, plt
from inference import load_run
from model_utils import set_training_phase
from analyze_multiscale_pilot import latency


def main():
    set_seed(42)
    path = ROOT/'outputs'/'capacity_control_results.json'
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    result = json.loads(path.read_text(encoding='utf-8'))
    recipes = ['control', 'fixed', 'capacity', 'legacy']
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    class_rows = []
    for recipe in recipes:
        metrics = [json.loads((ROOT/'outputs'/row['run_name']/'val_metrics.json').read_text(encoding='utf-8'))
                   for row in result['rows'] if row['recipe'] == recipe]
        if len(metrics) != 3:
            raise RuntimeError('Three completed seeds required for each recipe')
        for name in classes:
            supports = {int(item['classification_report'][name]['support']) for item in metrics}
            if len(supports) != 1:
                raise RuntimeError('Evaluation class counts differ')
            recalls = [item['classification_report'][name]['recall'] for item in metrics]
            f1s = [item['classification_report'][name]['f1-score'] for item in metrics]
            class_rows.append(dict(recipe=recipe, class_name=name, support=supports.pop(),
                                   recall_mean=statistics.mean(recalls), recall_seed_sd=statistics.stdev(recalls),
                                   f1_mean=statistics.mean(f1s), f1_seed_sd=statistics.stdev(f1s)))
    with (ROOT/'outputs'/'capacity_control_per_class.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_cpu = torch.randn(1, 3, 224, 224)
    dummy_device = dummy_cpu.to(device)
    profile = []
    for recipe in ['control', 'fixed', 'capacity']:
        row = next(row for row in result['rows'] if row['recipe'] == recipe and row['seed'] == 42)
        model, checkpoint = load_run(row['run_name'])
        set_training_phase(model, checkpoint['config']['finetune_scope'])
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        cpu_ms = latency(model, dummy_cpu, 'cpu')
        device_ms = latency(model, dummy_device, device) if device != 'cpu' else cpu_ms
        profile.append(dict(recipe=recipe, parameters=total, fine_tuning_parameters=trainable,
                            fp32_cpu_batch1_ms=cpu_ms, fp32_device_batch1_ms=device_ms))
        del model, checkpoint
    indexed = {row['recipe']: row for row in profile}
    if indexed['capacity']['parameters']-indexed['fixed']['parameters'] != 128:
        raise RuntimeError('Trained model capacity budget differs from plan')
    with (ROOT/'outputs'/'capacity_control_conditions.csv').open(encoding='utf-8', newline='') as stream:
        curves = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout='constrained')
    conditions = [['brightness_0.2', 'brightness_0.4', 'brightness_0.7', 'clean', 'brightness_1.3', 'brightness_1.7'],
                  ['clean', 'blur_1', 'blur_2', 'blur_4']]
    for recipe, label, style in [('control', 'Matched linear head', '-'), ('fixed', 'Cross-stage fixed fusion', '-'),
                                 ('capacity', 'Same-stage capacity control', '-'), ('legacy', 'Historical layer4 augmentation', '--')]:
        for axis, names, positions in zip(axes, conditions, [[.2, .4, .7, 1, 1.3, 1.7], [0, 1, 2, 4]]):
            groups = [[float(row['macro_f1']) for row in curves if row['recipe'] == recipe and row['condition'] == name] for name in names]
            if any(len(group) != 3 for group in groups):
                raise RuntimeError('Missing curve seed')
            axis.plot(positions, [statistics.mean(group) for group in groups], marker='o', linestyle=style, label=label)
    for axis, title, xlabel in zip(axes, ['Brightness changes', 'Gaussian blur'], ['Brightness multiplier', 'Blur radius']):
        axis.set(title=title, xlabel=xlabel, ylabel='Mean validation Macro-F1', ylim=(0, 1))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('Three training seeds, same validation set\nNear-equal head parameter count; topology and optimization are not identical', fontsize=11)
    fig.savefig(ROOT/'outputs'/'capacity_control_curves.png', dpi=160)
    plt.close(fig)
    analysis = dict(per_class=class_rows, parameter_and_latency=profile, device=device,
                    gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
                    cpu_threads=torch.get_num_threads(), benchmark='FP32 eval, batch1 224x224, 10 warmups and 50 timed iterations; synchronized GPU',
                    test_evaluated=False, note='Hardware-dependent inference-only timings; no end-to-end deployment guarantee')
    (ROOT/'outputs'/'capacity_control_analysis.json').write_text(json.dumps(analysis, indent=2), encoding='utf-8')
    report = ['# 容量对照：逐类分析与实际开销', '',
              '| 类别 | 固定融合召回 | 容量对照召回 | 固定融合 F1 | 容量对照 F1 |', '|---|---:|---:|---:|---:|']
    for name in classes:
        fixed = next(row for row in class_rows if row['recipe'] == 'fixed' and row['class_name'] == name)
        capacity = next(row for row in class_rows if row['recipe'] == 'capacity' and row['class_name'] == name)
        report.append(f"| {name} | {fixed['recall_mean']:.2%} | {capacity['recall_mean']:.2%} | {fixed['f1_mean']:.4f} | {capacity['f1_mean']:.4f} |")
    report += ['', '四组模型的完整逐类统计和种子标准差见 CSV；三次训练使用相同图片，并非独立新样本。', '',
               '| 模型 | 总参数 | 微调参数 | CPU 单图 ms | GPU/设备单图 ms |', '|---|---:|---:|---:|---:|']
    for row in profile:
        report.append(f"| {row['recipe']} | {row['parameters']:,} | {row['fine_tuning_parameters']:,} | {row['fp32_cpu_batch1_ms']:.2f} | {row['fp32_device_batch1_ms']:.2f} |")
    report += ['', f"本机设备：{analysis['gpu'] or device}，CPU 线程 {analysis['cpu_threads']}。FP32 单张 224×224，预热 10 次、计时 50 次；GPU 同步。只测模型推理，不含上传和预处理。", '',
               '参数量相近不代表计算顺序、特征来源、函数空间或优化难度相同，不能把结果扩大解释成唯一机制证明。', '']
    (ROOT/'CAPACITY_CONTROL_ANALYSIS.md').write_text('\n'.join(report), encoding='utf-8')
    if hashlib.sha256(path.read_bytes()).hexdigest() != fingerprint:
        raise RuntimeError('Selection results changed during analysis')
    print('Saved capacity-control class analysis, parameter checks, latency and curves', flush=True)


if __name__ == '__main__':
    main()
