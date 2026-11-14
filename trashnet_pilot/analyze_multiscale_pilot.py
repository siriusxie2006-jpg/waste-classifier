"""Inspect learned mixture weights and measured inference overhead after the pilot."""
import csv
import json
import statistics
import time
from torch.utils.data import DataLoader
from torchvision import transforms as T
from train_baseline import ROOT, CLASSES, TrashDataset, make_transform, evaluate, set_seed, torch, plt
from inference import load_run
from model_utils import set_training_phase
from preview_robustness import Corrupt


def latency(model, inputs, device, count=50):
    model.to(device).eval()
    with torch.inference_mode():
        for _ in range(10):
            model(inputs)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(count):
            model(inputs)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
    return (time.perf_counter()-start)*1000/count


def main():
    set_seed(42)
    result = json.loads((ROOT/'outputs'/'multiscale_pilot_results.json').read_text(encoding='utf-8'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_cpu = torch.randn(1, 3, 224, 224)
    dummy_device = dummy_cpu.to(device)
    metadata = []
    for row in result['rows']:
        model, checkpoint = load_run(row['run_name'])
        set_training_phase(model, checkpoint['config']['finetune_scope'])
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        cpu_ms = latency(model, dummy_cpu, 'cpu')
        device_ms = latency(model, dummy_device, device) if device != 'cpu' else cpu_ms
        metadata.append(dict(recipe=row['recipe'], total_parameters=total, fine_tuning_parameters=trainable,
                             fp32_cpu_batch1_ms=cpu_ms, fp32_device_batch1_ms=device_ms))
        del model
    model, _ = load_run('multiscale_pilot_gated_seed42', device)
    base = list(make_transform(False, framing='fit').transforms)
    conditions = [('clean', None, None), ('brightness_0.7', .7, None), ('brightness_0.4', .4, None),
                  ('brightness_0.2', .2, None), ('blur_1', None, 1), ('blur_2', None, 2),
                  ('blur_4', None, 4), ('brightness_1.3', 1.3, None), ('brightness_1.7', 1.7, None)]
    with (ROOT/'outputs'/'multiscale_pilot_gated_seed42'/'validation_robustness_extended.csv').open(encoding='utf-8', newline='') as stream:
        saved = {row['condition']: float(row['macro_f1']) for row in csv.DictReader(stream)}
    gate_rows, values = [], []
    handle = model.fc.gate[-1].register_forward_hook(lambda _module, _args, output: values.extend(output.sigmoid().detach().cpu().flatten().tolist()))
    try:
        for name, brightness, blur in conditions:
            values.clear()
            transform = T.Compose(base[:-2]+[Corrupt(brightness, blur)]+base[-2:])
            data = DataLoader(TrashDataset('val', transform), batch_size=32, num_workers=0)
            metrics, _ = evaluate(model, data, device)
            if len(values) != 375 or abs(metrics['macro_f1']-saved[name]) > 1e-12:
                raise RuntimeError('Gate probe and saved model evaluation differ')
            by_class = {class_name: statistics.mean(value for value, record in zip(values, data.dataset.records)
                                                   if record['class_name'] == class_name) for class_name in CLASSES}
            gate_rows.append(dict(condition=name, detail_weight_mean=statistics.mean(values),
                                  detail_weight_sd=statistics.pstdev(values), minimum=min(values), maximum=max(values),
                                  per_class_mean=by_class))
    finally:
        handle.remove()
    analysis = dict(parameters_and_latency=metadata, gate_probes=gate_rows, latency_device=device,
                    gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
                    cpu_threads=torch.get_num_threads(), latency_protocol='FP32 eval, batch1 224x224, 10 warmup + 50 timed iterations, synchronized GPU',
                    note='Gate values are mixture coefficients, not calibrated reliability or proof of causal dependence; latency is hardware dependent',
                    test_evaluated=False)
    (ROOT/'outputs'/'multiscale_pilot_analysis.json').write_text(json.dumps(analysis, indent=2), encoding='utf-8')
    with (ROOT/'outputs'/'multiscale_pilot_conditions.csv').open(encoding='utf-8', newline='') as stream:
        curves = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout='constrained')
    for recipe, label in [('control', 'Matched strong control'), ('fixed', 'Fixed fusion'), ('gated', 'Adaptive fusion')]:
        scores = {row['condition']: float(row['macro_f1']) for row in curves if row['recipe'] == recipe}
        axes[0].plot([.2, .4, .7, 1., 1.3, 1.7], [scores[name] for name in ['brightness_0.2', 'brightness_0.4', 'brightness_0.7', 'clean', 'brightness_1.3', 'brightness_1.7']], marker='o', label=label)
        axes[1].plot([0, 1, 2, 4], [scores[name] for name in ['clean', 'blur_1', 'blur_2', 'blur_4']], marker='o', label=label)
    for axis, title, xlabel in zip(axes, ['Brightness changes', 'Gaussian blur'], ['Brightness multiplier', 'Blur radius']):
        axis.set(title=title, xlabel=xlabel, ylabel='Validation Macro-F1', ylim=(0, 1))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('Structural pilot: seed42, same 375 validation images\nSingle-seed exploratory results; not an independent final test', fontsize=11)
    fig.savefig(ROOT/'outputs'/'multiscale_pilot_curves.png', dpi=160)
    plt.close(fig)
    lines = ['# 结构原型的门控与开销检查', '',
             '| 模型 | 总参数 | 微调参数 | CPU 单图 ms | GPU/设备单图 ms |', '|---|---:|---:|---:|---:|']
    for row in metadata:
        lines.append(f"| {row['recipe']} | {row['total_parameters']:,} | {row['fine_tuning_parameters']:,} | {row['fp32_cpu_batch1_ms']:.2f} | {row['fp32_device_batch1_ms']:.2f} |")
    lines += ['', f"设备：{analysis['gpu'] or device}；CPU 线程 {analysis['cpu_threads']}。计时为 FP32、单张 224×224，10 次预热＋50 次同步计时，仅反映本机。", '',
              '| 条件 | 中间层混合系数均值 | 图片间标准差 |', '|---|---:|---:|']
    for row in gate_rows:
        lines.append(f"| {row['condition']} | {row['detail_weight_mean']:.4e} | {row['detail_weight_sd']:.4e} |")
    lines += ['', '系数变化仅是描述性观察，不能证明学到了图像质量、材质或可靠性；不同分支特征的幅值也会影响实际贡献。',
              '门控检查重新计算的验证 F1 与保存的九条件结果逐项一致；所有 hook 已清理，没有使用测试集。', '']
    (ROOT/'MULTISCALE_PILOT_ANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved structural pilot parameter, latency, gate and curve analysis', flush=True)


if __name__ == '__main__':
    main()
