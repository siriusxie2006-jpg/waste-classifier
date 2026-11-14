"""Build a portable walkthrough notebook and summary after the pilot finishes."""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    run = ROOT/'outputs'/'baseline_resnet18_seed42'
    summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    audit = json.loads((ROOT/'outputs'/'data_audit.json').read_text(encoding='utf-8'))
    metrics = json.loads((run/'val_metrics.json').read_text(encoding='utf-8'))
    cells = []

    def md(text):
        cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))

    def code(text):
        cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.splitlines(True)))

    md('# TrashNet / ResNet18 基准试跑\n\n'
       '课程项目未限定软件。本试验用 Python + PyTorch，加载 ImageNet 预训练权重，再训练垃圾分类头并微调 layer4。\n\n'
       '**测试集尚未评估。下面展示的是用于选模型的验证集结果，不是最终泛化成绩。**')
    md('## 本次真实试跑结果\n\n'
       f"- 原始图片 {audit['source_images']}；排除标签冲突的 6 张后使用 {audit['total_images']} 张。\n"
       '- 正式划分：训练 1766、验证 375、测试 380。\n'
       f"- 验证准确率：{summary['validation_accuracy']:.2%}；Macro-F1：{summary['validation_macro_f1']:.4f}。\n"
       f"- 完成 {summary['epochs_completed']} 轮；训练与逐轮验证用时 {summary['total_training_validation_seconds']/60:.2f} 分钟。\n"
       f"- 最佳模型出现在第 {summary['best_epoch']} 轮。\n"
       '- 耗时不含首次 PyTorch 安装、数据和预训练权重下载，也不代表所有电脑的耗时。')
    code("from pathlib import Path\nimport json, csv, sys, subprocess\nroot = Path.cwd()\nif not (root / 'prepare_data.py').exists():\n    root = root / 'trashnet_pilot'\nassert (root / 'prepare_data.py').exists(), 'Open this notebook inside the project folder'\nprint('Python:', sys.executable)\nprint('Project:', root)\n")
    md('## 1. 数据准备\n\n如果分享包没有原始图片，取消下一格注释即可重新下载。固定清单已包含，相同参数会复现相同划分。')
    code("# subprocess.run([sys.executable, str(root/'prepare_data.py')], check=True)\nwith (root/'outputs/split_counts.csv').open(encoding='utf-8') as f:\n    print(f.read())\n")
    code("from IPython.display import display, Image\ndisplay(Image(filename=str(root/'outputs/training_samples.png')))\n")
    md('## 2. 基准训练\n\n官方 ResNet18 预训练权重 → 六分类层 → 分类头训练 3 轮 → layer4 微调 7 轮。\n\n'
       '使用已安装 torch 的解释器作为 notebook 内核。已有模型不会被覆盖；重新训练要使用不同 run-name。')
    code("# subprocess.run([sys.executable, str(root/'train_baseline.py'),\n#                 '--run-name', 'my_baseline_seed42'], check=True)\n")
    md('## 3. 查看已完成结果')
    code("summary = json.loads((root/'outputs/baseline_resnet18_seed42/summary.json').read_text())\nprint(json.dumps(summary, indent=2))\ndisplay(Image(filename=str(root/'outputs/baseline_resnet18_seed42/learning_curves.png')))\ndisplay(Image(filename=str(root/'outputs/baseline_resnet18_seed42/val_confusion_matrix.png')))\n")
    md('## 4. 改进入口（尚未训练）\n\n保持 ResNet18、数据划分和训练轮数不变，仅改变增强。\n\n'
       '- `--augmentation light`：额外亮度/对比度增强。\n'
       '- `--augmentation blur`：额外高斯模糊增强。\n'
       '- `--augmentation both`：两者结合。\n\n'
       '先用验证集选择强度，固定全部方案后才最终评估测试集。不要用测试集调参。')
    code("# subprocess.run([sys.executable, str(root/'train_baseline.py'),\n#                 '--augmentation', 'both', '--run-name', 'resnet18_both_seed42'], check=True)\n")
    if (run/'validation_robustness_preview.png').exists():
        md('## 5. 可选：验证集上的干扰预览\n\n仅为验证集探索结果，不能当作最终测试结论。')
        code("display(Image(filename=str(root/'outputs/baseline_resnet18_seed42/validation_robustness_preview.png')))\n")
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3 (select project environment)', language='python', name='python3'), language_info=dict(name='python',version='3.13')), nbformat=4, nbformat_minor=5)
    for i, c in enumerate(cells):
        c['id'] = f'pilot-cell-{i:02d}'
    (ROOT/'pilot_walkthrough.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = [
        '# TrashNet 基准试跑结果', '',
        '本次只完成数据准备与基准模型，改进模型尚未训练。', '',
        '## 数据', '',
        '原始 2527 张；3 对完全相同图片有矛盾标签，排除全部 6 张后使用 2521 张。',
        '正式划分：train 1766 / val 375 / test 380。相似图片候选分组不跨集合。',
        '所有审计记录和固定划分均保留；相似度审计不能证明不存在其他近重复。', '',
        '## 训练', '',
        'ResNet18 ImageNet 预训练；六分类头；3 轮训练分类头，7 轮微调 layer4。',
        f"设备：{json.loads((run/'config.json').read_text())['gpu']}。",
        f"实际完成 {summary['epochs_completed']} 轮；最佳第 {summary['best_epoch']} 轮。",
        f"训练＋逐轮验证耗时：{summary['total_training_validation_seconds']/60:.2f} 分钟（不含环境安装和下载）。",
        f"平均每轮：{summary['mean_epoch_seconds']:.1f} 秒。",
        f"峰值 PyTorch GPU allocated：{summary['peak_gpu_allocated_mb']:.1f} MiB（不是全部显存占用）。", '',
        '## 验证结果', '',
        f"验证准确率：{summary['validation_accuracy']:.2%}；Macro-F1：{summary['validation_macro_f1']:.4f}。",
        '验证集参与选择最佳模型；这些分数不是独立最终测试成绩。测试集尚未评估。', '',
        '| 类别 | Precision | Recall | F1 | 验证样本数 |', '|---|---:|---:|---:|---:|'
    ]
    for c in audit['classes']:
        m = metrics['classification_report'][c]
        lines.append(f"| {c} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1-score']:.3f} | {m['support']:.0f} |")
    if (run/'validation_robustness_preview.csv').exists():
        lines += ['', '## 验证集干扰预览', '', '| 条件 | 准确率 | Macro-F1 |', '|---|---:|---:|']
        import csv
        with (run/'validation_robustness_preview.csv').open() as f:
            for r in csv.DictReader(f):
                lines.append(f"| {r['condition']} | {float(r['accuracy']):.2%} | {float(r['macro_f1']):.4f} |")
        lines += ['', '干扰在 resize/crop 后、归一化前施加。此表仅用于了解问题，不能替代最终测试实验。']
    lines += ['', '## 工作量判断', '',
              '增强代码入口已实现，改进无需改变网络结构；主要工作在于选择合理强度、保持公平对照、多种子重复和错误分析。',
              '四组单种子实验的耗时可按本次每轮速度粗估，但增强、温度和其他程序会影响速度。正式实验应统一训练预算。',
              '首次 PyTorch CUDA 环境下载较大，不能把安装耗时当成每次训练耗时。', '',
              '## 复现', '', '见 README.md；分享包包含模型和清单，不含图片或 ImageNet 缓存，脚本可重新下载。']
    (ROOT/'PILOT_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    target = ROOT.parent/'trashnet_pilot_bundle.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for path in sorted(ROOT.rglob('*')):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if any(part in ['__pycache__', 'cache', 'downloads', 'source'] for part in rel.parts):
                continue
            z.write(path, Path('trashnet_pilot')/rel)
    print('BUNDLE', target, 'BYTES', target.stat().st_size)


if __name__ == '__main__':
    main()
