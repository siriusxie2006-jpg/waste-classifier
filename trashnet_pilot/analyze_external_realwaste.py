"""Summarize the locked external check, including unfavorable results."""
import csv
import json
import statistics
from pathlib import Path

from external_realwaste import ROOT, OUT, DATA, digest, read, save, verify_plan, write_csv

NAMES = {'control': '匹配训练的原模型', 'fixed': '跨层固定融合',
         'capacity': '同层容量对照', 'legacy': '历史增强方案'}
ENGLISH = ['Matched original', 'Fixed fusion', 'Capacity control', 'Historical augmentation']
RECIPES = list(NAMES)
MATERIALS = ['cardboard', 'glass', 'metal', 'paper', 'plastic']


def main():
    plan = verify_plan()
    results_path = OUT/'results.json'
    result = read(results_path)
    if result['plan'] != plan:
        raise RuntimeError('Results and frozen plan differ')
    for row in result['rows']:
        path = OUT/row['run_name']/'predictions.csv'
        if digest(path) != row['predictions_sha256']:
            raise RuntimeError('External prediction table changed')
    import numpy as np
    from train_baseline import plt
    from PIL import Image

    classes = []
    for recipe in RECIPES:
        group = [row for row in result['rows'] if row['recipe'] == recipe]
        for material in MATERIALS:
            records = [row['classification_report'][material] for row in group]
            item = dict(recipe=recipe, class_name=material, support=records[0]['support'])
            for key in ['precision', 'recall', 'f1-score']:
                item[key+'_mean'] = statistics.mean(record[key] for record in records)
                item[key+'_sample_sd'] = statistics.stdev(record[key] for record in records)
            classes.append(item)
    write_csv(OUT/'per_class.csv', classes)
    run_rows = [dict(recipe=row['recipe'], seed=row['seed'], accuracy=row['accuracy'],
                     macro_f1_five_materials=row['macro_f1_five_materials'],
                     plastic_recall=row['classification_report']['plastic']['recall'],
                     glass_recall=row['classification_report']['glass']['recall'],
                     unflagged_error_count=row['unflagged_error_count'],
                     unflagged_count=row['unflagged_count'],
                     predicted_trash_count=row['predicted_trash_count']) for row in result['rows']]
    write_csv(OUT/'runs.csv', run_rows)
    write_csv(OUT/'paired.csv', result['paired'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    colors = ['#2878ad', '#ec8b26', '#3c9d5d', '#b85e66']
    for axis, key, title in zip(axes, ['accuracy', 'macro_f1_five_materials'], ['Accuracy', 'Five-material Macro-F1']):
        for index, recipe in enumerate(RECIPES):
            values = [row[key] for row in result['rows'] if row['recipe'] == recipe]
            axis.bar(index, statistics.mean(values), color=colors[index], alpha=.8, width=.62)
            axis.scatter([index-.09, index, index+.09], values, color='black', marker='o', s=20, zorder=3)
        axis.set(xticks=range(4), xticklabels=ENGLISH, ylim=(0, 1), title=title)
        axis.tick_params(axis='x', labelrotation=18)
        axis.grid(axis='y', alpha=.2)
    fig.suptitle('RealWaste external transfer: same images, three frozen training seeds\nAll six outputs retained; no external fitting')
    fig.tight_layout()
    fig.savefig(OUT/'comparison.png', dpi=160)
    plt.close(fig)

    fixed = [row for row in result['rows'] if row['recipe'] == 'fixed']
    cm = np.sum([row['confusion_matrix'] for row in fixed], axis=0)[:5]
    percentages = cm/cm.sum(1, keepdims=True)*100
    fig, axis = plt.subplots(figsize=(8.8, 5.6))
    display = axis.imshow(percentages, vmin=0, vmax=100, cmap='Blues')
    for y in range(5):
        for x in range(6):
            axis.text(x, y, f'{percentages[y,x]:.1f}%', ha='center', va='center', color='white' if percentages[y,x]>50 else 'black')
    axis.set(xticks=range(6), yticks=range(5), xticklabels=MATERIALS+['trash'], yticklabels=MATERIALS,
             xlabel='Predicted class', ylabel='Original RealWaste material label',
             title='Fixed fusion: mean class prediction percentages\nThree training seeds, same external images')
    fig.colorbar(display, ax=axis, label='Percent within true class')
    fig.tight_layout()
    fig.savefig(OUT/'fixed_confusion.png', dpi=160)
    plt.close(fig)

    representative = next(row for row in fixed if row['seed'] == 42)
    with (OUT/representative['run_name']/'predictions.csv').open(encoding='utf-8', newline='') as stream:
        predictions = list(csv.DictReader(stream))
    examples = []
    for material in MATERIALS:
        examples.extend([row for row in predictions if row['true_class'] == material and row['predicted_class'] != material][:2])
    if examples:
        fig, axes = plt.subplots(2, 5, figsize=(15, 8.0))
        for axis in axes.flat:
            axis.axis('off')
        for axis, row in zip(axes.flat, examples):
            with Image.open(ROOT/row['path']) as image:
                axis.imshow(image.convert('RGB'))
            axis.set_title(f"{row['true_class']} -> {row['predicted_class']}\nscore {float(row['top_score']):.1%}", fontsize=10)
        fig.suptitle('Fixed fusion seed42: first two errors per material in path order\nSelected examples, not a representative accuracy sample; original material labels')
        fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.86, hspace=.35, wspace=.08)
        fig.savefig(OUT/'failure_examples.png', dpi=140)
        plt.close(fig)
        write_csv(OUT/'failure_examples.csv', examples)

    report = ['# RealWaste 外部场景评估', '',
              f"使用原作者固定版本的五类图片，审计前 3092 张，审计后 {result['audit']['retained']} 张。没有外部训练、种子选择或阈值调整。", '',
              '| 模型 | 准确率 | 五类 Macro-F1 ± 种子标准差 | 塑料召回 | 玻璃召回 |', '|---|---:|---:|---:|---:|']
    for recipe in RECIPES:
        aggregate = result['aggregate'][recipe]
        by_class = {row['class_name']: row for row in classes if row['recipe'] == recipe}
        score = aggregate['macro_f1_five_materials']
        report.append(f"| {NAMES[recipe]} | {aggregate['accuracy']['mean']:.2%} | {score['mean']:.4f} ± {score['sample_sd']:.4f} | {by_class['plastic']['recall_mean']:.2%} | {by_class['glass']['recall_mean']:.2%} |")
    report += ['', '## 全部材质 F1', '', '| 材质 | 原模型 | 固定融合 | 容量对照 | 历史增强 |', '|---|---:|---:|---:|---:|']
    indexed = {(row['recipe'], row['class_name']): row for row in classes}
    for material in MATERIALS:
        report.append('| '+material+' | '+' | '.join(f"{indexed[recipe,material]['f1-score_mean']:.4f}" for recipe in RECIPES)+' |')
    report += ['', '## 固定融合的配对五类 F1 差值', '', '| 种子 | 相对原模型 | 相对容量对照 | 相对历史增强 |', '|---|---:|---:|---:|']
    for row in result['paired']:
        report.append(f"| {row['seed']} | {row['fixed_minus_control']:+.4f} | {row['fixed_minus_capacity']:+.4f} | {row['fixed_minus_legacy']:+.4f} |")
    report += ['', '## 六输出与误判提示', '',
               '主要 F1 仅平均五个真实材质类别，但 argmax 始终保留全部六个输出；判成 trash 的图片计错。分类报告中的六标签 macro avg 包含无真实支持的 trash，不作为本轮主要指标。', '',
               '| 模型 | 判为 trash 的比例 | 未触发低分复核的图片中，错误比例 |', '|---|---:|---:|']
    for recipe in RECIPES:
        group = result['aggregate'][recipe]
        report.append(f"| {NAMES[recipe]} | {group['predicted_trash_rate']['mean']:.2%} | {group['unflagged_error_rate']['mean']:.2%} |")
    report += ['', '低分规则沿用得分低于 0.70 或前两名差低于 0.15，未做校准。该表错误比例的分母仅为未触发提示的图片，各模型数量不同，不是可靠性保证。', '',
               '## 数据与解释边界', '',
               '- 五类数量与排除原因见 `data/external_realwaste/audit.json`；整图输入、ImageNet 归一化、12 份权重和评估实现由预先保存的 JSON 计划约束。',
               '- 只对相同解码像素排除原数据重合与内部重复；没有保证发现所有近重复、同一物品或场景相关性，不报告独立场景置信区间。',
               '- 本轮是另一处垃圾处理环境下五种材质的迁移诊断，不是手机实拍准确率、完整六类测试或未知物体检测。保留来源标签，可能有混合、重叠和污损；没有添加检测或分割。',
               '- 原验证指标涉及六个真实类别，此外部主要 F1 涉及五个真实类别，两者不能直接相减当作同口径泛化下降。',
               '- 三次训练使用同一外部图片，标准差只描述训练种子波动；历史增强微调范围不同，属于性能背景。',
               '- 本轮不依据外部分数更新模型或演示。TrashNet 保留测试集未评估；如后续据此调整方法，本数据应视为开发诊断材料，不再当作新的最终独立测试。', '',
               '来源：[UCI](https://archive.ics.uci.edu/dataset/908/realwaste)、[原作者仓库](https://github.com/sam-single/realwaste)、[原论文](https://doi.org/10.3390/info14120633)。许可差异与取得方式见 `EXTERNAL_REALWASTE_PLAN.md`。', '']
    (ROOT/'EXTERNAL_REALWASTE_RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
    save(OUT/'analysis.json', dict(results_sha256=digest(results_path),
                                  analysis_source_sha256=digest(Path(__file__)), per_class=classes,
                                  failure_example_rule='First two errors per material, path order, preselected fixed seed42',
                                  examples_count=len(examples)))
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
