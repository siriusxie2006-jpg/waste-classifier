"""Describe paired input interventions without selecting a deployment transform."""
import hashlib
import statistics
from pathlib import Path

import numpy as np
from PIL import Image
from diagnose_domain_inputs import (ROOT, OUT, CONDITIONS, digest, read, read_csv,
                                    save, write_csv, verify)
from domain_probes import probe_frame

LABELS = ['Original', 'Existing pad:\nmean RGB', 'Shrink 75%:\nwhite',
          'Shrink 75%:\nmean RGB', 'Rotate 90°', 'Rotate 180°']
NAMES = {'fixed': '固定融合', 'legacy': '历史增强'}
MATERIALS = ['cardboard', 'glass', 'metal', 'paper', 'plastic']


def select_examples(rows, material, count=3):
    return sorted((row for row in rows if row['class_name'] == material),
                  key=lambda row: hashlib.sha256(row['path'].encode()).hexdigest())[:count]


def main():
    plan = verify()
    results = read(OUT/'results.json')
    if results['plan'] != plan or len(results['rows']) != 72:
        raise RuntimeError('Incomplete diagnosis')
    from train_baseline import plt
    from image_framing import fit_image
    rows = results['rows']
    indexed = {(row['dataset'], row['recipe'], row['seed'], row['condition']): row for row in rows}
    by_run = []
    for row in rows:
        original = indexed[row['dataset'], row['recipe'], row['seed'], 'original_white']
        by_run.append(dict(dataset=row['dataset'], recipe=row['recipe'], seed=row['seed'], condition=row['condition'],
                           accuracy=row['accuracy'], f1=row['macro_f1_five_materials'],
                           f1_change=row['macro_f1_five_materials']-original['macro_f1_five_materials'],
                           agreement=row['prediction_agreement'], lost_correct=row['original_correct_now_wrong'],
                           gained_correct=row['original_wrong_now_correct'], predicted_trash_rate=row['predicted_trash_rate']))
    write_csv(OUT/'paired_runs.csv', by_run)
    aggregate = []
    keys = ['accuracy', 'f1', 'f1_change', 'agreement', 'lost_correct', 'gained_correct', 'predicted_trash_rate']
    for dataset in ['validation5', 'realwaste']:
        for recipe in NAMES:
            for condition in CONDITIONS:
                group = [row for row in by_run if (row['dataset'], row['recipe'], row['condition']) == (dataset, recipe, condition)]
                item = dict(dataset=dataset, recipe=recipe, condition=condition)
                for key in keys:
                    item[key+'_mean'] = statistics.mean(row[key] for row in group)
                    item[key+'_sample_sd'] = statistics.stdev(row[key] for row in group)
                aggregate.append(item)
    write_csv(OUT/'aggregate.csv', aggregate)
    comparisons = []
    for dataset in ['validation5', 'realwaste']:
        for recipe in NAMES:
            for name, changed, reference in [('existing_pad_color', 'existing_mean_pad', 'original_white'),
                                             ('added_border_color', 'shrink75_mean', 'shrink75_white'),
                                             ('half_turn', 'rotate180', 'original_white')]:
                values = [indexed[dataset, recipe, seed, changed]['macro_f1_five_materials']-
                          indexed[dataset, recipe, seed, reference]['macro_f1_five_materials'] for seed in [42,43,44]]
                comparisons.append(dict(dataset=dataset, recipe=recipe, comparison=name,
                                        mean_f1_difference=statistics.mean(values), paired_seed_differences=values))
    save(OUT/'controlled_comparisons.json', comparisons)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    for axis, dataset, title in zip(axes, ['validation5', 'realwaste'], ['TrashNet validation: five materials', 'RealWaste: existing development diagnostics']):
        for recipe, color in [('fixed', '#2878ad'), ('legacy', '#da8b29')]:
            group = [next(row for row in aggregate if (row['dataset'],row['recipe'],row['condition']) == (dataset,recipe,condition)) for condition in CONDITIONS]
            means, sd = np.array([row['f1_mean'] for row in group]), np.array([row['f1_sample_sd'] for row in group])
            axis.plot(range(6), means, marker='o', color=color, label='Fixed fusion' if recipe == 'fixed' else 'Historical augmentation')
            axis.fill_between(range(6), means-sd, means+sd, color=color, alpha=.12)
        axis.set(xticks=range(6), xticklabels=LABELS, ylim=(0, 1), title=title, ylabel='Five-material Macro-F1')
        axis.tick_params(axis='x', labelrotation=18)
        axis.grid(alpha=.2)
        axis.legend(loc='lower left')
    fig.suptitle('Post-hoc paired interventions: full image content retained\nThree training seeds, same images; shaded seed SD is not an independent-scene confidence interval')
    fig.tight_layout()
    fig.savefig(OUT/'interventions.png', dpi=160)
    plt.close(fig)

    features = read_csv(OUT/'coverage_features.csv')
    feature_summary = []
    for group in ['train', 'validation5', 'realwaste']:
        # Use the same five labels for the descriptive comparison.
        subset = [row for row in features if row['group'] == group and row['class_name'] in MATERIALS]
        for key in plan['features']:
            values = np.array([float(row[key]) for row in subset])
            feature_summary.append(dict(group=group, feature=key, count=len(values), mean=float(values.mean()),
                                        q25=float(np.quantile(values,.25)), median=float(np.median(values)),
                                        q75=float(np.quantile(values,.75))))
    write_csv(OUT/'coverage_summary.csv', feature_summary)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for axis, key, title in zip(axes, ['luma_mean','rim_luma_mean','rim_luma_std'],
                                ['Full-image luminance', 'Border-pixel luminance', 'Border-pixel luminance SD']):
        for group, color, name in [('train','#2878ad','TrashNet train, shared labels'),('realwaste','#da8b29','RealWaste')]:
            values = [float(row[key]) for row in features if row['group'] == group and row['class_name'] in MATERIALS]
            axis.hist(values, bins=np.linspace(0,1,31), density=True, histtype='step', linewidth=2, color=color, label=name)
        axis.set(title=title, xlabel='Pixel luminance, 0–1', ylabel='Density')
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Descriptive image coverage; border pixels are not a ground-truth background mask')
    fig.tight_layout()
    fig.savefig(OUT/'coverage.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(6, 6, figsize=(12.5, 13.8))
    examples = []
    for row_index, material in enumerate(MATERIALS+['trash']):
        for group, offset in [('train',0),('realwaste',3)]:
            selected = select_examples([row for row in features if row['group'] == group], material)
            for column in range(3):
                axis = axes[row_index, offset+column]
                axis.set(xticks=[], yticks=[])
                if column < len(selected):
                    row = selected[column]
                    with Image.open(ROOT/row['path']) as source:
                        axis.imshow(fit_image(source.convert('RGB')))
                    examples.append(dict(group=group, class_name=material, path=row['path']))
                else:
                    axis.text(.5,.5,'Not in mapped\nexternal evaluation',ha='center',va='center', transform=axis.transAxes,fontsize=9)
                if offset == 0 and column == 0:
                    axis.set_ylabel(material, fontsize=10)
    fig.suptitle('Fixed examples selected by path hash, without model-score selection\nFull 224x224 model input: TrashNet training (left) | RealWaste (right)', y=.985)
    fig.subplots_adjust(left=.075,right=.99,bottom=.015,top=.94,hspace=.12,wspace=.08)
    fig.savefig(OUT/'dataset_examples.png', dpi=140)
    plt.close(fig)
    write_csv(OUT/'dataset_examples.csv', examples)

    prep = read(OUT/'preparation.json')
    fig, axes = plt.subplots(2, 6, figsize=(13, 5.0))
    for y, dataset in enumerate(['validation5','realwaste']):
        entry = prep['datasets'][dataset]
        records = read_csv(ROOT/entry['manifest'])
        chosen = select_examples(records, 'plastic', 1)[0]
        frame = Image.fromarray(np.load(ROOT/entry['frames'], mmap_mode='r')[int(chosen['index'])])
        box = [int(chosen[key]) for key in ['left','top','right','bottom']]
        for x, condition in enumerate(CONDITIONS):
            axis = axes[y,x]
            axis.imshow(probe_frame(frame, condition, box))
            axis.set(xticks=[],yticks=[])
            if y == 0:
                axis.set_title(LABELS[x], fontsize=9)
            if x == 0:
                axis.set_ylabel('TrashNet val' if y == 0 else 'RealWaste')
    fig.suptitle('Examples of prescribed interventions; all original content retained')
    fig.tight_layout()
    fig.savefig(OUT/'probe_examples.png', dpi=150)
    plt.close(fig)

    report = ['# 跨场景输入诊断结果', '',
              '这是外部结果之后开展的事后开发诊断，不是新的独立测试；六份既有权重、六个预定条件，未训练或替换演示。验证端保持原 375 张输入批次，仅统计五种对应材质的 353 张；外部统计原有 3092 张。所有指标保留六输出。', '']
    for dataset, title in [('validation5','原验证集五类'),('realwaste','RealWaste 开发诊断')]:
        report += ['', '## '+title, '', '| 条件 | 固定融合 F1 | 历史增强 F1 | 固定融合预测一致率 | 历史增强预测一致率 |', '|---|---:|---:|---:|---:|']
        for condition in CONDITIONS:
            selected = {row['recipe']:row for row in aggregate if row['dataset'] == dataset and row['condition'] == condition}
            report.append(f"| {condition} | {selected['fixed']['f1_mean']:.4f} | {selected['legacy']['f1_mean']:.4f} | {selected['fixed']['agreement_mean']:.2%} | {selected['legacy']['agreement_mean']:.2%} |")
    report += ['', '## 隔离边框颜色与半周旋转', '', '| 数据 | 模型 | 对比 | 平均 F1 差值 | 三个配对差值 |', '|---|---|---|---:|---|']
    for row in comparisons:
        report.append(f"| {row['dataset']} | {NAMES[row['recipe']]} | {row['comparison']} | {row['mean_f1_difference']:+.4f} | "+', '.join(f'{value:+.4f}' for value in row['paired_seed_differences'])+' |')
    reproduction = [row['original_reproduction'] for row in rows if row['condition'] == 'original_white']
    maximum = max(row['max_score_difference'] for row in reproduction)
    report += ['', '## 复核与限制', '',
               f'12 组原始输入复核的预测类别均一致，最大分数差 {maximum:.8g}，在预定 1e-5 容差内。正方形外部图的 existing_mean_pad 没有真实像素变化，是对照条件。', '',
               '- existing_pad_color 为已有边框均值色减原白色，内容与尺度相同；added_border_color 为缩小后均值框减白框，新增边框以外内容相同。',
               '- 缩小同时改变内容分辨率、占比及新增边框，不能证明真实物体大小的唯一作用；没有目标框或分割标注。',
               '- rotate90 对原横图还改变白边方向；rotate180 保持对称白边位置。两者均旋转物体及真实背景，不能只归因为物体朝向。',
               '- 背景覆盖只使用画面边缘像素的描述统计，没有自然背景掩码，不能据此证明背景因果。',
               '- 结果不能用来挑选最佳测试时变换后当作独立外部成绩；原数据和预测保留，原正式测试集未打开，演示默认未改。',
               '- 首次只加载五类的尝试在原值分数复核失败后停止；恢复全部验证批次后重新锁定，记录与源码另存，评分范围和容差未变。', '']
    (ROOT/'DOMAIN_DIAGNOSIS_RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
    save(OUT/'analysis.json', dict(results_sha256=digest(OUT/'results.json'),
                                  source_sha256=digest(Path(__file__)), aggregate=aggregate,
                                  controlled_comparisons=comparisons, feature_summary=feature_summary,
                                  original_reproduction_max_score_difference=maximum,
                                  interpretation='Paired post-hoc interventions and descriptive coverage, not a unique domain-gap cause'))
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
