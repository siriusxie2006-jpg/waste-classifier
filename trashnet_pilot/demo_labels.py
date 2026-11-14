"""Readable presentation labels; checkpoint class IDs and run IDs stay intact."""
import math

CLASS_LABELS = {
    'cardboard': '纸板', 'glass': '玻璃', 'metal': '金属',
    'paper': '纸张', 'plastic': '塑料', 'trash': '其他垃圾（数据集杂项类）',
}
RUN_LABELS = {
    'multiscale_pilot_control_seed42': ('结构试验：匹配增强对照', '与新结构使用相同微调范围、输入和训练增强的原网络。'),
    'multiscale_pilot_fixed_seed42': ('结构试验：固定双层融合', '中间层与深层特征按固定比例融合，用作消融。'),
    'multiscale_pilot_gated_seed42': ('结构试验：自适应双层融合', '由网络学习每张图片的两层特征融合权重；尚为研究原型。'),
    'full_image_control_seed42': ('完整图片模型（基础训练）', '训练和识别均完整保留图片，等比例缩放并留白。'),
    'full_image_both_weighted_seed42': ('完整图片模型（光照＋模糊＋类别平衡）', '完整图片训练，结合光照、模糊和类别加权。'),
    'baseline_resnet18_seed42': ('原始模型', '普通裁剪、翻转训练，作为改进前的参考。'),
    'resnet18_light_seed42': ('光照适应训练', '训练时加入亮度和对比度变化。'),
    'resnet18_blur_seed42': ('模糊适应训练', '训练时加入随机模糊。'),
    'resnet18_both_seed42': ('光照＋模糊适应训练', '训练时同时加入光照变化和模糊。'),
    'merged_control_seed42': ('原始模型（复现实验）', '用融合后的代码复现原始训练设置。'),
    'merged_weighted_seed42': ('少数类别平衡训练', '提高少数类别在训练损失中的权重。'),
    'merged_both_weighted_seed42': ('改进模型（光照＋模糊＋类别平衡）', '结合光照变化、模糊和少数类别加权训练。'),
    'merged_mlp_seed42': ('两层分类头实验', '将最后的分类部分改成两层，并加入 Dropout。'),
    'merged_deeper_seed42': ('深层微调实验', '解冻 ResNet18 的后两个模块进行训练。'),
}
for _seed in (43, 44):
    RUN_LABELS[f'multiscale_confirm_control_seed{_seed}'] = (
        f'结构复核：匹配增强对照（{_seed}）', '与固定融合匹配训练设置的重复试验。')
    RUN_LABELS[f'multiscale_confirm_fixed_seed{_seed}'] = (
        f'结构复核：固定双层融合（{_seed}）', '固定结构与参数，检查不同训练种子下的稳定性。')
    RUN_LABELS[f'full_image_control_seed{_seed}'] = (
        f'完整图片基础训练（重复实验 {_seed}）', '与基础训练保持相同设置，用于检查训练随机性的影响。')
    RUN_LABELS[f'full_image_both_weighted_seed{_seed}'] = (
        f'完整图片增强训练（重复实验 {_seed}）', '与光照、模糊及类别加权方案保持相同设置，用于稳定性检查。')
for _seed in (42, 43, 44):
    RUN_LABELS[f'capacity_residual_seed{_seed}'] = (
        f'容量对照：同层残差头（{_seed}）', '不加入中间层特征，总参数与固定融合近似相同。')
    RUN_LABELS[f'coverage_control_seed{_seed}'] = (
        'ResNet18（整图增强）' if _seed == 43 else f'ResNet18（整图增强 · {_seed}）',
        '在基础训练设置上增加整图旋转、缩放和留边颜色变化。')
    RUN_LABELS[f'coverage_fixed_seed{_seed}'] = (
        f'固定融合（整图增强 · {_seed}）', '在相同整图增强设置下评估固定融合结构。')
RUN_LABELS['multiscale_confirm_control_seed43'] = (
    '基础 ResNet18', '完整图片训练，使用光照、模糊和类别加权。')
RUN_LABELS['bottle_6_25_seed43'] = (
    'ResNet18（瓶类样本微调）', '补充塑料瓶和玻璃瓶样本，保留六类输出。普通六分类推荐使用基础 ResNet18。')
for _setting, _label in [('lower_lr', '较低学习率'), ('weight_decay', '较强正则化'), ('deeper', '更深微调范围')]:
    for _seed in (42, 43, 44):
        RUN_LABELS[f'baseline_tune_{_setting}_seed{_seed}'] = (
            f'基础训练调优（{_label}·{_seed}）', '保持网络结构与固定数据划分，单独比较训练设置。')


def run_label(name):
    return RUN_LABELS.get(name, (f'其他实验：{name}', '单独保存的实验模型。'))[0]


def run_description(name):
    return RUN_LABELS.get(name, ('', '单独保存的实验模型。'))[1]


def assess_scores(probabilities, classes, mixed_input=False, min_score=.70, min_margin=.15):
    """Display-only heuristic, not calibrated confidence or unknown-object detection."""
    if len(probabilities) != len(classes) or len(classes) < 2:
        raise ValueError('Scores do not match the class list')
    scores = [float(value) for value in probabilities]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
        raise ValueError('Invalid class scores')
    if not math.isclose(sum(scores), 1., rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError('Class scores must sum to one')
    ranking = sorted(range(len(scores)), key=lambda i: -scores[i])
    first, second = ranking[:2]
    reasons = []
    if mixed_input:
        reasons.append('图片被标记为混合材质或多个物品，单一类别不足以表达它。')
    if scores[first] < min_score:
        reasons.append('第一候选得分较低。')
    if scores[first] - scores[second] < min_margin:
        reasons.append('前两个候选得分接近。')
    return dict(needs_review=bool(reasons), reasons=reasons, predicted_class=classes[first],
                top_score=scores[first], margin=scores[first]-scores[second], ranking=ranking)
