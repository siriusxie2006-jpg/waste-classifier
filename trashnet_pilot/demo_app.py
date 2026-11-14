"""Single-model image workspace; optional comparison and automatic Grad-CAM."""
import csv
import hashlib
import html
import io
import json
import threading
import streamlit as st
from PIL import Image, ImageOps
from train_baseline import ROOT, CLASSES
from inference import load_run, predict_image, prepare_image, crop_image, gradcam_overlay
from crop_widget import manual_crop
from demo_labels import CLASS_LABELS, run_label, run_description, assess_scores

st.set_page_config(page_title='垃圾图像识别', page_icon='♻️', layout='wide')
st.markdown('<style>' + (ROOT/'demo_style.css').read_text(encoding='utf-8') + '</style>', unsafe_allow_html=True)
st.markdown('<div class="app-kicker">WASTE CLASSIFIER</div>', unsafe_allow_html=True)
st.title('垃圾图像识别')
st.markdown('<p class="app-intro">上传照片，调整光照或模糊，直接查看识别结果。</p>', unsafe_allow_html=True)
runs = sorted(path.parent.name for path in (ROOT/'outputs').glob('*/best_model.pt')
              if (path.parent/'summary.json').exists())
if not runs:
    st.info('还没有可用模型，请先完成训练。')
    st.stop()
framing_by_run = {name: json.loads((ROOT/'outputs'/name/'config.json').read_text(encoding='utf-8')).get('framing', 'center')
                  for name in runs}
full_image_runs = [name for name in runs if framing_by_run[name] == 'fit']
baseline_pair = ['multiscale_confirm_control_seed43', 'coverage_control_seed43']
legacy_pair = ['full_image_both_weighted_seed42', 'full_image_control_seed42']
bottle_run = 'bottle_6_25_seed43'
if all(name in full_image_runs for name in baseline_pair):
    demo_runs = baseline_pair
else:
    demo_runs = [name for name in legacy_pair if name in full_image_runs] or full_image_runs or runs
if bottle_run in full_image_runs and baseline_pair[0] in demo_runs and bottle_run not in demo_runs:
    demo_runs = [*demo_runs, bottle_run]
default_run = next((name for name in ('merged_both_weighted_seed42', 'resnet18_both_seed42') if name in demo_runs), demo_runs[0])
SHORT_LABELS = {**CLASS_LABELS, 'trash': '其他垃圾'}


@st.cache_resource
def cached_run(name, modified_ns):
    model, checkpoint = load_run(name)
    return model, checkpoint, threading.Lock()


@st.cache_data(show_spinner=False, max_entries=32)
def prediction_for(name, modified_ns, image_bytes, brightness, blur, framing, crop_box=None):
    with Image.open(io.BytesIO(image_bytes)) as raw:
        image = crop_image(ImageOps.exif_transpose(raw).convert('RGB'), crop_box)
    model, _, lock = cached_run(name, modified_ns)
    with lock:
        prediction, _ = predict_image(model, image, brightness=brightness, blur=blur, framing=framing)
    return prediction


@st.cache_data(show_spinner=False, max_entries=16)
def explanation_for(name, modified_ns, image_bytes, brightness, blur, framing, crop_box=None):
    prediction = prediction_for(name, modified_ns, image_bytes, brightness, blur, framing, crop_box)
    with Image.open(io.BytesIO(image_bytes)) as raw:
        prepared = prepare_image(crop_image(ImageOps.exif_transpose(raw).convert('RGB'), crop_box), brightness, blur, framing=framing)
    model, _, lock = cached_run(name, modified_ns)
    with lock:
        # Explain the model's own prediction. The user never supplies a class.
        return gradcam_overlay(model, prepared, class_index=CLASSES.index(prediction['predicted_class']))


@st.cache_data(show_spinner=False)
def example_bytes():
    with (ROOT/'data'/'split_manifest.csv').open(encoding='utf-8', newline='') as stream:
        record = next(row for row in csv.DictReader(stream)
                      if row['split'] == 'train' and row['class_name'] == 'plastic')
    return (ROOT/record['path']).read_bytes()


def reset_adjustments():
    st.session_state['brightness'] = 1.
    st.session_state['blur'] = 0.
    st.session_state['crop_box'] = None
    st.session_state['crop_reset_token'] = st.session_state.get('crop_reset_token', 0) + 1


def toggle_example():
    st.session_state['example_active'] = not st.session_state.get('example_active', False)


def short_label(name):
    return SHORT_LABELS[name]


image_column, result_column = st.columns([1.2, 1], gap='large')
with result_column:
    with st.container(key='result_workspace'):
        title_column, settings_column = st.columns([2.5, 1])
        title_column.subheader('识别结果')
        with settings_column.popover('模型设置'):
            selected_run = st.selectbox('识别模型', demo_runs, index=demo_runs.index(default_run),
                                       format_func=run_label, key='selected_run_full_image')
            st.caption(run_description(selected_run))
            mixed_input = st.checkbox('待识别物品包含混合材质或多个物品', key='mixed_input',
                                      help='这是可选的输入标记。拍摄用白纸背景不算另一件待识别物品。')
        selected_framing = framing_by_run[selected_run]
        comparison_run = next((name for name in demo_runs if name != selected_run and framing_by_run[name] == selected_framing), None)
        st.caption('当前模型 · ' + run_label(selected_run))
        result_area = st.container()
        toggle_left, toggle_right = st.columns(2)
        show_heat = toggle_left.toggle('显示关注区域', key='show_heat')
        compare = toggle_right.toggle('对比模型', key='compare_baseline', disabled=comparison_run is None)
        heat_area = st.container()
        comparison_area = st.container()

with image_column:
    with st.container(key='image_workspace'):
        st.subheader('图片')
        uploaded = st.file_uploader('上传图片', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed', key='upload')
        image_bytes = uploaded.getvalue() if uploaded is not None else (example_bytes() if st.session_state.get('example_active', False) else None)
        image_id = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
        if st.session_state.get('crop_image_id') != image_id:
            st.session_state.update(crop_image_id=image_id, crop_box=None, crop_last_event=None, crop_reset_token=0)
        crop_box = st.session_state.get('crop_box')
        image = None
        if image_bytes:
            try:
                with Image.open(io.BytesIO(image_bytes)) as raw:
                    image = ImageOps.exif_transpose(raw).convert('RGB')
            except (OSError, ValueError):
                st.error('图片无法读取，请换一张 JPG 或 PNG 图片。')
        tool_example, tool_reset, tool_original = st.columns([1.2, 1., 1.])
        tool_example.button('清除示例' if st.session_state.get('example_active', False) else '试用示例图片',
                            on_click=toggle_example, disabled=uploaded is not None, key='use_example')
        tool_reset.button('恢复原图', on_click=reset_adjustments, disabled=image is None, key='reset_image')
        with tool_original.popover('裁剪图片', disabled=image is None):
            if image is not None:
                payload = manual_crop(image, image_id, crop_box, st.session_state.get('crop_reset_token', 0))
                if isinstance(payload, dict) and payload.get('image_id') == image_id and payload.get('event_id') != st.session_state.get('crop_last_event'):
                    try:
                        selected_box = payload.get('box')
                        crop_image(image, selected_box)  # Validate before storing the selection.
                        crop_box = tuple(selected_box) if selected_box is not None else None
                        st.session_state['crop_box'] = crop_box
                        st.session_state['crop_last_event'] = payload.get('event_id')
                        st.rerun()  # Send the confirmed region back to the crop widget as well.
                    except (ValueError, TypeError):
                        st.error('裁剪区域无效，请重新框选。')
        light_column, blur_column = st.columns(2)
        st.session_state.setdefault('brightness', 1.)
        st.session_state.setdefault('blur', 0.)
        brightness = light_column.slider('亮度', min_value=.2, max_value=1.7, step=.1, key='brightness', disabled=image is None)
        blur = blur_column.slider('模糊', min_value=0., max_value=4., step=.5, key='blur', disabled=image is None)
        if image is not None:
            prepared = prepare_image(crop_image(image, crop_box), brightness, blur, framing=selected_framing)
            with st.container(key='processed_image'):
                st.image(prepared, width='stretch')
            if selected_framing == 'fit':
                st.caption('你选择的裁剪区域 · 等比例缩放' if crop_box is not None else '完整图片 · 等比例缩放并留白，不自动裁掉边缘')
            else:
                st.caption('中心裁剪后的模型输入')
        else:
            st.markdown('<div class="image-empty"><strong>在这里查看图片</strong><span>上传 JPG / PNG，或点击“试用示例图片”</span></div>', unsafe_allow_html=True)

if image is not None:
    try:
        stamp = (ROOT/'outputs'/selected_run/'best_model.pt').stat().st_mtime_ns
        with result_area:
            with st.spinner('正在识别…'):
                prediction = prediction_for(selected_run, stamp, image_bytes, brightness, blur, selected_framing, crop_box)
            decision = assess_scores(prediction['probabilities'], CLASSES, mixed_input=mixed_input)
            predicted = decision['predicted_class']
            label = html.escape(short_label(predicted))
            status = '第一候选 · 需要人工确认' if decision['needs_review'] else '当前更倾向于'
            reason = ' '.join(decision['reasons']) if decision['needs_review'] else '模型得分表示倾向，不保证判断正确。'
            tone = ' review' if decision['needs_review'] else ''
            st.markdown(f'<div class="prediction-card{tone}"><div class="prediction-eyebrow">{status}</div><div class="prediction-line"><strong>{label}</strong><span>{decision["top_score"]:.1%}</span></div><p>{html.escape(reason)}</p></div>', unsafe_allow_html=True)
            st.caption('候选类别')
            rows = []
            for index in decision['ranking'][:3]:
                score = prediction['probabilities'][index]
                rows.append(f'<div class="score-row"><span>{html.escape(short_label(CLASSES[index]))}</span><div class="score-track"><div class="score-fill" style="width:{score*100:.3f}%"></div></div><span class="score-number">{score:.1%}</span></div>')
            st.markdown(''.join(rows), unsafe_allow_html=True)
            if predicted == 'trash':
                st.caption('“其他垃圾”是训练集中的杂项类，不是“无法识别”。')
            with st.expander('全部类别得分'):
                st.dataframe([{'类别': short_label(name), '模型得分': f'{prediction["probabilities"][i]:.1%}'}
                              for i, name in enumerate(CLASSES)], hide_index=True)
        if show_heat:
            with heat_area:
                with st.spinner('生成关注区域…'):
                    overlay = explanation_for(selected_run, stamp, image_bytes, brightness, blur, selected_framing, crop_box)
                st.image(overlay, width=260, caption='对“' + short_label(predicted) + '”这一判断的贡献区域')
                st.caption('自动解释本次预测；暖色表示正向贡献，不是材质边界。')
        if compare and comparison_run is not None:
            with comparison_area:
                comparison_stamp = (ROOT/'outputs'/comparison_run/'best_model.pt').stat().st_mtime_ns
                reference = prediction_for(comparison_run, comparison_stamp, image_bytes, brightness, blur, selected_framing, crop_box)
                reference_label = short_label(reference['predicted_class'])
                reference_score = max(reference['probabilities'])
                different = reference['predicted_class'] != predicted
                description = '两个模型判断不同，需要结合实际物品确认。' if different else '两个模型的第一候选相同；得分高低不代表整体性能优劣。'
                comparison_title = run_label(comparison_run)
                st.markdown(f'<div class="compare-card"><div class="prediction-eyebrow">{html.escape(comparison_title)} · 相同图片条件</div><div class="prediction-line"><strong>{html.escape(reference_label)}</strong><span>{reference_score:.1%}</span></div><p>{description}</p></div>', unsafe_allow_html=True)
                if show_heat:
                    reference_overlay = explanation_for(comparison_run, comparison_stamp, image_bytes, brightness, blur, selected_framing, crop_box)
                    st.image(reference_overlay, width=200, caption=comparison_title + '对“' + reference_label + '”的贡献区域')
    except (ValueError, RuntimeError, OSError) as error:
        result_area.error(f'暂时无法完成识别：{error}')
else:
    with result_area:
        st.markdown('<div class="result-empty"><strong>等待一张图片</strong><p>识别结果会显示在这里。左侧调整图片后，结果会自动更新。</p></div>', unsafe_allow_html=True)

st.divider()
with st.expander('类别与适用范围'):
    st.write('模型将整张图片归入纸板、玻璃、金属、纸张、塑料、其他垃圾六类，主要适用于单件物品。白纸背景不等于纸张垃圾，也不会按画面中材质面积占比投票。')
    st.write('“其他垃圾”是数据集预先标注的杂项类，含部分软包装、糖果包装、纸巾状纸团等；与材质类别的边界有重叠，不是“认不出来”的兜底，也不直接等同于国内垃圾投放四分类。')
    with (ROOT/'data'/'split_manifest.csv').open(encoding='utf-8', newline='') as stream:
        examples = [row for row in csv.DictReader(stream) if row['split'] == 'train' and row['class_name'] == 'trash'][:3]
    for column, example in zip(st.columns(3), examples):
        column.image(str(ROOT/example['path']), caption='训练样本：其他垃圾', width=160)
with st.expander('验证实验结果'):
    records = []
    for name in runs:
        summary = json.loads((ROOT/'outputs'/name/'summary.json').read_text(encoding='utf-8'))
        records.append({'训练方案': run_label(name), '输入方式': '完整图片缩放' if framing_by_run[name] == 'fit' else '中心裁剪', '验证准确率': f'{summary["validation_accuracy"]:.2%}',
                        '验证 Macro-F1': f'{summary["validation_macro_f1"]:.4f}'})
    st.dataframe(records, hide_index=True)
    st.caption('验证集整体成绩用于选模，不是当前图片的得分，也不是最终测试成绩。切换模型仅加载已有权重。')
with st.expander('评分与局限'):
    st.write('模型只在六类中评分，未知物品或混合材质也可能被高分误判。第一候选低于 70%、前两名相差不到 15 个百分点，或在模型设置中标记混合输入时，会提示人工确认。')
    st.caption('这是未校准的演示规则，不是未知物体检测。热力图在预测后计算，用于辅助理解模型，不证明判断正确。')
