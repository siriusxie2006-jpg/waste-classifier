"""Check display rules and the two local demo workflows; no test images used."""
import csv
import hashlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from demo_labels import CLASS_LABELS, RUN_LABELS, run_label, assess_scores

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


class DisplayRuleTests(unittest.TestCase):
    def test_other_waste_is_a_learned_class_not_rejection(self):
        result = assess_scores([.01, .01, .01, .01, .01, .95], CLASSES)
        self.assertEqual(result['predicted_class'], 'trash')
        self.assertFalse(result['needs_review'])
        self.assertIn('数据集杂项类', CLASS_LABELS['trash'])

    def test_mixed_input_stays_uncertain_even_at_high_score(self):
        result = assess_scores([.99, .002, .002, .002, .002, .002], CLASSES, mixed_input=True)
        self.assertTrue(result['needs_review'])
        self.assertIn('混合材质', result['reasons'][0])

    def test_close_candidates_and_low_score_request_review(self):
        result = assess_scores([.45, .40, .05, .04, .03, .03], CLASSES)
        self.assertTrue(result['needs_review'])
        self.assertTrue(any('接近' in reason for reason in result['reasons']))

    def test_high_score_alone_cannot_detect_unknown_objects(self):
        # Explicit limitation of the heuristic: high scores can still be wrong.
        result = assess_scores([.01, .01, .01, .01, .95, .01], CLASSES)
        self.assertFalse(result['needs_review'])
        with self.assertRaises(ValueError):
            assess_scores([float('nan')] * 6, CLASSES)

    def test_completed_runs_have_distinct_readable_labels(self):
        runs = [path.parent.name for path in (ROOT/'outputs').glob('*/summary.json')]
        self.assertTrue(all(name in RUN_LABELS for name in runs))
        self.assertEqual(len(set(run_label(name) for name in runs)), len(runs))


def check_app():
    from PIL import Image
    from streamlit.testing.v1 import AppTest
    from inference import gradcam_overlay, predict_image
    with (ROOT/'data'/'split_manifest.csv').open(encoding='utf-8', newline='') as stream:
        record = next(row for row in csv.DictReader(stream) if row['split'] == 'val')
    uploaded = io.BytesIO()
    with Image.open(ROOT/record['path']) as image:
        image.save(uploaded, format='PNG')
    uploaded.seek(0)
    with patch('streamlit.file_uploader', return_value=uploaded):
        app = AppTest.from_file(str(ROOT/'demo_app.py')).run(timeout=60)
        assert not app.exception, [item.message for item in app.exception]
        assert not app.error, [item.value for item in app.error]
        assert len(app.radio) == 0, 'No redundant use-mode selection'
        assert len(app.selectbox) == 1, 'Only the optional model setting remains'
        assert app.selectbox[0].value == 'multiscale_confirm_control_seed43'
        options = ['基础 ResNet18', 'ResNet18（整图增强）']
        bottle_directory = ROOT/'outputs'/'bottle_6_25_seed43'
        has_bottle_run = all((bottle_directory/name).is_file() for name in ('best_model.pt','summary.json'))
        if has_bottle_run:
            options.append('ResNet18（瓶类样本微调）')
        assert app.selectbox[0].options == options
        assert sum('<div class="prediction-card' in item.value for item in app.markdown) == 1
        assert not any('<div class="compare-card' in item.value for item in app.markdown)
        assert not next(item for item in app.toggle if item.key == 'compare_baseline').value
        assert 'seed42' not in app.dataframe[0].value.to_string()

        crop_event = {'image_id': hashlib.sha256(uploaded.getvalue()).hexdigest(),
                      'box': [.1, .1, .9, .9], 'event_id': 'regression-crop-1'}
        uploaded.seek(0)
        with patch('crop_widget.manual_crop', return_value=crop_event), patch('inference.predict_image', wraps=predict_image) as crop_calls:
            app.run(timeout=60)
        assert not app.exception, [item.message for item in app.exception]
        assert app.session_state['crop_box'] == (.1, .1, .9, .9)
        assert any('你选择的裁剪区域' in item.value for item in app.caption)
        assert crop_calls.call_args_list and all(call.kwargs['framing'] == 'fit' for call in crop_calls.call_args_list)
        assert all(call.args[1].width < image.width and call.args[1].height < image.height
                   for call in crop_calls.call_args_list), 'Inference must receive the confirmed crop'

        next(item for item in app.slider if item.key == 'brightness').set_value(.2)
        next(item for item in app.slider if item.key == 'blur').set_value(4.)
        uploaded.seek(0)
        with patch('inference.predict_image', wraps=predict_image) as prediction_calls:
            app.run(timeout=60)
            assert not app.exception, [item.message for item in app.exception]
            assert not app.error, [item.value for item in app.error]
            assert any(call.kwargs['brightness'] == .2 and call.kwargs['blur'] == 4.
                       for call in prediction_calls.call_args_list)

        next(item for item in app.checkbox if item.key == 'mixed_input').check()
        next(item for item in app.toggle if item.key == 'show_heat').set_value(True)
        uploaded.seek(0)
        with patch('inference.gradcam_overlay', wraps=gradcam_overlay) as heat_calls:
            app.run(timeout=60)
            assert not app.exception, [item.message for item in app.exception]
            assert not app.error, [item.value for item in app.error]
            assert len(heat_calls.call_args_list) == 1, 'Explain the single active prediction automatically'
            assert 0 <= heat_calls.call_args_list[0].kwargs['class_index'] < 6
        assert len(app.selectbox) == 1, 'No manual heatmap-class selector'
        assert any('需要人工确认' in item.value for item in app.markdown)

        next(item for item in app.toggle if item.key == 'compare_baseline').set_value(True)
        uploaded.seek(0)
        app.run(timeout=60)
        assert not app.exception, [item.message for item in app.exception]
        assert not app.error, [item.value for item in app.error]
        comparison_cards = [item.value for item in app.markdown if '<div class="compare-card' in item.value]
        assert len(comparison_cards) == 1 and 'ResNet18（整图增强）' in comparison_cards[0]

        app.selectbox[0].set_value('coverage_control_seed43')
        uploaded.seek(0)
        app.run(timeout=60)
        assert not app.exception, [item.message for item in app.exception]
        assert not app.error, [item.value for item in app.error]
        comparison_cards = [item.value for item in app.markdown if '<div class="compare-card' in item.value]
        assert len(comparison_cards) == 1 and '基础 ResNet18' in comparison_cards[0]

        if has_bottle_run:
            app.selectbox[0].set_value('bottle_6_25_seed43')
            uploaded.seek(0)
            app.run(timeout=60)
            assert not app.exception and not app.error
            assert any('当前模型 · ResNet18（瓶类样本微调）' in item.value for item in app.caption)
            assert any('基础 ResNet18' in item.value for item in app.markdown if '<div class="compare-card' in item.value)

        next(item for item in app.button if item.key == 'reset_image').click()
        uploaded.seek(0)
        with patch('crop_widget.manual_crop', return_value=crop_event):
            app.run(timeout=60)
        assert not app.exception, [item.message for item in app.exception]
        assert next(item for item in app.slider if item.key == 'brightness').value == 1.
        assert next(item for item in app.slider if item.key == 'blur').value == 0.
        assert app.session_state['crop_box'] is None, 'Reset must not reapply a stale crop event'

    empty = AppTest.from_file(str(ROOT/'demo_app.py')).run(timeout=60)
    assert not empty.exception, [item.message for item in empty.exception]
    next(item for item in empty.button if item.key == 'use_example').click()
    empty.run(timeout=60)
    assert not empty.exception, [item.message for item in empty.exception]
    assert any('<div class="prediction-card' in item.value for item in empty.markdown)
    print('APP PASS: matched full-image models, explicit crop, live adjustments, automatic Grad-CAM, comparison, reset without stale crop, training example', flush=True)

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DisplayRuleTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    if '--app' in sys.argv:
        check_app()
