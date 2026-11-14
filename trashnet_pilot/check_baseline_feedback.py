"""Post-selection check of an already-seen feedback photo, never a test accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps
from train_baseline import ROOT, CLASSES, set_seed
from model_utils import set_training_phase
from inference import load_run, predict_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--expected', choices=CLASSES, default='plastic')
    args = parser.parse_args()
    set_seed(42)
    frozen_path = ROOT/'outputs'/'baseline_recipe.json'
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    if frozen.get('frozen_for_next_model_comparison') is not True:
        raise RuntimeError('Freeze the tuning recipe before inspecting this feedback case')
    with Image.open(args.image) as raw:
        photo = ImageOps.exif_transpose(raw).convert('RGB')
    names = list(dict.fromkeys(['full_image_control_seed42', 'full_image_both_weighted_seed42', frozen['representative_run']]))
    records = []
    for name in names:
        model, checkpoint = load_run(name)
        scope = checkpoint['config']['finetune_scope']
        set_training_phase(model, scope)
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        model.eval()
        prediction, _ = predict_image(model, photo, framing='fit')
        record = dict(run_name=name, predicted_class=prediction['predicted_class'],
                      probabilities=dict(zip(CLASSES, prediction['probabilities'])),
                      fine_tuning_scope=scope, fine_tuning_trainable_parameters=trainable)
        records.append(record)
        print(json.dumps(record), flush=True)
    report = dict(source_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
                  image_size=list(photo.size), expected_class=args.expected,
                  baseline_recipe_sha256=hashlib.sha256(frozen_bytes).hexdigest(),
                  records=records, diagnostic_only=True, previously_seen_failure_case=True,
                  used_for_tuning_selection=False, test_evaluated=False,
                  note='A single known feedback case cannot estimate real-world accuracy; results do not alter the frozen recipe')
    (ROOT/'outputs'/'baseline_tuning_feedback.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != report['baseline_recipe_sha256']:
        raise RuntimeError('Frozen baseline recipe changed during diagnostic check')


if __name__ == '__main__':
    main()
