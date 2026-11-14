"""Check a known photo across saved seeds after the confirmation decision is fixed."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps
from train_baseline import ROOT, CLASSES, set_seed
from inference import load_run, predict_image
from demo_labels import assess_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--expected', choices=CLASSES, default='plastic')
    parser.add_argument('--study', choices=['confirmation', 'capacity'], default='confirmation')
    args = parser.parse_args()
    basename = 'fixed_fusion_confirmation' if args.study == 'confirmation' else 'capacity_control'
    result_path = ROOT/'outputs'/f'{basename}_results.json'
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    set_seed(42)
    with Image.open(args.image) as raw:
        image = ImageOps.exif_transpose(raw).convert('RGB')
    records = []
    for row in result['rows']:
        model, checkpoint = load_run(row['run_name'])
        prediction, _ = predict_image(model, image, framing='fit')
        decision = assess_scores(prediction['probabilities'], CLASSES)
        record = dict(recipe=row['recipe'], seed=row['seed'], run_name=row['run_name'],
                      predicted_class=prediction['predicted_class'], top_score=decision['top_score'],
                      plastic=prediction['probabilities'][4], glass=prediction['probabilities'][1],
                      needs_review=decision['needs_review'], matches_expected=prediction['predicted_class'] == args.expected)
        records.append(record)
        print(json.dumps(record), flush=True)
        del model, checkpoint
    recipes = ['control', 'fixed', 'legacy'] if args.study == 'confirmation' else ['control', 'fixed', 'capacity', 'legacy']
    summary = {recipe: dict(weights_checked=3, first_candidate_matches=sum(row['matches_expected'] for row in records if row['recipe'] == recipe),
                            needs_review=sum(row['needs_review'] for row in records if row['recipe'] == recipe))
               for recipe in recipes}
    hash_key = 'confirmation_results_sha256' if args.study == 'confirmation' else 'capacity_results_sha256'
    report = dict(source_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(), image_size=list(image.size),
                  expected_class=args.expected, **{hash_key: hashlib.sha256(result_bytes).hexdigest()},
                  records=records, summary=summary, known_feedback_case=True, used_for_selection=False,
                  test_evaluated=False,
                  note='Multiple trained weights, one previously seen image. Counts are not accuracy on independent images; scores and review heuristic are uncalibrated.')
    (ROOT/'outputs'/f'{basename}_feedback.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if hashlib.sha256(result_path.read_bytes()).hexdigest() != report[hash_key]:
        raise RuntimeError('Confirmation results changed during feedback check')


if __name__ == '__main__':
    main()
