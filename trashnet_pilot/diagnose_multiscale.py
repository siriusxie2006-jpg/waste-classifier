"""Post-pilot diagnosis of routing collapse and an optional known feedback photo."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
from train_baseline import ROOT, torch, set_seed, TrashDataset, make_transform
from inference import load_run, predict_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feedback-image', type=Path)
    args = parser.parse_args()
    results_path = ROOT/'outputs'/'multiscale_pilot_results.json'
    result_hash = hashlib.sha256(results_path.read_bytes()).hexdigest()
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_run('multiscale_pilot_gated_seed42', device)
    ratios, alphas, predictions, targets = [], [], [], []
    changes = 0
    with torch.inference_mode():
        data = DataLoader(TrashDataset('val', make_transform(False, framing='fit')), batch_size=32, num_workers=0)
        for inputs, labels, _ in data:
            detail, semantic = model.forward_features(inputs.to(device))
            logits, alpha = model.fc.forward_with_gate(detail, semantic)
            aligned = torch.relu(model.fc.project_detail(detail))
            detail_norm = (alpha*aligned).norm(dim=1)
            semantic_norm = ((1-alpha)*semantic).norm(dim=1)
            ratios.extend((detail_norm/(detail_norm+semantic_norm).clamp_min(1e-12)).cpu().tolist())
            alphas.extend(alpha.flatten().cpu().tolist())
            predicted = logits.argmax(1)
            changes += int((model.fc.classifier(semantic).argmax(1) != predicted).sum())
            predictions.extend(predicted.cpu().tolist())
            targets.extend(labels.tolist())
    cm = torch.bincount(torch.tensor(targets)*6+torch.tensor(predictions), minlength=36).reshape(6, 6).tolist()
    saved = json.loads((ROOT/'outputs'/'multiscale_pilot_gated_seed42'/'val_metrics.json').read_text(encoding='utf-8'))
    if cm != saved['confusion_matrix']:
        raise RuntimeError('Contribution probe predictions differ from saved validation')
    collapse = dict(split='validation_clean', count=len(ratios), detail_weight_mean=statistics.mean(alphas), detail_weight_max=max(alphas),
                    detail_contribution_norm_fraction_mean=statistics.mean(ratios), detail_contribution_norm_fraction_max=max(ratios),
                    changed_predictions_when_detail_bypassed=changes,
                    interpretation='At this checkpoint the detail path contributes almost nothing on clean validation inputs; this does not identify the training-time cause',
                    weights_modified=False, test_evaluated=False)
    (ROOT/'outputs'/'multiscale_pilot_gate_collapse.json').write_text(json.dumps(collapse, indent=2), encoding='utf-8')
    print(json.dumps(collapse), flush=True)
    if args.feedback_image:
        with Image.open(args.feedback_image) as raw:
            photo = ImageOps.exif_transpose(raw).convert('RGB')
        records = []
        for recipe in ['control', 'fixed', 'gated']:
            candidate, _ = load_run(f'multiscale_pilot_{recipe}_seed42', device)
            prediction, _ = predict_image(candidate, photo, device, framing='fit')
            records.append(dict(recipe=recipe, predicted_class=prediction['predicted_class'],
                                glass=prediction['probabilities'][1], plastic=prediction['probabilities'][4]))
        feedback = dict(records=records, source_sha256=hashlib.sha256(args.feedback_image.read_bytes()).hexdigest(),
                        image_size=list(photo.size), diagnostic_only=True, previously_seen_failure_case=True,
                        used_for_selection=False, test_evaluated=False,
                        note='Single known feedback case; model scores are not correctness guarantees or real-world accuracy estimates')
        (ROOT/'outputs'/'multiscale_pilot_feedback.json').write_text(json.dumps(feedback, indent=2), encoding='utf-8')
        print(json.dumps(records), flush=True)
    if hashlib.sha256(results_path.read_bytes()).hexdigest() != result_hash:
        raise RuntimeError('Selection results changed during diagnostic check')


if __name__ == '__main__':
    main()
