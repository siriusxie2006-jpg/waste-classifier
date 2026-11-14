"""Export validation-only mistakes and Grad-CAM on the exact model input crop."""
import argparse
import json
from collections import Counter
from PIL import Image
from train_baseline import ROOT, CLASSES, TrashDataset, make_transform, loader, evaluate, plt, torch, set_seed
from inference import load_run, prepare_image, gradcam_overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='baseline_resnet18_seed42')
    parser.add_argument('--max-errors', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.max_errors <= 24:
        parser.error('--max-errors must be between 1 and 24')
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, checkpoint = load_run(args.run_name, device)
    framing = checkpoint['config'].get('framing', 'center')
    metrics, rows = evaluate(model, loader('val', 32, 0, 42, framing=framing), device)
    wrong = [row for row in rows if row['true_class'] != row['predicted_class']]
    pairs = Counter((r['true_class'], r['predicted_class']) for r in wrong)
    out = ROOT/'outputs'/args.run_name/'analysis'
    out.mkdir(exist_ok=True)
    selected = sorted(wrong, key=lambda row: (-row['confidence'], row['path']))[:args.max_errors]
    report = dict(split='validation', framing=framing, count=metrics['count'], error_count=len(wrong),
                  accuracy=metrics['accuracy'], macro_f1=metrics['macro_f1'],
                  classification_report=metrics['classification_report'],
                  confusion_pairs=[dict(true_class=a, predicted_class=b, count=count)
                                   for (a, b), count in pairs.most_common()],
                  example_selection='Highest-confidence validation errors, then path order',
                  selected_errors=selected, test_evaluated=False,
                  gradcam_note='Qualitative attribution only; does not establish absence of background bias')
    if selected:
        fig, axes = plt.subplots(len(selected), 2, figsize=(7, 3*len(selected)), squeeze=False)
        for axis, row in zip(axes, selected):
            with Image.open(ROOT/row['path']) as image:
                prepared = prepare_image(image, framing=framing)
            overlay = gradcam_overlay(model, prepared, device, CLASSES.index(row['predicted_class']))
            axis[0].imshow(prepared)
            axis[0].set_title(f"True: {row['true_class']} / Predicted: {row['predicted_class']}", fontsize=9)
            axis[1].imshow(overlay)
            axis[1].set_title(f"Grad-CAM / confidence {row['confidence']:.1%}", fontsize=9)
            for item in axis:
                item.axis('off')
        fig.suptitle(f'{args.run_name}: selected validation errors', fontsize=11)
        fig.tight_layout()
        fig.savefig(out/'validation_errors_gradcam.png', dpi=140)
        plt.close(fig)
    (out/'validation_error_analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['split', 'count', 'error_count', 'macro_f1']}, indent=2))


if __name__ == '__main__':
    main()
