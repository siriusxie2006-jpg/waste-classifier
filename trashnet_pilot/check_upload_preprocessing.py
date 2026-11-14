"""Validation-only diagnostic: historical center crop vs full-image letterbox."""
import json
from torchvision import transforms as T
from torch.utils.data import DataLoader
from train_baseline import ROOT, TrashDataset, make_transform, evaluate, torch, set_seed
from inference import prepare_image, predict_image, load_run


class FitTransform:
    def __call__(self, image):
        return T.Compose(make_transform(False).transforms[2:])(prepare_image(image, framing='fit'))


def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = DataLoader(TrashDataset('val', FitTransform()), batch_size=32, num_workers=0)
    rows = []
    for run in ['baseline_resnet18_seed42', 'merged_both_weighted_seed42']:
        model, _ = load_run(run, device)
        fitted, _ = evaluate(model, data, device)
        original = json.loads((ROOT/'outputs'/run/'val_metrics.json').read_text(encoding='utf-8'))
        row = dict(run_name=run, validation_count=fitted['count'],
                   center_crop_accuracy=original['accuracy'], center_crop_macro_f1=original['macro_f1'],
                   full_image_accuracy=fitted['accuracy'], full_image_macro_f1=fitted['macro_f1'],
                   full_image_confusion_matrix=fitted['confusion_matrix'])
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(split='validation', test_evaluated=False,
                  note='Same weights. Diagnostic of preprocessing only; not a retraining result.', results=rows)
    (ROOT/'outputs'/'upload_preprocessing_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
