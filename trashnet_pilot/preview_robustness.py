"""Exploratory validation-only robustness check. Never touches the test split."""
import argparse
import csv
import json
import time
from pathlib import Path
from PIL import ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import transforms as T
from torchvision.models import resnet18
from train_baseline import ROOT, CLASSES, TrashDataset, evaluate, make_transform, set_seed
from torch.utils.data import DataLoader
from model_utils import load_checkpoint


class Corrupt:
    def __init__(self, brightness=None, blur=None):
        self.brightness, self.blur = brightness, blur

    def __call__(self, im):
        if self.brightness is not None:
            im = ImageEnhance.Brightness(im).enhance(self.brightness)
        if self.blur is not None:
            im = im.filter(ImageFilter.GaussianBlur(self.blur))
        return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='baseline_resnet18_seed42')
    parser.add_argument('--include-bright', action='store_true')
    args = parser.parse_args()
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = ROOT / 'outputs' / args.run_name
    model, ckpt = load_checkpoint(out/'best_model.pt', device, ROOT/'data'/'split_manifest.csv')
    base = list(make_transform(False, framing=ckpt['config'].get('framing', 'center')).transforms)
    conditions = [('clean', None, None), ('brightness_0.7', .7, None), ('brightness_0.4', .4, None),
                  ('brightness_0.2', .2, None), ('blur_1', None, 1), ('blur_2', None, 2), ('blur_4', None, 4)]
    if args.include_bright:
        conditions += [('brightness_1.3', 1.3, None), ('brightness_1.7', 1.7, None)]
    rows = []
    for name, brightness, blur in conditions:
        # Apply deterministic corruption after resize/crop, before normalization.
        transform = T.Compose(base[:-2] + [Corrupt(brightness, blur)] + base[-2:])
        data = DataLoader(TrashDataset('val', transform), batch_size=32, num_workers=0)
        t0 = time.perf_counter()
        result, _ = evaluate(model, data, device)
        row = dict(condition=name, accuracy=result['accuracy'], macro_f1=result['macro_f1'],
                   count=result['count'], seconds=time.perf_counter()-t0, split='validation')
        rows.append(row); print(json.dumps(row), flush=True)
    stem = 'validation_robustness_extended' if args.include_bright else 'validation_robustness_preview'
    with (out/f'{stem}.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(9,4.5))
    ax.bar([r['condition'] for r in rows], [r['macro_f1'] for r in rows])
    ax.set(ylabel='Macro-F1', ylim=(0,1), title=f'{args.run_name}\nExploratory validation check - not final test results')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    fig.tight_layout(); fig.savefig(out/f'{stem}.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
