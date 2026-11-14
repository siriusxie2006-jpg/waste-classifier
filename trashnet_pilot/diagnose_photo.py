"""Read-only model diagnosis of a user photo; never train on or add it to a split."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from train_baseline import ROOT, CLASSES, TrashDataset, make_transform, set_seed, torch, plt
from inference import load_run, predict_image, crop_image, gradcam_overlay


DEFAULT_RUN = 'full_image_both_weighted_seed42'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--crop', type=float, nargs=4, required=True,
                        metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    args = parser.parse_args()
    set_seed(42)
    image_bytes = args.image.read_bytes()
    digest = hashlib.sha256(image_bytes).hexdigest()
    out = ROOT/'outputs'/'user_photo_diagnostics'/digest[:12]
    out.mkdir(parents=True, exist_ok=True)
    with Image.open(args.image) as raw:
        image = ImageOps.exif_transpose(raw).convert('RGB')
    cases = {'whole_photo': image, 'manual_bottle_crop': crop_image(image, args.crop)}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = []
    for checkpoint in sorted((ROOT/'outputs').glob('*/best_model.pt')):
        run_name = checkpoint.parent.name
        if not (checkpoint.parent/'summary.json').exists():
            continue
        model, state = load_run(run_name, device)
        native = state['config'].get('framing', 'center')
        modes = [native] if native == 'fit' else [native, 'fit']
        for mode in modes:
            for case_name, photo in cases.items():
                prediction, _ = predict_image(model, photo, device, framing=mode)
                row = dict(run_name=run_name, framing=mode, case=case_name,
                           predicted_class=prediction['predicted_class'],
                           **{name: score for name, score in zip(CLASSES, prediction['probabilities'])})
                rows.append(row)
                print(json.dumps({key: row[key] for key in ['run_name', 'framing', 'case', 'predicted_class', 'glass', 'plastic']}), flush=True)
        del model
    with (out/'predictions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    model, _ = load_run(DEFAULT_RUN, device)
    fig, axes = plt.subplots(2, 3, figsize=(10, 8), layout='constrained')
    fig.set_constrained_layout_pads(h_pad=.10, w_pad=.05, hspace=.08, wspace=.04)
    for axis, (case_name, photo) in zip(axes, cases.items()):
        prediction, prepared = predict_image(model, photo, device, framing='fit')
        prepared.save(out/f'{case_name}_input.png')
        axis[0].imshow(prepared)
        axis[0].set_title(case_name.replace('_', ' ') + '\nActual 224x224 input', fontsize=10)
        for column, name in enumerate(['glass', 'plastic'], 1):
            overlay = gradcam_overlay(model, prepared, device, CLASSES.index(name))
            axis[column].imshow(overlay)
            axis[column].set_title(f"{name} score {prediction['probabilities'][CLASSES.index(name)]:.1%}\nGrad-CAM (qualitative only)", fontsize=10)
        for item in axis:
            item.axis('off')
    fig.suptitle('User-reported plastic bottle: current model diagnosis', fontsize=12)
    fig.savefig(out/'input_and_attribution.png', dpi=150)
    plt.close(fig)

    # Similarity uses training images only, in deterministic evaluation mode.
    backbone = torch.nn.Sequential(*list(model.children())[:-1]).eval()
    transform = make_transform(False, framing='fit')
    dataset = TrashDataset('train', transform)
    features, paths, labels = [], [], []
    with torch.inference_mode():
        query = torch.stack([transform(photo) for photo in cases.values()]).to(device)
        query_features = torch.nn.functional.normalize(backbone(query).flatten(1), dim=1)
        for tensor, label, batch_paths in DataLoader(dataset, batch_size=32, num_workers=0):
            features.append(torch.nn.functional.normalize(backbone(tensor.to(device)).flatten(1), dim=1).cpu())
            paths.extend(batch_paths)
            labels.extend(label.tolist())
        similarities = query_features.cpu() @ torch.cat(features).T
    nearest = {}
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for case_index, case_name in enumerate(cases):
        selected = similarities[case_index].topk(6).indices.tolist()
        nearest[case_name] = []
        for axis, index in zip(axes[case_index], selected):
            record = dict(path=paths[index], label=CLASSES[labels[index]],
                          cosine_similarity=float(similarities[case_index, index]))
            nearest[case_name].append(record)
            with Image.open(ROOT/paths[index]) as sample:
                axis.imshow(sample.convert('RGB'))
            axis.set_title(f"{record['label']} / cosine {record['cosine_similarity']:.3f}\n{Path(paths[index]).name}", fontsize=9)
            axis.axis('off')
    fig.suptitle('Nearest TRAIN images in learned feature space\nTop row: whole photo; bottom: manual crop. Similarity does not establish causation.', fontsize=11)
    fig.tight_layout()
    fig.savefig(out/'nearest_training_examples.png', dpi=140)
    plt.close(fig)
    report = dict(source_path=str(args.image.resolve()), image_sha256=digest, image_size=list(image.size),
                  reported_ground_truth='plastic', crop_box=args.crop,
                  crop_selection='One manually specified box surrounding the visible bottle, before model scoring; not object detection',
                  default_run=DEFAULT_RUN, training_counts=dict(Counter(record['class_name'] for record in dataset.records)),
                  training_image_sizes=dict(Counter(record['width']+'x'+record['height'] for record in dataset.records)),
                  predictions=rows, nearest_training_examples=nearest,
                  diagnostic_only=True, image_added_to_training=False, test_evaluated=False,
                  limitations=['Single user feedback case, not an accuracy estimate',
                               'No model weights, class scores or split records were modified',
                               'Grad-CAM and nearest feature examples do not prove causal reliance on a feature'])
    (out/'diagnosis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'DIAGNOSIS SAVED: {out}', flush=True)


if __name__ == '__main__':
    main()
