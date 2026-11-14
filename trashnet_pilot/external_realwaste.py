"""Frozen, evaluation-only transfer check using the official RealWaste release."""
import argparse
import csv
import hashlib
import json
import statistics
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image
from external_metrics import CLASSES, material_metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs'/'external_realwaste'
DATA = ROOT/'data'/'external_realwaste'
PLAN = ROOT/'outputs'/'external_realwaste_plan.json'
MAPPING = {'Cardboard': 'cardboard', 'Glass': 'glass', 'Metal': 'metal',
           'Paper': 'paper', 'Plastic': 'plastic'}
COUNTS = {'Cardboard': 461, 'Food Organics': 411, 'Glass': 420, 'Metal': 790,
          'Miscellaneous Trash': 495, 'Paper': 500, 'Plastic': 921,
          'Textile Trash': 318, 'Vegetation': 436}
SOURCES = ['external_realwaste.py', 'external_metrics.py', 'train_baseline.py',
           'model_utils.py', 'image_framing.py', 'multiscale_model.py', 'capacity_control.py', 'download_realwaste.py']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze_plan():
    study = read(ROOT/'outputs/capacity_control_results.json')
    protected = read(ROOT/'outputs/capacity_control_artifacts.json')
    models = []
    for row in study['rows']:
        relative = Path('outputs')/row['run_name']/'best_model.pt'
        fingerprint = digest(ROOT/relative)
        assert fingerprint == protected['model_artifact_sha256'][str(relative)]
        models.append(dict(recipe=row['recipe'], seed=row['seed'], run_name=row['run_name'],
                           checkpoint=relative.as_posix(), checkpoint_sha256=fingerprint))
    plan = dict(dataset='RealWaste',
                source_url='https://raw.githubusercontent.com/sam-single/realwaste/156e8e9154eeb163df30fcfd767f55b19633bb06/RealWaste/',
                source_archive_kind='author_five_material_subset', author_commit='156e8e9154eeb163df30fcfd767f55b19633bb06',
                canonical_uci_url='https://archive.ics.uci.edu/static/public/908/realwaste.zip',
                previous_uci_attempt_sha256=digest(ROOT/'outputs/external_realwaste_uci_attempt_plan.json'),
                acquisition_change='Slow UCI download; same five classes and scoring, pinned author files; changed before any external prediction',
                dataset_doi='10.24432/C5SS4G', mapping=MAPPING, published_class_counts=COUNTS,
                expected_mapped_count=3092, models=models,
                candidate='fixed', candidate_selected_before_external_scores=True,
                capacity_results_sha256=digest(ROOT/'outputs/capacity_control_results.json'),
                split_manifest_sha256=digest(ROOT/'data/split_manifest.csv'),
                plan_document_sha256=digest(ROOT/'EXTERNAL_REALWASTE_PLAN.md'),
                implementation_sha256={name: digest(ROOT/name) for name in SOURCES},
                input='Full image aspect fit with white padding 224x224; original ImageNet normalization',
                conditions=['original'], primary_metric='Macro-F1 over five truth materials with all six outputs retained',
                duplicate_policy='Exclude original pixel matches and conflicting external pixel groups; keep first same-label pixel duplicate',
                review_heuristic=dict(min_score=.70, min_margin=.15, calibrated=False),
                batch_size=64, workers=2, dtype='float32', model_selection_on_external=False,
                external_training=False, original_test_evaluated=False, deployment_changed=False)
    if PLAN.exists():
        if read(PLAN) != plan:
            raise RuntimeError('Frozen external protocol differs; preserve this study')
    else:
        save(PLAN, plan)
    print('External protocol frozen: 12 existing checkpoints; five materials; no fitting', flush=True)
    return plan


def verify_plan():
    plan = read(PLAN)
    for name, fingerprint in plan['implementation_sha256'].items():
        if digest(ROOT/name) != fingerprint:
            raise RuntimeError('Evaluation implementation changed: '+name)
    for relative, key in [('data/split_manifest.csv', 'split_manifest_sha256'),
                          ('outputs/capacity_control_results.json', 'capacity_results_sha256'),
                          ('EXTERNAL_REALWASTE_PLAN.md', 'plan_document_sha256')]:
        if digest(ROOT/relative) != plan[key]:
            raise RuntimeError('Frozen reference changed: '+relative)
    for model in plan['models']:
        if digest(ROOT/model['checkpoint']) != model['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed: '+model['run_name'])
    return plan


def prepare(archive):
    plan = verify_plan()
    manifest = DATA/'manifest.csv'
    audit_path = DATA/'audit.json'
    archive_hash = digest(archive)
    if audit_path.exists():
        audit = read(audit_path)
        if digest(manifest) != audit['manifest_sha256'] or archive_hash != audit['archive_sha256'] or audit['plan_sha256'] != digest(PLAN):
            raise RuntimeError('Existing external preparation differs')
        print('Reusing audited external manifest', flush=True)
        return
    with (ROOT/'data/split_manifest.csv').open(encoding='utf-8', newline='') as stream:
        original_pixels = {row['pixel_sha256'] for row in csv.DictReader(stream)}
    records, sizes = [], Counter()
    with zipfile.ZipFile(archive) as zipped:
        members = []
        counts = Counter()
        for entry in zipped.infolist():
            parts = PurePosixPath(entry.filename).parts
            if len(parts) == 4 and parts[:2] == ('realwaste-main', 'RealWaste') and parts[2] in COUNTS and parts[3].lower().endswith('.jpg'):
                counts[parts[2]] += 1
                if parts[2] in MAPPING:
                    members.append((entry, parts[2], parts[3]))
        expected_counts = {name: COUNTS[name] for name in MAPPING}
        if dict(counts) != expected_counts or len(members) != plan['expected_mapped_count']:
            raise RuntimeError('Official archive inventory differs: '+str(counts))
        for index, (entry, category, name) in enumerate(sorted(members, key=lambda item: item[0].filename)):
            if entry.file_size > 20*1024*1024 or entry.file_size <= 0:
                raise RuntimeError('Unexpected image size: '+entry.filename)
            target = DATA/'images'/category/name
            if not target.resolve().is_relative_to(DATA.resolve()):
                raise RuntimeError('Unsafe image path')
            content = zipped.read(entry)
            file_hash = hashlib.sha256(content).hexdigest()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target) != file_hash:
                raise RuntimeError('Existing extracted image differs')
            if not target.exists():
                target.write_bytes(content)
            with Image.open(target) as source:
                source.load()
                image = source.convert('RGB')
                pixel_hash = hashlib.sha256(image.width.to_bytes(4, 'big')+image.height.to_bytes(4, 'big')+image.tobytes()).hexdigest()
                sizes[f'{image.width}x{image.height}'] += 1
            label = MAPPING[category]
            records.append(dict(path=target.relative_to(ROOT).as_posix(), class_name=label,
                                label=CLASSES.index(label), source_category=category,
                                file_sha256=file_hash, pixel_sha256=pixel_hash,
                                width=image.width, height=image.height))
            if (index+1) % 500 == 0:
                print(f'Audited {index+1}/{len(members)} external images', flush=True)
    groups = defaultdict(list)
    for row in records:
        groups[row['pixel_sha256']].append(row)
    keep, excluded = [], []
    for pixel_hash, group in groups.items():
        reason = ('original_pixel_match' if pixel_hash in original_pixels else
                  'conflicting_external_labels' if len({row['class_name'] for row in group}) > 1 else None)
        if reason:
            excluded.extend(dict(row, reason=reason) for row in group)
        else:
            keep.append(group[0])
            excluded.extend(dict(row, reason='same_label_pixel_duplicate') for row in group[1:])
    keep.sort(key=lambda row: row['path'])
    if set(row['class_name'] for row in keep) != set(MAPPING.values()):
        raise RuntimeError('Mapped material disappeared after audit')
    write_csv(manifest, keep)
    save(DATA/'excluded.json', excluded)
    audit = dict(dataset='RealWaste', plan_sha256=digest(PLAN), archive_sha256=archive_hash,
                 source_url=plan['source_url'], downloaded_inventory=dict(counts),
                 archive_kind=plan['source_archive_kind'],
                 mapped_before_audit=len(records), retained=len(keep),
                 class_counts=dict(Counter(row['class_name'] for row in keep)),
                 sizes=dict(sizes), excluded_counts=dict(Counter(row['reason'] for row in excluded)),
                 exact_pixel_matches_with_original=sum(row['reason'] == 'original_pixel_match' for row in excluded),
                 manifest_sha256=digest(manifest), original_test_images_opened=False,
                 limitation='Exact pixel matching cannot exclude reencoded near duplicates or common object/scene identities')
    save(audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


def evaluate():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from model_utils import load_checkpoint
    from train_baseline import make_transform, set_seed

    plan = verify_plan()
    audit = read(DATA/'audit.json')
    manifest = DATA/'manifest.csv'
    if audit['plan_sha256'] != digest(PLAN) or digest(manifest) != audit['manifest_sha256']:
        raise RuntimeError('External manifest changed')
    with manifest.open(encoding='utf-8', newline='') as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        if digest(ROOT/row['path']) != row['file_sha256']:
            raise RuntimeError('External image changed: '+row['path'])
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # A module-level dataset is needed by the Windows spawn worker context.
    dataset = RealWasteDataset(records, make_transform(False, framing='fit'))
    loader = DataLoader(dataset, batch_size=plan['batch_size'], shuffle=False,
                        num_workers=plan['workers'], persistent_workers=plan['workers'] > 0,
                        pin_memory=device == 'cuda')
    OUT.mkdir(parents=True, exist_ok=True)
    for position, model_record in enumerate(plan['models']):
        out = OUT/model_record['run_name']
        metrics_path = out/'metrics.json'
        if metrics_path.exists():
            old = read(metrics_path)
            if old['plan_sha256'] != digest(PLAN) or old['manifest_sha256'] != audit['manifest_sha256'] or not (out/'predictions.csv').exists() or digest(out/'predictions.csv') != old['predictions_sha256']:
                raise RuntimeError('Existing external evaluation differs or is incomplete')
            print('Reusing '+model_record['run_name'], flush=True)
            continue
        print(f"External evaluation {position+1}/12: {model_record['run_name']}", flush=True)
        started = time.perf_counter()
        model, checkpoint = load_checkpoint(ROOT/model_record['checkpoint'], device, ROOT/'data/split_manifest.csv')
        if checkpoint['config'].get('framing') != 'fit':
            raise RuntimeError('Checkpoint uses a different input protocol')
        scores = []
        with torch.inference_mode():
            for images in loader:
                scores.extend(model(images.to(device, non_blocking=True)).softmax(1).cpu().tolist())
        metrics, predicted, review = material_metrics([int(row['label']) for row in records], scores)
        out.mkdir(parents=True, exist_ok=True)
        predictions = [dict(path=row['path'], true_class=row['class_name'],
                            predicted_class=CLASSES[int(pred)], top_score=float(max(prob)),
                            needs_review=bool(flag), **{name: float(value) for name, value in zip(CLASSES, prob)})
                       for row, pred, flag, prob in zip(records, predicted, review, scores)]
        write_csv(out/'predictions.csv', predictions)
        metrics.update(model_record, plan_sha256=digest(PLAN), manifest_sha256=audit['manifest_sha256'],
                       predictions_sha256=digest(out/'predictions.csv'),
                       evaluation_seconds=time.perf_counter()-started, device=device,
                       original_test_evaluated=False, external_training=False)
        save(metrics_path, metrics)
        print(f"accuracy={metrics['accuracy']:.4f}; five-material F1={metrics['macro_f1_five_materials']:.4f}", flush=True)
        del model, checkpoint
        if device == 'cuda':
            torch.cuda.empty_cache()
    rows = [read(OUT/model['run_name']/'metrics.json') for model in plan['models']]
    aggregate = {}
    keys = ['accuracy', 'macro_f1_five_materials', 'predicted_trash_rate',
            'plastic_as_glass_rate', 'glass_as_plastic_rate', 'unflagged_error_rate']
    for recipe in ['control', 'fixed', 'capacity', 'legacy']:
        group = [row for row in rows if row['recipe'] == recipe]
        aggregate[recipe] = {key: dict(mean=statistics.mean(row[key] for row in group),
                                       sample_sd=statistics.stdev(row[key] for row in group)) for key in keys}
    paired = []
    for seed in [42, 43, 44]:
        group = {row['recipe']: row for row in rows if row['seed'] == seed}
        paired.append(dict(seed=seed, fixed_minus_control=group['fixed']['macro_f1_five_materials']-group['control']['macro_f1_five_materials'],
                           fixed_minus_capacity=group['fixed']['macro_f1_five_materials']-group['capacity']['macro_f1_five_materials'],
                           fixed_minus_legacy=group['fixed']['macro_f1_five_materials']-group['legacy']['macro_f1_five_materials']))
    save(OUT/'results.json', dict(plan=plan, audit=audit, rows=rows, aggregate=aggregate, paired=paired,
                                 original_test_evaluated=False, deployment_changed=False,
                                 note='External five-material transfer diagnostic; no tuning or per-seed selection'))
    print('Completed all twelve frozen external evaluations', flush=True)


class RealWasteDataset:
    def __init__(self, records, transform):
        self.records, self.transform = records, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(ROOT/self.records[index]['path']) as image:
            return self.transform(image.convert('RGB'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'prepare', 'evaluate'])
    parser.add_argument('--archive', type=Path, default=ROOT/'data/downloads/realwaste-author-156e8e9.zip')
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze_plan()
    elif args.stage == 'prepare':
        prepare(args.archive)
    else:
        evaluate()


if __name__ == '__main__':
    main()
