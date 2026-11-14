"""Post-hoc input-geometry diagnosis; preserve models, old results and test data."""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from domain_probes import CONDITIONS, agreement_metrics, probe_frame
from external_metrics import CLASSES, material_metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs/domain_input_diagnosis'
CACHE = ROOT/'cache/domain_input_diagnosis'
PLAN = OUT/'plan.json'
CORE = ['diagnose_domain_inputs.py', 'domain_probes.py', 'external_metrics.py',
        'train_baseline.py', 'model_utils.py', 'image_framing.py',
        'multiscale_model.py', 'capacity_control.py']


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


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze():
    external = read(ROOT/'outputs/external_realwaste_plan.json')
    models = [row for row in external['models'] if row['recipe'] in ('fixed', 'legacy')]
    plan = dict(purpose='Post-hoc geometry and padding diagnosis after external results; not an independent test',
                models=models, conditions=CONDITIONS, source_version='2026-11-10',
                datasets={'validation5': dict(count=353, input_count=375, original_split='val', labels=list(range(5)), batch_size=32),
                          'realwaste': dict(count=3092, input_count=3092, labels=list(range(5)), batch_size=64)},
                numerical_reproduction='Preserve all 375 original validation inputs and batch order, then score five-material subset; attempt1 stopped before intervention evaluation',
                primary_metric='Five-truth-material Macro-F1, all six outputs retained',
                intervention_space='Canonical full-image fit 224x224; retain all content; no target extraction',
                diagnostics_only=True, independent_new_test=False, training=False,
                original_test_opened=False, deployment_changed=False,
                existing_external_result_sha256=digest(ROOT/'outputs/external_realwaste/results.json'),
                existing_external_artifacts_sha256=digest(ROOT/'outputs/external_realwaste/artifacts.json'),
                external_manifest_sha256=digest(ROOT/'data/external_realwaste/manifest.csv'),
                original_manifest_sha256=digest(ROOT/'data/split_manifest.csv'),
                document_sha256=digest(ROOT/'DOMAIN_DIAGNOSIS_PLAN.md'),
                implementation_sha256={name: digest(ROOT/name) for name in CORE},
                baseline_tolerance=1e-5, workers=2,
                sample_rule='First three per material after SHA256(path) sorting, independent of model scores',
                features=['aspect_ratio', 'pad_fraction', 'luma_mean', 'rim_luma_mean', 'rim_luma_std', 'gradient_mean'])
    if PLAN.exists() and read(PLAN) != plan:
        raise RuntimeError('Diagnosis plan differs; do not overwrite')
    save(PLAN, plan)
    print('Frozen post-hoc diagnosis: six weights, six conditions, no training', flush=True)


def verify():
    plan = read(PLAN)
    for name, value in plan['implementation_sha256'].items():
        if digest(ROOT/name) != value:
            raise RuntimeError('Diagnosis source changed: '+name)
    for path, key in [('data/split_manifest.csv', 'original_manifest_sha256'),
                      ('data/external_realwaste/manifest.csv', 'external_manifest_sha256'),
                      ('outputs/external_realwaste/results.json', 'existing_external_result_sha256'),
                      ('outputs/external_realwaste/artifacts.json', 'existing_external_artifacts_sha256'),
                      ('DOMAIN_DIAGNOSIS_PLAN.md', 'document_sha256')]:
        if digest(ROOT/path) != plan[key]:
            raise RuntimeError('Protected diagnosis reference changed: '+path)
    for model in plan['models']:
        if digest(ROOT/model['checkpoint']) != model['checkpoint_sha256']:
            raise RuntimeError('Model changed')
    return plan


def describe(image, width, height):
    rgb = np.asarray(image.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)/255
    luma = rgb @ np.array([.299, .587, .114], dtype=np.float32)
    rim = np.ones((128, 128), dtype=bool)
    rim[6:-6, 6:-6] = False
    gradient = (np.abs(np.diff(luma, axis=0)).mean()+np.abs(np.diff(luma, axis=1)).mean())/2
    return dict(aspect_ratio=image.width/image.height, pad_fraction=1-width*height/(224*224),
                luma_mean=float(luma.mean()), rim_luma_mean=float(luma[rim].mean()),
                rim_luma_std=float(luma[rim].std()), gradient_mean=float(gradient))


def prepare():
    from image_framing import fit_image
    plan = verify()
    audit_path = OUT/'preparation.json'
    if audit_path.exists():
        audit = read(audit_path)
        if audit['plan_sha256'] != digest(PLAN):
            raise RuntimeError('Cache belongs to another plan')
        for item in audit['datasets'].values():
            for key, fingerprint in [('manifest', 'manifest_sha256'), ('frames', 'frames_sha256')]:
                if digest(ROOT/item[key]) != item[fingerprint]:
                    raise RuntimeError('Cached diagnosis input changed')
        if digest(OUT/'coverage_features.csv') != audit['features_sha256']:
            raise RuntimeError('Coverage features changed')
        print('Reusing fixed diagnosis frame caches', flush=True)
        return
    original = read_csv(ROOT/'data/split_manifest.csv')
    groups = {'train': [row for row in original if row['split'] == 'train'],
              'validation5': [row for row in original if row['split'] == 'val'],
              'realwaste': read_csv(ROOT/'data/external_realwaste/manifest.csv')}
    CACHE.mkdir(parents=True, exist_ok=True)
    features, caches = [], {}
    for group, records in groups.items():
        if group != 'train' and len(records) != plan['datasets'][group]['input_count']:
            raise RuntimeError('Unexpected diagnosis image count')
        temporary = CACHE/f'{group}.npy.part'
        frames = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8,
                                         shape=(len(records), 224, 224, 3)) if group != 'train' else None
        cached_rows = []
        for index, row in enumerate(records):
            path = ROOT/row['path']
            if row.get('file_sha256') and digest(path) != row['file_sha256']:
                raise RuntimeError('External source file changed')
            with Image.open(path) as source:
                source.load()
                image = source.convert('RGB')
                pixel_hash = hashlib.sha256(image.width.to_bytes(4, 'big')+image.height.to_bytes(4, 'big')+image.tobytes()).hexdigest()
                if pixel_hash != row['pixel_sha256']:
                    raise RuntimeError('Source pixels changed: '+row['path'])
                resized = ImageOps.contain(image, (224, 224), method=Image.Resampling.BILINEAR)
                width, height = resized.size
                frame = fit_image(image)
                features.append(dict(group=group, path=row['path'], class_name=row['class_name'],
                                     label=int(row['label']), **describe(image, width, height)))
                left, top = (224-width)//2, (224-height)//2
                if frames is not None:
                    frames[index] = np.asarray(frame)
                    cached_rows.append(dict(index=index, path=row['path'], class_name=row['class_name'],
                                            label=int(row['label']), left=left, top=top,
                                            right=left+width, bottom=top+height))
            if (index+1) % 1000 == 0:
                print(f'Checked {group} {index+1}/{len(records)}', flush=True)
        if frames is not None:
            frames.flush()
            del frames
            frame_path = CACHE/(f'{group}_inputs375.npy' if group == 'validation5' else f'{group}.npy')
            temporary.replace(frame_path)
            manifest = OUT/f'{group}_inputs.csv'
            write_csv(manifest, cached_rows)
            caches[group] = dict(count=len(records), frames=str(frame_path.relative_to(ROOT)),
                                 frames_sha256=digest(frame_path), manifest=str(manifest.relative_to(ROOT)),
                                 manifest_sha256=digest(manifest))
    write_csv(OUT/'coverage_features.csv', features)
    audit = dict(plan_sha256=digest(PLAN), datasets=caches,
                 coverage_counts=dict(Counter(row['group'] for row in features)),
                 features_sha256=digest(OUT/'coverage_features.csv'),
                 original_test_images_opened=False,
                 training_images_used_for_statistics_only=True,
                 edge_measurement='Outer six pixels of a descriptor-only 128x128 resize; not a background mask')
    save(audit_path, audit)
    print(json.dumps(audit), flush=True)


class ProbeDataset:
    def __init__(self, records, frame_path, condition, normalize):
        self.records, self.frame_path, self.condition, self.normalize = records, frame_path, condition, normalize
        self.frames = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self.frames is None:
            self.frames = np.load(self.frame_path, mmap_mode='r')
        row = self.records[index]
        frame = Image.fromarray(np.asarray(self.frames[index]))
        box = [int(row[key]) for key in ['left', 'top', 'right', 'bottom']]
        return self.normalize(probe_frame(frame, self.condition, box))


def verify_original(dataset, model, records, scores):
    folder = ROOT/'outputs'
    if dataset == 'realwaste':
        folder /= 'external_realwaste'
    filename = 'predictions.csv' if dataset == 'realwaste' else 'val_predictions.csv'
    prior = {row['path']: row for row in read_csv(folder/model['run_name']/filename)}
    errors, maximum = 0, 0.
    for row, probabilities in zip(records, scores):
        reference = prior[row['path']]
        if reference['true_class'] != row['class_name']:
            raise RuntimeError('Canonical labels differ')
        predicted = CLASSES[int(np.argmax(probabilities))]
        errors += predicted != reference['predicted_class']
        if dataset == 'realwaste':
            maximum = max(maximum, max(abs(float(reference[name])-probability) for name, probability in zip(CLASSES, probabilities)))
        else:
            maximum = max(maximum, abs(float(reference['confidence'])-max(probabilities)))
    if errors or maximum > read(PLAN)['baseline_tolerance']:
        raise RuntimeError(f'Canonical original not reproduced: {errors} labels, {maximum} score difference')
    return dict(class_mismatches=errors, max_score_difference=maximum)


def evaluate():
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from model_utils import load_checkpoint
    from train_baseline import make_transform, set_seed
    plan = verify()
    prep = read(OUT/'preparation.json')
    if prep['plan_sha256'] != digest(PLAN):
        raise RuntimeError('Prepared inputs differ from plan')
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    normalize = T.Compose(make_transform(False, framing='fit').transforms[1:])
    summary = []
    for dataset, entry in prep['datasets'].items():
        if digest(ROOT/entry['manifest']) != entry['manifest_sha256'] or digest(ROOT/entry['frames']) != entry['frames_sha256']:
            raise RuntimeError('Prepared frame cache changed')
        records = read_csv(ROOT/entry['manifest'])
        scored_mask = np.asarray([int(row['label']) < 5 for row in records])
        scored_records = [row for row in records if int(row['label']) < 5]
        truth = [int(row['label']) for row in scored_records]
        if len(truth) != plan['datasets'][dataset]['count']:
            raise RuntimeError('Scored five-material subset differs')
        for condition in CONDITIONS:
            data = ProbeDataset(records, ROOT/entry['frames'], condition, normalize)
            loader = DataLoader(data, batch_size=plan['datasets'][dataset]['batch_size'], shuffle=False,
                                num_workers=plan['workers'], persistent_workers=True, pin_memory=device == 'cuda')
            for model_entry in plan['models']:
                out = OUT/dataset/model_entry['run_name']/condition
                out.mkdir(parents=True, exist_ok=True)
                metrics_path = out/'metrics.json'
                if metrics_path.exists():
                    metrics = read(metrics_path)
                    if metrics['plan_sha256'] != digest(PLAN) or digest(out/'scores.npy') != metrics['scores_sha256'] or digest(out/'predictions.csv') != metrics['predictions_sha256']:
                        raise RuntimeError('Partial or changed saved diagnosis result')
                    summary.append(metrics)
                    continue
                started = time.perf_counter()
                model, checkpoint = load_checkpoint(ROOT/model_entry['checkpoint'], device, ROOT/'data/split_manifest.csv')
                scores = []
                with torch.inference_mode():
                    for batch in loader:
                        scores.extend(model(batch.to(device, non_blocking=True)).softmax(1).cpu().tolist())
                scores = np.asarray(scores, dtype=np.float32)[scored_mask]
                metrics, predicted, review = material_metrics(truth, scores)
                original_folder = OUT/dataset/model_entry['run_name']/'original_white'
                reference = scores if condition == 'original_white' else np.load(original_folder/'scores.npy')
                metrics.update(agreement_metrics(truth, scores, reference))
                if condition == 'original_white':
                    metrics['original_reproduction'] = verify_original(dataset, model_entry, scored_records, scores)
                np.save(out/'scores.npy', scores)
                predictions = [dict(path=row['path'], true_class=row['class_name'], predicted_class=CLASSES[int(label)],
                                    top_score=float(max(probability)), needs_review=bool(flag),
                                    **{name: float(value) for name, value in zip(CLASSES, probability)})
                               for row, label, flag, probability in zip(scored_records, predicted, review, scores)]
                write_csv(out/'predictions.csv', predictions)
                metrics.update(model_entry, dataset=dataset, condition=condition, plan_sha256=digest(PLAN),
                               scores_sha256=digest(out/'scores.npy'), predictions_sha256=digest(out/'predictions.csv'),
                               seconds=time.perf_counter()-started, training=False, original_test_opened=False)
                save(metrics_path, metrics)
                summary.append(metrics)
                print(f"{dataset} {model_entry['run_name']} {condition}: accuracy {metrics['accuracy']:.4f}, F1 {metrics['macro_f1_five_materials']:.4f}, agreement {metrics['prediction_agreement']:.4f}", flush=True)
                del model, checkpoint
                if device == 'cuda':
                    torch.cuda.empty_cache()
            del loader, data
    verify()
    save(OUT/'results.json', dict(plan=plan, rows=summary, training=False,
                                 original_test_opened=False, deployment_changed=False,
                                 interpretation='Post-hoc paired input interventions; not new independent accuracy claims'))
    print('All 72 model/dataset/condition records completed', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'prepare', 'evaluate'])
    stage = parser.parse_args().stage
    {'freeze': freeze, 'prepare': prepare, 'evaluate': evaluate}[stage]()


if __name__ == '__main__':
    main()
