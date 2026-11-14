"""Score all six pairs on the fixed validation protocol, then check known cases."""
import statistics
from pathlib import Path

from PIL import Image, ImageOps
from diagnose_domain_inputs import ProbeDataset
from external_metrics import material_metrics
from inference import predict_image
from model_utils import load_checkpoint
from preview_robustness import Corrupt
from train_baseline import evaluate, torch, T, make_transform, set_seed
from train_geometry_coverage import (CLASSES, MANIFEST, OUT, PLAN, REFERENCE, REFERENCE_ARTIFACTS,
                                     ROOT, digest, read, read_csv, save, write_csv)


class LabeledProbe(ProbeDataset):
    def __getitem__(self, index):
        record = self.records[index]
        return super().__getitem__(index), int(record['label']), record['path']


def cached_inputs(name):
    entry = read(ROOT/'outputs/domain_input_diagnosis/preparation.json')['datasets'][name]
    frames, manifest = ROOT/entry['frames'], ROOT/entry['manifest']
    if digest(frames) != entry['frames_sha256'] or digest(manifest) != entry['manifest_sha256']:
        raise RuntimeError('Diagnosis frame cache differs from its audit')
    return read_csv(manifest), frames


def verification_metrics(name):
    directory = ROOT/'outputs'/name
    return read(directory/'val_metrics.json'), read(directory/'summary.json')


def evaluate_validation(plan, device):
    records, frames = cached_inputs('validation5')
    expected = [(row['path'], row['label']) for row in read_csv(MANIFEST) if row['split'] == 'val']
    if [(row['path'], row['label']) for row in records] != expected or len(records) != plan['validation_count']:
        raise RuntimeError('Cached inputs must match all 375 validation images in order')
    normalization = make_transform(False, framing='fit').transforms[1:]
    probes = {condition: (condition, normalization) for condition in plan['geometry_conditions']}
    for condition in plan['robustness_conditions']:
        if condition == 'clean':
            continue
        family, amount = condition.split('_')
        corruption = Corrupt(**{family: float(amount)})
        probes[condition] = ('original_white', [corruption, *normalization])
    models = []
    rows, condition_rows, class_rows = [], [], []
    for run in plan['runs']:
        for coverage, name in [('original', run['reference']), ('coverage', run['run_name'])]:
            canonical, training = verification_metrics(name)
            path = ROOT/'outputs'/name/'best_model.pt'
            model, checkpoint = load_checkpoint(path, device, MANIFEST)
            if coverage == 'coverage' and (training['plan_sha256'] != digest(PLAN) or checkpoint['config']['plan_sha256'] != digest(PLAN)):
                raise RuntimeError('Training belongs to a different plan: '+name)
            metadata = dict(recipe=run['recipe'], seed=run['seed'], coverage=coverage, run_name=name)
            historical = {}
            if coverage == 'original':
                historical = {row['condition']: float(row['macro_f1']) for row in
                              read_csv(path.parent/'validation_robustness_extended.csv')}
            models.append(dict(model=model, path=path, canonical=canonical, training=training,
                               metadata=metadata, historical=historical, conditions={}))
            del checkpoint
    for condition, (probe, steps) in probes.items():
        dataset = LabeledProbe(records, frames, probe, T.Compose(steps))
        loader = torch.utils.data.DataLoader(dataset, batch_size=plan['common']['batch_size'], shuffle=False)
        batches = [(images.to(device), labels.to(device), paths) for images, labels, paths in loader]
        for entry in models:
            metrics, predictions = evaluate(entry['model'], batches, device)
            entry['conditions'][condition] = metrics
            f1 = metrics['macro_f1']
            path, metadata = entry['path'], entry['metadata']
            if condition == 'original_white':
                previous = read_csv(path.parent/'val_predictions.csv')
                pairs = [(row['path'], row['predicted_class']) for row in previous]
                if pairs != [(row['path'], row['predicted_class']) for row in predictions] or f1 != entry['canonical']['macro_f1']:
                    raise RuntimeError('Original predictions differ: '+str(path.parent.name))
            elif condition in entry['historical'] and abs(f1-entry['historical'][condition]) > 1e-12:
                raise RuntimeError('Historical corruption score differs: '+path.parent.name+' '+condition)
            condition_rows.append(dict(**metadata, condition=condition, accuracy=metrics['accuracy'], macro_f1=f1))
            for category in CLASSES:
                class_rows.append(dict(**metadata, condition=condition, class_name=category,
                                       **metrics['classification_report'][category]))
        del batches
        print('Validated 12 weights on', condition, flush=True)
    for entry in models:
        conditions, canonical = entry['conditions'], entry['canonical']
        brightness = statistics.mean(metrics['macro_f1'] for key, metrics in conditions.items() if key.startswith('brightness'))
        blur = statistics.mean(metrics['macro_f1'] for key, metrics in conditions.items() if key.startswith('blur'))
        row = dict(**entry['metadata'], checkpoint_sha256=digest(entry['path']), clean_accuracy=canonical['accuracy'],
                   clean_f1=canonical['macro_f1'], best_epoch=entry['training']['best_epoch'],
                   geometry_f1=statistics.mean(conditions[key]['macro_f1'] for key in plan['geometry_conditions']),
                   robustness_f1=(brightness+blur)/2,
                   plastic_recall=canonical['classification_report']['plastic']['recall'])
        rows.append(row)
        save(OUT/'validation'/f"{row['run_name']}.json", conditions)
        print('Validation', row, flush=True)
    write_csv(OUT/'validation_conditions.csv', condition_rows)
    write_csv(OUT/'validation_classes.csv', class_rows)
    return rows


def decide(plan, rows):
    metrics = ['geometry_f1', 'clean_f1', 'robustness_f1', 'plastic_recall']
    indexed = {(row['recipe'], row['seed'], row['coverage']):row for row in rows}
    deltas = []
    definitions = {'coverage_control': (('control','coverage'), ('control','original')),
                   'coverage_fixed': (('fixed','coverage'), ('fixed','original')),
                   'fusion_with_coverage': (('fixed','coverage'), ('control','coverage'))}
    for comparison, (new, reference) in definitions.items():
        for seed in (42,43,44):
            a, b = indexed[new[0],seed,new[1]], indexed[reference[0],seed,reference[1]]
            deltas.append(dict(comparison=comparison, seed=seed, **{key:a[key]-b[key] for key in metrics}))
    write_csv(OUT/'paired_differences.csv', deltas)
    decisions = {}
    for comparison, rule in [('coverage_control', plan['coverage_rule']), ('fusion_with_coverage', plan['fusion_rule'])]:
        group = [row for row in deltas if row['comparison'] == comparison]
        means = {key:statistics.mean(row[key] for row in group) for key in metrics}
        checks = dict(geometry_gain=means['geometry_f1'] >= rule['min_geometry_gain'],
                      positive_seeds=sum(row['geometry_f1']>0 for row in group) >= rule['min_positive_seeds'],
                      clean_preserved=means['clean_f1'] >= -rule['max_clean_loss'],
                      plastic_preserved=means['plastic_recall'] >= -rule['max_plastic_recall_loss'])
        if 'max_robustness_loss' in rule:
            checks['robustness_preserved'] = means['robustness_f1'] >= -rule['max_robustness_loss']
        decisions[comparison] = dict(mean_deltas=means, checks=checks, passed=all(checks.values()))
    result = dict(plan_sha256=digest(PLAN), rows=rows, paired_differences=deltas, decisions=decisions, test_evaluated=False)
    save(OUT/'validation_results.json', result)
    return result


def check_development(rows, device):
    records, frames = cached_inputs('realwaste')
    truth = [int(row['label']) for row in records]
    normalize = T.Compose(make_transform(False, framing='fit').transforms[1:])
    batches = torch.utils.data.DataLoader(ProbeDataset(records, frames, 'original_white', normalize), batch_size=64)
    case = read(ROOT/'outputs/user_photo_diagnostics/47f077c9d383/diagnosis.json')
    image_path = Path(case['source_path'])
    if digest(image_path) != case['image_sha256']:
        raise RuntimeError('Feedback photo differs from the known case')
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    previous = {row['run_name']:row for row in read(ROOT/'outputs/external_realwaste/results.json')['rows']}
    development = []
    for row in rows:
        name = row['run_name']
        path = ROOT/'outputs'/name/'best_model.pt'
        if digest(path) != row['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed after validation: '+name)
        model, checkpoint = load_checkpoint(path, device, MANIFEST)
        if row['coverage'] == 'original':
            metrics = previous[name]
        else:
            scores = []
            with torch.inference_mode():
                for batch in batches:
                    scores.extend(model(batch.to(device)).softmax(1).cpu().tolist())
            metrics, predictions, _ = material_metrics(truth, scores)
            write_csv(OUT/f'{name}_realwaste_predictions.csv', [
                dict(path=record['path'], true_class=record['class_name'], predicted_class=CLASSES[int(label)],
                     **dict(zip(CLASSES, values))) for record,label,values in zip(records,predictions,scores)])
        prediction, _ = predict_image(model.cpu(), image, framing='fit')
        development.append(dict(recipe=row['recipe'], coverage=row['coverage'], seed=row['seed'], run_name=name,
                                realwaste_accuracy=metrics['accuracy'], realwaste_f1=metrics['macro_f1_five_materials'],
                                realwaste_plastic_recall=metrics['classification_report']['plastic']['recall'],
                                bottle_prediction=prediction['predicted_class'], bottle_scores=dict(zip(CLASSES,prediction['probabilities']))))
        print('Development', name, metrics['accuracy'], metrics['macro_f1_five_materials'], prediction['predicted_class'], flush=True)
        del model, checkpoint
    return development


def main():
    plan = read(PLAN)
    if digest(REFERENCE) != plan['reference_results_sha256'] or digest(MANIFEST) != plan['split_manifest_sha256']:
        raise RuntimeError('Reference result or split changed')
    if digest(REFERENCE_ARTIFACTS) != plan['reference_artifacts_sha256']:
        raise RuntimeError('Reference artifact manifest changed')
    for relative, expected in read(REFERENCE_ARTIFACTS)['model_artifact_sha256'].items():
        if digest(ROOT/relative) != expected:
            raise RuntimeError('Reference artifact changed: '+relative)
    if any(digest(ROOT/name) != expected for name,expected in plan['source_sha256'].items()):
        raise RuntimeError('Training or probe source changed')
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = evaluate_validation(plan, device)
    result = decide(plan, rows)
    print('Validation decision', result['decisions'], flush=True)
    development = check_development(rows, device)
    save(OUT/'development_results.json', dict(validation_results_sha256=digest(OUT/'validation_results.json'),
                                              rows=development, used_for_selection=False, test_evaluated=False))


if __name__ == '__main__':
    main()
