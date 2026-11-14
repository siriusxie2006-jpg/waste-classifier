"""Lock the validation-selected model and evaluate the existing TrashNet holdout."""
import argparse
from collections import Counter
import hashlib
import statistics

from PIL import Image, ImageOps
from diagnose_domain_inputs import ROOT, digest, read, read_csv, save, write_csv
from domain_probes import CONDITIONS, probe_frame
from image_framing import fit_image
from model_utils import load_checkpoint
from preview_robustness import Corrupt
from run_baseline_check import dataset_audit
from train_baseline import CLASSES, T, evaluate, loader, make_transform, set_seed, torch


OUT = ROOT/'outputs/heldout_test'
LOCK = OUT/'plan.json'
MANIFEST = ROOT/'data/split_manifest.csv'
COVERAGE = ROOT/'outputs/training_coverage'
GROUPS = ['original_control', 'coverage_control', 'original_fixed', 'coverage_fixed']
SOURCES = ['evaluate_heldout_test.py', 'train_baseline.py', 'model_utils.py', 'multiscale_model.py',
           'image_framing.py', 'domain_probes.py', 'preview_robustness.py',
           'diagnose_domain_inputs.py', 'external_metrics.py', 'run_baseline_check.py']


def build_plan():
    validation_path = COVERAGE/'validation_results.json'
    validation = read(validation_path)
    reference = read(COVERAGE/'plan.json')
    integrity = read(COVERAGE/'integrity.json')
    if digest(validation_path) != integrity['validation_result_sha256'] or digest(COVERAGE/'plan.json') != validation['plan_sha256']:
        raise RuntimeError('Validation decision differs from the archived study')
    if not validation['decisions']['coverage_control']['passed'] or validation['decisions']['fusion_with_coverage']['passed']:
        raise RuntimeError('Unexpected validation decision')
    split_hash = digest(MANIFEST)
    if split_hash != reference['split_manifest_sha256']:
        raise RuntimeError('Fixed data split changed')
    audit = dataset_audit()
    records = read_csv(MANIFEST)
    if len({row['path'] for row in records}) != len(records) or any(CLASSES[int(row['label'])] != row['class_name'] for row in records):
        raise RuntimeError('Duplicate paths or inconsistent class mapping')
    test_counts = dict(Counter(row['class_name'] for row in records if row['split'] == 'test'))
    weights = []
    for row in validation['rows']:
        path = ROOT/'outputs'/row['run_name']/'best_model.pt'
        if digest(path) != row['checkpoint_sha256']:
            raise RuntimeError('Checkpoint differs from the validation decision: '+row['run_name'])
        weights.append(dict(run_name=row['run_name'], group=row['coverage']+'_'+row['recipe'],
                            seed=row['seed'], checkpoint_sha256=row['checkpoint_sha256'], best_epoch=row['best_epoch']))
    expected = {(group,seed) for group in GROUPS for seed in (42,43,44)}
    if len(weights) != 12 or {(row['group'],row['seed']) for row in weights} != expected:
        raise RuntimeError('Expected twelve paired checkpoints')
    if reference['geometry_conditions'] != CONDITIONS:
        raise RuntimeError('Geometry protocol differs from validation')
    return dict(split='test', split_manifest_sha256=split_hash, class_order=CLASSES, test_counts=test_counts,
                preflight_manifest_audit=audit, validation_results_sha256=digest(validation_path),
                candidate='coverage_control', selection_rule='Candidate and epochs frozen from validation before holdout inference',
                weights=weights, batch_size=reference['common']['batch_size'],
                geometry_conditions=reference['geometry_conditions'],
                light_blur_conditions=[condition for condition in reference['robustness_conditions'] if condition != 'clean'],
                geometry_metric='Mean of six condition-specific six-class macro-F1 values including original input',
                light_blur_metric='Mean of the brightness-family mean and blur-family mean',
                condition_preprocessing='Full aspect-preserving white fit at 224, then the locked probe and ImageNet normalization',
                source_sha256={name:digest(ROOT/name) for name in SOURCES})


def read_test_frames():
    items = []
    for row in read_csv(MANIFEST):
        if row['split'] != 'test':
            continue
        with Image.open(ROOT/row['path']) as source:
            image = source.convert('RGB')
        pixels = hashlib.sha256(image.width.to_bytes(4, 'big')+image.height.to_bytes(4, 'big')+image.tobytes()).hexdigest()
        if pixels != row['pixel_sha256']:
            raise RuntimeError('Test image pixels changed: '+row['path'])
        resized = ImageOps.contain(image, (224,224), method=Image.Resampling.BILINEAR)
        left, top = (224-resized.width)//2, (224-resized.height)//2
        box = (left, top, left+resized.width, top+resized.height)
        items.append((fit_image(image), box, int(row['label']), row['path']))
    return items


def batches_for(items, condition, batch_size, device, normalize):
    corruption = None
    if condition not in CONDITIONS:
        family, amount = condition.split('_')
        corruption = Corrupt(**{family:float(amount)})
    batches = []
    for start in range(0, len(items), batch_size):
        images, labels, paths = [], [], []
        for frame, box, label, path in items[start:start+batch_size]:
            transformed = corruption(frame) if corruption else probe_frame(frame, condition, box)
            images.append(normalize(transformed))
            labels.append(label)
            paths.append(path)
        batches.append((torch.stack(images).to(device), torch.tensor(labels, device=device), paths))
    return batches


def write_results(plan, rows, conditions, per_class, predictions):
    keys = ['clean_accuracy', 'clean_f1', 'geometry_f1', 'light_blur_f1', 'plastic_recall', 'glass_recall',
            'plastic_as_glass', 'glass_as_plastic']
    groups = {}
    for group in GROUPS:
        subset = [row for row in rows if row['group'] == group]
        groups[group] = {key:dict(mean=statistics.mean(row[key] for row in subset),
                                 sample_sd=statistics.stdev(row[key] for row in subset)) for key in keys}
    indexed = {(row['group'],row['seed']):row for row in rows}
    pairs = []
    for name, new, base in [('coverage_on_resnet18','coverage_control','original_control'),
                            ('coverage_on_fusion','coverage_fixed','original_fixed'),
                            ('fusion_without_coverage','original_fixed','original_control'),
                            ('fusion_with_coverage','coverage_fixed','coverage_control')]:
        for seed in (42,43,44):
            pairs.append(dict(comparison=name, seed=seed,
                              **{key:indexed[new,seed][key]-indexed[base,seed][key] for key in keys}))
    files = {'runs.csv':rows, 'conditions.csv':conditions, 'classes.csv':per_class,
             'predictions.csv':predictions, 'paired_differences.csv':pairs}
    for name, records in files.items():
        temporary = OUT/(name+'.tmp')
        write_csv(temporary, records)
        temporary.replace(OUT/name)
    payload = dict(plan_sha256=digest(LOCK), split='test', image_count=sum(plan['test_counts'].values()),
                   candidate=plan['candidate'], rows=rows, groups=groups, paired_differences=pairs,
                   artifact_sha256={name:digest(OUT/name) for name in files})
    temporary = OUT/'results.json.tmp'
    save(temporary, payload)
    temporary.replace(OUT/'results.json')


def run_test(plan):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalization = T.Compose(make_transform(False, framing='fit').transforms[1:])
    loaded = []
    val = [(x.to(device),y.to(device),paths) for x,y,paths in loader('val', plan['batch_size'], 0, 42, framing='fit')]
    for weight in plan['weights']:
        directory = ROOT/'outputs'/weight['run_name']
        model, state = load_checkpoint(directory/'best_model.pt', device, MANIFEST)
        if state['config']['framing'] != 'fit' or state['epoch'] != weight['best_epoch']:
            raise RuntimeError('Weight metadata differs from locked input or epoch')
        _, reproduced = evaluate(model, val, device)
        expected = read_csv(directory/'val_predictions.csv')
        pairs = lambda rows: [(row['path'],row['true_class'],row['predicted_class']) for row in rows]
        if pairs(reproduced) != pairs(expected):
            raise RuntimeError('Validation predictions did not reproduce: '+weight['run_name'])
        loaded.append((weight, model))
        del state
    del val
    print('Reproduced all 12 validation prediction files before opening test images.', flush=True)
    items = read_test_frames()
    if len(items) != 380:
        raise RuntimeError('Expected all 380 locked test images')
    results = {row['run_name']:{} for row in plan['weights']}
    conditions, per_class, predictions = [], [], []
    for condition in plan['geometry_conditions']+plan['light_blur_conditions']:
        batches = batches_for(items, condition, plan['batch_size'], device, normalization)
        for weight, model in loaded:
            metrics, predicted = evaluate(model, batches, device)
            results[weight['run_name']][condition] = metrics
            info = dict(run_name=weight['run_name'], group=weight['group'], seed=weight['seed'], condition=condition)
            conditions.append(dict(**info, accuracy=metrics['accuracy'], macro_f1=metrics['macro_f1'], count=metrics['count']))
            per_class.extend(dict(**info, class_name=name, **metrics['classification_report'][name]) for name in CLASSES)
            predictions.extend(dict(**info, **row) for row in predicted)
        del batches
        print('Test condition complete:', condition, flush=True)
    rows = []
    for weight in plan['weights']:
        metrics = results[weight['run_name']]
        clean = metrics['original_white']
        f1 = {name:entry['macro_f1'] for name,entry in metrics.items()}
        brightness = statistics.mean(value for name,value in f1.items() if name.startswith('brightness_'))
        blur = statistics.mean(value for name,value in f1.items() if name.startswith('blur_'))
        cm = clean['confusion_matrix']
        rows.append(dict(**weight, clean_accuracy=clean['accuracy'], clean_f1=clean['macro_f1'],
                         geometry_f1=statistics.mean(f1[name] for name in plan['geometry_conditions']),
                         light_blur_f1=(brightness+blur)/2, plastic_recall=clean['classification_report']['plastic']['recall'],
                         glass_recall=clean['classification_report']['glass']['recall'],
                         plastic_as_glass=cm[4][1], glass_as_plastic=cm[1][4], clean_confusion_matrix=cm))
    write_results(plan, rows, conditions, per_class, predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze','evaluate'])
    args = parser.parse_args()
    plan = build_plan()
    if LOCK.exists() and read(LOCK) != plan:
        raise RuntimeError('Protocol differs from the frozen plan')
    if args.stage == 'freeze':
        if not LOCK.exists():
            save(LOCK, plan)
        print('Locked 12 checkpoints, 380 test images, and 14 conditions using metadata only.', flush=True)
        return
    if not LOCK.exists():
        raise RuntimeError('Run freeze before evaluating test images')
    existing = OUT/'results.json'
    if existing.exists():
        previous = read(existing)
        if previous['plan_sha256'] != digest(LOCK) or any(digest(OUT/name) != value for name,value in previous['artifact_sha256'].items()):
            raise RuntimeError('Saved test results differ from their manifest')
        print('Verified saved test results.', flush=True)
        return
    run_test(plan)


if __name__ == '__main__':
    main()
