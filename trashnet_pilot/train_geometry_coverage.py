"""Train the fixed two-architecture, three-seed whole-image coverage study."""
import argparse
from collections import Counter
import random
import subprocess
import sys
import time

from PIL import Image, ImageOps
from diagnose_domain_inputs import ROOT, digest, read, read_csv, save, write_csv
from domain_probes import CONDITIONS, MEAN_RGB
from train_baseline import (CLASSES, T, build_model, build_optimizer, evaluate, loader,
                            make_transform, nn, save_evaluation, set_seed, set_training_phase,
                            torch, torchvision, training_class_weights)

OUT = ROOT/'outputs/training_coverage'
PLAN = OUT/'plan.json'
MANIFEST = ROOT/'data/split_manifest.csv'
REFERENCE = ROOT/'outputs/fixed_fusion_confirmation_results.json'
REFERENCE_ARTIFACTS = ROOT/'outputs/fixed_fusion_confirmation_artifacts.json'
PRETRAINED = ROOT/'cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
SOURCES = ['train_geometry_coverage.py', 'train_baseline.py', 'model_utils.py',
           'multiscale_model.py', 'capacity_control.py', 'image_framing.py',
           'domain_probes.py', 'diagnose_domain_inputs.py', 'external_metrics.py', 'preview_robustness.py']
POLICY = dict(canvas_colors=[[255, 255, 255], list(MEAN_RGB)], rotation_probability=.5,
              rotation_degrees=[90, 180, 270], shrink_probability=.5, scale_range=[.75, 1.0])


class WholeImageCoverage:
    def __call__(self, image):
        color = random.choice(POLICY['canvas_colors'])
        resized = ImageOps.contain(image.convert('RGB'), (224, 224), method=Image.Resampling.BILINEAR)
        frame = Image.new('RGB', (224, 224), tuple(color))
        frame.paste(resized, ((224-resized.width)//2, (224-resized.height)//2))
        if random.random() < POLICY['rotation_probability']:
            frame = frame.transpose(random.choice([Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270]))
        if random.random() < POLICY['shrink_probability']:
            side = round(224*random.uniform(*POLICY['scale_range']))
            canvas = Image.new('RGB', (224, 224), tuple(color))
            offset = (224-side)//2
            canvas.paste(frame.resize((side, side), Image.Resampling.BILINEAR), (offset, offset))
            frame = canvas
        return frame


def freeze():
    reference, artifacts = read(REFERENCE), read(REFERENCE_ARTIFACTS)
    if digest(REFERENCE) != artifacts['results_sha256'] or digest(MANIFEST) != reference['plan']['split_manifest_sha256']:
        raise RuntimeError('Reference result or split differs')
    common = reference['plan']['common']
    runs = []
    for row in reference['rows']:
        if row['recipe'] not in ('control', 'fixed'):
            continue
        old = row['run_name']
        config = read(ROOT/'outputs'/old/'config.json')
        if any(config[key] != value for key, value in common.items()):
            raise RuntimeError('Reference settings differ: '+old)
        runs.append(dict(recipe=row['recipe'], seed=row['seed'], reference=old,
                         run_name=f"coverage_{row['recipe']}_seed{row['seed']}", architecture=config['architecture']))
    if len(runs) != 6 or {(r['recipe'], r['seed']) for r in runs} != {(a,s) for a in ('control','fixed') for s in (42,43,44)}:
        raise RuntimeError('Expected the six paired references')
    for path, expected in artifacts['model_artifact_sha256'].items():
        if digest(ROOT/path) != expected:
            raise RuntimeError('Reference artifact changed: '+path)
    records = read_csv(MANIFEST)
    train_counts = Counter(row['class_name'] for row in records if row['split'] == 'train')
    if any(CLASSES[int(row['label'])] != row['class_name'] for row in records):
        raise RuntimeError('Class order differs from the manifest')
    examples = [row['path'] for row in read_csv(ROOT/'outputs/domain_input_diagnosis/dataset_examples.csv')
                if row['group'] == 'train' and row['class_name'] == 'trash']
    plan = dict(common=common, runs=runs, geometry_policy=POLICY, class_counts=dict(train_counts),
                label_boundary_examples=examples,
                split_manifest_sha256=digest(MANIFEST), pretrained_sha256=digest(PRETRAINED),
                reference_results_sha256=digest(REFERENCE), reference_artifacts_sha256=digest(REFERENCE_ARTIFACTS),
                source_sha256={name:digest(ROOT/name) for name in SOURCES},
                geometry_conditions=CONDITIONS, robustness_conditions=reference['plan']['conditions'],
                coverage_rule=dict(min_geometry_gain=.02, min_positive_seeds=2, max_clean_loss=.01,
                                   max_robustness_loss=.01, max_plastic_recall_loss=.02),
                fusion_rule=dict(min_geometry_gain=.005, min_positive_seeds=2, max_clean_loss=.01,
                                 max_plastic_recall_loss=.02),
                decision_split='val', validation_count=375, test_evaluated=False)
    if PLAN.exists() and read(PLAN) != plan:
        raise RuntimeError('Existing training plan differs')
    if not PLAN.exists():
        save(PLAN, plan)
    snapshot = ROOT/'sources/training_versions/geometry_coverage_20261111'
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        (snapshot/name).write_bytes((ROOT/name).read_bytes())
    save(snapshot/'fingerprints.json', plan['source_sha256'])
    print('Frozen six paired runs. Train counts:', dict(train_counts), flush=True)
    return plan


def train(run_name):
    plan = read(PLAN)
    record = next(row for row in plan['runs'] if row['run_name'] == run_name)
    if any(digest(ROOT/name) != expected for name, expected in plan['source_sha256'].items()):
        raise RuntimeError('Training sources differ from the frozen plan')
    if digest(MANIFEST) != plan['split_manifest_sha256'] or digest(PRETRAINED) != plan['pretrained_sha256']:
        raise RuntimeError('Split or initialization changed')
    config = dict(plan['common'], **record, geometry_policy=plan['geometry_policy'],
                  split_manifest_sha256=plan['split_manifest_sha256'], plan_sha256=digest(PLAN),
                  selection_metric='validation macro-F1', test_evaluated=False,
                  python=sys.version, torch=str(torch.__version__), torchvision=str(torchvision.__version__))
    out = ROOT/'outputs'/run_name
    out.mkdir(exist_ok=False)
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.update(device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None)
    model = build_model(config['head'], pretrained=True, architecture=config['architecture'])
    set_training_phase(model, 'head')
    model.to(device)
    train_data = loader('train', config['batch_size'], config['workers'], config['seed'], config['augmentation'], config['framing'])
    original = make_transform(True, config['augmentation'], config['framing'])
    train_data.dataset.transform = T.Compose([WholeImageCoverage(), *original.transforms[1:]])
    val_data = loader('val', config['batch_size'], config['workers'], config['seed'], framing=config['framing'])
    weights = training_class_weights(train_data.dataset.records, config['class_weighting']).to(device)
    config['class_weights'] = weights.cpu().tolist()
    save(out/'config.json', config)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    history, best_f1, best_epoch = [], -1., 0
    checkpoint = out/'best_model.pt'
    started = time.perf_counter()
    for scope, count in [('head', config['warmup_epochs']), (config['finetune_scope'], config['finetune_epochs'])]:
        if scope != 'head':
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['model_state_dict'])
        set_training_phase(model, scope)
        optimizer = build_optimizer(model, config['head_lr'], config['backbone_lr'], config['weight_decay'])
        for _ in range(count):
            epoch = len(history)+1
            t0 = time.perf_counter()
            set_training_phase(model, scope)
            correct, total, loss_sum, weight_sum = 0, 0, 0., 0.
            for x, y, _ in train_data:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, y, weight=weights)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)
                denominator = weights[y].sum().item()
                loss_sum += loss.item()*denominator
                weight_sum += denominator
            metrics, _ = evaluate(model, val_data, device)
            row = dict(epoch=epoch, phase=scope, train_loss=loss_sum/weight_sum, train_accuracy=correct/total,
                       val_loss=metrics['loss'], val_accuracy=metrics['accuracy'], val_macro_f1=metrics['macro_f1'],
                       seconds=time.perf_counter()-t0)
            history.append(row)
            write_csv(out/'history.csv', history)
            print(run_name, row, flush=True)
            if metrics['macro_f1'] > best_f1:
                best_f1, best_epoch = metrics['macro_f1'], epoch
                torch.save(dict(model_state_dict=model.state_dict(), classes=CLASSES, epoch=epoch,
                                validation_macro_f1=best_f1, config=config), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['model_state_dict'])
    metrics = save_evaluation(model, val_data, device, out, 'val')
    summary = dict(best_epoch=best_epoch, epochs_completed=len(history), validation_accuracy=metrics['accuracy'],
                   validation_macro_f1=metrics['macro_f1'], total_training_validation_seconds=time.perf_counter()-started,
                   test_evaluated=False, plan_sha256=digest(PLAN))
    save(out/'summary.json', summary)
    print('FINAL', run_name, summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single-run')
    args = parser.parse_args()
    if args.single_run:
        train(args.single_run)
        return
    plan = freeze()
    for record in plan['runs']:
        name = record['run_name']
        summary = ROOT/'outputs'/name/'summary.json'
        if summary.exists():
            existing = read(summary)
            if existing['plan_sha256'] != digest(PLAN) or existing['epochs_completed'] != 10:
                raise RuntimeError('Invalid completed run: '+name)
        else:
            subprocess.run([sys.executable, '-u', __file__, '--single-run', name], check=True)


if __name__ == '__main__':
    main()
