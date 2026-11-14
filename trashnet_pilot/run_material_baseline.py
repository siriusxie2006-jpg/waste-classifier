"""Adapt and verify a six-way ResNet18 baseline for ordinary single-item photos."""
import argparse
from copy import deepcopy
from statistics import mean

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

from diagnose_domain_inputs import ROOT, digest, read, read_csv, save, write_csv
from inference import load_run
from model_utils import load_checkpoint, training_class_weights
from train_baseline import CLASSES, TrashDataset, make_transform, set_seed, torch, nn, DataLoader, T


OUT = ROOT/'outputs/material_baseline'
TRASH = ROOT/'data/split_manifest.csv'
BOTTLES = ROOT/'data/bottle_material/manifest.csv'
BASES = ROOT/'outputs/training_coverage/validation_results.json'
REAL_CACHE = ROOT/'outputs/domain_input_diagnosis/preparation.json'
REAL_SOURCE = ROOT/'data/external_realwaste/manifest.csv'
PLAN, SELECTION = OUT/'plan.json', OUT/'selection.json'
RECIPES = {'in_domain':0, 'mixed_12_5':16, 'mixed_25':32}
BOTTLE_LABELS = [CLASSES.index(name) for name in ('glass', 'plastic')]
STEPS, BATCH, LR = 120, 128, .0005


class BottleDataset:
    def __init__(self, fold, transform):
        self.rows = [row for row in read_csv(BOTTLES) if row['split'] == fold]
        self.transform = transform
        for row in self.rows:
            if digest(ROOT/row['path']) != row['file_sha256']:
                raise RuntimeError('Bottle image changed: '+row['path'])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT/row['path']) as image:
            return self.transform(image.convert('RGB')), CLASSES.index(row['class_name']), row['path']


class RealWasteFrames:
    def __init__(self, source, cache, normalize):
        self.frames = np.load(ROOT/cache['frames'], mmap_mode='r')
        self.rows = read_csv(ROOT/cache['manifest'])
        self.normalize = normalize
        fields = ('path', 'label', 'class_name')
        expected = [tuple(row[key] for key in fields) for row in read_csv(source)]
        found = [tuple(row[key] for key in fields) for row in self.rows]
        if found != expected or len(found) != len(self.frames):
            raise RuntimeError('RealWaste frame order differs from the source manifest')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        frame = Image.fromarray(np.asarray(self.frames[index]))
        return self.normalize(frame), int(row['label']), row['path']


def freeze():
    bases = [row for row in read(BASES)['rows'] if row['recipe']=='control' and row['coverage']=='original']
    if sorted(row['seed'] for row in bases) != [42,43,44]:
        raise RuntimeError('Expected three original ResNet18 runs')
    cache = read(REAL_CACHE)['datasets']['realwaste']
    provenance = BOTTLES.with_name('provenance.json')
    if read(provenance)['manifest_sha256'] != digest(BOTTLES):
        raise RuntimeError('Bottle manifest does not match its acquisition record')
    files = {path.relative_to(ROOT).as_posix():digest(path) for path in (
        TRASH, BOTTLES, provenance, BASES, REAL_SOURCE, REAL_CACHE, ROOT/'inference.py',
        ROOT/'model_utils.py', ROOT/'train_baseline.py', ROOT/'image_framing.py',
        ROOT/'prepare_bottle_material.py', ROOT/'run_material_baseline.py')}
    files.update({cache['manifest']:cache['manifest_sha256'], cache['frames']:cache['frames_sha256']})
    for row in bases:
        directory = f"outputs/{row['run_name']}"
        files[f'{directory}/best_model.pt'] = row['checkpoint_sha256']
        predictions = f'{directory}/val_predictions.csv'
        files[predictions] = digest(ROOT/predictions)
    plan = dict(files=files, bases=bases, classes=CLASSES, realwaste_cache=cache,
                recipe_realwaste_samples_per_batch=RECIPES, steps=STEPS, batch=BATCH, lr=LR,
                weight_decay=.0001, distillation_weight=.5,
                shared_batch_description='First 96 clean TrashNet train features shared per step. '
                    'Same step count, inverse class weights and original-logit penalty in all adaptations.',
                development='TACO bottle development, evaluated through all six outputs',
                validation='Clean TrashNet validation (375). Historical TrashNet test is excluded.',
                selection=dict(min_development_gain=.02, min_gain_over_in_domain=.01,
                               max_validation_f1_loss=.01, positive_seeds=2,
                               tie_break='Highest mean development balanced recall; ties prefer fewer external samples'),
                holdout=dict(min_balanced_recall_gain=.02, positive_seeds=2,
                             checks='Mean recall for each observed label must be at least as high as original. '
                             'Same recipes and checkpoints for all rows. No reselection.'))
    if PLAN.exists():
        if read(PLAN) != plan:
            raise RuntimeError('The frozen material plan differs')
    else:
        save(PLAN,plan)


def verify():
    plan = read(PLAN)
    for path, expected in plan['files'].items():
        if digest(ROOT/path) != expected:
            raise RuntimeError('Pinned input changed: '+path)
    return plan


def features(backbone, dataset, device):
    inputs = DataLoader(dataset,batch_size=32,shuffle=False,pin_memory=device=='cuda')
    vectors, truth, paths = [], [], []
    with torch.no_grad():
        for x, y, paths_batch in inputs:
            vectors.append(backbone(x.to(device)))
            truth.append(y)
            paths.extend(paths_batch)
    return torch.cat(vectors), torch.cat(truth).to(device), paths


def metrics(truth, prediction, bottle):
    labels = BOTTLE_LABELS if bottle else list(range(len(CLASSES)))
    matrix = confusion_matrix(truth, prediction, labels=list(range(len(CLASSES))))
    recalls = {CLASSES[i]:float(matrix[i,i]/matrix[i].sum()) for i in labels}
    return dict(count=len(truth), accuracy=float(np.mean(truth==prediction)), recall=recalls,
                balanced_recall=mean(recalls.values()),
                macro_f1=float(f1_score(truth,prediction,labels=labels,average='macro',zero_division=0)),
                confusion_matrix=matrix.tolist())


@torch.no_grad()
def score(model, data, bottle=False):
    inputs, truth, paths = data
    probs = torch.cat([model(x).softmax(1).cpu() for x in inputs.split(32)]).numpy()
    prediction = probs.argmax(1)
    truth = truth.cpu().numpy()
    rows = [dict(path=path,true_class=CLASSES[y],predicted_class=CLASSES[p],
                 **{f'p_{name}':float(value) for name,value in zip(CLASSES,values)})
            for path,y,p,values in zip(paths,truth,prediction,probs)]
    return metrics(truth,prediction,bottle), rows


def adapt(original, train, real, seed, external, weights):
    head = deepcopy(original).train()
    optimizer = torch.optim.AdamW(head.parameters(),lr=LR,weight_decay=.0001)
    train_x, train_y, _ = train
    extra_x, extra_y, _ = real if external else train
    generator = torch.Generator().manual_seed(seed)
    extra_generator = torch.Generator().manual_seed(seed+1000)
    shared_ids = torch.randint(len(train_y),(STEPS,BATCH),generator=generator,device='cpu')
    extra_ids = torch.randint(len(extra_y),(STEPS,32),generator=extra_generator,device='cpu')
    for source, extra in zip(shared_ids,extra_ids):
        source = source[:BATCH-external].to(train_y.device)
        extra = extra[:external].to(train_y.device)
        original_inputs = train_x[source]
        x = torch.cat((original_inputs,extra_x[extra]))
        y = torch.cat((train_y[source],extra_y[extra]))
        logits = head(x)
        with torch.no_grad():
            previous = original(original_inputs).softmax(1)
        retention = nn.functional.kl_div(logits[:len(source)].log_softmax(1),previous,reduction='batchmean')
        loss = nn.functional.cross_entropy(logits,y,weight=weights)+.5*retention
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval()


def save_variant(backbone, head, checkpoint, recipe, seed):
    path = OUT/'weights'/f'{recipe}_seed{seed}.pt'
    path.parent.mkdir(parents=True,exist_ok=True)
    state = {name:value.detach().cpu() for name,value in backbone.state_dict().items()}
    state.update({f'fc.{name}':value.detach().cpu() for name,value in head.state_dict().items()})
    config = dict(checkpoint['config'], material_plan_sha256=digest(PLAN), material_recipe=recipe)
    torch.save(dict(model_state_dict=state, classes=CLASSES,config=config),path)
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=digest(path))


def development(plan, device):
    if SELECTION.exists():
        raise RuntimeError('The selection is locked; run holdout to evaluate it')
    transform = make_transform(framing='fit')
    datasets = dict(validation=TrashDataset('val',transform),development=BottleDataset('development',transform))
    train = TrashDataset('train',transform)
    real = RealWasteFrames(REAL_SOURCE,plan['realwaste_cache'],T.Compose(transform.transforms[-2:]))
    weights = training_class_weights(read_csv(TRASH),'inverse').to(device)
    rows = []
    for base in plan['bases']:
        seed = base['seed']
        print(f'Extracting fixed features for seed {seed}',flush=True)
        backbone, checkpoint = load_run(base['run_name'],device)
        original = backbone.fc
        backbone.fc = nn.Identity()
        evaluated = {name:features(backbone,dataset,device) for name,dataset in datasets.items()}
        learned = {}
        for name,data in evaluated.items():
            result,predictions = score(original,data,bottle=name=='development')
            learned[name] = result
            write_csv(OUT/f'original_seed{seed}_{name}.csv',predictions)
        historical = read_csv(ROOT/'outputs'/base['run_name']/'val_predictions.csv')
        reproduced = read_csv(OUT/f'original_seed{seed}_validation.csv')
        keys = ('path','true_class','predicted_class')
        if [tuple(r[key] for key in keys) for r in historical] != [tuple(r[key] for key in keys) for r in reproduced]:
            raise RuntimeError('Original validation predictions did not reproduce')
        rows.append(dict(seed=seed,recipe='original',**learned,weight=dict(
            path=f"outputs/{base['run_name']}/best_model.pt",sha256=base['checkpoint_sha256'])))
        print(f"Original seed {seed} bottle development balanced recall {learned['development']['balanced_recall']:.3f}",flush=True)
        train_features = features(backbone,train,device)
        real_features = features(backbone,real,device)
        for recipe, external in RECIPES.items():
            updated = adapt(original,train_features,real_features,seed,external,weights)
            learned = {}
            for name,data in evaluated.items():
                result,predictions = score(updated,data,bottle=name=='development')
                learned[name] = result
                write_csv(OUT/f'{recipe}_seed{seed}_{name}.csv',predictions)
            weight = save_variant(backbone,updated,checkpoint,recipe,seed)
            rows.append(dict(seed=seed,recipe=recipe,**learned,weight=weight))
            print(f"{recipe} seed {seed} val F1 {learned['validation']['macro_f1']:.3f}, "
                  f"bottle development balanced recall {learned['development']['balanced_recall']:.3f}",flush=True)
        save(OUT/'development.json',dict(plan_sha256=digest(PLAN),rows=rows))
    lock_selection(plan,rows)


def lock_selection(plan, rows):
    recipes = {recipe:[r for r in rows if r['recipe']==recipe] for recipe in ('original',*RECIPES)}
    original = recipes['original']
    rules = plan['selection']
    decisions = {}
    for recipe, runs in recipes.items():
        gains = [r['development']['balanced_recall']-b['development']['balanced_recall'] for r,b in zip(runs,original)]
        matched = [r['development']['balanced_recall']-b['development']['balanced_recall'] for r,b in zip(runs,recipes['in_domain'])]
        val_change = mean(r['validation']['macro_f1']-b['validation']['macro_f1'] for r,b in zip(runs,original))
        accepted = (RECIPES.get(recipe,0)>0 and mean(gains)>=rules['min_development_gain']
                    and mean(matched)>=rules['min_gain_over_in_domain']
                    and val_change>=-rules['max_validation_f1_loss']
                    and sum(g>0 for g in gains)>=rules['positive_seeds'])
        decisions[recipe] = dict(accepted_for_holdout=accepted,bottle_dev_change=mean(gains),
                                 bottle_dev_change_by_seed=gains,bottle_dev_change_over_in_domain=mean(matched),
                                 trashnet_val_f1_change=val_change)
    eligible = [recipe for recipe in RECIPES if decisions[recipe]['accepted_for_holdout']]
    winner = max(eligible,key=lambda recipe:(decisions[recipe]['bottle_dev_change'],-RECIPES[recipe])) if eligible else None
    tested = {'original','in_domain',winner} if winner else {'original'}
    weights = [dict(seed=r['seed'],recipe=r['recipe'],**r['weight']) for r in rows if r['recipe'] in tested]
    save(SELECTION,dict(plan_sha256=digest(PLAN),development_sha256=digest(OUT/'development.json'),
                        candidate=winner,decisions=decisions,weights=weights))
    print('Locked candidate:',winner,flush=True)


def holdout(plan, device):
    selection = read(SELECTION)
    if selection['plan_sha256'] != digest(PLAN) or selection['development_sha256'] != digest(OUT/'development.json'):
        raise RuntimeError('Locked plan or development scores changed')
    for row in selection['weights']:
        if digest(ROOT/row['path']) != row['sha256']:
            raise RuntimeError('Locked weight changed: '+row['path'])
    dataset = BottleDataset('holdout',make_transform(framing='fit'))
    inputs, labels, paths = next(iter(DataLoader(dataset,batch_size=len(dataset),shuffle=False)))
    data = inputs.to(device),labels.to(device),paths
    rows = []
    for row in selection['weights']:
        model,_ = load_checkpoint(ROOT/row['path'],device,TRASH)
        result,predictions = score(model,data,bottle=True)
        write_csv(OUT/f"{row['recipe']}_seed{row['seed']}_holdout.csv",predictions)
        rows.append(dict(seed=row['seed'],recipe=row['recipe'],**result))
    candidate = selection['candidate']
    original = [r for r in rows if r['recipe']=='original']
    compared = [r for r in rows if r['recipe']==candidate]
    gains = [r['balanced_recall']-b['balanced_recall'] for r,b in zip(compared,original)]
    label_gains = {name:mean(r['recall'][name]-b['recall'][name] for r,b in zip(compared,original))
                   for name in original[0]['recall']} if candidate else {}
    rules = plan['holdout']
    passed = (bool(gains) and mean(gains)>=rules['min_balanced_recall_gain']
              and sum(g>0 for g in gains)>=rules['positive_seeds']
              and all(g>=-1e-12 for g in label_gains.values()))
    save(OUT/'holdout.json',dict(selection_sha256=digest(SELECTION),candidate=candidate,
                                passed=passed,bottle_balanced_recall_change_by_seed=gains,
                                mean_recall_change_by_label=label_gains,rows=rows))
    print(f'Holdout confirmation: {passed}. Paired balanced recall changes: {gains}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze','development','holdout'))
    stage = parser.parse_args().stage
    if stage == 'freeze':
        freeze()
        return
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plan = verify()
    if stage == 'development':
        development(plan,device)
    else:
        holdout(plan,device)


if __name__ == '__main__':
    main()
