"""Check whether a small amount of targeted bottle replay improves the baseline."""
import argparse
from statistics import mean

import numpy as np

import run_material_baseline as baseline
from diagnose_domain_inputs import ROOT, digest, read, read_csv, save, write_csv
from inference import load_run
from model_utils import load_checkpoint, training_class_weights
from train_baseline import CLASSES, TrashDataset, make_transform, set_seed, torch, nn


OUT = baseline.OUT/'bottle_replay'
PLAN, LOCK = OUT/'plan.json', OUT/'selection.json'
GROUPS = (('batch_1','batch_4'), ('batch_3','batch_6','batch_10'), ('batch_5','batch_7','batch_9'))
RECIPES = {'bottle_3_125':4,'bottle_6_25':8}


def freeze():
    previous = baseline.verify()
    previous_selection = read(baseline.SELECTION)
    if previous_selection['candidate'] is not None or (baseline.OUT/'holdout.json').exists():
        raise RuntimeError('Targeted replay was proposed after the first development rejection, before holdout')
    paths = [baseline.PLAN, baseline.SELECTION, baseline.OUT/'development.json', ROOT/'run_bottle_replay.py']
    plan = dict(files={path.relative_to(ROOT).as_posix():digest(path) for path in paths},
                groups=GROUPS, recipe_bottles_per_batch=RECIPES,
                protocol='Same optimizer, 120 steps and batch of 128 as the first trial. '
                         'The replacement positions sample labeled TACO glass and plastic with equal probability. '
                         'Out-of-fold predictions use source-batch groups and exclude near neighbors from training. '
                         'The final checkpoint learns from all development crops, evaluated on TrashNet val.',
                selection=previous['selection'],holdout=previous['holdout'])
    if PLAN.exists():
        raise RuntimeError('Replay plan already exists')
    save(PLAN,plan)


def verify():
    previous = baseline.verify()
    plan = read(PLAN)
    for path, expected in plan['files'].items():
        if digest(ROOT/path) != expected:
            raise RuntimeError('Replay input changed: '+path)
    return previous,plan


def subset(data, indexes):
    x,y,paths = data
    return x[indexes],y[indexes],[paths[i] for i in indexes]


def similar(a,b):
    return (a['source_sha256']==b['source_sha256'] or
            any((int(a[key])^int(b[key])).bit_count()<=4 for key in ('image_phash','crop_phash')))


def folds(records, groups):
    partitions = [[i for i,row in enumerate(records) if row['source_batch'] in group] for group in groups]
    if sorted(i for fold in partitions for i in fold) != list(range(len(records))):
        raise RuntimeError('Development folds must cover each crop exactly once')
    return [(test,[i for i,row in enumerate(records)
                   if i not in test and not any(similar(row,records[j]) for j in test)]) for test in partitions]


def balanced(data, indexes):
    labels = data[1].cpu().tolist()
    groups = [[i for i in indexes if labels[i]==label] for label in baseline.BOTTLE_LABELS]
    if any(not group for group in groups):
        raise RuntimeError('Bottle replay requires both labels')
    # Each crop has the same sampling probability within its class.
    first, second = groups
    return subset(data, first*len(second)+second*len(first))


def development(previous, plan, device):
    if LOCK.exists():
        raise RuntimeError('Replay weights are already locked')
    transform = make_transform(framing='fit')
    dataset = baseline.BottleDataset('development',transform)
    partitions = folds(dataset.rows,plan['groups'])
    train,val = TrashDataset('train',transform),TrashDataset('val',transform)
    weights = training_class_weights(read_csv(baseline.TRASH),'inverse').to(device)
    prior = read(baseline.OUT/'development.json')['rows']
    rows = [r for r in prior if r['recipe'] in ('original','in_domain')]
    for base in previous['bases']:
        seed = base['seed']
        model, checkpoint = load_run(base['run_name'],device)
        original = model.fc
        model.fc = nn.Identity()
        train_x = baseline.features(model,train,device)
        val_x = baseline.features(model,val,device)
        bottle_x = baseline.features(model,dataset,device)
        for recipe, count in plan['recipe_bottles_per_batch'].items():
            predictions = []
            for test,fit in partitions:
                replay = balanced(bottle_x,fit)
                head = baseline.adapt(original,train_x,replay,seed,count,weights)
                _, out_of_fold = baseline.score(head,subset(bottle_x,test),bottle=True)
                predictions.extend(out_of_fold)
            truth = np.asarray([CLASSES.index(r['true_class']) for r in predictions])
            pred = np.asarray([CLASSES.index(r['predicted_class']) for r in predictions])
            dev = baseline.metrics(truth,pred,True)
            final = baseline.adapt(original,train_x,balanced(bottle_x,list(range(len(dataset)))),seed,count,weights)
            val_score,val_predictions = baseline.score(final,val_x)
            write_csv(OUT/f'{recipe}_seed{seed}_out_of_fold.csv',predictions)
            write_csv(OUT/f'{recipe}_seed{seed}_validation.csv',val_predictions)
            path = OUT/f'{recipe}_seed{seed}.pt'
            state = dict(checkpoint['model_state_dict'])
            state.update({f'fc.{name}':value.detach().cpu() for name,value in final.state_dict().items()})
            config = dict(checkpoint['config'],material_plan_sha256=digest(PLAN),material_recipe=recipe)
            torch.save(dict(model_state_dict=state,classes=CLASSES,config=config),path)
            rows.append(dict(seed=seed,recipe=recipe,development=dev,validation=val_score,
                             weight=dict(path=path.relative_to(ROOT).as_posix(),sha256=digest(path))))
            print(f"{recipe} seed {seed}: OOF balanced recall {dev['balanced_recall']:.3f}, val F1 {val_score['macro_f1']:.3f}",flush=True)
    save(OUT/'development.json',dict(plan_sha256=digest(PLAN),rows=rows,
                                    folds=[dict(test=[dataset.rows[i]['path'] for i in test],
                                                fit=[dataset.rows[i]['path'] for i in fit]) for test,fit in partitions]))
    lock(plan,rows)


def lock(plan,rows):
    by_recipe = {recipe:{r['seed']:r for r in rows if r['recipe']==recipe}
                 for recipe in ('original','in_domain',*RECIPES)}
    criteria = plan['selection']
    decisions = {}
    for recipe in RECIPES:
        gains, matched, val = [],[],[]
        for seed,r in by_recipe[recipe].items():
            original,control = by_recipe['original'][seed],by_recipe['in_domain'][seed]
            recall = r['development']['balanced_recall']
            gains.append(recall-original['development']['balanced_recall'])
            matched.append(recall-control['development']['balanced_recall'])
            val.append(r['validation']['macro_f1']-original['validation']['macro_f1'])
        passed = (mean(gains)>=criteria['min_development_gain'] and mean(matched)>=criteria['min_gain_over_in_domain']
                  and mean(val)>=-criteria['max_validation_f1_loss'] and sum(g>0 for g in gains)>=criteria['positive_seeds'])
        decisions[recipe] = dict(accepted_for_holdout=passed,bottle_dev_change=mean(gains),
                                 bottle_dev_change_by_seed=gains,bottle_dev_change_over_in_domain=mean(matched),
                                 trashnet_val_f1_change=mean(val))
    eligible = [recipe for recipe in RECIPES if decisions[recipe]['accepted_for_holdout']]
    winner = max(eligible,key=lambda name:(decisions[name]['bottle_dev_change'],-RECIPES[name])) if eligible else None
    tested = ('original','in_domain',winner) if winner else ('original',)
    weights = [dict(seed=seed,recipe=recipe,**r['weight'])
               for recipe in tested for seed,r in by_recipe[recipe].items()]
    save(LOCK,dict(plan_sha256=digest(PLAN),development_sha256=digest(OUT/'development.json'),
                   candidate=winner,decisions=decisions,weights=weights))
    print('Locked replay candidate:',winner,flush=True)


def holdout(plan,device):
    selection = read(LOCK)
    if selection['plan_sha256']!=digest(PLAN) or selection['development_sha256']!=digest(OUT/'development.json'):
        raise RuntimeError('Locked replay records changed')
    for row in selection['weights']:
        if digest(ROOT/row['path'])!=row['sha256']:
            raise RuntimeError('Locked replay weight changed: '+row['path'])
    dataset = baseline.BottleDataset('holdout',make_transform(framing='fit'))
    batches = baseline.DataLoader(dataset,batch_size=len(dataset),shuffle=False)
    x,y,paths = next(iter(batches))
    rows = []
    for row in selection['weights']:
        model,_ = load_checkpoint(ROOT/row['path'],device,baseline.TRASH)
        result,predictions = baseline.score(model,(x.to(device),y.to(device),paths),bottle=True)
        write_csv(OUT/f"{row['recipe']}_seed{row['seed']}_holdout.csv",predictions)
        rows.append(dict(seed=row['seed'],recipe=row['recipe'],**result))
    original = {r['seed']:r for r in rows if r['recipe']=='original'}
    candidate = selection['candidate']
    compared = [r for r in rows if r['recipe']==candidate]
    gains = [r['balanced_recall']-original[r['seed']]['balanced_recall'] for r in compared]
    label_gains = {name:mean(r['recall'][name]-original[r['seed']]['recall'][name] for r in compared)
                   for name in ('glass','plastic')} if candidate else {}
    rules = plan['holdout']
    passed = (bool(gains) and mean(gains)>=rules['min_balanced_recall_gain']
              and sum(g>0 for g in gains)>=rules['positive_seeds'] and all(g>=-1e-12 for g in label_gains.values()))
    save(OUT/'holdout.json',dict(selection_sha256=digest(LOCK),candidate=candidate,passed=passed,
                                bottle_balanced_recall_change_by_seed=gains,mean_recall_change_by_label=label_gains,rows=rows))
    print(f'Replay holdout confirmation: {passed}. Paired balanced recall changes: {gains}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','development','holdout'))
    stage = parser.parse_args().stage
    if stage=='freeze':
        freeze()
        return
    set_seed(42)
    previous,plan = verify()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if stage=='development':
        development(previous,plan,device)
    else:
        holdout(plan,device)


if __name__=='__main__':
    main()
