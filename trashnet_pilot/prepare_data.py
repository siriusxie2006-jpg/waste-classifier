"""Download official TrashNet images, audit them, and freeze a grouped split."""
import argparse
import csv
import hashlib
import json
import random
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
URL = 'https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip'
EXPECTED_ARCHIVE_SHA256 = '0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2'


def download(url, path):
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, tmp.open('wb') as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            tmp.replace(path)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def phash(im):
    gray = np.asarray(im.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    block = dctn(gray, norm='ortho')[:8, :8].reshape(-1)[1:]
    bits = block > np.median(block)
    return sum(int(v) << i for i, v in enumerate(bits))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    started = time.perf_counter()
    outputs = ROOT / 'outputs'
    outputs.mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'data' / 'downloads' / 'dataset-resized.zip'
    print('Downloading official resized archive...', flush=True)
    download(URL, archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != EXPECTED_ARCHIVE_SHA256:
        raise ValueError('Source archive differs from the audited pilot dataset; refusing to silently change data')
    source = ROOT / 'data' / 'source'
    source.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            dest = (source / entry.filename).resolve()
            if not dest.is_relative_to(source.resolve()):
                raise ValueError('Unsafe archive path')
        z.extractall(source)
    records, hashes, pixel_hashes = [], [], []
    for label, name in enumerate(CLASSES):
        for path in sorted((source / 'dataset-resized' / name).glob('*.jpg')):
            with Image.open(path) as raw:
                raw.load()
                im = raw.convert('RGB')
                digest = hashlib.sha256(im.size[0].to_bytes(4, 'big') + im.size[1].to_bytes(4, 'big') + im.tobytes()).hexdigest()
                hashes.append(phash(im))
                pixel_hashes.append(digest)
                records.append(dict(path=path.relative_to(ROOT).as_posix(), class_name=name, label=label,
                                    width=im.width, height=im.height, pixel_sha256=digest))
    if len(records) != 2527:
        raise ValueError(f'Expected 2527 images, got {len(records)}')
    original_count = len(records)
    pixel_labels = defaultdict(set)
    for r in records:
        pixel_labels[r['pixel_sha256']].add(r['label'])
    conflicting = {h for h, labels in pixel_labels.items() if len(labels) > 1}
    excluded = [r for r in records if r['pixel_sha256'] in conflicting]
    keep = [i for i, r in enumerate(records) if r['pixel_sha256'] not in conflicting]
    records = [records[i] for i in keep]
    hashes = [hashes[i] for i in keep]
    pixel_hashes = [pixel_hashes[i] for i in keep]
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    exact, near = [], []
    # Conservative grouping: same pixels or pHash Hamming distance <= 6.
    # Similarity is NOT proof of the same physical object. No images are deleted.
    for i in range(len(records)):
        for j in range(i):
            if pixel_hashes[i] == pixel_hashes[j]:
                exact.append([j, i])
                union(i, j)
            elif (hashes[i] ^ hashes[j]).bit_count() <= 6:
                near.append([j, i])
                union(i, j)
    groups = np.array([find(i) for i in range(len(records))])
    y = np.array([r['label'] for r in records])
    target = np.array([0.70, 0.15, 0.15])
    class_counts = np.bincount(y, minlength=6)
    best = None
    # Select only for class balance, never using model scores.
    for k in range(300):
        seed = args.seed + k
        trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=seed).split(y, y, groups))
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.15/.85, random_state=seed + 10000).split(trainval, y[trainval], groups[trainval]))
        train, val = trainval[tr], trainval[va]
        parts = [train, val, test]
        counts = np.array([np.bincount(y[idx], minlength=6) for idx in parts])
        if np.any(counts == 0):
            continue
        score = float(np.square(counts / class_counts - target[:, None]).sum())
        if best is None or score < best[0]:
            best = (score, seed, parts)
    if best is None:
        raise ValueError('Could not form balanced grouped split')
    for name, idx in zip(['train', 'val', 'test'], best[2]):
        for i in idx:
            records[i]['split'] = name
            records[i]['group_id'] = int(groups[i])
    group_splits = defaultdict(set)
    for r in records:
        group_splits[r['group_id']].add(r['split'])
    assert all(len(v) == 1 for v in group_splits.values())
    manifest = ROOT / 'data' / 'split_manifest.csv'
    if manifest.exists():
        old = list(csv.DictReader(manifest.open(encoding='utf-8', newline='')))
        keys = ['path', 'split', 'label', 'pixel_sha256', 'group_id']
        if [[str(r[k]) for k in keys] for r in old] != [[str(r[k]) for k in keys] for r in records]:
            raise RuntimeError('Existing split differs; refusing to replace a frozen split')
    with manifest.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {name: dict(Counter(r['class_name'] for r in records if r['split'] == name)) for name in ['train', 'val', 'test']}
    with (outputs / 'split_counts.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'train', 'val', 'test', 'total'])
        for c in CLASSES:
            vals = [summary[s].get(c, 0) for s in ['train', 'val', 'test']]
            writer.writerow([c, *vals, sum(vals)])
    pairs = [{'a': records[a]['path'], 'b': records[b]['path'], 'same_class': records[a]['label'] == records[b]['label'],
              'phash_distance': (hashes[a] ^ hashes[b]).bit_count(), 'split': records[a]['split']} for a, b in exact + near]
    audit = dict(source_images=original_count, total_images=len(records), classes=CLASSES, counts=summary,
                 excluded_conflicting_labels=excluded,
                 exclusion_rule='Exclude all copies of identical RGB pixels assigned different class labels; no subjective relabeling.',
                 archive_url=URL, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                 split_seed=args.seed, selected_balance_seed=best[1],
                 exact_duplicate_pairs=len(exact), perceptual_candidate_pairs=len(near),
                 group_count=len(set(groups)), largest_group=max(Counter(groups).values()),
                 grouping_rule='After excluding pixel-identical label conflicts, union connected components with identical RGB pixels or 63-bit pHash distance <=6, including cross-label candidates; no further images removed.',
                 limitation='Conservative similarity grouping reduces leakage but cannot establish physical-object identity or guarantee all near duplicates are found.',
                 candidate_pairs=pairs, preparation_seconds=time.perf_counter()-started)
    (outputs / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    # A fixed training-only contact sheet.
    rng = random.Random(args.seed)
    sheet = Image.new('RGB', (6 * 170, 4 * 150), 'white')
    draw = ImageDraw.Draw(sheet)
    for col, name in enumerate(CLASSES):
        candidates = [r for r in records if r['class_name'] == name and r['split'] == 'train']
        for row, r in enumerate(rng.sample(candidates, 4)):
            with Image.open(ROOT / r['path']) as im:
                thumb = ImageOps.contain(im.convert('RGB'), (164, 122))
                sheet.paste(thumb, (col * 170, row * 150 + 20))
            draw.text((col * 170 + 4, row * 150 + 3), name, fill='black')
    sheet.save(outputs / 'training_samples.png')
    if pairs:
        shown = pairs[:40]
        contact = Image.new('RGB', (520, len(shown) * 115), 'white')
        draw = ImageDraw.Draw(contact)
        for row, p in enumerate(shown):
            for col, key in enumerate(['a', 'b']):
                with Image.open(ROOT / p[key]) as im:
                    contact.paste(ImageOps.contain(im.convert('RGB'), (245, 90)), (col * 260, row * 115 + 20))
            draw.text((3, row * 115 + 2), f"distance={p['phash_distance']} same_class={p['same_class']} split={p['split']}", fill='black')
        contact.save(outputs / 'similarity_candidates.png')
    print(json.dumps({k:v for k,v in audit.items() if k != 'candidate_pairs'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
