"""Populate the pilot's data directory from the repository's Clean V1 images."""
import csv
import hashlib
from collections import Counter
from pathlib import Path
from shutil import copyfile


PILOT = Path(__file__).resolve().parent
REPO = PILOT.parent
SOURCE_PREFIX = Path('data/source/dataset-resized')


def read_manifest(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pilot_rows = read_manifest(PILOT / 'data/split_manifest.csv')
    shared_rows = read_manifest(REPO / 'data/manifests/split_manifest.csv')
    shared = {row['relative_path']: row for row in shared_rows}
    if len(shared) != len(shared_rows):
        raise ValueError('Duplicate image paths in the shared manifest')
    if len(pilot_rows) != len(shared):
        raise ValueError('The local and shared manifests have different image counts')
    split_counts = Counter()
    seen = set()
    for row in pilot_rows:
        path = Path(row['path'])
        image_id = path.relative_to(SOURCE_PREFIX).as_posix()
        if image_id in seen:
            raise ValueError(f'Duplicate image path in the pilot manifest: {image_id}')
        seen.add(image_id)
        entry = shared[image_id]
        if entry['class'] != row['class_name']:
            raise ValueError(f'Class labels differ for {image_id}')
        expected = entry['sha256'].lower()
        original = (REPO / entry['split_relative_path']).resolve()
        target = (PILOT / path).resolve()
        if not original.is_relative_to(REPO / 'data') or not target.is_relative_to(PILOT / SOURCE_PREFIX):
            raise ValueError(f'Image path is outside its expected data directory: {image_id}')
        if sha256(original) != expected:
            raise ValueError(f'Shared image hash differs for {image_id}')
        if target.exists():
            if sha256(target) != expected:
                raise ValueError(f'Existing local image hash differs for {image_id}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.jpg.tmp')
            copyfile(original, temporary)
            temporary.replace(target)
        split_counts[row['split']] += 1
    print(f'Checked {len(seen)} images. Pilot split: {dict(split_counts)}')


if __name__ == '__main__':
    main()
