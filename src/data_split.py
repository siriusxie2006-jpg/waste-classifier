"""Create and validate the project's fixed, leakage-aware TrashNet split."""

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn
from sklearn.model_selection import GroupShuffleSplit


CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
PHASH_HAMMING_THRESHOLD = 6
CANDIDATE_SEED_COUNT = 300
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
DEFAULT_SEED = 42
MANIFEST_FIELDS = [
    "relative_path", "class", "label", "width", "height", "file_sha256",
    "pixel_sha256", "phash", "split", "group_id", "random_state",
    "balance_seed", "phash_hamming_threshold", "candidate_seed_count",
    "train_ratio", "val_ratio", "test_ratio",
]


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_hashes(path):
    """Return byte SHA256, RGB-pixel SHA256, pHash, width, and height."""
    with Image.open(path) as raw:
        raw.load()
        image = raw.convert("RGB")
    pixel_bytes = (
        image.width.to_bytes(4, "big")
        + image.height.to_bytes(4, "big")
        + image.tobytes()
    )
    pixel_sha256 = hashlib.sha256(pixel_bytes).hexdigest()
    gray = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
        dtype=float,
    )
    dct_block = dctn(gray, norm="ortho")[:8, :8].reshape(-1)[1:]
    bits = dct_block > np.median(dct_block)
    phash = sum(int(bit) << index for index, bit in enumerate(bits))
    return _sha256_file(path), pixel_sha256, phash, image.width, image.height


def _load_images(raw_dir):
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")
    actual_classes = sorted(path.name for path in raw_dir.iterdir() if path.is_dir())
    if actual_classes != sorted(CLASSES):
        raise ValueError(
            f"Expected exactly the six TrashNet classes {CLASSES}, got {actual_classes}"
        )

    records = []
    for label, class_name in enumerate(CLASSES):
        class_dir = raw_dir / class_name
        for path in sorted(class_dir.glob("*.jpg")):
            file_sha, pixel_sha, phash, width, height = _image_hashes(path)
            records.append({
                "relative_path": path.relative_to(raw_dir).as_posix(),
                "class": class_name,
                "label": label,
                "width": width,
                "height": height,
                "file_sha256": file_sha,
                "pixel_sha256": pixel_sha,
                "phash": f"{phash:016x}",
                "_phash_int": phash,
            })
    if not records:
        raise ValueError(f"No .jpg images found under {raw_dir}")
    return records


def _connected_components(records):
    """Apply the source branch's exact-pixel/pHash union-find grouping."""
    labels_by_pixel = defaultdict(set)
    for record in records:
        labels_by_pixel[record["pixel_sha256"]].add(record["label"])
    conflicting_hashes = {
        pixel_hash for pixel_hash, labels in labels_by_pixel.items()
        if len(labels) > 1
    }
    excluded = [
        record for record in records
        if record["pixel_sha256"] in conflicting_hashes
    ]
    kept = [
        record for record in records
        if record["pixel_sha256"] not in conflicting_hashes
    ]

    parent = list(range(len(kept)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    exact_pairs = []
    near_pairs = []
    for index in range(len(kept)):
        for other in range(index):
            if kept[index]["pixel_sha256"] == kept[other]["pixel_sha256"]:
                exact_pairs.append((other, index))
                union(index, other)
            elif (
                kept[index]["_phash_int"] ^ kept[other]["_phash_int"]
            ).bit_count() <= PHASH_HAMMING_THRESHOLD:
                near_pairs.append((other, index))
                union(index, other)

    groups = np.asarray([find(index) for index in range(len(kept))])
    return kept, excluded, groups, exact_pairs, near_pairs


def _select_grouped_split(records, groups, random_state):
    labels = np.asarray([record["label"] for record in records])
    class_counts = np.bincount(labels, minlength=len(CLASSES))
    target = np.asarray([TRAIN_RATIO, VAL_RATIO, TEST_RATIO])
    best = None

    # This mirrors prepare_data.py: 300 consecutive seeds and two
    # GroupShuffleSplit stages; balance is based only on class frequencies.
    for offset in range(CANDIDATE_SEED_COUNT):
        seed = random_state + offset
        trainval, test = next(GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_RATIO,
            random_state=seed,
        ).split(labels, labels, groups))
        train, val = next(GroupShuffleSplit(
            n_splits=1,
            test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
            random_state=seed + 10000,
        ).split(trainval, labels[trainval], groups[trainval]))
        parts = [trainval[train], trainval[val], test]
        counts = np.asarray([
            np.bincount(labels[indices], minlength=len(CLASSES))
            for indices in parts
        ])
        if np.any(counts == 0):
            continue
        score = float(np.square(counts / class_counts - target[:, None]).sum())
        if best is None or score < best[0]:
            best = (score, seed, parts)

    if best is None:
        raise ValueError("Could not form a balanced Grouped Split")
    split_by_index = {}
    for split_name, indices in zip(("train", "val", "test"), best[2]):
        for index in indices:
            split_by_index[int(index)] = split_name
    for index, record in enumerate(records):
        record["split"] = split_by_index[index]
        record["group_id"] = int(groups[index])
        record["random_state"] = random_state
        record["balance_seed"] = best[1]
        record["phash_hamming_threshold"] = PHASH_HAMMING_THRESHOLD
        record["candidate_seed_count"] = CANDIDATE_SEED_COUNT
        record["train_ratio"] = TRAIN_RATIO
        record["val_ratio"] = VAL_RATIO
        record["test_ratio"] = TEST_RATIO
    return best[1]


def _read_fixed_manifest(manifest_path, records, groups, random_state):
    """Validate/reuse an existing fixed manifest without changing membership."""
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    by_path = {record["relative_path"]: record for record in records}
    if len(by_path) != len(records):
        raise ValueError("Input dataset contains duplicate relative paths")
    if len(rows) != len(records):
        raise ValueError(
            f"Manifest has {len(rows)} rows but input has {len(records)} images"
        )

    verification_records = [dict(record) for record in records]
    selected_seed = _select_grouped_split(
        verification_records, groups, random_state
    )
    expected_assignments = {
        record["relative_path"]: (record["split"], record["group_id"])
        for record in verification_records
    }
    used_paths = set()
    actual_assignments = {}
    for row in rows:
        relpath = row.get("relative_path") or row.get("path", "")
        class_name = row.get("class") or row.get("class_name", "")
        if relpath.startswith("data/source/dataset-resized/"):
            relpath = relpath.removeprefix("data/source/dataset-resized/")
        elif relpath.startswith("data/raw/"):
            relpath = relpath.removeprefix("data/raw/")
        if relpath not in by_path or relpath in used_paths:
            raise ValueError(f"Manifest path is missing or duplicated: {relpath}")
        record = by_path[relpath]
        if class_name != record["class"]:
            raise ValueError(f"Manifest class mismatch for {relpath}")
        if row.get("pixel_sha256", "").lower() != record["pixel_sha256"]:
            raise ValueError(f"Manifest pixel SHA256 mismatch for {relpath}")
        if row.get("phash") and row["phash"].lower() != record["phash"]:
            raise ValueError(f"Manifest pHash mismatch for {relpath}")
        if row.get("file_sha256") and row["file_sha256"].lower() != record["file_sha256"]:
            raise ValueError(f"Manifest file SHA256 mismatch for {relpath}")
        if row.get("sha256") and row["sha256"].lower() != record["file_sha256"]:
            raise ValueError(f"Manifest file SHA256 mismatch for {relpath}")
        split = row.get("split", "").lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split for {relpath}: {split}")
        record["split"] = split
        try:
            record["group_id"] = int(row["group_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing/invalid group_id for {relpath}") from exc
        actual_assignments[relpath] = (split, record["group_id"])
        for key, default in (
            ("random_state", DEFAULT_SEED),
            ("balance_seed", row.get("selected_balance_seed", "55")),
            ("phash_hamming_threshold", PHASH_HAMMING_THRESHOLD),
            ("candidate_seed_count", CANDIDATE_SEED_COUNT),
            ("train_ratio", TRAIN_RATIO),
            ("val_ratio", VAL_RATIO),
            ("test_ratio", TEST_RATIO),
        ):
            record[key] = int(row.get(key) or default) if key in {
                "random_state", "balance_seed", "phash_hamming_threshold",
                "candidate_seed_count",
            } else float(row.get(key) or default)
        used_paths.add(relpath)
    if used_paths != set(by_path):
        raise ValueError("Manifest does not cover the input dataset exactly")
    if actual_assignments != expected_assignments:
        raise ValueError(
            "Fixed manifest does not reproduce from the current Grouped Split "
            "algorithm and seed 42"
        )

    splits_by_group = defaultdict(set)
    for record in records:
        splits_by_group[record["group_id"]].add(record["split"])
    crossing_groups = [group for group, splits in splits_by_group.items() if len(splits) != 1]
    if crossing_groups:
        raise ValueError(f"Groups cross splits: {crossing_groups[:10]}")
    if any(record["balance_seed"] != selected_seed for record in records):
        raise ValueError(
            f"Manifest balance seed differs from recomputed seed {selected_seed}"
        )
    return rows


def _expected_split_files(records, raw_dir):
    raw_dir = Path(raw_dir)
    return {
        split: {
            (record["class"], Path(record["relative_path"]).name): _sha256_file(
                raw_dir / record["relative_path"]
            )
            for record in records if record["split"] == split
        }
        for split in ("train", "val", "test")
    }


def _validate_materialized_split(directory, expected):
    directory = Path(directory)
    if not directory.is_dir():
        return False
    actual = {}
    for class_dir in directory.iterdir():
        if not class_dir.is_dir():
            raise FileExistsError(f"Unexpected file in split directory: {class_dir}")
        for path in class_dir.iterdir():
            if path.is_file():
                actual[(class_dir.name, path.name)] = _sha256_file(path)
    if actual != expected:
        raise FileExistsError(
            f"Existing split directory differs from the fixed manifest: {directory}"
        )
    return True


def _write_csv_exclusive(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_audit_exclusive(path, records, excluded, exact_pairs, near_pairs,
                           original_count, random_state, balance_seed):
    counts = {
        split: dict(Counter(
            record["class"] for record in records if record["split"] == split
        ))
        for split in ("train", "val", "test")
    }
    groups = defaultdict(set)
    pixel_splits = defaultdict(set)
    for record in records:
        groups[record["group_id"]].add(record["split"])
        pixel_splits[record["pixel_sha256"]].add(record["split"])
    audit = {
        "source_images": original_count,
        "total_images": len(records),
        "classes": CLASSES,
        "counts": counts,
        "excluded_conflicting_labels": [
            {key: value for key, value in record.items() if not key.startswith("_")}
            for record in excluded
        ],
        "exclusion_rule": (
            "Exclude all copies of RGB-pixel-identical images with conflicting "
            "folder labels; no labels are adjudicated."
        ),
        "split_seed": random_state,
        "selected_balance_seed": balance_seed,
        "candidate_seed_count": CANDIDATE_SEED_COUNT,
        "phash_hamming_threshold": PHASH_HAMMING_THRESHOLD,
        "exact_duplicate_pairs": len(exact_pairs),
        "perceptual_candidate_pairs": len(near_pairs),
        "group_count": len(groups),
        "cross_split_group_count": sum(len(splits) > 1 for splits in groups.values()),
        "cross_split_pixel_hash_count": sum(
            len(splits) > 1 for splits in pixel_splits.values()
        ),
        "largest_group": max(Counter(record["group_id"] for record in records).values()),
        "grouping_rule": (
            "After excluding pixel-identical label conflicts, connected components "
            "join identical RGB pixels or 63-bit pHash distance <=6."
        ),
        "selection_rule": (
            "Choose the lowest class-distribution squared-error score across 300 "
            "GroupShuffleSplit candidates; no model scores are used."
        ),
    }
    audit_path = Path(path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("x", encoding="utf-8") as output:
        json.dump(audit, output, ensure_ascii=False, indent=2)


def split_dataset(raw_dir, train_dir, val_dir, test_dir, test_size=0.15,
                  val_size=0.15, random_state=42,
                  manifest_path="data/manifests/split_manifest.csv",
                  class_counts_path="data/manifests/class_counts.csv",
                  audit_path="data/manifests/data_audit.json"):
    """Materialize a fixed Grouped Split; never replace existing outputs.

    The original positional interface is retained. If a valid grouped manifest
    already exists, it is reused after verifying every input image hash.
    """
    if (test_size, val_size, random_state) != (TEST_RATIO, VAL_RATIO, DEFAULT_SEED):
        raise ValueError("This project fixes val/test ratios at 0.15 and seed at 42")

    raw_dir = Path(raw_dir)
    output_dirs = {
        "train": Path(train_dir),
        "val": Path(val_dir),
        "test": Path(test_dir),
    }
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        collisions = [
            str(path) for path in (*output_dirs.values(), manifest_path,
                                   Path(class_counts_path), Path(audit_path))
            if path.exists()
        ]
        if collisions:
            raise FileExistsError(
                "Grouped Split outputs already exist; refusing to create a new "
                "manifest or overwrite them: " + ", ".join(collisions)
            )
    original_records = _load_images(raw_dir)
    records, excluded, groups, exact_pairs, near_pairs = _connected_components(
        original_records
    )
    if manifest_path.exists():
        _read_fixed_manifest(manifest_path, records, groups, random_state)
        balance_seed = int(records[0]["balance_seed"])
    else:
        balance_seed = _select_grouped_split(records, groups, random_state)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_rows = [
            {key: record[key] for key in MANIFEST_FIELDS}
            for record in records
        ]
        _write_csv_exclusive(manifest_path, MANIFEST_FIELDS, manifest_rows)

    expected_by_split = _expected_split_files(records, raw_dir)
    existing_dirs = [path.exists() for path in output_dirs.values()]
    if any(existing_dirs):
        if not all(existing_dirs):
            raise FileExistsError(
                "Only some split directories exist; refusing partial overwrite."
            )
        for split, directory in output_dirs.items():
            _validate_materialized_split(directory, expected_by_split[split])
        print("Existing split directories match the fixed Grouped Split manifest.")
    else:
        for split, directory in output_dirs.items():
            directory.parent.mkdir(parents=True, exist_ok=True)
            for (class_name, filename), expected_sha in expected_by_split[split].items():
                source = raw_dir / class_name / filename
                destination = directory / class_name / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if _sha256_file(destination) != expected_sha:
                    raise IOError(f"Copied file hash mismatch: {destination}")

    class_rows = []
    for class_name in CLASSES:
        per_split = {
            split: sum(
                record["class"] == class_name and record["split"] == split
                for record in records
            )
            for split in ("train", "val", "test")
        }
        class_rows.append({
            "class": class_name,
            "train": per_split["train"],
            "validation": per_split["val"],
            "test": per_split["test"],
            "total": sum(per_split.values()),
        })
    if Path(class_counts_path).exists():
        with Path(class_counts_path).open(
            "r", encoding="utf-8-sig", newline=""
        ) as source:
            stored_counts = list(csv.DictReader(source))
        if stored_counts != [
            {key: str(value) for key, value in row.items()}
            for row in class_rows
        ]:
            raise FileExistsError(
                f"Existing class-count report differs from manifest: {class_counts_path}"
            )
    else:
        _write_csv_exclusive(
            class_counts_path,
            ["class", "train", "validation", "test", "total"],
            class_rows,
        )
    if Path(audit_path).exists():
        with Path(audit_path).open("r", encoding="utf-8-sig") as source:
            stored_audit = json.load(source)
        expected_counts = {
            split: {
                class_name: row["train" if split == "train" else
                               "validation" if split == "val" else "test"]
                for row in class_rows for class_name in [row["class"]]
            }
            for split in ("train", "val", "test")
        }
        expected_groups = len({record["group_id"] for record in records})
        if (
            stored_audit.get("total_images") != len(records)
            or stored_audit.get("group_count") != expected_groups
            or stored_audit.get("selected_balance_seed") != balance_seed
            or stored_audit.get("phash_hamming_threshold") != PHASH_HAMMING_THRESHOLD
            or stored_audit.get("counts") != expected_counts
            or stored_audit.get("cross_split_group_count") != 0
            or stored_audit.get("cross_split_pixel_hash_count") != 0
        ):
            raise FileExistsError(
                f"Existing data audit differs from manifest: {audit_path}"
            )
    else:
        _write_audit_exclusive(
            audit_path, records, excluded, exact_pairs, near_pairs,
            len(original_records), random_state, balance_seed,
        )
    print(
        f"Grouped Split ready: {len(records)} images, "
        f"{len(set(record['group_id'] for record in records))} groups, "
        f"balance seed {balance_seed}; manifest: {manifest_path}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--train-dir", default="data/train")
    parser.add_argument("--val-dir", default="data/val")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument(
        "--manifest-path", default="data/manifests/split_manifest.csv"
    )
    parser.add_argument(
        "--class-counts-path", default="data/manifests/class_counts.csv"
    )
    parser.add_argument("--audit-path", default="data/manifests/data_audit.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    split_dataset(
        args.raw_dir, args.train_dir, args.val_dir, args.test_dir,
        random_state=args.seed,
        manifest_path=args.manifest_path,
        class_counts_path=args.class_counts_path,
        audit_path=args.audit_path,
    )


if __name__ == "__main__":
    main()
