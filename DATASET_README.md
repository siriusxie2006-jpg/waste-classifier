# TrashNet Clean V1 with Grouped Split

The formal dataset split in this repository is the Clean V1 version of the six-class TrashNet image dataset. It contains 2,521 retained images from the original 2,527-image collection. The original `data/train`, `data/val`, and `data/test` paths are now materialized from the fixed Grouped Split manifest so that `src/train_transfer.py` can continue to read its existing paths.

## Source, labels, and exclusions

The images originate from Gary Thung and Mindy Yang's [TrashNet project](https://github.com/garythung/trashnet). The source class-folder names are used as labels; no new manual labeling was performed. The upstream project provides an MIT License, retained at [`data/THIRD_PARTY_LICENSES/TRASHNET_MIT_LICENSE.txt`](data/THIRD_PARTY_LICENSES/TRASHNET_MIT_LICENSE.txt). Please cite the original TrashNet project.

Clean V1 excludes the six images listed in [`data/manifests/excluded_files.csv`](data/manifests/excluded_files.csv). They form three pairs of byte-identical files in different class folders. Their correct labels were not adjudicated.

## Grouped Split method

The fixed assignment is in [`data/manifests/split_manifest.csv`](data/manifests/split_manifest.csv); it records each relative path, class, split, byte and RGB-pixel SHA256, pHash, and `group_id`. [`data/manifests/data_audit.json`](data/manifests/data_audit.json) records the pHash parameters and audit summary. Per-class split counts are in [`data/manifests/class_counts.csv`](data/manifests/class_counts.csv).

The split procedure follows the verified implementation from `damnshi/local-baseline`:

1. Detect conflicting labels for pixel-identical RGB images and exclude all copies in each conflict. No label is guessed.
2. Compute pHash from a 32×32 grayscale image's 8×8 DCT block, excluding the DC term.
3. Connect images with identical RGB pixels or pHash Hamming distance at most 6. Connected components define the indivisible groups.
4. Try 300 consecutive GroupShuffleSplit candidates starting at seed 42. Each candidate uses a 15% test holdout and a 15% validation split of the remaining 85%; the candidate with the lowest squared class-distribution error against 70% / 15% / 15% is selected. The selected balance seed is 55.

Groups are assigned wholly to one split. The audit contains 2,478 groups, 43 pHash candidate pairs, and no group or exact pixel hash crossing Train, Validation, and Test. The resulting counts are:

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| cardboard | 280 | 65 | 58 | 403 |
| glass | 345 | 75 | 78 | 498 |
| metal | 292 | 54 | 63 | 409 |
| paper | 413 | 91 | 90 | 594 |
| plastic | 340 | 68 | 72 | 480 |
| trash | 96 | 22 | 19 | 137 |
| **Total** | **1766** | **375** | **380** | **2521** |

The grouping is a conservative leakage-control measure. Perceptual similarity does not establish that images show the same physical object, and it cannot guarantee every near duplicate is detected. Do not regenerate the split for experiment comparisons. If split directories already exist, `src/data_split.py` validates them against the fixed manifest and stops rather than overwriting a mismatch.

The previous Clean V1 stratified split's manifest, class counts, and SHA256 index are retained under [`data/manifests/clean_v1_stratified/`](data/manifests/clean_v1_stratified/). The original Clean V1 dataset directory outside this repository was left unchanged.

## Train the Frozen ResNet18 baseline

With the dependencies installed, the project entry point reads `data/train` and `data/val` and saves the best validation checkpoint under `models/`:

```bash
python src/train_transfer.py
```

The test split is not read by this training entry point. Use it only for a final evaluation after model selection.

## Citation

Please cite the original [TrashNet project](https://github.com/garythung/trashnet) by Gary Thung and Mindy Yang.
