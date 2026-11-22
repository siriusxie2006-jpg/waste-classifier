# DS4023 E4 — Frozen Final Method

**Status:** FROZEN (2026-11-22)  
**Audit:** PASS WITH WARNINGS (as recorded in `FINAL_STATUS_RECONCILIATION.md`)  
**Project:** Robust Waste Classification Under Image Degradation

## Final selected method

**ResNet18 Layer4 fine-tuning + five-seed probability ensemble + horizontal-flip TTA.**

1. Initialize from E0 pretrained/frozen-backbone ResNet18 checkpoint with classification head `512 → 256 → ReLU → Dropout(0.4) → 6`.
2. Fine-tune **Layer4 and the head only**; freeze earlier backbone stages and all BatchNorm parameters/running statistics.
3. Train using five independent random seeds: **42, 43, 44, 45, 46**, retaining the fixed epoch-10 checkpoints.
4. Resize images to **224 × 224** using the E0 pipeline (aspect-ratio-preserving padding was tested and rejected).
5. At inference, each of the five models predicts both the original and horizontally flipped image. Average all **ten softmax probability vectors** and take the maximum-probability class.

Training: Adam, learning rate **1e-4** for Layer4 and **1e-3** for the classification head, batch size **32**, **10 epochs**, cross-entropy loss, fixed Grouped Split V1 (train **1766**, validation **375**, test **380**). No auxiliary loss, class weighting or architecture changes.

## Frozen results

| Metric | E0 Test | E4 Final Test | Change |
|---|---:|---:|---:|
| Accuracy | 81.05% | **88.16%** | **+7.11 pp** |
| Macro-F1 | 76.90% | **85.60%** | **+8.70 pp** |
| Weighted-F1 | 80.63% | **88.03%** | **+7.40 pp** |
| Trash F1 | 48.28% | **66.67%** | **+18.39 pp** |

Correctly classified test samples: **308/380 → 335/380**. Final ensemble Validation Accuracy **91.20%** and Macro-F1 **90.15%** (different from five single-model seed averages: **88.37% / 86.45%**).

## Experiment decisions

| Phase | Question / comparison | Decision |
|---|---|---|
| E4-1 | Alternative classifiers on frozen E0 features (256D/512D) | Diagnostic only; 512D RBF SVM Validation Accuracy 88.27%, Macro-F1 86.76%. |
| E4-2 | Unfreeze Layer4 vs Layer3+4 | **Retain Layer4 + head**: stronger Validation Macro-F1 in paired-seed comparison; five-seed mean 86.45%. |
| E4-3 | Direct Resize vs aspect-preserving Resize + Padding | **Retain direct Resize**. Padding underperformed in two seeds. |
| E4-4 | Single, ensemble, TTA, ensemble + TTA | **Retain five-model ensemble + hflip TTA**; Validation 91.20% / 90.15%. |

## Artifact provenance

- [Experiment configuration](E4_EXPERIMENT_CONFIG.json)
- [Audit status and limitations](FINAL_STATUS_RECONCILIATION.md)
- [Original report (historical; contains known §6/§8 metric-label inconsistencies)](E4_FINAL_REPORT_ORIGINAL.md)
- [Reconciled report derived from the uploaded original](E4_FINAL_REPORT_RECONCILED.md)

**For formal course submission, prefer the audited local `D:\ml_project\E4_AUDIT\E4_FINAL_REPORT_CORRECTED.md`.** The reconciled GitHub version was prepared from the original report and audit-status evidence; it is **not identical by provenance** to that local audited file.

**Actual training/inference source code and the five checkpoint `.pth` files are not part of this GitHub directory yet.** They remain in the frozen local `D:\ml_project\E4_FINAL\code\` and `runs\e4_2\layer4_resize_seed{42..46}\` folders. Do **not** treat this documentation-only archive as a runnable reproduction.

## Preserved limitations

E4-1 five-fold CV per-fold evidence unavailable; E4-1 selection/multiple-comparison bias; 512D feature provenance not directly verified; E4-3 only two seeds; final ensemble costs ten forward passes; Test access guard has non-atomic/force-path warnings. The archived audit reports indicate fixed Grouped Split V1 with no cross-split file/group overlap and a single final Test access.

**Freeze policy:** do not modify E4 methods or experiment artifacts, retrain, tune on Test, or revisit Test images. Future work is understanding the method, preparing the course report and Q&A.
