# Integration validation results

Seed 42 exploratory results; fixed split, 3 warmup + 7 fine-tuning epochs; test not evaluated.
All checkpoints selected using clean validation Macro-F1. Bright conditions are newly added validation probes.

| Run | Clean accuracy | Clean Macro-F1 | Trash recall | Dark 0.2 F1 | Blur 4 F1 | Bright 1.7 F1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline_resnet18_seed42 | 0.9067 | 0.8923 | 0.6818 | 0.6723 | 0.6817 | 0.7633 |
| resnet18_both_seed42 | 0.8987 | 0.8934 | 0.8182 | 0.7316 | 0.7439 | 0.7561 |
| merged_control_seed42 | 0.9067 | 0.8923 | 0.6818 | 0.6723 | 0.6817 | 0.7633 |
| merged_weighted_seed42 | 0.9040 | 0.8996 | 0.8182 | 0.6863 | 0.6415 | 0.7533 |
| merged_both_weighted_seed42 | 0.9067 | 0.9019 | 0.8182 | 0.7217 | 0.7590 | 0.7700 |
| merged_mlp_seed42 | 0.8960 | 0.8849 | 0.7273 | 0.6549 | 0.5938 | 0.7526 |
| merged_deeper_seed42 | 0.9093 | 0.9128 | 0.8636 | 0.5489 | 0.5215 | 0.7700 |

Refactored control confusion matrix matches historical baseline: True.
Use the new control for integration ablations. If controls differ, do not attribute that difference to a new method.
Do not infer that combining every available feature improves performance. Repeat across seeds before final selection.
