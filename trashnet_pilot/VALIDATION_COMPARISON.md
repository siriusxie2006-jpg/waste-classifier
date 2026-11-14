# Validation augmentation comparison

Seed 42; fixed split; 3 head epochs + 7 layer4 epochs for every run.
Checkpoints selected by clean validation Macro-F1. Test not evaluated.
Single-seed exploratory results; repeat across seeds before firm conclusions.

| Condition | Baseline | Light | Blur | Both |
|---|---:|---:|---:|---:|
| clean | 0.8923 | 0.8941 | 0.8864 | 0.8934 |
| brightness_0.7 | 0.8736 | 0.8542 | 0.8590 | 0.8776 |
| brightness_0.4 | 0.8189 | 0.8149 | 0.7715 | 0.8268 |
| brightness_0.2 | 0.6723 | 0.7387 | 0.6757 | 0.7316 |
| blur_1 | 0.8624 | 0.8371 | 0.8590 | 0.8773 |
| blur_2 | 0.8347 | 0.7953 | 0.8309 | 0.8507 |
| blur_4 | 0.6817 | 0.6723 | 0.7162 | 0.7439 |

Scores above are Macro-F1 on the same 375 validation images.
Brightness and blur are separate synthetic conditions applied after resize/crop.
Training blur and evaluation blur use different implementations, as in the existing preview.
These results do not estimate performance in all real-world environments.
