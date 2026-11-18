# E1: Focal Loss

This experiment tests whether unweighted multiclass Focal Loss improves Trash recognition while preserving overall six-class performance. It uses the team's Clean V1 / Grouped Split V1 manifest and starts from the shared E0 checkpoint. It retains the frozen ResNet18 and BatchNorm state and trains the existing classification head. Prediction uses the same six-class argmax.

For the true class probability $p_t$, Focal Loss is $-(1-p_t)^\gamma\log(p_t)$. It downweights easy examples based on prediction confidence. Every class has the same additional class weight of 1. See the [original paper](https://openaccess.thecvf.com/content_ICCV_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html).

## Reproduce

Use the repository's [fixed data split](../DATASET_README.md) and place the team's E0 checkpoint at `models/resnet18_frozen.pth`.

- E0 SHA-256: `6e67965a4c0b43a5706e8d11dde226322342ee982e62a5a5c7a4e6f25603e7d0`
- Split manifest SHA-256: `fa928903c389e66bba75fa1d2de91083170254b1fab4eebf56b217defe104851`

Run [the training script](../src/train_e1.py) from the repository root. Use a fresh output directory for each run.

```powershell
foreach ($seed in 42,43,44) {
    python src/train_e1.py --checkpoint models/resnet18_frozen.pth --output "runs/e1-focal/seed$seed" --seed $seed --epochs 10 --focal-gammas 0.5 1 2
}
```

The script trains an ordinary cross-entropy control and one head for each `gamma`. Within a seed, all arms share batches, augmented inputs, dropout seeds, Adam settings, and ten continuation epochs. The checkpoint with the highest validation accuracy is selected, with ties resolved in favor of the earlier epoch. Each run writes configuration, per-class and per-epoch metrics, individual validation predictions, and checkpoints to its output directory.

## Validation results

The validation set contains 375 images, including 22 Trash images. Continuation results are means across seeds 42, 43 and 44 on the same validation set. These runs did not use the Test split.

| Model | Accuracy | Macro-F1 | Trash F1 | Trash recall | Trash precision |
|---|---:|---:|---:|---:|---:|
| Official E0 | 83.20% | 80.30% | 61.11% | 50.00% | 78.57% |
| Matched CE continuation | 83.73% | 80.60% | 60.38% | 48.48% | 80.43% |
| Focal, gamma 0.5 | 83.29% | 80.42% | 60.17% | 53.03% | 71.04% |
| Focal, gamma 1 | 83.91% | 80.88% | 61.14% | 50.00% | 79.37% |
| Focal, gamma 2 | 83.29% | 80.81% | 62.93% | 59.09% | 68.10% |

| Class F1 | Official E0 | Matched CE | Focal 0.5 | Focal 1 | Focal 2 |
|---|---:|---:|---:|---:|---:|
| Cardboard | 89.76% | 89.68% | 88.65% | 89.62% | 89.24% |
| Glass | 79.19% | 80.72% | 81.37% | 80.73% | 81.84% |
| Metal | 82.26% | 84.15% | 85.63% | 85.72% | 84.76% |
| Paper | 88.89% | 90.42% | 88.66% | 90.17% | 87.51% |
| Plastic | 80.60% | 78.22% | 78.03% | 77.87% | 78.59% |
| Trash | 61.11% | 60.38% | 60.17% | 61.14% | 62.93% |

Gamma 2 increased Trash recall and false positives. Gamma 1 showed small gains against matched CE here, with sensitivity to training and splitting. In two internal grouped five-fold runs, gamma 1 reduced accuracy from 80.58% to 80.12% and from 81.82% to 81.31%. Trash F1 moved from 59.46% to 59.34% and from 57.30% to 59.26%. The two partitions reused the same development images.

In a later matched continuation run with image views shared across seeds and independently shuffled batch order, gamma 1 reached 84.36% accuracy, 81.36% Macro-F1 and 60.57% Trash F1. In that run its six class F1 scores in order were 90.09%, 81.81%, 86.04%, 90.25%, 79.38% and 60.57%. The matched CE had 84.36% accuracy, 81.17% Macro-F1 and 58.51% Trash F1. These results leave Focal Loss as an E1 candidate and comparison, with simultaneous gains in overall accuracy and Trash recognition still unconfirmed.
