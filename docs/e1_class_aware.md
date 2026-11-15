# E1 类别加权交接

E1 的验证候选为 **Trash 权重 1.5 倍**。分类任务继续采用六类 Frozen ResNet18。类序为 Cardboard、Glass、Metal、Paper、Plastic、Trash，相对权重为 `[1, 1, 1, 1, 1, 1.5]`。

## 用同一条件复现

从仓库根目录运行。使用 `main` 的 Clean V1 / Grouped Split V1，以及全组约定的 E0 文件 `models/resnet18_frozen.pth`。正式 E0 的 SHA-256 为 `6e67965a4c0b43a5706e8d11dde226322342ee982e62a5a5c7a4e6f25603e7d0`。分组清单的 SHA-256 为 `fa928903c389e66bba75fa1d2de91083170254b1fab4eebf56b217defe104851`。

```powershell
foreach ($seed in 42,43,44) {
    python src/train_e1.py --checkpoint models/resnet18_frozen.pth --output "runs/e1/seed$seed" --seed $seed --epochs 10 --trash-multipliers 1.25 1.5
}
```

每个种子会生成普通续训、Trash 1.25 倍和 1.5 倍的权重、训练配置、逐轮验证指标和逐图预测。复现首轮的逆频率与有效样本数方案时，使用同样的命令并去掉 `--trash-multipliers 1.25 1.5`，换一个输出目录。每组按最高验证准确率选择 checkpoint，同分取较早轮次。

## 验证结果

| 方案 | 六类 Accuracy | 六类 Macro-F1 | Trash Precision | Trash Recall | Trash F1 |
|---|---:|---:|---:|---:|---:|
| 正式 E0 | 83.20% | 0.8030 | 78.57% | 50.00% | 61.11% |
| 普通交叉熵续训 | 83.73% | 0.8060 | 80.43% | 48.48% | 60.38% |
| Trash 权重 1.25 倍 | 83.56% | 0.8071 | 72.79% | 53.03% | 60.89% |
| Trash 权重 1.5 倍 | 83.64% | 0.8078 | 77.30% | 51.52% | 61.81% |

续训组为三个种子在同一 375 张验证图上的均值。1.5 倍方案的 Trash F1 相对普通续训在 seed42 和 seed43 提高，在 seed44 下降。数据集中只有 22 张真实 Trash 验证样本，因此用这组结果作为选择候选的依据。380 张保留测试图没有参与本轮。用 `main` 的原版模型定义和验证预处理重新加载保存的 E0、三份 1.5 倍权重及三份普通续训权重，逐份复现全部 375 张验证图的预测。

## 与 E2 对接

在 E2 的训练代码中先核对六个输出的类别顺序。若沿用当前数据加载器，可以复用 E1 的类别权重实现。

```python
from train_e1 import CLASSES, class_weights
from torch import nn

assert train_loader.dataset.classes == CLASSES
_, weights = class_weights(train_loader.dataset, device, (1.5,))
criterion = nn.CrossEntropyLoss(weight=weights["trash_1.5"])
```

上面的权重用于训练的分类交叉熵。如果 E2 有辅助损失项，E3 按 E2 的实际公式合并该项。E3 从正式 E0 权重起训，使用与普通续训、E1、E2 相同的追加训练轮数、样本划分和配对种子。统一以普通交叉熵及分类指标验证，并列原始 E0 作为历史初始成绩。这样可以分别测量 E1、E2 及两者组合的效果。
