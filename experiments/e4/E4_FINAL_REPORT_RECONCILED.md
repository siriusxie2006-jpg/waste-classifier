# DS4023 E4 最终实验报告（指标口径对账修订稿）

Robust Waste Classification Under Image Degradation · E4-1 / E4-2 / E4-3 / E4-4 / 五 seed 验证 / 最终 Test

生成时间：2026-11-21（本机 RTX 5070 Ti / torch 2.11.0+cu128 / torchvision 0.26.0+cu128 / Python 3.12.10）
本稿基于用户上传的 E4 原始报告，依据 `FINAL_STATUS_RECONCILIATION.md` 的审计记录修正指标口径与部分不准确表述；并非本地 `E4_AUDIT/E4_FINAL_REPORT_CORRECTED.md` 的原件。原报告称数字来自 `E4_FINAL/runs/`、`E4_FINAL/results/`，参见冻结归档记录。

---

## 0. 结论总览

| 阶段 | 做了什么 | Val Acc | Val Macro-F1 | 相对 E0 |
|---|---|---|---:|---:|
| E0（基准） | Frozen ResNet18 + 256D head | 83.20% | 80.30% | — |
| E4-1 | E0 特征上的 7 类分类器（256D/512D） | 88.27% | 86.76% | +5.07 / +6.46 pp |
| E4-2 | 解冻 Layer4 + Head 微调（5 seed 均值） | 88.37 ± 1.73% | 86.45 ± 2.26% | +5.17 / +6.15 pp |
| E4-3 | 宽高比保持 + padding 预处理 | 85.60% | 83.30% | 比 Resize 差 2.0–3.4 pp（**不采用**） |
| E4-4 | 5 模型概率平均 + 水平翻转 TTA | **91.20%** | **90.15%** | +8.00 / +9.85 pp |
| **最终 Test** | 冻结的 ens5 + hflip TTA（只访问一次） | **88.16%** | **85.60%** | **+7.11 / +8.70 pp**（McNemar p = 4.2e-5） |

三条最重要的结论：

1. **微调 Layer4 是主要的训练阶段改进**：Val Macro-F1 从 E0 的 80.30% 提升至五 seed 均值 86.45%（±2.26 pp）。Test 上的 85.60% 则属于后续叠加五模型集成与 TTA 后的最终部署方案，不能单独归因于微调。
2. **集成 + TTA 进一步改善 Validation 表现**（单模型 seed46 的 Macro-F1 88.16% → 五模型 + TTA 的 90.15%），不增加额外训练轮次，但需要十次前向推理，不能称为零成本。
3. **E4-1 的替代分类器未进入最终方案**：512D RBF SVM 的 Val Macro-F1 为 86.76%，略高于五 seed 单模型微调的均值 86.45%，但低于最终五模型集成 + TTA 的 90.15%。各方法指标应在同一指标口径下比较。E4-1 的价值在于证明"冻结特征里还有约 5 pp 没被 E0 头用掉"，而不是终点。

---

## 1. E4-1 分类器优化（冻结 E0 特征）

### 1.1 目标
在不改动 backbone 的前提下，回答两个问题：(a) E0 的 83.20% 是"特征上限"还是"解码器不够好"；(b) 若换更合适的分类器，能到多高。

### 1.2 实际执行的实验
- 特征：`E2_FEATURE_DISTANCE_AUDIT/FEATURES_{TRAIN,VALIDATION}.csv`（256D，即 `fc[0]` 的 ReLU 输出，pre-Dropout，eval 模式提取）为主；
  另用 `E2_FORMAL_PHASE1/A6_FEATURES_{TRAIN,VALIDATION}.csv`（512D `avgpool`）作补充对照。
- 分类器（7 族 × 超参网格，共 14 个配置）：
  `LogisticRegression`、`LinearSVC`、`RBF SVC`、`KNN(euclidean, k=1/3/5/9/15)`、`KNN(cosine, k=1/3/5/9/15)`、
  `ShrinkageLDA`（shrinkage ∈ {auto,0.1,0.3,0.5,0.8}）、`NearestCentroid`。
- 协议（严格遵守"标准化只用 Train、分类器只用 Train 训练、Validation 选参"）：
  1) `StandardScaler` 在 Train 上 fit；
  2) **5 折分层交叉验证（仅在 Train 内，random_state=42）** 选超参；
  3) 用选中超参在全 Train 上重拟合；
  4) 在 Validation 上只预测一次，并存下完整预测用于配对检验。

### 1.3 真实结果（Validation，n=375，E0 = 312/375）

| 特征 | 分类器 | 选中超参 | Train-CV Macro-F1 | Val Acc | Val Macro-F1 | 正确数 | vs E0 | McNemar p |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 512D | RBF SVM | C=10, gamma=scale | 0.8433 | **88.27%** | **86.76%** | 331 | +19 | 0.0094 |
| 512D | ShrinkageLDA | auto | 0.8181 | 87.47% | 85.68% | 328 | +16 | 0.0226 |
| 512D | KNN(1, euclid) | k=1 | 0.7824 | 86.67% | 85.51% | 325 | +13 | 0.1654 |
| 256D | RBF SVM | C=10, gamma=0.01 | 0.8448 | 86.67% | 84.22% | 325 | +13 | 0.0470 |
| 256D | LogisticRegression | C=0.1 | **0.8494** | 84.80% | 82.30% | 318 | +6 | 0.3449 |
| 256D | ShrinkageLDA | auto | 0.8407 | 84.00% | 81.49% | 315 | +3 | 0.6900 |
| 256D | KNN(9, euclid) | k=9 | 0.8330 | 84.80% | 82.29% | 318 | +6 | 0.3771 |
| 256D | NearestCentroid | — | 0.8077 | 80.27% | 77.31% | 301 | −11 | 0.0192 |

完整 14 行见 `results/final_tables/E4_1_CLASSIFIERS.csv`，逐类 P/R/F1 与混淆矩阵见 `results/e4_1/e4_1_results.json`。

### 1.4 与 E0 的比较与结论
- **最好的是 512D RBF SVM：Val Acc 88.27%、Macro-F1 86.76%，比 E0 多 19 张正确，McNemar p = 0.0094。**
- 256D（严格按任务书口径）最好的是 RBF SVM：86.67% / 84.22%，+13 张，p = 0.047。
- **必须说明的保留意见**：这里是在 14 个候选里按 Validation Macro-F1 挑的最好一个，存在选择偏倚；若按 Train-CV 排序，第一是 256D 逻辑回归（0.8494），512D RBF SVM 只有 0.8433。也就是说"哪个分类器最好"依赖你信哪一个估计量：固定 375 张 Validation 上 512D 更好，Train 内部 5 折上 256D 更好。我在 E3 阶段已经证明过这个验证集比 Train 内部留出系统性偏乐观（同一解码器从 Train 抽 5 个 1391/375 留出，512D 反而低 2.4 pp），所以**不把 E4-1 的 +5 pp 当作"特征变好了"的证据**，只当作"E0 头不是最优解码器"的证据。
- 对后续阶段的意义：E4-1 的最佳 Val Macro-F1 为 86.76%，与五 seed 微调均值 86.45% 接近；但微调模型可以进一步集成，最终达到 90.15%。E4-1 主要用于诊断冻结特征的分类可分性，而非最终部署方案。

---

## 2. E4-2 ResNet18 微调

### 2.1 目标
在冻结 BN 的前提下，比较两种解冻范围：A = Layer4 + Head，B = Layer3 + Layer4 + Head，判断 Macro-F1 的提升是否真实、Trash 是否改善、是否过拟合、跨 seed 是否稳定。

### 2.2 固定协议（两臂完全一致，只改解冻范围）
- 初始化：E0 checkpoint `E0_GROUPED_BASELINE/models/resnet18_frozen_best.pth`（SHA256 `6e67965a…3e7d0`）。
- BN：**始终 eval**，BN 仿射参数与 running stats 全部冻结（`freeze_all_bn`）。
- 优化：Adam，backbone lr **1e-4**，head lr **1e-3**，batch 32，10 epoch。
- 增强：完全沿用 E0 的 `train_transform`（Resize224 / HFlip / Rot15 / ColorJitter）。
- 损失：纯交叉熵，**不加辅助 Loss、不加类别权重**。
- 评价：固定 epoch 10 的 Validation 指标（任务书口径），同时记录逐 epoch 曲线。

### 2.3 真实结果（Validation，ep10）

| 臂 | seeds | Val Acc | Val Macro-F1 | Val Weighted-F1 | Val Trash F1 | Train Acc(ep10) | 单次耗时 |
|---|---|---:|---:|---:|---:|---:|---:|
| A: layer4 | 42,43 | 88.00% | 86.02% | — | 72.9% | 96.9% | ~66 s |
| B: layer3+4 | 42,43 | 88.27% | 84.01% | — | 54.8% | 96.5% | ~70 s |
| **A: layer4** | **42–46** | **88.37 ± 1.73%** | **86.45 ± 2.26%** | **88.26 ± 1.73%** | **73.07 ± 8.54%** | 96.5 ± 0.6% | ~66 s |

逐 seed（layer4/resize/ep10）：

| seed | Val Acc | Val Macro-F1 | Val Trash F1 | 最好 epoch（按 mF1） |
|---|---:|---:|---:|---:|
| 42 | 86.67% | 83.60% | 62.86% | 9 (85.73%) |
| 43 | 89.33% | 88.43% | 82.93% | 10 |
| 44 | 86.67% | 84.44% | 68.42% | 5 (87.34%) |
| 45 | 88.53% | 87.62% | 80.85% | 9 (88.15%) |
| 46 | 90.67% | 88.16% | 70.27% | 10 |

### 2.4 结论
- **在本次两个相同 seed 的对照中，Layer4 优于 Layer3+4**：Macro-F1 高约 2.0 pp（86.02 vs 84.01），Trash F1 高约 18 pp（72.9 vs 54.8）。训练集总计 1766 张，其中 Trash 仅 96 张；额外解冻参数可能增加过拟合风险，但两 seed 对照不足以证明唯一因果解释。
- **提升是真实的**：5 个 seed 全部 ≥ 83.60%，均值 86.45%，全部高于 E0 的 80.30%（最小 +3.3 pp）。
- **过拟合已经出现**：ep10 Train Acc 96.5% 而 Val Acc 88.4%（差 8.1 pp），且 5 个 seed 中 4 个的最好 epoch 出现在 9–10、只有 seed44 在 epoch 5（87.34%），说明固定 10 epoch 对多数 seed 尚可但已到边界。这是本项目的一条真实限制（见第 9 节）。
- 跨 seed 标准差 2.26 pp，**大于** E3 阶段看到的 CDS 类效应量，所以单 seed 比较（如 seed42 的 83.60%）不足以代表该方法。

---

## 3. E4-3 预处理：直接 Resize vs 保持宽高比 + Padding

### 3.1 目标
在 E4-2 最佳方案（Layer4 + Head）基础上，只改几何预处理，比较：
A = `Resize((224,224))`（E0 原策略）；B = 等比缩放到 224×224 内 + 用 ImageNet 均值色 padding 补齐。

### 3.2 真实结果（同 seed 配对，ep10）

| seed | Resize Val Macro-F1 | Pad Val Macro-F1 | Δ Macro-F1 | Resize Acc | Pad Acc | Δ Acc |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 83.60% | 81.56% | −2.04 pp | 86.67% | 85.07% | −1.60 pp |
| 43 | 88.43% | 85.03% | −3.40 pp | 89.33% | 86.13% | −3.20 pp |

### 3.3 结论
**Padding 在两个 seed 上都更差（−2.0 / −3.4 pp Macro-F1），不采用。**
可能原因：TrashNet 原图大多是 512×384 的 4:3，直接 Resize 到正方形会把物体拉伸到满框（占像素更多），而 padding 后有约 25% 的面积是无效灰边，等效分辨率下降；E0 的 head 又是在 Resize 特征上训练的，换成 padding 等于同时改了输入分布。因为 Resize 更好且本来就是 E0 协议，**最终保留 Resize**，不扩大预处理组合。

---

## 4. E4-4 集成与 TTA

### 4.1 目标
只在不改变架构、不重新训练的前提下，比较四种部署形态：单模型、5 模型概率平均、水平翻转 TTA、两者组合。

### 4.2 真实结果（Validation，n=375，E0 = 312）

| 方案 | 成员 | TTA | Val Acc | Val Macro-F1 | 正确数 | vs E0 | McNemar p |
|---|---|---|---:|---:|---:|---:|---:|
| 单模型（seed46） | 1 | 无 | 90.67% | 88.16% | 340 | +28 | 0.0002 |
| 单模型（seed46） | 1 | hflip | 89.60% | 86.54% | 336 | +24 | 0.0018 |
| 5 模型平均 | 5 | 无 | 89.60% | 88.76% | 336 | +24 | 0.0009 |
| **5 模型平均** | **5** | **hflip** | **91.20%** | **90.15%** | **342** | **+30** | **<0.0001** |
| 3 模型平均（43/45/46） | 3 | hflip | 91.73% | 91.04% | 344 | +32 | <0.0001 |

### 4.3 结论
- **5 模型 + hflip 是最合理的部署方案**（+30 张正确，Val Macro-F1 90.15%）。
- 在此次 Validation 对照中，seed46 单模型加 TTA 后正确数从 340 降至 336；五模型平均搭配 TTA 后正确数从 336 增至 342。只能据此说 **本次实验里 TTA 与五模型平均的组合更有效**，不能推广为 TTA 必须搭配集成才有效。
- 3 模型那行（91.73%）看起来更好，但那是**用 Validation 挑出来的 top-3 seed**（43/45/46），属于选择偏倚，不作为部署方案，只作为"模型间有互补性"的旁证（5 模型两两预测一致率平均 **90.85%**，即约 9% 的样本存在分歧，平均才有收益）。
- 互补性数据：`results/e4_4/ens5_resize_hflip.json` 的 `pairwise_agreement`。

---

## 5. 最终五 seed 验证（Layer4 + Resize，ep10）

| 指标 | E0 | E4-2 五 seed 均值 ± 标准差 |
|---|---:|---:|
| Validation Accuracy | 83.20% | **88.37% ± 1.73** |
| Validation Macro-F1 | 80.30% | **86.45% ± 2.26** |
| Validation Weighted-F1 | 83.01% | **88.26% ± 1.73** |
| Train Accuracy (ep10) | 80.9% | 96.50% ± 0.61 |

逐类（5 seed 平均，Validation）：

| 类别 | Precision | Recall | F1 | F1 std | E0 F1（参考） |
|---|---:|---:|---:|---:|---:|
| cardboard | 96.11% | 84.00% | 89.63% | 2.40 | 89.76% |
| glass | 92.56% | 85.60% | 88.87% | 2.19 | 79.19% |
| metal | 88.07% | 90.00% | 88.81% | 2.52 | 82.26% |
| paper | 85.82% | 95.16% | 90.23% | 1.68 | 88.89% |
| plastic | 84.60% | 92.35% | 88.10% | 3.49 | 80.60% |
| **trash** | 83.60% | 66.36% | **73.07%** | 8.54 | **61.11%** |

- **Trash 是最大赢家**：F1 从 61.11% 提到 73.07%（+11.96 pp，6 类中提升最大），但仍是最难类，且 seed 间波动最大（std 8.5）。
- Metal 的变化是"重新分配"而不是"变好/变差"：F1 从 82.26% 升到 88.81%，但那是靠精确率从 72.86% 升到 88.07% 换来的，召回反而从 94.44% 降到 90.00%——E0 的 metal 是"几乎不拒收"的特化解，微调后与其他类更均衡。其余类（cardboard −0.13、glass +9.68、paper +1.34、plastic +7.50）与之同向。
- 训练曲线见 `results/figures/curve_val_macro_f1.png`、`curve_val_accuracy.png`、`curve_train_accuracy.png`（PIL 绘制，无需 matplotlib）。
- **稳定性结论（如实）**：5 个 seed 全部优于 E0，均值提升 6.15 pp，但单 seed 之间相差 4.8 pp（83.60 → 88.43），**不能只报最好的 seed**。

---

## 6. 最终 Test 结果（只访问一次）

访问记录：`results/final_test/TEST_ACCESSED.json`、`logs/test_access.log`（时间戳 2026-11-21）。

| 指标 | E0 | 最终方案（5 模型 + hflip TTA） | 差值 |
|---|---:|---:|---:|
| Test Accuracy | 81.05% | **88.16%** | **+7.11 pp** |
| Test Macro-F1 | 76.90% | **85.60%** | **+8.70 pp** |
| Test Weighted-F1 | 80.63% | **88.03%** | **+7.40 pp** |
| 正确数 / 380 | 308 | **335** | **+27** |

配对检验：McNemar b=8（E0 对、最终错）、c=35（E0 错、最终对），**p = 4.19e-5**。

逐类（Test）：

| 类别 | Precision | Recall | F1 | E0 F1 | 支撑数 |
|---|---:|---:|---:|---:|---:|
| cardboard | 94.64% | 91.38% | 92.98% | 94.02% | 58 |
| glass | 86.25% | 88.46% | 87.34% | 78.21% | 78 |
| metal | 91.80% | 88.89% | 90.32% | 80.00% | 63 |
| paper | 85.57% | 92.22% | 88.77% | 85.88% | 90 |
| plastic | 87.50% | 87.50% | 87.50% | 75.00% | 72 |
| trash | 78.57% | 57.89% | 66.67% | 48.28% | 19 |

混淆矩阵（Test，行=真值，列=cardboard/glass/metal/paper/plastic/trash）：

```
cardboard  53  0  0  5  0  0
glass       0 69  4  1  4  0
metal       0  5 56  1  1  0
paper       3  1  0 83  2  1
plastic     0  4  1  2 63  2
trash       0  1  0  5  2 11
```

- **最终部署方案的 Validation→Test 差距**：Accuracy 从 91.20% 降至 88.16%（−3.04 pp），Macro-F1 从 90.15% 降至 85.60%（−4.55 pp）。原报告混入了单模型均值与不使用 TTA 的方案，不能据此得出“无 Test 过拟合迹象”的确定结论。
- 主要残余错误：trash 只召回 11/19（57.9%），其中 5 张被判成 paper；glass/metal 之间仍有 4+5 张互错。这与 Val 上的失败模式一致。

---

## 7. 哪些有效、哪些无效

| 手段 | 效果 | 判断 |
|---|---|---|
| 换分类器（E4-1，512D RBF SVM） | Val +19 张正确（+5.07 pp Acc），p=0.0094 | **有效但有限**，且选择偏倚大，不能替代微调 |
| 解冻 Layer4 微调（E4-2 A） | 5 seed 均值 +6.15 pp Macro-F1 | **最有效**，稳定、成本低（66 s/run） |
| 解冻 Layer3+4（E4-2 B） | Macro-F1 比 Layer4 低 2.0 pp，Trash 崩 18 pp | **无效**，容量过大 |
| 保持宽高比 + padding（E4-3） | −2.0 ~ −3.4 pp | **无效**，明确不采用 |
| 5 模型概率平均（E4-4） | Val +0.6 pp Macro-F1（88.16→88.76） | 有效但小 |
| 单模型 + hflip TTA | −1.6 pp（变差） | **无效**，不要单独用 |
| 5 模型 + hflip TTA | Val +2.0 pp Macro-F1（88.16→90.15） | **有效**，最终方案 |

为什么选 "Layer4 + Resize + 5 模型平均 + hflip TTA" 作为最终模型：
1. 它不需要额外的训练成本（复用五 seed 的 ep10 checkpoint），只多 10 次前向；
2. 它在 Validation 上是最好的**非挑选**方案（没有按 seed 挑成员，5 个全用）；
3. 它的提升方向正确（Macro-F1 +9.85 pp），且 Test 上复现了 +8.70 pp；
4. 它没有引入任何新架构、新损失或新超参。

---

## 8. 与 E0 的完整对比（Validation + Test）

| | E0 | 最终 |
|---|---:|---:|
| Val Acc | 83.20% | **91.20%（五模型 + TTA）**；五 seed 单模型均值 88.37% |
| Val Macro-F1 | 80.30% | **90.15%（五模型 + TTA）**；五 seed 单模型均值 86.45% |
| Val Trash F1 | 61.11% | **73.07%（五 seed 单模型均值；不是最终集成的 Trash F1）** |
| Test Acc | 81.05% | 88.16% |
| Test Macro-F1 | 76.90% | 85.60% |
| Test Trash F1 | 48.28%（P 70.0 / R 36.8） | 66.67%（P 78.6 / R 57.9） |

---

## 9. 当前项目的限制（如实）

1. **数据规模小**：Train 1766、Val 375、Test 380；Trash 只有 96/22/19 张。Test 上 trash 的 95% 置信区间约 ±22 pp，任何关于 trash 的结论都不稳。
2. **固定 10 epoch**：任务书要求沿用 E0 预算，但 seed44 在 epoch 5 就已达到最好 Macro-F1（87.34% > ep10 的 84.44%），说明预算对部分 seed 偏长；没有做 early stopping（那会引入额外的规则）。
3. **过拟合可见**：ep10 Train Acc 96.5% vs Val 88.4%，说明还有正则化空间（但这超出任务书允许的改动范围）。
4. **E4-1 的选择偏倚**：14 个候选里按 Validation 挑最好，p 值未做多重比较校正；按 Train-CV 排序会得到不同赢家。
5. **Test 只评一次**：380 张、单次，无法给出 Test 上的方差；所有 Test 结论都是"这一份固定 Test 上的点估计"。
6. **集成成本**：5 模型 + TTA 需要 10 次前向，部署成本是 E0 的 10 倍（但对本项目的规模可接受）。
7. **未做的事**：没有做类别权重/重采样（任务书禁止）、没有做更激进的增强、没有做模型蒸馏、没有在 Test 上做任何调参。

---

## 10. 复现指令

```powershell
$PY = 'C:\Users\Administrator\AppData\Local\venvs\e3-windows\Scripts\python.exe'
cd D:\ml_project

# E4-1（CPU，约 25 s）
& $PY E4_FINAL\code\e4_1_classifiers.py

# E4-2 / E4-3（GPU，每次约 66 s）
& $PY E4_FINAL\code\e4_2_finetune.py --unfreeze layer4 --seed 42
& $PY E4_FINAL\code\e4_2_finetune.py --unfreeze layer3+4 --seed 42
& $PY E4_FINAL\code\e4_2_finetune.py --unfreeze layer4 --preprocess pad --seed 42

# E4-4（GPU，约 20 s/次）
& $PY E4_FINAL\code\e4_4_ensemble_tta.py --members <5 个 checkpoint> --tta hflip --name ens5_resize_hflip

# 最终 Test（只允许一次；TEST_ACCESSED.json 存在时会拒绝运行）
& $PY E4_FINAL\code\e4_final_test.py --members <5 个 checkpoint> --tta hflip --name final_ens5_hflip

# 汇总与画图
& $PY E4_FINAL\code\aggregate_e4_2.py
& $PY E4_FINAL\code\assemble_tables.py
& $PY E4_FINAL\code\plot_curves.py
```

---

## 11. 交付物清单

```
E4_FINAL/
├─ E4_FINAL_REPORT.md                 本报告
├─ SHA256SUMS.txt                     全部关键产物的 SHA256
├─ code/
│  ├─ e4_common.py                    数据/模型/指标/协议（唯一真源）
│  ├─ e0_val_predictions.py           E0 验证集参考预测（312/375 复现）
│  ├─ e4_1_classifiers.py             E4-1
│  ├─ e4_2_finetune.py                E4-2 / E4-3
│  ├─ e4_4_ensemble_tta.py            E4-4
│  ├─ e4_final_test.py                最终 Test（单次、带守卫）
│  ├─ aggregate_e4_2.py               运行汇总
│  ├─ assemble_tables.py              全部交付表
│  └─ plot_curves.py                  训练曲线（PIL）
├─ runs/e4_2/<run>/                   9 个 run：checkpoint_epoch10.pth、
│                                     EPOCH_HISTORY.json、RUN_CONFIG.json、val_probs_epoch10.npz
├─ results/
│  ├─ e0_val_summary.json、e0_val_probs.npz
│  ├─ e4_1/                           e4_1_results.csv/.json、selected、val_predictions.npz
│  ├─ e4_2/                           e4_2_runs.csv、e4_2_summary.json
│  ├─ e4_4/                           每个方案的 json + probs
│  ├─ final_tables/                   E4_1_CLASSIFIERS.csv、E4_2_RUNS.csv、E4_3_PREPROCESS.csv、
│  │                                  E4_4_ENSEMBLE_TTA.csv、E4_5SEED_*.csv/json
│  ├─ figures/                        3 张训练曲线
│  └─ final_test/                     final_ens5_hflip_test_results.json、TEST_ACCESSED.json
└─ logs/                              e4_2_screen.log、e4_2_seeds_pad.log、e4_4.log、test_access.log
```

## 12. 数据访问声明

- 本次 E4 的**选择、调参、集成决策全部只使用 Validation**。
- **Test 只在第 6 节被访问一次**（`results/final_test/TEST_ACCESSED.json` 与 `logs/test_access.log` 有记录），之后没有再训练或调参。
- 未读取/未使用 `TRASHNET_CLEAN_V1/data/test`（那是旧的随机划分 379 张），本次 Test 严格使用 Grouped Split V1 的 `third_party/waste-classifier/data/test`（380 张，58/78/63/90/72/19）。
- E0–E3 的所有原始目录与结果均未被修改。

---

## 修订说明与审计边界

此文档是基于上传的原始报告整理的**指标口径修订稿**，不是对训练代码或 Test 数据的重新执行，也不冒充独立审计目录中同名的原始修正版。主要修订包括：§0 微调与集成成效的归因、§1.4 分类器与微调的 Macro-F1 比较、§2.4 两 seed 对照的限定、§4.3 TTA 结论适用范围、§6/§8 最终模型 Validation 与 Test 指标的对应关系。

审计对账结论为 **PASS WITH WARNINGS**，残余限制包括 E4-1 逐折 CV 证据缺口、512D 特征直接来源验证缺口、E4-3 仅两个 seed、多重比较选择偏倚、Test 访问锁非原子及 SHA256 清单 BOM 问题。参考 [审计状态对账](FINAL_STATUS_RECONCILIATION.md)。
