# E1 Focal Loss 与 E2 CDS 的联合训练

E1 和 E2 的 Git 分支可以正常合并。原来的训练接口还需要连接。`src/train_e1.py` 只接收冻结 ResNet18 的 512 维输出，CDS 使用分类头中第一层 Linear 和 ReLU 产生的 256 维特征。组员提供的 CDS 组合器将分类项固定为交叉熵，而且包导入依赖手动添加 `integration` 搜索路径。

`src/train_e3.py` 为四种方法提供了同一个训练入口。`CDSBranch` 现在接收可选的分类损失，默认仍使用交叉熵。它保留原来的 `ce`、`cds` 和 `total` 字段，并新增 `classification` 字段。包导入和原来直接添加 `integration` 路径的方式都可使用。`E0Branches.train()` 会冻结主干与 BatchNorm，并按训练模式切换分类头。

## 四组对照

| 参数 | 目标函数 |
|---|---|
| `ce` | 普通交叉熵，作为同预算续训对照 |
| `e1_focal` | Focal Loss |
| `e2_cds` | 交叉熵 + lambda × CDS |
| `e3_focal_cds` | Focal Loss + lambda × CDS |

每组从同一份 E0 开始，分别训练自己的分类头。E3 通过联合目标重新训练分类头。单独训练好的 E1 和 E2 权重提供的是各自的实验结果，E3 的权重由联合训练产生。

共同路径为图片 → 冻结 ResNet18 → 512 维特征 → 可训练 Linear/ReLU → 256 维特征 → Dropout/Linear → 六类 logits。Focal 使用 logits 和真实标签，CDS 使用 Dropout 之前的 256 维特征、logits 和真实标签。

512 维输出可以缓存。256 维特征在每次训练前向中重新计算，梯度通过 CDS 回到可训练分类头。每个 CDS 训练组独立维护自己的 EMA 中心。中心只用 E0 的 Train 特征初始化，在该批反向传播和优化器更新后再做 EMA 更新。

## 运行

使用仓库已有的 Clean V1 / Grouped Split V1。把小组正式 E0 放到 `models/resnet18_frozen.pth`。本次验证使用的 SHA-256 是 `6e67965a4c0b43a5706e8d11dde226322342ee982e62a5a5c7a4e6f25603e7d0`。

从仓库根目录运行完整四组对照，输出目录需为新目录。

```powershell
python src/train_e3.py --checkpoint models/resnet18_frozen.pth --output runs/e3-comparison --epochs 10 --seeds 42 43 44 --gamma 1
```

默认使用组员 `integration/cds_config.json` 中已登记的参数，lambda 为 `0.17614097189991385`，theta 为 `0.30`，rho0 为 `0.10`。CDS 数学实现文件按 LF 换行计算的 SHA-256 与登记值一致；Windows checkout 转为 CRLF 后原始字节哈希会变化。这里采用 E1 已实验过的 gamma 1 进行接口验证，尚需以小组最终实验计划确定最终配置。

只运行联合方案也可指定训练组。

```powershell
python src/train_e3.py --checkpoint models/resnet18_frozen.pth --output runs/e3-only --epochs 10 --seeds 42 --gamma 1 --schemes e3_focal_cds
```

每个随机种子内，四组共享增强后的图片、批次顺序和 Dropout 随机序列，优化器为仅更新分类头的 Adam，学习率 0.001，batch size 32。中心初始化的数据加载会影响本次训练的随机序列，因此本次 CE/Focal 的结果应与本入口的配对对照比较。先前独立脚本的种子编号相同也可能产生不同训练轨迹。

## 输出与推理

每个 `seedXX` 目录保存训练配置、逐轮指标、每类 precision/recall/F1、混淆矩阵、逐图验证预测以及四组 checkpoint。主 checkpoint 按验证准确率最高选取，同分保留较早轮次。`epoch_last_heads.pth` 保存统一最后一轮的分类头，`summary.json` 同时列出主结果和最后一轮结果。

E3 checkpoint 的 `model_state_dict` 使用原来 ResNet18 的键名，类序为 cardboard、glass、metal、paper、plastic、trash。推理直接使用六类 argmax。训练时的 CDS 中心附加保存在 `cds_state_dict`，普通模型加载器可以继续读取 `model_state_dict`。

```powershell
python -m unittest discover -s tests -v
```

回归检查包括包导入、旧 CE 接口、Focal/CDS 目标公式、CDS 梯度路径、零系数时与原训练逐参数一致，以及标准 `.train()` 下主干与 BN 的冻结规则。完整训练还会逐张量检查主干和 BN，并重载每个已保存模型，复现它的验证预测。实验只使用 Train 和 Validation，Test 留待最终锁定方案后的统一比较。

## 本次完整验证

在 2026-11-20 使用本入口完成 seeds 42、43、44，各组续训 10 轮，gamma 1 和上述固定 CDS 参数。这是新入口的三种子配对验证。组员原五种子实验的完整训练入口与结果日志未包含在其上传分支中，本次结果用于验证当前上传模块的组合。

原 E0 在 375 张验证图上的预测与已有存档逐张一致。9 项数学和工程测试通过，12 个已保存模型重载后共复现 4,500 条验证预测。每个 CDS 组、每个种子的中心更新均为 560 次，主干和 BN 张量保持一致。对保存的逐图预测独立复算了所有种子、所有类别的 F1 及整体准确率，主 checkpoint 和最后一轮结果均通过核对。

下表续训指标是同一批 375 张验证图上的三种子均值，其中真实 Trash 只有 22 张。按相同规则选择验证准确率最高的一轮。

| 方案 | Accuracy | Macro-F1 | Trash F1 |
|---|---:|---:|---:|
| 正式 E0 | 83.20% | 80.30% | 61.11% |
| 同预算 CE 续训 | 84.09% | 81.03% | 59.70% |
| E1 Focal | 84.18% | 81.33% | 62.52% |
| E2 CDS | 84.53% | 81.50% | 60.25% |
| E3 Focal + CDS | 84.09% | 81.65% | 65.50% |

| 类别 F1 | 正式 E0 | 同预算 CE | E1 Focal | E2 CDS | E3 |
|---|---:|---:|---:|---:|---:|
| 纸板 | 89.76% | 89.31% | 89.60% | 89.31% | 88.94% |
| 玻璃 | 79.19% | 81.73% | 81.90% | 83.02% | 82.18% |
| 金属 | 82.26% | 85.63% | 84.51% | 86.97% | 84.69% |
| 纸张 | 88.89% | 89.67% | 90.05% | 89.63% | 89.44% |
| 塑料 | 80.60% | 80.12% | 79.42% | 79.86% | 79.16% |
| Trash | 61.11% | 59.70% | 62.52% | 60.25% | 65.50% |

统一取第 10 轮时，E3 的 Accuracy 为 83.11%、Macro-F1 为 80.16%、Trash F1 为 59.15%；CE 对照为 83.56%、80.45%、58.47%。主结果依赖选轮方式，在已多次用于开发的验证集上还需要保留这一限制。三个种子共享同一批图片，样本覆盖仍为 375 张和 22 张 Trash。

[完整指标与两种选轮口径](e3_integration_validation.json)
