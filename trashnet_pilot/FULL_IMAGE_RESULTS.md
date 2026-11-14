# 完整图片输入与裁剪修正

日期：2026-11-09。针对上传图片自动中心裁剪可能丢失物品部位的问题，演示改为保留整图、等比例缩放并白色留边至 224×224。裁剪必须由用户框选并点击确认；预览、分类、Grad-CAM 使用同一输入。

## 为什么补做训练

直接给旧权重更换输入流程会造成训练与使用不匹配。在同一 375 张验证图片上的诊断如下。这里没有重新训练：

| 旧模型 | 原中心裁剪准确率 / Macro-F1 | 同权重改为完整图片输入 |
|---|---|---|
| 原始 ResNet18 | 90.67% / 0.8923 | 87.20% / 0.8394 |
| 光照＋模糊＋类别平衡 | 90.67% / 0.9019 | 87.47% / 0.8585 |

原始记录：`outputs/upload_preprocessing_check.json`。这些差异说明不能只修改上传预览，就沿用旧模型的验证成绩。

## 与整图输入匹配的新训练

固定划分保持 1766 train / 375 val / 380 test；清单 SHA256 为 `dcb36e7e183a698c837aa682635ecc606d1a7c1ab360a776b1784e8226e2bf07`。测试集没有评估。

两组均为 seed42、ImageNet 预训练 ResNet18、线性分类头、3 轮仅分类头＋7 轮 layer4 微调，按验证 Macro-F1 选择权重。训练和验证都使用完整图片缩放留边，训练保留随机水平翻转，不使用旧的随机裁剪。第二组额外使用原有光照、模糊增强与按训练集频数计算的类别权重。

| 新运行 | 验证准确率 | 验证 Macro-F1 | 最佳轮次 |
|---|---|---|---|
| full_image_control_seed42 | 90.67% | 0.8966 | 10 |
| full_image_both_weighted_seed42 | 91.47% | 0.9019 | 8 |

后一组为新默认模型；基础对照使用前一组。模型设置仅提供这两组输入匹配的模型，原中心裁剪实验仍保留在验证表格和原输出目录中。

这些是单随机种子的验证选模结果，不能说明真实上传图片已可靠，也不能把旧中心裁剪模型与新整图模型的差值归因于单个组件。整图训练也改变了训练时的裁剪增强。

## 塑料被识别为玻璃的问题仍未解决

新默认模型的干净验证结果中，68 张塑料有 62 张被正确识别，另 6 张全部被识别为玻璃；75 张玻璃中有 4 张被识别为塑料。旧默认模型在原中心裁剪流程中，同样有 6 张塑料被识别为玻璃。

因此，这次输入修正没有消除塑料和玻璃的混淆。2026-11-10 已取得用户反馈的矿泉水瓶原图并复现：默认模型的玻璃得分为 81.96%，人工框选完整瓶子后仍为 77.36%。诊断见 `USER_PHOTO_DIAGNOSIS.md`。目前不能确定外形、反光、拍摄条件或训练样本覆盖各自的影响；需要收集独立的实际拍摄失败案例，检查错误模式，再决定训练或模型改动。反馈案例一旦用于调参，就不作为最终独立测试成绩。

## 干扰验证预览

当前默认模型同一验证集的结果如下。这是输入新流程下的结果，不能直接与旧流程当作严格的单项消融比较。

| 条件 | 准确率 | Macro-F1 |
|---|---|---|
| 原图 | 91.47% | 0.9019 |
| 亮度 0.7 | 89.87% | 0.8842 |
| 亮度 0.4 | 86.67% | 0.8545 |
| 亮度 0.2 | 78.13% | 0.7633 |
| 模糊 1 | 89.07% | 0.8821 |
| 模糊 2 | 84.53% | 0.8354 |
| 模糊 4 | 73.07% | 0.7156 |
| 亮度 1.3 | 90.13% | 0.8813 |
| 亮度 1.7 | 81.33% | 0.7826 |

数据来自 `outputs/full_image_both_weighted_seed42/validation_robustness_extended.csv`。强模糊和强光照变化仍会降低准确率。

## 复现

在 `D:\ML project` 中运行。输出目录已存在时训练程序会拒绝覆盖，复跑请使用新的 run-name。

```powershell
.\.venv-trashnet\Scripts\python.exe .\trashnet_pilot\train_baseline.py --run-name full_image_control_seed42 --framing fit
.\.venv-trashnet\Scripts\python.exe .\trashnet_pilot\train_baseline.py --run-name full_image_both_weighted_seed42 --framing fit --augmentation both --class-weighting inverse
.\.venv-trashnet\Scripts\python.exe .\trashnet_pilot\preview_robustness.py --run-name full_image_both_weighted_seed42 --include-bright
```

`--framing` 默认仍为 `center`，保留历史训练和评估流程；整图训练需显式传入 `fit`。评估已有 checkpoint 时读取其保存的 framing 配置。原 baseline 权重、固定划分和旧结果均保留，未上传 GitHub。
