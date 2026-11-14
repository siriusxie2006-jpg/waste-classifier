# TrashNet 本地基础模型

这份代码用于比较两套 ResNet18 基础方案。当前目录的做法是保留完整图像、等比例缩放并补白到 224×224，使用 ImageNet 预训练 ResNet18 和单层 `512 → 6` 分类头。先训练分类头 3 轮，再微调 `layer3`、`layer4` 和分类头 7 轮。训练使用亮度、对比度和模糊增强，以及按训练类别频率计算的加权交叉熵。

## 数据划分与实验结果

| 项目 | 本目录的分组划分 | 仓库根目录的 Clean V1 |
|---|---:|---:|
| 训练 | 1,766 | 1,764 |
| 验证 | 375 | 378 |
| 测试 | 380 | 379 |
| 总计 | 2,521 | 2,521 |

两份清单的 2,521 张图片及类别相同。这个目录把疑似近重复的图片成组分配，根目录按类别分层随机划分。本目录的 380 张测试图有 272 张在根目录 Clean V1 的训练划分。比较方案时，请先约定同一份清单并重新训练。 [本目录固定清单](data/split_manifest.csv) · [根目录 Clean V1](../DATASET_README.md)

本目录演示默认权重的历史结果为 375 张验证图准确率 93.33%、Macro-F1 0.9257，380 张测试图准确率 89.74%、Macro-F1 0.8704。瓶类微调权重在 28 张 TACO 保留瓶子裁图上准确率为 64.29%，同种子基础版为 25.00%。两项测试覆盖的数据不同。 [TrashNet 测试报告](HELDOUT_TEST_RESULTS.md) · [瓶类微调报告](BASELINE_MATERIAL_RESULTS.md)

## 重新训练并运行演示

先安装与你的运行环境匹配的 PyTorch 和 Torchvision，再从仓库根目录运行下方命令。数据准备脚本会下载和校验 TrashNet 原始缩小版，并核对本目录固定划分。首次训练还会获取 ImageNet 权重。

```sh
python -m pip install -r trashnet_pilot/requirements-demo.txt
python trashnet_pilot/prepare_data.py
python trashnet_pilot/train_baseline.py --run-name multiscale_confirm_control_seed43 --seed 43 --framing fit --augmentation both --finetune-scope layer3_layer4 --class-weighting inverse
python -m streamlit run trashnet_pilot/demo_app.py --server.address 127.0.0.1
```

本分支包含代码、清单、训练配置和历史评估输出。已有权重单独保存。上面命令会用相同设置重训通用基础模型，结果可能随硬件和依赖版本有波动。要打开整图增强或瓶类微调选项，需要取得各自的训练权重，放到对应的 `outputs/<运行名>/best_model.pt`。历史多种子评估及瓶类适配脚本还依赖额外运行和数据，请按各实验报告中的配置及数据来源复现。 [演示操作](DEMO_GUIDE.md) · [原始项目来源与融合](MERGE_GUIDE.md)
