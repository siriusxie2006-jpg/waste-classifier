# 垃圾分类项目融合版

以本地固定划分和抗光照、模糊实验为基础，吸收外部工程的两层分类头、类别加权和更深层微调方案。统一训练、加载、验证与演示接口，保留原有四组结果。

## 已接入的功能

- `--class-weighting inverse`：仅按训练集计算逆频率类别权重。
- `--head mlp`：512→256→6，ReLU 和 Dropout(0.4)。默认仍为单层头。
- `--finetune-scope layer3_layer4`：微调更深的两个模块；默认仍为 layer4。冻结模块的 BatchNorm 始终保持 eval。
- 所有配置继续使用固定划分、3+7 轮预算、AdamW、验证 Macro-F1 选模；结构因素可单独对照。
- 新旧本地 checkpoint 使用同一严格加载入口，校验类别顺序、划分指纹和参数形状。没有引入外部已训练权重或测试结果。
- `preview_robustness.py --include-bright`：单独输出扩展验证预览，增加亮度 1.3、1.7；保留原始 7 条件结果文件。
- `analyze_errors.py`：仅使用验证集，输出逐类指标、混淆对和最高置信度错误的 Grad-CAM。热力图叠加在模型实际输入裁剪上，hook 使用后移除。
- `demo_app.py`：中文本地演示，对同一上传图片比较两个模型，可调亮度、模糊，显示类别概率与 Grad-CAM。结果表读取实际 summary，不使用上游硬编码分数。

## 从项目主目录运行

```powershell
.\.venv-trashnet\Scripts\python.exe .\trashnet_pilot\test_fusion.py
.\.venv-trashnet\Scripts\python.exe .\trashnet_pilot\run_merged_experiments.py
.\.venv-trashnet\Scripts\python.exe -m streamlit run .\trashnet_pilot\demo_app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

演示依赖见 `requirements-demo.txt`。本机已安装环境后不需要重复安装。组员电脑按原 README 安装 PyTorch，再安装演示依赖。

根目录的旧 `trashnet_pilot_bundle.zip` 保留原始 pilot 内容，尚未更新为融合版；当前实现以 `trashnet_pilot` 文件夹为准。依赖版本快照另见 `environment_fusion_freeze.txt`。

## 本次探索性实验

| 新运行 | 相对新对照的单项差异 |
|---|---|
| merged_control_seed42 | 融合代码的默认基准，复核重构影响 |
| merged_weighted_seed42 | 仅类别加权 |
| merged_both_weighted_seed42 | 类别加权+光照与模糊增强 |
| merged_mlp_seed42 | 仅两层分类头 |
| merged_deeper_seed42 | 仅解冻 layer3+layer4 |

历史 baseline 和 both 也纳入扩展验证表；同一 seed 不同配置分别训练，不直接拼接两份模型权重。结果见 `MERGED_RESULTS.md` 和 `outputs/merged_validation_comparison.csv`，完整设置在 `outputs/merged_experiment_plan.json`。

本次是单种子可行性实验，不据此宣称某方法稳定优于另一方法。增加结构、参数和加权可能降低表现，最终按验证证据选择。扩展条件用于探索，不会为了调参使用测试集。

## 本次实际发现

默认对照复现了原基准的混淆矩阵。类别加权+组合增强的清晰 Macro-F1 为 0.9019，强暗光为 0.7217，强模糊为 0.7590；原基准分别为 0.8923、0.6723、0.6817。少数类 trash 的验证召回率从 0.6818 提高到 0.8182。

这不代表所有条件都优于已有组合增强：已有 both 在强暗光的 Macro-F1 为 0.7316，略高于加权组合的 0.7217。深层微调清晰分数达 0.9128，但强暗光、强模糊降到 0.5489、0.5215；两层头本次也未优于原基准。因此暂不把深层微调或两层头叠加进默认改进方案，下一步优先用多种子确认训练方案的取舍。

对照图见 `outputs/merged_validation_comparison.png`；错误案例和 Grad-CAM 见对应运行的 `analysis` 子目录。运行 `plot_merged_results.py` 可重建对照图。

## 来源和贡献边界

外部参考工程：

- 组员快照：<https://github.com/siriusxie2006-jpg/waste-classifier/tree/9520fe4c44d08a6860f8f457c05c240437fac21c>
- 该工程注明的上游：<https://github.com/kal-ngga/waste-classifier/tree/0bda35180855b2fee1e7a6df7961faedd6dcaad3>
- 两层头参考 `src/model_finetune.py` / `src/model_transfer.py`，逆频率加权和深层微调参考 `src/train_finetune.py`。融合实现为统一框架重新编写，没有复制上游静态指标或模型权重。
- 冻结统计量检查的思路与组员 frozen 分支一致，并扩展到深层微调。
- Grad-CAM 方法：Selvaraju et al., ICCV 2017，<https://arxiv.org/abs/1610.02391>。此处用 PyTorch autograd 和 Matplotlib 重新实现，不依赖 OpenCV。
- Streamlit 展示参考外部工程的上传预测及解释功能；界面和模型适配重新实现。

这些技术本身是已有方法。报告应把本组贡献写成具体方案、适配实现、公平消融和发现，不把上游已有功能归为本组发明。

外部训练与调参图片历史仍需组员确认。本次没有加载外部 checkpoint，只在本地固定 train 上从 ImageNet 重新训练。若小组之前基于与本地测试集重叠的图片调整了方法，最终评估的独立性需要如实说明。
