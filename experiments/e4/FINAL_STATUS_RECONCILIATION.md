# E4_FINAL 最终审计状态对账与归档

审计对象：`D:\ml_project\E4_FINAL`；证据目录：`D:\ml_project\E4_AUDIT`。本轮只读取既有 Markdown/JSON 证据，不运行项目代码，不访问 Test 图片，不改写历史报告、实验代码或产物。

## 1. 最终总体结论

**PASS WITH WARNINGS。** 对账结果支持 `E4_AUDIT_REPORT.md` 的总判断；此前将整个项目概括为 **BLOCKED** 的状态摘要过于宽泛，与现存审计证据矛盾，应以本文件和审计总报告的细分结论为准。没有证据表明核心验证/测试指标、五个部署模型、划分隔离或 Test 访问纪律无法核验。

这里的 **BLOCKED** 只适用于局部材料缺口（例如无法独立复算未保存的 E4-1 逐折 CV），不等价于项目整体被阻塞。已由现存证据完成的指标复算、Checkpoint 张量审计及 split/group/manifest/Test 时序核验不得重复标成未验证。

## 2. 五项审计状态

| 工作 | 状态 | 对账依据 |
|---|---|---|
| 完整性与归档 | PASS WITH WARNINGS | `integrity_check.json`：88 项，无缺失、无尺寸差异、无未登记；87 项哈希匹配。唯一 hash_mismatch 是 `code/aggregate_e4_2.py` 的 expected 字符串带 U+FEFF，去 BOM 后与 actual 完全一致。ZIP `testzip()` 无损坏；见 `INTEGRITY_REPORT.md`。 |
| 源码 | PASS WITH WARNINGS | `static_checks.json`：源码 AST/静态检查通过；Test 唯一代码访问点为最终测试脚本。锁非原子及 `--force` 属工程 WARNING，不推翻本次访问记录；详见 `CODE_AUDIT.md`。 |
| Checkpoint | PASS | `checkpoint_audit.json` 与 `CHECKPOINT_AUDIT.md`：五个模型 CPU 可读，均为 124 张量；相对 E0 的变化符合 layer4+head，layer1–3/BN 保持一致，五模型互异，测试 JSON 引用的 checkpoint 路径及哈希匹配。 |
| 指标复算 | PASS | `metric_recalc.json`、`METRIC_RECALCULATION.md`：验证集五 seed、各集成/TTA 方案及最终 Test 概率指标、混淆矩阵、逐类指标和 McNemar 均已独立复算并与报告/JSON 一致。 |
| 实验协议 | PASS WITH WARNINGS | `static_checks.json`、`PROTOCOL_AUDIT.md`：Grouped Split V1 manifest 哈希匹配；split 路径、group、跨集文件哈希重叠为 0；Test 访问单次且在配置冻结之后。限制见第 7 节。 |

## 3. BLOCKED 与 WARNING 的边界

**局部 BLOCKED：** E4-1 各分类器族的五折 CV 逐折分数与折划分没有留存，无法从现有材料重新计算逐折 CV。该缺口不影响已经复算的验证/最终测试指标，也不把全项目状态变成 BLOCKED。

**WARNING（已知限制，不阻止总体结论）：** E4-1 多重比较选择偏倚；E4-3 仅两个 Seed；512D avgpool 特征没有直接前向来源验证；E4-4 单次耗时未记录；Test 访问锁实现非原子；SHA256 清单有 BOM。详细证据和影响见各专题报告。

## 4. 最终模型指标与独立复算

最终部署配置为五模型概率平均 + 水平翻转 TTA。Validation（375 样本）独立结果：**Accuracy 91.20%（342/375），Macro-F1 90.1471%（报告四舍五入 90.15%）**。最终 Test（380 样本）独立结果：**Accuracy 88.1579%（335/380），Macro-F1 85.5973%（85.60%），Weighted-F1 88.0314%**。最终 Test 概率形状为 380×6，无 NaN/Inf，行概率和约为 1；混淆矩阵与逐类指标与 JSON 一致。

E0 对照的 Test Accuracy 为 308/380（81.0526%），Macro-F1 为 76.8956%；McNemar `b=8, c=35, p=4.1934157934520044e-05`，由保存的逐样本预测独立复算且与 JSON 一致。证据：`metric_recalc.json`、`METRIC_RECALCULATION.md`。

## 5. Checkpoint 完整性与路径

Seed 42–46 的最终部署 Checkpoint 均可独立定位：

`D:\ml_project\E4_FINAL\runs\e4_2\layer4_resize_seed{42..46}\checkpoint_epoch10.pth`

每个文件 45,319,133 字节、124 张量，CPU 可读；彼此 SHA256 不同，且最终 Test JSON 的五个成员引用与磁盘文件哈希匹配。结构及逐张量结论依据 `checkpoint_audit.json`、`CHECKPOINT_AUDIT.md`。因此此项为 **PASS**，不是 BLOCKED。

## 6. 数据划分与 Test 访问纪律

现存协议证据确认使用 Grouped Split V1，manifest SHA256 匹配；训练、验证、测试的路径交集及 Group 交集均为 0，跨集合 `file_sha256` 重复为 0。静态代码检查显示只有 `e4_final_test.py` 访问 Test；访问记录为一次，时间在配置冻结之后。最终预测与标签来自已经保存的 NPZ/JSON。此次状态对账没有打开原图或再次访问 Test。

## 7. 保留的剩余限制

- E4-1 缺少五折 CV 逐折结果及折划分，逐折 CV 独立复算为局部 BLOCKED。
- 512D avgpool 特征来源缺少直接前向验证；当前 CSV/预测一致性属于间接证据。
- E4-4 单次运行耗时缺少记录，报告中的约 20 秒不能由日志独立支持。
- E4-3 只有两个 Seed（42、43）。
- E4-1 在 70 个候选拟合中按 Validation 选择，存在多重比较选择偏倚；未校正 p 值不应作为确认性推断。
- ens3 曾按 Validation 选择成员，但没有用于最终部署；最终部署使用五个 Seed。
- SHA256SUMS.txt 有 UTF-8 BOM，导致首条解析字段携带 U+FEFF；清单中的 E4_EXPERIMENT_CONFIG.json 真实存在且已纳入 88 条登记项。
- E4_FINAL_REPORT.md 的指标数值本身与 JSON/复算吻合，但 §6/§8 有口径混用，已在修正版中调整。

## 8. 归档核验与报告版本

`E4_EXPERIMENT_CONFIG.json` 存在（4,674 字节），且 `integrity_check.json` 将其列入通过校验的登记文件。清单覆盖磁盘上全部应归档文件：88/88，无未登记文件。清单文本字节以 `EF BB BF` 开头（UTF-8 BOM）。由于唯一不匹配是该 BOM 被误读为首条哈希的一部分，原始清单保持原样；本轮不创建或替换哈希清单。五个 Checkpoint 可由上述绝对路径独立定位。指标交叉核对结果见 `metric_recalc.json` 与 `consistency_checks.json`，一致性检查确认 14/14 E4-1 CSV/预测行吻合。

**交付时采用 `E4_FINAL_REPORT_CORRECTED.md` 作为正文版本**，并同时提供本对账文件及专题审计报告。保留 `E4_FINAL_REPORT.md` 作为原始历史版本，不覆盖。修正版已针对集成指标/seed 均值混淆、Test 与 Validation 差距、E4-1 候选数口径等问题修订；剩余限制仍应与本文件第 7 节一并披露。

## 9. 是否需要额外实验

**不需要额外实验即可提交当前 E4 核心结论与审计材料。** 不建议为消除归档或方法学限制而重新访问 Test。逐折 CV、512D 来源直接验证及耗时记录属于材料/追溯限制；若课程评审明确要求这些证据，只能作为后续单独工作处理，不能补写为已验证结果。