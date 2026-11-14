# RealWaste 数据来源记录

Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste [Dataset]. UCI Machine Learning Repository. DOI: [10.24432/C5SS4G](https://doi.org/10.24432/C5SS4G)。访问日期 2026-11-10。

- [UCI 数据说明](https://archive.ics.uci.edu/dataset/908/realwaste)：4752 张、九类材质、524×524，来源为真实垃圾处理场景；页面列出 CC BY 4.0。
- [原作者仓库](https://github.com/sam-single/realwaste)：相同类目与公布数量，README 标注 CC BY-NC-SA 4.0。
- [IEEE DataPort 作者页面](https://doi.org/10.21227/65m5-rr72)：同样标注 CC BY-NC-SA 4.0。
- 论文：Single, S.; Iranmanesh, S.; Raad, R. *RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning*. Information 2023, 14, 633. [10.3390/info14120633](https://doi.org/10.3390/info14120633)。只借鉴其数据场景说明，本轮没有复现或直接比较论文内模型成绩。

因 UCI 整包下载过慢，未完成的下载停止并保留部分文件；在任何外部模型预测前改为原作者固定 commit `156e8e9154eeb163df30fcfd767f55b19633bb06` 的五类原始 JPG 文件。版本由 `git ls-remote` 核实。预期完整文件编号范围为各类 1 至公布数量；任何缺失下载均导致失败，不能依据预测补选图片。

本地 ZIP 是这五类文件的汇集，不是 UCI 官方 ZIP 的字节副本；ZIP、每个文件和解码像素的 SHA256 记录在审计文件及清单。原 UCI 尝试计划保存在 `outputs/external_realwaste_uci_attempt_plan.json`，对应计划文字保存于本 sources 目录。外部评估用途、模型、评分和排除规则没有因获取方式改变。

许可元数据存在不同口径，本地课程研究按更严格的非商业、署名与相同方式共享要求使用；没有上传或重新分发图片。导出包含图片的报告或幻灯片时也需保留数据来源与适用许可。
