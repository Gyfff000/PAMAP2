# 数据来源与许可

本项目使用 Attila Reiss 提供的 PAMAP2 Physical Activity Monitoring 数据集。

- 官方来源：https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
- 数据 DOI：https://doi.org/10.24432/C5NW2H
- 数据许可：Creative Commons Attribution 4.0 International（CC BY 4.0）
- 许可全文：https://creativecommons.org/licenses/by/4.0/legalcode
- 原始论文：A. Reiss and D. Stricker. Introducing a New Benchmarked Dataset for Activity Monitoring. 16th IEEE International Symposium on Wearable Computers, 2012.
- 原始说明另要求引用：A. Reiss and D. Stricker. Creating and Benchmarking a New Dataset for Physical Activity Monitoring. 5th Workshop on Affect and Behaviour Related Assistance, 2012.

原始压缩包不在 Git 仓库中。仓库中的演示窗口、模型评估标签和预测记录来自该公开数据。演示窗口经过列选择、短缺失插值、多相滤波降采样和切窗，不是原始记录的完整替代品。具体处理见项目 README 和论文。

数据作者不对本项目的模型、报告或结论作出认可。源文件和处理后数组的校验值保存在 `outputs/data_source.json` 与 `outputs/data_audit.json`。
