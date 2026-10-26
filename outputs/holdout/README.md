# 当前实验：严格 7:3 + 训练内三折分组验证

- `split_assignments.csv`：逐窗口冻结清单；partition=0 是测试集，1/2/3 为训练集的三个验证折。
- 训练 5187，测试 2223；每折训练 3458、验证 1729。所有完整片段互斥，十二类均覆盖，每个训练窗口验证一次。
- `split_manifest.json` 与 `split_class_counts.csv`：划分哈希、分配约束和各类数量；每类比例受整段约束，精确 7:3 针对总体。
- `cv1`、`cv2`、`cv3`：十种模型每折的指标、训练来源、保存模型与预测哈希。
- `cv_summary.csv`、`model_selection.json`：训练内交叉验证及测试之前冻结的模型选择。
- `test`：十个在完整训练集拟合后，对同一测试集的预测与指标。
- `model_comparison.csv`、`model_summary.json`：当前最终结果，论文取数来源。
- `result_audit.json`：比例、隔离、覆盖和逐预测指标重算。
- `verification.json`：四十个保存模型回放与测试日志、当前 PDF 哈希。

同一人的不同片段可出现在两侧。这不是受试者隔离，也不是未见人员泛化评价。历史 LOSO 用过同批公开记录，本次不称为全新的外部数据。中间模型只在本机，预测随 `artifacts/holdout_predictions.zip` 提供。
