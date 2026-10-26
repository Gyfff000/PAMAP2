# PAMAP2：多传感器融合人体活动识别

模式识别课程大作业，采用 UCI 官方 PAMAP2 数据完成十二类监督分类、模型对比与传感器输入对照。当前协议已按老师要求改为**训练集与测试集严格 7:3，并且仅在训练集内部进行三折分组交叉验证**。旧留一受试者实验单独留档，不作为本次提交的主要成绩。

## 当前实验结果（协议 2）

| 主模型 | 训练集三折 CV F1（均值 ± SD） | 测试 Macro-F1 | 测试 Accuracy |
| --- | ---: | ---: | ---: |
| 逻辑回归 | 0.8830 ± 0.0327 | 0.7998 | 0.8241 |
| 随机森林 | 0.9105 ± 0.0082 | 0.9261 | 0.9213 |
| 早期融合 CNN | 0.7935 ± 0.0704 | 0.5693 | 0.5879 |
| 部位与统计特征融合 | 0.8978 ± 0.0418 | 0.8526 | 0.8444 |

7,410 个窗口全部使用：训练集 **5,187（70%）**，测试集 **2,223（30%）**。完整连续活动片段只属于一个数据部分，避免 50% 重叠窗口及同段滤波上下文跨越训练/测试。三个验证折各 1,729 窗口，每折训练 3,458 窗口；每个训练样本恰好验证一次，测试集完全不参与交叉验证。

十个预设配置共完成 **30 次交叉验证拟合 + 10 次最终拟合**。依据四种主模型的训练 CV 均值，先选定随机森林，再生成测试预测。最终模型使用 70% 训练集拟合；演示导出的是同一个已评价版本，没有在全部数据上重训。全部四十个保存模型的回放与原预测完全一致，核验见 `outputs/holdout/verification.json`。

中文报告为 [paper/main.pdf](paper/main.pdf)。正文、摘要、图表及结论均使用新协议的真实运行成绩。署名仍为空，提交前填写真实姓名、班级、学号，并理解、审核全文。

## 给助教展示

1. 在 Visual Studio 打开 `PAMAP2.sln`，将 `main.py` 设为启动文件，按 Ctrl+F5：展示 7:3 数量、三折交叉验证成绩及最终测试成绩。当前个人 VS 启动设置予以保留，可自行切换入口。
2. 打开 `outputs/holdout/split_manifest.json` 与 `split_class_counts.csv`：说明 5,187/2,223 比例、每类覆盖和三个验证折。
3. 打开 `paper/main.pdf` 的实验协议和结果部分：交叉验证只使用训练集，测试成绩单列。
4. 将 `05_predict_demo.py` 设为启动文件：加载真实保存模型，对固定的 24 个测试窗口回放。选样不按预测正确筛选，演示正确率不能替代完整测试集成绩。
5. 展示原始数据时，打开 `data/raw/Protocol` 的 `.dat` 文件，运行 `07_inspect_raw_data.py`，查看 54 列原始值、标签与缺失值。原始预览不做插值、降采样或训练。

注意：本次隔离的是**完整录制片段**，不是受试者；同一人的不同片段可能进入两侧，所以论文不再声称未见受试者泛化。此前的 LOSO 曾评价同批公开数据，本次是课程要求下的协议修订，不是全新的外部数据测试。

## 项目入口与结果目录

| 文件 | 功能 |
| --- | --- |
| `00_download_data.py` | 下载、校验、解压 UCI 官方数据 |
| `01_prepare_data.py` | 分段、缺失值处理、降采样、切窗、150 维统计特征 |
| `02_run_experiments.py` | 训练集内三折交叉验证、冻结模型选择、70% 重训与 30% 测试 |
| `03_analyze_results.py` | 核验四十次预测、生成新协议图表与结果表 |
| `04_train_deployment.py` | 导出 CV 选定的 70% 训练模型及固定测试示例，不进行全数据重训 |
| `05_predict_demo.py` | 模型回放与特征一致性检查 |
| `06_build_report_assets.py` | 从核验记录生成论文数字和表格 |
| `07_inspect_raw_data.py` | 标准库只读原始数据预览 |
| `tools/verify_saved_models.py` | 重新加载四十个已训练模型，逐窗口比对原预测 |
| `src/pamap2/protocol.py` | 冻结的完整片段分组及训练内三折映射 |
| `outputs/holdout` | **本次提交依据**：划分、CV、测试、选择记录、核验和图表 |
| `artifacts/holdout_predictions.zip` | 当前四十份验证/测试预测，分析时自动恢复 |
| `artifacts/legacy_loso_protocol1.zip` | 旧配置、代码、论文与结果快照；旧预测另见 `outer_fold_predictions.zip` |
| `artifacts/final`、`artifacts/demo` | 新协议选定模型与可直接运行的小型示例 |

`outputs` 根目录其他旧模型表、`outputs/folds`、`outputs/figures` 是协议 1 历史记录，不与当前结果混用。`outputs/data_source.json`、`data_audit.json`、`class_subject_counts.csv` 仍是两种协议共同的数据来源与预处理审计。

## 数据与模型

- 数据源：[UCI PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)，DOI `10.24432/C5NW2H`，CC BY 4.0；原始数据不上传 Git。
- 只用 Protocol 十二类活动，受试者 101–108。109 录制不完整，仅保留审计。
- 每部位使用 ±16g 三轴加速度与三轴角速度，共十八通道；不使用心率、磁力计、温度、重复量程加速度及无效姿态列。
- 在活动变化、时间跳变、长缺失处切分；超过 0.5 秒的连续缺失区域排除，短缺失仅在当前段插值。`resample_poly` 从 100 Hz 降至 50 Hz；256 点窗口、128 点步长，形成 7,410 个窗口。
- 106 个完整活动段分配到测试和三个训练验证子集。根据片段标签与数量尽量保持类别分布，硬性满足精确总数、片段隔离与每部分十二类覆盖。限时整数分配求得可行解，不宣称类别比例偏差全局最优；逐窗清单与哈希已提交，复现直接使用冻结清单，勿删除后反复生成挑分。
- 跳绳仅四个完整片段，留出一段测试后剩三段，因此用三折分组验证以保证每个验证折十二类齐全；五折会有验证折缺少跳绳。
- 模型为逻辑回归、240 棵树随机森林、早期融合 CNN、部位时序与统计特征融合网络；另做六种随机森林部位对照。标准化和类别权重只从当前训练部分估计。神经网络每次固定训练 25 轮，不按测试或验证挑选轮数。
- 主指标固定十二类 Macro-F1，CV 报告折均值和样本 SD，测试分数单列。个体图按该人在测试中出现的类别计算，属描述性分析。
- 任务是标注活动内部的离线窗口分类，使用真实活动边界及同段前后信号，不评价在线分段或未知活动检测。

## 从头复现

本机环境为 Python 3.14、CPU；依赖版本见 `requirements-lock.txt`，实际环境见 `outputs/holdout/environment.json`。其他电脑建议使用干净虚拟环境。

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe 00_download_data.py
.\.venv\Scripts\python.exe 01_prepare_data.py
.\.venv\Scripts\python.exe 02_run_experiments.py
.\.venv\Scripts\python.exe 03_analyze_results.py
.\.venv\Scripts\python.exe 04_train_deployment.py
.\.venv\Scripts\python.exe 05_predict_demo.py
.\.venv\Scripts\python.exe 06_build_report_assets.py
.\paper\build.ps1
.\.venv\Scripts\python.exe tools/verify_saved_models.py
```

训练入口支持 `--models cnn fusion` 补跑指定模型的 CV，或 `--cv-only` 仅完成验证和冻结选择。全部十个配置的 CV 完成后，才允许最终测试。已有模型、预测和配置哈希一致时跳过对应拟合；从 GitHub 新克隆时没有本地中间模型，训练入口会重新拟合，不能只凭已有分数跳过。旧的 `--subjects` 不适用于当前协议。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

八项测试覆盖精确比例、训练内 CV、片段泄漏拦截、通道映射、缺失处理、部位特征、训练标准化及网络计算及验证折类别覆盖。分析程序另核验全部四十份预测与选择依据。原始压缩包约 656 MB，原始数据、缓存、虚拟环境及中间模型仅在本机保留。

`AI使用记录.md` 独立保存 AI 参与说明，包含代码生成、实际实验执行、论文初稿与本次协议修订。最终内容由学生审核、理解并承担提交责任。
