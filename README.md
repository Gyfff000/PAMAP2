# PAMAP2：多传感器融合与跨受试者人体活动识别

模式识别课程大作业。使用 UCI 官方 PAMAP2 原始记录，完成 12 类监督分类、8 折留一受试者评估、模型对比与传感器消融。真实实验结果由保存的逐窗口预测重新计算，论文不会引用预设或虚构成绩。

## 已完成的实验结果

| 主模型 | 汇总 Macro-F1 | Accuracy |
| --- | ---: | ---: |
| 逻辑回归 | 0.8853 | 0.8885 |
| 随机森林 | **0.9279** | **0.9236** |
| 早期融合 CNN | 0.8132 | 0.8086 |
| 部位与统计特征融合 | 0.9223 | 0.9232 |

十个预设配置的八十次外层评估均已完成；每个保存模型重新加载后的预测与原始逐折记录完全一致，核验见 `outputs/verification.json`。六项协议测试通过。最终选用随机森林，并在全部八名受试者上重新拟合；其泛化成绩以上述 LOSO 评估为依据，不另宣称独立测试成绩。

中文报告为 [paper/main.pdf](paper/main.pdf)，共十二页。模型误差、传感器对照、个体差异和实验局限均保留在报告中。

## 在 Visual Studio 打开

打开 `PAMAP2.sln`，启动项目为 `PAMAP2`，启动文件为 `main.py`。项目配置指向本目录的 `.venv`。本机首次环境复用已有 Python 3.14 全局的 NumPy、SciPy、PyTorch 与 Matplotlib，并在项目 `.venv` 安装 scikit-learn 及依赖；版本见 `outputs/environment.json`，完整运行所需版本见 `requirements-lock.txt`。其他电脑建议创建不继承全局包的干净虚拟环境。

按 **Ctrl+F5** 展示保存的完整结果，不会重新训练。将 `05_predict_demo.py` 设为启动文件可运行真实模型预测演示；该演示自带少量公开数据窗口和未训练过演示受试者的模型，无需下载完整数据。

## 项目入口

| 文件 | 功能 |
| --- | --- |
| `00_download_data.py` | 从 UCI 官方下载、校验、解压 Protocol 数据，记录 SHA-256 |
| `01_prepare_data.py` | 缺失值处理、断点分段、抗混叠降采样、窗口切分、150 维特征 |
| `02_run_experiments.py` | 四种主模型和六种传感器配置对照，共十个配置的八折评估 |
| `03_analyze_results.py` | 从外层预测核验成绩，生成图表和受试者配对分析 |
| `04_train_deployment.py` | 全部八人重训演示/部署模型，另导出严格隔离的少量预测示例 |
| `05_predict_demo.py` | 加载保存模型进行预测，并核查文件与特征一致性 |
| `main.py` | VS 默认入口，展示已有评估结果 |
| `paper/main.tex`、`paper/main.pdf` | 中文学术报告源码与 PDF |
| `AI使用记录.md` | 独立保存的 AI 参与说明及关键提示 |

## 数据与处理

- 数据源：[UCI PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)，DOI `10.24432/C5NW2H`，CC BY 4.0。原始数据不放进 Git。
- 采用 Protocol 的 12 类：躺卧、坐、站立、步行、跑步、骑车、北欧式行走、上楼、下楼、吸尘、熨烫、跳绳。
- 主实验为受试者 101–108。109 仅有部分 Protocol 记录，保留审计但不纳入主实验；101–108 也不是每个人都完整包含 12 类，具体见 `outputs/class_subject_counts.csv`。
- 每部位使用 ±16g 三轴加速度和三轴陀螺仪，共 18 通道。不使用心率、磁力计、温度、重复量程加速度或无效姿态值。该选择减少心率低采样率缺失以及磁场环境差异的影响，不是比较后挑选的配置。
- 原始 100 Hz 记录先按受试者、活动变化、时间跳变分段。传感器连续缺失超过 0.5 秒的区域排除；短缺失只在当前段内线性插值，端点采用最近有效值。
- `resample_poly` 抗混叠降采样至 50 Hz；窗口长 5.12 秒（256 点），步长 2.56 秒。窗口不跨活动、长缺失或时间跳变，得到 7,410 个窗口。
- 任务是**已知活动内部的离线窗口分类**。使用真实活动边界排除过渡，因此不是在线分段、未知活动检测或任意连续流部署评估。

## 评价协议与模型

配置在结果检查前写入 `configs/experiment.json`。每次留一个人做外层测试，其余七人为训练。神经网络先以循环顺序的下一位受试者作为内层验证，在六人上选训练轮数，再重新初始化并在全部七人上训练选定轮数。标准化参数和类别权重只来自当前训练数据。

主模型为统计特征逻辑回归、统计特征随机森林、18 通道早期融合 CNN、三个局部 CNN 与统计特征 MLP 拼接的融合模型。常见拼接结构不作为新架构创新。随机森林另做手腕/胸部/脚踝单独输入，以及逐一移除部位的对照；这些结果不能被当作神经网络分支的因果解释。

主指标是合并全部外层预测后固定 12 类的 Macro-F1。补充给出各受试者实际出现类别的 Macro-F1 均值与样本标准差；原始固定 12 类个人分数也保留。两者定义不同，不能混合比较。不存在另一个未触碰的独立最终测试集；LOSO 本身用于模型评估，挑选最好结果仍可能产生选择偏差。

## 从头复现

在项目目录的 PowerShell 中执行（Python 3.14）：

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
```

数据压缩包约 656 MB，解压、缓存和各折模型需要额外磁盘空间；当前实验使用 CPU。下载失败会重试。训练入口可通过 `--models cnn fusion`、`--subjects 101 102` 选择补跑配置/受试者，已完成折会校验配置及测试标签后跳过。更改实验配置后须使用新输出目录或备份旧结果并重新运行，不能把不同配置结果混在一起。PDF 编译使用 XeLaTeX/CTeX；本机已配置 TeX Live 2026，其他电脑也可将完整仓库上传至支持中文字体的 LaTeX 环境编译。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖通道索引、受试者隔离、长缺失处理、部位特征独立性、标准化拟合范围及网络梯度。结果分析另检查每个窗口恰好外层测试一次、存储成绩与逐样本预测重算一致。

本机保留了全部中间模型，可额外运行 `tools/verify_saved_models.py` 重放核验八十个模型；从 GitHub 新克隆时需先重新完成训练，才具备全部中间模型。随仓库提供的独立演示不依赖这些中间文件。

## 留档与同步范围

GitHub 保存源代码、VS 项目、固定配置、环境锁定、审计记录、每折预测与指标、图表、论文和可运行的模型示例。大体积原始数据、缓存、虚拟环境及全部中间折模型仅在本机保留。`artifacts/final` 是全部八人拟合的部署模型，不能用其训练集成绩代替泛化评估；`artifacts/demo` 使用外层测试人为 101 的模型。

学生署名保留为空，提交前应填写真实姓名、班级、学号，并核查内容。AI 参与如实记录在独立文件中。
