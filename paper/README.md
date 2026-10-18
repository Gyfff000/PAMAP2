# 中文学术报告

`main.tex` 为主文件，`main.pdf` 为编译结果，`author_info.tex` 用于填写真实署名。使用 CTeX 的 A4 学术排版、中英文摘要、编号公式、三线表、图题、文内引用和参考文献。

表格与结果描述来自 `06_build_report_assets.py`，图像来自 `03_analyze_results.py`，均以 `outputs` 中真实实验记录为依据。`asset_manifest.json` 记录来源哈希。不要独立改写表格里的实验数字；若改动实验，应重新执行分析和素材生成，并核查正文与结果一致。

在项目根目录运行 `paper/build.ps1` 进行两次 XeLaTeX 编译，或用 Codex LaTeX 编辑器打开 `main.tex`。编译不运行训练代码。图表位于上级 `outputs/figures`，因此复现排版时须保留完整仓库结构。

AI 参与说明单独保存在项目根目录的 `AI使用记录.md`；论文不嵌入该说明。
