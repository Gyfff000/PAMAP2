# 中文学术报告

`main.tex` 为现有主文件，`main.pdf` 为编译结果，`author_info.tex` 保留真实署名填写位置。采用 CTeX A4 排版，中英文摘要、公式、三线表、图题与参考文献。

当前协议为严格 7:3（训练 5187、测试 2223），仅训练集内部三折分组交叉验证。完整片段隔离，同一人的不同片段可进入两侧，论文不再声称未见受试者泛化。旧论文与结果已在 `artifacts/legacy_loso_protocol1.zip` 留档。

所有新表格和结果描述来自 `06_build_report_assets.py`，新图来自 `03_analyze_results.py`，均读取 `outputs/holdout` 的实际运行记录；`asset_manifest.json` 保存来源哈希。修改实验后应重新分析、生成素材并检查全文，不能只手改分数。

项目根目录运行 `paper/build.ps1`，或在当前 Codex LaTeX 编辑器编译现有项目。图像相对路径为 `../outputs/holdout/figures`，须保留仓库结构。AI 参与说明独立保存在根目录 `AI使用记录.md`。

排版随项目附带公开领域的 `placeins.sty`，来源为 CTAN placeins 2.2，用于防止浮动图表越过相关段落。
