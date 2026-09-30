# ResearchClawBench Material — 2026-08-10

本次使用 `CatMaster-codex-oauth` 完成四个 Material 任务，使用 ResearchClawBench checklist scorer 和 `openai/gpt-5.1` Judge 评分：

| Task | Score | Report |
|---|---:|---|
| Material_000 | 14.10 | [report](reports/Material_000/report.md) |
| Material_001 | 8.75 | [report](reports/Material_001/report.md) |
| Material_002 | 47.67 | [report](reports/Material_002/report.md) |
| Material_003 | 21.35 | [report](reports/Material_003/report.md) |
| **Material mean** | **22.97** | |

评测时删除了上游 scorer 的 `generated_images[:5]` 截断，改为读取全部满足单图大小限制的报告图。ResearchClawBench 的任务说明和 checklist 没有规定最多只能评五张图；本次 `Material_000` 和 `Material_001` 均有六张正文引用图，截断会漏掉第六张。除这一图像输入范围外，任务、checklist、target image、rubric 和 Judge 配置均未修改。

这里只保存四份最终 `report.md` 及其 `images/`。
