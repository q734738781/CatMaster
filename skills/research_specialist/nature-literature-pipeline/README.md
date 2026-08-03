# 文献发现与推送管线

该管线完成多源检索、去重、理由化筛选、选择性精读、推送和可选归档。

候选论文只使用三个状态：`selected`、`deferred`、`excluded`。每个状态必须附一条与
本轮任务直接相关的理由，并单独记录访问深度（metadata、abstract、full text、
supplementary/source data）。不对论文质量、证据强度或结论可信度计算综合分。

主要文件：

- `SKILL.md`：工作流和边界
- `references/selection-policy.md`：候选状态判定
- `references/push-format.md`：推送格式
- `references/note-template.md`：归档笔记格式
- `references/gap-analysis.md`：研究空白检索
- `templates/literature-push-template.md`：可配置任务模板

检索失败时保留已经获得的候选和稳定标识；来源不可访问时明确写出 blocker，不从题名、
期刊、作者或引用量推断方法与结论。
