# `nature-writing` 技能

`nature-writing` 是 Writing lane 唯一的通用科学论文写作 skill。它根据作者提供的 claims、图表、结果、笔记或中文草稿，起草或重建 Nature 系列及其他期刊的手稿，并按研究设计加载必要的报告规范。

## 功能

`nature-writing` 可帮助撰写：

- 标题
- 摘要
- 引言
- 结果叙事
- 讨论
- 结论
- significance paragraph
- 手稿大纲
- 通用 IMRAD 与研究设计报告规范

该技能用于论证构建和章节起草。如果已有英文草稿只需要句子级润色，应使用 `nature-polishing`。

## 来源基础

该技能来自对材料、能源系统、建筑脱碳和机器学习等领域 Nature 与 Nature Communications 研究论文的细读，并结合仓库中已有的 writing-strategy 规则。

章节级写作和面向审稿人的自审规则也参考了彭思达老师公开的科研写作笔记：

- https://pengsida.notion.site/c1a22465a0fa4b15a12985223916048e
- https://github.com/pengsida/learning_research

## 文件结构

```text
nature-writing/
├── README.md
├── SKILL.md
└── references/
    ├── abstract.md
    ├── article-architecture.md
    ├── chinese-author-workflow.md
    ├── conclusion.md
    ├── experiments.md
    ├── introduction.md
    ├── method.md
    ├── nature-summary-paragraph.md
    ├── paper-review.md
    ├── paragraph-flow.md
    ├── reporting-standards.md
    ├── related-work.md
    └── examples/
```

## 核心规则

| 领域 | 规则 |
|---|---|
| 证据优先 | 不编造数据、机制、统计量、样本量或创新性 |
| 发布会主线 | 识别证据最强的发表价值，并围绕它重构标题、章节、实验和图表 |
| 摘要 | 重要问题、关键缺口、独特方法、最强结果和意义 |
| 引言 | 领域尺度、瓶颈、已有尝试、未解决缺口、本文工作；面向 `Nature` 时使用分阶段 summary-paragraph funnel |
| 方法 | 解释模块动机、设计、正向流程和技术优势 |
| 结果 | 构建证据阶梯，而不是按实验流水账写作 |
| 实验 | 每个实验承担明确论证职责，只选择核心 claim 真正需要的比较、消融、机制或场景证据 |
| 报告规范 | 根据研究设计按需加载 CONSORT、STROBE、PRISMA 等当前规范；规范控制完整性，发布会主线控制叙事重点 |
| 讨论 | 解释优势为何成立、与恰当既有工作的关系及其科学或应用意义 |
| 自审 | 检查主线是否鲜明、主张与证据是否一致，并清除防御性写作和无关技术细节 |
| 中文笔记 | 翻译意图和论证，不照搬中文句序 |
