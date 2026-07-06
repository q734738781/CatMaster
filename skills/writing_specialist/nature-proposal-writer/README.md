# researchwrite

Proposal-first 科研写作状态机。不是"帮我写论文"——它强制执行写作前的论证架构，写完跑四层 QA pipeline。

## 这是什么

三个模式 + 四层质量闸门：

```
你的输入                  模式           做什么
─────────────            ─────          ──────────────────
题目、方向、模糊想法  →  compose       9步：从 research canon 到 .docx 导出
已有段落/章节          →  revise        9步：差距分析 → 对比润色前后
已有草稿 + 要扩写      →  hybrid        compose + revise 组合
```

写完后自动跑 QA：

```
Gate 2: professor 专家审查（内容层）
  ├── 论文 → 方法论专家 + 领域专家
  ├── proposal → 可行性专家 + 创新性专家
  └── 综述 → 覆盖面专家 + 批判深度专家
      ↓
Gate 1: avoid-ai-writing（语言层，仅英文）
      ↓
Gate 3: 自动校验（citation？可复现？编号连续？）
      ↓
Gate 4: 评分阈值（≥7.0 通过，<7.0 定向回退 ≤3 轮）
```

## 核心原则

| # | 原则 | 说明 |
|---|------|------|
| 1 | 证据先于文字 | 起草前必须建立 research_canon 和 evidence_table |
| 2 | 论证先于章节 | 写正文前必须完成 argument_map |
| 3 | 契约先于段落 | 每节需要 purpose / allowed claims / forbidden claims |
| 4 | 动态专家 | 按失败模式召唤对应审查专家 |
| 5 | 内容先于语言 | 诊断科学逻辑后再做语言打磨 |
| 6 | 该停就停 | 平台期、证据缺失是停止理由 |

## 安装

```bash
# Hermes / Claude Code
git clone https://github.com/Jiahao8595/research-pipeline.git
cp -r research-pipeline/researchwrite ~/.hermes/skills/
```

安装后 `/reload-skills`。

**前置依赖：**

```bash
hermes skills install brainstorm     # 入口追问
hermes skills install professor      # 动态专家审查
hermes skills install avoid-ai-writing  # 英文去 AI 味
hermes skills install docx           # Word 文档导出
```

## 使用示例

```
# 从零写 proposal
"用 researchwrite 帮我写一个关于钙钛矿稳定性优化的研究计划"

# 审查已有文本
"用 researchwrite 审查这段 discussion，paper 挡位"

# 快速扫读
"快速扫一下这个摘要"（只标记问题，不全跑 QA）
```

## 四挡位

| 挡位 | 场景 | 阈值 | 说明 |
|------|------|:---:|------|
| `paper` | 投稿论文 | 7.0 | 全跑 |
| `proposal` | 研究方案/开题 | 7.0 | 全跑 |
| `internal` | 内部汇报 | 5.0 | 跳过专家审查 |
| `quick` | 快速扫读 | — | 只标记 P0 问题 |

## 文件结构

```
researchwrite/
├── SKILL.md                       ← 技能入口
├── README.md                      ← 本文件
├── references/
│   ├── evaluation-rubric.md       ← 8 维 × 4 锚点评分体系
│   ├── stopping-rules.md          ← 迭代终止条件
│   ├── foundation-files.md        ← 建立 foundation 五文件
│   └── validation-checklist.md    ← 自动校验清单
├── scripts/
│   └── build_proposal_docx.py     ← .md → .docx 构建脚本
└── templates/                     ← 空模板（新建项目用）
    ├── 00_scope.md
    ├── 01_research_canon.md
    ├── 02_evidence_table.md
    ├── 03_argument_map.md
    ├── 04_section_contracts.md
    ├── 05_style_guide.md
    ├── qa_report.md
    └── revision_brief.md
```

> **完整 reference 文件**（compose/revise/hybrid 模式详解、专家分派、导出归档、综述框架等 12 份）未包含在公开版本中。如需获取，请通过 [GitHub issue](https://github.com/Jiahao8595/research-pipeline/issues) 或邮件联系作者。

## 作者

十五 (JL Lab) — 基于博士研究实战构建的写作框架。
