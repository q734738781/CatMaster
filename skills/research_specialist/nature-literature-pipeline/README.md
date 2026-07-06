# nature-literature-pipeline

文献管线引擎 + 每日推送应用层。一套完整的自动化文献发现和跟进系统。

## 这是什么

不是简单的"帮我搜几篇论文"——它是一个结构化管线：多源检索 → 六维评分筛选 → 精读 → 格式化推送 → 归档。支持 cron 每日自动运行。

## 与 nature-academic-search 的关系

`nature-academic-search` 是**按需搜索**（"现在帮我找几篇 XX 的论文"），本 skill 是**自动化订阅**（"每天帮我盯着这个领域，有新论文推送给我"）。互补关系，建议两个都装。

## 管线架构

```
多源检索（30 篇候选）
  arXiv / OpenAlex / Crossref / Semantic Scholar（自动降级）
       ↓
六维粗筛（30 → 5 篇）
  方向匹配 × 35 + 方法论 × 20 + 期刊质量 × 15
  + 网络关联 × 10 + 工程价值 × 10 + 归档价值 × 10
       ↓
精读（top 5）
  标注来源：Full-text / Abstract only / Metadata only
       ↓
推送
  🏅 排名 | 标题 | 期刊 | ⭐ 评分 | 💡 一句话 | 🔬 方法 | 📊 关键数据 | 🧭 点评
       ↓
归档
  DOI/arXiv 去重 → 分类 → 笔记 → 更新索引
```

## 安装

```bash
# Codex
请从这个仓库安装 nature-literature-pipeline：
https://github.com/Yuan1z0825/nature-skills.git
请把 skills/nature-literature-pipeline/ 完整安装，包括 references/ 和 templates/

# Hermes / Claude Code
git clone https://github.com/Yuan1z0825/nature-skills.git
cp -r nature-skills/skills/nature-literature-pipeline ~/.hermes/skills/research/
# 重启 session 或 /reload-skills
```

## 首次配置

装好后告诉 agent：

```
我的研究领域是钙钛矿太阳能电池，关键词：perovskite solar cell, PSC stability,
hole transport layer, interface passivation,
文献推送到飞书群"每日文献"，归档到 ~/research/literature/
```

Agent 会配置关键词、推送目标、归档路径。之后设置 cron：

```
帮我设一个每天早上 8:30 的文献推送，30 篇候选，推送 top 5
```

## 文件结构

```
nature-literature-pipeline/
├── SKILL.md                                ← 技能入口
├── README.md                               ← 本文件
├── references/
│   ├── scoring-system.md                   ← 六维评分体系
│   ├── gap-analysis.md                     ← 文献 gap 分析方法
│   ├── note-template.md                    ← 标准化文献笔记模板
│   ├── push-format.md                      ← 推送消息格式规范
│   ├── cron-setup.md                       ← cron 创建/验证/故障恢复
│   └── review-compilation-workflow.md      ← 综述文献汇编工作流
└── templates/
    └── literature-push-template.md         ← 可分享的通用配置模板
```

## 内置保护

- **评分校验**：每个维度不超过上限，总分自动重算
- **三重去重**：DOI / arXiv ID / OpenAlex ID
- **自动降级**：Semantic Scholar 不可用 → 自动切 OpenAlex + Crossref + arXiv
- **只写文献库**：每日管线只写 `raw/` 目录，不自动改 wiki/知识库

## 常见问题

**Q: 和 nature-citation 怎么配合？**
管线发现新论文 → 用 nature-citation 导出 CNS 格式引用插入手稿。

**Q: 需要什么模型？**
推荐 DeepSeek V3 以上。每日 cron 可用 flash 模型降低成本。

**Q: 我的领域不是材料/化学？**
全部可配置。关键词换成你的领域即可——医学、生物、CS、社科都一样用。

## 作者

十五 (JL Lab) — 基于真实科研工作流构建，已稳定运行数月。
