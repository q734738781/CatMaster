---
name: nature-ref-verifier
description: >-
  学术参考文献多源交叉验证技能。逐条对比作者、标题、年份、卷期、页码，
  标记卷年/DOI年冲突、作者顺序异常、页码偏差等问题，输出结构化验证报告。
---

# nature-ref-verifier

## 用途

逐条验证参考文献的元数据准确性。适合论文提交前自查、审稿意见回复时的引用核查、Zotero 库定期养护。

## 工作流概要

1. **解析输入** — 接受整篇论文的参考文献列表、单条引用、BibTeX 文件或 Zotero item key
2. **多源并行查询** — 根据可用工具，同时查 Crossref / IEEE Xplore / 网络搜索 / CNKI / Zotero
3. **字段级对比** — 逐字段比对，按严重程度分为 🔴 必须修正 / 🟡 建议核对 / 🟢 仅供参考
4. **置信度评估** — 输出 ✅ Verified / ⚠️ Check suggested / ❌ Needs fix / ❓ Unverifiable
5. **输出报告** — Markdown 摘要 / BibTeX patch / Zotero 更新指令

## 设计背景

本技能源于实际开题报告参考文献核查中遇到的问题：

| 问题类型 | 示例 | 发现方式 |
|---------|------|---------|
| 作者完全编造 | Smogavec P → Leckebusch J | CrossRef 查 DOI |
| 作者顺序颠倒 | Vainikainen→Mikhnev | IEEE 官网 |
| 卷年 vs DOI 年 | Dadrass 卷2025/上线2024 | 多源对比 |
| 页码完全错误 | Noon: 1444-1449→1309-1319 | IEEE Xplore |
| 中文作者名搞混 | 赵延刚→赵廷刚 | 知网 |
| DOI 指向不同论文 | Abidi: 错误 DOI | CrossRef |

单一搜索引擎（谷歌学术 / Bing / Crossref）都不足以覆盖所有问题，必须多源交叉验证。

## 环境和依赖

可选工具，有则启用：

- `zotero-cli` / `zotero-mcp` — 读取 Zotero 库、修正条目
- `kimi-datasource (scholar)` — 学术搜索
- `kimi-webbridge` — 带校园网登录态的知网查询
- `WebSearch` / `FetchURL` — 通用网络搜索兜底

## 使用方式

直接告诉 agent：

```
帮我校验这篇论文的参考文献
```

```
验证这条引用：Farquharson G, Langman A. ... 1999
```

```
检查我的 bib 文件，输出修正后的版本
```
