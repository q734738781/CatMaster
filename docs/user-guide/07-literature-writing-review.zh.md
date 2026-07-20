# 7. 文献、写作与审稿

[上一章](06-computational-workflows.zh.md) | [目录](README.zh.md) | [下一章](08-remote-execution.zh.md)

文献、写作和审稿共享项目文件，但证据职责不同。Literature Review 负责发现、阅读和引用定稿；Writing 负责基于现有材料生产文稿；Peer Review 负责从审稿人和编辑视角检查一份固定 PDF。把三类任务分开，能减少伪引用、版本混乱和“边找边编”的文本。

## 7.1 检索服务配置

常用环境变量：

```bash
export TAVILY_API_KEY="<KEY>"
export SEMANTIC_SCHOLAR_API_KEY="<KEY>"
export OPENALEX_API_KEY="<KEY>"
export NCBI_API_KEY="<KEY>"
export CROSSREF_MAILTO="you@example.org"
```

不是每项都必需。缺少某个服务时，agent 会使用当前可用的搜索和本地语料路径。API key 只提供访问能力，不保证全文权限或元数据正确。

## 7.2 受控浏览器

安装命令见[快速安装](01-quickstart.zh.md)。可选会话设置：

```bash
export CATMASTER_AGENT_BROWSER_PROFILE="$HOME/.config/catmaster/browser-profile"
export CATMASTER_AGENT_BROWSER_HEADED=true
```

或尝试连接已运行的 Chrome：

```bash
export CATMASTER_AGENT_BROWSER_AUTO_CONNECT=true
```

首次机构登录通常需要 headed 模式。Profile 必须位于项目空间之外，并设置为当前用户私有。不要把密码、cookie、OTP、session 导出或浏览器状态写入 `.env.local`、YAML、prompt 或 `files/`。

遇到 CAPTCHA、二维码、短信验证、许可确认或浏览器安全警告时，agent 应停下，由用户操作。CatMaster 不绕过付费墙或访问控制。

## 7.3 Literature Review 工作流

一个可审计的综述流程：

1. 明确研究问题、日期范围、材料/反应体系、文献类型和排除条件。
2. 记录数据库、网页入口和检索式。
3. 对 DOI、标题和版本去重。
4. 区分发现记录、摘要、全文和补充信息。
5. 对关键论文提取方法、体系、比较条件、结果和限制。
6. 建立 claim 到 evidence 的映射，不用一篇论文支撑过多不同主张。
7. 用 citation finalizer 核对 DOI、作者、期刊、年份和可导出记录。
8. 保存检索日志、证据表、未获取清单和最终综述。

推荐目录：

```text
literature/
  query.md
  corpus/
  metadata/
  evidence.csv
  unavailable.md
  references.bib
  review.md
```

## 7.4 证据级别

报告应标明信息来自：

| 级别 | 可以支持什么 |
|---|---|
| 检索结果或元数据 | 论文存在、标题、作者、期刊、年份等发现信息 |
| Abstract | 摘要明确陈述的主要目的和结果 |
| Full text | 方法、数值、限定条件、图表和讨论 |
| Supplementary Information | 详细实验、计算参数、扩展数据和额外图表 |
| 用户提供数据 | 与其来源、版本和完整性相匹配的主张 |

没有读到全文时应明确说明。不要从标题推断方法，不要从摘要补写精确参数，也不要把预印本和期刊版本重复计数。

## 7.5 本地语料库

已有 PDF、Markdown 或表格可以上传到 `literature/corpus/`，再要求 Literature Review ingest 和 query。大型语料建议同时维护 manifest：

```text
source path
DOI or stable identifier
title
document version
access date
parse status
notes
```

解析成功不代表图表、公式和补充材料全部被完整提取。对核心结论应回到 PDF 页面或 publisher HTML 核查。

## 7.6 Writing 的输入合同

Writing 请求应明确：

- 目标文种、期刊或读者。
- 文稿类型和本轮章节。
- 可用证据文件、表格、图和引用库。
- 必须保留的数字、术语、引用和结论边界。
- 禁止添加的内容，例如新实验结果、未核实引用或因果表述。
- 输出格式、文件名和是否需要修订记录。

示例：

```text
使用 notes/result_contract.md、calculations/summary.csv 和
writing/references.bib 起草 Results 的两个小节。保持所有数值和误差不变，
不要补引用；每一段末尾列出对应证据文件。输出到 writing/results_v1.md。
```

## 7.7 起草、润色与事实保持

Writing worker 可以重组论证、起草章节、生成图件和编译文件。Polisher 只做保守语言改进。终稿检查：

- 数字、单位、符号和误差是否与源文件一致。
- 引用是否真的支持相邻 claim。
- 相关性是否被误写成因果。
- 结论强度是否超过数据范围。
- 方法和结果是否混写。
- 限制、不确定性和失败 case 是否被删除。
- 图号、表号、补充材料和交叉引用是否一致。

任何自动润色都不能替代 author review。对重要段落保留修改前版本或 diff。

## 7.8 图件、Markdown PDF 和 TeX

Writing 能调用图件和编译工具。提供数据、图的结论、panel 逻辑、单位、颜色限制、输出格式和期刊尺寸要求。科研图应保留生成脚本和源数据。

Markdown PDF 路径通常需要 Pandoc、Chrome/Chromium、Fontconfig 和合适的 CJK 字体。LaTeX 需要 `pdflatex`，带 bibliography 时还需要 `bibtex`。编译成功后仍应检查生成 PDF 的空白页、溢出、图片裁切、字体替换、公式和链接。

环境配置见[部署、运维与安全](10-deployment-operations.zh.md)。

## 7.9 Peer Review 工作流

准备一份 canonical PDF，例如：

```text
writing/submission/manuscript.pdf
```

请求中说明目标期刊或审稿标准、文章类型、是否包含 Supplementary Information，以及希望关注的方法学或报告规范。不要让多个同名 PDF 留在不同目录而不说明主版本。

Peer Review 按 `peer_review_models` 逐个生成 reviewer report，再生成 editor synthesis。建议产物：

```text
writing/review/
  reviewer_1.md
  reviewer_2.md
  reviewer_3.md
  editor_synthesis.md
  review_memo.md
```

审稿输出是诊断材料，不是事实裁决。用户应回到论文页码、原始数据和方法文件核实每项批评。

## 7.10 从审稿转入修稿

不要直接要求 Peer Review 改稿。先确认哪些意见接受、部分接受或拒绝，然后在 Writing 或 Research 中提供：

- canonical manuscript 源文件。
- reviewer 和 editor 产物。
- 每条意见的决定与证据。
- 允许修改的章节。
- 是否需要 response letter 和 marked manuscript。

修订后重新编译 PDF，再用新的 canonical PDF 做一次独立检查，避免只审查旧版本。

## 7.11 交付清单

完整的文献和写作交付通常包括：

- 检索式与日期。
- 去重后的文献记录和稳定标识符。
- 全文可用性与证据等级。
- claim-evidence 表。
- 引用库和无法核实的条目。
- 可编辑源文稿、图源和编译后的 PDF。
- reviewer 原始报告、editor synthesis 和修订决定。
- 对数据、引用、版面和事实保持的最终检查记录。
