# 7. 文献、写作与审稿 Agent

[上一章](06-computational-workflows.zh.md) | [目录](README.zh.md) | [下一章](08-remote-execution.zh.md)

Literature Review、Writing 和 Peer Review 使用同一个 workspace，却承担三种不同的证据责任。Literature Review 负责找到并核实材料；Writing 负责用已有证据形成文稿和其他交付物；Peer Review 负责站在独立审稿人的位置检查一份固定稿件。把这三类角色分清，能避免一边写作一边补造证据，也能让审稿意见与实际修订保持可追溯。

## Literature Review Agent：从发现论文到证据库

Literature Review 可以处理快速查证，也可以承担较完整的主题综述。任务规模由研究问题决定。一个精确事实可能只需少量高质量来源；一篇 perspective 式综述则需要更宽的候选集合、清楚的检索边界和系统的筛选记录。Agent 会根据 prompt 判断深度，但用户最好说明时间范围、材料或反应体系、文献类型、排除条件和最终用途。

### 发现不是精读

公共网页搜索适合发现论文、项目页和数据库记录。搜索结果通常只能证明某篇论文存在，并提供题名、作者、摘要片段或 DOI。Agent 会把这些发现记录与真正读到的摘要、全文和 Supplementary Information 区分开。

如果论文可通过开放获取或用户本人授权的机构会话访问，受控浏览器可以打开动态页面并下载合法全文。遇到 CAPTCHA、二维码、OTP、许可确认或安全警告时，Agent 会停下来让用户操作，不会尝试绕过访问控制。无法取得全文的论文仍可留在候选表中，但不能用于支持摘要之外的精确方法和数值。

```text
使用 Literature Review 查找 2018 年至今关于单原子催化剂动态聚集和再分散的原位证据。
先做广泛发现并保存完整候选表，再根据是否包含 operando/in situ 表征、
是否直接讨论聚集或再分散、以及是否能取得全文筛选核心论文。

把数据库记录、摘要和全文证据分开。对核心论文提取催化剂、气氛、温度、
表征方法、观察到的动态过程、作者解释和主要限制。不要把搜索摘要改写成精读结论。
```

### 本地语料让项目材料可以反复查询

你可以把已有 PDF、Markdown、DOCX 或表格放进 `literature/`，让 Agent 导入本地 corpus。导入工具会建立可检索文本和来源记录，之后可以围绕多个问题反复查询。对于长 PDF、图表、公式和补充信息，解析文本不一定包含全部视觉信息；关键结论仍应回到原始页码或 publisher HTML 核查。

`nature-reader` skill 可以把一篇论文整理成中英文对照、图表感知的 Markdown reader，并保留原文锚点。它适合真正精读，而不是把整篇论文压缩成一页摘要。你也可以要求 Agent 只处理特定章节或图表，但应明确保留哪些原文证据。

```text
精读 literature/papers/pd_redispersion.pdf，生成中英文对照 reader。
保留文章的章节顺序，把每张关键图和表放到对应讨论附近，并为每个文本块保留页码或来源锚点。
方法、结果与作者推测要分开。不要只交付摘要；对影响"Pd 是否真正原子化"判断的
表征证据做详细解释，并列出论文没有排除的替代解释。
```

### 证据表把论文与具体主张连接起来

主题综述最有价值的中间产物往往不是一段文字，而是一张可审查的证据表。Agent 可以按论文记录材料体系、方法、实验或计算条件、主要结果、限制和证据等级，再把具体 claim 映射到支持或反驳它的来源。这样 Writing Agent 后续起草时，不必重新猜每句话应该引用哪篇论文。

对题名、DOI、预印本和期刊版本的去重应在检索早期进行。确定最终论文后，`finalize_citations` 会统一解析 DOI、作者、期刊、年份等字段并导出引用文件。`nature-ref-verifier` 还可以逐字段检查已有参考文献，标出卷年冲突、作者顺序、页码和 DOI 异常。

```text
把 literature/corpus/ 中关于 Pd/CeO2 的论文整理成 claim-evidence 表。
主张至少覆盖 Pd 稳定位点、氧空位作用、氧化还原气氛下的迁移、烧结温度和再分散证据。
每个主张分别列出支持、反例或证据不足的论文，并标记证据来自摘要、全文还是 SI。

完成选择后再统一核对 DOI 和元数据，输出 evidence.csv、references.bib 和
unavailable.md。不要让同一篇论文的预印本与期刊版本重复计数。
```

### 专项文献与引用能力

当前 literature skills 还支持按 Nature Portfolio、Science family 和 Cell Press 范围为段落寻找 claim 级引用，执行多源文献检索与引用指标整理，以及组织持续更新的文献流水线。是否能使用 Scopus、ScienceDirect、PubMed 等来源取决于部署的 MCP 和账户权限。Agent 应报告实际使用了哪些来源，不能因为 skill 描述存在就声称已经访问未配置的数据库。

<details>
<summary>Literature Review 的能力来源</summary>

直接 tools：`web_search`、`ingest_literature_files`、`query_literature_corpus`、`finalize_citations`，以及部署成功后由 `agent-browser` 提供的受控浏览器 tools。

主要 skills：`nature-academic-search`、`nature-downloader`、`nature-reader`、`nature-citation`、`nature-ref-verifier` 和 `nature-literature-pipeline`。Tool 负责搜索、导入、查询和引用定稿；skill 负责检索策略、证据分级、全文获取边界、精读格式和交付质量。

</details>

## Writing Agent：把已有证据变成可交付文稿

Writing Agent 的输入可以很杂：中文笔记、结果表、图、代码输出、参考文献库、LaTeX 工程、PDF 旧稿或审稿意见。它的工作不是把这些材料"润色一下"，而是先理解写作目标与证据边界，再选择适合的 writing skills 组织论证、起草、修改、制图或编译。

### 起草论文、报告和项目书

`nature-writing` 与 `scientific-writing` 适合从已有 claims、结果和图件构建论文结构。`researchwrite` 面向项目书或 proposal，强调先明确论证与证据合同，再写章节。Agent 可以起草摘要、引言、方法、结果、讨论和结论，也可以重组已有章节。

好的 Writing prompt 不需要规定每段第一句话，但应说明读者、文稿类型、当前章节、可用证据、必须保留的数字和禁止补写的内容。正文默认使用连贯段落，不应把研究结果写成密集短语和清单。

```text
使用 Writing 为一篇催化计算论文重写 Discussion。现有草稿在 writing/discussion_old.md，
可信证据在 notes/claims.md、data/final_results.csv、figures/ 和 references.bib。

先判断当前论证哪里只是重复 Results，哪里缺少文献比较或限制。自主选择合适的写作 skill，
重组为连贯段落；所有数值和引用必须来自给定文件，不得补造机理。
保留对模型适用范围和未验证动力学的限制。输出新稿和一份简短修改说明。
```

### 润色、翻译和事实保持

`nature-polishing` 与 writing polisher 用于改善语言、段落逻辑和学术表达。它们应保留数值、单位、引用、结论强度和科学结构。如果用户提供中文草稿，可以翻译成投稿级英文；如果只要求语言修订，则不应把保守结论改成宣传性表达。

对于重要稿件，建议保留原文件，让 Agent 写出新版本或修订记录。你可以明确哪些术语、符号和句子不能改，也可以要求它逐段列出科学含义可能发生变化的地方。

```text
润色 writing/abstract_v3.md 的英文。保持所有数字、催化剂名称、时态、引用和结论强度不变，
不要新增背景或把相关性改成因果。目标期刊为 Nature Communications，但不要模仿宣传性摘要。

先检查摘要的科学逻辑，再做语言修改。输出 abstract_v4.md，并列出任何你认为需要作者
确认的术语或过满结论。正文必须是自然的完整段落，不要改成要点。
```

### 引用、参考文献和数据声明

Writing 可以调用 citation skills 为现有段落寻找支持文献，也可以核验 DOI、作者、卷期和页码。引用任务应从具体 claim 出发，而不是在段落末尾随意堆几篇相关论文。Agent 会把每个引用与相邻主张对应，并标记无法获得全文或支持强度不足的条目。

`nature-data` 用于准备 Data Availability、Code Availability、仓库选择、数据集引用和 FAIR metadata 检查。它可以根据数据类型和访问限制起草声明，但不会替用户上传数据或编造 accession number。

```text
检查 writing/introduction.md 中标记为 [CITATION NEEDED] 的句子。
逐条提取可以被外部文献验证的主张，优先寻找真正直接支持该主张的论文，
并说明证据来自全文还是摘要。不要给常识性过渡句硬加引用。

把建议以 claim、候选来源、支持程度和 DOI 的对应表保存下来，
确认后再更新 references.bib；不要直接覆盖正文。
```

### 科研图件、示意图和 PDF

`nature-figure` 与 `scientific-visualization` 可以用 Python 或 R 生成投稿级图件，包括多 panel、误差、显著性、色盲友好配色和期刊尺寸。用户应说明图要支持的结论、数据文件、单位、比较关系和输出格式。Agent 会保留绘图脚本与源数据，使图件可以重跑，而不是只交付一张不可编辑图片。

如果用户明确需要图形摘要、机制示意或概念图，可以使用图像生成路线先做草稿，再由研究者核对科学对象和标签。原子结构、能量图和定量关系不应由生成图像代替数据绘图。

`markdown-pdf-export` 可以把现有 Markdown 直接渲染成 PDF；`compile_text` 处理 LaTeX 静态检查和编译。ACS 稿件可使用本地 achemso skill，其他期刊和会议可参考 venue templates。编译成功后仍需检查图片裁切、公式、字体、交叉引用和空白页。

```text
使用 Writing 根据 data/activity.csv 和 data/stability.csv 制作论文主图。
图的核心结论是活性与稳定性存在权衡，并突出三个候选催化剂。
先检查数据列、单位、重复实验和误差定义，再提出 panel 逻辑；绘图后做尺寸、字体、
颜色和标注审计。保存 Python 源码、处理后的绘图数据、SVG、PDF 和 600 dpi TIFF。
不要为了视觉效果删除不利数据点。
```

### PPT、审稿回复和专利草稿

`nature-paper2ppt` 可以从论文、PDF 或阅读笔记制作中文学术汇报，选择支撑论证所需的图，生成 slide 内容和 speaker notes，并做溢出与图像质量复查。它适合组会、journal club、答辩和学术报告，不是把论文段落逐页粘贴到模板。

`nature-response` 可以把 editor letter 和 reviewer comments 整理成逐点回复、revision cover letter 和 marked manuscript 修改计划。它会区分作者接受、部分接受和有证据拒绝的意见，并要求每条回复指向实际修改或证据。

`nature-paper-to-patent` 可以从论文、报告、代码和图中提取有证据支持的技术贡献，生成中国发明专利的权利要求、说明书、摘要和摘要附图草稿。专利性判断和正式提交仍需专业人员审核。

```text
根据 writing/submission/manuscript.pdf 制作一套 20 分钟中文组会汇报。
请先理解论文的研究问题、主要证据链和局限，再选择真正需要的图。
不要按论文页序机械搬运，也不要为每个小节都做一张标题页。

输出可编辑 PPTX 和 speaker notes，复查所有图的清晰度、文字溢出、颜色、
页码和引用。结尾用论文能支持的结论与尚未解决的问题收束。
```

<details>
<summary>Writing 的能力来源</summary>

入口 tools 包括 `generate_nanobanana_figure` 和 `review_pdf_manuscript`。Writing worker 还使用 `polish_academic_prose`、`compile_text` 和 `render_markdown_pdf`，并可通过 workspace 的文件和脚本能力生成实际交付物。

Skills 覆盖论文与项目书写作、润色、引用、数据声明、图件、全文阅读、参考文献核验、审稿回复、投稿前审稿、PPT、专利、ACS LaTeX、期刊模板和 Markdown PDF。`citation-management` 提供通用引用管理方法，`humanizer` 负责最终文风审计。某个 skill 能否完成所有外部动作仍取决于本机安装的软件、可用 API 和用户提供的源材料。

</details>

## Peer Review Agent：让多位 reviewer 独立检查同一稿件

Peer Review 需要一份明确的 canonical PDF。PDF 是审稿对象，因为它同时包含正文、图、表、公式和最终版面。LaTeX 或 Word 源文件可以保留在 workspace 供后续修订，但不要把多个相似 PDF 一起交给 Agent 而不说明哪个版本有效。

`peer_review_request` 会向 `peer_review_models` 中配置的模型分别发起审稿。每个 reviewer 独立形成意见，然后 editor 层综合新颖性、方法可靠性、证据与结论、报告完整性和投稿风险。多个 reviewer 说法一致不自动证明它们正确；用户仍需回到页码、原始数据和方法文件核实。

```text
使用 Peer Review 审查 writing/submission/manuscript_r2.pdf，目标期刊为 Journal of Catalysis。
这是唯一 canonical PDF；SI 为 writing/submission/si_r2.pdf。

请让 reviewer 独立检查模型构建、DFT 设置、吸附能与自由能基准、NEB 证据、
实验对照、图表和可重复性。每条主要意见都指向具体页码、图表或段落。
保留完整 reviewer 报告，再给 editor synthesis；不要直接改稿，也不要替作者写回复。
```

### 从审稿进入修订

审稿完成后，先建立一张决策表：每条意见是接受、部分接受、需要澄清还是有证据拒绝；需要哪份数据或分析；会修改哪里。随后把 canonical 源稿、reviewer reports、editor synthesis 和作者决定交给 Writing。Writing 可以起草逐点回复并修改源文件，但每条"已修改"都必须对应真实 diff。

修订结束后重新编译 PDF，再对新 PDF 做版面和科学一致性检查。若进行第二轮 Peer Review，应明确这是新版本，避免 reviewer 继续评论已经修复的旧页码。

```text
使用 Writing 处理 writing/review_round1/ 中的审稿意见。源稿为 writing/manuscript.tex，
作者决定记录在 writing/review_round1/decisions.md。

先逐条核对 reviewer 原文、作者决定和现有证据，再起草 response letter 和修改计划。
只有 decisions.md 标为接受或部分接受的内容可以进入稿件；需要新增计算的意见先列为待办，
不要编造结果。每条回复指向实际修改位置，并保留修改前后对照。
```

## 三类 Agent 的推荐交付顺序

对于一项完整论文工作，常见顺序是先用 Literature Review 建立来源与证据表，再让 Writing 起草或修订，编译得到 canonical PDF 后交给 Peer Review。审稿结果回到 Writing 形成修订与回复，必要时再由 Literature Review 补证据，或由 Research 协调新增计算。

这个顺序不是强制流水线。已有完整证据时可以直接进入 Writing；只想精读一篇论文时无需启动 Research；只检查版面时也不必发起多 reviewer 审稿。选择最窄且足够的入口，Agent 才能把自主性用在任务本身，而不是在角色之间来回规划。
