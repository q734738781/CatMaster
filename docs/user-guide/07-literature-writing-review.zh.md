# 7. 文献、写作与审稿 Agent

[上一章](06-computational-workflows.zh.md) | [目录](README.zh.md) | [下一章](08-remote-execution.zh.md)

Literature Review、Writing 和 Peer Review 使用同一个 workspace，却承担三种不同的证据责任。Literature Review 负责检索、核实并综合解释文献发现；Writing 负责用已有证据形成文稿和其他交付物；Peer Review 负责站在独立审稿人的位置检查一份固定稿件。把这三类角色分清，能避免一边写作一边补造证据，也能让审稿意见与实际修订保持可追溯。

## Literature Review Agent：从发现论文到证据库

Literature Review 可以处理快速查证，也可以承担较完整的主题综述。任务规模由研究问题决定。一个精确事实可能只需少量高质量来源；主题综述围绕重要进展、定量比较和机制展开。用户可以说明时间范围、材料或反应体系及最终用途，Agent 据此选择检索和精读深度。

Agent 会保留真实问题、交付物、相关性边界、未解决项和用户明确的停止要求，不会把每个请求都改写成完整综述 checklist。每获得一批有用证据后，它会重新判断继续搜索是否还可能改变当前答案、答案边界或下一步决策；有足够证据支撑有界答案时立即综合，用户要求缩小或停止时不再开启新分支。这个判断背后没有固定论文数、分支数或形式化 completion state。

### 发现不是精读

公共网页搜索适合发现论文、项目页和数据库记录。只有题名、作者和 DOI 时，通常只能证明论文存在；但检索结果中如果包含完整摘要或有信息量的作者摘要，就可以支持其明确陈述的结论。Agent 会说明证据边界，不会因为没有全文就放弃这些信息，也不会把摘要扩写成其中没有的方法或数值。

Literature Review 及其 worker 可直接检索 OpenAlex 和 Semantic Scholar、查询已知文献记录，或根据种子论文获取 Semantic Scholar 推荐。OpenAlex 检索返回 provider cursor，Semantic Scholar relevance search 返回 offset，单页均最多 100 条。小默认值只是页大小，不要求自动取完整个结果集。记录保留 provider 返回的全部作者和完整摘要。需要进一步检查原文时，专用来源获取工具会返回可直接读取的本地文件。

当关键判断确实依赖摘要中没有的方法、条件、数值、图表或补充信息时，受控浏览器可以作为一次升级路径，读取开放获取内容或用户本人授权的机构会话。一次合理访问失败后，Agent 会说明限制并继续综合其他来源，而不是反复尝试不同页面或下载。遇到 CAPTCHA、二维码、OTP、许可确认或安全警告时会停止，不会绕过访问控制。

同一 run 内，选中文献获取会规范化 DOI、arXiv、PMID 和 URL identity，并让 parent 与 worker 共享已经完成或仍在进行的结果。同一 identifier 通过同一路径重复请求时，会直接返回已有 source handle 和 access state，不会再次下载。只有换用实质不同且已授权的路径，或前次结果属于 transient/expired，才允许重试；不同版本不会被错误合并。

```text
调研 2018 年至今单原子催化剂动态聚集和再分散的原位研究。
讲清主要发现、影响因素和机制，结合代表性实验数据解释，并附参考文献。
```

### 本地语料让项目材料可以反复查询

你可以把已有 PDF、Markdown、DOCX 或表格放进 `literature/`，让 Agent 导入本地 corpus。导入工具会建立可检索文本和来源记录，之后可以围绕多个问题反复查询。对于长 PDF、图表、公式和补充信息，解析文本不一定包含全部视觉信息；关键结论仍应回到原始页码或 publisher HTML 核查。

需要精读时，可以要求 Agent 编写保留原文锚点的双语阅读文档，并明确章节、图表及阅读深度。文件阅读与 corpus 检索能力不依赖固定 reader 模板。

```text
精读 literature/papers/pd_redispersion.pdf，生成中英文对照 reader。
保留文章的章节顺序，把每张关键图和表放到对应讨论附近，并为每个文本块保留页码或来源锚点。
详细解释 Pd 再分散的关键表征、实验结果及其机理意义。
```

### 保留研究发现、解释与来源

文献交接保留有用发现、综合解释及其来源。需要跨体系或条件比较时，可以用证据表组织观察、方法、条件与相关主张；正文或交接仍应解释这些差异的意义。访问深度、来源独立性等属性在影响判断时记录，不要求每项来源填满同一张核验表，也不给整篇论文评高、中、低等级。

Literature Review 通过紧凑自然语言交接研究问题、发现及其意义，并保留恢复相关证据所需的来源句柄。下游复用已完成的核验，也可以为理解、解释或使用原图而按需读取来源；不需要重复核验或另建平行 evidence manifest。

文献 coordinator 按科学问题组织 worker 分支，返回研究发现、数据和综合解释。明确的查证请求或已经发现的具体矛盾才进入针对性核验。证据属性附录服务于这类问题，普通综述围绕主题展开。文献笔记不规定最终措辞、章节顺序或图注规则；文献侧的返回指导与计算执行的返回要求分别维护。

文献候选集使用 `selected`、`deferred` 或 `excluded`，并写明具体理由。selected 表示论文会影响当前综合或决策；deferred 表示有关但当前不需要；excluded 表示范围不符、重复或无法回答问题。访问深度与 claim relationship 分开记录，不给论文计算综合分，也不把期刊声望当作证据等级。

对题名、DOI、预印本和期刊版本的去重应在检索早期进行。需要交付参考文献库时，`finalize_citations` 会统一解析 DOI、作者、期刊、年份等字段，默认只导出一份 `.bib`，不附带同内容的 JSON 或 Markdown。明确需要其他格式时才设置 `output_format` 为 `json` 或 `md`；未解析的标识符仍在工具响应中说明。按需导出脚本同样默认 BibTeX。`citation-management` 支持核对已有参考文献，标出卷年冲突、作者顺序、页码和 DOI 异常。

```text
结合 literature/corpus/ 中的论文，解释 Pd/CeO2 的稳定位点、氧空位作用、
氧化还原气氛下的迁移及再分散机制。用表格比较关键实验条件和结果，并附参考文献。
```

### 专项文献与引用能力

检索范围由用户的问题及明确的期刊、日期等条件决定。可用检索和获取来源取决于部署配置；Agent 应报告实际访问的来源，不能把 skill 描述当成数据库访问凭据。

<details>
<summary>Literature Review 的能力来源</summary>

直接 tools：`web_search`、`acquire_literature_source`、`batch_acquire_literature_sources`、`ingest_literature_files`、`query_literature_corpus`、`finalize_citations`。`web_search` 会按 provider 路由：OpenAI/Codex 角色使用托管搜索，其他 provider 使用带配置降级的 CatMaster 实现。选中文献获取优先走直接 OA 路径，匹配的 DOI 可使用已配置的 Elsevier 官方 API，必要时内部通过 ScanSci 浏览器访问一次 DOI 落地页；主文 PDF 必须经过校验，可按需保存 SI，没有 PDF 时只缓存一次静态页面。批量接口接收直接标识符列表或工作区内的 `.txt`、`.csv`、`.tsv` 清单，通常按同一科学目的组织 10–30 篇，输入超过 50 行时整批拒绝，不会静默截断。

原生元数据工具包括 `search_openalex`、`search_semantic_scholar`、`get_openalex_record`、`get_semantic_scholar_record` 和 `recommend_semantic_scholar`，与网页搜索互补，Literature Review 及其 worker 均可直接调用。

本地 `literature-evidence-use` skill 提供科学范围与证据使用指导。原生工具负责检索、来源获取、导入、查询和引用定稿；元数据转换脚本按需读取。

</details>

## Writing Agent：把已有证据变成可交付文稿

Writing Agent 的输入可以很杂：中文笔记、结果表、图、代码输出、参考文献库、LaTeX 工程、PDF 旧稿或审稿意见。它的工作不是把这些材料"润色一下"，而是先理解写作目标与证据边界，再选择适合的 writing skills 组织论证、起草、修改、制图或编译。

论文、报告和演示都采用自由文本 brief，说明用户问题、读者、关键发现及其意义、相关证据与图件路径、用户明确要求和交付物。未指定的写作选择交由作者按语境和学科惯例判断；交接不追加写作禁令、假想误读清单或统一图注字段。科学条件随证据保留，由作者判断哪些影响读者理解结果或作出选择。制作要求在执行中落实，正文围绕发现及其意义展开。章节任务保留整体目的和必要的相邻上下文，协调者负责覆盖、论证衔接和读者相关性，worker 自主取舍证据笔记中的表达并完成最终整合。参考文献、格式转换、编译和局部修订留在相关写作任务内，复用已经完成的检查。

Research 与 Writing 在交接时区分面向读者解释研究的报告、熟悉项目者使用的内部技术记录，以及面向期刊的论文，并说明读者已知什么。科研报告和 PPT 未明确受众时，默认面向不了解本项目计算理论与内部过程的导师或实验合作者，补齐理解问题、比较量和结果意义所需的解释。明确要求的内部技术摘要保持简洁，可用表格、列表和适合该读者的术语；论文遵循期刊及论文写作指引。简短的交接或完成消息不限制正文深度，精选示例也不代替用户要求覆盖的全部材料。

写作采用有经验的同行向认真阅读者解释问题的方式：判断哪些发现值得展开，说明证据之间的关系，形成有依据的解释和判断。详略随解释价值分配，常识和自然可推知的内容可以留白。避免竞争解释和限定。

Writing 检查实际成品及整篇讲述顺序是否满足该用途，以及读者能获得什么理解、作者判断是否有证据支持。为解释结果或使用已有图件而阅读来源，不等于重新验证已经核查的科学结论。修改可以针对局部表达，也可以在用户授权重构时重排整章；不为另一种同样可行的审美选择无限返修。

### 用 Research Graph 快速定位原始证据

如果当前 Writing thread 已显式 Attach 一个 Research Graph，Agent 会先读取 partial focus，并用只读查询定位相关 Hypothesis、所有直接支持、反对或无法区分的 Results、产生它们的 Experiment 及 Sources。随后只打开本节真正需要的 note、artifact、run、thread message 或文献来源；涉及关键数值、条件、机理或限制时，仍以原始 owner 内容为准。Graph 覆盖不足时才局部搜索 workspace，并说明缺口。这个过程不增加稿件专用 schema，也不允许 Writing 修改科学图。

### 起草论文、报告和项目书

`publication-launch-writing` 围绕证据支持的核心贡献组织论文；研究设计报告规范按需读取，期刊格式由 venue templates 提供。Writing 可基于给定证据起草或重组论文、报告与项目书，不预设通用章节配方。

论文以作者身份叙述科学工作，省略与研究无关的写稿过程。agents、prompts、tools、工作流、文件和运行记录如果本身是研究对象、相关方法或数据，或属于必须披露的内容，仍应正常介绍。

共享 `scientific-communication` skill 将具体材料或分子体系、实验观察、性能比较与科学解释联系起来。报告中的结构图、实验图像和曲线应与正文解释相互对应；综述围绕科学问题和证据组织，进展汇报保留有意义的负结果和研究方向调整。方法与数值用于解释发现，执行记录和 QC 诊断采用各自适合的文体。Research 可以将基于已完成计算的正式报告交给 Writing，不为写报告重开实验。图文交付所需的关键图件应从已有证据准备并整合，不能用一份待制图说明代替。

Research 在科学收尾前判断结果的合理性及证据是否支持结论，把影响理解的条件和未决问题融入回答。普通报告、PPT 和文件交付不要求额外附加自检段；明确要求核验或诊断时，按该任务解释检查结果。

好的 Writing prompt 不需要规定每段第一句话，但应说明读者、文稿类型、当前章节、可用证据、必须保留的数字和禁止补写的内容。正文默认使用连贯段落，不应把研究结果写成密集短语和清单。

```text
使用 Writing 为一篇催化计算论文重写 Discussion。现有草稿在 writing/discussion_old.md，
可信证据在 notes/claims.md、data/final_results.csv、figures/ 和 references.bib。

先判断当前论证哪里只是重复 Results，哪里缺少文献比较或限制。自主选择合适的写作 skill，
重组为连贯段落；所有数值和引用必须来自给定文件，不得补造机理。
保留对模型适用范围和未验证动力学的限制。输出新稿和一份简短修改说明。
```

### 润色、翻译和事实保持

Writing worker 使用 `scientific-communication` 及其按需措辞附录改善语言和段落连贯性，同时保留数值、单位、引用、结论强度与科学含义。中文草稿可以翻译成投稿英文；仅要求语言修订时，不得擅自改变主张。

中文正文、标题和表格使用常见术语，写清对象、事实和判断理由，避免自创压缩词串或用内部状态标签代替解释。必要的编号和缩写在首次出现时说明含义及所指对象；与读者理解无关的追踪编号保留在来源记录中。是否分类及分类维度由读者的问题决定，不预设分类模板；不同判断分别表达，名称前后一致。协调者审阅实际文稿时也检查这些要求。

对于重要稿件，建议保留原文件，让 Agent 写出新版本或修订记录。你可以明确哪些术语、符号和句子不能改，也可以要求它逐段列出科学含义可能发生变化的地方。

```text
润色 writing/abstract_v3.md 的英文。保持所有数字、催化剂名称、时态、引用和结论强度不变，
不要新增背景或把相关性改成因果。目标期刊为 Nature Communications，但不要模仿宣传性摘要。

先检查摘要的科学逻辑，再做语言修改。输出 abstract_v4.md，并列出任何你认为需要作者
确认的术语或过满结论。正文必须是自然的完整段落，不要改成要点。
```

### 引用、参考文献和数据声明

Writing 可以调用 citation skills 为现有段落寻找支持文献，也可以核验 DOI、作者、卷期和页码。引用任务应从具体 claim 出发，而不是在段落末尾随意堆几篇相关论文。Agent 会把每个引用与相邻主张对应，并标记无法获得全文或支持强度不足的条目。

Writing 根据实际数据情况与期刊要求起草 Data Availability、Code Availability 等声明，不编造 accession number，也不擅自上传材料。

```text
检查 writing/introduction.md 中标记为 [CITATION NEEDED] 的句子。
逐条提取可以被外部文献验证的主张，优先寻找真正直接支持该主张的论文，
并说明证据来自全文还是摘要。不要给常识性过渡句硬加引用。

把建议以 claim、候选来源、claim relationship、访问深度和 DOI 的对应表保存下来，
确认后再更新 references.bib；不要直接覆盖正文。
```

### 科研图件、示意图和 PDF

Plot worker 使用 `publication-data-plotting` 制作定量图，遵守 Origin 风格、经验证的 NPG 分类配色与实际成图检查要求。色彩语义和用户明确要求决定适用的例外。用户应说明图要支持的结论、数据文件、单位、比较关系和输出格式。每个逻辑图件只保留一种最终格式；未指定时使用高分辨率 PNG，期刊或下游接口明确要求其他格式时采用该格式。绘图脚本、源数据和最终图件承担不同作用，可以同时保留；仅用于检查的转换图放在 `/tmp/`，不作为第二份交付物。

图形摘要、机制示意和概念图可用 `generate_figure` 制作，并检查科学对象和标签。工具支持按调用选择模型、附加参考图和继续编辑已有图片，详见[图片生成接口](../figure_generation.md)。原子结构、能量图和定量关系应使用结构渲染或数据绘图。

`markdown-pdf-export` 可以把现有 Markdown 直接渲染成 PDF；`compile_text` 处理 LaTeX 静态检查和编译。ACS 稿件可使用本地 achemso skill，其他期刊和会议可参考 venue templates。编译成功后仍需检查图片裁切、公式、字体、交叉引用和空白页。

```text
使用 Writing 根据 data/activity.csv 和 data/stability.csv 制作论文主图。
图的核心结论是活性与稳定性存在权衡，并突出三个候选催化剂。
先检查数据列、单位、重复实验和误差定义，再提出 panel 逻辑；绘图后做尺寸、字体、
颜色和标注审计。保存 Python 源码、处理后的绘图数据，以及一张 600 dpi PNG 主图；
不要另外保存同内容的 SVG、PDF 或 TIFF。
不要为了视觉效果删除不利数据点。
```

### PPT 与审稿回复

Writing 将 PPT 工作交给 `presentation_worker`，通过 EasySlides 技能和预装运行包，从论文、笔记及给定图件制作可编辑汇报。worker 保留完整文件操作、命令执行能力，并绑定 `generate_figure` 制作插图。标题、正文、普通表格和页面布局保留为 PPT 原生可编辑对象，数据图与复杂插图作为局部素材嵌入。版面根据听众和证据决定，可以复用内置或用户给定模板，并检查实际渲染结果。安装与配置见 [EasySlides](../easyslides.md)。

Writing 可以整理逐点审稿回复和修改计划；每条声称已完成的修改都应对应实际编辑或支持证据。

```text
根据 writing/submission/manuscript.pdf 制作一套 20 分钟中文组会汇报。
请先理解论文的研究问题、主要证据链和局限，再选择真正需要的图。
不要按论文页序机械搬运，也不要为每个小节都做一张标题页。

输出可编辑 PPTX 和 speaker notes，复查所有图的清晰度、文字溢出、颜色、
页码和引用。结尾用论文能支持的结论与尚未解决的问题收束。
```

<details>
<summary>Writing 的能力来源</summary>

入口 tools 包括只读的 `query_research_graph_sql`、`generate_figure` 和 `review_pdf_manuscript`。Graph 查询由 Writing coordinator 用于证据导航；单一 Writing worker 通过 workspace 文件与脚本能力以及 `generate_figure`、`compile_text`、`render_markdown_pdf` 处理连贯的文稿、章节或整合范围。Reconciliation 会把已知实质修改合并后交回同一 worker。

本地 skills 覆盖论文论证、引用管理、科研绘图、ACS LaTeX、期刊格式和 Markdown PDF，语言指导与按需措辞示例由 `scientific-communication` 提供。其他文档任务使用相同的写作能力，具体执行方式取决于可用软件和用户提供的源材料。

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
