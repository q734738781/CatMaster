# 3. 五类 Agent：从研究目标到可交付结果

[上一章](02-concepts.zh.md) | [目录](README.zh.md) | [下一章](04-webui.zh.md)

WebUI 提供 Research、Experiment、Writing、Peer Review 和 Literature Review 五个入口。它们不是同一个聊天模型换了五套开场白，而是拥有不同职责、worker、tools 和 skills 的研究角色。

入口选得合适，Agent 就能从你的目标出发安排工作。入口选得太宽，简单任务可能多出不必要的规划；选得太窄，Agent 可能缺少需要的 worker。最实用的判断不是"哪个最强"，而是"这次工作的主要产物是什么"。

## Research Agent：统筹开放研究目标

Research 适合尚未被拆成单一任务的问题。它会维护原始研究目标，判断目前缺的是文献证据、计算证据、文稿整合还是独立审查，然后按顺序委派 Literature Review、Experiment、Writing 或 Peer Review。每次委派结束后，它会检查新结果是否真正回答了当前问题，再决定是否需要下一阶段。

例如"阐明 Pd 单原子在 CeO2 上抗烧结的可能机制"不是一个单独计算。Research 可以先让 Literature Review 建立已报道机制和关键表征证据，再让 Experiment 检查哪些结构或能量差仍需计算，最后让 Writing 把文献与计算结果组织成报告。它不会因为文献里缺一个数值就默认启动 DFT；计算需要明确的科学理由和用户授权。

Research 特别适合以下工作：

- 从一个开放问题建立可执行的研究路线，并随证据更新路线。
- 把文献、已有项目文件和新计算放在同一个论证中比较。
- 管理多阶段项目中的假设、证据缺口、阶段产物和尚未解决的问题。
- 在完成一个明确阶段后交付结论与限制，而不是无限扩张任务。

它不适合代替 Experiment 做一次简单结构转换，也不适合代替 Literature Review 只查一组论文。Research 的价值在于跨阶段判断和整合。

<details>
<summary>Research 当前可调用的角色、tools 与 skills</summary>

Research 的领域动作主要通过四个 specialist 完成：`experiment_specialist`、`writing_specialist`、`peer_review_specialist` 和 `litreview_agent`。它自己保留文件、任务计划和项目记忆等通用能力，不直接持有 VASP、slab 或远程提交工具。

它可按需读取的研究 skills 是 `nature-citation`、`nature-data`、`nature-experiment-log`、`nature-figure`、`nature-literature-pipeline`、`nature-paper-to-patent`、`researchwrite`、`nature-reader`、`nature-ref-verifier` 和 `nature-writing`。真正执行计算时会进入 Experiment 及其 worker 的 skill 范围。

</details>

参考 prompt：

```text
使用 Research 研究 Pd 单原子在 CeO2 表面抗烧结的可能机制。

先检查 workspace 中 literature/、structures/ 和 calculations/ 已有材料，
把已经有证据支持的判断与仍然缺证据的假设分开。自主决定何时需要文献、
计算或写作 specialist，但每次只推进一个有边界的阶段并检查产物。

本轮目标是形成一份证据地图和下一阶段建议，不要因为文献缺少某个数值就自动启动 DFT。
如果确实建议新增计算，请说明它能区分哪些假设、需要哪些输入和大致成本，等我确认。
```

## Experiment Agent：组织建模、计算和结果检查

Experiment 负责边界明确的计算研究。它先理解体系、输入和预期结果，再把工作交给 Materials、Dynamics、ML 或 ORCA/xTB worker。它可以直接检索和下载 Materials Project 结构，也可以查看部署中有哪些远程 task；具体的结构构建、输入准备、科学分析和远程提交由相应 worker 完成。

Experiment 的自主性体现在选择正确的 worker、组合多个准备与检查步骤，并根据中间结果修正后续工作。例如一项吸附筛选可能先由 Materials worker 建 slab 和位点，发现候选过多后使用 MLFF 做预筛，再只为少数结构准备 VASP。用户不必手工切换 worker，但应说明允许哪些近似、是否可以提交计算，以及哪些科学选择要先确认。

它覆盖四组主要能力：

- Materials worker 处理材料发现、体相与表面、吸附、缺陷、VASP/CP2K、MLFF 推理、NEB、能带、声子、弹性和热力学。
- Dynamics worker 处理 CP2K AIMD、LAMMPS、MLFF MD、restart、轨迹健康和扩散等分析。
- ML worker 处理训练数据、MACE 训练与评估、主动学习候选选择。
- ORCA/xTB worker 处理分子生成、构象、xTB、CREST、ORCA、TS、IRC、TDDFT 和 NMR。

这四类 worker 的 tools、skills 和参考 prompt 在[第 5 章](05-agents-and-modules.zh.md)详细展开，完整建模能力在[第 6 章](06-computational-workflows.zh.md)说明。

<details>
<summary>Experiment 自己直接拥有的 tools</summary>

Experiment coordinator 可以使用 `mp_search_materials` 和 `mp_download_structure` 查找或下载 Materials Project 结构，也可以用 `get_avail_remote_task` 查看当前部署公开给 worker 的远程任务。它不会越过 worker 直接调用 `remote_submission`。

</details>

参考 prompt：

```text
使用 Experiment 检查 structures/POSCAR，并为 CO 吸附研究建立一组可复核的表面候选。

请先识别材料、晶胞和现有 Selective Dynamics，再自主选择合适的 worker、skills 和 tools。
比较 (111) 面的合理终止方式，生成表面结构、代表性吸附位和 CO 初始构型；
对每一步保留来源、参数和结构检查。不要一开始就批量准备所有 VASP 任务。

先用几何与配位审计缩小候选集，并说明还需要我决定的化学问题。
本轮可以写文件和生成结构，但不要提交远程计算。
```

## Literature Review Agent：建立可追溯的文献证据

Literature Review 不是把搜索结果改写成一篇流畅摘要。它的任务是发现文献、取得合法可用的文本、区分元数据与全文证据、去重、精读、建立证据表，并在论文确定后核对引用记录。

它可以从公共网页搜索开始，必要时打开受控浏览器。受控浏览器可以复用用户本人已登录的机构会话，但不会绕过验证码、付费墙或安全警告。已有 PDF、Markdown 和表格也可以导入本地语料库，再围绕研究问题检索。最终引用由确定性工具统一解析，避免每条文献都依赖模型猜测元数据。

Literature Review 可以完成主题综述、方法比较、关键论文精读、中英文对照阅读、claim-evidence 表、引用补充、参考文献核验和全文获取记录。它不会运行材料计算，也不应把只读到摘要的论文写成掌握了全部方法细节。

<details>
<summary>Literature Review 当前 tools 与 skills</summary>

直接 tools 包括 `web_search`、`ingest_literature_files`、`query_literature_corpus` 和 `finalize_citations`。部署了 `agent-browser` 时，还会得到经过筛选的浏览器工具，用于动态页面和用户授权会话。

主要 skills 包括 `nature-academic-search`、`nature-downloader`、`nature-reader`、`nature-citation`、`nature-ref-verifier` 和 `nature-literature-pipeline`。它们分别处理检索范围、合法全文获取、图表感知的全文阅读、claim 级引用、元数据核验和较完整的文献流水线。

</details>

参考 prompt：

```text
使用 Literature Review 调研 2021 年至今 Pd 催化剂抗烧结策略，重点关注氧化物载体上的
单原子稳定和可逆再分散。请先设计覆盖面足够的检索策略，再对题名、DOI 和版本去重。

把"只发现记录""读到摘要""读到全文或补充信息"明确区分。优先精读真正讨论稳定机制、
迁移或烧结实验的论文，建立一张包含材料体系、条件、证据类型、结论和限制的表。
保存检索式、候选文献表、未能取得全文的条目和最终引用库，不要编造无法核实的参数。
```

## Writing Agent：把已有证据变成文稿和图件

Writing 面向已经有材料的写作任务。你可以给它研究笔记、结果表、图、引用库、已有章节或期刊模板，让它起草、重构、润色、排版和编译。Writing coordinator 会把实质性起草交给 writing worker，把保守语言修改交给 polisher。这样可以避免一次润色顺手改变技术立场或结构。

Writing 的能力远不止"改英文"。当前 skills 覆盖论文各章节、项目书、数据可用性声明、文献引用、参考文献核验、科研图件、PPT、投稿回复、投稿前审稿、中文专利草稿、ACS LaTeX 模板、Markdown PDF 和通用 venue 模板。它还可以读取 PDF 或 Office 文档的有界文本，处理已有 LaTeX，生成可编辑图和编译后的 PDF。

Writing 不会替用户发明实验结果，也不应为了让段落更完整而补造引用。缺少文献证据时，可以把任务转给 Literature Review；缺少计算证据时，应明确指出而不是自行扩大为计算项目。

<details>
<summary>Writing 当前角色、tools 与 skills</summary>

入口 Agent 可以调用 `generate_nanobanana_figure` 和 `review_pdf_manuscript`，并委派 `writing_worker_agent` 与 `writing_polisher_agent`。Writing worker 还可以使用 `polish_academic_prose`、`compile_text` 和 `render_markdown_pdf`，同时保留通用文件与轻量脚本能力。

可加载 skills 包括 `nature-writing`、`nature-polishing`、`nature-citation`、`citation-management`、`nature-data`、`nature-figure`、`nature-reader`、`nature-response`、`nature-reviewer`、`nature-paper2ppt`、`nature-paper-to-patent`、`nature-ref-verifier`、`nature-academic-search`、`researchwrite`、`scientific-writing`、`scientific-visualization`、`achemso-latex-manuscript`、`venue-templates`、`markdown-pdf-export` 和质量检查 skill `humanizer`。

</details>

参考 prompt：

```text
使用 Writing 根据 notes/result_contract.md、data/summary.csv、figures/ 和
writing/references.bib 起草 Results 中关于表面稳定性的两个小节。

请先阅读证据并提出段落论证顺序，再自主选择相关 writing skills。
所有数值、误差、体系名称和引用必须能追溯到给定文件；不要补写缺失数据或新引用。
正文使用连贯段落，不要写成要点堆叠。将草稿写到 writing/results_surface_v1.md，
并附一份简短的证据对应说明，列出仍需作者判断的地方。
```

## Peer Review Agent：从固定稿件出发做独立审查

Peer Review 面向一份已经编译好的 canonical manuscript PDF。它会把同一份 PDF 交给 `peer_review_models` 中配置的 reviewer 模型，让它们分别检查新颖性、方法、证据、报告质量和可重复性，再由编辑层综合共识、分歧和风险。

这和 Writing 中的"帮我修改一段"不同。Peer Review 应保持审稿人视角，不直接把稿件改成它喜欢的版本。原始 reviewer 报告应保留，因为 editor synthesis 可能会压缩或取舍意见。审稿结束后，用户决定接受、部分接受或拒绝哪些意见，再把决定和源文件交给 Writing 处理修订与回复。

<details>
<summary>Peer Review 当前 tools 与 skills</summary>

主要执行工具是 `peer_review_request`，它会把一份本地 PDF 发送给所有已配置 reviewer 模型并收集原始报告。入口还会委派 `peer_review_worker_agent` 完成一次有边界的审稿。Worker 可以读取 writing 和 writing-quality skills，用于投稿前审查标准、报告组织和避免模板化措辞，但不会获得计算 worker 的执行工具。

</details>

参考 prompt：

```text
使用 Peer Review 审查 writing/submission/manuscript.pdf。这是本轮唯一的 canonical manuscript；
Supplementary Information 位于 writing/submission/si.pdf。

按催化与材料计算论文的标准分别检查新颖性、计算方法、结构模型、统计与对照、
证据是否支持结论、图表可读性和可重复性。请保留每位 reviewer 的完整报告，
再给出 editor synthesis，明确共识、分歧、必须解决的问题和可选改进。
本轮只审稿，不修改源文件，也不要把意见写成作者回复。
```

## 五类 Agent 怎样交接

不同入口共享同一个 workspace，但职责不会自动混在一起。Research 可以在一个开放目标中委派其他 specialist。直接使用 Experiment、Writing、Peer Review 或 Literature Review 时，它们只处理自己的主要任务。

如果一项工作自然进入下一阶段，先让当前 Agent 把产物保存完整，再在合适的入口继续。例如 Literature Review 交付证据表和引用库后，可以新建 Writing thread 起草综述；Peer Review 交付审稿意见后，可以新建 Writing thread 处理修稿；Experiment 完成计算并写出结果合同后，也可以交给 Writing。这样比在一个 thread 中频繁改变角色更容易追溯。

下一章介绍 WebUI 中怎样选择入口、观察委派、查看文件和在运行中补充方向。模型 provider、角色路由和部署配置已移到[第 10 章](10-deployment-operations.zh.md)，新用户不需要先理解那些字段才能认识 CatMaster 的功能。
