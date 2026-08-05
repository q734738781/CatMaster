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

当研究需要跨 thread 延续、保留竞争解释，或让一条结果同时作用于多个假设时，可以建立 workspace 级 Research Graph。它不是聊天记录，也不属于创建它的 thread。图只保存三类简短科学节点：

- Hypothesis 保存可证伪的 claim、rationale 和 observable predictions。
- Experiment 保存 objective、plan summary、decision rule 和 execution lane。
- Result 保存简短观察或结论，再用关系分别表示它支持、反对或不能区分哪些假设。来自图中 Experiment 的 Result 保留 produces 关系；文献发现、合作组结果或历史观察可以直接进入图，不需要倒填一个虚假的 Experiment。

Hypothesis 不会因为一条结果变成不可修改的 "supported" 或 "rejected" 终态。界面根据所有 `supports`、`opposes` 和 `inconclusive` 入边显示 Result 关系概览，而不是给证据打等级。一个 Experiment 可以有多个 Result，所以复现实验会增加一条观察，不会覆盖已有结果。

图与 note 的职责不同。论文精读、长分析、结构、图像、日志、报告和远程 receipt 继续放在 Files、artifact、run 或 note 中。Graph 节点保存科学命题，并用受控引用连接这些来源。planning turn 先收到明确标注为 partial 的 focus snippet，其中包含 focus 邻域和完整 runnable frontier。绑定的只读 `query_research_graph_sql` 让当前 graph 的全部节点、关系、引用和引用实际指向的 owner 记录继续可达，不需要把全图复制进 prompt。多个 active graph 同时存在时必须显式选择。

每个 graph 都有完成条件；创建时可以自己填写，也可以留空使用可见默认值，之后仍可编辑或重新打开。用户可以自己创建 seed hypotheses、实验 proposal，以及来自本项目、合作组或文献的观察/结果；创建科学输入时可以同时附 DOI、URL、note、artifact、run、thread 或 message 来源。也可以从 Result 节点启动一个绑定的 Research planning thread。`hypothesis_proposer` 会先把该 Result 与已有预测和旧 Result 比较，再判断现有 Hypothesis 是否已经足够，或是否需要一个真正不同、可证伪的新 Hypothesis。它可以查询完整绑定图、检索网络与本地 corpus，并读取或获取已选来源。共享的 `evidence_judge` 独立判断证据真正涉及哪些 Result-to-Hypothesis 关系；空 judgments 是有效结果。自动 Experiment 和 Literature Review 路径会在原子写回 Result 前完成这一步。

Result 用普通科学语言保留观察、派生分析、解释、科学模态、适用条件和 provenance，不接受统一强度等级。`supports`、`opposes`、`inconclusive` edge 描述它与某一 Hypothesis 的关系，不表示 Result 本身强弱。planning 的 staging 只保存临时分支，不会实体化。独立 evaluator 为当前 graph revision 的每个候选 Experiment 给出创新分和保守分；永久 Experiment 不保存这些分数，任何 graph mutation 都会让旧分数失效。manual 模式同时显示两种推荐。auto 模式采用明确的保守推荐；推荐或评价缺失、无效、陈旧或明确为空时保持等待。每轮仍最多运行一个真实 Experiment，完成条件满足后停止自动推进。

Research Graph 不改变执行边界。文献任务仍由 Literature Review 完成，DFT 或实验任务仍经过 Experiment、相应 worker、受管执行和必要的人工审批。一次性问答和简单线性任务可以不创建 graph。

<details>
<summary>Research 当前可调用的角色、tools 与 skills</summary>

Research 把科学计划形成交给 `hypothesis_proposer`，把候选评价交给 `experiment_evaluator`，把证据解释交给 `evidence_judge`，把实际执行交给 `experiment_specialist`、`writing_specialist`、`peer_review_specialist` 或 `litreview_agent`。它自己保留文件、任务计划和项目记忆等通用能力，不直接持有 VASP、slab 或远程提交工具。proposer、evaluator 和 judge 只加载窄的 `research_reasoning` skills 及只读图和来源能力，不持有 graph mutation、文件写入、shell、patch 或科学执行工具。

Research 可以列出、创建、查询和编辑 graph，加入 Hypothesis、Experiment、Result、证据判断与来源，也可以记录真实 blocker。`query_research_graph_sql` 只接受只读 SQL；host 根据受信任 thread 绑定 workspace、graph、revision 和引用实际可达的 owner rows。普通 mutation 需要 graph ID 和当前 revision，并返回准确的 changed entity 与最新 revision。内部 planning actions 从 planning thread 取得绑定；`stage_research_plan` 只写 disposable preview，评价、实体化和 launch 是后续独立转换。持久数据位于 workspace 的 `metadata/workspace.sqlite`，详细执行记录和资源用量仍在原有 thread、receipt 与 artifact store 中。

它可按需读取的研究 skills 包括 `research-graph-control`、`nature-citation`、`nature-data`、`nature-experiment-log`、`nature-figure`、`nature-literature-pipeline`、`nature-paper-to-patent`、`researchwrite`、`nature-reader`、`nature-ref-verifier` 和 `nature-writing`。真正执行计算时会进入 Experiment 及其 worker 的 skill 范围。

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

Experiment 的自主性体现在选择正确的 worker、组合多个准备与检查步骤，并根据中间结果修正后续工作。例如一项吸附筛选可能先由 Materials worker 建 slab 和位点，发现候选过多后使用 MLFF 做预筛，再只为少数结构准备 VASP。用户不必手工切换 worker，但应说明允许哪些近似、是否可以提交计算，以及哪些科学选择要先确认。Experiment brief 会保留这些科学边界，但把 tool 顺序、兼容执行路径、输入层修正和有边界的恢复交给 worker。Specialist 选错 worker 或执行路线时，应先在科学等价范围内改写 brief 并重新委派，而不是询问用户；只有触及用户控制的科学选择、授权、成本、时间或安全边界时才等待人类输入。

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

Literature Review 会按实际获得的证据工作。检索摘要和论文摘要可以支持其明确陈述的结论；只有题名和书目信息时，只能确认论文存在。Agent 会区分这些边界、去重、综合证据，并在论文确定后核对引用记录，而不会把“拿到全文”当成每篇论文的验收条件。

它从搜索摘要和可信学术元数据开始。选中论文需要深入阅读时，一个高层获取工具会先尝试合法开放获取仓储与索引；DOI 仍未获取 PDF 时，可以在内部通过 ScanSci/CloakBrowser 访问一次 DOI 落地页。得到的 PDF 必须通过校验，否则只保存一次静态公开页面。Agent 只读取本地文件，不自己控制浏览器状态和页面动作；只有需要跨多篇文档反复检索时才导入本地 corpus。

Literature Review 可以完成主题综述、方法比较、关键论文精读、中英文对照阅读、claim-evidence 表、引用补充和参考文献核验。它不会运行材料计算，也不应把只读到摘要的论文写成掌握了全部方法细节；如果摘要证据会实质影响结论，会用自然语言说明把握和限制，而不是要求逐篇填写置信度字段。

<details>
<summary>Literature Review 当前 tools 与 skills</summary>

直接 tools 包括 `web_search`、`acquire_literature_source`、`ingest_literature_files`、`query_literature_corpus` 和 `finalize_citations`。搜索实现跟随该角色实际绑定的模型：`codex_oauth` 和 OpenAI Responses 模型使用托管的原生 `web_search`，其他 provider 使用 CatMaster 搜索函数。该函数会在 Tavily 可用时使用 Tavily，分类失败后可降级为学术索引发现，并在结果中标明真实后端；同一个 agent 只绑定一种搜索实现。来源获取在内部使用固定版本的 ScanSci 与 CloakBrowser，不向模型暴露低层浏览器操作。

主要 skills 包括 `nature-academic-search`、`nature-reader`、`nature-citation`、`nature-ref-verifier` 和 `nature-literature-pipeline`。来源获取流程已经融合进 `nature-academic-search`；具体下载和校验行为由工具负责。

</details>

参考 prompt：

```text
使用 Literature Review 调研 2021 年至今 Pd 催化剂抗烧结策略，重点关注氧化物载体上的
单原子稳定和可逆再分散。请先设计覆盖面足够的检索策略，再对题名、DOI 和版本去重。

把"只发现记录""读到摘要""读到全文或补充信息"明确区分。先用摘要形成有边界的综合；
只有结论依赖精确条件、数值或图表时才继续读取相关原文。建立一张包含材料体系、条件、
证据来源、结论和限制的表，保存检索式、候选文献表和最终引用库，不要编造无法核实的参数。
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
