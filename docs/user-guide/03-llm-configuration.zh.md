# 3. 五类 Agent：从研究目标到可交付结果

[上一章](02-concepts.zh.md) | [目录](README.zh.md) | [下一章](04-webui.zh.md)

WebUI 提供 Research、Experiment、Writing、Peer Review 和 Literature Review 五类研究角色，并为 Research 额外提供 Persistent Research 持续入口。它们不是同一个聊天模型换开场白，而是拥有不同职责、worker、tools 和 skills 的研究角色；Persistent Research 与 Research 复用同一个角色和科学状态。

入口选得合适，Agent 就能从你的目标出发安排工作。入口选得太宽，简单任务可能多出不必要的规划；选得太窄，Agent 可能缺少需要的 worker。最实用的判断不是"哪个最强"，而是"这次工作的主要产物是什么"。

## Research Agent：统筹研究目标

Research 根据当前任务委派文献、计算、写作或正式审查，完成用户要求的阶段后交付。Persistent Research 与它共用同一个 ResearchSpecialist、模型角色、工具和 specialist 层级；持续模式在同一原生线程中推进已授权目标。

首次选择 Persistent Research 时，系统在需要时创建并绑定 auto Graph。Graph 属于 workspace，跨线程保存 Hypothesis、Experiment、Result 与来源；当前根线程负责执行协调。子任务完成后由 DBOS 原生续接根线程。图谱后台负责接收证据变化和恢复已接受工作，不另开 planner 或候选比赛来选择科研动作。

对有独立科学问题的研究，主 Research 可以并行启动多个 ResearchSpecialist 分支。每个分支保留独立上下文，自行提出或修正假说、确定方法、委派工作、解释 Result，并提出下一项有用检验。分支内的 Experiment、Literature Review 等仍通过原生同步委派返回；主 Research 负责跨路线综合与整体完成判断。研究分支也可发起新的独立研究，所有后代共用同一个部署池；仅等待这些独立结果时先结束中间 turn，结果到达后接续原研究者。

并发按整条 agent 任务的预期最高验证成本粗分：文献与小规模数据基线为 low，MLFF 探索或较大训练为 medium，DFT 研究为 high，较便宜的前置步骤归入同一任务。默认总池上限 16，low/medium/high 分别为 8/4/2，可在部署 LLM 配置的 `persistent_research` 中调整。计算等待期间持续占槽，不限制 Slurm 作业或 batch 数量。开放问题可先从低成本研究开始；形成需要更高成本的方法后，研究者申请变档，系统安全暂停、排队并自动接续同一上下文。未填写任务档位时使用 medium；明确的文献任务可选择 low。档位不是计算授权，也不要求预估算时。

Result 的 `methods`、`summary`、`conclusion` 分别保存实际方法、观察和有适用范围的解释。Graph 详情可查看和编辑三个部分。旧记录缺失的方法或结论显示 `missing due to old record`，不会据此判为负面证据；后续研究者仍可沿来源补查。

`hypothesis_proposer` 是独立科学推理角色：从实际 Result、旧证据和反证出发，解释适用范围、修订假说并推荐下一项有用检查。它可以增量写图，保留完整 SQL、文件读取和来源检索能力；实际实验仍由对应 specialist 执行。普通结果无需再交给一个独立总结器。

持续研究准备因科学停滞而结束、但用户阶段尚未完成时，根 agent 声明停止理由，原生运行路径会进行一次独立复查。若发现具体、可执行且已授权的补救，默认执行一次有界验证；只有真实授权/资源/输入障碍、检查已等价完成或失去意义、用户目标已达成等理由可以调整这一默认。没有新依据则保存部分成果、开放前提和恢复条件。用户暂停、实际预算和授权边界优先；同一问题与同一关键依据不会因为更换标题或 agent 而重复复查。

文献调研或实验推荐不会自动授权 DFT、MD 或实验室操作。可操作的实验室方案保存为 `external` Experiment，供以后回填 Result；交付达到用户要求即可完成。图谱编辑、判断修订或晚到结果不会自动重开已完成的阶段。

Result 判断边用 `scope` 与 `rationale` 保存适用条件和理由。新的 H 或 R 可以通过 `revises` 说明对旧主张的替换、限定或撤回，两条记录都保留。不同条件下的结果可以并存，某个代理方法失败也不自动否定其他观测。重要停止决定和补救结果保存在 `research_decisions`，与 H/E/R 关联。

Graph 保存科学索引；原始论文、数据、报告和长分析继续保存在 Files、artifact、run 或 note 中。当前 focus 只是局部导航，`query_research_graph_sql` 支持完整绑定图的普通 SQL、JSON1、递归关系查询和分页。读取旧结论时应同时查看修订、反证和决定性来源。

Research 可用 `add_research_hypothesis`、`add_research_experiment`、`record_research_result` 增量保存科学记录，使用 `set_research_result_judgment` 更新局部解释、`revise_research_claim` 建立修订关系，使用 `record_research_disposition` 保存未完成阶段的处置。独立 reasoner 用 `record_research_review` 保存拟停止复查。实际写入采用现有图谱事务与 revision 冲突处理；执行、排队和恢复仍由DBOS 管理。

示例：

```text
研究 Cu 电极 CO₂ 还原的近期进展，给出有来源依据的潜在实验新思路。
先核对关键文献和反应器条件，区分事实、推断和待验证假说。
本轮只做文献综合及实验推荐，不启动科学计算或实验室操作。
保存报告、来源和 Research Graph；交付实验建议后结束。
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

它从搜索摘要和可信学术元数据开始。选中论文需要深入阅读时，一个高层获取工具会先尝试合法开放获取仓储与索引；匹配的 Elsevier DOI 可使用已配置的官方 API，之后仍未获取 PDF 时，可以在内部通过 ScanSci 浏览器访问一次 DOI 落地页。主文 PDF 必须通过校验；显式请求时还可保存 DOI 对应的补充材料，否则只保存一次静态公开页面。Agent 只读取这些本地文件，不自己控制浏览器状态和页面动作；只有需要跨多篇文档反复检索时才导入本地 corpus。

Literature Review 可以完成主题综述、方法比较、关键论文精读、中英文对照阅读、claim-evidence 表、引用补充和参考文献核验。它不会运行材料计算，也不应把只读到摘要的论文写成掌握了全部方法细节；如果摘要证据会实质影响结论，会用自然语言说明把握和限制，而不是要求逐篇填写置信度字段。

<details>
<summary>Literature Review 当前 tools 与 skills</summary>

直接 tools 包括 `web_search`、`acquire_literature_source`、`batch_acquire_literature_sources`、`ingest_literature_files`、`query_literature_corpus` 和 `finalize_citations`。网页搜索实现跟随该角色实际绑定的模型：`codex_oauth` 和 OpenAI Responses 模型使用托管的原生 `web_search`，其他 provider 使用 CatMaster 搜索函数。该函数会在 Tavily 可用时使用 Tavily，分类失败后可降级为学术索引发现，并在结果中标明真实后端；同一个 agent 只绑定一种 `web_search` 实现。来源获取在内部使用固定版本的 ScanSci、可选的 Elsevier 官方 API 与浏览器后端，不向模型暴露低层浏览器操作。批量获取接收直接列表或工作区清单文件，硬上限为 50 条。

Literature Review 及其 worker 还可直接调用 `search_openalex`、`search_semantic_scholar`、`get_openalex_record`、`get_semantic_scholar_record` 和 `recommend_semantic_scholar` 进行学术元数据查询。引用导出默认只有一份 BibTeX `.bib`，明确需要时才选择其他格式。

本地 `literature-evidence-use` skill 负责限定范围的检索、来源阅读、证据解释和引用定稿；获取与访问处理由工具负责，按需脚本保留完整引用元数据。

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

Writing 面向已经有材料的写作任务。你可以给它研究笔记、结果表、图、引用库、已有章节或期刊模板，让它起草、重构、润色、排版和编译。Writing coordinator 会把起草、改写、明确的语言润色和最终整合统一交给一个 writing worker，把定量和数据原生图件交给 plot worker。因此，YAML 中配置的 `section_writer` 模型负责完整的作者可见正文处理。

当前默认模板把 `write_director`、`section_writer`、`presentation_worker`、`plot_worker`、`write_reviewer` 和 `tex_compile_fixer` 绑定到独立的 `codex-oauth-writing`：GPT-6 Astra、medium reasoning，使用已有 Codex OAuth 登录。Codex 模板中的研究主角色使用 high，计算 worker/helper 使用 medium；默认 `literature_worker` 使用 GPT-6 Luna xhigh，标题生成使用 Luna low。这些参数不改变角色的工具或技能。模型标签只是配置引用名，实际推理级别以对应 `reasoning.effort` 为准。Astra 的 Codex 配置应省略 temperature。新任务读取当前配置；已构建的运行中 agent 保留原模型。

`timeout_s` 对所有 provider 都以秒为单位。OpenRouter 适配器在构建模型时将其转换为 SDK 要求的毫秒，配置文件仍填写秒。

Writing 总领和文字 worker 使用共享 `scientific-communication` 指导，按用户问题、读者和产物组织发现、证据与解释。论文论证、期刊格式与投稿细则按任务选择相关 skill。措辞示例位于该技能的按需附录，不要求全文禁词检查、固定段落数或多稿输出。协调者保留整体论证判断，章节组写作与最终整合由 worker 完成；紧凑交接区分用户约束、科学条件和可调整的编辑建议。

Writing 的能力远不止"改英文"。当前 skills 覆盖论文各章节、项目书、数据可用性声明、文献引用、参考文献核验、科研图件、PPT、投稿回复、投稿前审稿、中文专利草稿、ACS LaTeX 模板、Markdown PDF 和通用 venue 模板。它还可以读取 PDF 或 Office 文档的有界文本，处理已有 LaTeX，生成可编辑图和编译后的 PDF。

Writing 不会替用户发明实验结果，也不应为了让段落更完整而补造引用。缺少文献证据时，可以把任务转给 Literature Review；缺少计算证据时，应明确指出而不是自行扩大为计算项目。

<details>
<summary>Writing 当前角色、tools 与 skills</summary>

入口 Agent 可以调用 `generate_figure` 和 `review_pdf_manuscript`，并委派 `writing_worker_agent`、`plot_worker` 与 `presentation_worker`。Writing worker 通过正常 workspace 文件能力以及 `generate_figure`、`compile_text`、`render_markdown_pdf` 完成交付；系统不再保留独立 polisher Agent 或直接覆写正文的润色工具。Plot worker 直接读取给定定量数据，以 matplotlib 编写可复现绘图代码，显式设置 Origin 风格的坐标轴、刻度、字体和线条，不交付默认样式图；普通分类图使用 Nature/NPG 色系，并检查最终渲染图的裁切、文字碰撞以及文字与科学信号重叠。

Writing 可加载 `publication-launch-writing`、`citation-management`、`scientific-visualization`、`achemso-latex-manuscript`、`venue-templates`、`markdown-pdf-export`，以及共享的 `scientific-communication` 指导。研究设计报告规范是 `publication-launch-writing` 中的按需资料。Plot worker 使用 `publication-data-plotting`，保留客户确认的 Origin 风格、NPG 配色与实际成图检查要求。`presentation_worker` 获得 presentation、writing-quality 和 plotting 三个 skill root：EasySlides 提供可编辑演示制作，共享指导负责科学内容和语言，绘图 skill 支持直接制作数据图。它保留完整文件和命令能力，并绑定 `generate_figure`；模型角色可单独配置，省略时沿用 `section_writer`。见 [EasySlides](../easyslides.md)。

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
