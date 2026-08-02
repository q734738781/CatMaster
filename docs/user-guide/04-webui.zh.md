# 4. 在 WebUI 中与 Agent 一起工作

[上一章](03-llm-configuration.zh.md) | [目录](README.zh.md) | [下一章](05-agents-and-modules.zh.md)

WebUI 把对话、项目文件、Agent 活动、人工审批和运行观测放在同一个页面。日常使用不需要理解后端的每个状态字段，但要养成两个习惯：让重要结果落到 workspace 文件中，并在涉及结构修改或远程计算时查看 Agent 实际做了什么。

## Workspace 与 thread 怎样划分

登录后，左侧最上方是 workspace。它代表一个长期项目，包含文件、对话历史、运行记录和项目经验。一个催化体系、一篇论文或一套机器学习数据通常各用一个 workspace。这样 Agent 读取项目记忆和检索文件时，不会把互不相关的研究混在一起。

同一 workspace 可以有多个 thread。Thread 适合保存一条连续的研究上下文，例如"CeO2 表面模型""ORR 自由能""论文修订第二轮"。在同一个 thread 中继续，Agent 可以利用已有 checkpoint 和产物；换到新 thread，则应把必要的输入路径和前提重新说明。

左侧同时有一个文件树，适合快速打开结构、报告或日志。完整的上传、预览和下载操作在页面上方的 Files 视图中完成。

## 先选择与主要产物匹配的 Entry

Composer 上方可以选择 Research、Experiment、Writing、Peer Review 或 Literature Review。运行开始后不能切换 Entry，因为每个入口会建立不同的 Agent、tools 和 workers。

如果目标是一个明确的结构、计算或轨迹任务，选择 Experiment。只查文献时选择 Literature Review；已经有材料并准备写作时选择 Writing；对固定 PDF 做投稿前审查时选择 Peer Review；只有当目标确实跨越多个阶段、需要 Agent 判断先后关系时，才选择 Research。

Entry 选错不会一定报错，但会让工作变得绕。例如用 Research 扩一个 3x3x1 超胞，会多一层不必要的协调；用 Writing 询问是否应该重新计算吸附能，则缺少计算 worker。第 3、5 和 7 章提供了更完整的选择例子。

## Prompt 应给出研究边界，而不是工具调用脚本

你可以直接用自然语言描述任务。通常需要说明目标、输入文件、不可丢失的约束、允许的工作范围和希望保留的交付物。方法细节已经确定时可以写明；尚未确定时，可以要求 Agent 比较选择并解释依据。

下面这个请求既没有替 Agent 指定每个 tool，也不会让它无限扩张：

```text
使用 Experiment 检查 structures/slab.vasp，并为 CO 吸附建立一组初始候选。
保留现有 Selective Dynamics；先检查表面配位、周期边界和可用吸附区域，
再自主选择合适的 skills 和 tools 枚举去重位点、放置 CO 并生成结构图。

把候选、位点来源和几何审计写到 structures/co_candidates/ 和 notes/co_sites.md。
如果表面模型本身存在问题，请先停下来说明，不要在有问题的 slab 上继续。
本轮不要准备或提交 VASP。
```

输入有单位时写明单位，有电荷或自旋时明确数值，有随机过程时说明 seed 或可重复性要求。已有文件应使用 workspace 相对路径，例如 `structures/slab.vasp`，不要粘贴宿主机上的私人绝对路径。

## Attachments 进入项目后怎样被使用

点击 Attach 可以随当前消息上传图片、PDF、DOCX、XLSX、PPTX、结构或其他文件。附件会先保存到 `files/attachments/<thread_id>/`，并在消息中显示为 artifact。Agent 收到的是可追溯文件，而不是只在浏览器中临时存在的数据。

图片可以在模型 profile 支持视觉输入时直接发送给模型。PDF 和现代 Office 文档会被有界解析为文本；需要看 PDF 图页时，Agent 可以选择页码渲染后检查。音频、视频、旧式 Office 文件和超出限制的媒体可能只被保存，不一定进入模型。Monitor 中的 `multimodal.prepared` 事件会记录是否发送、以什么形式发送以及是否降级。

附件适合当前消息的输入。若文件会在项目中反复使用，最好在 Files 中移动或上传到有意义的目录，例如 `literature/corpus/`、`structures/` 或 `data/`，然后在后续 prompt 中引用稳定路径。

## Chat 中能看到 Agent 的哪些工作

Agent 回复时，Chat 不只显示最终文字。Progress 卡会保留当前推理与阶段说明；Activity 会汇总 tools、subagents 和远程 receipts；单个 tool 卡可以展开查看输入、状态和返回摘要。Agent 写出的结构、表格、图或报告会显示为 artifact，点击后在右侧 inspector 打开。

这些信息用来回答不同问题。Progress 说明 Agent 正在怎样理解任务，subagent 卡说明工作交给了哪个角色，tool 卡说明实际执行了什么动作，artifact 是可以继续使用的结果，remote receipt 则是远程作业的身份与状态证据。

不需要逐条监视每个读文件动作，但出现以下情况时应展开查看：

- Agent 修改了重要结构或源文稿，需要确认输入路径和目标文件。
- 候选数量、筛选条件或参数与预期不同。
- Tool 返回 warning、partial、error 或空结果。
- 远程提交涉及较多任务、GPU、许可证或长 walltime。
- 最终回复与 Files 中的实际产物不一致。

## Research Graph 连接跨 thread 的研究进展

Research Graph 是 workspace 级科学图，不属于当前 thread。顶部 catalog 会列出每个 graph 的研究问题、节点数量、可运行的 frontier、手动或自动模式、最近更新时间，以及当前 thread 是否已经附着。只有一个 active graph 时界面可以预选；有多个时必须显式选择。Attach、Detach 或切换 graph 只改变当前 thread 的关注点，不会复制或删除科学状态。

New graph 只强制填写研究问题。标题、完成条件、编排模式和 seed hypotheses 都收在可选设置中；完成条件留空时，系统使用可见默认值：由已记录 Result 和可追溯来源支撑一个站得住脚的答案。seed Hypothesis 只需一条 claim，也可以在创建时直接附上启发它的论文、note 或其他来源。没有 seed 时，可以直接点击 “Ask Research to propose starting routes”，不必先在 Chat 中让 Agent 初始化。

图中只有三种科学节点：

- Hypothesis 显示简短命题、相对重要性及由所有 Result 派生的关系概览；这不是证据等级。
- Experiment proposal 显示 objective、plan、decision rule、execution lane、预期决策价值、粗粒度算力成本和准备或执行状态。
- Result 显示简短观察或结果，并通过带文字标签的关系连接它支持、反对或无法区分的 Hypothesis。文献发现、合作组结果和历史观察可以不绑定图中 Experiment 直接记录。

画布支持平移、缩放、fit、minimap、键盘访问、focus neighborhood，以及 5、25、100 个节点的密度选择。节点卡保留完整标题和可访问名称。点击节点后，右侧 inspector 显示完整科学字段和来源，不会用不可恢复的字符截断代替内容。

“Add scientific input” 支持先写一两句话：Hypothesis 只需 claim，draft Experiment 只需 objective，Observation/Result 只需 summary；标题、rationale、predictions、关系、优先级、解释和来源都在可选细节中。draft Experiment 可以暂时不完整，但没有 plan 和 decision rule 时不能标记为 Ready，也不能运行。Hypothesis 可以发展实验 proposal、编辑或查看关联证据。Experiment 可以准备、运行、复现、查看 active launch、添加依赖、记录结果或标记阻塞。Result 可以由用户直接发展新 Hypothesis 或 follow-up Experiment；它对任一 Hypothesis 的支持、反对或无法区分判断也可以事后新增、替换或清除，不必重建 Result。图中允许科学循环和分叉；只有 Experiment 的 dependency 关系必须无环。

Research Specialist 创建 graph 后会自动附着到当前 thread。由 Experiment 或
Literature Review 启动的 child 会得到同一个有界 graph focus，但只能为绑定的
Experiment 写回 Result 或具体阻塞原因，且系统会自动把 child thread 附成来源。
如果 child 结束时没有 Result，launch 会显示为 blocked，而不是 completed。

运行 Experiment 会原子占用一次 launch，再创建绑定 graph 和 focus node 的普通 child thread。同一个 active launch 的重复点击会合并，但完成后的 Experiment 可以显式启动 replicate。来源 thread 正在运行、停止或已经删除，都不会阻塞 graph。远程状态不明时，系统先对账已有 thread、run 和 receipt，不会自动重提。

Research planning 会先让 `hypothesis_proposer` 阅读当前证据，并按需检索网络、受控浏览器和本地文献 corpus。它给 Research 返回普通科学语言的 memo，也可以通过绑定的 staging action 发布带来源的临时 Hypothesis/Experiment 分支。分支数量由当前证据支持的科学差异决定，不要求固定的 Hypothesis/Experiment 数量或比例；开始重复已有解释时就停止扩展。临时 Experiment 可以只是只有 objective 的 draft，只有补齐可执行 plan 和 decision rule 后才会成为 runnable。规划会同时考虑完整 runnable frontier，但推荐依据是一段科学理由，而不是持久化数值分数。候选分支在落图前以半透明节点显示。规划 run 属于内部编排，不会污染普通 thread 列表；对用户有用的进度和推荐直接显示在 graph 上。Manual 模式下可以点击任一临时节点把相关路线原子加入图中；未选分支会随下一次规划替换，不进入永久科学图。

Automatic orchestration 会在每次 graph 变化后先重新规划，再最多运行一个真实 Experiment；这只限制执行并发，不限制图中并行 Hypothesis 的数量。如果规划推荐临时路线，系统只实体化该路线；如果推荐已有 ready Experiment，则直接推进它。完成条件被已有 Result 满足后，graph 标记为 Completed 并停止自动推进。切回 Manual 只停止后续自动启动，不取消当前 thread 或远程任务。

Completed 是停止推进标记，不是写保护：新增或修改科学内容会自动重新打开 graph，只补一条来源不会改变完成状态。Archived graph 则是只读的，必须显式 Restore 后才能继续修改。

Graph 节点只保存短科学命题。论文、详细笔记、结构、日志、报告、artifact 和 receipt 仍在原有位置，通过 Sources 连接。来源被移动或删除后会显示 "Source unavailable"，不会静默删除引用。Graph 操作也不等于批准受保护执行；计算仍经过相应 specialist、受管执行和原有审批卡。

其他 thread 更新同一 graph 时，页面通过持久事件流刷新。若你提交编辑前 graph 已变化，服务端会拒绝覆盖并显示可读的冲突说明。刷新后核对新内容，再重新提交。

## Auto 与 Review 代表不同的协作方式

Auto 允许 Agent 在当前权限范围内连续工作，适合读取、分析和已建立信任的项目流程。Review 会在 `remote_submission` 和 `remote_submission_batch` 前暂停，消息中出现审批卡。本地 `write_file`、`edit_file` 与 Codex OAuth 的 `apply_patch` 不会弹出审批卡。

线程可能提交真实远程计算时，可以使用 Review。审批卡提供四种处理方式：

- Approve 按当前 action 执行。
- Reject 拒绝这次 action，可以附上原因。
- Respond 给 Agent 补充说明，让它根据反馈重新处理。
- Edit action 直接修改 action JSON，适合熟悉 tool schema 的高级用户。

Review 不是所有行为的总开关。读取、搜索、分析和本地文件编辑仍可自动进行。它的价值是把会真实提交远程计算的动作交给用户确认。审批应在消息卡中完成；不要另发一条普通消息冒充审批结果。

`write_file`、`edit_file`、Codex OAuth `apply_patch`，以及会生成声明输出的 `supercell`、`build_slab` 等领域 tool 都会直接写入 workspace。对这类操作，应在 prompt 中给出目标目录，在 tool 卡中核对输入与输出路径，并在 Files 中审查产物。Review 是远程提交保护层，不是 workspace 变更的事务锁。

## 运行中可以 Steer，但不必把每个想法都打断进去

空闲时按钮为 Send。Agent 正在运行时，发送纯文本会变成 Steer。Steer 不会强行终止正在执行的 tool；当前 tool 完成并写入 checkpoint 后，排队消息会从同一 thread 状态继续。如果当前模型不再调用 tool，则本轮正常结束后再应用 Steer。它适合补充刚发现的约束，例如"不要覆盖原结构""只分析前 20 ps""把两个终止面都保留"。

如果新要求彻底改变任务，等待当前安全停下后开一个新 thread 通常更清楚。运行期间不能附加新文件，所以需要新增输入时，可以先 Stop 或等待结束，再上传并继续。

Stop 会请求本地 Agent 在流事件边界停止；连续请求会升级为 emergency cancel。它不会自动取消已经提交到远程调度器的作业。远程作业必须根据 receipt 和集群状态单独处理。

## Files 是交付物所在的地方

Files 视图提供 Browse、Preview 和 Uploads。它可以预览文本、Markdown、JSON、图片、PDF、CSV/TSV、常见晶体与分子结构、轨迹、体数据以及部分 OUTCAR 振动内容。

晶体、slab、defect、adsorbate 和普通分子预览以 MatterViz 为主。点击 **Open Structure Workbench** 后进入全屏工作台，可以按 base atom 选择，编辑坐标、晶胞与约束，测量距离和角度，undo/redo，预览 supercell、对称性、termination、defect 和 adsorption candidates，再明确 Save As。显示复制只用于观察；要建立一个真实单缺陷，必须先 Make supercell。大结构仍以完整源模型执行选择和保存，画布只切换为有界显示。

分子文件按需加载 Ketcher 二维编辑器，并用 MatterViz 查看三维构象。SDF/MOL 的 connection table 是权威数据。分子改存 XYZ 会丢失键、芳香性、键级、电荷和立体化学，改存 SMILES 会丢失当前三维坐标；Workbench 会先阻止保存并要求确认。周期结构约束可以通过 POSCAR/VASP 和 ASE `.traj` 往返；目标格式无法表达约束时也会给出同样明确的警告。

轨迹以只读方式打开，显示真实总帧数；可以 scrub、play、查看标量性质，并在 Extract frame 后编辑单帧。CUBE、CHGCAR、LOCPOT、ELFCAR 和 XSF 作为独立 volume artifact 打开，支持结构 overlay、正负等值面和切片。JSmol 只保留给 OUTCAR vibration 和主 renderer 无法打开的兼容格式，不维护第二份可编辑状态。VESTA 生成的标准视图仍可作为图片 artifact 打开。

Agent 报告完成后，至少检查核心交付物是否真的存在，文件名和目录是否符合约定。结构任务看候选与审计，计算任务看 stage、status、stdout/stderr 和分析，文献任务看候选表、证据表与引用库，写作任务看可编辑源文件而不只看编译 PDF。

Files 上传同名文件会覆盖，删除目录是永久递归操作。重要原始数据在 workspace 外应有备份。普通文件树只展示用户交付物和工作文件；内部 metadata、tool-result offload 与临时抽取仍保留给 diagnostics，但不会伪装成用户交付物。

## Monitor 用来判断过程是否正常

Monitor 将一次运行的模型、Agent、tools、tasks、token、费用和机器时间汇总起来。每次 LLM 调用完成后都会更新 token 统计；provider 提供时会分别记录 input、output、cache 和 reasoning token，仍在运行的单次调用则暂时没有最终用量。Overview 适合快速看状态和规模；Live 显示当前阶段、活动工具、Todo、subagent 与近期日志；Events 可以按 thread、run、agent、tool、category 和 channel 过滤；Raw 和 Details 用于排查更具体的问题。

当 Agent 看似停住时，先看 Live 中是否仍有远程 tool 或 subagent 在运行。当结果不完整时，查 Events 中的 tool error、document warning 或 multimodal 状态。当成本异常时，查看模型调用、token 和远程机器时间。Monitor 是诊断界面，不需要作为日常报告手工抄写。

当前界面没有历史 run 选择器，Overview 也可能汇总 workspace 与 lane 的当前或最近运行。精确追踪某次远程计算时，应同时核对 thread ID、run ID、artifact 和 receipt。

## 右侧 inspector 适合边对话边审阅

点击 artifact 或文件后，右侧 inspector 会打开标签页。你可以保持 Chat 可见，同时对照结构、报告、表格或日志继续提问。Todo 标签显示当前 turn 中最新 `write_todos` 的 canonical 只读投影，消息分页和页面刷新不会改变其完成状态；它不是用户需要手工维护的项目管理器。

一个自然的复查请求可以直接指向刚打开的文件：

```text
我正在看 notes/slab_audit.md。请结合对应结构重新检查第 3 个终止面，
解释报告中 CN=1 的判定阈值，并把该终止面的俯视和侧视图与第 1 个并排比较。
先分析，不要删除或覆盖任何候选。
```

## Skill Evolution 处理项目中反复出现的经验

登录模式下，每个有用户任务的 terminal run 会进入一次 Skill Evolution 语义反思。模型读取完整的已记录轨迹和结果，区分无需学习、执行 Agent 没有遵守现有 skill，以及确实需要修改长期行为。系统不用正则、embedding 或固定复现次数替模型判断。一个明确的长期 correction 可以单独形成证据；多次措辞相似也不自动产生候选。产品/schema 缺陷与详细科学事实不会写成 skill；已有 owner skill 能承接时优先修订。

候选卡片优先展示行为变化、evidence episode 与来源、适用边界、静态检查、reviewer counterexample、concerns 和人工核对项，而不是 raw JSON。Candidate 与 observation 列表为 newest first，可按状态筛选并 Load more；精确 revision 的 diff 只在 Technical details 中按需打开。生命周期只使用 `pending`、`review`、`revision`、`canary`、`stable`、`rejected` 和 `inactive`。AI reviewer 只给建议。用户可以 Request revision 或 Reject；skill 必须先绑定到明确指定的 thread/run 做 canary，并有一次成功的真实使用，才会出现 Promote stable。Start canary 只改变精确版本指针，不会新建对话或触发额外模型调用。目标已变化时 candidate 回到 `revision`，不能覆盖；canary 失败只停止对应 pointer，stable revision 以后可以 Quarantine、Retire 或 Roll back。

## 继续一项已经中断或隔天再做的工作

回到原 workspace 和原 thread，先让 Agent 重新读取关键产物，而不是只发"继续"。说明应保留哪些结果、上次停在哪里、是否禁止重复计算，以及这一次的目标：

```text
继续上次的 CO 吸附筛选。先读取 notes/co_sites.md、structures/co_candidates/ 和
calculations/mlff_screen/output/，核对已有候选、失败项和排序依据。

不要重新生成或提交已经完成的结构。请判断哪些候选值得进入 VASP，
列出原因和需要我确认的共同设置；本轮停在 VASP stage 审阅前。
```

如果上一次涉及远程错误，先读取 receipt 并判断旧作业状态，禁止把"重试"理解为重新提交。第 8 章给出了相应恢复 prompt，第 11 章收录更具体的诊断方法。

## 当前界面的重要限制

WebUI 目前不能重命名、删除、分支或 retry thread，也不能从历史 run 选择器恢复某个任意节点。Research Graph 可以跨 thread 管理科学分支，但它不是 thread 历史或 rollback 控件。Files 上传同名文件会覆盖，删除不会进入回收站。审批中断必须用消息内卡片恢复。Stop 不取消远程 job。Skill Evolution 只在登录模式显示，并从下一次 run 生效。

这些限制不会阻止正常研究流程，但会影响如何备份、续跑和停止任务。重要文件使用版本化名称或 Git/外部备份；远程任务依靠 receipt；相互独立的目标或不兼容的项目范围使用不同 thread。这样可以避免把 UI 中缺少的操作误认为 Agent 会自动补齐。
