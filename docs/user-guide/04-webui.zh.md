# 4. 在 WebUI 中与 Agent 一起工作

[上一章](03-llm-configuration.zh.md) | [目录](README.zh.md) | [下一章](05-agents-and-modules.zh.md)

WebUI 把对话、项目文件、Agent 活动、人工审批和运行观测放在同一个页面。日常使用不需要理解后端的每个状态字段，但要养成两个习惯：让重要结果落到 workspace 文件中，并在涉及结构修改或远程计算时查看 Agent 实际做了什么。

## 页面导航

没有选中 workspace 时，页面提供工作区选择和创建入口。首次使用需先填写名称创建工作区；登录与无登录模式均不会自动创建 `default` 或 `admin`。已有工作区可从左栏选择，也可以通过链接中的 `project_space` 打开。失效链接会回到选择/创建页面，不会创建同名目录。

左侧栏集中放置 workspace 选择器、**New conversation**、页面导航、会话文件夹和紧凑文件树。Chat 是主要工作区；Research Graph、Files、Monitor 和 Skill Evolution 都从左侧进入。点击已有会话会直接回到 Chat。

空白会话提供文献调研、计算和研究规划三个起点。点击只会填入可编辑的草稿，不会发送消息或启动任务。输入区内的 Agent 选择器与 Review/Auto 控制跟随草稿显示。桌面端可用顶部的 Task Context 按钮收起或展开右栏；窄屏时，导航和任务上下文分别作为抽屉打开。

在 Research Graph 中展开 **Research graphs** 可浏览或切换图。选中图的完整问题、完成标准和决策偏好保留在 **Research question & completion criteria** 下，节点和关系因此获得更多展示空间。

## Workspace 与 thread 怎样划分

登录后，左侧最上方是 workspace。它代表一个长期项目，包含文件、对话历史、运行记录和项目经验。一个催化体系、一篇论文或一套机器学习数据通常各用一个 workspace。这样 Agent 读取项目记忆和检索文件时，不会把互不相关的研究混在一起。

同一 workspace 可以有多个 thread。Thread 适合保存一条连续的研究上下文，例如"CeO2 表面模型""ORR 自由能""论文修订第二轮"。普通 turn、审批恢复和 checkpoint continuation 都复用同一个 本地执行宿主 thread 及其 checkpoint；每次提交则创建独立的原生 run。CatMaster 保存的是供 WebUI 检索和展示的投影，不另行决定 run 的排队、停止或恢复。在同一个 thread 中继续，Agent 可以利用已有 checkpoint 和产物；换到新 thread，则应把必要的输入路径和前提重新说明。尚未发送消息的新 thread 显示为 `New thread`；首条普通消息提交后会立即出现本地标题，并在配置了轻量标题模型时于后台更新为更短的语义标题。标题生成失败不会影响研究任务，手动改名始终优先。点击 thread 右侧的铅笔可以就地改名；Enter 或勾选按钮保存，Escape 或取消按钮放弃修改。

同一浏览器标签页刷新后会恢复刷新前明确打开的 thread，包括从 Task Context 打开的后台执行 thread；该 thread 已不存在时才回到可见的根对话。已有消息加载完成后，Chat 会落到最新消息，而不会停在旧的后台回执处。新标签页没有这份标签页内选择记录，仍默认打开用户可见的根对话。

左侧同时有一个文件树，适合快速打开结构、报告或日志。桌面端可以拖动左栏右边界调整宽度，设置会被浏览器记住；聚焦该分隔条后也可用方向键、Home 和 End 调整。左侧文件树与完整 Files 浏览器都会为超长文件名提供横向滚动条，不会把名称后半段变成不可达内容。完整的上传、预览和下载操作在左侧导航进入的 Files 视图中完成。

## 先选择与主要产物匹配的 Entry

Composer 上方可以选择 Research、Persistent Research、Experiment、Writing、Peer Review 或 Literature Review。同一个长期 thread 可以在不同轮之间切换 Entry，并继续复用原有对话 checkpoint；运行中的一轮不能切换，因为每个 Entry 会建立相应的 Agent、tools 和 workers。Chat 中每条 Agent 消息会显示该轮实际使用的 Entry，不能用 thread 当前选择反推旧消息的角色。

Graph 附着与 Persistent Research 编排是两件事。普通 Research 可以自动附着 workspace 中唯一的 open Graph、聚焦其中的 Experiment，并独立执行或回写 Result；它不会因此成为 Research Session 根，也不会进入额外规划线程或 session steering。若当前入口是 Persistent Research，点击 New thread 后，新 thread 使用 workspace 默认入口，而不是复制当前入口；标准配置下就是普通 Research。在两个回合之间把 Persistent Research 根切换到普通 Entry 时，Graph 和 focus 会保留，但持续编排随即结束；之后发送消息会启动该 thread 的正常新回合。

如果目标是一个明确的结构、计算或轨迹任务，选择 Experiment。只查文献时选择 Literature Review；已经有材料并准备写作时选择 Writing；对固定 PDF 做投稿前审查时选择 Peer Review；目标跨越多个阶段但希望按需收束时选择 Research；希望同一目标通过 Research Graph 自动长程推进时选择 Persistent Research。

Entry 选错不会一定报错，但会让工作变得绕。例如用 Research 扩一个 3x3x1 超胞，会多一层不必要的协调；用 Writing 询问是否应该重新计算吸附能，则缺少计算 worker。第 3、5 和 7 章提供了更完整的选择例子。

## Persistent Research 会话显示为文件夹

一次 Persistent Research 在左侧栏保留稳定的根入口，原生异步 specialist 的活动显示在任务上下文中。已有实验执行线程仍可在 Research Session 文件夹中打开；历史规划/比较记录仍可查询。根线程处理子任务结果和相关科学证据变化。Research Graph 属于 workspace，文件夹不会复制科学记录或产物。

文件夹显示整项研究的聚合状态，而不是只显示根 thread 当前是否正在生成消息。因此，即使根对话已经 idle，只要子 thread 还在规划、调用工具或等待远程计算，它仍会显示当前实验和最近进度。`waiting_continue`、`waiting_review`、`operationally_incomplete`、暂停和完成会分别呈现，不会都压成一个含糊的 waiting。打开根对话后，右侧 Task Context 的 Research progress 会保留当前阶段和 `Running now` 活动；需要完整上下文时再打开对应的实验子 thread。

研究分支或执行者登记有意义的 Result 后，根对话会显示 `Research milestone`，包含结果、结论和已记录的方法；修正已有 Result 会更新同一条消息。计算输出文件本身不会自动变成 Result。Research Session 面板显示最新科学结果；若 Result 引用了报告，`Open latest report` 可直接打开预览。活动状态包含嵌套的后台研究与执行任务，根对话空闲不代表研究已经停止。`Waiting — research unfinished` 表示目标尚未完成。

根对话也是持续控制入口。发给根对话的消息始终留在这个 thread，不会被隐式转发给某个执行 child；当原生 run 尚未结束时，Composer 会要求明确选择 **Steer**、**Queue**、**Replace** 或 **Reject if busy**，并把相应的 `interrupt`、`enqueue`、`rollback` 或 `reject` 策略直接交给 本地执行宿主。要修改具体后台分支，应在 Task Context 对相应 async specialist 使用 Steer；Stop 也只针对界面所示的原生 run。Pause automation 只阻止后续规划和实验启动，已经提交的远程任务继续运行。恢复自动编排后，系统继续使用已有 graph、launch 和 thread 关系，不会重复提交现有执行。

搜索会同时匹配文件夹标题和实验子 thread，子项结果显示 `根会话 / 子 thread` 面包屑并自动展开所属文件夹。活跃或需要处理的会话默认展开，其余展开偏好由浏览器保存。刷新、重连或服务重启不会把子 thread 重新摊平。无法可靠确定旧执行归属的记录会进入 `Related research activity` 临时组，不会猜测一个错误的父会话。

## 原生 async specialist 怎样回到根对话

后台卡片显示任务的 low/medium/high 成本档位。容量不足时显示“等待 … 研究槽位 · 将自动继续”，无需点击审批或继续；原任务会在获得名额后接续。等待计算结果的研究者仍占用名额。独立研究分支可使用 ResearchSpecialist，拥有自己的完整研究上下文，主对话仍可继续交互。

Research 可以通过 后台 specialist 接口，把独立的 Literature Review、Experiment、Writing 或 Peer Review 分支提交到各自的 本地执行宿主 thread/run。主对话保留每个子任务的卡片；右侧 Activity 显示仍在运行的子任务、最新进展和当前工具。点击 **查看过程** 打开大幅过程视图，可按全部、进展与结果、Reasoning、工具筛选，并区分 specialist 与嵌套 worker 的输出。工具详情支持完整输入、结果及分页读取，打开后可返回子任务过程。

worker 的同一次回复从流式生成到结束会持续更新同一条消息，重连后也保持对应关系。同一 worker 的多条消息表示先后的回复，不代表新增了多个 worker。完成、中断和失败的活动均保留已收到的文字与工具结果，并显示对应状态。

切回浏览器页面或断线重连时，界面会先读取最新状态，再接入实时更新。Activity 卡片直接显示当前进展，不逐条播放离开期间的旧工具调用；完整过程仍保留在详情中。

卡片与任务说明把最初委派和最新“补充指令”分开显示。补充指令按对应执行轮次显示等待处理、正在处理、该轮已完成或中断/失败；这不表示模型已经理解或解决了指令中的问题。旧记录缺少对应轮次时只显示原文。主 agent 更正子任务正在使用的前提、来源判断、方法或范围时应选择 `interrupt`；不影响当前工作的新增证据、后续问题和纯措辞修改可以 `enqueue`。排队指令要等当前轮及其同步 worker 返回后才会处理。

过程视图中的 **Steer** 将新指令发给所选子任务，在同一 thread 上接续；**停止子任务** 针对当时显示的 run。关闭视图或刷新浏览器不会停止任务。父对话完成或继续下一轮时，独立子任务仍在 Activity 中；完成后的过程可从原卡片重新打开。旧 run 未记录过的嵌套 worker 消息无法事后补回，已有 checkpoint 消息仍可查看。

后台任务默认完成后将结果作为新输入排入主对话，由主 agent 综合交付。提交方也可选择 `on_completion=notify`，只保存结果和更新 Activity。停止/暂停优先。通知属于持久执行流程，浏览器关闭或重连不影响投递。

因此，`Thread ready` 只表示根 thread 当前没有活动 run；它可以与仍在运行的 async specialist 同时出现。并行 specialist 应保持只读或写入不同输出；可能写入同一路径时，应在任务设计中拆分路径或安排先后关系。

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

图片可以在模型 profile 支持视觉输入时直接发送给模型。较小的 PDF、DOCX、XLSX 和 PPTX 可以作为原生文件块发送；较大或文本很多的文档只先保存，随后由同一个 `read_file` 按 offset 分页读取有界文本。需要看 PDF 图页时，Agent 可以选择页码渲染后检查。音频、视频、旧式 Office 文件和不支持的媒体可能只被保存，不一定进入模型。Monitor 中的 `multimodal.prepared` 事件会记录是否发送、以什么形式发送以及是否降级。

附件适合当前消息的输入。若文件会在项目中反复使用，最好在 Files 中移动或上传到有意义的目录，例如 `literature/corpus/`、`structures/` 或 `data/`，然后在后续 prompt 中引用稳定路径。

## Chat 中能看到 Agent 的哪些工作

Agent 回复时，Chat 不只显示最终文字。同一用户任务中的 Todo 更新（包括 checkpoint 恢复产生的新 assistant message）会按语义 agent 角色合并到最后一条回复顶部，每个角色只显示一张最终 Plan 卡片；中间快照仍保留在底层轨迹中。推理、阶段说明和 tool calls 作为中间活动层显示在最终正文之前，并按一次具体的 subagent 生命周期归组。同名 worker 的两次调用仍是两个独立活动组。短组直接展开；活动较多或包含单条大段推理的组默认折叠，只显示当前或最后一项活动，展开后仍可查看未经摘要替换的完整轨迹。远程 receipts 和 artifacts 继续独立显示；点击 artifact 或完整活动详情会打开居中的大预览窗口，不会压缩 Chat 或占用右栏。

长任务运行时，Composer 正上方会固定显示 **Running now**。它只是同一份持久 active tool 记录的醒目投影，不是第二套 Monitor：运行最久的项目排在最前，多项并发时可以展开，并显示该 tool 自己的实际开始时间和前端本地更新的已用时。因此，即使受管计算阻塞期间没有新模型事件，或原始 tool 卡已被大量轨迹埋住，当前操作仍然可见；只有收到 terminal tool event 后才会离开该区域。刷新页面会从 current turn 恢复这张卡；服务异常退出后重启时，由同一 WebUI 实例拥有的旧任务会标为 Interrupted，不再永久显示 Running。界面不会猜测 ETA、scheduler phase 或完成百分比。

Research 与执行 Agent 也可以在主要科学阶段切换、重要 delegation 或阻塞式受管计算之前，以及新证据实质改变计划时，发布一条简洁的语义进度。最新一条会直接显示在 Plan 卡片上方，较早的阶段更新保留在同一卡片的紧凑历史中，不再埋进 tool trace。这类更新刻意保持低频，不是执行前置条件，也不是周期心跳。原始 reasoning 仍可在下方展开；独立 block 现在会显示来源和边界，不再直接拼成一串文字。

这些信息用来回答不同问题。Progress 说明 Agent 正在怎样理解任务，subagent 卡说明工作交给了哪个角色，tool 卡说明实际执行了什么动作；当紧凑卡片不够时，Technical details 可通过认证的字符分页读取完整已存 input 和安全脱敏后的 output。Artifact 是可以继续使用的结果，remote receipt 则是远程作业的身份与状态证据。Artifact 与 receipt 会完整注册，再由 UI 分页展示，不会因为卡片内联数量限制而丢掉后续记录。

不需要逐条监视每个读文件动作，但出现以下情况时应展开查看：

- Agent 修改了重要结构或源文稿，需要确认输入路径和目标文件。
- 候选数量、筛选条件或参数与预期不同。
- Tool 返回 warning、partial、error 或空结果。
- 远程提交涉及较多任务、GPU、许可证或长 walltime。
- 最终回复与 Files 中的实际产物不一致。

## Research Graph 连接跨 thread 的研究进展

Persistent Research 的“研究协作”显示同图分支目标、状态、当前工作和下一步，并支持完整任务说明、讨论、回复、节点过滤和历史查询。分支可以直接交流。普通 Research 会话不会启用讨论工具或讨论收尾检查；已有图谱证据和历史讨论仍可查阅。

主研究者在接收结果、综合和收尾时查看相关留言，包括分支之间尚未回答的方法问题。它根据实际证据决定直接回答、说明为何暂不接续，或给原 subagent 下发一个具体的后续问题。不影响当前工作的追加问题排队；纠正当前前提或指令时使用中断接续，两者都保留原线程上下文。深层 worker 通过所属研究分支继续，不改变父子任务控制关系。

留言本身不唤醒任何 agent。主线程已经空闲时，新留言等待下一次用户请求或原有任务完成接续。讨论卡片可显示“已处理”“暂不接续”“已安排后续调查”，并保留处理回复；安排调查不等于得到结论，也不是额外的启动按钮。未逐条回复不会阻止完成研究。暂停设置、任务完成通知选项和用户授权继续有效，讨论不自动重开已完成阶段或授权新计算、实验。

Research Graph 是 workspace 级科学图，不属于当前 thread。顶部 catalog 列出研究问题、节点数量、可运行的 frontier、最近更新时间，以及当前 thread 是否已经附着。图谱目录供用户浏览和显式选择，不传给 Agent。Attach、Detach 或切换 graph 只改变当前 thread 的关注点，不会复制或删除科学状态。

后台在接收请求时确定本轮图谱绑定，并在 Agent 输入中明确提示查询目标；后台委派
继承相应绑定。Agent 的图谱查询和科学写入只使用这张图，工具不要求它提供 graph ID，
也不提供列出全工作区图谱、另建图谱或切图的能力。节点选择、科学方法和 revision
冲突处理仍由 Agent 完成。图谱里保存的问题和完成条件与当前请求分开显示；新授权
可以使用已完成阶段的证据，附着本身不会自动改写科学完成状态。

New graph 只强制填写研究问题。标题、完成条件、用户明确表达的 decision preferences 和 seed hypotheses 收在可选设置中；完成条件留空时，系统使用可见默认值：由已记录 Result 和可追溯来源支撑一个站得住脚的答案。seed Hypothesis 只需一条 claim，也可以直接附上启发它的论文、note 或其他来源。用户可以从一个问题开始，在 Persistent Research 对话中交给根研究者探索。

Research、Persistent Research、Experiment 或 Literature Review 提交回合时，后台优先
保留 thread 的有效已有绑定；未绑定时复用创建时间最新的未归档图，包括已完成图。
修改较早图的内容不会使它成为“最新创建”的图。仅当 workspace 没有未归档图时，
后台才用首次请求初始化一张图，后续新 thread 默认复用它。用户仍可在界面显式新建
或切换图谱。子任务继承已接受的绑定，不自行创建或重新选择。Writing 不会静默选择
或创建 Graph，一次性任务也无需人为凑齐 H/E/R 节点。

图中只有三种科学节点：

- Hypothesis 显示简短命题、相对重要性及由所有 Result 派生的关系概览；这不是证据等级。
- Experiment proposal 显示 objective、plan、decision rule、execution lane、粗粒度算力成本和准备或执行状态。`external` lane 表示可交给实验室或协作者的完整交接，只有实际返回的观察才作为实验结果。
- Result 显示简短观察或结果，并通过带文字标签的关系连接它支持、反对或无法区分的 Hypothesis。文献发现、合作组结果和历史观察可以不绑定图中 Experiment 直接记录。

画布支持平移、缩放、fit、minimap、键盘访问、focus neighborhood，以及 5、25、100 个节点的密度选择。节点卡保留完整标题和可访问名称。点击节点只会在 Research Graph 页面自己的详情抽屉中显示完整科学字段和来源，不会改变 thread 的 graph focus；需要用 **Set focus** 或 **Clear focus** 显式设置。保存后的 focus 会在画布上标出，并成为该 thread 后续回合的起始分支。

“Add scientific input” 支持先写一两句话：Hypothesis 只需 claim，draft Experiment 只需 objective，Observation/Result 只需 summary；标题、rationale、predictions、关系、优先级、解释和来源都在可选细节中。draft Experiment 可以暂时不完整，但没有 plan 和 decision rule 时不能标记为 Ready，也不能运行。选择 **External lab / collaborator** 后，Ready 表示交接内容已经完整，页面不会显示 Run 或 Replicate，而会显示 **Record external result**；实验组完成后在这里填写观察、来源和 Hypothesis 判断。Hypothesis 可以发展实验 proposal、编辑或查看关联证据。内部 Experiment 可以准备、运行、复现、查看 active launch、添加依赖、记录结果或标记阻塞。Result 可以由用户直接发展新 Hypothesis 或 follow-up Experiment；它对任一 Hypothesis 的支持、反对或无法区分判断也可以事后新增、替换或清除，不必重建 Result。图中允许科学循环和分叉；Experiment 的 dependency 关系和同类主张的修订链各自必须无环。

Research Specialist 显式创建 graph 后也会自动附着到当前 thread。Research、顶层 Experiment、Literature Review 和 Writing 可以在已绑定 graph 内显式移动或清除当前 thread 的 focus；child agent 不会自动继承新 focus。Graph 身份和 active launch 仍由 host 绑定，因此 focus 不能把已经存在的 launch 改指另一个 Experiment。

顶层 Experiment 或 Literature Review 只有在绑定 graph 时才获得只读查询和 Result 写回。由 Experiment 产生的 Result 必须有真实 Experiment focus；缺少时会以非破坏性错误返回。临时计算得到值得复用的科学 Result 后，顶层 Experiment 可以显式创建并 focus 一个 Experiment，再记录关联 Result。Literature Review 的有来源发现不需要虚构回溯性 Experiment；没有 Experiment focus 时可以记录 standalone Result。普通一次性任务仍可不生成图节点。授权、输入准备、source/model 获取、兼容恢复、build 或 scheduler 诊断和平台可行性工作可以在不伪造 Result 的情况下结束；只有科学 decision rule 确实无法完成时才把 Experiment 标为 blocked。系统自动附上实际 thread 和 run 来源；Writing 可以改变自己的导航 focus，但仍不能修改科学图内容。

运行 Experiment 会原子占用一次 launch，再创建绑定 graph 和 focus node 的普通 child thread。同一个 active launch 的重复点击会合并，但完成后的 Experiment 可以显式启动 replicate。准备与科学等价的恢复仍属于同一个 Experiment。完成的科学观察才创建 Result；同一 run、dataset、条件与观察的澄清或来源补充会经过 revision 检查原位更新该 Result，并保留既有关系和来源。新的 run、条件、dataset 或独立观察才新建 Result。error 或 stop 只形成可重试的 operationally incomplete 状态，不会被解释成科学结论。真正写回时只完成该轮精确绑定的 launch，后续 Writing、追问或补充分析不会改写旧 launch 的 run。远程状态不明时，系统先对账已有 thread、run 和 receipt，不会自动重提。

根 ResearchSpecialist 负责整体目标与综合交付，可以将独立科学问题交给 Research 分支。每个分支自行提出假说、选择方法、委派实验、解释结果，并在已有授权内继续下一轮；阶段成果由产出者写回 Graph。Literature、Experiment、Writing 等领域 specialist 保留各自职责。`hypothesis_proposer` 用于按需独立解释和停止前复查，不是每轮研究的必经规划入口。只有 objective 的 Experiment 可以先作为 Draft 保存。

Persistent Research 准备因科学停滞结束、而用户要求仍未满足时，会进行一次独立复查；存在已授权的具体补救后，默认完成一次有界验证。文献调研和实验推荐只授权来源核对、综合与建议，不会因此开始计算或实验室工作。达到用户请求即交付并停止。开放前提、复查意见与恢复条件保存在图谱中。

Research Session 面板的 `Pause research` 停止根线程当前回合及后续自动续接，使用与聊天 Stop 相同的控制；已运行的子任务保留独立 Stop，远程计算也保留原有执行控制。`Resume research` 向同一根线程提交继续原目标的请求，复用在途工作和已完成结果；直接发送继续或补充指令也可恢复会话。后台结果通知不会自行解除暂停。普通 Research 的图谱附着不会把它变成持续模式。输入区的 Auto/Review 是工具审批偏好，与持续研究的暂停/恢复不同。

Completed 表示用户当前阶段已经完成，仍允许记录新来源、晚到 Result 和科学修订；这些编辑不会自动重开阶段。用户可以显式重开目标。Archived Graph 为只读，须先 Restore。

Graph 节点只保存短科学命题。论文、详细笔记、结构、日志、报告、artifact 和 receipt 仍在原有位置，通过 Sources 连接。来源被移动或删除后会显示 "Source unavailable"，不会静默删除引用。Graph 操作也不等于批准受保护执行；计算仍经过相应 specialist、受管执行和原有审批卡。

Writing thread 显式 Attach graph 后，每个 turn 会得到同一份 partial focus context，Writing coordinator 也可只读查询完整绑定图。它先定位与当前章节有关的 Result、相反或无法区分的判断及其 Sources，再定点打开原始 note、artifact、run、thread message、DOI 或 URL。Result summary 只是导航，不替代原始证据；Writing 不能修改 graph。未 Attach graph 的 Writing 行为保持不变，存在多个 graph 时也不会按标题猜测。

其他 thread 更新同一 graph 时，页面通过持久事件流刷新。若你提交编辑前 graph 已变化，服务端会拒绝覆盖并显示可读的冲突说明。刷新后核对新内容，再重新提交。用户可以经过 revision 检查编辑 Result，或填写 audit reason 后删除。Agent 的 retraction 更窄，只允许删除当前 run 意外写入、尚未 judgment 且没有 dependent scientific node 的 Result。Overview 把 recent mutation history 与科学节点分开展示，更早的记录仍可按稳定 event ID 分页读取。

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

输入区把 Agent/权限设置、当前运行的停止控制、新消息发送分行对齐。停止时可选择保留进展或丢弃本轮；发送策略带独立图标，窄屏仍保留文字与操作入口。Ctrl+Enter 与点击发送使用同一策略。空闲时按钮为 Send。Agent 正在运行时，输入框会明确提供 Steer、Queue、Replace 和 Reject if busy。Steer 使用 本地执行宿主 的 `interrupt` 策略：请求中断当前 run，并从已保存 checkpoint 接受新消息；Queue 保留当前 run 并排队；Replace 回滚当前 turn 的 checkpoint 效果后执行新消息；Reject if busy 则在仍有活动 run 时拒绝提交。正常 Steer/Stop 后，原消息保留已有输出并显示为已中断。当前 provider 调用或 tool 若不能立即取消，界面只能表示中断请求已经接纳，不能声称模型已经读到新指令。

如果新要求彻底改变任务，等待当前安全停下后开一个新 thread 通常更清楚。运行期间不能附加新文件，所以需要新增输入时，可以先 Stop 或等待结束，再上传并继续。

Stop 直接针对当前 本地执行宿主 run。Keep progress 使用 `interrupt` 保留已有 checkpoint，Discard this turn 使用 `rollback` 丢弃本 turn 的 checkpoint 效果。它不会自动取消已经提交到远程调度器的作业；远程作业必须根据 receipt 和集群状态单独处理。

## Files 是交付物所在的地方

Files 视图提供 Browse、Preview 和 Uploads。它可以预览文本、Markdown、JSON、图片、PDF、CSV/TSV、常见晶体与分子结构、轨迹、体数据以及部分 OUTCAR 振动内容。

Agent 的文件系统根目录就是这个 Files 树，因此 `reports/result.md` 与界面路径写法 `files/reports/result.md` 指向同一文件。聊天中返回的文件链接（包括 `sandbox:/files/...`）会在遮罩式大预览窗口中直接打开对应文件。成功回复若在明确的 `## Files` 小节列出真实存在的 workspace 文件，Chat 也会把它们注册为可直接打开的 artifact 卡片，使最终报告不必再靠用户手工寻找。

Agent 也可以把 workspace 中的图片直接放进回答，例如 `![拉伸曲线](sandbox:/files/figures/tensile.png "应力-应变曲线")`。WebUI 会在消息原来的位置显示一张自适应图卡，把图题和“View larger”入口收在简洁的底栏中；点击图片本身也会打开同一个大预览。消息只保存 workspace 路径，不保存 base64 图片副本。这个方式适合结构渲染、关键曲线和机理示意图；其他文件继续使用普通链接或 artifact 卡片。

晶体、slab、defect、adsorbate 和普通分子预览以 MatterViz 为主。点击 **Open Structure Workbench** 后进入全屏工作台，可以按 base atom 选择，编辑坐标、晶胞与约束，测量距离和角度，undo/redo，预览 supercell、对称性、termination、defect 和 adsorption candidates，再明确 Save As。显示复制只用于观察；要建立一个真实单缺陷，必须先 Make supercell。大结构仍以完整源模型执行选择和保存，画布只切换为有界显示。

XYZ、extxyz 等坐标文件默认打开 MatterViz 三维视图。预览和轨迹播放不要求价态检查通过，也不自动指定键级或电荷；显示连线只是几何估计，无法推断连线时仍可查看原子。SDF/MOL 等分子文件按需加载 Ketcher 二维编辑器，其 connection table 是权威数据；明确的化学编辑和构象生成保留相应检查。分子改存 XYZ 会丢失键、芳香性、键级、电荷和立体化学，改存 SMILES 会丢失当前三维坐标；Workbench 会先阻止保存并要求确认。周期结构约束可以通过 POSCAR/VASP 和 ASE `.traj` 往返；目标格式无法表达约束时也会给出同样明确的警告。

轨迹以只读方式打开，显示真实总帧数；可以 scrub、play、查看标量性质，并在 Extract frame 后编辑单帧。CUBE、CHGCAR、LOCPOT、ELFCAR 和 XSF 作为独立 volume artifact 打开，支持结构 overlay、正负等值面和切片。JSmol 只保留给 OUTCAR vibration 和主 renderer 无法打开的兼容格式，不维护第二份可编辑状态。VESTA 生成的标准视图仍可作为图片 artifact 打开。

Agent 报告完成后，至少检查核心交付物是否真的存在，文件名和目录是否符合约定。结构任务看候选与审计，计算任务看 stage、status、stdout/stderr 和分析，文献任务看候选表、证据表与引用库，写作任务看可编辑源文件而不只看编译 PDF。

Files 上传同名文件会覆盖，删除目录是永久递归操作。重要原始数据在 workspace 外应有备份。普通文件树只展示用户交付物和工作文件；内部 metadata、tool-result offload 与临时抽取仍保留给 diagnostics，但不会伪装成用户交付物。

## Monitor 用来判断过程是否正常

Monitor 将一次运行的模型、Agent、tools、tasks、token、费用和机器时间汇总起来。每次 LLM 调用完成后都会更新 token 统计；provider 提供时会分别记录 input、output、cache 和 reasoning token，仍在运行的单次调用则暂时没有最终用量。Checkpoint continuation 可能向 stream 重放旧消息，但其中的历史 usage 不会计入新 run。展开 **Token details by model** 可以按 `llm.yaml` 中的模型标签查看未缓存输入、缓存输入、缓存写入、输出、总 token 和已完成调用数。Overview 适合快速看状态和规模；Live 显示当前阶段、活动工具、Todo、subagent 与近期日志；Events 可以按 thread、run、agent、tool、category 和 channel 过滤；Raw 和 Details 用于排查更具体的问题。

当 Agent 看似停住时，先看 Live 中是否仍有远程 tool 或 subagent 在运行。当结果不完整时，查 Events 中的 tool error、document warning 或 multimodal 状态。当成本异常时，查看模型调用、token 和远程机器时间。Monitor 是诊断界面，不需要作为日常报告手工抄写。

当前界面没有历史 run 选择器，Overview 也可能汇总 workspace 与 lane 的当前或最近运行。精确追踪某次远程计算时，应同时核对 thread ID、run ID、artifact 和 receipt。

## 右栏用于任务上下文，产物使用大预览

Chat 右栏是一条 Task Context 信息流，不是内嵌文件查看器。Outputs 收录会话完成时明确报告的文件，包括工作区报告链接和中文输出文件列表，点击后进入统一大预览；Activity 显示当前运行和后台 specialist，之后依次是 Plan 与适用时的 Research progress，后者提供醒目的最新报告按钮。Research Graph 保留完整独立页面。Plan 跟随最新的 `write_todos` 状态：主回复结束不会把未完成事项或独立运行的后台专家自动标成完成。后台完成通知单独显示为可展开的通知，不会冒充用户发言。刷新后，页面从消息快照对应的游标继续接收推流。桌面端可以拖动右栏左边界调整宽度，也可以用顶部按钮收起或展开；窄屏上通过 Task Context 按钮打开抽屉，按钮会显示 active 或 attention 数量。

点击聊天里的图片、文件链接、artifact 卡片或需要完整查看的 activity，会在页面上方打开一个独立预览窗口。页面四周变暗，中间窗口使用接近完整 viewport 的空间继续复用图片、Markdown、表格、PDF、结构和文本 renderer。右上角叉号、Escape 或点击窗口外的遮罩均可关闭，键盘焦点随后回到原入口。这个窗口只负责查看产物；关闭它不会改变后台任务或 Research Graph 状态。

已完成的 specialist task 卡直接使用返回的科学 Markdown：卡片展示内容标题、核心判断和简短分节提纲；Details 中保留扩展提纲和技术引用。

一个自然的复查请求可以直接指向刚打开的文件：

```text
我正在看 notes/slab_audit.md。请结合对应结构重新检查第 3 个终止面，
解释报告中 CN=1 的判定阈值，并把该终止面的俯视和侧视图与第 1 个并排比较。
先分析，不要删除或覆盖任何候选。
```

## Skill Evolution 处理项目中反复出现的经验

登录模式和可信 no-login 模式下，只要 workspace mode 不是 `off`，每个有用户任务的 terminal run 都会进入一次 Skill Evolution 语义反思。登录部署使用已认证用户名和隔离的用户根目录；no-login 使用固定本地 `admin` actor 和直接本地 workspace 范围。模型先收到紧凑、确定性的 run index，再通过 host-bound 只读查询按需打开语义事件；完整轨迹不会复制进 prompt。它区分无需学习、执行 Agent 没有遵守现有 skill、等待更多真实使用的 `defer`、确认不应固化的 `ignore`，以及确实需要修改长期行为。系统不用正则、embedding、固定复现次数或 replay 评分替模型判断。一个明确的长期 correction 可以单独形成证据；多次措辞相似也不自动产生候选。最终 resolved owner 决定唯一 revision chain；普通新 evidence 从当前有效版本开始，rejected 草稿不会隐式成为父版本。Processing history 可分页查看每个 reflect 的变化、理由、原始和 resolved target、evidence handle 与终态；失败结果保持原状且不会自动重试，卡片只为用户明确选中的 job 或 failed item 提供授权重试。

Reviewer 对精确不可变 revision 的 `approve` 就是语义决策。Workspace 处于 `auto` 时，只要 target 已启用并使用自动更新，获批且可加载的 revision 会自动成为下一次 run 的有效版本；普通流程不再产生人工 review、canary 或 promotion 队列。`needs_revision` 会触发有上限的自动 proposer-reviewer 修订，每个尝试版本仍可追溯。只有真实的授权、安全选择或未解决的用户偏好才会暂停激活，并在原聊天中提出具体问题。自动激活和 `no_change` 也会在原聊天中给出简短、非阻塞通知。

需要精确后验控制时，在 **Skill Evolution** 页面点击 **Manage skills**。弹窗展示完整分页的有效 skill 清单、workspace 持久化模式（`off`、`observe`、`auto`）、启用状态、所选精确版本、`auto_head`、最新草稿、不可变版本、evidence 引用，以及自动、聊天和 GUI 产生的有序历史。手动选版本会固定为 Pinned；切回 Follow auto 会恢复自动选择。禁用只会让该 skill 不再进入下一次 run 的 staging，不会删除文件、revision 或历史。这些操作从后续 run 生效，不会热更新当前 run。

## 继续一项已经中断或隔天再做的工作

回到原 workspace 和原 thread，先让 Agent 重新读取关键产物，而不是只发"继续"。说明应保留哪些结果、上次停在哪里、是否禁止重复计算，以及这一次的目标：

```text
继续上次的 CO 吸附筛选。先读取 notes/co_sites.md、structures/co_candidates/ 和
calculations/mlff_screen/output/，核对已有候选、失败项和排序依据。

不要重新生成或提交已经完成的结构。请判断哪些候选值得进入 VASP，
列出原因和需要我确认的共同设置；本轮停在 VASP stage 审阅前。
```

如果上一次涉及远程错误，先读取 receipt 并判断旧作业状态，禁止把"重试"理解为重新提交。第 8 章给出了相应恢复 prompt，第 11 章收录更具体的诊断方法。

如果最新一轮是在可恢复的 LangGraph 步骤内失败，错误卡片会显示
**Continue from checkpoint**。它不会添加新的用户消息，而是以空输入续接同一个持久化
thread；已经写入 checkpoint 的步骤继续保留，只重新执行失败的图尾部。它不会从一段
中断的 LLM token 流中间接着生成。如果卡片仍显示 **Review and try again**，说明该失败
没有可安全续接的图状态，需要从 composer 调整后再提交。Thread 一旦已经前进，旧失败
卡片的续跑操作就会失效。

## 当前界面的重要限制

WebUI 目前不能删除或分支 thread，也不能回放任意历史 checkpoint 或从历史 run 选择器恢复某个节点；它只能原生续接最新且可恢复的失败图尾部。Research Graph 可以跨 thread 管理科学分支，但它不是 thread 历史或 rollback 控件。Files 上传同名文件会覆盖，删除不会进入回收站。审批中断必须用消息内卡片恢复。Stop 不取消远程 job。Skill Evolution 的变更从下一次 run 生效，不会热更新当前 run。

这些限制不会阻止正常研究流程，但会影响如何备份、续跑和停止任务。重要文件使用版本化名称或 Git/外部备份；远程任务依靠 receipt；相互独立的目标或不兼容的项目范围使用不同 thread。这样可以避免把 UI 中缺少的操作误认为 Agent 会自动补齐。

图谱节点详情展示判断的适用范围和理由，以及科学修订关系。Research decisions and open premises 展示关键停止决定、独立复查、补救实验与实际结果、开放前提和恢复条件。运行中的 Plan 属于当前对话，长期科研认识保存在 Graph。
