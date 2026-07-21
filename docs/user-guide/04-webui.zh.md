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

## Auto 与 Review 代表不同的协作方式

Auto 允许 Agent 在当前权限范围内连续工作，适合读取、分析和已建立信任的项目流程。Review 会在 `write_file`、`edit_file`、`remote_submission` 和 `remote_submission_batch` 前暂停，消息中出现审批卡。

首次处理重要项目、可能覆盖文件或准备真实计算时，建议使用 Review。审批卡提供四种处理方式：

- Approve 按当前 action 执行。
- Reject 拒绝这次 action，可以附上原因。
- Respond 给 Agent 补充说明，让它根据反馈重新处理。
- Edit action 直接修改 action JSON，适合熟悉 tool schema 的高级用户。

Review 不是所有行为的总开关。读取、搜索和部分分析仍可自动进行。它的价值是把受保护的文件编辑和远程提交交给用户确认。审批应在消息卡中完成；不要另发一条普通消息冒充审批结果。

这里的"写文件"特指当前受保护的 `write_file` 与 `edit_file`。`supercell`、`build_slab` 等领域 tool 会在自己的调用中直接生成已声明的输出，当前不会仅因为产生文件就自动弹卡。对这类操作，应在 prompt 中给出目标目录，在 tool 卡中核对输入与输出路径，并在 Files 中审查产物。Review 是针对明确调用的保护层，不是所有 workspace 变更的事务锁。

## 运行中可以 Steer，但不必把每个想法都打断进去

空闲时按钮为 Send。Agent 正在运行时，发送纯文本会变成 Steer。Steer 不会强行终止正在执行的 tool，而是在下一个安全边界成为后续指令。它适合补充刚发现的约束，例如"不要覆盖原结构""只分析前 20 ps""把两个终止面都保留"。

如果新要求彻底改变任务，等待当前安全停下后开一个新 thread 通常更清楚。运行期间不能附加新文件，所以需要新增输入时，可以先 Stop 或等待结束，再上传并继续。

Stop 会请求本地 Agent 在流事件边界停止；连续请求会升级为 emergency cancel。它不会自动取消已经提交到远程调度器的作业。远程作业必须根据 receipt 和集群状态单独处理。

## Files 是交付物所在的地方

Files 视图提供 Browse、Preview 和 Uploads。它可以预览文本、Markdown、JSON、图片、PDF、CSV/TSV、常见晶体与分子结构、轨迹以及部分 OUTCAR 振动内容。结构和轨迹可以通过 JSmol 查看，VESTA 生成的标准视图也可以作为图片 artifact 打开。

Agent 报告完成后，至少检查核心交付物是否真的存在，文件名和目录是否符合约定。结构任务看候选与审计，计算任务看 stage、status、stdout/stderr 和分析，文献任务看候选表、证据表与引用库，写作任务看可编辑源文件而不只看编译 PDF。

Files 上传同名文件会覆盖，删除目录是永久递归操作。重要原始数据在 workspace 外应有备份。`metadata/` 虽然会出现在文件树中，却不是普通项目文件区，不要随意改名、移动或删除。

## Monitor 用来判断过程是否正常

Monitor 将一次运行的模型、Agent、tools、tasks、token、费用和机器时间汇总起来。Overview 适合快速看状态和规模；Live 显示当前阶段、活动工具、Todo、subagent 与近期日志；Events 可以按 thread、run、agent、tool、category 和 channel 过滤；Raw 和 Details 用于排查更具体的问题。

当 Agent 看似停住时，先看 Live 中是否仍有远程 tool 或 subagent 在运行。当结果不完整时，查 Events 中的 tool error、document warning 或 multimodal 状态。当成本异常时，查看模型调用、token 和远程机器时间。Monitor 是诊断界面，不需要作为日常报告手工抄写。

当前界面没有历史 run 选择器，Overview 也可能汇总 workspace 与 lane 的当前或最近运行。精确追踪某次远程计算时，应同时核对 thread ID、run ID、artifact 和 receipt。

## 右侧 inspector 适合边对话边审阅

点击 artifact 或文件后，右侧 inspector 会打开标签页。你可以保持 Chat 可见，同时对照结构、报告、表格或日志继续提问。Todo 标签显示当前 turn 中 `write_todos` 产生的只读计划，适合了解 Agent 还在做什么，但它不是用户需要手工维护的项目管理器。

一个自然的复查请求可以直接指向刚打开的文件：

```text
我正在看 notes/slab_audit.md。请结合对应结构重新检查第 3 个终止面，
解释报告中 CN=1 的判定阈值，并把该终止面的俯视和侧视图与第 1 个并排比较。
先分析，不要删除或覆盖任何候选。
```

## Skill Evolution 处理项目中反复出现的经验

登录模式下，终态 run 可以产生 workspace 范围的改进候选。例如同一项目反复要求固定的目录命名、单位、结构审计或报告格式，系统可以提出 memory 或 skill 候选。默认 `observe` 模式不会自动生效，用户在 Skill Evolution 视图中审阅后才能 Promote 或 Reject。

这里适合保存稳定、项目特定且经过验证的做法，不适合把一次网络错误、临时文件名或未经证实的科学猜测固化成规则。Promote 后从下一次 run 生效；目标内容已变化时会进入 conflict，而不是直接覆盖。造成退化的已提升内容可以在目标仍匹配时 Rollback。

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

WebUI 目前不能重命名、删除、分支或 retry thread，也不能从历史 run 选择器恢复某个任意节点。Files 上传同名文件会覆盖，删除不会进入回收站。审批中断必须用消息内卡片恢复。Stop 不取消远程 job。Skill Evolution 只在登录模式显示，并从下一次 run 生效。

这些限制不会阻止正常研究流程，但会影响如何备份、续跑和停止任务。重要文件使用版本化名称或 Git/外部备份；远程任务依靠 receipt；需要不同假设时新建 thread。这样可以避免把 UI 中缺少的操作误认为 Agent 会自动补齐。
