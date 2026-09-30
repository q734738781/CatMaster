# 9. 项目文件、连续工作与可复用经验

[上一章](08-remote-execution.zh.md) | [目录](README.zh.md) | [下一章](10-deployment-operations.zh.md)

CatMaster 的价值不只在一次回答，而在于同一项目能够持续积累结构、数据、脚本、文献、文稿和可复核的决策。Workspace 是这项工作的载体。你和 Agent 共同维护 `files/`，系统用 `metadata/` 保存 thread、checkpoint、运行观测和远程状态。

## 目录应服务研究，而不是服务代码模块

不必按 tool 名或 Agent 名建立目录。更自然的做法是按研究对象和交付物组织。例如一个表面催化项目可以这样开始：

```text
files/
  literature/
  structures/
    bulk/
    slabs/
    adsorption/
  calculations/
    bulk_reference/
    slab_screen/
    adsorption/
  data/
  scripts/
  notes/
  figures/
  writing/
```

Materials worker、Dynamics worker 和 Writing Agent 都可以读取这些目录，不需要把同一份结构复制到每个 Agent 专属位置。项目已经有成熟布局时，在第一个 prompt 中告诉 Agent 延续原有约定即可。

```text
这是一个已有项目。请先阅读 files 根目录、notes/project_conventions.md 和最近相关结果，
理解现有目录、命名、单位和版本习惯。不要为了符合 CatMaster 示例重新整理整个项目。

本轮只给出你理解到的项目结构、可信输入、派生文件和需要澄清的地方，
并建议后续产物应放在哪里；不要移动、删除或覆盖文件。
```

## 原始输入、派生结果和最终交付物要分得开

数据库下载、仪器数据、用户上传的结构和投稿源稿属于原始输入，应该保留原件及来源。标准化结构、过滤后的数据、计算 stage 和图片是派生结果，应能追溯到输入与生成方法。论文表格、最终图和报告是交付物，应指向其源数据与脚本。

Manifest、README 或审计清单不是默认产物；只有用户要求、现有接口要求，或其中确有下游决策所需信息时才创建。OCR 文本、格式转换、试验性片段、一次性脚本、日志和中间表统一放在 `tmp/`（文件工具中显示为 `/tmp/`），下游默认忽略；需要长期使用的结果再移入主题目录。

对于结构修改，可以保留原文件并使用能表达变化的名称，例如 `ceo2_111_t0_raw.vasp`、`ceo2_111_t0_fixed.vasp` 和 `ceo2_111_t0_pd_site03.vasp`。更复杂的批量候选应配一张 CSV 或 Markdown 清单，而不是把全部信息塞进文件名。

## Agent 写脚本时怎样保持可复现

已有 skill 工具脚本可直接执行，无需先复制或打包。文件工具读取
`/.deepagents/skills/<相对路径>`，shell 使用
`python "$CATMASTER_SKILLS_ROOT/<相对路径>"`；这个变量指向当前运行实际采用的 skill
快照。输入输出使用工作区路径，只有修改实现时才另存项目副本。诊断类脚本先返回状态、
关键异常和报告路径，完整原子距离表、逐原子数组等留在文件中按需查询。

现有 tools 能覆盖许多常见动作，但研究项目总会出现特殊分析。边界清楚的轻量工作可以由 worker 使用 Python 或 shell 完成。若逻辑会重复使用、影响科学结论或需要处理大量结构，Agent 应把它保存到 `scripts/`，而不是把整个过程藏在一次临时命令中。

可复用脚本应说明创建日期、相关 Agent、实现思路、用途、输入输出、单位、关键参数和失败方式。结果报告要记录实际运行命令或配置。这样下一次 thread 可以直接复查脚本，而不用根据聊天摘要重新发明分析。

```text
为 trajectories/run1.traj 编写一个可复用的 Pd 团簇连通性分析脚本。
脚本放在 scripts/，输入路径、Pd-Pd 截断、周期边界和抽帧间隔都用明确参数，
不要写死当前文件。输出逐帧连通分量、最大团簇大小和代表帧清单。

先用当前轨迹做最小验证，再把命令、阈值依据、结果路径和已知限制写入 notes/。
不要只在一次 execute 调用中完成后丢掉代码。
```

## Artifact 让对话与文件互相连接

Agent 写出的文件可以注册为 artifact，并在 Chat 中显示可点击卡片。点击后，大预览窗口根据文件类型选择文本、表格、图片、PDF、结构或轨迹 renderer。Artifact 不是文件副本，它指向 workspace 中的实际产物，因此移动或删除文件会影响后续打开。

工具返回内容很长时，Chat 只显示预览，完整结果会写到 `files/_tool_outputs/`。最终报告应引用这些文件或更清楚的整理结果，不应把一个被截断的 tool preview 当作全部证据。

远程 receipt 也是一种重要 artifact。它连接本地 stage、远程作业和回传状态。计算项目备份时，不要把 `files/.deepagents/` 一概视作可删除缓存，其中可能包含仍需恢复的 receipts。

## Project memory 保存稳定约定，不保存流水账

Agent 可以使用 workspace 范围的长期 memory，保存会影响未来任务的稳定信息。例如项目固定使用的能量零点、结构命名、单位、不可丢失的 Selective Dynamics 规则，或用户明确要求长期遵守的写作偏好。

一次失败的 SSH 连接、临时文件路径、当前任务进度和未经证实的机理猜测不应进入长期 memory。它们应留在 thread、日志或阶段报告中。Memory 越像一份简洁的项目约定，后续 Agent 越容易正确使用；把所有聊天内容都塞进去反而会污染判断。

## Skill Evolution 把反复验证的方法变成项目能力

本地后台运行结束后，学习任务读取该轮绑定的用户要求、最终答复及模型和工具记录。
即使你已经开始下一轮，或从旧答复选择 Learn，学习证据仍指向选中的那一轮。
可恢复的中断或错误会等待续跑完成后再自动学习。学习结果通过原线程的聊天通知展示，
不会因此启动新的科研任务；暂缓、忽略或失败的详情可在 Processing history 查看。

Skill 比 memory 更适合保存一套可重复流程。假设一个项目多次验证了特定阶梯 CeO2 模型的终止面检查、原子命名、固定层和报告格式，系统可以提出 workspace skill 候选。候选应包含完整 `SKILL.md`，必要时还可包含参考文件和脚本。

系统不会把每个完成的 run 都变成 skill，但会把每个 terminal 用户 episode 交给一次语义反思。一个 episode 如果中断后跨多个物理 run 恢复，episode identity 保持不变，只有最终 terminal run 成为反思 anchor；显式 Learn 则保持用户实际选中的 run。反思模型先收到确定性 run index，再通过 host-bound 只读查询读取完整模型响应、tool input、模型实际可见的 tool result、任务边界和最终 outcome；provider envelope、加密 transport、streaming delta 和重复 callback 不进入这层语义视图。它区分 `no_change`、execution lapse 和确实需要修改长期行为的证据。没有值得固化的 SOP 改进时，job 明确返回 `no_change`，并显示 reflector 给出的理由；execution lapse 只记入 job outcome，不写 observation，也不生成 candidate。一次反思可以返回多个彼此独立、有证据的 finding，也可以有多个 finding 初始指向同一 owner；各项独立 proposal/review，某一项失败不会丢掉其他项。系统不用关键词、正则、embedding 或固定复现次数替模型做判断。用户明确要求长期遵守的 correction 可以单独形成证据；多次措辞相似也不自动成为 skill。模型负责判断产品、schema、科学或一次性问题是否蕴含可复用的 SOP 改进，宿主不会按问题类别否决；已有 skill 能承接时优先修订已有 owner。

每个可学习 finding 提供一个初始 target anchor，并引用查询结果中的完整事件 handle。Proposer 可以检查完整的已挂载 skill tree 和授权 workspace history，然后保留或纠正 owner；最终 resolved owner 决定唯一 candidate ID、锁和 revision chain。轻量 history 默认分页，包含 observation、candidate revision、review、job、同一精确 skill version 的真实 read/helper/outcome 对照，以及授权 `run_ref`；worker 可以按需打开事件正文或原始记录；查询返回的 handle 可直接交给读取工具，默认读取与该引用对应的正文。新一轮默认只装配当前 open delta，已经被 revision 吸收的历史不会反复写进 evidence 或在启动前全部解析。Proposer 和独立 reviewer 收到紧凑的 claim/handle index，可以用只读 SQL/JSON1 或显式 continuation 重新打开授权范围内的 event field；完整轨迹不会复制进 prompt 或 candidate 目录。普通查询直接返回完整行；大结果返回完整 JSON 文件的 `result_path`，可以用 `read_file`、`grep` 继续检查。长事件字段也可通过 `next_offset` 分段读取。查询描述包含可用列和两个示例；错误返回具体原因和恢复方式，不重复整份表结构。这里没有相似度聚类或“至少三个 episode”门槛：证据可能形成 candidate，也可用 `defer` 保持 open 等待后续真实使用，或用 `ignore` 记录并消费已确认不应固化的 finding。

反思、提案和审查都通过各自声明的结果工具提交决定，普通回复中的 JSON 不会被当作已提交结果。说明写在结果的文本字段中；skill 或 memory 的修改由 proposer 用普通文件工具写入候选文件，不需要 patch 字段。若模型只用普通回复结束，系统会保留同一会话、已查证据和文件修改，提醒一次补交结果工具；仍未提交时保留完整日志并将该项记为失败。

每个 candidate revision 都是不可变版本。CatMaster 不会为候选生成测试题，也不会额外启动多组对话比较版本。宿主只执行真实的路径、授权、不可变 revision 和 selection pointer 边界；candidate 能否加载由 active runtime 所用的同一套 DeepAgents skill middleware 检查。Loader 报错时，proposer 会收到当前 candidate 和精确诊断，继续修改后再检查。标题、章节顺序、文风、可选 metadata、文件数量、reference、代码布局和 `allowed-tools` 都不会成为宿主质量 gate。独立 reviewer 读取完整 evidence 和精确 diff，负责 SOP 的语义审核：`approve` 按 workspace policy 推进；`auto` 下启用且 Follow auto 的 target 会从下一次 run 自动使用，不再要求用户二次确认。`needs_revision` 基于同一 evidence 和 exact draft 启动有上限的自动修订，`reject` 则终止该 branch。普通新 evidence 从当前有效版本开始；rejected draft 仍可查，但不会被静默复制为下一版父内容。

Candidate 与 Processing history 是诊断视图，不是审批 inbox。它们展示行为变化、原始和 resolved target、处理理由、evidence handle、content parent、reviewer 的 counterexample 和 concerns，并为每个 job 给出明确终态；即使没有 candidate，也会区分 `defer`、`ignore`、`no_change`、existing-guidance execution lapse 或具体错误，同时保留每个独立 finding。完整 diff 只从当前精确 revision 的 Technical details 按需读取；candidate response 还给出该 revision 的完整 proposal、review、prior review、validation 以及 proposer/reviewer 可见响应证据的认证分页引用。Raw semantic event payload 可通过授权的 raw trajectory view 打开，transport 内部信息仍留在受控 Developer Diagnostics。Candidate、observation 和 Processing history 都是 newest first，并可继续分页加载。

Workspace mode 决定 review 后的动作。没有保存 workspace 设置且部署未显式覆盖时，默认使用 `auto`。`off` 停止新的 post-run evolution，但不禁用已经选中的 skill；`observe` 记录获批 revision 的 `auto_head`，但保持 dormant；`auto` 会在 target 已启用且采用 Follow auto 时，为下一次 run 选择获批的精确 revision。它不会重新启用被禁用的 target，也不会替换 Pinned 版本。真正未解决的授权、安全选择或主观偏好会回到原聊天提问；回答以精确 user-message 引用记录，并恢复 held revision。自动激活和 `no_change` 会产生简短、非阻塞聊天通知。失败的 evolution job 不会自动重试；认证用户可以只重试一个精确失败 item，已完成项保持不变。

Skill Evolution 页面中的 **Manage skills** 按钮会打开可选的精确控制弹窗。它通过分页列出完整有效集合，包括 repository base、不可变 workspace revision、所选版本、reviewer 批准的 `auto_head`、最新草稿、启用状态，以及 Follow auto 或 Pinned 策略。选择一个合格历史版本会固定它，也就是精确 rollback；切回 Follow auto 会恢复自动选择。Rejected、未完成、无效、不可读或尚未解决边界的 revision 仍可查看，但不能选择。每次自动、聊天和 GUI 转换都会记录前后状态与来源。所有变化从后续 run 生效。

适合提升为项目 skill 的内容包括：

- 项目长期使用的结构生成、筛选和 QC 方法。
- 稳定的目录、命名、单位和交付合同。
- 特定远程 task 的 stage 准备与结果验收方式。
- 经多次使用证明有效的写作、图件或报告流程。

不适合提升的内容包括一次网络错误、暂时可用的文件名、某个单独样本上的偶然参数、没有改变任务决策的 checksum，以及未经独立验证的科学结论。Skill 只改变 Agent 的工作方法，不会给它新增 tool 权限，也不会自动启用缺失的 remote task。

```text
回顾这个 workspace 中最近三次 slab 任务及其审计报告。
找出真正重复且已经验证的项目约定，区分稳定规则与只适用于某个结构的临时选择。

如果确有值得复用的流程，先根据完整 episode 与结果判断根因，
并优先修订已有 owner skill。写清适用与不适用范围，以及预期改变的具体决策。
不要把一次 CN 阈值、某个原子索引或顺手做的 checksum 写成通用规则。
让独立 reviewer 判断 revision；在 `auto` 下，获批变化会从下一次 run 生效，除非 target 被固定、禁用，或确实需要我回答。
```

## 隔天继续时先恢复事实，再恢复计划

回到旧 thread 后，Agent 能使用 checkpoint，但项目文件仍是权威现状。最好让它先重新读取关键产物，确认哪些文件存在、哪些计算真正完成、哪些判断仍待用户决定。这样可以发现对话记忆与磁盘状态之间的差异。

```text
继续这个项目。先读取 notes/progress.md、calculations/summary.csv、最近的 receipts
和相关 stage，不要直接沿用聊天中的"已完成"说法。

请根据当前文件重新列出可信完成项、失败或不完整项、仍在远程运行的任务和需要我决定的事项。
保留所有成功结果，禁止重复计算；只有恢复事实后再建议下一阶段。
```

如果新目标与原 thread 明显不同，例如从表面计算转为论文写作，可以新建 Writing thread，并把 `notes/result_contract.md`、结果表、图和引用库作为交接材料。这样新 Agent 得到的是清楚的证据包，而不是一段很长的聊天转述。

## 备份时要保留什么

完整恢复一个 workspace 需要同时备份 `files/` 和 `metadata/`。只保存 `files/` 可以保住结构和结果，却会失去 thread checkpoint、审批状态、运行观测和部分 artifact 索引。启用登录的部署还要备份项目根目录下的 `.webui_auth/auth.sqlite`。

备份最好在 WebUI 停止或没有写入中的 run 时进行。大规模轨迹和计算结果可以采用站点自己的增量备份策略，但 receipts、manifest、报告和关键配置应与数据一起保留。升级、部署和权限设置见下一章。
