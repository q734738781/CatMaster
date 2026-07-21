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

Agent 可以帮助建立 manifest 或 README，记录路径、日期、来源、关键参数和版本。但不要让它在项目刚开始时生成大量空目录和模板文件。只有当目录中真的出现需要管理的内容时再补记录，文档会更容易保持真实。

对于结构修改，可以保留原文件并使用能表达变化的名称，例如 `ceo2_111_t0_raw.vasp`、`ceo2_111_t0_fixed.vasp` 和 `ceo2_111_t0_pd_site03.vasp`。更复杂的批量候选应配一张 CSV 或 Markdown 清单，而不是把全部信息塞进文件名。

## Agent 写脚本时怎样保持可复现

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

Agent 写出的文件可以注册为 artifact，并在 Chat 中显示可点击卡片。右侧 inspector 根据文件类型选择文本、表格、图片、PDF、结构或轨迹预览。Artifact 不是文件副本，它指向 workspace 中的实际产物，因此移动或删除文件会影响后续打开。

工具返回内容很长时，Chat 只显示预览，完整结果会写到 `files/_tool_outputs/`。最终报告应引用这些文件或更清楚的整理结果，不应把一个被截断的 tool preview 当作全部证据。

远程 receipt 也是一种重要 artifact。它连接本地 stage、远程作业和回传状态。计算项目备份时，不要把 `files/.deepagents/` 一概视作可删除缓存，其中可能包含仍需恢复的 receipts。

## Project memory 保存稳定约定，不保存流水账

Agent 可以使用 workspace 范围的长期 memory，保存会影响未来任务的稳定信息。例如项目固定使用的能量零点、结构命名、单位、不可丢失的 Selective Dynamics 规则，或用户明确要求长期遵守的写作偏好。

一次失败的 SSH 连接、临时文件路径、当前任务进度和未经证实的机理猜测不应进入长期 memory。它们应留在 thread、日志或阶段报告中。Memory 越像一份简洁的项目约定，后续 Agent 越容易正确使用；把所有聊天内容都塞进去反而会污染判断。

## Skill Evolution 把反复验证的方法变成项目能力

Skill 比 memory 更适合保存一套可重复流程。假设一个项目多次验证了特定阶梯 CeO2 模型的终止面检查、原子命名、固定层和报告格式，系统可以提出 workspace skill 候选。候选应包含完整 `SKILL.md`，必要时还可包含参考文件和脚本。

在默认 `observe` 模式中，候选先经过静态检查和独立 reviewer，再出现在 Skill Evolution 页面。用户可以查看它来自哪次 run、准备覆盖什么内容以及为什么建议固化，然后选择 Promote 或 Reject。Promote 从下一次 run 生效，不会改变正在运行的 Agent。

适合提升为项目 skill 的内容包括：

- 项目长期使用的结构生成、筛选和 QC 方法。
- 稳定的目录、命名、单位和交付合同。
- 特定远程 task 的 stage 准备与结果验收方式。
- 经多次使用证明有效的写作、图件或报告流程。

不适合提升的内容包括一次网络错误、暂时可用的文件名、某个单独样本上的偶然参数，以及未经独立验证的科学结论。Skill 只改变 Agent 的工作方法，不会给它新增 tool 权限，也不会自动启用缺失的 remote task。

```text
回顾这个 workspace 中最近三次 slab 任务及其审计报告。
找出真正重复且已经验证的项目约定，区分稳定规则与只适用于某个结构的临时选择。

如果确有值得复用的流程，请提出一个 project skill 候选，说明适用条件、输入、
检查步骤、输出和不能推广的边界。不要把一次 CN 阈值或某个原子索引写成通用规则。
候选只进入 observe 审阅，不要自动 Promote。
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
