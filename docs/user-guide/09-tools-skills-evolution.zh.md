# 9. 工具、技能与自进化

[上一章](08-remote-execution.zh.md) | [目录](README.zh.md) | [下一章](10-deployment-operations.zh.md)

工具决定 agent 能执行什么动作，skill 决定它应按什么流程执行，自进化决定一个 workspace 能否提出并启用项目级改进。三者相互配合，但不互相替代。

## 9.1 基础工具

DeepAgents 运行时提供文件和任务基础工具，例如：

```text
write_todos
ls
read_file
write_file
edit_file
glob
grep
execute
read_document
```

`read_document` 对 PDF、DOCX、XLSX 和 PPTX 做有界解析。`execute` 用于项目内准备、检查、轻量脚本、依赖探测和后处理，不是绕过 DPDispatcher 运行 VASP、CP2K、LAMMPS、ORCA、xTB、CREST 或受管 MLFF 的入口。

## 9.2 领域工具

CatMaster 的注册工具大致分为：

- 结构构建、转换、表面、吸附和几何分析。
- VASP、CP2K、LAMMPS、ORCA、xTB/CREST 输入准备和结果分析。
- 轨迹、振动、热力学、能带、DOS、弹性和数据分析。
- ML 数据、主动学习、符号回归、MACE train/eval 和 MLFF task schema。
- Materials Project、文献检索、本地语料、网页和引用管理。
- 图件、文档读取、Markdown PDF、TeX 编译和图像生成。
- DPDispatcher task catalog、单 stage 提交、batch 提交和 receipt。

工具不会全部出现在每个 agent 上。Runtime allowlist 和 task `audiences` 决定 specialist/worker 的实际界面。提示词中无需强制具体工具名，除非你是在核对 task schema 或复现一个已知调用。

## 9.3 Tool schema 是用户界面

Agent 根据工具描述和 JSON schema 构造参数。可选字段通常应省略或传空对象/数组，而不是猜测 `null`。远程 task 的参数尤其应先用 catalog 查询，因为 backend 启用状态、模型、默认值和 override key 由部署决定。

如果 agent 反复把关键参数传成空值：

1. 在 Chat 展开工具卡，核对最终参数。
2. 查询 catalog 的 full spec。
3. 在 Review 卡中修正 action，或拒绝并让 agent 重建。
4. 把可复用的正确合同写进项目 skill，而不是在每条 prompt 重复粘贴。

## 9.4 Skill 是什么

Skill 是带 `SKILL.md` 的任务 SOP，可包含参考文件、脚本、模板和验收规则。当前主要 skill 组覆盖：

| 组 | 典型范围 |
|---|---|
| Materials | bulk、slab、termination、adsorption、defect、VASP、CP2K、NEB、phonon、MLFF |
| Dynamics | CP2K AIMD、LAMMPS、MLFF MD、restart、trajectory analysis |
| ML | dataset、MACE training/evaluation、active learning |
| ORCA/xTB | conformer、xTB、CREST、ORCA opt/freq/TS/IRC/TDDFT/NMR |
| Research | 研究规划、状态、证据和跨 specialist 协调 |
| Literature | 检索、阅读、语料、引用和报告 |
| Writing | 论文、回复、图件、LaTeX、PDF 和语言润色 |
| Execution | remote stage layouts 和 DPDispatcher receipt 恢复 |
| Writing quality | 避免模板化和 AI 腔，同时保持事实 |

目录存在但没有有效 `SKILL.md` 的占位项不算可用能力。

## 9.5 Skill 的加载位置

每次 run 会把适用的内置 skills staged 到：

```text
files/.deepagents/skills/<group>/
```

Workspace 的项目级覆盖位于 self-development 区，并可覆盖同名内置 skill。Staged 内容是运行时快照，不应在任务中随意修改后误以为仓库 skill 已更新。

Skill 只提供方法，不授予权限。例如远程 stage layout skill 能说明 VASP 目录，但没有 `remote_submission` allowlist 的 agent 仍不能提交。

## 9.6 项目脚本

Agent 为缺失的轻量操作编写脚本时，应放在 `scripts/`，并至少记录：

```text
创建日期
创建 agent 或来源
目的和科学原理
输入与输出
单位和关键参数
依赖
失败模式
最小使用示例
```

一次性 shell 片段适合探索；会重复使用、影响科学结果或被后续 thread 调用的逻辑应固化成脚本并保留审计。

## 9.7 顺序执行和共享 workspace

Specialist 和 worker 共用同一个 workspace。运行时默认每次只做一个委派并等待返回，不允许多个 agent 同时写相同目录。某个 provider 支持 `parallel_tool_calls` 不代表项目写入可以安全并行。

真正独立的远程 stage 可以由 `remote_submission_batch` 在一个受管调用中批量提交。不要用多个 agent 并行提交来代替 batch 合同。

## 9.8 Review 模式的边界

Review 只在 `write_file`、`edit_file`、`remote_submission` 和 `remote_submission_batch` 前中断。它不能替代：

- 项目空间外部备份。
- SSH 和队列访问控制。
- 远程 task/resource audience。
- 对科学参数和成本的人工检查。
- 对已有远程作业的调度器管理。

需要更严格的部署，应在网络、账号、文件权限、集群队列和 secret manager 层增加控制。

## 9.9 自进化模式

环境变量：

```bash
export CATMASTER_SELF_EVOLUTION_MODE=observe
```

可选值：

| 模式 | 行为 |
|---|---|
| `off` | 不生成候选 |
| `observe` | 默认；生成和审查候选，等待人工 Promote 或 Reject |
| `auto` | 通过 gate 和 reviewer 后自动 Promote |

`--no-login` 模式完全关闭 WebUI 自进化。`auto` 会改变后续 run 的行为，只应在管理员已验证候选质量、回滚和监控流程后启用。

## 9.10 候选生命周期

```text
终态 run trace
  -> proposer: ignore / memory / skill
  -> 静态 gate
  -> 独立 reviewer
  -> observe: 人工决定
  -> auto: 自动提升
  -> 下一次 run 激活
```

候选只能提出完整 memory 文件或一个完整 skill bundle，不能直接改仓库内置 skill。静态 gate 检查路径、大小、符号链接、frontmatter、章节、引用和 Python/Shell 语法。Reviewer 为只读角色。

Promotion 使用内容哈希、目标哈希和锁。如果候选生成后目标已经改变，操作进入 conflict，而不是覆盖新内容。Rollback 同样要求目标仍与被提升版本一致。

## 9.11 在 UI 中处理候选

Skill Evolution 面板中：

1. 先读候选来源 run 和提出原因。
2. 比较目标、旧内容和候选内容。
3. 确认它是项目特定规则，不应污染其他 workspace。
4. 检查是否把一次偶然失败过度概括成永久规则。
5. `Promote` 后新建一次最小验证 run。
6. 无益候选 `Reject`，造成退化时 `Rollback`。

候选在同一 workspace 的 thread 间共享，从下一次 run 生效。Promote 不会修改当前正在运行的上下文。

## 9.12 适合与不适合固化的内容

适合：

- 本项目固定的目录、命名、单位和交付合同。
- 经多次验证的结构检查或分析步骤。
- 特定远程 task 的稳定 stage 准备流程。
- 用户明确要求长期保持的写作或报告规范。

不适合：

- 单次网络错误、临时文件名或一个失败样本。
- SSH key、token、账号和私有主机信息。
- 未经验证的科学默认值。
- 绕过 worker、Review 或远程执行约束的方法。
- 只对当前 turn 有意义的指令。
