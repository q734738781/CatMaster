# 5. Agent 与模块功能

[上一章](04-webui.zh.md) | [目录](README.zh.md) | [下一章](06-computational-workflows.zh.md)

五个 Entry 是用户可见的顶层工作模式，不是同一个 agent 的五个提示词皮肤。每个入口绑定不同模型角色、工具、skill 和委派拓扑。选错入口通常不会让系统完全失效，但会增加无关规划、缺少领域工具或越过职责边界的风险。

## 5.1 快速选择

| 你的目标 | 入口 | 原因 |
|---|---|---|
| 一个明确结构、计算、轨迹或 ML 任务 | Experiment | 直接协调四个计算 worker |
| 文献、计算和写作交织的开放研究问题 | Research | 能按阶段委派其他四类 specialist |
| 基于已有证据起草或修改文稿 | Writing | 写作 worker、润色、图件和编译链 |
| 对一份固定 PDF 做正式审稿 | Peer Review | 多 reviewer 报告和 editor synthesis |
| 系统检索、全文阅读、证据表和引用整理 | Literature Review | 搜索、受控浏览器、本地语料和引用定稿 |

单一、边界明确的任务选最窄入口。Research 不是默认的“更强模式”，它更适合真正需要跨模块决策的目标。

## 5.2 Research

Research 是研究协调器。它可以按需要委派：

```text
Research
  -> Experiment Specialist
  -> Writing Specialist
  -> Peer Review Specialist
  -> LitReview Agent
```

适合：

- 从开放科学问题建立计划和证据缺口。
- 先做文献审查，再决定是否需要计算。
- 把已有计算结果和文献证据整合为报告或论文段落。
- 在多阶段工作中维护决策、假设、产物和下一步。

不适合：

- 已知输入和输出的一次结构转换。
- 只找论文或只润色一段文字。
- 用宽泛“研究一下”掩盖未定义的计算体系。

Research 自身是决策和整合层，不直接拥有全部计算工具。属性数值查询默认先查 workspace 和文献。没有证据时，它应说明缺口并询问是否进行计算，不能未经允许启动 DFT 或 ORCA 来补一个数值。

同一 workspace 中的委派按顺序进行。系统一次只委派一个 specialist 或 worker，等待结果写回后再决定下一步，避免多个 agent 并行改写相同文件。

## 5.3 Experiment

Experiment 负责计算研究的协调和质量控制。协调器可以查看可用远程任务，也能直接使用 Materials Project 检索和下载结构；具体领域工作优先委派给四个 worker。

### Materials worker

主要能力：

- Bulk、晶胞、超胞和结构标准化。
- 表面切割、终止面、台阶、缺陷、掺杂和吸附位。
- VASP 与 CP2K 输入准备、批量 stage、NEB 和 dimer 路径。
- 声子、弹性、能带、DOS、热力学和 k-path 辅助。
- MACE、UMA、MatterSim、ORB 等 MLFF 的 SP、relax 和路径任务。
- 结构、配位、约束、轨迹和输出文件审查。

### Dynamics worker

主要能力：

- CP2K AIMD 的准备、restart 和分析。
- LAMMPS 最小化、MD、restart 和势文件布局。
- MLFF MD 的准备与执行。
- 轨迹健康、温度、能量、漂移、扩散和结构演化分析。

### ML worker

主要能力：

- 训练和验证数据集整理。
- 主动学习候选管理。
- MACE 训练、微调、评估和 benchmark。
- 专用工具未覆盖时，编写项目内可复用的轻量 ML 脚本。

### ORCA/xTB worker

主要能力：

- 从 SMILES 或结构生成分子和构象。
- xTB 优化、能量、溶剂和短时 MD。
- CREST 构象搜索。
- ORCA 几何优化、频率、热化学、扫描、TS、IRC、TDDFT 和 NMR。
- 构象集合与分子 MLFF 预筛选。

这些 worker 准备和检查 stage，但注册科学引擎使用受管远程执行。若机器、资源或任务卡未配置，正确行为是报告缺失，而不是在 control plane 上静默运行外部程序。

## 5.4 Writing

Writing 处理已有证据的文稿生产：

```text
Writing Specialist
  -> writing_worker_agent
  -> writing_polisher_agent
```

`writing_worker_agent` 负责一个有边界的章节、集成任务、图件、TeX 编译或 Markdown PDF。`writing_polisher_agent` 做保守语言润色，不应改变数值、引用、证据范围、结论强度或技术结构。

适合：

- 从 notes、结果表和图件起草摘要、方法、结果或讨论。
- 修改已有 Markdown、LaTeX 或 Word 提取文本。
- 统一术语、修复段落逻辑和语言问题。
- 生成科研图件，编译 PDF，并基于 PDF 复查版面。

Writing 不负责凭空补实验结果或启动计算。文献证据不足时，应转给 Literature Review 或 Research，而不是生成看似合理的引用。

## 5.5 Peer Review

Peer Review 接收一份 canonical PDF，按 `peer_review_models` 列表生成独立 reviewer report，再由 editor 层综合决定。一个模型标签对应一份 reviewer 意见。

典型产物：

- 各 reviewer 的完整意见。
- 主要问题、次要问题和可复核证据位置。
- 跨 reviewer 的一致与分歧。
- Editor synthesis 或 decision memo。

适合投稿前预审、返修前质量检查和独立模型交叉审稿。不适合直接改写论文或替作者回答全部意见。需要修改文稿时，把审稿产物交给 Writing 或 Research。

输入必须明确哪一份 PDF 是 canonical manuscript。不要同时给多个近似版本而不说明优先级。

## 5.6 Literature Review

Literature Review 使用单一 LitReview DeepAgent，直接组合：

- 公共网页搜索。
- 受控 `agent-browser`，可利用用户本人已授权的机构会话。
- 本地文献 ingest 和 query。
- DOI、元数据、证据表和引用 finalizer。
- 文献阅读与写作质量 skills。

它适合论文发现、全文阅读、主题综述、证据矩阵、方法比较和引用整理。它不运行计算，也不承担完整 manuscript 生产。

受控浏览器不会绕过登录墙、CAPTCHA、OTP 或安全警告。需要登录时由用户亲自操作。检索到摘要不等于读到了全文，最终报告应区分 metadata、abstract、full text 和用户提供文件的证据级别。

## 5.7 通用子 agent

DeepAgents 运行时可能显示 `general-purpose` 子 agent。它用于同一职责 lane 内的上下文隔离，继承父层可用工具，不获得新的权限，也不能绕过 worker、远程任务或安全边界。用户通常不需要点名要求它。

## 5.8 工具和 skill 的关系

工具执行一个具体动作，例如读文件、分析结构或提交远程 stage。Skill 是一套工作方法和检查清单。一个 skill 出现在某个 worker 的上下文中，不代表该 worker 自动获得所有同名工具；最终调用权限仍由 allowlist 和 task audience 决定。

运行时会把内置 skills 复制到项目的 staged skill 区，并叠加 workspace 的 `self_develop_skills`。项目级同名 skill 可以覆盖内置版本，但只从下一次 run 加载。

## 5.9 选择入口的例子

```text
“把这个 POSCAR 扩成 3x3x1 并保留 Selective Dynamics”
-> Experiment

“系统比较 Pt 单原子在 CeO2 不同位点的文献和计算证据，列出下一步计算计划”
-> Research

“根据 results.csv 和图 2 起草 Results，不添加新数据或引用”
-> Writing

“对 manuscript.pdf 给出三位 reviewer 和 editor 的投稿前评审”
-> Peer Review

“检索 2021 至今抗烧结 Pd 催化剂工作，建立 DOI 去重证据表”
-> Literature Review
```

如果一项任务在执行中改变性质，可以在同一 thread 停下，明确保存当前产物，再新建合适入口的 thread 继续。Entry 运行中不能切换。
