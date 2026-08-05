# 2. Agent、worker、skill 与 tool

[上一章](01-quickstart.zh.md) | [目录](README.zh.md) | [下一章](03-llm-configuration.zh.md)

CatMaster 接受研究目标，并把工作落实为可以检查和继续使用的文件、计算阶段与证据。本章只说明各类能力怎样衔接。五个入口的详细角色、可用 tools、skills 和参考 prompt 见后续章节。

## 四类执行单元

| 单元 | 作用 | 例子 |
|---|---|---|
| Agent | 接收一个研究阶段的目标，并决定由谁完成 | Research、Experiment、Literature Review、Writing、Peer Review |
| Worker 或 specialist | 承担有边界的领域任务 | Materials、Dynamics、ML、ORCA/xTB，或文献、写作、审稿 specialist |
| Skill | 提供领域方法、检查项和交付标准 | 表面构建、终止面筛选、轨迹分析、文献精读、论文写作 |
| Tool | 读取或生成真实结果 | 解析结构、建立 slab、准备输入、提交远程 task、分析文件、编译文档 |

Research 也是可执行入口。它可以把开放问题拆成多个阶段，向 Literature Review、Experiment、Writing 或 Peer Review 下发任务，并在每个阶段返回后读取文件和证据，再决定继续、补查、返工或收束。具体领域动作由拥有相应 tools 与 skills 的 specialist 或 worker 完成。

Experiment 会继续把计算工作交给四类 worker：Materials 处理晶体、表面、吸附、缺陷、反应路径和性质；Dynamics 处理 AIMD、LAMMPS、MLFF MD、restart 和轨迹；ML 处理数据集、MACE 和主动学习；ORCA/xTB 处理分子、构象、xTB、CREST、ORCA、TS、IRC、TDDFT 和 NMR。

跨层委派传递的是科学边界，不是底层运行剧本。Research 只需说明目标、已有证据、必须保持不变的科学条件或比较规则、计算授权和停止点；Experiment 可以补充负责的 worker 以及 canonical 输入/输出路径。Worker 在这些边界内自行完成准备、选择兼容执行路径、修正实现细节、故障恢复和领域 QC。准备、smoke、提交和恢复即使属于不同机械步骤，只要服务于同一个科学目标，也不应被强制拆成多次委派。

Specialist 的一次错误选择不自动构成人类 blocker。如果指定的 worker、task 关键字、backend 或步骤不合适，worker 应把具体不匹配返回给直接委派者；Experiment 优先改写 brief 或改派另一条兼容 worker/执行路径，同时保持科学模型、体系、条件、比较规则和目标证据不变。如果科学等价路线超出 Experiment 的 worker 权限，而本轮来自 Research 委派，则返回 Research 重新路由。只有不存在已授权的科学等价路线，或者继续执行会改变用户明确要求、必须由用户选择的科学问题、已批准成本/时间、安全或授权边界时，才需要等待人类输入。

## 一项研究任务可以推进到哪里

以"解释 Pd 单原子在 CeO2 上的稳定机制"为例，Research 可以先让 Literature Review 建立机制和表征证据表，再让 Experiment 判断哪些结构或能量问题能够计算。获得计算结果后，它可以要求 Writing 整合文献与计算证据，也可以把固定稿件交给 Peer Review 独立审查。

```text
Research 目标
  -> Literature Review：检索记录、证据表、引用库
  -> Experiment：结构候选、计算 stage、远程结果、分析报告
  -> Writing：Markdown、LaTeX、DOCX、图件或 PPTX
  -> Peer Review：reviewer reports、editor synthesis、修订问题单
```

实际工作可以停在任一明确边界，例如只完成文献证据、只建立候选结构、准备好计算但不提交、等待远程结果，或只整理现有结果。Prompt 中写清目标、输入路径、科学约束、是否允许远程计算和本轮停止点即可。

## 能力范围取决于入口和部署

每个 worker 只获得与职责相符的 tools 和 skills。Materials 可以建立 slab 和 VASP 输入；Writing 可以编译文稿和制作图件；Literature Review 可以检索、阅读、导入语料和整理引用。跨领域工作由 Research 或 Experiment 委派给合适的执行者。

远程执行还取决于部署端登记的 task、resource、machine 和 MLFF backend。Agent 可以查询当前 catalog，并只提交其中已启用且属于自身职责的 task。准备输入与远程执行是两项独立能力，详见[第 8 章](08-remote-execution.zh.md)。

## Workspace 与 thread

每个 workspace 包含两个部分：

```text
workspace/
  files/
  metadata/
```

`files/` 是用户与 Agent 共同使用的项目区。上传的结构、论文和数据，以及生成的候选、脚本、报告、图件和远程结果都保存在这里。Prompt 中使用 `files/` 内的相对路径，例如 `structures/slab.vasp` 或 `writing/results.md`。

`metadata/` 保存 thread checkpoint、运行观测、artifact 索引和远程恢复信息。用户通常不直接编辑这里。完整备份应同时包含 `files/` 与 `metadata/`。

Thread 保存一条连续研究上下文。继续已有工作时，说明需要重新读取的文件、必须保留的条件和禁止重复的步骤，比只发送"继续"更可靠：

```text
继续表面筛选。先读取 notes/termination_review.md 和
structures/ceo2_111_candidates/，核对现有候选与上次审计。
不要重新生成哈希一致的结构。从尚未决定的终止面继续，仍然不要提交远程计算。
```

## 用户可以核对什么

Chat 显示委派、Progress 和 tool 卡；Files 保存真实交付物；Monitor 记录执行过程；远程 receipt 提供可恢复的任务身份。Review 模式会保护通用文件写入和远程提交，但部分领域 tool 会在一次调用中直接生成声明的输出文件，因此仍应在 prompt 中写明输出路径和停止点，并在完成后检查文件内容。

这些记录便于复查过程，但不会代替科学判断。提交计算前仍需确认体系、电荷、自旋、约束、方法、收敛标准、采样条件、能量基准和成本。下一章按入口介绍五类 Agent 的能力与参考 prompt。
