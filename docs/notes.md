
## 一、新计划的构想与设计原则

### 1) 你当前的“可用核心”是什么

你已经有一个非常关键的基础：一个单循环的自规划执行器（Orchestrator），具备：

* plan：产出 ToDo 与 NextStep
* step：每轮决定一个 tool call 或 finish
* registry：工具全量注册，且每个工具有 Pydantic schema
* observation：把每轮执行结果写入记忆

它的优势是：**“能跑起来，且能从有限工具库里自我安排”**。
它的短板是：**“不可恢复、不可审计、容易参数错、容易早停、工具扩展后可见性会拖垮决策”**。

所以里程碑 0 到里程碑 2 的核心目标不是改变“能不能规划”，而是把它升级为一个工程可用的执行内核：可持久化、可校验、可恢复、可扩展到多 Team。

### 2) 新计划的“薄腰”是什么

为了避免未来几千行代码后每次改架构都推倒重来，新计划采用一个明确的“薄腰”：

* **World Model（全局可读状态）**
* **Event Log（过程可追溯）**
* **Task/ToolCall（可调度、可恢复的执行单元）**
* **Tool Graph + Router（只把相关工具暴露给某个角色）**

所有未来扩展（Literature Team、SurSim、更多表面工具、更多筛选流程）都通过这四个接口接入，避免在 Orchestrator 内部堆越来越多特例。

### 3) MultiAgent 的策略：先“角色化”，再“多进程化”

你现在担心的不是多 agent 的数量，而是可靠性与工程复杂度。

因此新计划的 MultiAgent System 采用“逐步上强度”的路线：

* 早期：同一个 LLM，不同 Prompt 角色（Planner/Parameterizer/Validator/Executor/Analyst/Evaluator），由确定性的 Supervisor 调度
* 中期：ComputeTeam 与 LiteratureTeam 分离，各自内部仍然是角色化多 agent
* 后期：当确有必要，再将部分角色并行化或异步化

这样既能在论文叙事上满足 MultiAgent，又不在工程上一次引入过多不可控复杂性。

### 4) World Model 的策略：分层与证据链

你提出的顶层三件套 Hypothesis/Observation/Conclusion 我认为非常合适，并且要与分层 world model 搭配：

* 全局 World Model：面向人类可读证据链，存 H/O/C + literature/design_space/targets/analysis/tasks 的摘要与引用
* Team 内部 World Model：

  * Compute 内部存 Material/Surface/Site/作业细节/中间构型/失败原因等
  * Literature 内部存文献切片、RAG 索引、抽取表格等
* 向上“只抛 Observation”：把大量细节压缩成可审计的 Observation 节点，并附带 artifact 引用

这可以成为你与 SciSciGPT 等工作的主要差异点之一，因为它面向“科研证据链”和“复现性”。

---

## 二、线性开发与测试计划（阶段化里程碑）

说明：每个阶段都包含 3 部分

* 开发重点：本阶段必须落地的系统能力
* 对现有代码的改造原则：尽量复用 Orchestrator 和现有 registry
* 验收测试：用任务来验证，避免只做架构不出结果

你说 Literature Team 可稍晚实现，因此阶段 0 到阶段 2 以 ComputeTeam 为主。

---

## 阶段 0：执行内核工程化（持久化、校验、可恢复、反早停）

### 开发重点

1. **Run Context 与 StateStore**

   * 每次运行都有 project_id 与 run_dir
   * 生成并维护 world_model 的快照文件
   * 追加写 event log（jsonl），记录 plan、decision、toolcall、result、finish 事件

2. **ToolCall 结构化记录与结果归档**

   * 每次工具调用都有唯一 id
   * params、校验结果、返回值摘要、artifact 引用都被记录
   * tool 的全量返回值不直接塞进 LLM memory，而是落盘，memory 只保留 digest 与路径引用

3. **Schema 校验闭环**

   * 工具执行前必须用 Pydantic schema 校验 params
   * 校验失败不执行工具，把错误写成 Observation，并驱动 LLM 在下一轮修正 params
   * 这一步是解决你“run 五次崩三次”的核心工程方案

4. **Finish Guard（反早停）**

   * LLM 想 finish 时，系统检查“成功条件是否满足”
   * 成功条件不引入固定 kind，而是来自 plan 的 success_criteria 或由系统默认规则构造
   * 不满足则阻止 finish，并把缺失项写入下一轮指令

5. **主入口支持 resume**

   * 允许强制中断后从 world_model + workflow_state 恢复继续跑
   * 至少保证“不会丢已完成的工具输出与关键状态”

### 对现有代码的改造原则

* 不改变你 plan/step 的基本交互范式
* 把“记录”“校验”“归档”“finish 守门”作为 Orchestrator 的外置组件注入
* 工具 registry 维持现状，只补 meta 查询接口即可

### 验收测试

* 用你现有 O₂ demo 作为回归任务（不需要 task kind）

  1. 正常跑通：得到 vasp_summarize 输出的关键数值并生成最终 summary
  2. 注入错误参数：让 LLM 输出一个错误字段，系统能捕获校验失败并驱动其修正，最终仍跑通
  3. 中途 kill 后 resume：能继续执行并在最终报告中给出 artifact 路径

---

## 阶段 1：引入 Task 系统（持久化调度单元，不绑定 demo kind）

Task 系统依然必须引入，因为它是 MultiAgent 调度、批处理、长周期恢复、分阶段工作流的基础。

### 开发重点

1. **Task 的定义与生命周期**

   * Task 是持久化对象，有 id、goal、inputs、constraints、success_criteria、status
   * status 至少包括 pending/running/blocked/done/failed
   * Task 不用预先定义很多 kind，前期只需要支持“可描述的 goal + success_criteria”

2. **Task 与 Orchestrator 的关系**

   * Orchestrator 不再直接以 user_request 作为唯一上下文
   * 而是以 “当前 task.goal + world_model 摘要” 作为输入
   * 你现有 ToDo 可以自然迁移为 Task 列表或 TaskGroup

3. **Task Scheduler（确定性调度）**

   * 一个简单的调度器选择下一个可运行 task
   * blocked task 需要等待某个 artifact 或 job 完成
   * 这一步为后续 HPC 长作业奠定结构

4. **ToolCall 与 Task 的绑定**

   * 每个 ToolCall 归属于某个 Task
   * ToolCall 的输出以 artifact 引用形式反馈到 Task 的 outputs

### 对现有代码的改造原则

* 保留 plan/step，但 plan 的输出升级为：

  * Task 列表（或 ToDo 仍可保留但同时落成 Task）
  * success_criteria 写入每个 Task，而不是写死 kind
* Scheduler 与 Orchestrator 解耦：Scheduler 负责选择任务，Orchestrator 负责推进一个任务一步

### 验收测试

* 仍以 O₂ 为例，但验证点变为“任务化”

  1. 运行后 world_model 中存在至少 1 个 Task，且 ToolCall 都挂在该 Task 上
  2. 中断 resume 时，Task 状态与 ToolCall 状态一致，且不会重复执行已完成工具
  3. finish 时，Task.success_criteria 被满足才允许 done

---

## 阶段 2：工具图谱与工具可见性路由（Tool Graph + Router）

你已经预判到工具变多会让 LLM 变笨，这是事实。阶段 2 的目标是让系统具备“只暴露相关工具”的能力，并为后续异相表面工具爆炸做好治理。

### 开发重点

1. **Tool Graph 的元数据层**

   * 每个工具除了 schema，还应有：

     * tags（surface, adsorption, vasp, ml, file, analysis 等）
     * input/output 的 artifact 类型声明（哪怕是轻量的）
     * preconditions（例如必须存在某个目录或文件）
   * Tool Graph 不必上复杂图数据库，先做结构化元数据与依赖关系即可

2. **Tool Router**

   * 对每个角色或 task，选择一小部分工具暴露给 LLM
   * 选择策略先用 tags 与规则（例如 adsorption 任务只给 slab、adsorbate、vasp、summarize、file 工具）
   * 后续再加 embedding 搜索作为优化，不是本阶段必须

3. **Parameterizer 与 Validator 角色化拆分**

   * 现在是 Controller 一步输出 tool params
   * 阶段 2 将它拆成两个角色（不一定要两个模型，但两个 prompt）

     * Parameterizer：生成 ToolCall params
     * Validator：检查 schema、路径存在性、关键参数范围，失败则返回结构化错误并驱动重试

4. **WorkflowState 初版**

   * 记录 toolcall 的状态与产生的 artifact
   * 为 HPC job 管理做接口预留

### 对现有代码的改造原则

* Orchestrator 的 step_prompt 不再包含全量 tools，而是包含 Router 选出的 subset
* Validator 首先使用你已有 Pydantic schema，其次用少量硬规则补齐（比如必须在 workspace 下写文件）

### 验收测试

* 引入一个新任务验证工具治理能力：单表面吸附计算的最小闭环

  1. 生成分子或吸附物（create_molecule_from_smiles）
  2. 结构松弛（mace_relax 或 mp_relax_prepare，视你工具实际含义）
  3. vasp_execute
  4. vasp_summarize
* 验收重点：

  * LLM 看不到无关工具也能完成
  * 参数错误时 Validator 能拦截并纠错
  * world_model 的 artifact 索引中能找到输入输出目录与总结结果

---

## 阶段 3：ComputeTeam 子系统化（MultiAgent roles + 内部 world model）

到这里，你已经有了可运行的任务执行内核、任务系统、工具路由与校验。阶段 3 的重点是把 Compute 侧真正变成“可写进论文的 MultiAgent Compute Team”。

### 开发重点

1. **ComputeTeam 的角色划分固定化**

   * Planner（生成子计划或 task 内步骤）
   * Parameterizer（生成 ToolCall）
   * Validator（校验并反馈）
   * Executor（执行工具与提交作业）
   * Analyst（把大量结果压缩成 Observation）

2. **Compute 内部 world model**

   * 存 Material/Surface/Site/构型版本/中间结果/失败原因
   * 顶层 world_model 不展开这些细节，只保存引用与 Observation

3. **Observation 生成机制**

   * Analyst 在关键里程点生成 Observation 节点
   * Observation 必须包含可审计引用：toolcall id、artifact 路径、关键数值

### 对现有代码的改造原则

* 不要求你一开始就把 Orchestrator 拆成很多类
* 允许用“同一个模型，多套 prompt + Supervisor 逻辑”实现多 agent
* ComputeTeam 对外暴露的接口是：推进 task、产出 observation、更新 artifacts

### 验收测试

* 扩展吸附任务为“批量小实验”，验证 Analyst 的价值

  * 同一材料不同位点或不同构型跑 2 到 3 个变体
  * Analyst 产出一个 Observation，总结哪个更稳定、差多少能量、证据路径在哪里
* 验收重点：

  * 顶层 world_model 能读懂结论与证据
  * 内部 compute world model 可用于复现与调试

---

## 阶段 4：引入 Literature Team（晚一点上，但一上就接入任务系统与证据链）

你希望 Literature Team 晚些实现，我建议在阶段 4 才引入，并且一开始就让它产出“可执行的设计输入”，否则会沦为泛化摘要。

### 开发重点

1. **LiteratureTeam 的最小闭环**

   * 文献检索或本地 corpus 读取
   * 抽取描述符定义、常见材料族、关键结论
   * 输出 design_space 与 targets 的候选定义
   * 产出 Hypothesis 节点，注明引用来源

2. **与 Task 系统对接**

   * LiteratureTeam 不直接控制 ComputeTeam
   * 它只生成或更新 Task：例如创建一个 compute task，目标是验证某个描述符或某组候选
   * 这样才符合 Manager + Specialists 的 MultiAgent 调度叙事

3. **引用与可审计性**

   * literature 的证据以 excerpt + 文献信息 + artifact 引用形式存入 world_model
   * 便于后续 Conclusion 引用

### 验收测试

* 使用一个小型固定语料（哪怕先手工放 3 到 5 篇 pdf/markdown）
* 让 LiteratureTeam 给出：

  * 一个 design_space（元素集合或材料族）
  * 一个 targets（目标描述符）
  * 两条 Hypothesis
* 随后自动生成至少一个 compute task 去验证其中一条 hypothesis 或筛一小组候选

---

## 阶段 5：全局证据链 World Model（Hypothesis/Observation/Conclusion）与 Evaluation Specialist

这是你想要的“能发表的创新点区域”。此阶段把证据链机制制度化，并引入一个评价与收尾角色。

### 开发重点

1. **顶层 world model 定型**

   * literature / design_space / targets / analysis / tasks
   * hypotheses / observations / conclusions
   * 明确三类节点的引用关系：

     * hypothesis 被哪些 observation 支持或反驳
     * conclusion 由哪些 observation 支撑
     * observation 引用哪些 toolcalls 与 artifacts

2. **Evaluation Specialist**

   * 检查结论是否有证据
   * 检查关键数值是否来源可追溯
   * 检查任务是否真的完成 success_criteria
   * 必要时创建补证 task，而不是让系统草率结束

3. **人类可读报告生成**

   * 输出一个报告文档（markdown 或 pdf）
   * 每个 conclusion 后列出证据引用（observation 列表与 artifact 路径）

### 验收测试

* 端到端小例子：

  * LiteratureTeam 给出 design_space 与 hypothesis
  * ComputeTeam 做少量计算产生 observations
  * Evaluation 生成至少 1 条 conclusion，并带证据链
* 验收重点：

  * 任何 conclusion 都能回溯到 observation，再回溯到 toolcall 与文件

---

## 阶段 6：长周期与规模化（HPC job、批任务、重启、容错策略）

这一阶段才是真正面向“几天 ML 筛选与 DFT 验证”的工程强化。

### 开发重点

1. **WorkflowState 与 Job 管理**

   * ToolCall 可能提交 HPC job，状态为 submitted/running/done/failed
   * Scheduler 能识别 blocked task 并轮询 job 状态
   * 失败可自动重试，或标注需人类介入

2. **批处理与并发策略**

   * 支持批量候选的分片运行（避免一次计划爆几十个 toolcall）
   * 允许任务队列分批推进，保证稳定

3. **重启与幂等性**

   * 工具执行尽量幂等：同一 toolcall id 再次执行不会覆盖或造成混乱
   * 通过 artifact 命名与状态检查避免重复计算

### 验收测试

* 以你规划的里程碑 2 类任务为目标：

  * 划定二元材料空间
  * ML 快速筛选
  * top-K DFT 验证
* 重点验收：

  * 可暂停、可 resume
  * 部分 job 失败不会导致全局崩溃
  * 结果与证据链仍然完整

---

## 三、关于“是否需要最大的 SupervisorAgent”

在本计划里，顶层 Supervisor 建议优先做成**确定性调度器**，而不是把所有控制权交给一个大 LLM。

原因很实际：

* 长周期任务需要可预测的恢复逻辑
* 多 agent 的对话式调度如果没有硬状态机，很容易出现漂移或重复执行
* LLM 更适合做：提出任务、拟定 success_criteria、解释与总结

因此本计划的 Supervisor 的定位是：

* 负责选择下一个 task
* 负责调用哪个 team 去推进它
* 负责持久化与重启
* 负责触发 Evaluation 时机

而不是每一步都让它“自由规划”。

这也符合 SciSciGPT 类系统的 Manager + Specialists 思路，只是把 Manager 的关键控制部分做成可控的工程逻辑。

---

## 四、这份计划的“前文总结”与可独立性确认

为了便于你把这份计划复制到新的讨论环境中，这里用几句话把背景假设与关键术语总结清楚：

1. 现状：你已有一个 LLM 驱动的 Orchestrator，采用 plan 和 step 两段 prompt，在一个循环中选择 registry 里的工具执行任务；工具 registry 已使用 Pydantic schema 注册了工具的输入结构。
2. 痛点：随着工具增多会导致决策混乱；参数错误会导致崩溃；LLM 容易早停；长周期任务需要可重启；未来需要 MultiAgent System 与更强的证据链 world model。
3. 目标：构建一个面向异相催化的科研智能体，能从文献与数据设定设计空间，在 ML 筛选与 DFT 验证之间自动编排，并输出人类可读的证据链结论。
4. 核心设计：以 World Model + Event Log + Task/ToolCall + Tool Graph/Router 为系统薄腰；ComputeTeam 先行，LiteratureTeam 后接入；顶层用确定性 Supervisor 调度；World Model 分层并采用 Hypothesis/Observation/Conclusion 证据链结构。

基于以上总结，这份开发与测试计划**可以独立作为新的讨论起点**，不依赖我们之前对话的上下文内容。

---
