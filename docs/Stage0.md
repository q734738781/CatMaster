阶段 0 的定位很明确：**把你现有的单循环 Orchestrator 升级为一个“工程可用的 Compute Core v0”**。它解决的不是“会不会规划”，而是“能不能稳定跑、能不能断点恢复、能不能追溯证据、能不能防止早停、能不能把工具扩展做起来”。

为避免概念混淆，阶段 0 我建议你把产物叫 **Project/Run State**（例如 `project_state.json`），而不是最终论文叙事里的 Evidence World Model。阶段 0 做的是“执行状态骨架”，阶段 5 才在其上叠加 Hypothesis/Observation/Conclusion 的证据链语义层。

下面是阶段 0 的实现目标（按交付物与验收标准组织，纯语言描述）。

---

## 阶段 0 的核心目标

### 目标 A：可持久化、可恢复的执行状态骨架

你需要一个进程外的“单一真相源”，使得任何时刻进程崩溃或手动中断后，都能继续跑，而不是重新来过。

**交付物**

1. 每次运行都有唯一 `project_id` 与标准化 `run_dir`（位于 CATMASTER_WORKSPACE 下）。
2. `project_state.json`：运行快照，能完整恢复下一步需要的上下文。
3. `events.jsonl`：追加式事件日志，记录每一步发生了什么，便于审计与写论文补充材料。

**project_state 的最小内容（建议）**

* `meta`：project_id、user_request、workspace、创建时间、llm 配置摘要、schema_version
* `memories`：todo、next_step、observations_digest（只存摘要）
* `tool_calls`：结构化 toolcall 列表（见目标 B）
* `artifacts_index`：本次运行产生的关键输出路径索引
* `run_state`：当前 step、是否 finished、finish_reason、last_error
* `objective`：成功条件 success_criteria（用于 FinishGuard，见目标 D）

**事件日志 events.jsonl 的最小事件类型**

* PLAN_CREATED
* DECISION_MADE
* TOOLCALL_VALIDATION_FAILED
* TOOLCALL_STARTED
* TOOLCALL_FINISHED
* FINISH_ATTEMPTED
* FINISH_BLOCKED
* RUN_FINISHED

---

### 目标 B：每一次工具调用可追溯、可复现、可归档

现在你把 tool 返回值直接塞进 observations，会越来越臃肿，也不利于人类审计。阶段 0 要把“过程与证据”分开：LLM 记摘要，证据落盘可追溯。

**交付物**

1. 每次 tool call 都生成一条 `ToolCallRecord`（写入 project_state），包含：

   * tool_name
   * raw_params（LLM 给的）
   * validated_params（schema 校验后的）
   * status（planned/running/done/failed/invalid）
   * result_ref（结果文件引用路径）
   * error（如失败，保存结构化错误摘要信息Traceback）
2. 工具的完整返回值落盘为 artifact，例如：

   * `artifacts/tool_results/step_0003_vasp_execute.json`
3. `observations` 不再存完整结果，只存 digest 与引用，例如：

   * “vasp_execute 已提交/已完成，workdir=xxx，result_ref=xxx”

**验收要点**

* 任意一个结论数值（例如能量）都能从 final_answer 回溯到：
  final_answer → observations_digest → toolcall record → result_ref 文件 → 原始输出目录

---

### 目标 C：参数强校验闭环，解决“run 五次崩三次”

阶段 0 的策略是“**型式校验先兜底，错误结构化回传，交给同一个 LLM 迭代修复**”

这里把“参数问题”分成两类来处理：

1) **型式错误**：缺字段、类型不对、非法枚举值、出现了未声明的字段等  
2) **语义错误**：字段都对，但逻辑不对，例如路径不存在、目录里缺关键文件、组合不合理导致执行过程中工具失败

阶段 0 的闭环重点是让系统对这两类错误都能做到“不崩溃，可记录，可驱动下一轮修复”。

**交付物**

1. 引入统一的 ToolExecutor（模块或类均可），职责是：
   - 替代当前的直接parse resp.content的json的简单逻辑，首先先校验回复的是否是一个valid json，如果不是，返回TOOLCALL_VALIDATION_FAILED与原因"Response is not valid json"
   - 从 registry 获取工具函数与其 Pydantic schema  
   - 对 LLM 产出的 raw_params 进行 Pydantic 校验，得到 validated_params  
   - **型式校验失败时**：
     - 不执行工具
     - 写入 `TOOLCALL_VALIDATION_FAILED` 事件
     - 生成 ToolCallRecord，状态置为 `invalid`
     - 将校验错误整理成可读的 error digest（包含错误字段、期望类型、缺失字段、禁止字段等），写入 observations_digest
     - 覆写 next_step，引导同一个 LLM 修复参数并重试（带重试上限 N）
   - **型式校验通过时**：
     - 写入 `TOOLCALL_STARTED` 事件
     - 执行工具，并捕获所有异常，保证 Orchestrator 不因工具抛错而崩溃
     - 写入 `TOOLCALL_FINISHED` 或 `TOOLCALL_FAILED` 事件
     - 将工具完整返回值落盘为 artifact（见目标 B 的 result_ref 机制）
     - observations_digest 只记录摘要与 result_ref 引用

2. 统一的 ToolResult 返回约定（阶段 0 最小化版本）

   为避免“语义错误无法静态校验”导致系统崩溃，阶段 0 要求所有工具都遵循一个最小返回结构，至少包含：

   - `status`: `ok` | `failed`（你可按现有执行层语义裁剪）
   - `failed`
     - `traceback`: 工具报错信息（人类可读）

   约束要点：**工具内部必须捕获常见异常并转成 ToolResult.failed，而不是向上抛异常**。这样语义错误会进入“结构化失败回传 + LLM 修复循环”，而不是直接导致进程中断。

3. 重试与死循环控制

   - 对每个 ToolCallRecord 维护 `attempt_count`
   - 型式校验失败与工具执行失败都消耗 attempt
   - 达到上限 N 后，进入“受控失败”：
     - 写入 `last_error` 与失败摘要
     - 将 next_step 改为需要用户介入或切换策略（例如要求用户确认路径/材料体系、用 list_files 检查）

---

### 目标 D：FinishGuard（反早停）

Stage 0 的 FinishGuard 要解决两类失败模式：

早停：只准备输入或只跑到提交作业就 finish。
假完成：模型输出了总结，但缺少关键数值、缺少收敛证据、缺少路径引用，无法复现或审计。

Stage 0 不引入完整 World Model 语义层，但必须做到：
任何允许 finish 的 run，都满足“可检查的最小交付物” + “LLM 基于证据的语义充分性审计”。

**交付物**
   当 step 决策为 `action=finish` 时，Orchestrator 不直接结束，而是进入 finish attempt 流程：

   1) 记录 `FINISH_ATTEMPTED` 事件  

   2) 引入下面的LLM Auditor进行审计，实现 FinishGuard
   其输入应为：
   - user_request
   - ToolCall情况
   - observations_digest

   Auditor 输出固定 JSON，例如：
   - `decision`: allow | block
   - `missing_items`: [...]
   - `suggest_next_step`: ...

   3) 如果不允许结束，需要把这部分反思内容以FINISH_BLOCKED的结果引入世界模型中，由主Orchestrator再次决定应该用的工具。
   4) Summarize 作为“允许 finish 后”的结尾阶段

   - 只有当 FinishGuard 允许 finish，才进入 summarize 阶段
   - summarize 输出除了自然语言总结，还建议额外落盘一个 `final_report.json`，包含：
     - 引用过的 artifact refs 列表
     - 关键数值摘要
   - 写入 `RUN_FINISHED` 事件，并在 `project_state.run_state` 标记 finished 与 finish_reason


---

### 目标 E：支持 resume 的主入口与幂等运行

你提到里程碑 2 可能跑几天，因此阶段 0 就要把“重启机制”做成确定性的工程能力。

**交付物**

1. 一个统一入口支持：

   * 新开 run（生成新 project_id 或用户指定）
   * `--resume` 从已有 run_dir 恢复
2. resume 的最小保证：

   * 已完成的 toolcall 不重复跑（至少不重复写坏结果）
   * 已生成的 artifact 不丢失
   * run_state.step 能继续推进

**验收要点**

* 在 vasp_execute 后强制中断进程，恢复后能继续到 vasp_summarize 与最终 summary。
* 恢复后不会把同一个 job 重复提交（如果你现有 vasp_execute 已经有 job_id 或工作目录检查机制，则应利用它；如果没有，阶段 0 至少要做到“检测输出目录存在时不盲目重复”）。

---

## 阶段 0 的非目标（明确不做，避免扩张）

为了让阶段 0 聚焦并尽量不推倒重构，这些内容在阶段 0 不要求完成：

1. MultiAgent Team 的完整拆分与多角色对话协作（阶段 2/3）
2. 工具图谱的完整查询与依赖推理（阶段 2）
3. Literature Team 与文献 RAG（阶段 4）
4. 最终证据链语义层 world model（Hypothesis/Observation/Conclusion 的完整体系，阶段 5）
5. 大规模任务队列与并发调度（阶段 6）

阶段 0 只做“执行层骨架”，但要保证后续这些能力都能在骨架上扩展，而不需要推倒重写。

---

## 阶段 0 的最小验收任务集（建议你用作 CI/回归）

你可以把阶段 0 的验收固定为三条自动化回归任务，每条都能在你当前工具集中完成：

1. **O2 能量计算正常跑通**
   预期LLM自动规划的调用链：* create_molecule →（可选 mace_relax）→ vasp_execute → vasp_summarize → final summary
2. **参数错误纠错回归**
   * 人为让 LLM 第一次 toolcall 缺字段或类型不对
   * 观察系统拦截、记录、驱动修正，最终跑通
3. **中断恢复回归**
   * 在关键步骤后 kill
   * resume 后继续完成，并且 toolcall/artefact 不重复、不丢失

---

## 你现有 Orchestrator 在阶段 0 的定位

阶段 0 不否定你现有 Orchestrator 的价值，反而是对它做“外设加装”：

* 你现有的 plan 与 step prompt 仍然是主逻辑
* 变化点在于：

  1. toolcall 前强校验
  2. tool result 落盘归档
  3. state 与事件持久化
  4. finish 受 guard 约束
  5. 支持 resume

这能保证你继续高速迭代工具，而不会被工程问题拖死。

---

如果你愿意，我们下一步可以把阶段 0 的目标再进一步“定稿”为两份规范文档（仍然不写代码）：

1. `project_state.json` 的字段规范与版本演进约定（schema_version 0.1）
2. `events.jsonl` 的事件规范与必填字段（event_type、timestamp、step_id、toolcall_id、refs）

这两份规范定下来，你后续无论是引入 Task 系统、工具图谱，还是 MultiAgent Team，都能直接接在薄腰上，不会再出现概念漂移。
