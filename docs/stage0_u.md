### 目标 C：参数强校验闭环，解决“run 五次崩三次”

阶段 0 的策略是“**型式校验先兜底，错误结构化回传，交给同一个 LLM 迭代修复**”

这里把“参数问题”分成两类来处理：

1) **型式错误**：缺字段、类型不对、非法枚举值、出现了未声明的字段等  
2) **语义错误**：字段都对，但逻辑不对，例如路径不存在、目录里缺关键文件、组合不合理导致执行过程中工具失败

阶段 0 的闭环重点是让系统对这两类错误都能做到“不崩溃，可记录，可驱动下一轮修复”。

**交付物**

1. 引入统一的 ToolExecutor（模块或类均可），职责是：

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

### 目标 D：FinishGuard（反早停），把“完成条件”制度化

阶段 0 的 FinishGuard 不能依赖 LLM 去“自由生成 success_criteria”，因为随着任务复杂度与工具数量增长，LLM 很难稳定知道“什么叫完成”。因此阶段 0 采用“**完成契约 Completion Contract**”的做法：

- **完成条件由运行入口显式传入（demo/CLI 配置）并写入 project_state**  
- LLM 可以在 plan 阶段提出建议，但不作为唯一真相源  
- 当 LLM 试图 finish 时，由 FinishGuard 依据契约进行检查，未满足则强制继续

**交付物**

1. `project_state.objective` 存储完成契约（Completion Contract）

   由运行入口传入并落盘，最小字段建议：

   - `contract_version`
   - `required_deliverables`: 需要产出的交付物类型或引用（建议用“artifact 键/路径模式/结果字段”的组合描述，而不是硬编码工具链）
   - `required_evidence`: 必须出现的证据条件（例如必须有至少一次成功的计算类结果引用，必须有包含关键数值字段的分析结果）
   - `finish_policy`: 最大 finish 尝试次数、是否启用 LLM 审计等

   注：对阶段 0 的 O2 demo，完成契约完全可以很具体，例如“必须产生一次 vasp_summarize 且包含能量字段，并且 result_ref 文件存在”。对后续筛选任务，则可改为“必须生成 candidates 表格 artifact 与 top-N 分析摘要”等。

2. FinishGuard 的两段式收口流程

   当 step 决策为 `action=finish` 时，Orchestrator 不直接结束，而是进入 finish attempt 流程：

   1) 记录 `FINISH_ATTEMPTED` 事件  
   2) 运行 **确定性 ContractChecker**（非 LLM），检查 `project_state` 是否满足完成契约，包括但不限于：
      - 必需的 artifact 引用是否存在且可读取
      - 必需的结果字段是否已出现在某个 tool result artifact 中
      - 必需的 evidence 条件是否满足（例如至少存在一次 `status=ok` 的计算结果引用）
   3) 若不满足：
      - 写入 `FINISH_BLOCKED` 事件
      - 生成缺失项列表 `missing_items`
      - 覆写 next_step，强制引导继续（例如“缺少能量总结，请执行 vasp_summarize 并产出 energy 字段”或“缺少候选表格，请先输出 candidates.csv”）
      - 继续主循环

3. 可选的 LLM 审计式 FinishGuard（建议作为开关，而不是强依赖）

   在 ContractChecker 判定“已满足”或“接近满足”时，可以启用一个低温度的 LLM Auditor 进行二次审计，用来处理少量难以形式化的情况。其输入应严格限制为：

   - user_request
   - completion_contract
   - observations_digest
   - artifacts_index（只给索引，不给全量日志）

   Auditor 输出固定 JSON，例如：
   - `decision`: allow | block
   - `missing_items`: [...]
   - `next_step`: ...

   关键约束：Auditor **不得发明完成条件**，只能基于 completion_contract 判断。

4. Summarize 作为“允许 finish 后”的结尾阶段

   - 只有当 FinishGuard 允许 finish，才进入 summarize 阶段
   - summarize 输出除了自然语言总结，还建议额外落盘一个 `final_report.json`，包含：
     - 引用过的 artifact refs 列表
     - 关键数值摘要
   - 写入 `RUN_FINISHED` 事件，并在 `project_state.run_state` 标记 finished 与 finish_reason

**验收要点**

- 模型即使尝试在“只生成了 vasp 输入文件”后 finish，也会被 FinishGuard 阻止，直到完成契约要求的关键结果与总结产出为止  
- 完成契约由 demo/入口显式传入，避免未来任务扩展时把“必须调用哪些工具”写死在系统里或完全交给 LLM 生成  
- 若开启 LLM 审计，其作用是补充审计而非决定性约束，系统最终仍以 completion_contract 与 ContractChecker 的确定性检查为主
```
