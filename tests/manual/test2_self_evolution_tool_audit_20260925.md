# 公共 demo test2：self-evolution 与工具接口检查

检查日期：2026-09-25。结论依据公共部署的实际记录、当前开发代码、最终模型可见 schema，以及本地回归和小样例复现。公共工作区只读检查，没有重跑计算、重试失败的学习任务或改变已激活版本。

## 结论

test2 的计算已结束，但 self-evolution 没有全部正常完成：6 个学习 job 中，4 个完成，2 个反思阶段失败；两个候选通过 reviewer 并被选为后续 run 使用的版本。失败的两次模型其实写出了符合 `ReflectionBatch` 的 JSON，却放在普通回复中，没有调用结果工具。运行器只读取 `structured_response`，因此将它们记为失败。

这里确实有接口说明和结束处理的缺口。现有实现使用标准 DeepAgent 加 LangChain `ToolStrategy`，并非从最终回复抽取 JSON；原 prompt 的 “Return a compact batch / structured response” 没有把提交位置说清楚。仅绑定 schema 不足以避免这次失败。当前修改明确了终止工具调用，并对普通回复结束提供一次原会话内纠正。

普通工具另有已经复现的实现缺陷，不能只补 skill：ORCA 优化结构提取与托管输出不匹配、可能取轨迹第一帧；构象筛选会把绝对能量直接当相对能量，且读取能量摘要失败后静默继续。它们的实现本次均未修改。

## 1. 实际检查范围

| 层次 | 本次覆盖 | 不能从中推出的结论 |
|---|---|---|
| 公共 test2 | workspace messages、run observations、self-evo job/candidate/review、有效版本选择、候选文件差异 | 没有重新运行科学计算或重新验证全部科学结论 |
| 注册工具 | 101 个工具、39 个实现文件的入口定位；全量最终 schema、字段说明及 LangChain/OpenAI 两种导出比较 | 101 个工具并不同时暴露给一个 agent；未对每个外部引擎逐项在线运行 |
| 运行时附加工具 | 异步任务、容量切换、self-evo SQL/事件读取/候选准备/源码查看、普通文件和执行工具 | 注册器清单不能代表全部运行时能力 |
| 调用链 | ORCA 准备→托管执行→分析/提取→构象筛选，Research 建 E→派工→写回，self-evo 查询→文件修改→review→选择 | 定向检查不是所有科学算法的完整审计 |
| 回归 | self-evo、specialist、原生调度、工具适配、远程提交替身、量化/MD/ML、Research Graph、文献/材料检索和晶体工具 | 本地替身通过不代表付费 provider、远程队列或所有外部 API 在线验收 |

全量 schema 检查结果：101 个工具均有工具级描述；`StructuredTool.args_schema` 与 `as_openai_tools()` 的参数定义一致（忽略重复的顶层 description）。最终导出没有本次扫描所针对的 nullable/`anyOf` 控制字段问题。14 个 Research Graph 工具有合计 51 个顶层字段缺少 description，其中包括 `decision_rule`、`plan_summary`、`refs`、`expected_revision` 等。这属于可见说明不完整，不等于对应调用必然失败。

已有 [2026-09-16 工具输出审查](tool_skill_output_audit_20260916.md) 的结论用作背景；本次没有把那次的检查量算成本次新验证。

## 2. self-evolution 的结构化结果

### 2.1 test2 的失败发生在哪里

两次失败反思的观测目录是：

- `self_evolution_313f1bb47acc4934a8dac030e7264b15`：对应 MLFF Experiment，记录了 7 次模型调用。
- `self_evolution_3026c9ccab7d4feeb65cbeca593c56ba`：对应 Research 接收 MLFF 完成结果的回合，记录了 3 次模型调用。

末次响应都是普通正文中的 JSON 代码块，包含 `items` 和 `no_change`，没有 `tool_calls`，结束原因是 `stop`。从记录取出的两个 JSON 都能通过部署版本的 `ReflectionBatch` 校验。因此，已定位的问题是提交通道错误，不是 JSON 缺字段或字段类型错误。它们在数据库中仍然是失败，不能把正文里的 `no_change` 当作宿主已接受的结果。

当前安装的 LangChain 会为这类 ToolStrategy 请求工具调用；桥接层把 `any` 转成 provider 的 `required`。但这两次没有完整的历史 HTTP 请求可核对，不能进一步断言究竟是哪层忽略了选择要求。原生 agent 循环在没有工具调用时可以结束；ToolStrategy 对无效工具参数的纠正，并不自动覆盖这种普通回复结束。

相关代码：[agent 构建与结果读取](../../catmaster/runtime/self_evolution/agents.py)、[结果模型](../../catmaster/runtime/self_evolution/models.py)、[三阶段 prompt](../../catmaster/prompts/self_evolution)。框架依据：[LangChain structured output](https://docs.langchain.com/oss/python/langchain/structured-output)、[middleware hooks](https://docs.langchain.com/oss/python/langchain/middleware/custom)，并核对了本地安装源码。

### 2.2 当前结构是否过重、有无 patch

| 阶段 | 真正的提交工具 | 当前字段作用 |
|---|---|---|
| 反思 | `ReflectionBatch` | 一组 finding；每项判断、目标、变化、证据引用和解释 |
| 提案 | `ProposerResult` | action、目标和解释；可选适用范围、修改方式说明及预期行为变化 |
| 审查 | `ReviewerResult` | 审查决定、说明及面向界面展示的变化/证据/范围信息；`human_checks` 供真实未决授权问题使用 |

这三份 schema 都没有 patch 字段。`delta_operation` 只是可选的修改方式说明，不是要 agent 编写的补丁语法。提案阶段先把 skill 复制到 `/proposed/<group>/<name>/`，再用 `read_file`、`edit_file`、`write_file` 修改；reviewer 检查实际候选文件及差异，获批后系统选中那个精确版本。

Reviewer 的 10 个顶层字段和嵌套变化条目确实比最小决策协议重。不过它们主要是已有审查展示信息，很多可省略，本次失败也发生在更小的 ReflectionBatch 上。此次保留这些兼容字段，修正提交接口；没有增加新的输出语言、patch 协议或 Markdown JSON 提取器。如果后续简化审查展示，应同时调整 reviewer schema 和消费它的界面，避免只删模型字段、留下隐含读取要求。

### 2.3 本次已修正

- 三阶段 prompt 和终止工具描述明确：通过声明的结果工具提交，解释写入文本字段，文件修改通过普通文件工具完成。
- 普通回复结束且没有结果时，用标准 `after_model` hook 在同一会话添加一次 HumanMessage 提醒，再回到模型。已有证据、工具记录和文件修改保留，不重跑整个提案。
- 再次未提交仍然失败，不把可能是草稿的正文猜成正式决定；失败调用和部分输出继续保留在普通 observability 中。
- 同步/异步路径、三种结果类型、正常一次提交、两次普通回复、文件修改不重复都有实际 DeepAgent 循环测试。

## 3. 两次激活通知的含义

| 激活版本 | 实际变化 | 本次判断 |
|---|---|---|
| `sec_df1b9afa38aec55e68bbb14721fb@r0001` | ORCA skill 增加 native/LibXC 拼写说明及 SCAN 示例 | 有真实输入失败和成功修正支撑；适合作为输入语法说明 |
| `sec_c22cd7351838ccee8fd1bbd1b294@r0001` | `research_execution/research-graph-writeback` 增加先复用同目标 Experiment，再 focus/写回的说明 | 有重复建 E 的轨迹支撑；方向合理，但原句未解释 Graph 绑定不等于 Experiment focus |

UTC+8 记录的激活时间分别约为 01:11:54、01:26:16。`active_skills.json` 中对应目标启用、采用 Follow auto、选中获批版本。用户看到的 “activated ... for subsequent runs” 表示 workspace 为后续新 run 选择了这个版本；没有改仓库源 skill，也没有把新说明追溯注入已经结束的计算。

ORCA reviewer 提到约 917 秒的成本，但那是包含另外五个成功任务的整批等待时间，不能全部归因于 SCAN 失败。另发现普通执行 agent 已把 SCAN 说明和几何提取 workaround 写入 workspace memory，而 self-evo 后来又提升了 SCAN skill：这存在重复存放和把产品缺陷固化为 SOP 的风险。本次只记录，没有改历史 memory 或已激活候选。

## 4. 普通工具摩擦与待修问题

“优先”表示建议修复顺序，不表示本次已修改普通工具。

| 优先级 | 问题与证据 | 影响与建议 |
|---|---|---|
| 高 | `extract_optimized_molecules` 只识别 `job.xyz`、`job_trj.xyz` 等固定名，托管 ORCA 使用 `job.runtime...`；test2 七次提取均 count=0 | 工具链不兼容。应与现有 ORCA analyzer 的最终结构识别一致，返回未提取原因；不能要求 agent 反复换目录碰运气 |
| 高 | 提取只扫描根目录的子目录；单个结果目录本身被跳过。`job_trj.xyz` 又因只判断 `.trj` 后缀而读取第 0 帧 | 既可能空结果，也可能静默给错结构。应明示并支持单个结果目录/批次根；正确识别轨迹和最终帧 |
| 高 | `filter_conformer_ensemble` 把 `energy_kcal_mol` 原值直接当 `relative_energy_kcal_mol` | 5 kcal/mol 窗口下，绝对能量 100 和 102 的两个构象被全部剔除，正确的相对差应为 0 和 2。需要明确能量基准及单位，再计算相对值 |
| 高 | 构象筛选读取摘要异常时设置空 energy map 并继续；未知能量也照常保留，结果只说 completed | 错误 JSON 或不支持的摘要可能让能量筛选实际失效。应明确报告缺失/无效能量和实际执行了哪种筛选 |
| 中 | Research 子线程继承 Graph，但 `start_async_task` 没有继承具体 E focus。test2 已有 `exp_980c93ce86674a2b`，子线程又建 `exp_60fe11d8c02d4f93` | 这是派工信息和写回职责衔接问题；本次 skill 已说明传递实际 E ID、查询并 focus，普通工具自动绑定/去重逻辑未改 |
| 中 | 旧 E 最后用 failed/blocked 标记来表达“被重复 E 取代” | “重复计划”不等于“科学执行失败”。需要单独审视现有 Graph 生命周期如何表达替代/撤回；skill 已说明不要为清理重复计划伪造失败 |
| 中 | `orca_prepare` 接受原生关键词并原样写入；它不会验证每个功能关键词。plain `SCAN` 在 ORCA 输入解析时报错 | 说明应明确准备工具边界，并给少量正确示例。已补 SCAN 的 `LibXC(SCAN)` 和 native `r2SCAN`；不引入隐藏的方法翻译或强制预跑 |
| 中 | self-evo 查询的 `run_ref` 必须有 `run:` 且包含完整后缀；两次 proposer 合计六次句柄错误 | 已补空值使用 anchor、完整句柄示例和事件句柄区别；查询实现暂不做自动补全，避免把不完整 ID 猜成另一个 run |
| 中 | 14 个 Graph 工具合计 51 个顶层字段缺少 description | 不宜只靠模型从字段名猜写入范围、引用含义和 revision。后续应按对象职责补字段说明及少量 skill 示例；本次不改这些工具 schema |
| 低 | shell 中前面的 Python 或管道命令失败，尾部命令仍返回 0；test2 出现无 calculator 的 ASE `get_forces()`，以及 `rg` 缺失后的管道 | 属于命令组合/科学读取错误，不能用 stderr 有无文字替代退出码。执行接口保留原输出；不应泛化成所有命令都必须套自定义 wrapper |
| 低 | `edit_file` 的精确文本不匹配后重读修正 | 正常可恢复的 agent 操作失误；没有证据需要新编辑语言或放弃原生工具 |

实现证据主要位于 [molecular_qchem.py](../../catmaster/tools/geometry_inputs/molecular_qchem.py)、[qchem_analysis.py](../../catmaster/tools/analysis/qchem_analysis.py)、[ORCA runtime](../../catmaster/remote/cpu/orca_boot.py)、[local_execution.py](../../catmaster/webui/local_execution.py)、[Research Graph 工具](../../catmaster/tools/misc/research_graph.py)、[self-evo 查询](../../catmaster/runtime/self_evolution/query.py)。

### 4.1 小样例复现结果

在独立临时 workspace 写入两个 H₂ 帧，键长分别为 0.7 Å 和 1.4 Å。使用 `status.json` 中 `returncode=0`、普通 ORCA 输出标识，不启动 ORCA：

| 输入 | 实测 |
|---|---|
| 批次子目录内 `job.runtime.xyz` | count=0 |
| 直接传含 `job.xyz` 的结果目录 | count=0 |
| 传其父目录，子目录内 `job.xyz` | count=1，1.4 Å |
| 子目录只有双帧 `job_trj.xyz` | count=1，但为 0.7 Å，取了第一帧 |
| 两构象摘要只有绝对能量 100、102 kcal/mol，窗口 5 | count=0，应保留两个 |
| 将上述摘要改成无效 JSON | count=2，未报能量摘要错误 |

对应直接入口是 `extract_optimized_molecules` 和 `filter_conformer_ensemble`，均在 `workspace_scope` 内运行。这些是本次主动复现，后两个构象筛选缺陷尚无证据说明已影响 test2 的最终结果。

提取器的 `include_failed=False` 还只检查任务结束标志，不等于验证优化收敛。现有 `analyze_orca_results` 已区分 process/task 状态，skill 中也有这一边界。后续修复应复用这套结果含义，避免再造一个互相矛盾的“成功”定义。

### 4.2 其余工具检查结果

- 远程提交的单 stage / 第一层 batch、阻塞至终态、模板参数、失败输出路径已有明确 schema 和对应回归；此次没有发现需要统一包装其接口的证据。
- CP2K/LAMMPS 输入准备保留完整原生文件，分析区分进程完成与科学任务收敛；轨迹 inventory 与扩散/RDF 分析的用途也已分开。代表性回归通过。
- `build_dataset_from_runs` 虽然名字宽泛，最终描述已明确仅用 VASP `vasprun.xml` 构建 extxyz；不应按名字假设能读取 ORCA、CP2K 或任意 MLFF 输出。
- 文献查询和 self-evo 都依赖可继续读取的句柄；本次保留 SQL、JSON1、分页、原文及事件字段读取，没有添加只返回摘要的替代工具。文献 acquisition/corpus、Graph SQL、Materials Project 条件映射的本地测试通过，未发起新下载或外部检索验收。
- 注册器仍保留 `apply_aider_edits` 这个兼容工具，但 self-evo 使用的工具集合没有绑定它，活跃 specialist/worker 也按各自工具列表装配。它不能作为“self-evo 要求 agent 写 patch”的证据；本次未删兼容工具。

## 5. 调度状态信号

原问题在通用工具错误 middleware：`except Exception` 把 LangGraph `GraphInterrupt` 包装成普通 error ToolMessage。模型于是收到“工具出错”，继续执行，而不是停在原生 checkpoint 等待恢复。test2 那次任务原本已经属于 high 档，不能据这条错误单独认定发生了越过额度的科学执行；但这个错误处理方式本身不正确。

本次让 `GraphBubbleUp` 及其 `GraphInterrupt` 子类原样传播。排队时保留 `capacity_state=waiting`，agent 不继续运行；准入后从同一 checkpoint 恢复，工具返回 `{"task_cost":"high","admitted":true}`。这些是调度状态信号，没有把 agent 型号混进协议。

测试已把真实的通用错误 middleware 纳入容量等待的 DeepAgent/DBOS 集成路径，验证等待不执行后续工作、停止后可继续、原证据保留、前置工作不重复、恢复返回明确准入状态；普通工具错误仍返回可恢复的 error ToolMessage。[原生中断语义](https://docs.langchain.com/oss/python/langgraph/interrupts) 保持不变。

## 6. 本次修改与未修改范围

已修改 self-evo 终止接口说明及一次会话内纠正、调度控制信号透传、相应测试/手册/changelog。补充的说明放在实际接收方能够看到的位置：

- [Research 写回 skill](../../skills/research_execution/research-graph-writeback/SKILL.md)：现有 E 的复用和 focus 示例、重复计划与失败的区别。
- [Research 派工 skill](../../skills/research_specialist/research-graph-control/SKILL.md)：传递目标 E ID；Graph 绑定不自动设置 E focus。
- [ORCA skill](../../skills/orca_xtb_worker/orca-optfreq-thermochemistry/SKILL.md) 及 [方法输入参考](../../skills/orca_xtb_worker/orca-optfreq-thermochemistry/references/orca_method_selection.md)：准备工具边界、两种功能关键词示例。[ORCA 官方输入表](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/DensityFunctionalTheory.html#simple-input-of-libxc-functionals) 与实际成功修正相符。
- self-evo 的 run 查询字段：省略/空值使用 anchor；`run:abc:initial` 是完整 run handle，事件引用另用 `read_evolution_event`。这些 agent 的接口入口是 prompt 和动态工具描述，因此放在它们实际可见的 schema 中，没有新增一个它们不会加载的 skill。

未修改 `catmaster/tools/` 下普通工具实现、远程任务模板、工具授权、科学默认参数、历史 job 状态、候选文件或有效版本。修改尚未部署到公共 demo。

## 7. 验证

本地执行采用 `catmaster` 环境，模型/外部任务调用使用测试替身：

| 检查组 | 结果 |
|---|---|
| self-evo、query、usage、effective selection、local execution、capacity | 140 passed |
| specialist、最终工具提交及容量恢复、ORCA 参考边界定向复测 | 99 passed |
| 注册器、ORCA/xTB、CP2K/LAMMPS、分析/ML、远程提交/模板 | 初次 110 passed，1 项文档边界失败；把功能关键词示例移到方法参考后，该项已通过上组复测 |
| 工具输出、Graph 工具/查询、文献存储/获取、材料查询、晶体、local execution/controls | 87 passed |

以上是分组结果，含重复用例，不将它们相加声称独立用例总数。初次失败来自已有测试要求“操作关键词参考不混入电子方法”，已经通过调整文档位置解决；未放宽测试。新增六个工具小样例的实测结果见 4.1，不将缺陷复现称为修复验收。没有进行新的付费模型 smoke 或公共 demo 部署后的运行验证。

## 附录：注册接口覆盖清单

以下按实现文件列出本次导出的全部工具。仅表示接口已纳入静态检查，在线和行为验证范围以上文为准。

| 实现文件 | 数量 | 注册工具 |
|---|---:|---|
| [catmaster/runtime/literature/acquisition.py](../../catmaster/runtime/literature/acquisition.py) | 2 | `acquire_literature_source`, `batch_acquire_literature_sources` |
| [catmaster/runtime/literature/citations.py](../../catmaster/runtime/literature/citations.py) | 1 | `finalize_citations` |
| [catmaster/runtime/literature/corpus.py](../../catmaster/runtime/literature/corpus.py) | 2 | `ingest_literature_files`, `query_literature_corpus` |
| [catmaster/runtime/literature/tools.py](../../catmaster/runtime/literature/tools.py) | 8 | `search_openalex`, `search_semantic_scholar`, `get_openalex_record`, `get_semantic_scholar_record`, `recommend_semantic_scholar`, `web_search`, `open_public_page`, `find_in_page` |
| [catmaster/tools/analysis/agentic_compile_tex.py](../../catmaster/tools/analysis/agentic_compile_tex.py) | 1 | `compile_text` |
| [catmaster/tools/analysis/fragment_probe.py](../../catmaster/tools/analysis/fragment_probe.py) | 1 | `identify_structure_fragments` |
| [catmaster/tools/analysis/generate_figure.py](../../catmaster/tools/analysis/generate_figure.py) | 1 | `generate_figure` |
| [catmaster/tools/analysis/markdown_pdf.py](../../catmaster/tools/analysis/markdown_pdf.py) | 1 | `render_markdown_pdf` |
| [catmaster/tools/analysis/peer_review_pdf_manuscript.py](../../catmaster/tools/analysis/peer_review_pdf_manuscript.py) | 1 | `peer_review_pdf_manuscript` |
| [catmaster/tools/analysis/peer_review_request.py](../../catmaster/tools/analysis/peer_review_request.py) | 1 | `peer_review_request` |
| [catmaster/tools/analysis/qchem_analysis.py](../../catmaster/tools/analysis/qchem_analysis.py) | 2 | `analyze_xtb_results`, `analyze_orca_results` |
| [catmaster/tools/analysis/results_analysis.py](../../catmaster/tools/analysis/results_analysis.py) | 2 | `analyze_vasp_neb_results`, `analyze_trajectory` |
| [catmaster/tools/analysis/review_pdf_manuscript.py](../../catmaster/tools/analysis/review_pdf_manuscript.py) | 1 | `review_pdf_manuscript` |
| [catmaster/tools/analysis/vaspkit_thermo.py](../../catmaster/tools/analysis/vaspkit_thermo.py) | 2 | `vaspkit_adsorbate_thermo_correction`, `vaspkit_gas_thermo_correction` |
| [catmaster/tools/analysis/vesta_render.py](../../catmaster/tools/analysis/vesta_render.py) | 1 | `render_vesta_views` |
| [catmaster/tools/dynamics/cp2k_analysis.py](../../catmaster/tools/dynamics/cp2k_analysis.py) | 1 | `cp2k_output_summary` |
| [catmaster/tools/dynamics/lammps_tools.py](../../catmaster/tools/dynamics/lammps_tools.py) | 3 | `lammps_prepare`, `lammps_log_summary`, `md_trajectory_summary` |
| [catmaster/tools/execution/remote_submission.py](../../catmaster/tools/execution/remote_submission.py) | 5 | `remote_submission`, `remote_submission_batch`, `get_avail_remote_task`, `get_remote_task_spec`, `get_avail_resources` |
| [catmaster/tools/geometry_inputs/adsorbate_tool.py](../../catmaster/tools/geometry_inputs/adsorbate_tool.py) | 3 | `enumerate_adsorption_sites`, `place_adsorbate`, `generate_batch_adsorption_structures` |
| [catmaster/tools/geometry_inputs/cp2k_prepare.py](../../catmaster/tools/geometry_inputs/cp2k_prepare.py) | 1 | `cp2k_prepare` |
| [catmaster/tools/geometry_inputs/crest_prepare.py](../../catmaster/tools/geometry_inputs/crest_prepare.py) | 1 | `crest_prepare` |
| [catmaster/tools/geometry_inputs/crystal_tool.py](../../catmaster/tools/geometry_inputs/crystal_tool.py) | 8 | `supercell`, `enumerate_unique_sites`, `create_vacancy`, `substitute_species`, `insert_interstitial_at_coords`, `generate_strained_structures`, `generate_kpath`, `generate_phonon_displacements` |
| [catmaster/tools/geometry_inputs/dimer_tools.py](../../catmaster/tools/geometry_inputs/dimer_tools.py) | 4 | `vasp_dimer_prepare`, `make_dimer_mode_from_neb`, `make_dimer_mode_from_mace`, `mace_analyze_frequencies` |
| [catmaster/tools/geometry_inputs/molecular_qchem.py](../../catmaster/tools/geometry_inputs/molecular_qchem.py) | 3 | `enumerate_molecular_conformers`, `filter_conformer_ensemble`, `extract_optimized_molecules` |
| [catmaster/tools/geometry_inputs/molecule.py](../../catmaster/tools/geometry_inputs/molecule.py) | 1 | `create_molecule_from_smiles` |
| [catmaster/tools/geometry_inputs/neb_tools.py](../../catmaster/tools/geometry_inputs/neb_tools.py) | 4 | `estimate_neb_image_count`, `remap_neb_endpoint_atoms`, `make_neb_geometry`, `vasp_neb_prepare` |
| [catmaster/tools/geometry_inputs/orca_prepare.py](../../catmaster/tools/geometry_inputs/orca_prepare.py) | 2 | `orca_prepare`, `orca_nebts_prepare` |
| [catmaster/tools/geometry_inputs/slab_tools.py](../../catmaster/tools/geometry_inputs/slab_tools.py) | 4 | `build_slab`, `fix_atoms_by_layers`, `fix_atoms_by_height`, `fix_atoms_by_indices` |
| [catmaster/tools/geometry_inputs/vasp_band_prepare.py](../../catmaster/tools/geometry_inputs/vasp_band_prepare.py) | 1 | `vasp_band_prepare` |
| [catmaster/tools/geometry_inputs/vasp_prepare.py](../../catmaster/tools/geometry_inputs/vasp_prepare.py) | 1 | `vasp_prepare` |
| [catmaster/tools/geometry_inputs/xtb_prepare.py](../../catmaster/tools/geometry_inputs/xtb_prepare.py) | 1 | `xtb_prepare` |
| [catmaster/tools/machine_learning/dataset_tools.py](../../catmaster/tools/machine_learning/dataset_tools.py) | 1 | `build_dataset_from_runs` |
| [catmaster/tools/machine_learning/mace_ml.py](../../catmaster/tools/machine_learning/mace_ml.py) | 1 | `calculate_al_candidates` |
| [catmaster/tools/misc/effective_skills.py](../../catmaster/tools/misc/effective_skills.py) | 1 | `manage_effective_skills` |
| [catmaster/tools/misc/export_builtin_tool_source.py](../../catmaster/tools/misc/export_builtin_tool_source.py) | 1 | `export_builtin_tool_source` |
| [catmaster/tools/misc/memory_patch_apply.py](../../catmaster/tools/misc/memory_patch_apply.py) | 1 | `apply_aider_edits` |
| [catmaster/tools/misc/progress.py](../../catmaster/tools/misc/progress.py) | 1 | `notify_progress` |
| [catmaster/tools/misc/research_graph.py](../../catmaster/tools/misc/research_graph.py) | 23 | `list_research_graphs`, `revise_research_claim`, `record_research_disposition`, `record_research_review`, `mark_research_planning_no_change`, `create_research_graph`, `query_research_graph_sql`, `set_research_graph_focus`, `create_bound_research_experiment`, `update_bound_research_result`, `resume_bound_research_experiment`, `retract_bound_research_result`, `update_research_graph_scope`, `add_research_hypothesis`, `add_research_experiment`, `record_research_result`, `set_research_result_judgment`, `stage_research_plan`, `record_research_experiment_comparison`, `set_research_graph_completion`, `record_bound_research_result`, `mark_research_experiment_failed`, `mark_bound_research_experiment_failed` |
| [catmaster/tools/retrieval/matdb.py](../../catmaster/tools/retrieval/matdb.py) | 2 | `mp_search_materials`, `mp_download_structure` |
