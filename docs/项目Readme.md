# 项目总览与实施路线 README

本项目的目标是在研究型办公室电脑作为控制平面，打通与 CPU 集群的第一性原理计算以及与独立 GPU 服务器的机器学习力场与结构搜索计算，形成可审计、可扩展并能闭环演进的计算基座。路线以作业包为统一契约，PC 端负责任务打包与调度，CPU 端通过精简的 jobflow Flow 执行 DFT 作业，GPU 端以守护进程执行结构预筛与势场训练等任务。该组织与申请书中的技术路线保持一致，并面向“动态结构 表面机理 宏观动力学”的主线逐步推进。报告中的路线图与智能体框架见第十页与第十四页的图示，可作为本 README 的理论依据与验收参照。

## 架构与交互

PC 端提供 Orchestrator 与策略插件，负责把结构与参数生成 jobflow Flow，并将其与输入文件打包成作业包。作业包包含元清单、事件日志与产物清单三类受控文件，统一记录环境、策略、状态与可下载路径。CPU 与 GPU 侧均通过 jobflow-remote runner 在各自的队列系统中调度 Flow：CPU 跑 VASP（Slurm）、GPU 跑结构预筛与势能训练。VASP 输入规则来自你现有的 Prepare_vasp_inputs 脚本，但以策略插件形式在 PC 侧注入到 Flow，不再把 run.yaml 作为通信信封，从而既保留 INCAR/KPOINTS 经验，又保持与 jobflow 生态的结合度。

## 实现阶段

**阶段一：基座贯通与轻量检索落地**
本阶段的目标是在 PC 到 CPU 与 PC 到 GPU 的最小通道贯通的同时，立即提供可审计的在线检索与材料数据库查询工具，并以统一的数据契约输出可被后续 LLM 和人审共用的结构化结果。PC 端新增一个“检索工具层”，包含基于 chromedriver 的文献浏览器与基于 pymatgen 生态的材料数据库适配器。文献浏览器实现为 `retrieval/lit_browser.py`，提供 `search(query: str, site: str|None) -> search.json` 与 `capture(target_url: str) -> page.json` 两个入口，前者在校园网环境下用 Selenium 以 headless Chrome 打开学术搜索引擎并收集题名、作者、期刊、年份、DOI、可用链接与两到三句摘要，后者在合法站点上抓取正文主栏的可读文本与首张图题说明，所有输出用 `evidence_anchor.json` 的固定模式保存 DOI、标题、出处与定位片段的选择器。材料数据库适配器实现为 `retrieval/matdb.py`，以 `mp-api` 的 `MPRester` 为主入口，提供 `get_structures(formula_or_mpid)` 与 `search(props, criteria)` 两类查询，输出 `structures/*.cif` 与 `matdb_hits.json` 并记录查询条件与 API 版本。两个工具都注册到 `tools/registry.yaml`，输入输出用 JSONSchema 明确字段，便于后续作为 MCP 或 LangChain 的标准工具被 LLM 调用。为了贯通计算通道，本阶段保留前述作业包契约与最小 DFT 与 GPU 任务执行面：CPU 与 GPU 端均由 jobflow-remote runner 读取 Flow 并在队列中真实调度作业（CPU 运行 VASP、GPU 运行结构预筛/势能训练）。你现有的 VASP 输入规则脚本只作为策略参考，规则在 PC 侧转化为策略插件注入 Flow，不再下发 `run.yaml`。
本阶段的最小验证包含三个小任务。第一个是 O₂ 的单点能或极少步弛豫以验证 CPU 通道，结束时由 `vasp_summarizer.py` 写出 `work/vasp_summary.json`。第二个是在 GPU 上对盒内 O₂ 进行 MACE 预筛，守护进程正确选择空闲设备并写出 `prescreen.csv` 与可选二维嵌入图，你现有全局搜索脚本中的 AMP 与 TF32 选项在重构后仍可通过任务 JSON 参数化开启。第三个是“文献 + 数据库”的检索回合，文献浏览器以“ZnCr₂O₄ surface reconstruction CO₂/CO 条件”为查询式生成一个 `search.json` 与一份 `page.json`，材料数据库适配器以公式或 mp‑id 拉取 ZnCr₂O₄ 的参考晶体并落地 `structures/*.cif`，进行晶体结构优化；三类任务的日志都写入 `events.jsonl`，以统一格式供 LLM 与人审读取。

**阶段二：简易 LLM 自驱动原型与证据锚点回填**
本阶段在不引入复杂多智能体编排的前提下，完成一个可运行的“单代理原型”。PC 端新增 `agent/proto_planner.py`，它以“高层任务字符串 + 工具集注册表”为输入，按照固定的三步套路自动运行：先调用 `lit_browser.search` 聚合若干候选文献并抽取 DOI 与关键信息，再调用 `matdb.get_structures` 或 `search` 拉取与当前任务相关的结构与物性条目，最后按简单的决策规则选择要发起的计算作业或要继续检索的关键词。原型的决策规则不做黑箱学习，只做可审计的启发式打分，并把每一步的“工具调用输入输出、选择原因与下一步计划”写成 `plan_trace.jsonl`，便于复盘。为保证与后续闭环一致，本阶段统一产出“证据锚点”对象，字段包含 DOI、标题、页码或图号选择器与引用的原句片段，参考报告正文在“知识先验与证据核验”的角色定位，这里先以最小 schema 落地。
本阶段的最小验证选两条路径。其一是“问题驱动检索到计算”的短链条，例如输入“为 ZnCr₂O₄ 在 CO₂/CO 条件下构建工况相图的必要数据”，原型应先以 `lit_browser.search` 找到可复现实验工况与常用控氧对，再用 `matdb.get_structures` 拉取母相结构，随后由人审一键触发 `surface_energy_diagram` 的小网格试跑生成 CSV 与图件，字段与 `core_utils.py` 的化学势实现保持一致。其二是“计算驱动反查证据”的短链条，例如输入“要求给出 O₂ 的基准参考能与推荐设置”，原型应列出近期计算实践的推荐 INCAR 片段与 DOs，回填为策略插件的参数建议并把引用的 DOI 与片段落入锚点清单。

**阶段三：计算侧工具层重构与策略化输入生成**
使用SciToolKG架构，大规模扩展需要的工具和对应的工具连接方式

**阶段四：CATDA 与证据图的深度接入与闭环演进**
本阶段把你的 CATDA 接入作为知识侧的一等公民，统一落到“证据图 + GraphRAG”的形式。接口以 `knowledge/catda_client.py` 实现，提供 `query_entities(chem_query)` 与 `pull_evidence(dois|keywords)` 两类入口，返回以材料、工况、可观测量与图表锚点为节点的最小知识子图，再把该子图登记为作业包的知识产物，与计算侧产物并列。LLM 侧在原型之上增加一个“证据核验回路”，其规则是当计算结果与证据图的可观测量偏差超过阈值时自动生成“再计算或再检索”的计划，并交由人审门控再执行，逐轮压缩不确定度直到收敛。报告正文在技术路线与智能体编排部分已经说明了“知识先验与证据核验”在起点与终点的作用，这里对应为可运行的轻量闭环原型。

**跨阶段的配置与字段约定补充说明**
检索工具层新增 `configs/search.yaml` 与 `~/.orchestrator/secrets.yaml` 的两个区段。前者声明 `chrome_binary`、`driver_path`、`headless`、`timeout_s`、`throttle_ms` 与 `allow_sites` 等参数，用于在校园网与不同 Chrome 版本下稳定运行；如果使用 Selenium Manager 自动管理驱动，`driver_path` 可留空而由运行时解析。后者保存 `MP_API_KEY` 等材料数据库令牌，以及可能的机构内网 HTTP 代理与白名单域。`evidence_anchor.json` 的字段为 `title`、`authors`、`venue`、`year`、`doi`、`url`、`selector`、`quoted_text` 与 `acquired_at`，作为 LLM 侧引用链路的唯一接口。`matdb_hits.json` 的字段为 `query`、`api_version`、`hits[]`，每条 hit 至少包含 `mpid`、`formula`、`spacegroup`、`a,b,c,alpha,beta,gamma` 与可选的 `structure_path`。这些文件都被写入作业包并以 `artifacts.json` 建索引，使检索任务与计算任务一样可追溯。报告正文把“知识库与工具编排”作为智能体框架的中层能力，这里通过统一契约把它们落到可复核的数据对象上。

**阶段验收与最终目标**
阶段一的退出准则是三条最小链路全部可跑并产出统一摘要，文献与材料数据库检索能稳定返回结构化结果并写入锚点对象。阶段二的退出准则是单代理原型可以从自然语言目标出发，独立完成“检索、拉数、决策、发起单步计算”的闭环一次，并把全流程的决策轨迹写入 `plan_trace.jsonl`。阶段三的退出准则是计算侧工具重构完成且与策略插件对齐，GPU 队列在共享服务器条件下连续稳定运行。阶段四的退出准则是 CATDA 的实体与证据能被自动拉取并与计算产物并列入作业包，LLM 侧能基于证据与模型偏差自动给出“再计算或再检索”的计划并通过人审门控执行。最终目标是交付一个可持续演进的研究基座，它把“在线文献与材料库的轻量检索、LLM 原型的自驱闭环、计算侧的稳健执行与证据图的核验”纳入同一作业契约与日志体系，能够在示范体系上复现实验锚点并形成可迁移的工程样板。报告正文对技术路线与年度计划的叙述为该目标提供了方法学支撑与进度锚点，这里对应为可运行的工程化阶段划分与产出清单。

为方便你与同事按阶段实现，我已把需要重写与迁移的现有脚本定位在阶段三的工具层重构中，包括把 `Prepare_vasp_inputs.py` 的规则迁入策略插件并映射到 Maker 参数，把 `adsorbate_generator.py` 规范化为种子生成器并升级元数据版本，把 `global_search.py` 改造成“任务入口 + 可组合内核”的双层结构，以及把 `add_h2_to_pressure.py` 与 `surface_energy_diagram.py` 规整为分析工具并延续 `core_utils.py` 的物理与单位约定。这样既保证通用性与向后兼容，又为阶段一二的检索与 LLM 原型提供直接可用的数据对象。

## 环境与配置

PC端采用Python 3.12与jobflow和atomate2构建Flow，通过Paramiko或等价库进行SSH传输。CPU端提供VASP与最小Python环境以运行flow_runner和custodian。GPU端提供PyTorch与MACE推理或训练环境，并安装pynvml以进行显存探测。SSH私钥建议存放在用户家目录的隐藏目录下并启用known_hosts校验，集群与GPU服务器的地址与账户写入PC侧公共配置文件，敏感项写入用户家目录的私有secrets文件。Slurm模板位于项目的templates目录并由PC侧渲染提交，GPU队列目录在服务器本地固定路径由守护进程轮询，这一配置满足共享服务器的使用习惯并避免与他人产生冲突。

示例的Orchestrator配置可在`configs/orchestrator.yaml`中声明主机地址、端口、用户名、私钥路径、远端工作根目录与GPU队列目录，Slurm分区与账户以及默认资源模板在同一处集中管理。VASP势能库路径等敏感环境变量放在`~/.orchestrator/secrets.yaml`并在运行时注入。该方式可以在不改代码的情况下迁移到新集群或新服务器。

## 与现有脚本的衔接

VASP输入生成逻辑来自Prepare_vasp_inputs脚本中的任务安全默认值与K网格规则，现统一迁移到PC侧的策略插件层，并在Flow构建时注入到atomate2的Maker参数，从而淘汰run.yaml的通信功能。
吸附种生成脚本保留“AdsorbateSeeds”元数据模式与键约束推断，用于保证几何操作的物理合理性，并作为GPU预筛的种子来源。
全局搜索脚本保留MACE推理、AMP与TF32加速、RMSD或SOAP的多样性选择与Minima Hopping功能，接口与摘要对齐作业包契约，从而支持智能体按需组合与调度。
气相填充与表面能相图脚本与核心物性库按工况生成统一摘要与相图网格，作为后续动力学与文献对比的可观测量锚点。

