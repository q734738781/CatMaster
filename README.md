# CatMaster

[中文](#从研究目标出发) | [English](#start-from-a-research-objective)

CatMaster 是一个面向计算催化、材料建模、文献研究和科研写作的自主 Agent 工作台。它把对话、项目文件、领域 skills、可执行 tools、人工审批和受管远程计算放在同一个 workspace 中。用户可以从研究目标出发，不必先把工作拆成一串工具调用。

CatMaster is an autonomous agent workbench for computational catalysis, materials modeling, literature research, and scientific writing. It brings conversations, project files, domain skills, executable tools, approvals, and managed remote computation into one workspace. Users can begin with a research objective instead of scripting a sequence of tool calls.

## 从研究目标出发

CatMaster 可以直接从研究目标安排工作。Agent 会根据现有文件、科学约束和当前部署能力选择 skill，调用 tool，并检查中间产物。用户仍然决定会改变科学含义、花费远程算力或影响重要文件的事项。

WebUI 提供五个研究入口：

| Agent | 主要角色 | 能力来源举例 | 适合交付 |
|---|---|---|---|
| Research | 拆解开放研究目标并下发实际执行 | 可委派 Literature Review、Experiment、Writing、Peer Review，并读取项目文件、记忆与返回产物 | 已执行的文献、计算、写作和审稿阶段，证据地图与跨阶段结论 |
| Experiment | 组织有边界的建模、计算和验证 | Materials、Dynamics、ML、ORCA/xTB workers，以及结构、计算、轨迹和远程执行 skills | 结构候选、计算 stage、数据集、轨迹分析和结果合同 |
| Literature Review | 从论文发现走到可追溯证据 | 网页检索、受控浏览、本地语料、全文精读和引用核验 tools 与 skills | 检索记录、证据表、双语精读、引用库和综述 |
| Writing | 把已有证据组织成科研交付物 | 论文写作、润色、引用、数据声明、图件、PPT、投稿回复和专利 skills | Markdown、LaTeX、DOCX、PPTX、图件和 PDF |
| Peer Review | 独立审查一份固定稿件 | 多 reviewer 模型、审稿 worker 和 `peer_review_request` | 原始 reviewer reports、editor synthesis 和修订问题单 |

Research 会负责方向判断，也会继续把阶段任务交给可执行的 specialist。在用户授权的范围内，一条 Research thread 可以从文献证据缺口推进到结构建模、远程计算、结果复核、写作和独立审稿。具体领域操作由拥有相应 tools 与 skills 的 specialist 或 worker 完成；结果返回同一 workspace 后，Research 可以核对证据、补发下一阶段任务或收束结论。对于包含竞争解释或共享证据的问题，Research 还可以运行持久假设 campaign。专门的 hypothesis proposer 负责提出和修订可证伪假设，执行 specialist 完成验证，独立 evidence judge 判断结果支持、反对还是无法区分各目标假设。确定性 controller 只校验和保存科学状态并给出排序，不替代任何科学角色。WebUI 的 Research Map 点击会为所选验证建立一条完整、独立的普通 Research thread；Automatic Research 则由异步 worker 串行建立同样的 thread。每条执行 thread 有自己的 Research Kernel 和 checkpoint，只共享父 campaign 的科学状态。原有单 thread Research 请求和普通线性任务保持不变。

Experiment 下的四类 worker 进一步分工：Materials 负责晶体、表面、吸附、缺陷、反应路径和性质计算；Dynamics 负责 AIMD、LAMMPS、MLFF MD、restart 与轨迹；ML 负责数据集、MACE 训练评估和主动学习；ORCA/xTB 负责分子、构象、xTB、CREST、ORCA、TS、IRC、TDDFT 与 NMR。

完整说明和可直接改写的参考 prompt 见[中文用户手册](docs/user-guide/README.zh.md)。手册先解释 Agent 能完成的研究工作，再在可展开区域列出当前 tools 与 skills；远程 task 独立成章，不与结构准备混为一谈。

## 快速启动

```bash
conda env create -f requirements/pc-conda.yml
conda activate catmaster

cp -n configs/llm.template.yaml configs/llm.yaml
export OPENROUTER_API_KEY="<YOUR_KEY>"

mkdir -p "$HOME/catmaster_projects"
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
CATMASTER_HOST=127.0.0.1 \
CATMASTER_PORT=7991 \
./start_webui.sh
```

打开 `http://127.0.0.1:7991`，新建 workspace 和 thread，把权限模式设为 Review。在 Files 上传一份 CIF 或 POSCAR；源码安装可以使用 `tests/assets/Fe.cif`。然后先做一个不提交远程计算的任务：

```text
使用 Experiment 检查我刚上传的晶体结构。
识别文件路径、材料、晶胞、元素和约束，检查周期边界下的异常短距，
生成一份结构审计报告并说明后续可以开展哪些建模。
不要修改原文件，也不要查询或提交远程任务。
```

如果管理员已经提供了 CatMaster 地址，可直接从[第一次进入 WebUI](docs/user-guide/01-quickstart.zh.md#第一次进入-webui)开始。模型路由、服务器部署和外部程序配置集中在[第 10 章](docs/user-guide/10-deployment-operations.zh.md)。

## 中文手册路线

- [CatMaster 如何组合 Agent、worker、skill 与 tool](docs/user-guide/02-concepts.zh.md)
- [五类 Agent 的角色、能力来源和参考 prompt](docs/user-guide/03-llm-configuration.zh.md)
- [Experiment 与四类计算 worker](docs/user-guide/05-agents-and-modules.zh.md)
- [表面、吸附、缺陷、动力学、MLFF 与分子计算](docs/user-guide/06-computational-workflows.zh.md)
- [文献、写作与审稿 Agent](docs/user-guide/07-literature-writing-review.zh.md)
- [远程 task、receipt、停止与恢复](docs/user-guide/08-remote-execution.zh.md)
- [Prompt 库与故障排查](docs/user-guide/11-reference-troubleshooting.zh.md)

## Start from a research objective

CatMaster can plan work directly from a research objective. An agent interprets the existing files, scientific constraints, and deployment capabilities before selecting skills and calling tools. The user retains control over choices that change the scientific question, consume remote compute, or affect important files.

The WebUI exposes five research entries:

| Agent | Role | Capability sources | Typical deliverables |
|---|---|---|---|
| Research | Decomposes open objectives and dispatches real execution | Literature Review, Experiment, Writing, and Peer Review specialists, plus project files, memory, and returned artifacts | Completed literature, computation, writing, and review stages; evidence maps; cross-stage conclusions |
| Experiment | Organizes bounded modeling, computation, and validation | Materials, Dynamics, ML, and ORCA/xTB workers with structure, calculation, trajectory, and remote-execution skills | Candidate structures, calculation stages, datasets, trajectory analyses, result contracts |
| Literature Review | Moves from discovery to traceable evidence | Web search, controlled browsing, local corpora, full-paper reading, and citation-verification tools and skills | Search records, evidence tables, readers, reference libraries, reviews |
| Writing | Turns existing evidence into scientific deliverables | Manuscript, polishing, citation, data, figure, slide, response, and patent skills | Markdown, LaTeX, DOCX, PPTX, figures, PDF |
| Peer Review | Independently assesses one fixed manuscript | Multiple reviewer models, a review worker, and `peer_review_request` | Raw reviewer reports, editor synthesis, revision issue lists |

Research decides how to advance an open objective and dispatches stages to specialists that can execute them. Within the authority granted by the user, one Research thread can move from a literature evidence gap to structure modeling, remote computation, result checks, writing, and independent review. For questions with competing explanations or shared evidence, Research can run a persistent hypothesis campaign. A dedicated hypothesis proposer forms and revises falsifiable hypotheses, an execution specialist performs each verification, and an independent evidence judge decides whether the result supports, opposes, or fails to distinguish every target hypothesis. The deterministic controller only validates and persists scientific state and ranks available checks. Clicking a WebUI Research Map action creates a complete, independent ordinary Research thread; Automatic Research uses an asynchronous worker to create the same threads serially. Every execution thread has its own Research Kernel and checkpoint while sharing only the source campaign's scientific state. Existing single-thread Research requests and ordinary linear work remain unchanged.

Experiment delegates crystal, surface, adsorption, defect, path, and property work to Materials; AIMD, LAMMPS, MLFF MD, restart, and trajectory work to Dynamics; datasets, MACE, and active learning to ML; and molecular, conformer, xTB, CREST, ORCA, TS, IRC, TDDFT, and NMR work to ORCA/xTB.

The [English user manual](docs/user-guide/README.en.md) describes these capabilities as connected research work. Exact tool and skill names remain available in expandable reference sections. Remote tasks have a separate chapter because preparing a valid calculation and executing it on a configured machine are different capabilities.

## English manual paths

- [Quick installation and first conversation](docs/user-guide/01-quickstart.en.md)
- [How agents, workers, skills, and tools fit together](docs/user-guide/02-concepts.en.md)
- [Roles, capability sources, and prompts for the five agents](docs/user-guide/03-llm-configuration.en.md)
- [Experiment and its four computation workers](docs/user-guide/05-agents-and-modules.en.md)
- [Modeling and computation capabilities](docs/user-guide/06-computational-workflows.en.md)
- [Literature, Writing, and Peer Review agents](docs/user-guide/07-literature-writing-review.en.md)
- [Remote tasks, receipts, stopping, and recovery](docs/user-guide/08-remote-execution.en.md)
- [Prompt library and troubleshooting](docs/user-guide/11-reference-troubleshooting.en.md)

The launch example binds a local installation to loopback. A shared deployment needs authentication, access control, and the operational configuration described in the manual.

## Demo / 在线演示

A hosted demo may be available at `https://cm.cccgg.cyou`. Availability and compute capacity depend on the current deployment. Use a local or institution-managed installation for private data and substantial calculations.

在线 Demo 可能位于 `https://cm.cccgg.cyou`。在线状态和可用算力取决于当前部署。私有数据和正式计算应使用本地或机构管理的实例。

## Acknowledgements and third-party software / 致谢与第三方软件

CatMaster 的主体代码采用 [Apache License 2.0](LICENSE)。以下项目为仓库中的 skill、参考材料或运行组件提供了直接来源。相关作者与许可证仍归各上游项目所有。

CatMaster's main code is released under the [Apache License 2.0](LICENSE). The projects below directly supply skills, source material, or runtime components. Their authorship and license terms remain with the upstream projects.

| Project or contributor | Used in CatMaster | Attribution and license |
|---|---|---|
| Yuan Yizhe's [`nature-skills`](https://github.com/Yuan1z0825/nature-skills) | Selected literature, reading, citation, writing, data, figure, review, response, and presentation skills | Adapted under Apache-2.0; the redistributed license is at [`skills/NATURE_SKILLS_LICENSE`](skills/NATURE_SKILLS_LICENSE) |
| Siqi Chen's [`Humanizer`](https://github.com/blader/humanizer) | Runtime writing-quality skill used for prose-heavy deliverables | MIT; the unmodified upstream `SKILL.md` and license are retained at [`skills/writing_quality/humanizer`](skills/writing_quality/humanizer) |
| 十五 (JL Lab), [`research-pipeline`](https://github.com/Jiahao8595/research-pipeline) | `researchwrite`, experiment logging, and related research workflow material | MIT metadata retained in the relevant skills; see the [`researchwrite` README](skills/research_specialist/nature-proposal-writer/README.md) |
| [`snipp-zha/Paper-to-patent-Skill`](https://github.com/snipp-zha/Paper-to-patent-Skill) | Evidence-grounded Chinese patent drafting skill | Contributor and source retained in the local [`nature-paper-to-patent` README](skills/research_specialist/nature-paper-to-patent/README.md) |
| [`figures4papers`](https://github.com/ChenLiu-1996/figures4papers) and [Peng Sida's research notes](https://github.com/pengsida/learning_research) | Figure patterns and scientific-writing references | Source notes are retained in the local [`nature-figure`](skills/research_specialist/nature-figure/README.md) and [`nature-writing`](skills/research_specialist/nature-writing/README.md) documentation |
| [K-Dense scientific agent skills](https://github.com/K-Dense-AI/scientific-agent-skills) | Scientific writing, visualization, citation management, and venue guidance | MIT upstream; author metadata is retained in the bundled skill files |
| baihe26 | Institution-authorized literature download workflow | MIT; local notice at [`skills/litreview_agent/nature-downloader/LICENSE`](skills/litreview_agent/nature-downloader/LICENSE) |

The WebUI installs pinned JSmol 16.3.13 assets from the [official Jmol package](https://sourceforge.net/projects/jmol/) for interactive structure and trajectory previews. Jmol/JSmol is distributed upstream under LGPLv2. JSmol is a viewer only; calculation engines and remote execution continue to work if its preview assets are unavailable.

WebUI 会从 [Jmol 官方发布包](https://sourceforge.net/projects/jmol/)安装固定版本的 JSmol 16.3.13，用于交互式预览结构和轨迹。Jmol/JSmol 的上游许可证为 LGPLv2。JSmol 只负责预览；其资源缺失不会影响计算引擎或远程任务。

Core dependencies include DeepAgents, LangGraph, LangChain, FastAPI, Pydantic, React, assistant-ui, ASE, pymatgen, RDKit, and DPDispatcher. The exact Python and frontend dependency lists are maintained in [`requirements/pc-conda.yml`](requirements/pc-conda.yml) and [`catmaster/webui/frontend/package.json`](catmaster/webui/frontend/package.json).

核心依赖包括 DeepAgents、LangGraph、LangChain、FastAPI、Pydantic、React、assistant-ui、ASE、pymatgen、RDKit 和 DPDispatcher。完整的 Python 与前端依赖分别以 [`requirements/pc-conda.yml`](requirements/pc-conda.yml) 和 [`catmaster/webui/frontend/package.json`](catmaster/webui/frontend/package.json) 为准。

VASP, ORCA, CP2K, LAMMPS, xTB, CREST, VESTA, VASPKIT, and other scientific programs are configured separately. They are not licensed by CatMaster, and each deployment remains responsible for the applicable upstream license, citation, and site policy.

VASP、ORCA、CP2K、LAMMPS、xTB、CREST、VESTA、VASPKIT 等科学软件需要单独配置，不随 CatMaster 获得许可。部署者需要遵守各软件的许可证、引用要求和所在机构的使用规则。
