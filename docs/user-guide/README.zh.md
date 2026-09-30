# CatMaster 用户手册

[English](README.en.md) | 中文

CatMaster 是一个面向计算催化、材料建模和科研写作的自主 Agent 工作台。你不需要先学会一串工具名，也不需要把研究过程拆成几十条命令再逐条发送。更常见的用法是把研究目标、已有材料和必须遵守的科学约束交给合适的 Agent，让它选择相关 skill，调用可以真正读写文件和处理数据的 tool，并把过程和结果留在项目空间里。

这种自主性不等于完全放手。CatMaster 会自行完成许多技术判断，例如选择结构检查工具、组织中间文件或决定先读哪份结果；遇到会改变科学含义、消耗远程算力或覆盖重要文件的选择时，你仍然可以要求它先比较方案、解释依据并等待确认。本手册的重点就是帮助你理解这条协作边界，以及系统究竟能把哪些研究工作做到什么程度。

## 从哪一章开始

第一次使用时，依次阅读下面四章即可：

1. [快速安装与第一次对话](01-quickstart.zh.md)带你启动 WebUI，并用一个不会提交计算的任务验证系统。
2. [CatMaster 如何工作](02-concepts.zh.md)解释 Agent、worker、skill、tool 和远程 task 如何组合。读完这一章，再看功能列表时就不会把 CatMaster 误解成普通聊天机器人或工具菜单。
3. [五类 Agent](03-llm-configuration.zh.md)介绍 Research、Experiment、Literature Review、Writing 和 Peer Review 各自负责什么，也说明它们之间怎样交接。
4. [WebUI 使用指南](04-webui.zh.md)说明如何组织项目、上传材料、观察 Agent 工作、审阅文件和在运行中补充方向。

之后按研究任务选读：

| 你正在做的事 | 建议阅读 |
|---|---|
| 建表面、放吸附物、做缺陷、准备或分析 VASP/CP2K | [Experiment Agent 与四类 Worker](05-agents-and-modules.zh.md)，再读[建模与计算能力](06-computational-workflows.zh.md) |
| 做 AIMD、LAMMPS、MLFF MD、轨迹或扩散分析 | [Dynamics worker](05-agents-and-modules.zh.md#dynamics-worker原子动力学与轨迹)和[动力学能力](06-computational-workflows.zh.md#从初始结构到可分析轨迹) |
| 整理训练集、训练 MACE、做主动学习 | [ML worker](05-agents-and-modules.zh.md#ml-worker数据集训练与主动学习) |
| 做构象搜索、xTB、ORCA、TS、IRC、NMR | [ORCA/xTB worker](05-agents-and-modules.zh.md#orcaxtb-worker分子与量子化学) |
| 查文献、精读论文、建证据表、整理引用 | [文献、写作与审稿 Agent](07-literature-writing-review.zh.md) |
| 写论文、润色、作图、做 PPT、回复审稿人或写专利 | [Writing Agent](07-literature-writing-review.zh.md#writing-agent把已有证据变成可交付文稿) |
| 把准备好的任务提交到集群或 GPU 服务器 | [远程 Task](08-remote-execution.zh.md) |
| 继续长期项目、管理结果、保留项目经验 | [项目文件与连续工作](09-tools-skills-evolution.zh.md) |
| 安装、配置模型、接入服务器或管理多人部署 | [安装、模型配置与部署](10-deployment-operations.zh.md) |
| 想直接复制一个参考 prompt，或正在排查问题 | [Prompt 参考与故障排查](11-reference-troubleshooting.zh.md) |

## CatMaster 的能力版图

下面这张表只给出入口。各章会继续说明 Agent 如何组合 tools 和 skills，而不是把这些能力写成互不相干的按钮。

| 能力域 | CatMaster 可以参与的工作 | 常见交付物 |
|---|---|---|
| 材料发现与结构建模 | 检索体相结构，建立基准结构，生成超胞、表面、终止面、缺陷、掺杂、吸附位和反应路径 | POSCAR/CIF/XYZ、候选结构集、位点清单、结构审计、可复现脚本 |
| 第一性原理与性质计算 | 准备并检查 VASP、CP2K 输入，组织 relax、static、频率、能带、DOS、声子、弹性、NEB 与热力学校正 | 计算 stage、参数说明、收敛检查、能垒或性质表、分析报告 |
| 动力学 | 准备 CP2K AIMD、LAMMPS 和 MLFF MD，处理 restart，检查轨迹健康并分析 MSD、RDF、扩散和结构演化 | 输入与 restart stage、轨迹、健康检查、时间序列、扩散与配位分析 |
| 机器学习势 | 整理 VASP 结果为训练数据，划分数据集，训练或微调 MACE，做独立评估和主动学习候选筛选 | extxyz 数据集、固定划分、训练配置、checkpoint、误差与候选报告 |
| 分子与量子化学 | 从 SMILES 或结构生成三维分子，搜索和筛选构象，运行 xTB/CREST/ORCA，并处理频率、热化学、TS、IRC、TDDFT 和 NMR | 构象集合、ORCA/xTB stage、优化结构、频率与热化学结果、反应路径 |
| 文献与证据 | 多源发现论文，使用受控浏览器访问合法全文，导入本地语料，精读、去重、核对元数据并建立 claim-evidence 对应 | 检索记录、全文语料、证据表、双语精读、BibTeX/RIS/ENW、综述 |
| 科研写作与传播 | 起草和重构论文、润色学术英语、补充引用、制作图件和 PPT、整理数据声明、回复审稿意见、准备专利草稿 | Markdown/LaTeX/DOCX/PPTX、图件、PDF、response letter、专利文件 |
| 独立审稿 | 让多个 reviewer 模型分别检查同一份 canonical PDF，再汇总共识、分歧和投稿风险 | reviewer reports、editor synthesis、修订问题单 |
| 研究统筹 | 在一个开放目标中串联文献、计算、写作和审稿，保留假设、证据缺口、已完成产物和下一步判断 | 研究计划、阶段产物、证据综合、限制说明、可继续的项目状态 |

## 怎样读手册中的工具名和 skill 名

正文会优先描述用户能完成的研究工作。确实需要知道底层能力来源时，可以展开每节的"当前 tools 与 skills"部分。

- Tool 是 Agent 可以实际执行的动作，例如生成 slab、枚举吸附位、准备 VASP 输入、读取 PDF、分析轨迹或提交远程任务。
- Skill 是一个领域工作方法。它告诉 Agent 何时该用哪些动作、要检查什么、产物怎样组织，以及哪些结果不能过度解释。
- Worker 是带有特定 tool 权限和 skills 的领域执行者。Experiment Agent 会根据目标把工作交给 Materials、Dynamics、ML 或 ORCA/xTB worker。
- Remote task 是管理员登记好的计算合同。它把一个合格的本地 stage 送到指定服务器，执行受管科学程序，再收回结果和运行凭据。

用户通常不需要在 prompt 里指定具体 tool。说明科学目标、输入、必须保留的约束、允许的计算范围和希望得到的产物，Agent 会自行选择能力。只有在复现既有流程、核对远程 task 或明确要求某种方法时，才有必要点名工具或 task。

## 使用边界

CatMaster 可以组织任务、处理文件、生成和检查输入、调用注册工具、提交已配置的远程计算并保留证据。它不会替你获得 VASP 或 ORCA 许可证，也不会绕过机构登录、集群权限或论文付费墙。模型给出的科学判断、自动生成的结构和计算结果仍需要领域审查，特别是电荷、自旋、约束、能量基准、收敛、训练域外预测和反应路径等会直接影响结论的部分。

本手册以当前 DeepAgent specialist runtime 和 WebUI v2 为准，核对日期为 2026-07-20。
