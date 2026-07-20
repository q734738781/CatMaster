# CatMaster 用户手册

[English](README.en.md) | 中文

本手册面向实际使用和管理 CatMaster 的用户。内容以当前 DeepAgent specialist 运行时和 WebUI v2 为准，覆盖本地安装、模型配置、项目空间、五类任务入口、计算模块、远程机器连接、文件与运行管理、自进化、部署和故障排查。

核对日期：2026-07-20。

如果只想先启动系统，请从[快速安装与启动](01-quickstart.zh.md)开始。如果要接入集群，请先完成本地验证，再阅读[远程机器与任务执行](08-remote-execution.zh.md)。

## 手册地图

| 章节 | 解决的问题 |
|---|---|
| [1. 快速安装与启动](01-quickstart.zh.md) | 建环境、配置第一个模型、安全启动 WebUI |
| [2. 系统概念与项目空间](02-concepts.zh.md) | 理解 control plane、workspace、thread、run、artifact 和目录边界 |
| [3. LLM 与运行时配置](03-llm-configuration.zh.md) | 配置 provider、角色模型、API key、推理参数和输出策略 |
| [4. WebUI 操作手册](04-webui.zh.md) | 登录、工作区、线程、附件、审批、Monitor、Files |
| [5. Agent 与模块功能](05-agents-and-modules.zh.md) | 选择 Research、Experiment、Writing、Peer Review、Literature Review |
| [6. 计算与建模工作流](06-computational-workflows.zh.md) | 结构、DFT、MD、MLFF、分子计算和结果核查 |
| [7. 文献、写作与审稿](07-literature-writing-review.zh.md) | 文献检索、语料库、论文写作、润色和多模型审稿 |
| [8. 远程机器与任务执行](08-remote-execution.zh.md) | SSH、Slurm、DPDispatcher、资源卡、stage、receipt 和恢复 |
| [9. 工具、技能与自进化](09-tools-skills-evolution.zh.md) | 工具权限、skills、项目级改进、审批和回滚 |
| [10. 部署、运维与安全](10-deployment-operations.zh.md) | 服务器部署、SSH 隧道、备份、升级和外部程序 |
| [11. 参考与故障排查](11-reference-troubleshooting.zh.md) | 环境变量、任务矩阵、限制、诊断顺序和验收清单 |

## 三条使用原则

1. 用户和 agent 共同维护的内容放在项目空间的 `files/`。`metadata/` 保存线程、checkpoint、运行记录和内部索引，不要把它当普通文件目录编辑。
2. VASP、CP2K、LAMMPS、ORCA、xTB、CREST 和受管 MLFF 任务使用已登记的远程执行通道。CatMaster 不会在缺少远程配置时静默改成本地运行。
3. 计算完成不等于结果已经可靠。结构、参数、收敛、日志、物理合理性和回传文件仍需核查。

## 文档约定

- 命令默认从仓库根目录执行。
- 示例端口统一使用 `7991`，并显式绑定 `127.0.0.1`。启动脚本自己的内嵌默认值可能面向服务器部署，因此不要依赖隐式地址。
- `project path` 指宿主机上的真实路径；`workspace path` 指 agent 在 `files/` 中看到的相对路径。
- `<LIKE_THIS>` 是必须替换的站点值。示例中的 token、主机名、用户名和路径都不是可直接使用的凭据。
- 集群队列、核数、GPU 数和程序路径只是模板起点，必须按实际机器修改。

## 支持边界

CatMaster 负责组织任务、生成和检查文件、调用注册工具、保留执行证据，并在已配置的机器上提交任务。它不提供 VASP、ORCA 等商业软件许可证，也不替代集群账号、网络权限、势函数授权、机构订阅或科研判断。
