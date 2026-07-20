# CatMaster

CatMaster is a local, project-space-based agent workbench for computational
catalysis and materials research. Its current user surface is WebUI v2, with
persistent threads, files, artifacts, tool activity, approvals, observability,
and managed remote execution in one workspace.

CatMaster 是一个面向计算催化与材料研究的本地 agent 工作台。当前用户入口为
WebUI v2，围绕持久化 thread、项目文件、artifact、工具过程、人工审批、运行观测和
受管远程计算组织工作。

## Main capabilities / 主要能力

- `Research`: coordinates open goals across literature, computation, writing,
  and review.
- `Experiment`: delegates structures, VASP/CP2K/LAMMPS, dynamics, ML and MLFF,
  ORCA, xTB, and CREST work to domain workers.
- `Writing`: drafts and revises evidence-grounded manuscripts, figures, and
  compiled documents.
- `Peer Review`: produces independent reviewer reports and an editor synthesis
  for one canonical PDF.
- `Literature Review`: combines web search, controlled browsing, local corpora,
  evidence tables, and citation finalization.
- DPDispatcher connects registered tasks to site-managed SSH, Slurm, Shell, CPU,
  GPU, and MLFF environments.

- `Research`：协调跨文献、计算、写作与审稿的开放研究目标。
- `Experiment`：将结构、VASP/CP2K/LAMMPS、动力学、ML/MLFF、ORCA、xTB 和
  CREST 任务交给领域 worker。
- `Writing`：基于已有证据起草和修改论文、图件与编译文档。
- `Peer Review`：对一份 canonical PDF 生成独立 reviewer 报告和 editor 综合。
- `Literature Review`：组合网页搜索、受控浏览器、本地语料、证据表和引用定稿。
- DPDispatcher 将注册任务连接到站点管理的 SSH、Slurm、Shell、CPU、GPU 和
  MLFF 环境。

## Quick start / 快速启动

```bash
conda env create -f requirements/pc-conda.yml
conda activate catmaster

cp configs/llm.template.yaml configs/llm.yaml
export OPENROUTER_API_KEY="<YOUR_KEY>"

mkdir -p "$HOME/catmaster_projects"
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
CATMASTER_HOST=127.0.0.1 \
CATMASTER_PORT=7991 \
./start_webui.sh
```

Open / 打开：

```text
http://127.0.0.1:7991
```

The explicit host and port are intentional. The launch script has server-oriented
embedded defaults, while the direct Python CLI has another default port. Keep a
local installation on loopback unless a protected deployment has been designed.

这里显式设置 host 和 port 是有意的。启动脚本与 Python CLI 的隐式默认值不同；除非
已经配置受保护的服务器部署，本地使用时应只监听 loopback。

## User manual / 用户手册

- [中文用户手册](docs/user-guide/README.zh.md)
- [English user manual](docs/user-guide/README.en.md)
- [快速安装与启动](docs/user-guide/01-quickstart.zh.md) / [Quick start](docs/user-guide/01-quickstart.en.md)
- [WebUI 操作](docs/user-guide/04-webui.zh.md) / [WebUI guide](docs/user-guide/04-webui.en.md)
- [模块功能](docs/user-guide/05-agents-and-modules.zh.md) / [Agents and modules](docs/user-guide/05-agents-and-modules.en.md)
- [远程机器与任务执行](docs/user-guide/08-remote-execution.zh.md) / [Remote execution](docs/user-guide/08-remote-execution.en.md)
- [部署与运维](docs/user-guide/10-deployment-operations.zh.md) / [Deployment and operations](docs/user-guide/10-deployment-operations.en.md)
- [参考与排障](docs/user-guide/11-reference-troubleshooting.zh.md) / [Reference and troubleshooting](docs/user-guide/11-reference-troubleshooting.en.md)

The manual is the source of truth for user operation. The short files under
`docs/readme/` remain only as compatibility links for older URLs.

用户操作以新手册为准。`docs/readme/` 下的短文只保留旧链接兼容入口。

## Demo

A hosted demo may be available at:

```text
https://cm.cccgg.cyou
```

Availability and compute capacity depend on the current deployment. Use a local
or institution-managed installation for private data and substantial
calculations.

公共 demo 可能部署在上述地址，在线状态和算力取决于当前服务器。私有数据和较重计算
应使用本地或机构管理的部署。

## Acknowledgements / 致谢

CatMaster includes and adapts selected Apache-2.0 skills from Yuan Yizhe's
[`nature-skills`](https://github.com/Yuan1z0825/nature-skills) project for
literature and academic-writing workflows. The redistributed license is kept at
[`skills/NATURE_SKILLS_LICENSE`](skills/NATURE_SKILLS_LICENSE).

CatMaster 引入并适配了袁一哲
[`nature-skills`](https://github.com/Yuan1z0825/nature-skills) 项目中的部分
Apache-2.0 skill，用于文献和学术写作工作流。随附 license 位于
[`skills/NATURE_SKILLS_LICENSE`](skills/NATURE_SKILLS_LICENSE)。
