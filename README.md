# CatMaster

CatMaster is a local agent workbench for computational catalysis and materials workflows. It provides a WebUI for project spaces, task lanes, file browsing, run history, tool traces, and result inspection.

CatMaster 是一个面向计算催化和材料工作流的本地 agent 工作台。日常入口是 WebUI：选择项目空间、选择任务模式、提交需求，并在同一界面查看文件、运行记录、工具调用和结果。

## Public Web Demo

A public CatMaster WebUI demo is available at:

```text
https://cm.cccgg.cyou
```

This server is configured with the full CatMaster feature set, but its compute resources may be limited. It is best for trying the interface, workflow, file handling, and agent behavior; run heavier calculations on your own configured machine or cluster.

公共 WebUI demo：

```text
https://cm.cccgg.cyou
```

该服务器具备 CatMaster 的完整功能，但算力可能偏低。适合体验界面、工作流、文件管理和 agent 行为；较重的计算任务建议在自己配置的机器或集群上运行。

## What It Can Do

- Prepare and analyze atomistic structures, slabs, adsorbates, VASP/CP2K/LAMMPS/ORCA/xTB inputs, and MACE workflows.
- Run bounded experiment tasks, broader research planning, literature review, writing, and PDF peer-review style checks.
- Submit prepared calculation stages to remote machines through DPDispatcher when your cluster access is configured.
- Keep user artifacts inside project spaces so runs, reports, intermediate files, and histories stay inspectable.

## 能力概览

- 结构、表面、吸附物、VASP/CP2K/LAMMPS/ORCA/xTB 输入、MACE 相关任务的准备与分析。
- 支持计算实验、研究规划、文献综述、写作和 PDF 审稿式检查。
- 配好集群访问后，可通过 DPDispatcher 提交远程计算任务。
- 使用项目空间保存输入、输出、中间文件、运行历史和报告，方便继续任务和复查结果。

## Start Here

For a step-by-step setup, use the user guide series:

- 中文总览：[docs/readme/README.zh.md](docs/readme/README.zh.md)
- 本地配置：[docs/readme/01-local.zh.md](docs/readme/01-local.zh.md)
- 远程配置：[docs/readme/02-remote.zh.md](docs/readme/02-remote.zh.md)
- 功能使用：[docs/readme/03-features.zh.md](docs/readme/03-features.zh.md)

English guide:

- Overview: [docs/readme/README.en.md](docs/readme/README.en.md)
- Local setup: [docs/readme/01-local.en.md](docs/readme/01-local.en.md)
- Remote setup: [docs/readme/02-remote.en.md](docs/readme/02-remote.en.md)
- Features and workflows: [docs/readme/03-features.en.md](docs/readme/03-features.en.md)

## Minimal Local Launch

```bash
conda create -n catmaster python=3.11
conda activate catmaster
pip install -r requirements/pc.txt

cp configs/llm.template.yaml configs/llm.yaml
export OPENROUTER_API_KEY="..."

mkdir -p ~/catmaster_projects
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh
```

Then open:

```text
http://127.0.0.1:7990
```

If you need a different provider, remote execution, optional external programs, or troubleshooting notes, follow the full guide above.

如果你使用其他模型供应商、远程计算、可选外部程序，或需要排查启动问题，请按上面的中文分章指南操作。
