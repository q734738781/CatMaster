# CatMaster

CatMaster is a local agent workbench for computational catalysis and materials workflows. Its WebUI opens a thread-centric workspace with project-space files, agent messages, tool cards, artifact previews, interrupts, and result inspection in one view.

CatMaster 是一个面向计算催化和材料工作流的本地 agent 工作台。日常入口是 WebUI v2 工作区：选择项目空间和线程、提交需求，并在同一界面查看文件树、agent 消息、工具卡片、artifact 预览、中断审批和结果。

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

- Prepare and analyze atomistic structures, slabs, adsorbates, VASP/CP2K/LAMMPS/ORCA/xTB inputs, and MACE/UMA workflows.
- Run bounded experiment tasks, broader research planning, literature review, writing, and PDF peer-review style checks.
- Submit prepared calculation stages to remote machines through DPDispatcher when your cluster access is configured.
- Keep user artifacts inside project spaces so threads, reports, intermediate files, remote receipts, and histories stay inspectable.

## Acknowledgements

CatMaster includes and adapts selected Apache-2.0 skills from Yuan Yizhe's
`nature-skills` project for literature, researcher, and academic-writing
workflows: <https://github.com/Yuan1z0825/nature-skills>. The redistributed
license text is kept at [skills/NATURE_SKILLS_LICENSE](skills/NATURE_SKILLS_LICENSE).

## 能力概览

- 结构、表面、吸附物、VASP/CP2K/LAMMPS/ORCA/xTB 输入、MACE/UMA 相关任务的准备与分析。
- 支持计算实验、研究规划、文献综述、写作和 PDF 审稿式检查。
- 配好集群访问后，可通过 DPDispatcher 提交远程计算任务。
- 使用项目空间保存输入、输出、中间文件、线程历史、远程回执和报告，方便继续任务和复查结果。

## 致谢

CatMaster 引入并适配了袁一哲 `nature-skills` 项目中的部分 Apache-2.0
科研 skills，用于文献、researcher 和学术写作工作流：
<https://github.com/Yuan1z0825/nature-skills>。随附 license 文本保存在
[skills/NATURE_SKILLS_LICENSE](skills/NATURE_SKILLS_LICENSE)。

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

## Development WebUI v2 Check

For migration and compatibility work, use the `catmaster-dev` environment with
the same pinned control-plane requirements used by deployment:

```bash
conda activate catmaster-dev
/home/chenhh/miniconda3/envs/catmaster-dev/bin/python -m pip install -r requirements/pc.txt

cd catmaster/webui/frontend
npm install
npm run build

cd ../../..
/home/chenhh/miniconda3/envs/catmaster-dev/bin/python -m pytest \
  tests/test_webui_thread_v2.py tests/test_specialist_runtime.py
```

WebUI v2 is the default built frontend. Legacy run endpoints remain available
for compatibility/debugging, but new UI work should use the thread endpoints
and artifact registry.
