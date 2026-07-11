# CatMaster User Guide

This guide is written for users with basic Linux experience: opening a terminal, running commands, changing directories, and exporting environment variables. It is organized as "get the local WebUI running first, learn the workflow, then configure remote execution only when needed."

## Recommended Order

1. [Local setup](01-local.en.md)
   Install Python dependencies, configure an LLM, prepare a project space, and start the WebUI.
2. [Features and daily workflows](03-features.en.md)
   Learn the WebUI, task lanes, project spaces, run history, and prompt patterns.
3. [Remote setup](02-remote.en.md)
   Read this only when you need cluster submission through DPDispatcher.
4. [External materials programs](04-external-tools.en.md)
   Configure VASPKIT, VESTA, and headless structure-rendering dependencies when needed.

Chinese guide:

- [中文总览](README.zh.md)
- [本地配置要点](01-local.zh.md)
- [远程配置要点](02-remote.zh.md)
- [功能介绍与日常使用](03-features.zh.md)
- [外部材料软件](04-external-tools.zh.md)

## Public Web Demo

A public CatMaster WebUI demo is available at:

```text
https://cm.cccgg.cyou
```

The demo server is configured with the full CatMaster feature set, but compute resources may be limited. Use it to try the interface, workflow, file handling, and agent behavior; run heavier calculations on your own configured machine or cluster.

## Shortest Path

To start the WebUI quickly:

```bash
conda env create -f requirements/pc-conda.yml
conda activate catmaster

cp configs/llm.template.yaml configs/llm.yaml
export OPENROUTER_API_KEY="..."

CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh
```

Then open:

```text
http://127.0.0.1:7990
```

If this fails, start with the troubleshooting section in [Local setup](01-local.en.md).

## Configuration Map

Common files:

- `configs/llm.yaml`: default LLM config, usually copied from `configs/llm.template.yaml`.
- `configs/llm.full.template.yaml`: full field template for all supported providers and config fields.
- `.env.example`: environment-variable checklist. CatMaster does not auto-load it; source a local copy manually or export variables from your shell profile.
- `configs/tool_policy.yaml`: tool allow/block policy.
- `configs/tool_output.yaml`: long tool-output preview and offload policy.
- `configs/dpdispatcher/`: remote-execution config directory, used only for cluster submission.

## Documentation Rule

The root `README.md` stays short and points to this guide. Installation, configuration, and usage details live in this chaptered guide so one README does not become dense and hard to follow.
