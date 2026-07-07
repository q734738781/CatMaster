# CatMaster 用户文档

这组文档面向有 Linux 基础的用户：能进入终端、会 `cd`、会执行命令、知道环境变量是什么即可。文档按“先本地跑起来，再理解功能，最后按需配置远程计算”的顺序组织。

## 推荐阅读顺序

1. [本地配置要点](01-local.zh.md)
   安装 Python 环境、配置 LLM、准备项目空间并启动 WebUI。
2. [功能介绍与日常使用](03-features.zh.md)
   了解 WebUI、任务模式、项目空间、运行历史和常见提示词写法。
3. [远程配置要点](02-remote.zh.md)
   只有需要提交集群任务时再读，覆盖 DPDispatcher 的机器、资源和任务配置。

英文版：

- [English overview](README.en.md)
- [Local setup](01-local.en.md)
- [Remote setup](02-remote.en.md)
- [Features and workflows](03-features.en.md)

## 公共 Web Demo

公共 CatMaster WebUI demo 可在这里访问：

```text
https://cm.cccgg.cyou
```

该服务器具备 CatMaster 的完整功能，但算力可能偏低。建议用于体验界面、工作流、文件管理和 agent 行为；较重的计算任务请在自己配置的机器或集群上运行。

## 最短路径

只想先把 WebUI 跑起来，可以先做这几步：

```bash
conda env create -f requirements/pc-conda.yml
conda activate catmaster

cp configs/llm.template.yaml configs/llm.yaml
export OPENROUTER_API_KEY="..."

CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh
```

然后打开：

```text
http://127.0.0.1:7990
```

如果这里失败，优先看 [本地配置要点](01-local.zh.md) 的排查部分。

## 配置文件地图

常见文件和用途：

- `configs/llm.yaml`：默认 LLM 配置文件，通常由 `configs/llm.template.yaml` 复制得到。
- `configs/llm.full.template.yaml`：完整字段模板，适合查所有 provider 和字段写法。
- `.env.example`：环境变量清单模板；程序不会自动读取它，需要你手动 `source` 或把变量写进 shell profile。
- `configs/tool_policy.yaml`：工具允许/禁用策略。
- `configs/tool_output.yaml`：长工具输出的预览和落盘策略。
- `configs/dpdispatcher/`：远程计算配置目录；只在需要集群提交时使用。

## 文档维护原则

根目录 `README.md` 只保留能力介绍和入口链接。具体安装、配置和使用细节放在本目录的分章文档里，避免所有信息堆在一个 README 中。
