# 1. 快速安装与启动

[English](01-quickstart.en.md) | [目录](README.zh.md) | [下一章](02-concepts.zh.md)

本章给出一条可复现的本地启动路径。完成后，你应能注册或登录、创建工作区、选择任务入口并得到一次模型回复。远程科学软件不是这一步的前提。

## 1.1 前提

- Linux 主机或 Linux 服务器。
- 可用的 conda 安装。
- 能访问所选 LLM provider 的网络和 API key，或当前系统用户已有 Codex OAuth 凭据。
- 安装期间可访问 conda、pip、npm 和 JSmol 下载源。离线部署见[部署与运维](10-deployment-operations.zh.md)。
- 建议至少预留 20 GB 磁盘空间给 control plane 环境，项目数据和远程回传结果另算。

CatMaster 的 control plane 环境由一个文件管理：

```text
requirements/pc-conda.yml
```

不要用 `requirements/mace.txt`、`requirements/uma.txt`、`requirements/mattersim.txt` 或 `requirements/orb.txt` 替代它。这些是远程 MLFF provider 的隔离环境依赖。

## 1.2 创建环境

在仓库根目录执行：

```bash
conda env create -f requirements/pc-conda.yml
conda activate catmaster
```

更新已有环境：

```bash
conda env update -n catmaster -f requirements/pc-conda.yml
```

确认解释器和 WebUI 命令可用：

```bash
python --version
python -m catmaster.webui --help
```

## 1.3 安装 Literature Review 浏览器

只有使用 Literature Review 的受控浏览器路径时才需要 `agent-browser`，但建议在首次安装时一起完成：

```bash
npm install -g agent-browser@0.31.1
agent-browser install
agent-browser doctor --offline --quick
agent-browser mcp --help
```

CatMaster 自己启动 MCP 子进程。不要把 Codex 的全局 MCP 配置复制进 CatMaster。机构登录、验证码、二维码和 OTP 必须由用户在浏览器中完成，不要把 cookie、密码或浏览器 profile 放进项目空间。

## 1.4 配置第一个模型

复制标准模板：

```bash
cp configs/llm.template.yaml configs/llm.yaml
```

标准模板使用 OpenRouter 模型标签。提供 key：

```bash
export OPENROUTER_API_KEY="<YOUR_KEY>"
```

如果使用其他 provider，请不要只替换 key。按[LLM 与运行时配置](03-llm-configuration.zh.md)修改 `provider`、`model`、角色绑定和 provider 专属字段。

配置文件和 key 可以分开保存。一个实用做法是从清单创建本地文件：

```bash
cp .env.example .env.local
chmod 600 .env.local
```

`.env.local` 被 shell 读取时必须导出变量。由于模板使用 `KEY=value` 格式，请这样加载：

```bash
set -a
source .env.local
set +a
```

程序不会自动读取 `.env.local`。不要提交包含真实 key 的文件。

## 1.5 创建项目根目录

项目根目录用于容纳用户的多个 workspace：

```bash
mkdir -p "$HOME/catmaster_projects"
```

默认启用账号登录时，每个用户的数据会放在这个根目录下的 `users/<username>/`。项目布局详见[系统概念与项目空间](02-concepts.zh.md)。

## 1.6 安全启动 WebUI

显式指定项目根、监听地址和端口：

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
CATMASTER_HOST=127.0.0.1 \
CATMASTER_PORT=7991 \
./start_webui.sh
```

打开：

```text
http://127.0.0.1:7991
```

`start_webui.sh` 默认后台运行。它的内嵌默认值是 `0.0.0.0:7991`，而 `python -m catmaster.webui` 的默认值是 `127.0.0.1:7860`。手册始终显式传值，避免把服务意外暴露到网络，也避免端口混淆。

首次启动可能下载并安装固定版本的 JSmol 资源，用于结构预览。首次启动比后续启动慢是正常现象。

## 1.7 第一次登录和验收

默认启用登录和注册。用户名会转换为小写，允许字母、数字、点、下划线和连字符，长度为 3 到 40；密码长度为 8 到 256。注册页会要求完成简单算术验证码。

登录后执行以下检查：

1. 保留默认 workspace，或新建一个测试 workspace。
2. 新建 thread。
3. 选择 `Experiment`，权限模式先选 `Review`。
4. 发送：`请列出当前项目文件，并说明 files 和 metadata 的用途。不要创建文件。`
5. 在 Chat 中确认收到增量回复，在 Monitor 中确认出现一次 run。

这一步只验证 LLM、线程存储和流式界面。它不会验证集群或科学软件。

## 1.8 日常运维命令

查看状态和日志：

```bash
./start_webui.sh --status
tail -f .runtime/webui.log
```

前台启动以便排错：

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
CATMASTER_HOST=127.0.0.1 \
CATMASTER_PORT=7991 \
./start_webui.sh --foreground
```

停止后台服务：

```bash
./start_webui.sh --stop
```

如果 conda 环境名不同：

```bash
CATMASTER_CONDA_ENV=<ENV_NAME> \
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
CATMASTER_HOST=127.0.0.1 \
CATMASTER_PORT=7991 \
./start_webui.sh
```

## 1.9 仅本机的无登录模式

无登录模式使用开放的 `admin` 空间，并关闭 Skill Evolution。只在可信单机环境使用：

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
./start_webui.sh --foreground --host 127.0.0.1 --port 7991 --no-login
```

不要让 `--no-login` 监听局域网或公网地址。

## 1.10 接下来做什么

- 先读[系统概念与项目空间](02-concepts.zh.md)，再向 workspace 上传真实数据。
- 需要调整模型角色时读[LLM 与运行时配置](03-llm-configuration.zh.md)。
- 需要集群计算时，从[远程机器与任务执行](08-remote-execution.zh.md)开始，不要直接编辑活动私有配置后就提交正式任务。
