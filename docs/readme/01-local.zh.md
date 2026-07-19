# 本地配置要点

本章目标：在一台 Linux 机器上启动 CatMaster WebUI，并能让 agent 调用一个可用的 LLM。远程计算、VASP、MACE 等外部程序不是本章的前提。

## 1. 准备 Python 环境

建议使用独立 conda 环境：

```bash
conda env create -f requirements/pc-conda.yml
conda activate catmaster
```

`requirements/pc-conda.yml` 是 PC/control-plane 环境的唯一入口：科学/材料栈交给 conda 求解，精确 pin 的 LLM/WebUI pip 包也内联在同一个文件里。不要直接用 pip 安装科学栈。

如果这台机器还要跑本地 MACE 任务，再额外安装 MACE provider 依赖。`requirements/mace.txt` 不是完整 WebUI/agent 环境，不能替代 `requirements/pc-conda.yml`：

```bash
pip install -r requirements/mace.txt
```

如果需要重新构建 WebUI 前端或运行部署脚本，确认 Node.js 和 npm 可用：

```bash
node -v
npm -v
```

### 安装 LitReview 浏览器 MCP

Literature Review lane 使用 `agent-browser` 提供受控浏览器、动态网页和用户已授权的机构访问会话。Node.js 需要 22 或更高版本；`requirements/pc-conda.yml` 已在 CatMaster 环境中包含兼容的 Node 工具链。

```bash
npm install -g agent-browser@0.31.1
agent-browser install
agent-browser doctor --offline --quick
agent-browser mcp --help
```

CatMaster 会自行启动 MCP 子进程，不要把 Codex 的全局 MCP 配置复制到 CatMaster。LitReview 打开页面时会按配置启动受控 Chrome，或复用当前已运行的 Chrome。用户可能已经处于校园网、机构代理或已登录的浏览器 profile 中，此时直接打开 DOI/出版商页面就可能获得全文；不要仅因检索结果没有开放获取链接就默认全文不可访问。需要交互式机构登录时，由用户本人在受控 Chrome profile/session 中完成；密码、cookie、OTP、session 导出和浏览器状态文件都不能写入 YAML、环境变量模板、prompt 或项目空间。浏览器 profile 属于本机状态，不会进入部署包。

可选的本地会话配置：

```bash
export CATMASTER_AGENT_BROWSER_PROFILE="$HOME/.config/catmaster/browser-profile"
export CATMASTER_AGENT_BROWSER_HEADED=true
# 或复用当前已运行的 Chrome：
export CATMASTER_AGENT_BROWSER_AUTO_CONNECT=true
```

首次机构登录通常需要 headed 模式。遇到 CAPTCHA、二维码、短信/OTP、安全警告或不明确的授权确认时，agent 必须停下并交给用户操作。无图形登录会话的远程部署仍可使用已有本地文件和全文语料库，但不能声称获得了机构授权浏览访问。

## 2. 配置 LLM

CatMaster 默认读取：

```text
configs/llm.yaml
```

第一次使用时，从模板复制：

```bash
cp configs/llm.template.yaml configs/llm.yaml
```

如果你想查看所有可用字段和 provider 写法，用完整模板：

```bash
cp configs/llm.full.template.yaml configs/llm.yaml
```

也可以从一个预设开始：

```bash
cp configs/llm_gemini.yaml configs/llm.yaml
# 或
cp configs/llm_sonnet.yaml configs/llm.yaml
```

LLM 配置主要看两块：

- `models`：给每个模型起一个本地标签，设置 `provider`、`model`、`base_url` 和 provider 相关参数。
- `agents`：把不同任务角色绑定到模型标签。最少需要 `proposal`、`director`、`task_runner`、`memory_patch`、`summary`。

Reasoning 字段按 provider 区分填写：

- `openrouter` 和官方 `openai`：使用 `reasoning.effort`，例如 `reasoning: {effort: high}`。
- `oai_compatible`：使用顶层 `reasoning_effort`，例如 `reasoning_effort: high`。当前 CatMaster 走 `langchain-openai` 的 chat-completions 路径，不会把 `reasoning.effort` 自动翻译成 `reasoning_effort`。
- `deepseek`：使用顶层 `reasoning_effort`；DeepSeek 专用 `thinking` 等字段放到 `provider_options.deepseek.extra_body`。

## 3. 提供 API key

不要把真实 API key 写进 YAML。用环境变量：

```bash
export OPENROUTER_API_KEY="..."
# 或
export OPENAI_API_KEY="..."
# 或
export DEEPSEEK_API_KEY="..."
# 或
export ANTHROPIC_API_KEY="..."
```

可选服务：

```bash
export TAVILY_API_KEY="..."   # LitReview 和其他 agent 的直接网页搜索
export MP_API_KEY="..."       # Materials Project 结构检索
```

`.env.example` 是变量清单模板。程序不会自动读取 `.env.local`，如果你复制了本地变量文件，需要自己执行：

```bash
source .env.local
```

## 4. 不使用 YAML 的快速单模型配置

如果只是快速试跑，可以完全不写 `configs/llm.yaml`，直接用环境变量指定一个模型：

```bash
export CATMASTER_LLM_PROVIDER=openrouter
export CATMASTER_LLM_MODEL=openai/gpt-5.2
export OPENROUTER_API_KEY="..."
```

如果 `configs/llm.yaml` 存在，默认优先读取 YAML。想临时换一个配置文件：

```bash
export CATMASTER_LLM_CONFIG=configs/llm_gemini.yaml
```

## 5. Codex OAuth

Codex OAuth 使用 `langchain-openai` 的 `_ChatOpenAICodex`，不使用 API key。第一次使用前，在 `catmaster` 环境里登录：

```bash
python -c "from langchain_openai.chatgpt_oauth import login_chatgpt_device; login_chatgpt_device()"
```

`configs/llm_codex_oauth.template.yaml` 是本地 Codex OAuth 部署 profile，默认使用已验证的 `gpt-5.6-sol` 和 `xhigh` 推理强度；`configs/llm.template.yaml` 和 `configs/llm.full.template.yaml` 中也有示例。旧 `langchain-codex-oauth` token store 只作为兼容 fallback 读取，新环境不要再依赖第三方 adapter。

## 6. 准备项目空间

项目空间保存输入文件、输出文件、运行历史、中间文件和报告。建议放在仓库外或专门目录：

```bash
mkdir -p ~/catmaster_projects
```

启动 WebUI 时指定它：

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh
```

## 7. 启动 WebUI

后台启动：

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh
```

打开：

```text
http://127.0.0.1:7990
```

如果 conda 环境名不是 `catmaster`：

```bash
CATMASTER_CONDA_ENV=your_env_name CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh
```

前台运行，方便看日志：

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh --foreground
```

查看状态或停止：

```bash
./start_webui.sh --status
./start_webui.sh --stop
```

指定端口：

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh --port 7991
```

也可以直接用 Python 启动：

```bash
python -m catmaster.webui --project-space-root ~/catmaster_projects --host 127.0.0.1 --port 7860
```

## 8. 常见问题

`Missing API key`

确认对应环境变量已经在当前 shell 中导出：

```bash
echo "$OPENROUTER_API_KEY"
```

`conda is not available in PATH`

先初始化 conda，或进入已经激活 conda 的 shell 再运行启动脚本。

WebUI 起不来

先前台启动看报错：

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh --foreground
```

端口冲突

换端口：

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh --port 7991
```

## 9. 下一步

- 先读 [功能介绍与日常使用](03-features.zh.md)，理解任务模式和项目空间。
- 需要集群任务时，再读 [远程配置要点](02-remote.zh.md)。
