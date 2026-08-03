# 10. 安装、模型配置与部署

[上一章](09-tools-skills-evolution.zh.md) | [目录](README.zh.md) | [下一章](11-reference-troubleshooting.zh.md)

本章供自己安装 CatMaster、配置模型或管理服务器的用户查阅。普通使用者不需要理解全部 YAML 和环境变量，只需知道当前部署开放了哪些 Agent、远程 tasks 和外部程序。

## Control plane 环境

CatMaster WebUI、Agent runtime、材料工具和大部分本地分析共用 `requirements/pc-conda.yml`。这是唯一的 control plane 环境定义：

```bash
conda env create -f requirements/pc-conda.yml
conda activate catmaster
```

更新现有环境：

```bash
conda env update -n catmaster -f requirements/pc-conda.yml
```

MACE、UMA、MatterSim 和 ORB-v3 的 requirements 文件用于远程隔离环境。把它们全部安装到 control plane 容易造成 torch、CUDA 和模型依赖冲突，也不能自动创建可用 remote task。

## 配置 LLM

CatMaster 按角色选择模型。一个模型可以承担所有角色，也可以把研究协调、worker、写作、审稿、图像理解和低频候选 proposal/review 分给不同模型。第一次安装先用标准模板：

```bash
cp -n configs/llm.template.yaml configs/llm.yaml
export OPENROUTER_API_KEY="<YOUR_KEY>"
```

最小 profile 可以只定义一个模型，并把必需角色都指向它：

```yaml
models:
  main:
    provider: openrouter
    model: <OPENROUTER_MODEL_ID>
    temperature: 1.0
    reasoning:
      effort: high
    api_key_env: OPENROUTER_API_KEY
    base_url: https://openrouter.ai/api/v1

agents:
  proposal: main
  director: main
  task_runner: main
  memory_patch: main
  summary: main
```

`models` 下的 `main` 是 CatMaster 内部标签，`model` 才是 provider 模型 ID。`agents` 中的值必须引用已经定义的标签。

### 角色如何映射到五类 Agent

基础角色必须存在，其他角色可以回退：

| 角色 | 主要用途 | 常见回退 |
|---|---|---|
| `proposal` | 任务提案和初步拆解 | 必需 |
| `director` | Experiment 协调与通用决策 | 必需 |
| `task_runner` | Materials、Dynamics、ML、ORCA/xTB worker | 必需 |
| `memory_patch` | 项目记忆或 skill 候选 | 必需 |
| `summary` | 总结与通用审查 fallback | 必需 |
| `research_lead` | Research Agent | `director` |
| `research_state_updater` | Research 状态更新 | `research_lead` |
| `hypothesis_proposer` | 形成可证伪假设与验证计划 | `research_lead` |
| `evidence_judge` | 独立解释验证证据 | `research_state_updater` |
| `write_director` | Writing coordinator | `research_lead` |
| `section_writer` | Writing worker | `task_runner` |
| `write_reviewer` | 写作检查与审稿 | `summary` |
| `academic_polisher` | 保守语言润色 | `summary` |
| `tex_compile_fixer` | TeX 编译修复 | `academic_polisher` |
| `tool_selector` | 通用 tool 选择辅助 | `task_runner` |
| `image_analyzer` | 图片理解 | `task_runner` |
| `literature_deep_research` | Literature Review | `director` |
| `literature_worker` | 有边界的文献发现、阅读、提取与证据审计 | `literature_deep_research` |
| `self_evolution_proposer` | 改进候选生成 | `memory_patch` |
| `self_evolution_reviewer` | 候选独立审查 | `write_reviewer` |

成本受限时，可以让 `task_runner` 使用速度较快的模型，把 Research、Writing 和 reviewer 角色分配给更强模型。模型是否支持工具调用、图片和长上下文必须通过 provider 文档与真实 smoke test 验证，不能只根据模型名称判断。

### Provider 与凭据

当前 profile 支持 `openai`、`openrouter`、`deepseek`、`gemini`、`oai_compatible`、`langchain`、`anthropic` 和 `codex_oauth`。常用 key 变量为 `OPENAI_API_KEY`、`OPENROUTER_API_KEY`、`DEEPSEEK_API_KEY` 和 `ANTHROPIC_API_KEY`。兼容服务通过 `api_key_env` 指定变量，并明确填写 endpoint。

Provider 的 reasoning 参数并不通用。OpenAI 与 OpenRouter 使用 `reasoning.effort`；部分兼容服务使用 `reasoning_effort`；Anthropic 的原生 thinking 字段位于 provider 专属 kwargs。优先从仓库模板复制对应结构，不要把一个 provider 的字段原样搬给另一个。

真实 key 只放环境变量或外部 secret manager。`configs/llm.yaml` 可以包含私有 endpoint，但不应保存明文 key。`.env.local` 不会自动加载，如需使用：

```bash
set -a
source .env.local
set +a
```

Codex OAuth 使用当前系统用户的凭据：

```bash
python -c \
'from langchain_openai.chatgpt_oauth import login_chatgpt_device; login_chatgpt_device()'

export CATMASTER_LLM_CONFIG=configs/llm_codex_oauth.template.yaml
```

Codex OAuth 模板会传入 `timeout_s: 180`，但不显式设置 `max_retries`；传输
错误、限流和 HTTP 服务端错误使用已锁定 OpenAI SDK 的默认重试值。Codex
后端也可能先接受一个 HTTP 200 stream，随后以结构化的
`server_is_overloaded` 错误结束；SDK 无法在 HTTP 层重试这种情况。CatMaster
只对这一种 stream 错误再做最多六次重试，依次等待
30、60、120、240、480 和 600 秒；该行为覆盖所有 DeepAgent 层级，包括
CatMaster 显式配置的 `general-purpose` 子代理，不会捕获其他模型异常。

不要复制 OAuth token store，也不要把个人 OAuth profile 当作共享多用户服务的公共身份。

### Reviewer、图片与多模态

`peer_review_models` 是 reviewer 模型标签列表。每个标签会产生一份独立 reviewer report，因此数量直接影响调用次数、费用和耗时。

```yaml
peer_review_models:
  - reviewer-a
  - reviewer-b
```

图片生成可以绑定单独模型：

```yaml
image_generation:
  model_label: image-model
  image_config:
    aspect_ratio: "4:3"
```

模型能否接收图片由 profile 的 multimodal 能力和 provider 行为共同决定。当前默认只为 OpenAI、OpenRouter、Anthropic、Gemini 和 LangChain provider 开启图片块；其他 provider 需要明确声明并做真实调用验证。附件保存成功不代表模型看到了内容，排查时检查 `multimodal.prepared` 事件。

### Profile 选择与离线解析

配置路径优先级为：代码显式路径、`CATMASTER_LLM_CONFIG`、`configs/llm.yaml`，最后在所选 YAML 不存在时进入单模型环境变量模式。环境变量模式示例：

```bash
export CATMASTER_LLM_PROVIDER=openrouter
export CATMASTER_LLM_MODEL=<OPENROUTER_MODEL_ID>
export OPENROUTER_API_KEY="<YOUR_KEY>"
```

先做不调用模型的解析检查：

```bash
python -c 'from catmaster.llm.config import LLMProfile; p=LLMProfile.from_env_or_file(); print("models:", sorted(p.models)); print("roles:", p.agents)'
```

解析成功只证明 YAML 结构有效。Key、endpoint、模型 ID、tool calling 和多模态仍需在 WebUI 中做最小真实对话。

## 配置文献检索与受控浏览器

公共检索可按部署需要提供以下变量：

```bash
export TAVILY_API_KEY="<KEY>"
export SEMANTIC_SCHOLAR_API_KEY="<KEY>"
export OPENALEX_API_KEY="<KEY>"
export NCBI_API_KEY="<KEY>"
export CROSSREF_MAILTO="you@example.org"
```

对使用 `codex_oauth` 或 OpenAI Responses 的角色，`TAVILY_API_KEY` 不是必需项，
这些角色会得到托管的原生 `web_search`。如果其他 provider 需要公共检索，则仍需配置
Tavily；两种实现不会以同名工具同时暴露给同一个 agent，specialist、worker 和
self-evolution role 都经过同一个 provider resolver。对于 CatMaster 搜索函数，
`literature.public_web_on_search_failure` 控制学术索引降级。额度、鉴权、限流或网络
失败会在当前 run 内熔断 Tavily，后续搜索不再继续消耗或重试该后端；降级结果会标明
实际学术索引来源，不会伪装成通用网页覆盖。

实际可见 tools 以当前 Literature Review runtime 为准。API key 提供访问能力，不保证全文权限或元数据完全正确。`requirements/pc-conda.yml` 会安装 `scansci-pdf==1.9.0` 和 `cloakbrowser==0.5.3`。CatMaster 先调用合法、非浏览器的 OA adapters，再把一次 ScanSci/CloakBrowser DOI 落地页访问作为内部低优先级兜底。`UNPAYWALL_EMAIL`、`OPENALEX_MAILTO`、`CORE_API_KEY` 和 `SCANSCI_PDF_PROXY` 都是可选配置；不需要额外安装浏览器 CLI、暴露模型浏览器 tools 或配置 browser profile。

## 启动方式与访问范围

本地工作站应显式绑定 loopback：

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
CATMASTER_HOST=127.0.0.1 \
CATMASTER_PORT=7991 \
./start_webui.sh
```

部署在远程服务器、只供自己使用时，仍让服务监听服务器的 `127.0.0.1:7991`，再从本机建立 SSH tunnel：

```bash
ssh -L 7991:127.0.0.1:7991 <USER>@<SERVER>
```

浏览器打开本机 `http://127.0.0.1:7991`。这种方式不会直接暴露 WebUI。

多人共享服务需要反向代理或 VPN、TLS、外部身份控制、最小文件权限、日志与备份。内置登录包含账号隔离和基础注册，但不是完整公网身份平台；默认开放注册，应用本身不终止 TLS，cookie 也不应被当作公网安全边界。至少预置一个用户后，可用 `--disable-registration` 启动，或设置 `CATMASTER_DISABLE_REGISTRATION=1`，在继续要求登录的同时拒绝新账号。此时状态 API 返回 `registration_enabled: false`，前端隐藏创建账号入口，注册端点返回 HTTP 403。

`--no-login` 只适用于可信单机并绑定 loopback。它进入开放 `admin` workspace，同时关闭 Skill Evolution。

## 配置远程计算

Remote task 的用户语义在[第 8 章](08-remote-execution.zh.md)。管理员需要从四个模板建立私有活动配置。下面的 `-n` 会保留已经存在的活动文件；升级时应逐项合并模板变化，不要覆盖站点配置：

```bash
cp -n configs/dpdispatcher/machines_template.yaml configs/dpdispatcher/machines.yaml
cp -n configs/dpdispatcher/resources_template.yaml configs/dpdispatcher/resources.yaml
cp -n configs/dpdispatcher/tasks_template.yaml configs/dpdispatcher/tasks.yaml
cp -n configs/dpdispatcher/mlff_backends_template.yaml configs/dpdispatcher/mlff_backends.yaml
```

这些活动文件包含主机名、用户名、SSH key 路径、队列、远程目录和环境脚本，已被 Git 与部署包排除。不要把真实内容贴进 issue、prompt 或共享 workspace。

### Machine、resource、task 和 backend

Machine card 定义 SSH 连接、Slurm 或 Shell 类型、`remote_root` 和基础环境。首次连接先由管理员交互确认 host key，再使用 BatchMode 测试。`remote_root` 必须存在且可写，Slurm machine 还要验证 `sbatch`、`squeue` 和 `scancel`。

Resource card 把 machine 与 CPU/GPU、queue、walltime、环境 `source_list` 和 worker audience 绑定。模板中的核数和队列只是示例，必须按站点修改。不要为了方便移除 audience 限制。

Task card 定义科学程序、输入布局、默认 resource、boot script 和回传文件。模板默认支持 VASP、CP2K、LAMMPS、通用 MLFF、MACE train/eval、xTB、CREST 和 ORCA。只有经过验证的 tasks 应保持 enabled。

MLFF backend card 决定 MACE、UMA、MatterSim 或 ORB-v3 的启用状态、resource、operation 和模型。每个 model profile 都必须声明精确的 provider model、官方 task/domain 能力和 charge/spin 能力；MACE 还声明实际 loader、允许的 heads 与默认 head。UMA、MatterSim 和 ORB-v3 的模型键必须与官方命名完全一致，不能另造缩写或大小写别名。每个 backend 使用独立远程环境。模板默认启用 MACE `mh-1` 与独立 `omol-0`；其他 backend 只有在依赖、权重、device 和最小真实 case 通过后才整体开放。

### 远程环境加载

远程命令环境依次由 machine `env_setup`、resource `source_list`、提交 prepend script 和 task command 构造。Program modules、conda activate、许可证变量和库路径应放在站点受控脚本中，不应写进 stage 或 prompt。

DPDispatcher 通常启动非交互 shell，不能假设它会读取用户的 `.bashrc`。若 GPU 节点访问模型仓库需要代理，复制并编辑 `configs/dpdispatcher/env_templates/catmaster_env_proxy.sh`，在相关 GPU resource 的 `source_list` 中把它放在 provider conda 环境脚本之前；不需要代理的节点应删除该项。代理脚本只应绑定使用该代理的 machine，不要把指向 GPU 节点 `localhost` 的代理脚本复用到其他 CPU/SSH 节点。

投入使用前，每个已启用引擎至少跑一个成本可控的 smoke case，确认 task catalog、环境、结果回传、`status.json`、stdout/stderr 和 receipt。`python scripts/remote_execution_smoke.py --list` 只列 case；其他参数会提交真实作业，不要一开始运行全部 suite。

## Structure Workbench、JSmol、VESTA 与 VASPKIT

生产前端包含精确固定版本的 MatterViz/Svelte 和按需加载的 Ketcher chunks，统一从 `/static` 提供，不依赖 CDN 或外部字体。Server 会发送 Content Security Policy，只允许同源 chunks、本地字体、data/blob 图片和体数据解析所需 worker。

JSmol 16.3.13 只作为 OUTCAR vibration 与未支持格式的兼容 fallback。启动器会在缓存缺失时安装固定资源。离线服务器可先预热持久 cache：

```bash
CATMASTER_JSMOL_CACHE_DIR=/persistent/cache/jsmol \
python scripts/install_jsmol_assets.py
```

JSmol 缺失只影响这些 fallback 预览；MatterViz 支持的结构、Workbench、LLM 和远程 task 不受影响。

修改前端依赖后，在 `catmaster/webui/frontend` 执行 `npm run build`，并在部署使用的 base path 中分别验证一个周期结构、一次分子 2D/3D 切换、一个轨迹帧请求和一个 volume grid。版本必须同时精确写入 `package.json` 与 lockfile，不能改成 CDN 脚本。

VASPKIT 可通过 `CATMASTER_VASPKIT_BIN` 指定：

```bash
export CATMASTER_VASPKIT_BIN=/opt/vaspkit/bin/vaspkit
```

VESTA 渲染可设置：

```bash
export CATMASTER_VESTA_BIN=/opt/VESTA/VESTA
export CATMASTER_XVFB_RUN=/usr/bin/xvfb-run
```

无 DISPLAY 的服务器通常需要 Xvfb。VESTA 和 VASPKIT 是可选辅助程序，不随 CatMaster 提供许可证。

## Pandoc、Chrome、字体、TeX 与 Julia

Markdown PDF 需要 Pandoc 和 Chrome/Chromium，CJK 文档还要有合适字体：

```bash
export CATMASTER_PANDOC_BIN=/usr/bin/pandoc
export CATMASTER_CHROME_BIN=/usr/bin/chromium

pandoc --version
chromium --version
fc-match "Noto Sans CJK SC"
```

LaTeX 文稿至少需要 `pdflatex`，使用 BibTeX 时还需要 `bibtex`。编译成功后应人工查看 PDF，确认字体、图片、公式和分页。

PySR 首次 import 可能下载 Julia 并预编译。联网维护期可运行：

```bash
python scripts/pysr_julia_smoke.py --fit
```

离线机器应预装 Julia，并通过 `PYTHON_JULIACALL_BINDIR` 指向其 `bin` 目录。不要让第一个用户任务承担首次下载和预编译。

## 运行边界与长输出

LLM profile 中的 `recursion_limit`、`max_tool_calls` 和上下文压缩阈值是安全边界，不是越大越好的质量旋钮。长任务过早停止时，先查实际 tool error、上下文与任务范围，再决定是否调整。

`configs/tool_output.yaml` 控制长工具结果。默认会在 Chat 保留预览，并把大输出写到 workspace 的 `_tool_outputs/`。不要把 `configs/tool_policy.yaml` 当作当前 Agent 权限入口；可见 tools 由 runtime allowlists、task audiences 和 Review 中断共同决定。

## 部署包、升级与回滚

`scripts/package_remote_deploy.sh` 生成不包含 `.git`、私有配置、key、用户项目和运行日志的离线包。部署后使用 `scripts/deploy_runtime.sh` 同步 runtime，并在目标环境完成依赖与外部工具检查。实际命令和选项以脚本 `--help` 为准。

升级前记录当前 Git commit、conda 环境、活动 LLM profile、四个 DPDispatcher 配置、启动参数和外部程序版本。备份项目根与认证数据库，再在副本或测试 workspace 做一次对话、文件、结构预览和至少一个已启用远程 task 的最小验收。

从旧 Research Kernel 或 hypothesis campaign 升级时，每个 workspace 单独迁移。先停止该 workspace 的旧 writer，再运行 dry-run：

```bash
/home/chenhh/miniconda3/envs/catmaster/bin/python \
  scripts/migrate_research_graph.py /absolute/path/to/workspace
```

报告会区分可确定迁移的 v3/v4 campaign、需要人工检查的 v2 或不完整 Kernel，以及损坏文件。确认数量后执行：

```bash
/home/chenhh/miniconda3/envs/catmaster/bin/python \
  scripts/migrate_research_graph.py /absolute/path/to/workspace --apply
```

命令返回 rollback manifest 路径。旧文件会移到
`metadata/legacy_research_state/`，新版本只写
`metadata/workspace.sqlite`。迁移批次有稳定的 in-progress 指针；进程中断后重复
`--apply` 会续跑同一批次。不要让旧版和新版服务在同一 workspace 滚动混跑。

如果尚未用新版本写入 Research Graph，可以用返回的 manifest 回滚：

```bash
/home/chenhh/miniconda3/envs/catmaster/bin/python \
  scripts/migrate_research_graph.py /absolute/path/to/workspace \
  --rollback metadata/legacy_research_state/<batch>/rollback_manifest.json
```

回滚会删除该批次导入的 graph、恢复原 thread 绑定并把旧文件移回原路径。已经在新 graph 上继续研究后，不要用这个命令覆盖新科学状态，应从备份恢复到独立 workspace 后人工合并。

Workspace SQLite 只在已识别的本地文件系统上默认使用 WAL；网络或未知文件系统
默认使用 rollback journal。只有部署已经独立验证存储语义时，才设置
`CATMASTER_WORKSPACE_SQLITE_JOURNAL_MODE=WAL`。

回滚代码时不要覆盖用户项目。恢复先前 commit 或部署包后，还要恢复与它兼容的依赖和配置。不要把项目数据、私有 YAML 和密钥打进代码发布包作为回滚手段。

## 备份与日志

默认运行目录为 `.runtime/`，常用日志是 `.runtime/webui.log`。共享服务应配置日志轮转，并避免长期打开可能记录原始 prompt 或请求体的调试选项。

完整备份包括：

- 项目根下每个 workspace 的 `files/` 与 `metadata/`。
- 登录部署的 `.webui_auth/auth.sqlite`。
- 版本控制之外的 LLM 与 DPDispatcher 活动配置。
- 外部 secret manager 或站点环境脚本的独立备份。

备份最好在没有写入中的 run 时进行，并定期演练恢复。下一章给出面向用户和管理员的故障诊断顺序与参考 prompts。
