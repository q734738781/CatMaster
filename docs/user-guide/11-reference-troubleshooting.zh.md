# 11. 参考与故障排查

[上一章](10-deployment-operations.zh.md) | [目录](README.zh.md)

本章汇总常用变量、默认行为、限制和诊断顺序。先定位故障属于启动、模型、文件、浏览器、远程配置还是科学任务，再修改对应层。不要用“重装全部依赖”代替证据检查。

## 11.1 配置文件地图

| 文件 | 作用 | 是否含私密信息 |
|---|---|---|
| `requirements/pc-conda.yml` | 唯一 control-plane 环境定义 | 否 |
| `.env.example` | 环境变量清单，不会自动加载 | 否 |
| `configs/llm.yaml` | 活动 LLM profile | 不应含 key，可能含私有 endpoint |
| `configs/llm*.template.yaml` | Provider 和角色模板 | 否 |
| `configs/tool_output.yaml` | 长输出预览与落盘策略 | 否 |
| `configs/tool_policy.yaml` | 兼容配置，不是当前 specialist runtime 的权限 source of truth | 否 |
| `configs/dpdispatcher/*_template.yaml` | 公开 machine/resource/task/backend 模板 | 否 |
| `configs/dpdispatcher/{machines,resources,tasks,mlff_backends}.yaml` | 活动远程配置 | 是 |
| `configs/dpdispatcher/env_templates/` | 远程环境激活脚本参考 | 需要替换站点值 |

## 11.2 常用环境变量

### LLM 和检索

| 变量 | 用途 |
|---|---|
| `CATMASTER_LLM_CONFIG` | 选择 YAML profile，默认 `configs/llm.yaml` |
| `CATMASTER_LLM_PROVIDER` | 无 YAML 模式 provider，或填补空 provider |
| `CATMASTER_LLM_MODEL` | 无 YAML 模式 model，或填补空 model |
| `CATMASTER_API_KEY_ENV` | 无 YAML模式的 key 变量名 |
| `CATMASTER_BASE_URL` | 无 YAML或空字段时的 endpoint |
| `CATMASTER_TEMPERATURE` | 无 YAML或空字段时的 temperature |
| `CATMASTER_REASONING_EFFORT` | 无 YAML或空 reasoning 时的 effort |
| `OPENAI_API_KEY`、`OPENROUTER_API_KEY` | 对应 provider key |
| `DEEPSEEK_API_KEY`、`ANTHROPIC_API_KEY` | 对应 provider key |
| `TAVILY_API_KEY`、`MP_API_KEY` | 网页搜索和 Materials Project |
| `SEMANTIC_SCHOLAR_API_KEY`、`OPENALEX_API_KEY`、`NCBI_API_KEY` | 文献服务 |
| `CROSSREF_MAILTO` | Crossref 礼貌联系地址 |

### 浏览器和本地工具

| 变量 | 用途 |
|---|---|
| `CATMASTER_AGENT_BROWSER_BIN` | `agent-browser` 可执行文件 |
| `CATMASTER_AGENT_BROWSER_PROFILE` | 浏览器 profile，必须在 workspace 外 |
| `CATMASTER_AGENT_BROWSER_AUTO_CONNECT` | 连接已运行 Chrome |
| `CATMASTER_AGENT_BROWSER_HEADED` | 显示浏览器窗口 |
| `CATMASTER_AGENT_BROWSER_MAX_OUTPUT` | 受控浏览器输出上限 |
| `CATMASTER_VASPKIT_BIN`、`CATMASTER_VESTA_BIN` | 本地辅助程序路径 |
| `CATMASTER_XVFB_RUN` | 无 DISPLAY 渲染 wrapper |
| `CATMASTER_PANDOC_BIN`、`CATMASTER_CHROME_BIN` | Markdown PDF 工具路径 |
| `CATMASTER_JSMOL_CACHE_DIR` | JSmol 持久缓存 |

### Runtime 和 WebUI

| 变量 | 用途 |
|---|---|
| `CATMASTER_PROJECT_SPACE_ROOT` | 多用户 project 根目录 |
| `CATMASTER_CONDA_ENV` | 启动脚本使用的 conda 环境名 |
| `CATMASTER_HOST`、`CATMASTER_PORT` | WebUI 监听地址和端口 |
| `CATMASTER_RUNTIME_DIR` | PID 和默认日志目录 |
| `CATMASTER_WEBUI_LOG`、`CATMASTER_WEBUI_PID` | 日志和 PID 文件 |
| `CATMASTER_TOOL_OUTPUT_CONFIG` | 工具输出策略 |
| `CATMASTER_SELF_EVOLUTION_MODE` | `off`、`observe`、`auto` |
| `CATMASTER_RECURSION_LIMIT`、`CATMASTER_MAX_TOOL_CALLS` | 主要用于无 YAML profile |
| `CATMASTER_DEEPAGENT_CONTEXT_TRIGGER_TOKEN_CAP` | 无 YAML profile 的上下文压缩 cap |
| `CATMASTER_PRINT_HTTP_RAW_POST` | 调试原始请求，可能暴露敏感信息，默认 false |

`CATMASTER_PRINT_HTTP_RAW_POST=true` 可能把 prompt 或请求数据写到日志，只在隔离诊断环境短时使用。

## 11.3 地址和优先级

启动脚本解析顺序：CLI 参数优先于 `CATMASTER_*` 环境变量，环境变量优先于脚本 `LOCAL_*` 常量，再落到代码 fallback。

| 启动方式 | 未显式设置时 |
|---|---|
| `./start_webui.sh` | 脚本内嵌 `0.0.0.0:7991` |
| `python -m catmaster.webui` | `127.0.0.1:7860` |
| 本手册推荐 | 显式 `127.0.0.1:7991` |

因此排查“页面打不开”时，先运行：

```bash
./start_webui.sh --status
tail -n 100 .runtime/webui.log
ss -ltnp | grep 7991
```

## 11.4 WebUI 启动失败

前台运行：

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
./start_webui.sh --foreground --host 127.0.0.1 --port 7991
```

按第一条真实 traceback 排查：

- `conda is not available`：初始化 conda 或设置正确 `CATMASTER_CONDA_ENV`。
- `Address already in use`：找到占用进程，或改用另一个显式端口。
- JSmol 下载失败：预热 cache；如果其他页面正常，把它作为结构预览问题单独处理。
- 静态前端缺失：确认部署包完整；维护者部署不要使用不完整的 `--include-path` 集合代替 runtime 包。
- Project root permission denied：修复目录 ownership，不要以 root 临时绕过。

## 11.5 LLM 配置或调用失败

先做无网络解析：

```bash
python -c 'from catmaster.llm.config import LLMProfile; p=LLMProfile.from_env_or_file(); print(sorted(p.models)); print(p.agents)'
```

常见问题：

- `Missing API key`：确认变量已 export 到启动 WebUI 的同一进程环境。
- 复制了 `.env.local` 但无效：用 `set -a; source .env.local; set +a`。
- 角色引用未知标签：修正 `agents` 或 `peer_review_models`。
- 旧字段报错：移除 `tool_calling_profiles`、模型级 `tool_calling` 和错误位置的 `extra_body`。
- Provider 400：核对模型 ID、base URL、reasoning 字段和 provider options。
- 只文字回复、不调用工具：确认模型支持当前工具 schema，并检查 tool card 和 provider 日志。
- 长任务过早停止：先看 `max_tool_calls`、recursion 和实际错误，不要直接把边界调到极大。

## 11.6 登录和 workspace

- 注册失败：用户名需 3 到 40 个允许字符，密码至少 8 个字符，并完成新验证码。
- 登录后看不到旧项目：核对 `CATMASTER_PROJECT_SPACE_ROOT` 和 username，不要手工把项目放在 root 的错误层级。
- 旧 `.catmaster` 项目被拒绝：按 `files/`、`metadata/` 新布局迁移。
- Thread 历史丢失：检查 `metadata/threads/` 和 DeepAgent SQLite 是否随备份恢复。
- Skill Evolution 不显示：确认不是 `--no-login`，并检查 mode 是否为 `off`。

## 11.7 附件和文件预览

- Composer 拒绝文件：先看 64 MiB 浏览器限制。
- 文件已保存但模型没看：检查媒体 32 MiB inline、模型 multimodal 能力和 `multimodal.prepared` warning。
- PDF/Office 内容不全：检查 50 MiB、20 页/slide、60,000 字符和 spreadsheet 限制。
- 旧 Office 格式只保存：转换为 PDF、DOCX、XLSX 或 PPTX。
- JSmol 空白：检查 cache、浏览器控制台和结构格式，不要重启远程 task。
- 文件上传后内容变了：Files 使用同名覆盖；从外部备份恢复。
- 误删 `metadata/`：立即停止写入，从一致性备份恢复，不能靠重新上传 `files/` 修复 thread。

## 11.8 Literature Review

- `agent-browser` 不可用：依次运行 `agent-browser doctor --offline --quick` 和 `agent-browser mcp --help`。
- 登录页或 CAPTCHA：切到 headed，由用户完成；不要让 agent 反复自动尝试。
- 找到 DOI 但无全文：记录证据级别，检查机构会话或由用户上传合法全文。
- 引用元数据冲突：以 DOI/publisher 页面和文献自身为主，记录版本差异。
- 本地 corpus 查询漏文：检查 ingest manifest、parse status 和文件是否超出解析限制。

## 11.9 远程 catalog 或连接失败

按层排查：

1. 四个活动配置文件是否存在。
2. 文件名是否错误地包含 `template`。
3. YAML 是否能解析，key 是否被其他活动文件覆盖。
4. Machine SSH 是否支持 BatchMode。
5. `remote_root` 是否可写。
6. Resource 的 machine、queue、audience 和 `source_list` 是否正确。
7. Task 是否 enabled，backend 是否 enabled。
8. Worker audience 是否匹配。

`command not found` 或 127 常见于 `machine.env_setup`、`source_list` 或 task binary。先在相同非交互 SSH 环境中执行 `command -v`，不要通过修改科学 stage 掩盖环境错误。

## 11.10 远程运行和结果失败

- Tool 仍在 pending：等待它返回，不要轮询 receipt 或重投。
- SSH 断开：保存 receipt 身份，在调度器检查作业仍否存在。
- Scheduler completed 但无结果：下载 finished task 和 terminated log，检查 backward files 和远程权限。
- `status.json` 成功但科学不收敛：按程序日志和领域 QC 判为科学失败。
- Batch 部分失败：逐个一级 stage 分类，不要把成功子任务一起重算。
- 想停止：WebUI Stop 不取消远程 job；由管理员按 receipt 对应 job ID 使用调度器。
- 想清理远程目录：先确认结果、stdout、stderr 和 receipt 已本地保存。

恢复命令和顺序见[远程机器与任务执行](08-remote-execution.zh.md)。

## 11.11 当前 UI 限制

- 没有历史 run 选择器。
- 没有 thread 重命名、删除、branch 或 retry UI。
- Interrupted 状态必须使用消息内审批卡，composer 的 `Respond` 不是审批恢复。
- Monitor 总览可能对应 workspace/lane 的当前或最近 run，不一定严格对应选中 thread。
- Files 上传同名覆盖，删除永久递归。
- Files 树显示 `metadata/`，但没有专门的保护开关。
- 后端支持安全 ZIP 解压，Files UI 暂无解压开关。
- WebUI Stop 不取消已经提交的远程作业。
- Skill Evolution 仅登录模式可见，从下一次 run 生效。

文档明确列出这些限制，是为了让用户选择正确路径，不表示可以绕过安全边界。

## 11.12 高级线程 API 示例

下面示例只适合绑定本机的 `--no-login` 测试服务。现代主接口使用 workspace/thread/artifact API；旧 run API 只作为兼容和调试路径。

```bash
curl -s http://127.0.0.1:7991/api/bootstrap

THREAD_ID="$(
  curl -s -X POST \
    -H 'Content-Type: application/json' \
    -d '{"title":"CO adsorption","entrypoint":"experiment","permission_mode":"hitl"}' \
    http://127.0.0.1:7991/api/workspaces/admin/threads |
  jq -r '.thread.thread_id'
)"

curl -s -X POST \
  -H 'Content-Type: application/json' \
  -d '{"text":"Inspect structures/slab.vasp and prepare three adsorption structures.","entrypoint":"experiment","permission_mode":"hitl"}' \
  "http://127.0.0.1:7991/api/threads/$THREAD_ID/submit"

curl -N \
  "http://127.0.0.1:7991/api/threads/$THREAD_ID/stream?last_seq=0"
```

登录模式需要正确的 session cookie 和 CSRF/访问上下文，不应把无登录示例直接改成公网自动化。

## 11.13 安装验收清单

### 本地 control plane

- [ ] `conda env create/update` 成功。
- [ ] LLM YAML 离线解析成功。
- [ ] API key 确实进入 WebUI 进程。
- [ ] WebUI 显式监听预期地址和端口。
- [ ] 注册、登录和用户隔离通过。
- [ ] Workspace 同时有 `files/`、`metadata/`。
- [ ] Thread 对话、SSE、artifact 和 Monitor 可用。
- [ ] `agent-browser` doctor 通过或明确禁用该路径。
- [ ] JSmol、PDF、结构和表格预览按部署需求通过。

### 远程执行

- [ ] 四个活动 DPDispatcher 配置存在且不在 Git。
- [ ] SSH、remote root、scheduler 和环境脚本通过。
- [ ] Task/resource/audience/backend catalog 符合实际安装。
- [ ] `python scripts/remote_execution_smoke.py --list` 可用。
- [ ] 每类启用引擎至少一个最小真实 case 通过。
- [ ] Stage 收到 status、stdout、stderr 和 receipt。
- [ ] 已演练 receipt 驱动的下载和失败分类。

### 运维和安全

- [ ] 默认不直接暴露公网，`--no-login` 仅 loopback。
- [ ] TLS、VPN或外部访问控制已配置。
- [ ] Project、账号数据库、私有配置和 secret 有备份。
- [ ] 已验证升级、回滚和日志留存流程。
- [ ] 用户知道 Stop 不会取消远程作业，计算完成仍需科学 QC。
