# 3. LLM 与运行时配置

[上一章](02-concepts.zh.md) | [目录](README.zh.md) | [下一章](04-webui.zh.md)

CatMaster 按角色路由模型。一个基础模型可以承担全部角色，也可以把研究协调、写作、审稿、图像理解和后台自进化分给不同模型。配置目标不是堆叠最多的模型，而是让每个必需角色有明确、可用且成本可控的绑定。

## 3.1 配置来源和优先级

LLM profile 按以下顺序选择：

1. 代码调用显式传入的配置路径。
2. 环境变量 `CATMASTER_LLM_CONFIG`。
3. 默认路径 `configs/llm.yaml`。
4. 如果所选文件不存在，使用单模型环境变量模式。

模板：

| 文件 | 用途 |
|---|---|
| `configs/llm.template.yaml` | 推荐起点，包含常用 provider 和完整角色表 |
| `configs/llm.full.template.yaml` | 字段和 provider 参考，不建议原样用于生产 |
| `configs/llm_codex_oauth.template.yaml` | 当前系统用户的 Codex OAuth profile |
| `configs/llm_gemini.yaml`、`llm_sonnet.yaml` 等 | 站点预设或示例，使用前逐项检查 |

YAML 存在时，`CATMASTER_LLM_PROVIDER` 和 `CATMASTER_LLM_MODEL` 只会填补模型块中的空字段，不会整体覆盖 YAML。要切换完整 profile，应修改 `CATMASTER_LLM_CONFIG`。

## 3.2 最小 YAML

一个模型承担所有必需角色的最小结构：

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

`models` 的 key 是 CatMaster 内部标签，不是 provider 的模型 ID。`agents` 的值必须引用已存在的标签。

至少要绑定以下五个角色：

| 角色 | 主要职责 |
|---|---|
| `proposal` | 任务提案和初步拆解 |
| `director` | Experiment 协调与通用决策 |
| `task_runner` | 具体 worker 和一般任务执行 |
| `memory_patch` | 记忆或改进候选处理 |
| `summary` | 总结、压缩和审稿 fallback |

## 3.3 完整角色路由

复杂部署建议显式配置关键入口，而不是全部依赖 fallback：

| 角色 | 使用位置 | 未配置时 |
|---|---|---|
| `research_lead` | Research 主协调 | `director` |
| `research_state_updater` | 研究状态更新 | `research_lead` |
| `write_director` | Writing 协调 | `research_lead` |
| `section_writer` | 写作 worker | `task_runner` |
| `write_reviewer` | Peer Review 与写作检查 | `summary` |
| `academic_polisher` | 保守语言润色 | `summary` |
| `tex_compile_fixer` | TeX 编译修复 | `academic_polisher` |
| `tool_selector` | 工具选择 | `task_runner` |
| `image_analyzer` | 图片理解 | `task_runner` |
| `literature_deep_research` | Literature Review | `director` |
| `self_evolution_proposer` | 自进化候选生成 | `memory_patch` |
| `self_evolution_reviewer` | 自进化独立审查 | `write_reviewer` |

成本有限时，可以让 `task_runner` 使用较快模型，把 `research_lead`、`write_director` 和审稿模型分配给更强模型。不要仅凭模型名称推断是否支持工具调用、视觉输入或长上下文，应以 provider 当前能力和一次真实 smoke test 为准。

## 3.4 支持的 provider

当前 `provider` 值：

```text
openai
openrouter
deepseek
gemini
oai_compatible
langchain
anthropic
codex_oauth
```

常用 key 环境变量：

| Provider | 常用变量 | 备注 |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | 可设置 `base_url` 或 request options |
| `openrouter` | `OPENROUTER_API_KEY` | 默认 endpoint 为 OpenRouter API |
| `deepseek` | `DEEPSEEK_API_KEY` | 使用 DeepSeek 专属 provider options |
| `anthropic` | `ANTHROPIC_API_KEY` | 原生 thinking 等参数放在 Anthropic chat kwargs |
| `oai_compatible` | `api_key_env` 指定的变量 | 必须核对兼容服务的 endpoint 和 schema |
| `gemini`、`langchain` | 由 `langchain_class` 和 kwargs 决定 | 使用模板中的已验证类路径 |
| `codex_oauth` | 无 API key | 凭据来自当前系统用户的 OAuth store |

真实 key 应只放在环境变量或外部 secret manager 中。`api_key` 字段虽然存在，但不应在受版本控制的 YAML 中写明文。

## 3.5 推理参数的 provider 差异

推理字段不是通用的：

- `openrouter` 和 `openai` 使用 `reasoning.effort`。
- `oai_compatible` 和 `deepseek` 使用顶层 `reasoning_effort`。
- Anthropic 原生配置放在 `provider_options.anthropic.chat_kwargs`，CatMaster 不会自动把 OpenAI 风格的 `reasoning` 翻译成 Anthropic `thinking`。
- Provider 专属请求字段放在 `provider_options.<provider>`。不要恢复旧的顶层 `tool_calling_profiles`、模型级 `tool_calling` 或顶层 `extra_body`。

标准 OpenRouter 模板中的 `prompt_cache_retention` 当前会被适配器忽略并写 warning，不能把它当成已经生效的缓存保证。

## 3.6 单模型环境变量模式

当所选 YAML 不存在时，可以快速启用一个模型：

```bash
export CATMASTER_LLM_PROVIDER=openrouter
export CATMASTER_LLM_MODEL=<OPENROUTER_MODEL_ID>
export OPENROUTER_API_KEY="<YOUR_KEY>"
```

可选变量：

```bash
export CATMASTER_API_KEY_ENV=OPENROUTER_API_KEY
export CATMASTER_BASE_URL=https://openrouter.ai/api/v1
export CATMASTER_TEMPERATURE=1.0
export CATMASTER_REASONING_EFFORT=high
```

该模式把同一模型绑定到所有角色，适合首次验证，不适合长期维护复杂 profile。

## 3.7 Codex OAuth

在 `catmaster` 环境中执行设备登录：

```bash
python -c \
'from langchain_openai.chatgpt_oauth import login_chatgpt_device; login_chatgpt_device()'

cp configs/llm_codex_oauth.template.yaml configs/llm.yaml
```

OAuth 凭据属于当前系统用户。不要复制 token store，不要打进部署包，也不要把个人 OAuth profile 当成共享多用户服务的通用身份。

## 3.8 审稿、图像和写作配置

`peer_review_models` 是 reviewer 模型标签列表。每个标签产生一份独立 reviewer report，因此列表长度直接影响调用次数、成本和耗时：

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

写作署名：

```yaml
writing:
  author_name: "<AUTHOR_NAME>"
```

模型是否接收图片、音频或视频由 `multimodal` 能力配置和 provider 行为共同决定。当前运行时默认只为 `openai`、`openrouter`、`anthropic`、`gemini` 和 `langchain` provider 开启图片块；`codex_oauth`、`deepseek` 和 `oai_compatible` 默认关闭，除非 profile 明确声明并经过真实调用验证。附件被保存不代表它已发送给模型，具体状态在 Monitor 的 `multimodal.prepared` 事件中检查。

## 3.9 Runtime 和 Literature 配置

标准 YAML 的 runtime 默认值：

```yaml
agent_runtime:
  recursion_limit: 300
  max_tool_calls: 120
  deepagent_context_trigger_token_cap: 270000
  print_state_messages: false
  print_http_raw_post: false
```

这些是安全边界和上下文压缩控制，不是质量滑块。盲目提高会增加失控循环、费用和超时风险。YAML 模式下应修改 YAML；`.env.example` 中对应变量主要用于无 YAML profile。

`literature` 控制不同角色的检索深度、公开网页 fallback、重试和预算。先沿用模板，再根据一次有记录的 Literature Review 调整。检索深度越高，网页调用和上下文成本通常越高。

## 3.10 工具输出策略

`CATMASTER_TOOL_OUTPUT_CONFIG` 默认指向 `configs/tool_output.yaml`。当前策略：

```yaml
offload:
  inline_data_enabled: true
  preview_chars: 3000
  offload_chars: 20000
  offload_dir_rel: "_tool_outputs"
```

超过阈值的完整输出写到 workspace 的 `files/_tool_outputs/`，Chat 只保留预览和路径。不要把 `configs/tool_policy.yaml` 当成当前 specialist runtime 的用户权限入口；实际工具授权由 runtime allowlist、task audience 和 Review 中断策略决定。

## 3.11 无网络配置检查

下面的命令只解析配置，不调用模型：

```bash
python -c 'from catmaster.llm.config import LLMProfile; p=LLMProfile.from_env_or_file(); print("models:", sorted(p.models)); print("roles:", p.agents)'
```

如果解析成功，再启动 WebUI 做一次最短对话。配置解析成功仍不证明 key、模型 ID、endpoint、工具调用或多模态能力可用。
