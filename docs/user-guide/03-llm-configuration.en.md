# 3. LLM and runtime configuration

[Previous](02-concepts.en.md) | [Contents](README.en.md) | [Next](04-webui.en.md)

CatMaster routes models by role. One base model can cover every role, or a
deployment can assign research coordination, writing, review, image
understanding, and background evolution to different models. The goal is not to
use the largest possible set. Each required role needs an explicit, available,
and affordable binding.

## 3.1 Configuration source and precedence

The LLM profile is selected in this order:

1. A path supplied explicitly by code.
2. `CATMASTER_LLM_CONFIG`.
3. The default path `configs/llm.yaml`.
4. Single-model environment mode when the selected file does not exist.

Templates:

| File | Purpose |
|---|---|
| `configs/llm.template.yaml` | Recommended starting point with common providers and the full role map |
| `configs/llm.full.template.yaml` | Field and provider reference, not a production file to copy blindly |
| `configs/llm_codex_oauth.template.yaml` | Codex OAuth profile for the current system user |
| `configs/llm_gemini.yaml`, `llm_sonnet.yaml`, and others | Site profiles or examples that must be reviewed before use |

When YAML exists, `CATMASTER_LLM_PROVIDER` and `CATMASTER_LLM_MODEL` only fill
empty fields inside a model block. They do not replace the whole YAML profile.
Set `CATMASTER_LLM_CONFIG` to switch profiles.

## 3.2 Minimal YAML

This is the smallest shape in which one model covers all required roles:

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

A `models` key is a CatMaster label, not the provider's model ID. Values under
`agents` must reference existing labels.

Five role bindings are required:

| Role | Main responsibility |
|---|---|
| `proposal` | Task proposal and initial decomposition |
| `director` | Experiment coordination and general decisions |
| `task_runner` | Concrete workers and ordinary task execution |
| `memory_patch` | Memory or improvement candidate handling |
| `summary` | Summaries, compaction, and review fallback |

## 3.3 Full role routing

For a complex deployment, bind important entrypoints explicitly instead of
depending entirely on fallbacks:

| Role | Used for | Fallback |
|---|---|---|
| `research_lead` | Research coordinator | `director` |
| `research_state_updater` | Research-state updates | `research_lead` |
| `write_director` | Writing coordinator | `research_lead` |
| `section_writer` | Writing worker | `task_runner` |
| `write_reviewer` | Peer Review and writing checks | `summary` |
| `academic_polisher` | Conservative language polishing | `summary` |
| `tex_compile_fixer` | TeX compile repair | `academic_polisher` |
| `tool_selector` | Tool selection | `task_runner` |
| `image_analyzer` | Image understanding | `task_runner` |
| `literature_deep_research` | Literature Review | `director` |
| `self_evolution_proposer` | Evolution candidate proposal | `memory_patch` |
| `self_evolution_reviewer` | Independent evolution review | `write_reviewer` |

With a limited budget, a faster model can serve `task_runner`, while stronger
models serve `research_lead`, `write_director`, and review. Do not infer tool
calling, vision, or context support from a model name alone. Check current
provider capability and run a real smoke test.

## 3.4 Supported providers

Current `provider` values are:

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

Common secret variables:

| Provider | Typical variable | Note |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | Supports a base URL and request options |
| `openrouter` | `OPENROUTER_API_KEY` | Defaults to the OpenRouter API endpoint |
| `deepseek` | `DEEPSEEK_API_KEY` | Uses DeepSeek-specific provider options |
| `anthropic` | `ANTHROPIC_API_KEY` | Native thinking options belong in Anthropic chat kwargs |
| `oai_compatible` | Variable named by `api_key_env` | Verify the compatible service's endpoint and schema |
| `gemini`, `langchain` | Determined by `langchain_class` and kwargs | Use a verified class path from a template |
| `codex_oauth` | No API key | Uses the current system user's OAuth store |

Keep real keys in environment variables or an external secret manager. The
`api_key` field exists, but plaintext secrets do not belong in versioned YAML.

## 3.5 Provider-specific reasoning fields

Reasoning fields are not universal:

- `openrouter` and `openai` use `reasoning.effort`.
- `oai_compatible` and `deepseek` use top-level `reasoning_effort`.
- Anthropic-native settings belong under
  `provider_options.anthropic.chat_kwargs`. CatMaster does not translate
  OpenAI-style `reasoning` into Anthropic `thinking`.
- Provider request fields belong under `provider_options.<provider>`. Do not
  restore the removed top-level `tool_calling_profiles`, model-level
  `tool_calling`, or top-level `extra_body` forms.

The `prompt_cache_retention` setting shown in the standard OpenRouter template
is currently ignored by the adapter with a warning. Do not treat it as an active
cache guarantee.

## 3.6 Single-model environment mode

When the selected YAML file does not exist, a single model can be enabled with:

```bash
export CATMASTER_LLM_PROVIDER=openrouter
export CATMASTER_LLM_MODEL=<OPENROUTER_MODEL_ID>
export OPENROUTER_API_KEY="<YOUR_KEY>"
```

Optional values:

```bash
export CATMASTER_API_KEY_ENV=OPENROUTER_API_KEY
export CATMASTER_BASE_URL=https://openrouter.ai/api/v1
export CATMASTER_TEMPERATURE=1.0
export CATMASTER_REASONING_EFFORT=high
```

This binds one model to every role. It is useful for initial validation, but not
ideal for maintaining a complex deployment.

## 3.7 Codex OAuth

Run device login in the `catmaster` environment:

```bash
python -c \
'from langchain_openai.chatgpt_oauth import login_chatgpt_device; login_chatgpt_device()'

cp configs/llm_codex_oauth.template.yaml configs/llm.yaml
```

OAuth credentials belong to the current system user. Do not copy the token
store, package it, or use a personal OAuth profile as a shared identity for a
multi-user service.

## 3.8 Review, image, and writing settings

`peer_review_models` is a list of reviewer model labels. Each label produces an
independent reviewer report, so list length directly affects calls, cost, and
runtime:

```yaml
peer_review_models:
  - reviewer-a
  - reviewer-b
```

Image generation can use a separate label:

```yaml
image_generation:
  model_label: image-model
  image_config:
    aspect_ratio: "4:3"
```

Writing attribution:

```yaml
writing:
  author_name: "<AUTHOR_NAME>"
```

Whether a model receives images, audio, or video depends on `multimodal`
capabilities and provider behavior. The runtime enables image blocks by default
only for `openai`, `openrouter`, `anthropic`, `gemini`, and `langchain`.
`codex_oauth`, `deepseek`, and `oai_compatible` default to disabled unless the
profile declares support and a real call has verified it. A stored attachment is
not necessarily sent to the model. Inspect `multimodal.prepared` events in
Monitor.

## 3.9 Runtime and Literature settings

The standard YAML runtime defaults are:

```yaml
agent_runtime:
  recursion_limit: 300
  max_tool_calls: 120
  deepagent_context_trigger_token_cap: 270000
  print_state_messages: false
  print_http_raw_post: false
```

These are safety and context-compaction controls, not quality sliders. Raising
them blindly can increase loops, cost, and timeout exposure. In YAML mode, edit
the YAML. Corresponding entries in `.env.example` mainly affect the no-YAML
profile.

The `literature` section controls search depth by role, public-web fallback,
retries, and budgets. Start with the template, then tune it from a recorded
Literature Review run. Deeper search usually means more web calls and context
cost.

## 3.10 Tool-output policy

`CATMASTER_TOOL_OUTPUT_CONFIG` defaults to `configs/tool_output.yaml`. The
current policy is:

```yaml
offload:
  inline_data_enabled: true
  preview_chars: 3000
  offload_chars: 20000
  offload_dir_rel: "_tool_outputs"
```

Complete output above the threshold is written under
`files/_tool_outputs/`; Chat keeps a preview and path. Do not treat
`configs/tool_policy.yaml` as the active specialist-runtime authorization
surface. Actual access is defined by runtime allowlists, task audiences, and
Review interrupts.

## 3.11 Offline configuration check

This command parses the profile without calling a model:

```bash
python -c 'from catmaster.llm.config import LLMProfile; p=LLMProfile.from_env_or_file(); print("models:", sorted(p.models)); print("roles:", p.agents)'
```

After parsing succeeds, start the WebUI and run a short conversation. Successful
parsing does not prove that the key, model ID, endpoint, tool calling, or
multimodal support works.
