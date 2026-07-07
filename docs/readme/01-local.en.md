# Local Setup

Goal: start the CatMaster WebUI on a Linux machine and connect it to a working LLM. Remote execution, VASP, MACE, and other external scientific programs are not required for this chapter.

## 1. Prepare The Python Environment

Use a dedicated conda environment:

```bash
conda env create -f requirements/pc-conda.yml
conda activate catmaster
```

`requirements/pc-conda.yml` is the single source of truth for the PC/control-plane environment. It lets conda solve the scientific/materials stack and keeps the exact-pinned LLM/WebUI pip packages inline in the same file. Do not install the scientific stack directly with pip.

If this machine will also run local GPU / MACE tasks, install the GPU/MACE add-on dependencies as well. `requirements/gpu.txt` is not a complete WebUI/agent environment and does not replace `requirements/pc-conda.yml`:

```bash
pip install -r requirements/gpu.txt
```

If you need to rebuild the WebUI frontend or use the deployment script, confirm Node.js and npm are available:

```bash
node -v
npm -v
```

## 2. Configure The LLM

CatMaster reads this default file:

```text
configs/llm.yaml
```

For a first setup:

```bash
cp configs/llm.template.yaml configs/llm.yaml
```

To inspect every supported field and provider shape:

```bash
cp configs/llm.full.template.yaml configs/llm.yaml
```

You can also start from a preset:

```bash
cp configs/llm_gemini.yaml configs/llm.yaml
# or
cp configs/llm_sonnet.yaml configs/llm.yaml
```

The LLM config mainly has two sections:

- `models`: local labels for `provider`, `model`, `base_url`, and provider-specific parameters.
- `agents`: role-to-model bindings. At minimum configure `proposal`, `director`, `task_runner`, `memory_patch`, and `summary`.

Reasoning fields are provider-specific:

- `openrouter` and official `openai`: use `reasoning.effort`, for example `reasoning: {effort: high}`.
- `oai_compatible`: use top-level `reasoning_effort`, for example `reasoning_effort: high`. CatMaster currently uses the `langchain-openai` chat-completions path here and does not translate `reasoning.effort` into `reasoning_effort`.
- `deepseek`: use top-level `reasoning_effort`; DeepSeek-specific fields such as `thinking` belong under `provider_options.deepseek.extra_body`.

## 3. Provide API Keys

Do not put real API keys in YAML. Use environment variables:

```bash
export OPENROUTER_API_KEY="..."
# or
export OPENAI_API_KEY="..."
# or
export DEEPSEEK_API_KEY="..."
# or
export ANTHROPIC_API_KEY="..."
```

Optional services:

```bash
export TAVILY_API_KEY="..."   # public web / literature search
export MP_API_KEY="..."       # Materials Project access
```

`.env.example` is a variable checklist. CatMaster does not automatically load `.env.local`; if you create one, source it manually:

```bash
source .env.local
```

## 4. Quick Single-Model Setup Without YAML

For a quick trial, you can skip `configs/llm.yaml` and use environment variables:

```bash
export CATMASTER_LLM_PROVIDER=openrouter
export CATMASTER_LLM_MODEL=openai/gpt-5.2
export OPENROUTER_API_KEY="..."
```

If `configs/llm.yaml` exists, CatMaster reads it by default. To temporarily use another config file:

```bash
export CATMASTER_LLM_CONFIG=configs/llm_gemini.yaml
```

## 5. Codex OAuth

Codex OAuth uses the non-official `langchain-codex-oauth` package and does not use an API key. Log in once inside the `catmaster` environment:

```bash
langchain-codex-oauth auth login
```

For remote machines or restricted callback ports:

```bash
langchain-codex-oauth auth login --manual
```

Examples are included in `configs/llm.template.yaml` and `configs/llm.full.template.yaml`.

## 6. Prepare A Project Space

A project space stores inputs, outputs, run history, intermediate files, and reports. Use a dedicated directory:

```bash
mkdir -p ~/catmaster_projects
```

Pass it to the WebUI:

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh
```

## 7. Start The WebUI

Start in the background:

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh
```

Open:

```text
http://127.0.0.1:7990
```

If your conda environment is not named `catmaster`:

```bash
CATMASTER_CONDA_ENV=your_env_name CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh
```

Run in the foreground for logs:

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh --foreground
```

Check status or stop:

```bash
./start_webui.sh --status
./start_webui.sh --stop
```

Use a custom port:

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh --port 7991
```

Start directly with Python:

```bash
python -m catmaster.webui --project-space-root ~/catmaster_projects --host 127.0.0.1 --port 7860
```

## 8. Troubleshooting

`Missing API key`

Confirm the variable is exported in the current shell:

```bash
echo "$OPENROUTER_API_KEY"
```

`conda is not available in PATH`

Initialize conda first, or run the launcher from a shell where conda is already available.

WebUI does not start

Run it in the foreground:

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh --foreground
```

Port conflict

Use another port:

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh --port 7991
```

## 9. Next Steps

- Read [Features and daily workflows](03-features.en.md) to understand task lanes and project spaces.
- Read [Remote setup](02-remote.en.md) only if you need cluster submission.
