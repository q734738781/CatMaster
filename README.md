# CatMaster

CatMaster 是一个面向计算催化工作的本地 agent 工具。日常使用入口是 WebUI：打开一个项目空间，选择任务模式，提交需求，然后在同一个界面里查看运行状态、文件、历史记录和结果。

> 中文为主；英文使用指南见后半部分。

---

## 中文使用指南

### 1. 安装

建议使用独立的 conda 环境：

```bash
conda create -n catmaster python=3.11
conda activate catmaster
pip install -r requirements/pc.txt
```

如果这台机器也要跑本地 GPU / MACE 相关任务，再安装 GPU 依赖：

```bash
pip install -r requirements/gpu.txt
```

如需重新构建 WebUI 前端或使用部署脚本，请确保已安装 Node.js 和 npm：

```bash
node -v
npm -v
```

### 2. 配置

CatMaster 默认读取 `configs/llm.yaml`。第一次使用时建议从模板开始：

```bash
cp configs/llm.template.yaml configs/llm.yaml
```

也可以直接选一个已有预设作为起点：

```bash
cp configs/llm_gemini.yaml configs/llm.yaml
# 或
cp configs/llm_sonnet.yaml configs/llm.yaml
```

如果想临时使用另一个配置文件，而不覆盖 `configs/llm.yaml`：

```bash
export CATMASTER_LLM_CONFIG=configs/llm_gemini.yaml
```

LLM 配置文件里主要改两处：

- `models`：给每个模型起一个本地标签，设置 provider、model、base URL 和 provider 专属参数。
- `agents`：把不同任务角色绑定到这些模型标签。最少需要 `proposal`、`director`、`task_runner`、`memory_patch`、`summary`；模板里已经给出完整示例。

内置 provider 包括 `openrouter`、`openai`、`deepseek`、`anthropic`、`codex_oauth`、`oai_compatible` 和通用 `langchain`。Anthropic 与 Codex OAuth 都只做薄适配：YAML 中的 `provider_options.<provider>.chat_kwargs` 会直接传给对应 LangChain ChatModel；DeepSeek 的 reasoning/tool-call 兼容 patch 仍只作用于 `provider: deepseek`。

不要把 API key 写进 YAML。常用 key 通过环境变量提供：

```bash
export OPENROUTER_API_KEY="..."
# 或
export OPENAI_API_KEY="..."
# 或
export DEEPSEEK_API_KEY="..."
# 或
export ANTHROPIC_API_KEY="..."
```

Codex OAuth 使用非官方 `langchain-codex-oauth` 包，不使用 API key。第一次使用前在 `catmaster` conda 环境里登录一次：

```bash
langchain-codex-oauth auth login
# 远程机器或端口受限时：
langchain-codex-oauth auth login --manual
```

`configs/llm.template.yaml` 已包含 Anthropic 与 Codex OAuth 示例。最小模型片段如下：

```yaml
models:
  anthropic-main:
    provider: anthropic
    model: claude-sonnet-4-5-20250929
    api_key_env: ANTHROPIC_API_KEY
    provider_options:
      anthropic:
        chat_kwargs: {}

  codex-oauth-main:
    provider: codex_oauth
    model: gpt-5.2-codex
    provider_options:
      codex_oauth:
        chat_kwargs:
          system_prompt_mode: strict
```

可选服务：

```bash
export TAVILY_API_KEY="..."   # 公共网页/文献检索
export MP_API_KEY="..."       # Materials Project 结构检索
```

如果完全不使用 YAML，也可以用环境变量快速指定一个单模型配置：

```bash
export CATMASTER_LLM_PROVIDER=openrouter
export CATMASTER_LLM_MODEL=openai/gpt-5.5
export OPENROUTER_API_KEY="..."
```

其他常用配置文件：

- `configs/tool_output.yaml`：控制长工具输出是否写入 `_tool_outputs/`。
- `configs/tool_policy.yaml`：允许或禁用某些工具。
- `configs/dpdispatcher/resources.yaml` 和 `configs/dpdispatcher/tasks.yaml`：远程计算任务的资源和命令。
- `configs/machines_template.yaml`：远程机器模板；复制为 `configs/dpdispatcher/machines.yaml` 后填写自己的集群账号和环境。

### 3. 准备项目空间

项目空间用于保存输入、输出、运行记录和中间文件。可以使用任意本地目录：

```bash
mkdir -p ~/catmaster_projects
```

启动 WebUI 时通过 `CATMASTER_PROJECT_SPACE_ROOT` 或 `--project-space-root` 指定。

### 4. 启动 WebUI

最常用方式：

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh
```

如果 conda 环境名不是 `catmaster`：

```bash
CATMASTER_CONDA_ENV=your_env_name CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh
```

查看状态或停止后台服务：

```bash
./start_webui.sh --status
./start_webui.sh --stop
```

前台运行，方便查看日志：

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh --foreground
```

指定端口：

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh --port 7991
```

也可以直接用 Python 启动：

```bash
python -m catmaster.webui --project-space-root ~/catmaster_projects --host 127.0.0.1 --port 7860
```

默认脚本端口通常是：

```text
http://127.0.0.1:7990
```

如果你手动指定了端口，请打开对应端口。

### 5. 基本使用流程

1. 打开 WebUI。
2. 选择或新建一个 project space。
3. 上传已有结构、计算结果、文稿或数据文件；也可以让 agent 新建文件。
4. 选择任务模式。
5. 在输入框里写清楚目标、已知约束和希望输出的文件格式。
6. 运行后在界面中查看进度、工具调用、文件树和结果。
7. 需要继续未完成任务时，选择历史 run，并使用 `resume_selected_run`。

### 6. 任务模式

#### Experiment

用于边界明确的计算任务，例如准备结构和输入文件、提交或分析计算、从已有结果中提取数据。

```text
读取当前项目里的 Ni slab 和 CO 分子，生成 CO 在顶位、桥位、空位的吸附结构，并为每个结构准备 VASP 输入。
```

#### Research

用于更开放的研究问题，适合把一个催化方向拆成若干轮文献、计算和结果整理任务。

```text
围绕 CO2 加氢到甲醇，帮我制定一轮 Cu 基催化剂筛选计划。先结合文献给出候选体系，再安排可执行的计算任务。
```

#### Literature Review

用于直接启动 LitReview Agent，完成文献综述、公开来源核查、代表性论文整理和 DOI / 年份 / 期刊 / 作者等元数据确认。

```text
综述近五年单原子 Ni 催化 CO2 电还原生成 CO 的代表性工作，按催化剂结构、活性指标、关键证据和 DOI 整理。
```

#### Writing

用于基于项目空间里的证据写作、修改和整理文稿。

```text
根据当前项目中的计算结果和图表，写一版 ACS 风格的 Results and Discussion，输出 TeX。
```

#### Peer Review

用于对已有 PDF 文稿做审稿式检查，并给出编辑意见和 reviewer 风格评论。

```text
审阅 files/manuscript.pdf，重点检查催化机理证据是否充分、计算方法是否可复现、结论是否被数据支持。
```

### 7. 可选外部程序

只在你需要对应任务时安装：

- OVITO：结构渲染和结构视图导出。
- LaTeX / `pdflatex`：TeX 文稿编译。
- VASPKIT：吸附物和气相热力学校正。
- ORCA、xTB、CREST、VASP、MACE：对应量化、半经验、构象搜索、第一性原理和机器学习势任务。

如果要准备或提交 VASP 任务，请先在本机或远程机器上准备好 VASP 运行环境，并按 pymatgen 的要求配置 POTCAR。

远程提交使用 `configs/dpdispatcher/` 下的配置文件。可以从模板开始：

```bash
cp configs/machines_template.yaml configs/dpdispatcher/machines.yaml
```

然后按你的集群账号、队列、环境加载方式和执行命令修改：

```text
configs/dpdispatcher/machines.yaml
configs/dpdispatcher/resources.yaml
configs/dpdispatcher/tasks.yaml
```

### 8. 部署到运行目录

如果想把当前代码同步到一个独立运行目录：

```bash
scripts/deploy_runtime.sh --target ../CatMaster_Run --no-autorun
```

然后进入目标目录启动：

```bash
cd ../CatMaster_Run
CATMASTER_PROJECT_SPACE_ROOT=./project_space ./start_webui.sh
```

默认会保留目标目录已有的 `start_webui.sh` 本地端口和路径设置。若要强制同步当前仓库里的启动脚本：

```bash
scripts/deploy_runtime.sh --target ../CatMaster_Run --sync-start-webui
```

### 9. 说明

- 计算任务是否能成功，取决于你本机或远程机器上的外部程序、队列系统和环境变量是否已经配置好。
- WebUI 的 run 历史、日志、产物和中间文件都会保存在对应 project space 中。
- `devdocs/` 和 `docs/` 主要面向开发和内部记录；日常使用以本 README 为准。

---

## English Guide

### 1. Installation

Use a dedicated conda environment:

```bash
conda create -n catmaster python=3.11
conda activate catmaster
pip install -r requirements/pc.txt
```

If the same machine will run local GPU / MACE tasks, also install:

```bash
pip install -r requirements/gpu.txt
```

If you need to rebuild the WebUI frontend or use the deployment script, make sure Node.js and npm are available:

```bash
node -v
npm -v
```

### 2. Configuration

CatMaster reads `configs/llm.yaml` by default. For a first setup, start from the template:

```bash
cp configs/llm.template.yaml configs/llm.yaml
```

You can also start from one of the included presets:

```bash
cp configs/llm_gemini.yaml configs/llm.yaml
# or
cp configs/llm_sonnet.yaml configs/llm.yaml
```

To use another config file without replacing `configs/llm.yaml`:

```bash
export CATMASTER_LLM_CONFIG=configs/llm_gemini.yaml
```

The LLM config mainly has two sections:

- `models`: local labels for provider/model/base URL and provider-specific parameters.
- `agents`: role-to-model bindings. At minimum configure `proposal`, `director`, `task_runner`, `memory_patch`, and `summary`; the template includes the optional roles.

Built-in providers include `openrouter`, `openai`, `deepseek`, `anthropic`, `codex_oauth`, `oai_compatible`, and generic `langchain`. Anthropic and Codex OAuth are thin adapters: `provider_options.<provider>.chat_kwargs` is passed through to the matching LangChain ChatModel. The DeepSeek reasoning/tool-call compatibility patch remains scoped to `provider: deepseek`.

Keep secrets out of YAML. Provide API keys through environment variables:

```bash
export OPENROUTER_API_KEY="..."
# or
export OPENAI_API_KEY="..."
# or
export DEEPSEEK_API_KEY="..."
# or
export ANTHROPIC_API_KEY="..."
```

Codex OAuth uses the non-official `langchain-codex-oauth` package and does not use an API key. Log in once inside the `catmaster` conda environment:

```bash
langchain-codex-oauth auth login
# On remote machines or when the callback port is unavailable:
langchain-codex-oauth auth login --manual
```

`configs/llm.template.yaml` includes Anthropic and Codex OAuth examples. Minimal model snippets:

```yaml
models:
  anthropic-main:
    provider: anthropic
    model: claude-sonnet-4-5-20250929
    api_key_env: ANTHROPIC_API_KEY
    provider_options:
      anthropic:
        chat_kwargs: {}

  codex-oauth-main:
    provider: codex_oauth
    model: gpt-5.2-codex
    provider_options:
      codex_oauth:
        chat_kwargs:
          system_prompt_mode: strict
```

Optional services:

```bash
export TAVILY_API_KEY="..."   # public web / literature search
export MP_API_KEY="..."       # Materials Project access
```

For a quick single-model setup without YAML:

```bash
export CATMASTER_LLM_PROVIDER=openrouter
export CATMASTER_LLM_MODEL=openai/gpt-5.5
export OPENROUTER_API_KEY="..."
```

Other useful config files:

- `configs/tool_output.yaml`: controls whether long tool outputs are stored under `_tool_outputs/`.
- `configs/tool_policy.yaml`: allows or blocks selected tools.
- `configs/dpdispatcher/resources.yaml` and `configs/dpdispatcher/tasks.yaml`: remote-job resources and commands.
- `configs/machines_template.yaml`: remote-machine template; copy it to `configs/dpdispatcher/machines.yaml` and fill in your cluster account and environment.

### 3. Project Space

A project space stores inputs, outputs, run records, and intermediate files. Any local directory can be used:

```bash
mkdir -p ~/catmaster_projects
```

Pass it to the WebUI with `CATMASTER_PROJECT_SPACE_ROOT` or `--project-space-root`.

### 4. Start The WebUI

Recommended:

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh
```

If your conda environment is not named `catmaster`:

```bash
CATMASTER_CONDA_ENV=your_env_name CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh
```

Check or stop the background service:

```bash
./start_webui.sh --status
./start_webui.sh --stop
```

Run in the foreground:

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh --foreground
```

Use a custom port:

```bash
CATMASTER_PROJECT_SPACE_ROOT=~/catmaster_projects ./start_webui.sh --port 7991
```

You can also start it directly with Python:

```bash
python -m catmaster.webui --project-space-root ~/catmaster_projects --host 127.0.0.1 --port 7860
```

The default launcher port is usually:

```text
http://127.0.0.1:7990
```

If you set a custom port, open that port instead.

### 5. Basic Workflow

1. Open the WebUI.
2. Select or create a project space.
3. Upload structures, calculation outputs, manuscripts, or data files; the agent can also create files for you.
4. Choose a task lane.
5. Describe the goal, constraints, and desired output format.
6. Watch progress, tool calls, files, and results in the interface.
7. To continue an interrupted run, select the historical run and use `resume_selected_run`.

### 6. Task Lanes

#### Experiment

Use this for bounded computational tasks, such as preparing structures and input files, launching or analyzing calculations, and extracting data from existing outputs.

```text
Read the Ni slab and CO molecule in the current project, generate CO adsorption structures at top, bridge, and hollow sites, and prepare VASP inputs for each structure.
```

#### Research

Use this for broader research questions that may require literature review, multiple computational steps, and result synthesis.

```text
For CO2 hydrogenation to methanol, design a first-round Cu-based catalyst screening plan. Start from literature-supported candidates, then propose executable calculations.
```

#### Literature Review

Use this to launch LitReview Agent directly for literature synthesis, public-source checking, representative paper lists, and DOI / year / venue / author metadata verification.

```text
Review representative work from the last five years on single-atom Ni catalysts for electrochemical CO2-to-CO conversion. Organize by catalyst structure, activity metrics, key evidence, and DOI.
```

#### Writing

Use this to draft, revise, or organize manuscripts from evidence already present in the project space.

```text
Based on the calculation results and figures in this project, draft an ACS-style Results and Discussion section in TeX.
```

#### Peer Review

Use this for reviewer-style assessment of an existing PDF manuscript.

```text
Review files/manuscript.pdf. Focus on whether the catalytic mechanism evidence is sufficient, the computational methods are reproducible, and the conclusions are supported by data.
```

### 7. Optional External Programs

Install only what your tasks require:

- OVITO: structure rendering and exported structure views.
- LaTeX / `pdflatex`: TeX manuscript compilation.
- VASPKIT: adsorbate and gas-phase thermochemistry corrections.
- ORCA, xTB, CREST, VASP, MACE: quantum chemistry, semiempirical calculations, conformer search, first-principles calculations, and machine-learning potential tasks.

For VASP preparation or submission, prepare the VASP runtime and configure POTCAR for pymatgen.

Remote submission uses files under `configs/dpdispatcher/`. Start from:

```bash
cp configs/machines_template.yaml configs/dpdispatcher/machines.yaml
```

Then edit these files for your cluster account, queue, environment setup, and execution commands:

```text
configs/dpdispatcher/machines.yaml
configs/dpdispatcher/resources.yaml
configs/dpdispatcher/tasks.yaml
```

### 8. Deploy To A Runtime Directory

To sync the current checkout to a separate runtime directory:

```bash
scripts/deploy_runtime.sh --target ../CatMaster_Run --no-autorun
```

Then start from the target directory:

```bash
cd ../CatMaster_Run
CATMASTER_PROJECT_SPACE_ROOT=./project_space ./start_webui.sh
```

By default, an existing target `start_webui.sh` is preserved so local port and path settings remain stable. To force-sync the launcher from this checkout:

```bash
scripts/deploy_runtime.sh --target ../CatMaster_Run --sync-start-webui
```

### 9. Notes

- Computational success depends on your local or remote external programs, queue system, and environment variables.
- WebUI run history, logs, outputs, and intermediate files are stored inside the selected project space.
- `devdocs/` and `docs/` are mainly for development notes; this README is the user-facing guide.
