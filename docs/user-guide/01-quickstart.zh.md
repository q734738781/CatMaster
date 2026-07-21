# 1. 快速安装与第一次对话

[English](01-quickstart.en.md) | [目录](README.zh.md) | [下一章](02-concepts.zh.md)

如果管理员已经给你 CatMaster 地址，直接从"第一次进入 WebUI"开始。自己在本机部署时，先完成下面的最小安装。更完整的模型路由、服务器部署和外部程序配置在[第 10 章](10-deployment-operations.zh.md)。

## 最小本地安装

CatMaster 的 control plane 使用统一 conda 环境。在仓库根目录执行：

```bash
conda env create -f requirements/pc-conda.yml
conda activate catmaster
```

已有环境可以更新：

```bash
conda env update -n catmaster -f requirements/pc-conda.yml
```

`requirements/mace.txt`、`requirements/uma.txt`、`requirements/mattersim.txt` 和 `requirements/orb.txt` 是远程 MLFF 环境依赖，不要用它们替代 control plane 环境。

## 配置一个可用模型

首次安装且 `configs/llm.yaml` 不存在时，复制标准模板。已有文件应保留并直接编辑，不要覆盖：

```bash
cp -n configs/llm.template.yaml configs/llm.yaml
```

模板默认使用 OpenRouter 模型标签。设置 key：

```bash
export OPENROUTER_API_KEY="<YOUR_KEY>"
```

如果需要长期保存本机变量，可以从 `.env.example` 创建私有文件：

```bash
cp -n .env.example .env.local
chmod 600 .env.local
```

程序不会自动读取 `.env.local`。启动前这样加载：

```bash
set -a
source .env.local
set +a
```

不要把真实 key 提交到 Git。使用 OpenAI、Anthropic、DeepSeek、Gemini、兼容 endpoint 或 Codex OAuth 时，按[模型配置](10-deployment-operations.zh.md#配置-llm)修改 provider、model 和对应字段，不要只替换环境变量名。

## 启动 WebUI

创建项目根目录，并显式绑定本机地址：

```bash
mkdir -p "$HOME/catmaster_projects"

CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
CATMASTER_HOST=127.0.0.1 \
CATMASTER_PORT=7991 \
./start_webui.sh
```

浏览器打开：

```text
http://127.0.0.1:7991
```

首次启动可能安装固定版本的 JSmol 资源，用于结构和轨迹预览，因此会比后续启动慢。查看状态和日志：

```bash
./start_webui.sh --status
tail -f .runtime/webui.log
```

需要直接看到错误时以前台模式启动：

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
./start_webui.sh --foreground --host 127.0.0.1 --port 7991
```

停止后台服务：

```bash
./start_webui.sh --stop
```

## 第一次进入 WebUI

默认启动会显示登录页。注册一个账号后，系统为你创建个人项目区域和 `default` workspace。共享服务器上的不同用户只能进入各自目录。

第一次体验建议新建一个名为 `quickstart` 的 workspace，再新建 thread。选择 `Experiment`，把权限模式设为 `Review`。这样可以看到 Agent 如何委派 Materials worker 和使用工具，也会让后续通用文件编辑或远程提交进入人工审批。

上传任意一份 CIF 或 POSCAR。仓库源码安装的用户可以直接使用 `tests/assets/Fe.cif` 作为无科学意义的界面测试样例。发送：

```text
请使用 Experiment 检查我刚上传的晶体结构。先识别文件路径、元素、晶胞、
周期性、原子数和是否存在异常短距，再让 Materials worker 生成 2x2x2 超胞，
写到 quickstart/Fe_2x2x2.vasp。

自主选择合适的结构 tool，并说明变换前后的晶胞和原子数。
本轮只做结构操作，不查询远程 task，也不提交任何计算。
```

这条请求会让你看到 CatMaster 的基本工作方式。Chat 中先出现 Progress，随后可能看到 `materials_worker` 委派和 `supercell` tool 卡。`supercell` 会在一次领域 tool 调用中直接写出声明的目标文件，所以当前 Review 模式不一定为这一步显示审批卡。发送前把输出路径写清楚，调用时核对 tool 参数，完成后再检查 artifact 和 Files 中的真实文件。生成的结构应能用 JSmol 预览。

这不是对 Fe 的正式建模，只是同时验证以下组件：

- LLM profile 能正常调用并支持当前 tool schema。
- Experiment 可以委派 Materials worker。
- Worker 能读取附件、执行结构工具并写入 workspace。
- Artifact、Files 预览和 Monitor 事件能够对应同一次结构操作。

Review 的审批卡可以在后续 `write_file`、`edit_file` 或远程提交时看到。它并不拦截所有会产生文件的领域 tool；第 4 章给出准确边界。

如果你更关心文献工作，也可以上传一篇 PDF，选择 Literature Review，并发送：

```text
精读这篇论文。先确认附件路径和可解析页数，再说明论文的研究问题、主要证据链和限制。
把方法、作者直接观察到的结果和作者推测分开，保留页码或原文锚点。
本轮先在 Chat 中给出阅读计划，不下载其他论文，也不写综述。
```

## 第一次运行时看哪里

Chat 用于对话和查看 Agent、worker 与 tools 的活动。Files 用于确认结构、表格和报告是否真实写入项目。Monitor 可以查看模型调用、工具状态、错误和运行规模。第一次任务结束后，建议依次打开三处，建立"回复、过程、文件"之间的对应关系。

如果 Agent 回复正常但没有产生你要求的文件，展开 tool 卡检查是否被 Review 拒绝、路径是否错误或工具是否返回 warning。如果模型完全不能调用工具，先看 Monitor 和 WebUI 日志，再按[故障排查](11-reference-troubleshooting.zh.md)核对模型能力与配置。

## 暂时不需要配置的内容

第一次本地对话不需要 VASP、CP2K、LAMMPS、ORCA、xTB、CREST、MACE 或集群账号。也不需要先配置全部文献 API、浏览器 profile、VESTA、VASPKIT、Pandoc 或 LaTeX。这些能力可以在基础 WebUI 验证通过后按需添加。

如果只在可信单机上临时测试，可以关闭登录：

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
./start_webui.sh --foreground --host 127.0.0.1 --port 7991 --no-login
```

无登录模式进入开放 `admin` workspace，并关闭 Skill Evolution。不要让它监听局域网或公网地址。

下一章解释刚才看到的 Agent、worker、skill、tool 和 artifact 之间是什么关系。理解这套关系后，再阅读具体功能会更自然。
