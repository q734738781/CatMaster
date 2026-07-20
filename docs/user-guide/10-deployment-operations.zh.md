# 10. 部署、运维与安全

[上一章](09-tools-skills-evolution.zh.md) | [目录](README.zh.md) | [下一章](11-reference-troubleshooting.zh.md)

本章讨论 control plane 的部署和本地辅助程序。VASP、CP2K、LAMMPS、ORCA、xTB、CREST 和 MLFF provider 属于受管远程执行环境，不应混装进 CatMaster control plane。

## 10.1 三种常见部署

### 本地工作站

WebUI 和浏览器在同一台机器：

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
CATMASTER_HOST=127.0.0.1 \
CATMASTER_PORT=7991 \
./start_webui.sh
```

### 服务器加 SSH 隧道

服务器仍只监听 loopback：

```bash
CATMASTER_PROJECT_SPACE_ROOT=/srv/catmaster/projects \
CATMASTER_HOST=127.0.0.1 \
CATMASTER_PORT=7991 \
./start_webui.sh
```

用户电脑建立隧道：

```bash
ssh -L 7991:127.0.0.1:7991 <USER>@<SERVER>
```

然后打开 `http://127.0.0.1:7991`。

### 共享 Web 服务

共享服务应置于 TLS reverse proxy、VPN、IP allowlist 或外部身份层之后。内置认证默认允许自助注册，session cookie 没有 Secure 标志，应用本身不提供 TLS。不要把 `0.0.0.0:7991` 直接暴露到公网，也不要公开 `--no-login`。

## 10.2 账号和文件权限

运行 WebUI 的系统用户可以访问全部 project-space 数据和活动配置，因此：

- 使用专用、非 root 系统用户。
- Project root、`.webui_auth`、活动 YAML、SSH key 和 browser profile 只授权给该用户。
- 不让 WebUI 用户对仓库源代码和共享 secret 目录拥有不必要写权限。
- Project root 不放在 Web server 的静态目录。
- 多用户部署使用独立外部访问控制，审查开放注册风险。

## 10.3 `.env.local` 和 secret

从清单建立私有文件：

```bash
cp .env.example .env.local
chmod 600 .env.local
```

加载：

```bash
set -a
source .env.local
set +a
```

生产环境更适合 systemd EnvironmentFile 或 secret manager。不要把 key 写进 `configs/llm.yaml`，不要把 SSH key、license、token 或 cookie 放进 workspace。确认 `.env.local`、活动 `configs/dpdispatcher/*.yaml` 和 project root 不在版本控制或部署包中。

## 10.4 运行目录和日志

默认后台状态：

```text
.runtime/webui.pid
.runtime/webui.log
```

可覆盖：

```bash
export CATMASTER_RUNTIME_DIR=/var/tmp/catmaster-runtime
export CATMASTER_WEBUI_LOG=/var/log/catmaster/webui.log
export CATMASTER_WEBUI_PID=/var/run/user/<UID>/catmaster-webui.pid
```

目标目录必须由运行用户创建并可写。常用命令：

```bash
./start_webui.sh --status
tail -f .runtime/webui.log
./start_webui.sh --stop
```

启动脚本停止时先等待最多 30 秒，再对仍未退出的记录进程发送强制终止。停止本地 WebUI 不会取消远程调度作业。

## 10.5 运行时同步部署

`scripts/deploy_runtime.sh` 更新另一个 runtime 目录。默认 target 是 `../CatMaster_Run`，默认 runtime-only、会删除目标中源端已删除文件、构建前端、自动启动，并保留目标已有 config 和 launcher。

第一次先做非破坏预览：

```bash
scripts/deploy_runtime.sh \
  --target /path/to/CatMaster_Run \
  --project-space-root /path/to/catmaster_projects \
  --dry-run \
  --no-delete \
  --no-autorun
```

确认变更后去掉 `--dry-run`。重要选项：

- `--sync-configs` 会覆盖目标 `configs/`，可能破坏私有 LLM 和机器配置。
- `--sync-start-webui` 会覆盖目标 launcher。
- 不加 `--no-delete` 时，目标中已从源端删除的 runtime 文件会被删除。
- 默认 `--autorun`，维护窗口不希望立即启动时显式用 `--no-autorun`。
- `--full-repo` 扩大同步范围，不能当普通 runtime 更新默认使用。

## 10.6 离线部署包

生成包：

```bash
scripts/package_remote_deploy.sh --output-dir dist
```

默认包只含公开 DPDispatcher 模板，不含活动机器配置、`.env`、project space、日志和大型计算中间文件。生成后检查 archive 清单和 checksum，再传输。

`--include-path` 会把额外路径加入包。使用前确认其中没有 key、token、POTCAR、WAVECAR、CHGCAR、个人浏览器状态或未授权数据。`--no-verify` 会跳过包后验证，不建议在正式交付中使用。

## 10.7 升级和回滚

升级顺序：

1. 停止 WebUI，确认没有正在写本地状态的 run。
2. 备份 project root、账号数据库和私有配置。
3. 更新代码或部署包。
4. 更新 control plane 环境：

   ```bash
   conda env update -n catmaster -f requirements/pc-conda.yml
   ```

5. 对 LLM YAML 做无网络解析检查。
6. 前台启动，做登录、thread、文件读取和最短 LLM smoke。
7. 远程配置只先做 `--list` 和单个最小 case。
8. 验收后切回后台。

回滚代码时不要回滚 project-space 数据到不兼容的旧格式。保留升级前快照，并分别记录代码版本、配置版本和数据快照时间。

## 10.8 备份

完整备份范围：

```text
<PROJECT_SPACE_ROOT>/users/.../<workspace>/files/
<PROJECT_SPACE_ROOT>/users/.../<workspace>/metadata/
<PROJECT_SPACE_ROOT>/.webui_auth/auth.sqlite
configs/llm.yaml
configs/dpdispatcher/ 里的活动私有配置
.env.local 或外部 secret 定义
```

Secret 和项目数据可使用不同加密备份策略。恢复时先恢复目录权限，再启动 WebUI。只恢复 `files/` 会丢失线程和 checkpoint；只恢复 `metadata/` 则缺少实际 artifact。

## 10.9 agent-browser

版本固定为：

```bash
npm install -g agent-browser@0.31.1
agent-browser install
agent-browser doctor --offline --quick
agent-browser mcp --help
```

Profile 放在 workspace 之外：

```bash
export CATMASTER_AGENT_BROWSER_PROFILE="$HOME/.config/catmaster/browser-profile"
```

无图形服务器可以使用已有本地语料和网页搜索，但不能声称已获得需要交互登录的机构全文访问。需要交互式登录时使用安全的图形会话或把全文文件由用户合法上传。

## 10.10 JSmol

WebUI 用 JSmol 做结构和轨迹预览。启动器会调用安装脚本，在缓存缺失时下载固定资源。离线服务器先在有网络环境预热持久缓存：

```bash
CATMASTER_JSMOL_CACHE_DIR=/persistent/cache/jsmol \
python scripts/install_jsmol_assets.py
```

部署时保持同一 `CATMASTER_JSMOL_CACHE_DIR`，并确认运行用户可读。JSmol 缺失只影响相应预览，不应被误诊为 LLM 或远程执行失败。

## 10.11 VASPKIT 和 VESTA

VASPKIT 查找顺序：

1. `CATMASTER_VASPKIT_BIN`。
2. `PATH` 中的 `vaspkit`。
3. 常见用户路径，例如 `~/vaspkit/bin/vaspkit`。

```bash
export CATMASTER_VASPKIT_BIN=/opt/vaspkit/bin/vaspkit
```

VESTA 查找顺序类似，可显式设置：

```bash
export CATMASTER_VESTA_BIN=/opt/VESTA/VESTA
export CATMASTER_XVFB_RUN=/usr/bin/xvfb-run
```

无 DISPLAY 的服务器渲染通常需要 Xvfb。VESTA 和 VASPKIT 是可选本地辅助工具，不是受管 VASP 计算引擎，也不随 CatMaster 提供许可证。

## 10.12 Pandoc、Chrome、字体和 TeX

Markdown PDF 使用 Pandoc 生成 HTML5/MathML，再由 headless Chrome/Chromium 打印。可显式设置：

```bash
export CATMASTER_PANDOC_BIN=/usr/bin/pandoc
export CATMASTER_CHROME_BIN=/usr/bin/chromium
```

检查：

```bash
pandoc --version
chromium --version
fc-match sans
fc-match "Noto Sans CJK SC"
pdflatex --version
bibtex --version
```

缺少 CJK 字体时即使编译成功也可能出现方框或字体替换。最终以 PDF 视觉检查为准。

## 10.13 PySR 和 Julia

PySR 首次 import 可能下载 Julia 并预编译。联网部署期执行：

```bash
python scripts/pysr_julia_smoke.py --fit
```

离线机器预装 Julia，并指定：

```bash
export PYTHON_JULIACALL_BINDIR=/opt/julia/bin
python scripts/pysr_julia_smoke.py \
  --julia-bindir "$PYTHON_JULIACALL_BINDIR" --fit
```

把首次下载和预编译放在维护窗口，不要让第一个用户任务承担它。

## 10.14 远程科学引擎

以下程序配置在 resource 的远程环境中：

- VASP，通常为 `vasp_std`，Gamma case 可用 `vasp_gam`。
- CP2K，模板命令使用 `cp2k.psmp`。
- LAMMPS，boot script 探测常见 CPU/GPU/KOKKOS binary，也可用远程 `CATMASTER_LAMMPS_BIN`。
- ORCA，多 rank 时还需要正确的 MPI 启动器。
- xTB 和 CREST。
- MACE、FairChem UMA、MatterSim 和 ORB-v3 隔离环境。

程序可执行不等于许可证、模型权重、势文件和队列政策允许使用。管理员必须在开放 task 前完成站点验收。
