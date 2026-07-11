# 外部材料软件配置

CatMaster 不随部署包分发 VASPKIT、VESTA、VASP 等外部程序。程序应由用户按各自许可证安装到本机或计算节点；CatMaster 只探测可执行文件并调用已安装版本。

## 1. 可执行文件发现顺序

VASPKIT 的发现顺序为：

1. `CATMASTER_VASPKIT_BIN` 指定的可执行文件或安装目录。
2. `PATH` 中的 `vaspkit`。
3. 兼容路径 `~/vaspkit/bin/vaspkit`。

VESTA 的发现顺序为：

1. `CATMASTER_VESTA_BIN` 指定的 `VESTA` launcher 或安装目录。
2. `PATH` 中的 `VESTA` 或 `vesta`。
3. 少量常见用户目录。

建议显式配置，避免 WebUI、IDE 和终端使用不同的 `PATH`：

```bash
export CATMASTER_VASPKIT_BIN="$HOME/vaspkit/bin/vaspkit"
export CATMASTER_VESTA_BIN="$HOME/.local/opt/VESTA-gtk3/VESTA"
```

这些变量也列在 `.env.example` 中。CatMaster 不会自动加载 `.env` 文件；应在启动 WebUI 的 shell 中导出变量，或写入自己的 shell profile。

## 2. VASPKIT

按照 VASPKIT 官方说明完成安装后，先验证可执行文件：

```bash
test -x "$CATMASTER_VASPKIT_BIN" && echo "VASPKIT executable found"
```

当前 CatMaster 使用 VASPKIT 任务 501 和 502 做吸附物及气相热力学校正。若 VASPKIT 不可用，部分工具可明确标记后使用 ASE 近似；报告必须保留实际 backend 标签。

## 3. VESTA

VESTA 用于导出适合报告和多模态模型读取的原子结构图。`materials_worker` 的 `render_vesta_views` 会生成 top、side、isometric 三个标准视角、一个拼版 PNG 和一份 JSON 元数据；随后 agent 用 `read_file` 读取拼版图进行视觉检查。

从 [VESTA 官方下载页](https://jp-minerals.org/vesta/en/download.html) 获取适合系统的版本。Linux 示例：

```bash
mkdir -p "$HOME/.local/opt"
tar -xjf VESTA-gtk3.tar.bz2 -C "$HOME/.local/opt"
export CATMASTER_VESTA_BIN="$HOME/.local/opt/VESTA-gtk3/VESTA"
test -x "$CATMASTER_VESTA_BIN" && echo "VESTA executable found"
```

Linux 还需要 GTK/OpenGL 运行库。Ubuntu/Debian 的常见依赖为：

```bash
sudo apt-get update
sudo apt-get install -y libglu1-mesa xvfb xauth
```

VESTA 的图像导出命令依赖 GUI/X11。已有桌面会话和 `DISPLAY` 时，CatMaster 直接运行 VESTA；无显示的服务器上会自动调用 `xvfb-run`。如它不在 `PATH` 中，可显式设置：

```bash
export CATMASTER_XVFB_RUN=/usr/bin/xvfb-run
```

`-nogui` 不能替代 Xvfb 完成 VESTA 图像导出。启动 WebUI 前可检查：

```bash
printf 'VESTA=%s\n' "$CATMASTER_VESTA_BIN"
printf 'DISPLAY=%s\n' "${DISPLAY:-<unset>}"
command -v xvfb-run || true
```

默认使用 VESTA 自动取景。对 slab 需要更多面内环境时，让 agent 使用 `supercell=[2,2,1]`；不要为了填满画面而重复真空方向。视觉结论只能用于结构 sanity check、位点和终止面上下文以及报告展示，关键键长、配位数和排序仍应由数值分析确认。

## 4. 许可证与部署

VESTA 官方许可证不允许未经书面许可重新分发其安装文件，因此 CatMaster 主仓库、Deploy 目录和远程部署包都不包含 VESTA 二进制。每台需要渲染的机器都要单独安装并配置路径。

VESTA 生成的图用于论文或其他出版物时，需要明确致谢并引用：K. Momma and F. Izumi, *J. Appl. Crystallogr.* **44**, 1272-1276 (2011)。`render_vesta_views` 会把该要求和引用写入每次渲染的 metadata JSON。

远程 DPDispatcher 计算节点通常不需要 VESTA，因为结构渲染发生在运行 CatMaster agent 的 control-plane 主机。只有明确把渲染任务移到远端时，才需要在对应机器的启动环境中重复配置。
