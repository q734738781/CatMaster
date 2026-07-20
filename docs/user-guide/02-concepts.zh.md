# 2. 系统概念与项目空间

[上一章](01-quickstart.zh.md) | [目录](README.zh.md) | [下一章](03-llm-configuration.zh.md)

CatMaster 不是一个把所有程序安装在同一台电脑上的聊天窗口。它由 control plane、项目空间、specialist agent、工具与 skill、远程执行后端共同组成。先理解这些边界，能避免路径、续跑和计算资源方面的大多数误操作。

## 2.1 系统边界

```text
浏览器
  -> CatMaster WebUI 和 specialist runtime
      -> 项目空间 files 与 metadata
      -> LLM provider、网页与本地辅助工具
      -> DPDispatcher
          -> SSH/Slurm/Shell 机器
              -> VASP、CP2K、LAMMPS、ORCA、xTB、CREST、MLFF
```

Control plane 负责对话、规划、工具调用、文件编排、运行记录和远程提交。科学引擎在资源卡指定的执行环境中运行。二者可以在同一台主机，也可以完全分开。

## 2.2 账号、项目根和 workspace

`CATMASTER_PROJECT_SPACE_ROOT` 指向 WebUI 管理的总目录，不是某一个项目本身。

默认登录模式：

```text
<PROJECT_SPACE_ROOT>/
  .webui_auth/
    auth.sqlite
  users/
    <username>/
      default/
        files/
        metadata/
      <another-workspace>/
        files/
        metadata/
```

无登录模式：

```text
<PROJECT_SPACE_ROOT>/
  admin/
    files/
    metadata/
```

Workspace 是长期项目边界。不同催化体系、论文或数据集通常应使用不同 workspace。左侧栏允许切换、新建和删除 workspace。删除是实质性操作，先切换到其他 workspace，再按界面要求输入名称确认。

## 2.3 `files/` 和 `metadata/`

每个 workspace 必须同时有两个目录：

```text
workspace/
  files/
  metadata/
```

`files/` 包含用户输入、agent 生成物、结构、脚本、计算 stage、报告和远程 receipt。它是用户与 agent 共同工作的区域。

`metadata/` 包含线程记录、checkpoint、运行观测、artifact 索引、远程临时 staging 和自进化状态。用户通常只备份和诊断它，不直接编辑、重命名或删除其中的文件。

旧式只有 `.catmaster` 的单根目录不会被自动迁移，并会被当前运行时拒绝。迁移旧项目时，应建立 `files/` 与 `metadata/`，再有选择地复制用户数据，不能把旧内部状态整包塞进新目录。

## 2.4 Agent 看到的路径

对 agent 而言，虚拟根目录映射到 workspace 的 `files/`。在提示词中推荐写：

```text
读取 structures/slab.vasp
把报告写到 writing/surface_report.md
分析 calculations/co_adsorption/opt/OUTCAR
```

不要要求 agent 访问宿主机上的任意绝对路径，例如 `/home/user/private/...`。界面上传文件后，先确认它在 `files/` 下的相对路径，再把该路径告诉 agent。

一个适合多数研究项目的目录：

```text
files/
  literature/
  structures/
  calculations/
  scripts/
  notes/
  writing/
  attachments/
  .deepagents/
```

不必预先创建全部目录。已有清晰布局时让 agent 延续现状。可复用脚本放进 `scripts/`，并记录日期、用途、输入输出和关键假设。

## 2.5 Thread、turn 和 run

这些概念不能互换：

| 概念 | 含义 | 是否持久 |
|---|---|---|
| Workspace | 一个项目的数据和历史边界 | 是 |
| Thread | 连续对话与 checkpoint | 是 |
| Turn | 一次用户提交及其 agent 响应 | 是 |
| Run | 一次 turn、steering 或审批恢复对应的执行与观测记录 | 是 |
| Artifact | 注册到界面的文件或结果对象 | 是 |
| Receipt | 远程提交的可恢复身份与状态记录 | 是 |

左侧栏选择的是 thread，不是 run。同一 thread 内直接继续对话，checkpoint 会保持连续。当前 v2 没有历史 run 选择器、thread 分支、retry 或 `resume_selected_run` 控件。

线程状态决定下一步：

- `idle`、`stopped`、`error`：直接发送明确的继续指令。
- `running`：发送文本会成为 `Steer`，排队到当前运行后的安全边界。
- `interrupted`：使用消息内审批卡恢复，不要在 composer 里另发一条普通回复。

## 2.6 Artifact、日志和证据

Agent 写出的文件、附件和远程结果可以注册成 artifact，并在右侧 inspector 中打开。Chat 中的工具卡展示调用参数和返回摘要；Monitor 保存运行事件、模型文本、工具结果、token、费用和机器时间。

长工具输出可能只在对话中显示预览，完整内容会按 `configs/tool_output.yaml` 落到 `files/_tool_outputs/`。因此，最终结论应指向文件、日志、receipt 或结构，而不只依赖一段聊天摘要。

## 2.7 备份与恢复

要完整恢复一个 workspace，必须一起备份 `files/` 和 `metadata/`。只备份 `files/` 会丢失 thread、checkpoint、审批状态和观测历史。

登录部署还应单独备份：

```text
<PROJECT_SPACE_ROOT>/.webui_auth/auth.sqlite
```

建议在 WebUI 停止或确认没有写入中的 run 后做一致性备份。`files/.deepagents/` 内可能有 staged skills 和 DPDispatcher receipts，不要把整个隐藏目录当缓存清理。

## 2.8 执行权限和科学责任

CatMaster 按 entrypoint 把任务交给合适的 specialist 和 worker。协调层不会拥有所有科学工具，worker 也只能调用自己 allowlist 中的工具。受管远程任务还受 task、resource、machine 和 audience 约束。

这些约束能减少误用，但不证明计算设置正确。用户仍需确认：

- 体系、电荷、自旋、周期性和约束。
- 赝势、泛函、基组、色散和收敛标准。
- 温度、系综、时间步、采样长度和随机种子。
- 单位、能量基准、原子映射和结果可比性。
- 软件许可、集群策略和计算成本。
