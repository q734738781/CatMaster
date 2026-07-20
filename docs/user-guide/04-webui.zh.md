# 4. WebUI 操作手册

[上一章](03-llm-configuration.zh.md) | [目录](README.zh.md) | [下一章](05-agents-and-modules.zh.md)

WebUI v2 的主线是：登录，选择 workspace，选择 thread，设置 Entry 和权限模式，提交 turn，在 Chat 和 Monitor 中观察，再从同一 thread 继续。左侧栏不是历史 run 选择器。

## 4.1 登录与注册

默认启动启用登录。新用户可以在登录页注册，完成算术验证码后进入自己的用户根目录。Session 默认持续 14 天。不同账号的 workspace 被限制在各自的 `users/<username>/` 下。

内置登录适合本地或受保护网络，不是完整的公网身份平台。默认存在开放注册，cookie 没有设置 Secure 标志，服务本身也不终止 TLS。共享部署见[部署、运维与安全](10-deployment-operations.zh.md)。

`--no-login` 直接进入开放的 `admin` 空间，并禁用 Skill Evolution。它只适合绑定 `127.0.0.1` 的可信单机环境。

## 4.2 页面结构

当前界面由左侧导航、中间工作区和右侧 inspector 组成，并提供四类主要视图：

| 视图 | 用途 |
|---|---|
| Chat | 对话、Progress、工具卡、子 agent 活动、审批卡、artifact 和 remote receipt |
| Monitor | Overview、Live、Events、Raw、Details 运行观测 |
| Skill Evolution | 候选改进、Promote、Reject、Rollback，仅登录模式显示 |
| Files | Browse、Preview、Uploads 文件操作与预览 |

右侧 inspector 可以同时打开多个文件或 artifact 标签，并显示从当前 turn 的 `write_todos` 派生的只读 Todo。

## 4.3 Workspace 操作

左栏顶部选择 workspace。默认登录用户第一次进入时会得到 `default` workspace。

- 新建：使用有意义的短名称，例如 `pt_co_oxidation` 或 `paper_revision_r2`。
- 切换：切换 workspace 会同时切换文件、线程和运行历史边界。
- 删除：先切换到其他 workspace，再输入名称确认。该操作会删除项目数据和内部状态，不能只靠 thread 恢复。

不要用一个 workspace 混放互不相关的多个研究项目。也不要在 Files 中随意操作 `metadata/`。

## 4.4 Thread 操作

左栏的条目是 thread。当前 UI 支持搜索和新建，但没有 thread 重命名、删除、分支或 retry 控件。

建议：

- 一个持续研究问题使用一个 thread。
- 需要不同前提、不同体系或独立可追溯记录时新建 thread。
- 续跑时选择原 workspace 和原 thread，先读最近消息和 artifact，再发明确指令。
- 不要用新 thread 模拟原任务续跑，除非你准备在提示词中重新提供全部上下文。

## 4.5 选择 Entry

Composer 上方可以选择五个入口：

```text
Research
Experiment
Writing
Peer Review
Literature Review
```

运行中不能修改 Entry。选择方法和模块边界见[Agent 与模块功能](05-agents-and-modules.zh.md)。如果目标只是一次明确的结构或计算任务，优先用 Experiment；只有跨文献、计算和写作的开放研究目标才用 Research。

## 4.6 Auto 与 Review

默认权限模式是 `Auto`。`Review` 会在以下受保护工具执行前中断：

```text
write_file
edit_file
remote_submission
remote_submission_batch
```

Review 不是所有工具的总审批开关。读取、搜索、部分分析和其他不在中断表中的工具仍可自动执行。首次接触一个项目、准备覆盖文件或提交计费任务时，建议使用 Review。

运行中不能切换权限模式。中断后，在消息里的审批卡选择：

- `Approve`：按当前 action 执行。
- `Reject`：拒绝这次 action。
- `Respond`：提供补充说明，让 agent 重新处理。
- `Edit action`：编辑 action JSON 后继续，必须保持 action 数量一致，并保留每项的 `name` 和 `args`。

审批恢复必须使用消息内的卡片。中断时 composer 也可能显示 `Respond`，但当前实现仍走普通提交，不应把它当审批恢复按钮。

## 4.7 写一个可执行的请求

高质量请求至少说明目标、输入、约束、输出和停止点：

```text
Entry: Experiment

读取 structures/POSCAR。先检查元素、晶胞、周期性和 Selective Dynamics。
生成 (111) slab 的候选终止面，保留原有约束；用配位数审查悬挂原子。
把候选放到 structures/slabs/，并写 notes/slab_audit.md。
本轮不要提交远程计算。遇到需要我选择的终止面时停下。
```

涉及数值时写单位；涉及随机过程时写 seed；涉及已有文件时写 workspace 相对路径；涉及提交时写明是否允许远程执行、资源限制和预期产物。

## 4.8 附件

空闲时可在 composer 添加附件，运行中附件按钮会禁用。

附件先保存到 workspace 实体路径：

```text
files/attachments/<thread_id>/
```

Agent 使用的是 `attachments/<thread_id>/...` 相对路径。附件同时注册为 artifact，消息历史不会保存原始 base64。

主要限制：

| 层级 | 限制 |
|---|---|
| Composer 浏览器端 | 单文件 64 MiB |
| 后端保存 | 单文件 512 MiB |
| 媒体当前 turn inline | 默认 32 MiB，超出后只保存 |
| 文本附件 | 当前 turn 最多 20,000 字符 |
| PDF、DOCX、XLSX、PPTX 解析 | 文件最多 50 MiB，每次最多 60,000 字符 |
| PDF、PPTX | 最多 20 页或幻灯片 |
| XLSX | 最多 20,000 行、256 列 |

图片按模型视觉能力发送。音频和视频默认不发送，除非模型 profile 明确启用。旧 `.doc`、`.xls`、`.ppt` 和未知格式通常只保存，不解析。到 Monitor 的 Events 中查找 `multimodal.prepared`，可确认 `sent_to_model`、`sent_as` 和 warning。

## 4.9 Send、Steer 与 Stop

空闲时按钮为 `Send`，可以用 `Ctrl+Enter` 提交。

运行期间发送纯文本时，按钮变为 `Steer`。Steer 不会立即打断当前工具或科学任务，而是排队，在当前 run 结束后的安全边界成为下一次 turn。运行时不能附加新文件。

Stop 的前两次请求为优雅停止，在下一次流事件边界生效；第三次请求升级为 emergency cancel。Stop 只停止本地 agent turn，不代表已经提交到 Slurm 或远程 Shell 的作业被取消。远程作业必须依据 receipt 和集群调度器另行核查。

## 4.10 Chat 中的活动和结果

Chat 会呈现：

- 增量文本和 Progress。
- 工具名称、参数、返回摘要和错误。
- specialist 或 worker 的委派活动。
- 写出的 artifact。
- 远程 receipt 和状态摘要。
- Review 审批卡。

连续活动较多时会折叠到 `Activity`。不要只看最后一句回复。展开失败工具、检查 artifact 路径，并在涉及计算时核对 `status.json`、`stdout.log` 和 `stderr.log`。

## 4.11 Monitor

Monitor 运行中约每 5 秒刷新一次，并合并当前线程的 SSE 事件。

- `Overview`：状态、持续时间、LLM 调用、token、费用、机器时间、工具成功和失败数。
- `Live`：当前阶段、活动工具、Todo、子 agent、近期模型文本和任务日志。
- `Events`：按 thread、run、agent、tool、category 和 channel 过滤事件。
- `Raw`：原始聊天和日志数据。
- `Details`：task state 和 memory。

当前 Monitor 没有 run 选择器。总览指标查询以 workspace 和 lane 为主，可能显示该范围内当前或最近 run，不一定严格等于左栏所选 thread。需要精确追踪时，以事件中的 thread ID、run ID、artifact 和 receipt 相互核对。

## 4.12 Files

Files 提供 Browse、Preview 和 Uploads。支持文本、Markdown、JSON、图片、PDF、CSV/TSV、JSmol 结构、轨迹和 OUTCAR 振动等预览。常见结构格式包括 CIF、PDB、XYZ、VASP、POSCAR、CONTCAR、OUTCAR、XDATCAR 和 TRAJ。

限制和风险：

- 文本预览最多约 160 KiB，目录预览最多 40 项，文件树一次最多 500 项。
- 轨迹预览最多 240 帧。
- Files 上传的后端单文件上限是 512 MiB。
- 上传固定使用覆盖模式，同名文件会直接覆盖，没有第二次确认。
- 删除是永久递归删除，只有浏览器确认。
- 文件树同时显示 `files/` 和 `metadata/`，不要删除或改写 `metadata/`。
- 目录下载 ZIP 最多包含 20,000 个文件，总量最多 2 GiB。
- 后端支持安全 ZIP 解压，但当前 Files UI 没有解压开关。

重要数据上传前先改成唯一文件名。批量移动、覆盖或删除前，在 workspace 外保留备份。

## 4.13 继续、修正和复查

同一 thread 的常见继续指令：

```text
继续上一次任务。先重新读取 notes/slab_audit.md 和现有候选结构，
列出已完成、未完成和需要我决定的事项；不要重复生成已存在的候选。
```

发生错误后，不要只写“重试”。说明应保留的文件、失败证据、允许修改的范围，以及是否禁止重新计算。远程失败先做 receipt 驱动的状态审计，详见[远程机器与任务执行](08-remote-execution.zh.md)。

## 4.14 Skill Evolution

登录模式下，终态 run 可触发后台候选生成和审查。默认 `observe` 模式只显示候选，不自动生效。候选在同一 workspace 的所有 thread 间共享，Promote 后从下一次 run 起加载；不合适的候选可 Reject，已提升内容在目标未变化时可 Rollback。

具体安全边界见[工具、技能与自进化](09-tools-skills-evolution.zh.md)。
