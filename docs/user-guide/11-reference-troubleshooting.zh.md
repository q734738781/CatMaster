# 11. Prompt 参考与故障排查

[上一章](10-deployment-operations.zh.md) | [目录](README.zh.md)

本章前半部分收录可以直接修改的 prompts，后半部分按症状排查安装、模型、文件、文献和远程任务。参考 prompt 不是固定表单。保留其中对科学边界和交付物有用的部分，删掉与你的项目无关的限制，让 Agent 有空间选择 skills 与 tools。

## 怎样改写参考 prompt

Prompt 最重要的是让 Agent 理解你真正想解决的问题。输入路径、必须保持的约束、允许的计算范围和希望保留的结果能够减少误解；tool 名通常可以省略。若方法尚未确定，可以让 Agent 比较方案并在关键歧义处停下。若方法已经由项目标准确定，则直接写明，不必要求它重新论证。

"自主选择"也不是让 Agent 无限延伸。你可以授权它完成目标所需的技术判断，同时明确本轮停止在结构候选、输入审查、远程审批或结果分析的哪一阶段。

## Research 参考 prompt

```text
使用 Research 研究 <研究问题>。先读取 <已有目录或文件>，区分已有事实、
工作假设和真正缺少的证据。自主决定何时需要 Literature Review、Experiment、
Writing 或 Peer Review，但一次只推进一个有边界的阶段，并在每次委派后检查结果。

本轮要交付 <证据地图/研究计划/阶段综合>。没有文献或项目证据时不要擅自运行计算；
如果建议新增实验或计算，请说明它能区分什么假设、需要什么输入和成本，等我确认。
```

适合把 `<研究问题>` 改成开放目标，例如"解释某催化剂在氧化还原循环中的可逆结构变化"。如果你已经知道只需建一个 slab，则直接使用 Experiment。

## Slab 与吸附参考 prompt

```text
使用 Experiment 从 <体相结构路径> 建立 <Miller 指数> 表面，目标是 <后续研究用途>。
让 Materials worker 自主使用适合的 slab、termination 和 visual inspection skills。

表面至少满足 <厚度或层数> 和 <真空>，说明是否采用对称 slab、面内扩展与固定层策略。
保留现有 Selective Dynamics；若需要改动约束，先给出理由和候选方案。
检查上下表面、化学计量、配位、异常短键和孤立原子，保存全部合理终止面、结构图和审计。
本轮不要提交计算；遇到极性或终止面取舍时停下来问我。
```

继续到吸附时补充：

```text
在已确认的 slab 上为 <吸附物> 建立吸附候选。先定义吸附物构象和锚点，
枚举并去重有化学意义的位点与朝向，记录位点来源、初始距离、覆盖度和约束继承。
检查碰撞与跨周期距离，不要为了凑数量生成明显重复结构。

候选过多时可以建议 MLFF 单点或优化预筛，但要先查询当前 backend 能力并等待我批准。
最终给出候选清单、结构图、筛除理由和建议进入 DFT 的集合。
```

## VASP 或其他远程计算参考 prompt

```text
使用 Experiment 复查 <stage 路径>，目标是运行 <计算类型>。
让负责的 worker 先检查结构、约束、输入文件、科学参数和预期输出，
再查询当前部署的 remote task、resource 和完整 schema。不要从旧 prompt 猜 overrides。

如果准备或配置有问题，停下并写清缺什么。全部通过后，在 Review 审批卡展示 task、
work_dir、任务数量、资源、关键参数和清理策略，等我批准后再提交。
回传后检查 status、stdout/stderr、程序级收敛和科学结果，不要只看 scheduler completed。
```

Batch 时再说明一级 stage 目录与共同设置。MLFF 任务还应要求记录 backend、model、device 和 dtype，并把结果标为模型预测。

## 动力学与 restart 参考 prompt

```text
使用 Experiment 继续 <已有 MD 目录>。让 Dynamics worker 先审计最后有效步、
结构、速度、积分器或 thermostat 状态、随机状态、时间轴和可用 restart 文件。
禁止覆盖原目录，也不要在证据不足时从最后一帧重新赋速后称为连续续跑。

在新目录建立 continuation stage，说明新旧段怎样连接以及哪些设置必须保持一致。
查询当前 remote task，等我批准后再提交。结果回传后先做温度、能量、体积、
轨迹连续性、异常短距和 restart 可用性检查，再决定是否进行 MSD/RDF/扩散分析。
```

## 数据集与 MACE 参考 prompt

```text
使用 Experiment 让 ML worker 从 <VASP 结果目录> 建立 MACE 数据集。
先区分收敛、未收敛和标签不完整的 runs，检查单位、参考能、元素覆盖、重复结构、
异常值和不同计算设置混用。固定随机种子并保留 train/valid/test manifest。

输出 extxyz、划分文件和数据审计报告。只有数据审计通过后才准备训练参数；
在我确认数据范围、foundation model、replay/E0 设置和 GPU 成本前不要提交 mace_train。
```

## 分子、xTB 与 ORCA 参考 prompt

```text
使用 Experiment 处理 <SMILES 或结构路径>。总电荷为 <charge>，自旋多重度为 <multiplicity>，
溶剂和目标性质为 <设置>。让 ORCA/xTB worker 自主选择构象生成、CREST/xTB 预筛、
去重和 ORCA 方法，但保留每一步的结构、相对能和筛除理由。

先交付可审阅的构象集合与 ORCA stage 计划。说明频率、热化学、TS/IRC、TDDFT 或 NMR
中哪些步骤与当前目标有关；不要默认把所有计算都跑一遍。任何远程 task 等我批准。
```

## Literature Review 参考 prompt

```text
使用 Literature Review 调研 <主题>，范围为 <年份、体系、文献类型和排除条件>。
先设计并保存检索策略，建立足够宽的候选集合并对 DOI、题名和版本去重。

把发现记录、摘要、全文和 SI 证据分开。围绕 <具体问题> 精读核心论文，
提取体系、条件、方法、结果、限制和相互矛盾之处，建立 claim-evidence 表。
保存候选表、全文可用性、未获取清单和最终引用库，不要用标题或摘要补写精确参数。
```

精读单篇论文时，要求保留章节顺序、图表位置、页码或原文锚点，并明确"不接受只给摘要"。

## Writing 与审稿参考 prompt

```text
使用 Writing 根据 <证据文件、数据、图和引用库> 起草或修改 <章节/文稿>，
目标读者或期刊为 <目标>。先检查材料能支持怎样的论证，再自主选择 writing skills。

所有数字、单位、图和引用必须可追溯；保持 <必须保留的术语或结论边界>，
禁止添加 <新结果、未核实引用或因果表述>。正文写成连贯段落，输出到 <路径>，
并列出证据不足和需要作者决定的地方。
```

```text
使用 Peer Review 审查 <canonical PDF>，目标期刊和文章类型为 <说明>。
让 reviewer 独立检查新颖性、方法、证据、图表、报告完整性和可重复性，
主要意见必须指向页码或图表。保留每位 reviewer 的完整报告，再生成 editor synthesis，
明确共识、分歧、必须解决的问题和可选改进。本轮不修改源稿，也不代写作者回复。
```

## 继续旧任务或恢复失败参考 prompt

```text
继续原 thread。先重新读取 <关键报告、目录、receipts 和日志>，以当前文件为准，
不要直接沿用聊天中的完成状态。列出可信完成项、不完整项、仍在运行的远程任务和待决定事项。

保留所有成功结果，禁止重复生成或重复计算。恢复事实后再从 <明确阶段> 继续，
本轮停止在 <新的停止点>。
```

远程调用断开时：

```text
上次远程调用返回异常。不要重新提交或清理。
先读取 receipt，确认 remote_context_id、submission_hash、task、原 stage 和 job 状态，
再结合调度器或 DPDispatcher record 判断作业是否仍在运行、已完成待下载或真正终止。
优先收集结果和失败日志，给出恢复方案；只有确认旧作业不会继续写入后才讨论重投。
```

---

## WebUI 打不开

先以前台模式启动，阅读第一条真实 traceback：

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
./start_webui.sh --foreground --host 127.0.0.1 --port 7991
```

另一个终端检查：

```bash
./start_webui.sh --status
tail -n 100 .runtime/webui.log
ss -ltnp | grep 7991
```

`conda is not available` 表示启动 shell 没有初始化 conda，或 `CATMASTER_CONDA_ENV` 指向错误环境。`Address already in use` 表示端口被占用。Project root permission denied 应修复目录 ownership，不要临时用 root 绕过。JSmol 下载失败只影响 OUTCAR vibration 和 fallback 预览，不影响 MatterViz 主工作台。

启动脚本和 Python CLI 的隐式默认地址不同，所以诊断时总是显式写 host 与 port。

## LLM 配置能解析，但对话失败

先做离线解析：

```bash
python -c 'from catmaster.llm.config import LLMProfile; p=LLMProfile.from_env_or_file(); print(sorted(p.models)); print(p.agents)'
```

如果解析失败，检查 YAML 缩进、角色引用和 provider 字段。如果解析成功但调用失败，按顺序检查：

1. Key 是否 export 到启动 WebUI 的同一进程环境。
2. Model ID 和 base URL 是否属于当前 provider。
3. Reasoning 与 provider options 是否使用了正确字段。
4. 模型是否支持工具调用和当前 schema。
5. Provider 返回的第一条 4xx/5xx 或 timeout 是什么。

只得到文字、不调用 tool 时，不要先增加 prompt 强度。查看 Chat tool 卡和 Monitor，确认当前 Entry 是否正确、模型是否支持 tool calling、worker 是否被委派，以及 tool schema 是否真正发送。

长任务过早停止时，先看实际 tool error、`max_tool_calls`、recursion 与上下文，不要直接把所有边界调到极大。

## 附件保存了，但 Agent 没有读到

图片需要模型 profile 支持视觉输入。PDF、DOCX、XLSX 和 PPTX 走有界文档解析；旧 `.doc`、`.xls`、`.ppt` 和未知格式通常只保存。检查 Monitor 中的 `multimodal.prepared`，看 `sent_to_model`、`sent_as` 和 warning。

常见限制：Composer 单文件 64 MiB；后端保存上限 512 MiB；媒体当前 turn inline 默认 32 MiB；PDF 与 Office 解析单文件最多 50 MiB、最多 60,000 字符；PDF/PPTX 默认最多处理 20 页或 slides。大型文档应要求 Agent 按页或章节读取，或先拆分。

## Files 中结构、PDF 或表格不能预览

主结构画布空白时，先看浏览器 console 与 network，确认 `chunk-MatterVizHost.js` 及其本地资源返回 200，再阅读 renderer boundary 给出的错误。用 **Source** 判断文件本身是否损坏。大结构会主动显示有界画布提示，但 Properties 中的完整原子数和分页坐标表仍以源结构为准。

只有分子二维编辑失败时检查按需加载的 `chunk-KetcherEditor.js`，三维构象和 source 仍应可用。Volume 失败时查看 worker request，先 Cancel 再换 grid。只有 OUTCAR vibration 或明确进入 JSmol fallback 时才检查固定版本的 JSmol cache 和格式。文本、目录和文件树预览都有明确大小或数量边界，因此"看不到预览"不代表文件不存在。PDF 字体或页面异常应打开原文件核对。

Files 上传同名文件会覆盖。若内容被覆盖，从外部备份恢复。误删 `metadata/` 后应立即停止写入并恢复一致性备份，重新上传 `files/` 不能恢复 thread checkpoint。

## Literature Review 找到题目却没有全文

这通常不是搜索失败，也不一定需要继续找全文。摘要或有信息量的检索结果足以支持当前表述时，直接使用并说明边界。只有关键判断依赖摘要中没有的细节时，才确认开放获取或合法机构权限，并做一次合理的浏览器访问；失败后继续使用其他来源，不反复尝试。Agent 不会绕过 CAPTCHA 或付费墙。

```bash
agent-browser doctor --offline --quick
agent-browser mcp --help
```

只有摘要时，把证据级别记录为 abstract。引用元数据冲突时，以 DOI、publisher 页面和论文自身为主要依据，并保存版本差异。本地 corpus 漏文时检查 ingest manifest、parse status 和文件是否超出解析限制。

## Remote task 在 catalog 中不存在

按以下顺序检查管理员配置：

1. `machines.yaml`、`resources.yaml`、`tasks.yaml` 和 `mlff_backends.yaml` 是否存在。
2. 活动文件名是否错误地包含 `template`。
3. YAML 是否能解析，是否有多个活动文件覆盖同一 key。
4. Task 与 backend 是否 enabled。
5. Resource 的 audience 是否包含当前 worker。
6. Machine SSH、remote root、queue 和 `source_list` 是否有效。

缺少 task 时不要让 Agent 改用本地 scientific engine。先由管理员完成配置和最小 smoke case。

## Remote task 连接或启动失败

使用与 CatMaster 相同的非交互 SSH 环境验证主机、Python、目录和 scheduler：

```bash
ssh -o BatchMode=yes -i <SSH_KEY> <USER>@<HOST> 'hostname; python3 --version'
ssh -o BatchMode=yes -i <SSH_KEY> <USER>@<HOST> 'test -w <REMOTE_ROOT>'
ssh -o BatchMode=yes -i <SSH_KEY> <USER>@<HOST> 'command -v sbatch; command -v squeue; command -v scancel'
```

`command not found` 或退出码 127 通常来自 machine `env_setup`、resource `source_list` 或科学程序路径。不要通过修改结构 stage 掩盖远程环境问题。

## 远程调用断开、作业状态不明

不要重投。先找到 receipt 和 `submission_hash`。在确认 DPDispatcher record 后，可按实际需要选择命令：

```bash
dpdisp submission <submission_hash> --download-finished-task
dpdisp submission <submission_hash> --download-terminated-log
dpdisp submission <submission_hash> --reset-fail-count
dpdisp submission <submission_hash> --clean
```

这些命令不是固定顺序。先下载已完成结果和失败日志；只有明确理解 fail count 与清理后果时才使用后两条。`submission_hash` 为空通常表示没有可恢复 record，需要回到 receipt、调度器和远程目录判断。

Scheduler completed 只说明调度结束。若结果缺失，检查 backward files、远程权限和程序日志。`status.json` 成功但科学不收敛时，应判为科学失败。Batch 部分失败时逐个 stage 分类，只处理失败项。

## Stop 后远程作业仍在运行

这是预期行为。WebUI Stop 取消本地 Agent turn，不会调用 `scancel` 或终止远程 Shell。使用 receipt 中的 job 信息到对应调度器处理，并保留取消证据。不要删除本地 stage 或 receipt 后再尝试找作业。

## 登录、workspace 或 thread 看起来丢失

先确认 `CATMASTER_PROJECT_SPACE_ROOT` 是否与原部署一致，登录用户名是否相同。登录模式的数据位于 `users/<username>/`；无登录模式使用 `admin/`。旧 `.catmaster` 单根项目不能直接当作当前 workspace，需要迁移为 `files/` 与 `metadata/`。

只恢复 `files/` 不会恢复 thread。检查 `metadata/`、DeepAgent SQLite 和认证数据库是否来自同一份一致性备份。

## Skill Evolution 没有产生或发布 candidate

这通常是正确边界，不一定是 queue 故障。每个有用户任务的 terminal run 会先进入
语义反思；`no_change` 或 execution lapse 不会写 observation。系统根据完整轨迹和
结果判断，不要求固定数量的 run、thread 或 counterexample。用户明确要求长期遵守
的 correction 可以单独形成证据，但 candidate 仍要经过静态检查、独立建议与人工
审核。先在 Skill Evolution 查看 job、observation 和 candidate，不要要求开发者用
关键词强制路由 raw job。Tool/schema 问题与详细科学 notes 本来就不会进入 skill。

`pending` 和 `revision` 都不能 Promote。打开精确 revision，检查 evidence、
counterexample、适用边界、静态 validation 与 reviewer concerns，再选择 Request
revision 或 Reject。状态为
`review` 的 skill 还要先在明确 thread/run 上 Start canary；只有该精确 revision
有一次成功的真实使用，而且没有 failure 或 false activation，才会出现 Promote
stable。Start canary 不会创建新对话或模型调用，它只为所选 scope 绑定精确版本。

Canary 消失时先看 candidate card：精确 revision 失败或错误激活后，只会自动移除
对应 canary pointer，stable 不受影响。Builtin 或目标 hash 改变时会回到
`revision`，不会静默遮蔽新版 skill。需要开发者排查时复制卡片上的 diagnostics
reference，不要粘贴整页 raw event JSON。

## 当前 UI 限制速查

- 没有历史 run 选择器，也没有 thread 重命名、删除、branch 或 retry UI。
- Interrupted 状态必须用消息内审批卡恢复。
- Monitor 总览可能对应 workspace 与 lane 的当前或最近 run。
- Files 上传同名覆盖，删除永久递归；后端支持的 ZIP 解压尚未在 Files UI 提供开关。
- Stop 不取消远程 job。
- Skill Evolution 只在登录模式显示，并从下一次 run 生效。

这些限制应通过版本化文件名、外部备份、明确 thread 划分和 receipt 驱动的远程管理来规避，不要假设 Agent 会自动提供 UI 尚未实现的恢复操作。

## 部署验收

一个可交给用户的 CatMaster 部署至少应通过以下真实检查：

- 新账号可以登录并只能看到自己的 workspace。
- 五类 Entry 可选择，基础模型能对话并调用工具。
- 附件、artifact、Files 预览、Review 审批和 Monitor 可用。
- 一个本地结构任务能由 Experiment 委派 worker 并写出结构。
- Literature Review 的实际搜索与浏览器能力和文档宣称一致。
- 每个启用的 remote task 至少有一个最小真实 case，能回传 status、stdout/stderr 和 receipt。
- 模拟一次传输或本地中断后，可以依据 receipt 恢复，而不重复计算。
- 项目、认证数据库、私有配置和 secrets 有备份与恢复方案。

验收的目标不是让每个可选工具都安装，而是让界面展示的能力与实际部署一致，并让失败时有证据可查。
