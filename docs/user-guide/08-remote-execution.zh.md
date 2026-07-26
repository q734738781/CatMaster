# 8. 远程 Task：把准备好的计算真正跑起来

[上一章](07-literature-writing-review.zh.md) | [目录](README.zh.md) | [下一章](09-tools-skills-evolution.zh.md)

CatMaster 能在本地生成结构、检查文件和准备计算输入，但 VASP、CP2K、LAMMPS、ORCA、xTB、CREST、MACE 和其他受管 MLFF 的实际运行通常发生在集群或 GPU 服务器。Remote task 把本地 stage、远程程序、计算资源和结果回传连接成一个受控合同。

用户不需要记住每条提交命令。Experiment 会把科学任务交给合适的 worker，worker 查询当前部署的 task catalog，核对输入目录和参数，再在获得允许后提交。运行状态、日志和结果会回到原 workspace，并留下 receipt 供恢复和审计。

## "会准备"不等于"已经配置为可运行"

许多建模能力在任何 CatMaster 安装中都可以使用。例如 Materials worker 可以生成 slab、吸附候选和 VASP 输入，Dynamics worker 可以准备 LAMMPS stage。真正执行这些 stage 还需要管理员配置 SSH、远程目录、队列、环境脚本、许可证和对应 remote task。

所以同一个 Agent 可能告诉你"VASP 输入已经准备并通过检查，但当前部署没有可用的 `vasp_execute`"。这是准确的能力边界，不是系统应该绕开的错误。CatMaster 不会因为 remote task 缺失就悄悄在 WebUI 所在机器运行科学引擎。

在每次实际提交前，worker 会查询当前 catalog。Task 的启用状态、resource、MLFF backend、模型和允许的参数来自部署配置，旧文档或旧 prompt 都不能替代这次查询。

## 当前模板包含哪些 remote tasks

下面列出仓库模板支持的任务。实际 WebUI 能看到哪些，取决于管理员启用的配置和 worker 权限。

### VASP：单 stage、路径和 dimer

`vasp_execute` 运行一个已经准备好的 VASP 目录，适用于 relax、static、frequency、DOS、MD 等由 `vasp_prepare` 建立的 stage。`vasp_execute_neb` 使用较大的 VASP resource 运行 NEB 或 dimer 风格目录。模板还包含默认关闭的 `vasp_execute_k8s`，用于经过验证的 SSH 到 Kubernetes bridge。

Agent 会在提交前检查 INCAR、POSCAR、POTCAR 和 KPOINTS 是否齐全，元素与 POTCAR 顺序是否一致，以及目录是否符合 task 合同。NEB 还要检查根目录输入和编号连续的图像目录。Remote task 负责把这个已接受的 stage 送到远程执行，并不替代科学输入审查。

```text
使用 Experiment 复查 calculations/co_adsorption/site_03/ 的 VASP relax stage。
让 Materials worker 先核对结构、Selective Dynamics、POTCAR 顺序、INCAR、KPOINTS、
自旋和收敛设置，再查询当前 vasp_execute 的 task spec。若 execution binding 为 configured，
直接接受已注册的部署绑定，不再索要调度器、module、license、revision 或历史 receipt 细节。

只有输入存在问题或 catalog/spec 返回具体错误时才停下；如果全部通过，在 Review 审批卡中展示
task、work_dir、资源和关键设置，等我批准后再提交。完成后检查程序收敛和最终结构，
不要只根据调度状态回答"成功"。
```

### CP2K 与 LAMMPS

`cp2k_execute` 可由 Materials worker 或 Dynamics worker 使用。它运行包含 `job.inp` 和 manifest 所引用文件的 CP2K stage，可用于常规 DFT、频率、路径准备后的计算或 AIMD。Worker 会根据主要目标选择自己的 skills：材料性质侧重输入方法和电子结构，动力学侧重 ensemble、restart 和轨迹连续性。

LAMMPS 属于 Dynamics worker，并提供两条显式执行路径：`lammps_execute` 使用部署绑定的 CPU 资源，适合不完整支持 KOKKOS 的 style；`lammps_execute_kokkos` 使用 GPU/KOKKOS，且加速不可用或启动失败时不会静默回退 CPU。CPU 路径把 `SLURM_NTASKS` 映射为实际 MPI rank，并在运行 LAMMPS 前探测 MPI build、launcher 和进程数；serial/stub build、launcher 缺失或 rank 数不一致都会明确失败。对于没有安装 `srun` 的单节点 Slurm 计算节点，Intel MPI 会把缺省或失效的 Slurm bootstrap 改为本地 Hydra `fork`，同时仍严格验证进程数；`ssh` 等其他显式设置不会被覆盖。两者使用完全相同的 prepared-stage 布局，Agent 应先查询当前部署启用的 task，再根据输入中的 pair、fix、compute 等 style 是否支持 KOKKOS 选择任务。资源和机器由 task 绑定，不通过 submission 参数临时分流。

Task 能启动 LAMMPS，不代表势函数适合当前材料；元素映射、units、边界、neighbor 和势模型适用范围仍然要在提交前确认。

```text
继续 calculations/cp2k_aimd_600K_part1/ 的 AIMD。
让 Dynamics worker 先验证最后有效 restart、坐标、速度、随机状态和时间轴，
在新目录建立 part2，绝不覆盖 part1。查询 cp2k_execute 的当前 task spec，
说明续跑参数与原计算是否一致，并在真正提交前等待我批准。

结果回传后先检查 restart 连续性、温度、能量和轨迹完整性，再决定能否拼接分析。
```

### 通用 MLFF：SP、Relax、MD、NEB 与 TS

`mlff_sp`、`mlff_relax`、`mlff_md`、`mlff_neb`、`mlff_vib` 和 `mlff_ts` 使用统一 task 名称，再由 backend 配置选择 MACE、FairChem UMA、MatterSim 或 ORB-v3。这样同一个科学工作流可以在已启用的模型之间选择，而不需要为每个 provider 复制整套工具。

模板默认启用 MACE `mh-1` 和独立的 `omol-0`：`mh-1` 通过 `mace_mp()` 提供严格枚举的多 heads，`omol-0` 通过 `mace_omol()` 提供显式 charge/spin 条件化分子推理。`mh-1` 的 `omol` head 与独立 `omol-0` 不是同一个模型入口。UMA、MatterSim 和 ORB-v3 只有在管理员安装隔离环境、模型权重、resource 和最小 smoke case 后才会出现。Worker 调用 `get_remote_task_spec` 后会得到当前 backend、官方 model 名、operation、model-specific head 或 task allowlist；UMA 的 `auto` 和非官方模型缩写会被拒绝。用户不应从旧项目复制一组 overrides 后直接提交。

`mlff_sp`、`mlff_relax` 与 `mlff_vib` 可以在一个 stage 的 `input/` 中直接处理多个结构，所以“有多个候选”不自动意味着要用 remote batch。`mlff_md` 要求 `input/` 中只有一个起始或 restart 结构；`mlff_neb` 接受已经在本地建立并检查的固定图像路径；`mlff_ts` 要求 `input/` 中恰好一个 TS-like 结构。

受限 MLFF 优化直接继承结构文件中的原子约束，而不是另传一套 `fixed_atoms` 参数。POSCAR/VASP 使用 Selective Dynamics；extxyz 使用标准 `move_mask`（`L:1` 表示整原子，`L:3` 表示笛卡尔分量，false 表示固定）。`mlff_sp`、`mlff_relax` 和 `mlff_ts` 会把 extxyz 输入分别保存为 `sp.extxyz`、`opt.extxyz` 和 `ts.extxyz`，并在写出后核对约束掩码。extxyz 不能表达 ASE 的缩放坐标 `FixScaled` 约束，这种情况应继续使用 POSCAR/VASP。固定表面原子时通常保持 `relax_cell=false`，否则原子可能随晶胞形变发生仿射移动。

`mlff_ts` 固定晶胞并做一阶受约束 RS-pRFO。`auto` Hessian 策略在 calculator 有公开解析 Hessian 时优先使用；小体系在约束子空间做完整有限差分，大体系使用迭代对角化。终态频率验证与优化收敛相互独立：只有优化收敛且低于所设虚频阈值的模式恰好为一个时，`validated_first_order_saddle` 才为 true。

`mlff_vib` 是通用简正模分析，不带 TS 专属语义，可用于最低点、过渡态、分子、吸附物和受约束材料结构。它与 `mlff_ts` 共用精确约束投影和最终质量加权本征求解，但不会优化几何。完整频谱优先使用 calculator 的解析 Hessian，否则只在自由坐标子空间做有限差分。输出保持紧凑：`vibrations.npz` 保存结构、质量、约束基底、reduced Hessian、频率和质量归一化模式；`frequencies.csv` 是频率表；`modes.extxyz` 是单个多帧可视化文件。不会保留 ASE 的逐位移 JSON cache。对周期超胞而言这是有限超胞 Γ 点分析，不是声子色散。

```text
对 structures/adsorption_candidates/ 中的候选做 MLFF 预筛。
让 Materials worker 查询当前启用的 backend 和 mlff_sp、mlff_relax schema，
结合元素覆盖与任务目的建议模型。先做批量单点并检查异常能量或失败结构，
再只对值得保留的候选做 relaxation。

在提交前展示模型、device、dtype、输入数量、输出目录和排序方法。
最终报告必须把 MLFF 排名写成预筛结果，并列出建议进入 DFT 的候选与风险。
```

### MACE 训练与评估

`mace_train` 和 `mace_eval` 属于 ML worker。训练 task 从 `dataset/` 和 `params/train_params.json` 读取数据与配置，把 checkpoint、日志和其他输出收回 `output/`。评估 task 使用独立的 `params/eval_params.json`，应面向固定测试集或明确 benchmark。

Agent 会在远程训练前完成数据审计和 stage 准备。Dataset 中的单位、标签、划分、E0、head、replay 或 fine-tuning 设置不能靠 remote task 自动纠正。训练完成后，ML worker 应分析 held-out 误差、失败样本和适用范围，而不只报告最后一个 epoch。

```text
检查 ml/mace_finetune_v1/ 是否可以提交 mace_train。
让 ML worker 重新读取 dataset manifest、train/valid/test 划分和 train_params.json，
检查标签、单位、E0、随机种子、foundation checkpoint 与 replay 设置。

查询当前 mace_train resource，估算输入规模并在 Review 卡中展示关键训练参数。
只有我批准后才提交。训练完成后保存 checkpoint、完整日志和配置，
再单独准备 mace_eval，不要用训练误差替代独立测试。
```

### xTB、CREST 与 ORCA

`xtb_prepare` 先把坐标、运行模式、GFN、溶剂、电荷、未配对电子数和优化级别固化为完整 stage；常用约束可由 prepare 参数生成，也可以通过 `xcontrol_path` 原样带入完整 `xtb.inp`。随后 `xtb_execute` 只执行 `manifest.json` 描述的 stage，不再接收科学参数的 `template_overrides`。`crest_run` 用于构象搜索；`orca_execute` 运行包含 `job.inp` 及其本地引用文件的 ORCA stage。这些能力都属于 ORCA/xTB worker。

Worker 会在提交前确认总电荷、未配对电子数或多重度、溶剂、方法、基组和输入结构。CREST 与 xTB 常用于低成本预筛，ORCA 用于选定构象的高层级优化、频率、热化学、TDDFT、NMR 或反应路径。一个 xTB stage 包含 `manifest.json`、清单引用的坐标文件和可选 `xtb.inp`；多个结构必须先准备为一级子目录，再用一个 batch 提交。

```text
对 molecules/conformers_selected/ 中的 6 个构象做 ORCA opt+freq。
让 ORCA/xTB worker 先核对构象去重记录、总电荷、自旋多重度、溶剂、方法和基组，
再为每个构象建立独立 stage。查询 orca_execute 的当前 task spec；binding 为 configured 时，
不要再向用户索要管理员侧 ORCA、MPI、调度器、license 或历史 receipt 信息。

使用受管 batch 提交前，展示 6 个 stage 的路径和共同设置并等待批准。
回传后逐个检查正常终止、梯度和虚频，不能因为 batch 大多数成功就忽略失败构象。
```

## Agent 怎样选择 task、resource 和参数

Worker 先用 `get_avail_remote_task` 查看当前可用 tasks，再用 `get_remote_task_spec` 获取完整 schema。若 task 已列出且 spec 返回 `execution_binding.status=configured`，表示部署侧的 task/backend、resource 与 machine 绑定已通过平台预检，正常提交所需的基础设施依据已经充分。`get_avail_resources` 只列通用 custom-boot 资源，不负责再次审计注册 domain task。

Task 决定执行合同和默认 resource；resource 决定机器、CPU/GPU、队列、walltime、环境脚本和哪些 worker 可以看到它；machine 决定怎样通过 SSH 连接和在哪里放远程工作目录。用户通常只需要关心任务是否可用、资源是否合适、预计成本和科学参数。管理员配置详见[第 10 章](10-deployment-operations.zh.md)。

队列/account 细节、resource card revision、module 或 licensed executable 标识以及历史 smoke receipt 都属于管理员配置，worker 界面会有意隐藏。缺少这些字段不是停止理由；只有 catalog/spec 返回实际绑定错误，或受管提交返回具体运行故障时，Agent 才应把远程配置列为 blocker。

科学或方法参数通过 `template_overrides` 提供，提交层控制通过 `submission_config` 提供。可接受的 key 由当前 task spec 返回。不要把 model、optimizer、温度或计算方法藏进提交配置，也不要把检查间隔和清理策略混进科学参数。

## 单个 stage 和 batch 的区别

`remote_submission` 接受一个完整 stage。`remote_submission_batch` 接受一个父目录，其中每个一级子目录都是独立完整的 stage，并共享同一 task 与提交配置。它不会递归猜测更深目录，也不适合把一个 MLFF stage 内的多结构拆成 batch。

Batch 的价值是用一个受管调用提交一组同构任务，同时保留每个子任务的状态。Agent 仍应在提交前列出实际发现的一级目录并检查数量。部分失败时，只重试失败 stage，不应把已经成功的计算一起重算。

```text
准备批量提交 calculations/vacancy_screen/ 下的 VASP stages。
请列出所有一级子目录，并逐个验证 canonical VASP 输入；任何一个目录缺文件或设置不一致时，
先报告而不是提交部分集合。确认候选数量、共同 task、resource 和设置后，
在 Review 卡中等待批准。不要递归寻找更深目录，也不要自动重算已有成功结果。
```

## 一次远程运行在项目中留下什么

提交前，CatMaster 会把 stage 复制到 workspace 的 metadata staging 区，再由 DPDispatcher 上传。终态后，远程结果会合并回原始 `files/` stage。每个 task 都应回传至少 `status.json`、`stdout.log` 和 `stderr.log`；具体科学程序还会回传自己的输出。

与此同时，系统在 `files/.deepagents/dpdispatcher/receipts/` 保存 receipt。它记录 task、原始 work directory、提交时间、远程 context、submission hash、作业状态、资源和更新信息。Receipt 的作用不是增加内部术语，而是让一次远程作业在 WebUI 断开、网络异常或本地进程退出后仍然可以被识别和恢复。

Chat 会显示 remote receipt 卡，Files 中可以打开结果，Monitor 会记录工具调用和状态。判断计算是否可靠时，应同时查看 receipt、调度状态、stdout/stderr、程序级收敛和科学结果。

## Stop、断线与失败恢复

WebUI 的 Stop 只停止当前 Agent turn，不会自动取消已经进入 Slurm 或远程 Shell 的作业。网络断开也不代表远程作业停止。最危险的处理方式是看不到回复就再次提交同一 stage，因为这可能产生两个计费作业并让结果互相覆盖。

恢复时先保存或找到 receipt 中的 `remote_context_id`、`submission_hash` 和 `receipt_rel`，再到集群调度器和本地 receipt 判断作业是否未创建、排队、运行、终止或已经完成但尚未下载。已有结果应优先下载，失败日志应先收集。只有确认旧作业不会继续运行或写入后，才考虑重投。

```text
上一次 remote_submission 因 SSH 断开返回错误。不要重新提交。
先读取消息中的 receipt 和 .deepagents/dpdispatcher/receipts/ 下对应记录，
确认 remote_context_id、submission_hash、task、原 stage 和已知 job 状态。

结合调度器与 DPDispatcher record 判断作业是否仍在运行、已经完成待下载或真正终止。
优先收集 finished results 和 terminated logs，给出恢复方案；在我确认旧作业状态前禁止重投和清理远程目录。
```

如果 `submission_hash` 存在，管理员可使用 `dpdisp submission` 的下载和诊断命令恢复；具体命令与风险放在[故障排查](11-reference-troubleshooting.zh.md)中。`clean_remote` 只应在结果与日志已经安全回传后使用。

## Review 模式如何保护远程提交

在 Review 模式下，`remote_submission` 和 `remote_submission_batch` 会在执行前中断。审批卡会展示 Agent 准备调用的 task、work directory 和参数。此时应核对：

- 目录是不是你刚审查过的 stage，而不是旧版本或临时副本。
- Task、backend、model 和 operation 是否符合目标。
- CPU、GPU、walltime 和任务数量是否符合预算。
- Overrides 是否来自当前 schema，单位和物理含义是否正确。
- 是否会清理远程目录，是否存在同一 stage 的旧作业。

这些核对项适合审批时使用，但不需要写进每次 prompt。你可以在 prompt 中简单说明"远程提交必须等待 Review 批准"，Agent 会在准备完成后形成具体 action。

## Remote task 的能力来源

<details>
<summary>Worker 可见的远程 tools 与 execution skills</summary>

Materials、Dynamics、ML 和 ORCA/xTB worker 按各自 audience 使用 `get_avail_remote_task`、`get_remote_task_spec`、`get_avail_resources`、`remote_submission` 和 `remote_submission_batch`。

`remote-stage-layouts` skill 说明每种注册 task 的 canonical 输入布局和提交前检查。`dpdispatcher-remote-receipts` 只在远程工具返回失败、传输结果含糊或可能存在孤儿作业时使用，用于 receipt 驱动的恢复；它不应在正常 pending 调用期间触发轮询或重复提交。

</details>

## 第一次接入远程机器时

管理员应先配置 machine、resource、task 和可选 MLFF backend，再让 worker 查询 catalog。不要直接运行全部 smoke suite。先选择一个已经安装的最小 task，提交一个成本可控的真实 case，确认远程环境、回传文件、receipt 和失败恢复都正常，再逐项开放其他引擎。

普通用户不需要编辑这些 YAML。只要在 prompt 中要求 Agent 查询当前能力并在提交前等待批准即可。如果 catalog 中没有目标 task，把缺失信息交给管理员，而不是让 Agent 猜测服务器配置。
