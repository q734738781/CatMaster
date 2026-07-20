# 8. 远程机器与任务执行

[上一章](07-literature-writing-review.zh.md) | [目录](README.zh.md) | [下一章](09-tools-skills-evolution.zh.md)

CatMaster 用 DPDispatcher 把已经准备好的计算 stage 提交到 SSH 可达的 Slurm 或 Shell 机器。远程配置分为四层：machine 定义怎么连接，resource 定义在哪台机器用多少资源，task 定义执行什么合同，MLFF backend 定义模型和 operation 映射。

## 8.1 角色分工

| 角色 | 负责内容 |
|---|---|
| 管理员 | SSH、远程根目录、队列、资源卡、环境脚本、程序和许可证 |
| CatMaster worker | 按 task schema 准备 stage、做 QC、提交、收集和记录 receipt |
| 用户 | 选择科学设置、批准提交、承担费用、检查结果和失败恢复决定 |

Agent 不能把注册 task 任意切换到另一台 machine 或 resource。CPU/GPU 覆盖也只应在用户明确要求且站点允许时使用。

## 8.2 准备四个活动配置

模板文件名包含 `template`，不会被注册表加载。复制四个文件：

```bash
cp configs/dpdispatcher/machines_template.yaml \
   configs/dpdispatcher/machines.yaml
cp configs/dpdispatcher/resources_template.yaml \
   configs/dpdispatcher/resources.yaml
cp configs/dpdispatcher/tasks_template.yaml \
   configs/dpdispatcher/tasks.yaml
cp configs/dpdispatcher/mlff_backends_template.yaml \
   configs/dpdispatcher/mlff_backends.yaml
```

活动文件包含主机名、用户名、SSH key、路径和站点环境，已被 Git 和部署包排除。不要把真实配置复制到文档、issue、prompt 或共享 workspace。

目录中的非模板 YAML、YML 和 JSON 都可能被加载。多个活动文件出现重复 key 时，后读值可能覆盖前值，因此应保持一个清晰的 source of truth。

## 8.3 Machine card

Slurm CPU 机器的基本形状：

```yaml
cpu_server:
  batch_type: Slurm
  context_type: SSHContext
  local_root: <LOCAL_WORK_ROOT>
  remote_root: <REMOTE_WORK_ROOT>
  retry_count: 0
  remote_profile:
    hostname: <CPU_LOGIN_HOST>
    port: 22
    username: <USERNAME>
    key_filename: <PATH_TO_SSH_KEY>
  env_setup: |
    ulimit -s unlimited
    module load <SITE_MODULES>
```

模板还包含：

| Machine | Batch | 典型用途 |
|---|---|---|
| `cpu_server` | Slurm over SSH | VASP、CP2K、LAMMPS、xTB、CREST、ORCA、通用 CPU |
| `k8s_ssh_server` | Shell over SSH | 阻塞式 SSH 到 Kubernetes bridge，默认 VASP task 禁用 |
| `gpu_server` | Shell over SSH | MACE、UMA、MatterSim、ORB 和通用 GPU |

`local_root` 在实际提交时由 CatMaster 的 metadata staging root 接管。仍应保留合法占位值；真正需要重点核对的是 `remote_root`、SSH profile 和资源环境。

## 8.4 SSH 和目录验收

使用非交互 key，限制权限：

```bash
chmod 600 <PATH_TO_SSH_KEY>
ssh -i <PATH_TO_SSH_KEY> -p 22 <USERNAME>@<CPU_LOGIN_HOST>
```

在 control plane 上做非交互测试：

```bash
ssh -o BatchMode=yes -i <PATH_TO_SSH_KEY> \
  <USERNAME>@<CPU_LOGIN_HOST> 'hostname; python3 --version'
```

确认远程根目录存在且可写：

```bash
ssh -o BatchMode=yes -i <PATH_TO_SSH_KEY> \
  <USERNAME>@<CPU_LOGIN_HOST> \
  'mkdir -p <REMOTE_WORK_ROOT> && test -w <REMOTE_WORK_ROOT>'
```

Slurm 机器还应验证：

```bash
ssh -o BatchMode=yes -i <PATH_TO_SSH_KEY> \
  <USERNAME>@<CPU_LOGIN_HOST> \
  'command -v sbatch; command -v squeue; command -v scancel'
```

首次连接先由管理员交互确认 host key。不要在自动化中用 `StrictHostKeyChecking=no` 掩盖主机身份变化。

## 8.5 环境加载顺序

远程命令环境按以下顺序构造：

```text
machine.env_setup
  -> resource.source_list
      -> submission prepend_script
          -> task command
```

`source_list` 中的脚本必须在远程机器存在。路径错误会在任务启动前以 127 失败。把程序 module、conda activate、许可证变量和库路径放在站点受控脚本里，不要把 secret 写进 task stage。

## 8.6 Resource card

Resource 把能力绑定到 machine、audience、队列和核数。模板默认值只是示例：

| Resource | Machine | CPU | GPU | Queue | Audience/用途 |
|---|---|---:|---:|---|---|
| `vasp_cpu` | `cpu_server` | 52 | 0 | `batch` | Materials, VASP stage |
| `vasp_k8s_cpu` | `k8s_ssh_server` | 4 | 0 | `k8s` | Materials, K8s VASP |
| `vasp_cpu_neb` | `cpu_server` | 104 | 0 | `batch` | Materials, VASP path |
| `cp2k_cpu` | `cpu_server` | 32 | 0 | `batch` | Materials/Dynamics |
| `lammps_cpu` | `cpu_server` | 16 | 0 | `batch` | Dynamics |
| `general_cpu` | `cpu_server` | 4 | 0 | `batch` | 允许的 custom CPU boot |
| `general_gpu` | `gpu_server` | 16 | 1 | `main` | 允许的 custom GPU boot |
| `mace_gpu` | `gpu_server` | 16 | 1 | `main` | MACE |
| `uma_gpu` | `gpu_server` | 16 | 1 | `main` | FairChem UMA |
| `mattersim_gpu` | `gpu_server` | 16 | 1 | `main` | MatterSim |
| `orb_gpu` | `gpu_server` | 16 | 1 | `main` | ORB-v3 |
| `xtb_cpu` | `cpu_server` | 32 | 0 | `batch` | xTB |
| `crest_cpu` | `cpu_server` | 32 | 0 | `batch` | CREST |
| `orca_cpu` | `cpu_server` | 32 | 0 | `batch` | ORCA |

必须按站点修改 queue、核数、GPU、walltime 和 `source_list`。Resource 的 `audiences` 决定哪些 worker 可以看到它；不要为了方便删除所有 audience 限制。

## 8.7 Task card

当前模板的注册任务：

| Task | 默认 resource | 主要输入 |
|---|---|---|
| `vasp_execute` | `vasp_cpu` | 单个 VASP stage |
| `vasp_execute_k8s` | `vasp_k8s_cpu` | 同 VASP stage，默认 disabled |
| `vasp_execute_neb` | `vasp_cpu_neb` | NEB/dimer 目录 |
| `cp2k_execute` | `cp2k_cpu` | CP2K stage |
| `lammps_execute` | `lammps_cpu` | LAMMPS stage |
| `mlff_sp` | backend 决定 | 多结构 single point |
| `mlff_relax` | backend 决定 | 多结构 relaxation |
| `mlff_md` | backend 决定 | 单结构 trajectory |
| `mlff_neb` | backend 决定 | 固定图像路径 |
| `mace_train` | `mace_gpu` | 数据集和训练参数 |
| `mace_eval` | `mace_gpu` | 数据集和评估参数 |
| `xtb_run` | `xtb_cpu` | 分子输入和模式参数 |
| `crest_run` | `crest_cpu` | 分子与构象搜索参数 |
| `orca_execute` | `orca_cpu` | `job.inp` stage |

未写 `enabled` 的 task 默认可用；模板中只有 `vasp_execute_k8s` 明确为 `false`。启用前必须先验证 bridge、共享目录和阻塞语义。

## 8.8 MLFF backend 环境

模板默认：

| Backend | Enabled | Default model | Operations |
|---|---|---|---|
| `mace` | true，且为默认 | `mh-1` | SP、relax、MD、NEB |
| `fairchem_uma` | false | `uma-s-1p2` | SP、relax、MD、NEB |
| `mattersim` | false | `mattersim-v1-1m` | SP、relax、MD、NEB |
| `orb_v3` | false | `orb-v3-conservative-inf-omat` | SP、relax、MD、NEB |

每个 provider 使用隔离环境。不要把这些 requirements 安装进 control plane：

```text
requirements/mace.txt
requirements/uma.txt
requirements/mattersim.txt
requirements/orb.txt
```

远程机器上创建单独环境，并让相应 resource 的 `source_list` 指向激活脚本。仓库提供的参考脚本位于：

```text
configs/dpdispatcher/env_templates/
```

模型权重 token、缓存和许可证变量只放远程私有环境。不要进入 YAML、stage 或 prompt。Backend 只有在依赖、模型权重、device 和最小 smoke case 都通过后才改为 `enabled: true`。

## 8.9 Canonical stage 布局

| Task | Stage 根目录要求 |
|---|---|
| `vasp_execute` | `INCAR`, `POTCAR`, `POSCAR`, `KPOINTS` |
| `vasp_execute_neb` | 根目录含 `INCAR`, `POTCAR`, `KPOINTS`；`00/POSCAR ... NN/POSCAR` |
| `cp2k_execute` | `job.inp`, `manifest.json`, 以及 manifest 引用文件 |
| `lammps_execute` | `in.lammps`, `manifest.json`, `system.data` 或 restart，及势文件 |
| `orca_execute` | `job.inp` 和它引用的本地文件 |
| `xtb_run`, `crest_run` | 默认 `input.xyz`，或 task override 指定输入 |
| `mlff_sp`, `mlff_relax` | `input/` 下直接放结构，可选 `models/` |
| `mlff_md` | `input/` 下恰好一个 start 或 restart 结构 |
| `mlff_neb` | `input/path/00.vasp ... NN.vasp` |
| `mace_train` | `dataset/`, `params/train_params.json` |
| `mace_eval` | `dataset/`, `params/eval_params.json` |

所有输入引用必须留在 stage 内。不要用符号链接指向 project space 外部。`mlff_sp` 和 `mlff_relax` 自身支持 `input/` 下多个结构，不要因为多结构就改用 `remote_submission_batch`。

## 8.10 先查询 task schema

Task schema 是运行时 source of truth。给 agent 的请求可以写：

```text
先调用 get_avail_remote_task，确认 mlff_relax 可用；再用
get_remote_task_spec 查询 backend=mace 的 full schema。
只准备 calculations/si_relax/，列出默认值和需要我确认的 override，暂不提交。
```

`template_overrides` 控制 task 的科学或方法参数。`submission_config` 控制提交层，例如检查间隔、允许的 CPU/GPU override 和清理选项。两者不能混用。Machine 和 resource 由 task/backend 注册关系决定，不是 agent 的自由参数。

## 8.11 单 stage 与 batch

`remote_submission`：

- `work_dir` 本身就是一个完整 stage。
- 调用同步等待到 terminal 状态。
- 默认 `check_interval=30` 秒，`clean_remote=false`。

`remote_submission_batch`：

- 父目录下至少有两个一级子目录。
- 每个一级子目录是独立完整 stage。
- 不递归发现更深目录。
- 全部子任务共享 task 和 config。
- 调用等待所有子任务到 terminal 状态。

工具尚未返回时，不要另开轮询或重复提交。同一个 stage 的重复提交可能产生两个计费作业。

## 8.12 Staging、回传和 receipt

CatMaster 先把 stage 复制到 workspace 的 metadata staging，再由 DPDispatcher 上传。终态后，结果合并回原始 `files/` stage。每个 stage 都会强制回传：

```text
status.json
stdout.log
stderr.log
```

Receipt 的实体路径：

```text
files/.deepagents/dpdispatcher/receipts/
  dp_<timestamp>_<hash8>.json
```

Agent 看到的相对路径从 `.deepagents/...` 开始。关键字段：

| 字段 | 用途 |
|---|---|
| `remote_context_id` | CatMaster 的远程上下文身份 |
| `submission_hash` | DPDispatcher 恢复和下载身份 |
| `receipt_rel` | receipt 的 workspace 相对路径 |
| `task_name`、`work_dir_rel` | 提交的 task 和原 stage |
| `submitted_at`、`updated_at`、`duration_s` | 时间线 |
| `jobs`、`job_status_counts` | 调度作业和状态计数 |
| `resources` | 实际资源摘要 |

成功返回还应包含 `task_count`、`task_state_counts` 和 `submission_dir`。保留这些字段，不要只复制一句“计算完成”。

## 8.13 失败和恢复

网络中断或本地异常不代表远程作业已经取消。处理顺序：

1. 保存 `remote_context_id`、`submission_hash` 和 `receipt_rel`。
2. 确认原工具调用是否已经 terminal；pending 时不检查、不重投。
3. 在集群调度器和 receipt 中判断作业是未创建、排队、运行、终止还是已完成未下载。
4. 优先收集已完成结果和终止日志。
5. 只有确认旧任务不会继续消耗资源或覆盖结果后，才决定重投。
6. `clean_remote=true` 或清理命令只在结果和日志已经下载后使用。

`submission_hash` 为空通常表示没有可恢复的 DPDispatcher record。非空时，可从相应 project 的 `files/` 目录运行：

```bash
dpdisp submission <submission_hash> --download-finished-task
dpdisp submission <submission_hash> --download-terminated-log
dpdisp submission <submission_hash> --reset-fail-count
dpdisp submission <submission_hash> --clean
```

不要按顺序盲目执行四条命令。先下载和分类；只有知道 fail count 或远程清理的后果时再执行后两条。

## 8.14 远程 smoke test

只读列出 suite 和 case：

```bash
python scripts/remote_execution_smoke.py --list
```

实际运行示例：

```bash
python scripts/remote_execution_smoke.py \
  --case mace_sp \
  --project-space /tmp/catmaster_remote_smoke \
  --stop-on-failure
```

除 `--list` 外，该脚本提交真实计算，可能排队、计费并占用许可证。不要一开始运行 `--suite all`。按已配置的单个 backend 或单个 CPU 引擎做最小 case，检查 stage、receipt、日志和回传，再扩大范围。

## 8.15 管理员验收

投入用户使用前逐项确认：

1. 三类 machine 中实际使用的连接都能非交互登录。
2. `remote_root` 可写，Slurm 或 Shell 行为与 card 一致。
3. 所有 `source_list` 文件存在并能加载正确程序。
4. Resource 的 queue、CPU、GPU、walltime 和 audience 正确。
5. 四个活动配置存在，模板未被误当活动文件。
6. Task catalog 只展示已安装、已授权的能力。
7. 每个启用的 MLFF backend 通过独立最小 case。
8. 每个科学引擎产生 `status.json`、stdout、stderr 和 receipt。
9. 模拟传输失败后能从 receipt 收集，而不是重复计算。
10. 活动配置、SSH key、token 和许可证信息不在 Git 或项目文件中。
