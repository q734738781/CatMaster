# 远程配置要点

本章只在你需要把计算任务提交到远程机器或集群时使用。如果只是本地启动 WebUI、写作、文献综述或分析已有文件，可以先跳过。

CatMaster 的远程提交基于 DPDispatcher。你需要把三类事情配置清楚：

- `machine`：远程机器怎么连接，例如 SSH host、用户名、key、工作目录。
- `resource`：用哪台机器、哪个队列、多少 CPU/GPU、加载哪些环境。
- `task`：提交什么命令、上传哪些文件、下载哪些结果。

## 1. 配置机器

程序会读取：

```text
configs/dpdispatcher/machines.yaml
```

第一次使用时复制模板：

```bash
cp configs/machines_template.yaml configs/dpdispatcher/machines.yaml
```

模板里默认给出两个机器键：

- `cpu_server_2`
- `gpu_server`

这些名字和 `configs/dpdispatcher/resources.yaml` 里的默认资源卡匹配。你可以改名字，但如果改了机器键，也要同步修改 resource 里的 `machine` 字段。

一个简化示例：

```yaml
cpu_server_2:
  batch_type: Slurm
  context_type: SSHContext
  local_root: /path/to/local/dpdispatcher_work
  remote_root: /path/to/remote/dpdispatcher_work
  retry_count: 0
  remote_profile:
    hostname: login.cluster.edu
    port: 22
    username: your_user
    key_filename: /home/your_user/.ssh/id_ed25519
  env_setup: |
    ulimit -s unlimited
    module load python/3.10
```

先确认普通 SSH 能连通：

```bash
ssh -i /home/your_user/.ssh/id_ed25519 your_user@login.cluster.edu
```

## 2. 理解默认资源卡

默认资源卡在：

```text
configs/dpdispatcher/resources.yaml
```

常用 key：

- `vasp_cpu`：VASP 单阶段 CPU 任务。
- `vasp_cpu_neb`：VASP NEB / dimer 路径任务。
- `cp2k_cpu`：CP2K CPU MPI 任务。
- `lammps_cpu`：LAMMPS CPU 任务。
- `mace_gpu`：MACE GPU 任务。
- `general_cpu`：自定义 CPU boot script。
- `general_gpu`：自定义 GPU boot script。
- `xtb_cpu`、`crest_cpu`、`orca_cpu`：分子量化和构象任务。

resource 里最需要检查：

- `machine` 是否指向 `machines.yaml` 里存在的机器。
- `queue_name` 是否是你的集群队列名。
- `cpu_per_node` / `gpu_per_node` 是否符合队列规则。
- `custom_flags` 是否符合你的 Slurm/PBS 语法。
- `source_list` 或 `prepend_script` 是否能加载 VASP、CP2K、MACE 等环境。

## 3. 使用本地资源覆盖文件

如果默认 `resources.yaml` 不适合你的集群，建议复制模板到本地覆盖文件：

```bash
cp configs/dpdispatcher/resources_template.yaml configs/dpdispatcher/resources_local.yaml
```

你可以在 `resources_local.yaml` 里写同名 key 来覆盖默认资源卡。例如覆盖 `vasp_cpu`：

```yaml
vasp_cpu:
  kind: domain
  capabilities: [vasp]
  description: "My VASP CPU queue."
  audiences: [materials_worker]
  machine: cpu_server_2
  number_node: 1
  cpu_per_node: 64
  queue_name: normal
  group_size: 1
  custom_flags:
    - "#SBATCH -t 2-00:00:00"
    - "#SBATCH --export=ALL"
  source_list:
    - /path/to/remote/vasp_env.sh
```

同名 resource key 会覆盖默认定义。这样任务仍然可以继续引用 `vasp_cpu`，不需要改所有任务模板。

## 4. 理解任务模板

默认任务在：

```text
configs/dpdispatcher/tasks.yaml
```

任务 key 例子：

- `vasp_execute`
- `vasp_execute_neb`
- `cp2k_execute`
- `lammps_execute`
- `mace_relax_dir`
- `mace_sp_dir`
- `mace_md_dir`
- `mace_neb_dir`
- `mace_train_dir`
- `mace_eval_dir`
- `xtb_run`

每个 task 定义：

- `resources`：默认用哪个 resource key。
- `boot_script`：会复制到远程 stage 的启动脚本。
- `command`：远程执行命令。
- `defaults`：命令模板参数默认值。
- `forward_files`：上传文件。
- `backward_files`：下载文件。
- `task_work_path`：远程命令在哪个 stage 子目录执行。

## 5. 自定义任务

如果需要自定义远程任务，复制模板：

```bash
cp configs/dpdispatcher/tasks_template.yaml configs/dpdispatcher/tasks_local.yaml
```

示例：

```yaml
my_python_task:
  audiences: [materials_worker]
  description: "Run my prepared Python stage."
  boot_script: "catmaster/remote/cpu/vasp_boot.py"
  resources: general_cpu
  requires: [python]
  command: "python task_script/vasp_boot.py"
  forward_files:
    - "*"
    - "task_script/vasp_boot.py"
  backward_files:
    - "*"
  task_work_path: "."
```

文件名里带 `template` 的 YAML 只作为示例，不会被当成真实配置读取。真正想启用的配置应放在 `resources_local.yaml` 或 `tasks_local.yaml`。

## 6. 在 WebUI 中使用远程任务

远程配置完成后，可以在任务里明确说明：

```text
把当前项目中的 vasp_inputs/CO_on_Ni_top 作为一个已准备好的 VASP stage，用 remote_submission 提交 task_name=vasp_execute，并返回 remote_context_id、receipt_rel 和 status.json。
```

对于 MACE relax：

```text
将 structures/ 里的结构整理成 mace_relax_dir 需要的 input/ stage，然后用 remote_submission 提交，模型用 mh-1，head 用 omat_pbe。
```

## 7. 可选真实远程 smoke test

真实提交测试默认不运行。确认集群配置完成后，可以显式开启：

```bash
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 pytest tests/remote_execution -s -vv
```

常用覆盖变量见：

```text
tests/remote_execution/README.md
```

## 8. 常见问题

`Machine 'xxx' not found`

`resources.yaml` 或 `resources_local.yaml` 的 `machine` 字段没有对应到 `configs/dpdispatcher/machines.yaml` 中的机器 key。

`Resources 'xxx' not found`

任务引用的 resource key 没有在 `resources.yaml` 或 `resources_local.yaml` 中定义。

SSH 失败

先在终端直接跑 `ssh`，不要先从 CatMaster 排查。确认 host、port、username、key 和远程防火墙。

远程命令找不到 VASP/MACE/CP2K

检查 resource 的 `source_list` 或 `prepend_script`，确认远程 job 环境里能找到对应程序。

结果没有下载回来

检查 task 的 `backward_files`，以及远程目录权限和 DPDispatcher 日志。

## 9. 下一步

远程提交成功后，回到 [功能介绍与日常使用](03-features.zh.md)，把远程任务作为 Experiment lane 的一部分使用。
