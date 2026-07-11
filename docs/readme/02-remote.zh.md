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

第一次使用时复制三类模板：

```bash
cp configs/dpdispatcher/machines_template.yaml configs/dpdispatcher/machines.yaml
cp configs/dpdispatcher/resources_template.yaml configs/dpdispatcher/resources.yaml
cp configs/dpdispatcher/tasks_template.yaml configs/dpdispatcher/tasks.yaml
```

`machines.yaml`、`resources.yaml`、`tasks.yaml` 是实际部署配置，包含登录主机、SSH key、本地/远端 work root、队列、`source_list` 和任务命令，默认被 `.gitignore` 和部署打包脚本排除。公开发布只带 `*_template.yaml`。

模板里默认给出两个机器键：

- `cpu_server_2`
- `gpu_server`

这些名字和 `configs/dpdispatcher/resources.yaml` 里的资源卡匹配。你可以改名字，但如果改了机器键，也要同步修改 resource 里的 `machine` 字段。

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

## 2. 理解资源卡

实际资源卡在：

```text
configs/dpdispatcher/resources.yaml
```

常用 key：

- `vasp_cpu`：VASP 单阶段 CPU 任务。
- `vasp_cpu_neb`：VASP NEB / dimer 路径任务。
- `cp2k_cpu`：CP2K CPU MPI 任务。
- `lammps_cpu`：LAMMPS CPU 任务。
- `mace_gpu`：MACE GPU 任务。
- `uma_gpu`：FairChem UMA GPU 任务，建议使用独立于 MACE 的 conda 环境。

如需启用 MACE cuEquivariance 加速，先安装 `requirements/gpu.txt`，然后按
`torch.version.cuda` 只选择一个 kernel add-on：CUDA 12.x 使用
`requirements/gpu-cueq-cu12.txt`，CUDA 13.x 使用
`requirements/gpu-cueq-cu13.txt`。不要根据 NVIDIA 驱动版本选择 wheel。
在目标模型、体系规模和 GPU 完成基准测试前，MD 配置中的 `enable_cueq`
和 `compile_mode` 应保持显式设置，不要作为无条件默认值。
- `general_cpu`：自定义 CPU boot script。
- `general_gpu`：自定义 GPU boot script。
- `xtb_cpu`、`crest_cpu`、`orca_cpu`：分子量化和构象任务。

resource 里最需要检查：

- `machine` 是否指向 `machines.yaml` 里存在的机器。
- `queue_name` 是否是你的集群队列名。
- `cpu_per_node` / `gpu_per_node` 是否符合队列规则。
- `custom_flags` 是否符合你的 Slurm/PBS 语法。
- `source_list` 或 `prepend_script` 是否能加载 VASP、CP2K、MACE、UMA 等环境。

## 3. 修改实际资源卡

如果 `resources.yaml` 不适合你的集群，直接编辑这个私有文件。它不会进入 git 或部署包。重点检查 `source_list` 和 `prepend_script` 是否能加载远程任务环境。例如 `vasp_cpu`：

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

任务继续引用 `vasp_cpu`，不需要改所有任务模板。

### UMA / FairChem 独立环境

UMA 不建议装进现有 MACE 环境。FairChem/UMA 会牵涉独立的 PyTorch、模型下载和 Hugging Face 鉴权；把它和 `mace_gpu` 共用环境容易互相污染。推荐新建远端环境，例如：

```bash
conda create -n catmaster-uma python=3.11 -y
conda activate catmaster-uma
pip install -r requirements/uma.txt
```

然后让 `uma_gpu.source_list` 指向一个只负责 UMA 的脚本。可直接按下面模板修改：

```bash
#!/usr/bin/env bash
set -euo pipefail

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate catmaster-uma

export HF_HOME=/path/to/shared/cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME=/path/to/shared/cache/torch
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.config/huggingface/token" ]; then
  export HF_TOKEN="$(tr -d '\n' < "$HOME/.config/huggingface/token")"
fi

# 仅在模型已经预热到 HF_HUB_CACHE 后再打开离线模式。
# export HF_HUB_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-8}}"
```

不要把 `HF_TOKEN` 写进 `configs/dpdispatcher/tasks.yaml`、`params`、stage 文件或提示词参数。UMA checkpoint 体积大，也不应放进 `forward_files` 传输。第一次运行可以让远端下载模型；如果计算节点没有公网，先在登录节点或可联网节点执行一次预热：

```bash
mkdir -p "$HOME/.config/huggingface"
chmod 700 "$HOME/.config/huggingface"
printf '%s' '<hf_token_with_facebook_UMA_access>' > "$HOME/.config/huggingface/token"
chmod 600 "$HOME/.config/huggingface/token"
```

确认 token 文件存在后再预热模型：

```bash
source /path/to/catmaster_env_uma.sh
python - <<'PY'
from fairchem.core import pretrained_mlip
pretrained_mlip.get_predict_unit("uma-s-1p2", device="cpu")
PY
```

预热完成并确认计算节点共享同一个 cache 后，才考虑设置 `HF_HUB_OFFLINE=1`。

## 4. 理解任务配置

实际任务在：

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
- `uma_sp_dir`
- `uma_relax_dir`
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

如果需要自定义远程任务，编辑私有的 `configs/dpdispatcher/tasks.yaml`。

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

文件名里带 `template` 的 YAML 只作为示例，不会被当成真实配置读取。真正想启用的配置应复制为 `machines.yaml`、`resources.yaml`、`tasks.yaml`。

## 6. 在 WebUI 中使用远程任务

远程配置完成后，可以在任务里明确说明：

```text
把当前项目中的 vasp_inputs/CO_on_Ni_top 作为一个已准备好的 VASP stage，用 remote_submission 提交 task_name=vasp_execute，并返回 remote_context_id、receipt_rel 和 status.json。
```

对于 MACE relax：

```text
将 structures/ 里的结构整理成 mace_relax_dir 需要的 input/ stage，然后用 remote_submission 提交，模型用 mh-1，head 用 omat_pbe。
```

对于周期材料或催化体系的 UMA single point：

```text
将 structures/ 里的结构整理成 uma_sp_dir 需要的 input/ stage，用 remote_submission 提交，params 设置 model=uma-s-1p2、uma_task=omat。若是 OC20/OC25/ODAC/OMC 语义任务，不要依赖 auto，明确设置对应 uma_task。
```

对于分子或团簇的 UMA 预优化：

```text
将 molecules/ 里的结构整理成 uma_relax_dir 需要的 input/ stage，用 remote_submission 提交，params 设置 uma_task=omol、charge=0、spin=1、relax_cell=false。后续正式能量、频率、TS 或光谱仍需 ORCA/xTB/DFT 验证。
```

`uma_task=auto` 只做保守的分子/周期粗分：有有效周期 cell 时默认 `omat`，否则默认 `omol`。`omol` 的 `charge` 和 `spin` 会写入 ASE `Atoms.info`；非 `omol` 任务请保持 `charge=0`、`spin=0`。

## 7. Single 与 Batch 提交

`remote_submission` 和 `remote_submission_batch` 的区别是 boot script 在哪里启动：

- `remote_submission`：`work_dir` 本身就是一个 stage，boot script 直接在 `work_dir` 里启动一次。
- `remote_submission_batch`：`work_dir` 是父目录，boot script 会在每个第一层子目录里各启动一次；父目录本身不会作为任务 cwd 启动。

```text
remote_submission:
stage/
  INCAR
  POSCAR
  KPOINTS
  POTCAR

remote_submission_batch:
batch_root/
  job_a/
    INCAR
    POSCAR
    KPOINTS
    POTCAR
  job_b/
    INCAR
    POSCAR
    KPOINTS
    POTCAR
```

不要因为一个 stage 内部包含多个科学输入就自动改用 `remote_submission_batch`。例如 `mace_sp_dir`、`mace_relax_dir`、`uma_sp_dir` 和 `uma_relax_dir` 的一个 stage 可以在 `input/` 下放多个结构，这仍然是一次 `remote_submission`；只有当你准备了多个独立的 stage 子目录时才用 `remote_submission_batch`。

## 8. 真实远程 smoke test

部署到远端后，优先跑脚本式 smoke test。它会准备极小 O2/H2O 输入，走 CatMaster 当前 agent 可见的 `remote_submission` 路径，并提交真实 DPDispatcher 任务，不是 dry-run。

查看可用 suite 和 case：

```bash
python scripts/remote_execution_smoke.py --list
```

常用最小覆盖：

```bash
python scripts/remote_execution_smoke.py --suite core --check-interval 30
```

`core` 会实跑 `mace_sp`、`xtb_sp`、`orca_sp`。更完整的材料侧覆盖：

```bash
python scripts/remote_execution_smoke.py --suite materials --check-interval 60
```

全部常规远程执行覆盖：

```bash
python scripts/remote_execution_smoke.py --suite all --check-interval 60
```

UMA 需要额外的 FairChem 环境、Hugging Face 权限和模型 cache，因此单独放在 `uma` suite，不并入 `core` 或 `all`。该 suite 会覆盖分子/周期 single point 和短 relax：

```bash
python scripts/remote_execution_smoke.py --suite uma --uma-model uma-s-1p2 --uma-task omat --uma-check-interval 60
```

脚本默认把阶段文件和 JSON 报告写到 `/tmp/catmaster_remote_execution_smoke`。需要固定目录时：

```bash
python scripts/remote_execution_smoke.py --suite core --project-space /path/to/catmaster_remote_smoke
```

仍然保留 pytest 入口，适合开发时跑：

```bash
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 pytest tests/remote_execution -s -vv
```

常用覆盖变量见：

```text
tests/remote_execution/README.md
```

## 9. 常见问题

`Machine 'xxx' not found`

`configs/dpdispatcher/resources.yaml` 的 `machine` 字段没有对应到 `configs/dpdispatcher/machines.yaml` 中的机器 key。

`Resources 'xxx' not found`

任务引用的 resource key 没有在 `configs/dpdispatcher/resources.yaml` 中定义。

SSH 失败

先在终端直接跑 `ssh`，不要先从 CatMaster 排查。确认 host、port、username、key 和远程防火墙。

远程命令找不到 VASP/MACE/CP2K/UMA

检查 resource 的 `source_list` 或 `prepend_script`，确认远程 job 环境里能找到对应程序。

UMA 报模型下载、鉴权或离线 cache 错误

先在远端直接 `source catmaster_env_uma.sh`，确认 `python -c "from fairchem.core import pretrained_mlip"` 可运行；再检查 `HF_TOKEN` 是否只在远端环境里设置、`HF_HOME/HF_HUB_CACHE` 是否指向计算节点可读写的持久目录、是否在未预热模型前打开了 `HF_HUB_OFFLINE=1`。如果错误包含 `GatedRepoError` 或 `Access to model facebook/UMA is restricted`，说明远端 token 不存在、没有被 source script 读到，或该 Hugging Face 账号尚未获准访问 `facebook/UMA`。

结果没有下载回来

检查 task 的 `backward_files`，以及远程目录权限和 DPDispatcher 日志。

## 9. 下一步

远程提交成功后，回到 [功能介绍与日常使用](03-features.zh.md)，把远程任务作为 Experiment lane 的一部分使用。
