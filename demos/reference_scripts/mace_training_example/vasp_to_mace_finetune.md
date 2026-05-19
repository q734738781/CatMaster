# VASP 到 MACE 微调工作流

本文档整理当前仓库里已经落地并验证过的 VASP 数据抽取、MACE 训练数据导出、`estimated E0s` 评估，以及 `mace-mh-1` 的在线微调方案。

适用环境：

- `conda activate catmaster`
- 已验证可用版本：
  - `ase 3.27.0`
  - `torch 2.10.0+cu128`
  - `mace 0.3.15`

## 1. 总体流程

推荐顺序：

1. 从 `vasprun.xml` 抽取全部 ionic steps 到 `ase db`
2. 用 `step_electronic_converged_guess` 做后过滤，不在第一步硬编码丢帧
3. 从 `ase db` 导出 `extxyz`
4. 用 `mace-mh-1` 的 `omat_pbe` head 评估 `estimated E0s`
5. 准备 `train/valid`，必要时再加 `test`
6. 决定训练时是直接 `--E0s estimated`，还是固定成 `json` 后再训练
7. 在 `naive / multihead replay / LoRA` 三条官方路径里选一条起跑

## 2. VASP 抽取

脚本：

- [extract_elec_converged_ase_db.py](/mnt/ssd/chenhh/python_projects/Battery/prep_dft_ft_data/extract_elec_converged_ase_db.py)

设计原则：

- 结构、能量、自由能、力、应力、约束、calculator 参数都交给 `ase.io.vasp.read_vasp_xml()`
- XML 自解析只负责：
  - 读取 `NELM`
  - 统计每个 ionic step 的 `scstep` 数量
  - 用于生成 `step_electronic_converged_guess`

当前默认行为：

- 默认写入全部 ionic steps
- 不默认丢弃电子不收敛步
- 额外保存 `step_electronic_converged_guess`
- 可选 `--only-electronically-converged` 做硬过滤

默认输出：

- `DFT_files/ionic_steps.db`
- `DFT_files/ionic_steps.db.metadata.csv`

如果加 `--only-electronically-converged`，则可直接输出过滤后的：

- `DFT_files/ionic_steps_converged.db`
- `DFT_files/ionic_steps_converged.db.metadata.csv`

示例：

```bash
source /home/chenhh/miniconda3/etc/profile.d/conda.sh
conda activate catmaster

python prep_dft_ft_data/extract_elec_converged_ase_db.py \
  --root DFT_files/v0 \
  --pattern vasprun.xml \
  --db-path DFT_files/ionic_steps_converged.db \
  --overwrite \
  --workers 32
```

如果你要直接硬过滤，只保留 `electronic_step_count < NELM` 的步，再额外加：

```bash
--only-electronically-converged
```

### 2.1 对齐检查

这一步很重要，当前脚本默认开启对齐检查：

- 数量对齐：XML `<calculation>` 数量 vs ASE 读出的 ionic steps 数量
- 逐步对齐：XML `e_fr_energy` vs ASE `free_energy`

相关参数：

- `--alignment-check` / `--no-alignment-check`
- `--alignment-energy-atol 1e-6`

如果逐步 free energy 对不上，脚本会把该文件记为失败，不会静默写入错位数据。

### 2.2 已验证行为

在 `catmaster` 环境里已经做过以下验证：

- 单文件样本 `A__S208__Na3p5__r05/vasprun.xml`：通过
- 多步样本 `A__S208__Na3p5__base/vasprun.xml`：通过
  - `84` 个 ionic steps
  - `81` 个 `step_electronic_converged_guess=True`
  - `3` 个 `False`
- 并行一致性测试：
  - 测试集：4 个真实 `vasprun.xml`
  - 总计 `376` 个 ionic steps
  - `--workers 1` 和 `--workers 4` 输出完全一致

### 2.3 DB 中保留的信息

除了 `Atoms` 本身携带的结构/能量/力/应力/约束之外，`db.write(..., key_value_pairs=..., data=...)` 里还保留：

- `frame_uid`
- `source_relpath`
- `source_dirname`
- `ionic_step_index`
- `ionic_step_number`
- `electronic_step_count`
- `nelm`
- `step_electronic_converged_guess`
- `has_constraints`
- `constraint_types`
- `selected_parameters`

`selected_parameters` 当前保留：

- `isif`
- `pstress`
- `ibrion`
- `nsw`
- `nelm`
- `algo`
- `lepsilon`
- `ediff`
- `ediffg`

## 3. 从 ASE DB 导出到 MACE extxyz

脚本：

- [export_ase_db_to_mace_xyz.py](/mnt/ssd/chenhh/python_projects/Battery/prep_dft_ft_data/export_ase_db_to_mace_xyz.py)

该脚本会把 DB 中参考量显式写成 MACE 友好的键名：

- `REF_energy`
- `REF_forces`
- `REF_stress`

推荐导出“已经过后过滤”的子集，而不是直接把所有步全塞给训练。

示例：

```bash
python prep_dft_ft_data/export_ase_db_to_mace_xyz.py \
  --db-path DFT_files/ionic_steps_converged.db \
  --out-path DFT_files/mace_ft/all.extxyz \
  --head omat_pbe \
  --config-type dft \
  --overwrite
```

说明：

- 如果输入库已经是 `ionic_steps_converged.db`，通常不再需要额外写 `--selection`
- 如果输入库还是全量 `ionic_steps.db`，则可加 `--selection "step_electronic_converged_guess=1"`
- 如果你想先做小样本评估，可以加 `--limit`

## 4. `estimated E0s` 评估

脚本：

- [estimate_mace_e0s.py](/mnt/ssd/chenhh/python_projects/Battery/prep_dft_ft_data/estimate_mace_e0s.py)

它直接复用 MACE 官方的 `estimate_e0s_from_foundation()`：

- 先用 foundation model 预测训练集总能量
- 计算残差 `E_ref - E_foundation`
- 解最小二乘 `A @ delta ≈ residual`
- 得到 `E0_new = E0_foundation + delta`

示例：

```bash
python prep_dft_ft_data/estimate_mace_e0s.py \
  --train-file DFT_files/mace_ft/train.extxyz \
  --foundation-model models/mace-mh-1.model \
  --foundation-head omat_pbe \
  --out-path DFT_files/mace_ft/e0s_estimated.json \
  --device cuda \
  --overwrite
```

### 4.1 当前这次评估的结论

当前你已经跑过一次 `estimated E0s`，结果大意如下：

- 训练集大小：`43873` 个构型
- 元素数：`27`
- 线性系统秩：`25/27`
- `RMSE before`: `2.141791 eV`
- `RMSE after`: `1.556828 eV`
- `MAE before`: `1.628301 eV`
- `MAE after`: `1.216965 eV`

解释：

- 这说明 `estimated E0s` 是有帮助的，误差下降约 `25% - 27%`
- 但 `rank 25/27` 表明体系在成分空间里有共线性，至少有 2 个自由度不可唯一辨识
- 所以它适合当训练前的能量零点修正，但不适合直接当“物理上最终可信的 isolated-atom E0”

简化判断：

- 作为训练初始化：可以用
- 作为严格标定：不够干净

### 4.2 是否可以直接在训练里用 `--E0s estimated`

可以。

MACE 官方训练入口支持：

```bash
--E0s estimated --foundation_model ...
```

也就是说，训练时可以直接在线估计，不必先离线写成 `json`。

但两种做法有区别：

- 快速起跑：直接 `--E0s estimated`
- 更可复现：先离线生成 `e0s_estimated.json`，训练时显式传这个文件

当前建议：

- 如果只是评估或快速试跑，可以直接用 `estimated`
- 如果进入正式对比实验，建议固定成 `json`

## 5. 训练/验证划分

MACE 官方训练入口支持三种数据集划分方式：

- 只给 `--train_file`，再用 `--valid_fraction`
- 显式给 `--train_file` 和 `--valid_file`
- 在上面基础上再加 `--test_file`

对当前项目，建议分成两种用法。

### 5.1 快速试跑

如果你只是想先看微调是否工作，最省事的方式是：

1. 先导出一个总文件 `all.extxyz`
2. 在 `mace_run_train` 里直接用 `--valid_fraction 0.05`

这条路适合：

- 快速验证 `mh-1 + omat_pbe` 是否能正常起训
- 快速比较 `E0s=foundation` 和 `E0s=estimated`
- 不要求严格可复现的数据拆分

### 5.2 正式实验

如果你要做正式对比，建议显式固定：

- `train.extxyz`
- `valid.extxyz`
- 可选 `test.extxyz`

这样后面比较：

- `E0s=foundation`
- `E0s=estimated`
- `E0s=e0s_estimated.json`
- `naive / replay / LoRA`

时，不会掺进随机划分噪声。

当前仓库里还没有单独的 split 脚本，所以现在最简单的做法是先导出 `all.extxyz`，再固定随机种子拆分一次。

示例：

```bash
python prep_dft_ft_data/export_ase_db_to_mace_xyz.py \
  --db-path DFT_files/ionic_steps_converged.db \
  --out-path DFT_files/mace_ft/all.extxyz \
  --head omat_pbe \
  --config-type dft \
  --overwrite
```

```bash
python - <<'PY'
from random import Random
from ase.io import read, write

seed = 3
valid_fraction = 0.1
test_fraction = 0.1

frames = read("DFT_files/mace_ft/all.extxyz", index=":")
idx = list(range(len(frames)))
Random(seed).shuffle(idx)

n_total = len(idx)
n_valid = int(round(n_total * valid_fraction))
n_test = int(round(n_total * test_fraction))
n_train = n_total - n_valid - n_test

train_idx = idx[:n_train]
valid_idx = idx[n_train:n_train + n_valid]
test_idx = idx[n_train + n_valid:]

write("DFT_files/mace_ft/train.extxyz", [frames[i] for i in train_idx])
write("DFT_files/mace_ft/valid.extxyz", [frames[i] for i in valid_idx])
write("DFT_files/mace_ft/test.extxyz", [frames[i] for i in test_idx])

print("total =", n_total)
print("train =", len(train_idx))
print("valid =", len(valid_idx))
print("test  =", len(test_idx))
PY
```

### 5.3 当前数据量对应的划分量级

你当前过滤后的训练池是 `43873` 个构型。

如果按 `5% valid + 5% test` 拆分，大约是：

- `train`: `39485`
- `valid`: `2194`
- `test`: `2194`

## 6. `mace-mh-1` 微调建议

本项目当前使用：

- foundation model: `models/mace-mh-1.model`
- foundation head: `omat_pbe`

这点很重要。多头模型训练时应显式写：

- `--foundation_model models/mace-mh-1.model`
- `--foundation_head omat_pbe`

不要依赖默认 head 选择。

### 6.1 `E0s` 选择建议

对当前数据和现有评估结果，建议优先级：

1. 先做快速试跑时：`estimated`
2. 做正式可复现实验时：固定 `e0s_estimated.json`
3. 如果后面愿意补 isolated atom 单点，再考虑自算 `E0s`

不建议：

- 对 `multiheads_finetuning=True` 使用 `average`

### 6.2 推荐训练起点

当前最务实的第一版方案：

- 用 `mh-1`
- 指定 `omat_pbe`
- 用已经过滤过的 `train.extxyz`
- 先试 `estimated E0s`
- 先不要自定义复杂 freeze 策略

如果后面要走官方推荐的 foundation finetuning 路线，可以继续评估：

- `multihead replay finetuning`
- `LoRA` 低秩微调

但第一轮建议先把基线跑通。

## 7. 在线微调方法

MACE 官方当前主要给出三条路径：

1. naive fine-tuning
2. multihead replay fine-tuning
3. LoRA fine-tuning

- `multihead replay` 是更稳、更推荐的 foundation model 微调方式
- `LoRA` 更适合小数据，且官方更推荐和 naive 结合

### 7.1 Naive fine-tuning

适合：

- 快速 baseline
- 没准备 replay 数据集
- 想先验证 `estimated E0s` 是否有帮助

推荐起点：

```bash
mace_run_train \
  --name battery_mh1_naive \
  --foundation_model models/mace-mh-1.model \
  --foundation_head omat_pbe \
  --multiheads_finetuning=False \
  --train_file DFT_files/mace_ft/train.extxyz \
  --valid_fraction 0.05 \
  --E0s estimated \
  --energy_weight 1.0 \
  --forces_weight 10.0 \
  --stress_weight 1.0 \
  --scaling rms_forces_scaling \
  --batch_size 2 \
  --max_num_epochs 20 \
  --ema \
  --ema_decay 0.99 \
  --amsgrad \
  --default_dtype float64 \
  --device cuda \
  --seed 3
```

如果你已经拆好了固定数据集，直接把：

- `--valid_fraction 0.05`

换成：

- `--valid_file DFT_files/mace_ft/valid.extxyz`
- 可选 `--test_file DFT_files/mace_ft/test.extxyz`

如果你已经固定了 `e0s_estimated.json`，可以把：

- `--E0s estimated`

改成：

- `--E0s DFT_files/mace_ft/e0s_estimated.json`

### 7.2 Multihead replay fine-tuning

适合：

- 想尽量避免 catastrophic forgetting
- 准备做正式 foundation-model 微调

官方文档说明，`multihead replay` 需要 replay 数据集。对于你当前这种本地路径 foundation model，最稳妥的方式是显式提供：

- `--pt_train_file path/to/replay_dataset.xyz`

而不是依赖 `mp` shortcut。

关键点：

- 需要 `--multiheads_finetuning True`
- 需要 `--pt_train_file`
- 需要 `--atomic_numbers` 指定目标数据集元素
- 官方 troubleshooting 建议的起始超参数是 `lr=1e-4`、`ema_decay=0.99999`、`num_samples_pt=100000`、`forces_weight=10`、`energy_weight=1`、`stress_weight=1`

你当前数据集元素可直接写成：

```text
[8, 11, 12, 13, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 38, 39, 40, 41, 42, 44, 45, 46, 49, 50]
```

推荐起点：

```bash
mace_run_train \
  --name battery_mh1_replay \
  --foundation_model models/mace-mh-1.model \
  --foundation_head omat_pbe \
  --train_file DFT_files/mace_ft/train.extxyz \
  --valid_file DFT_files/mace_ft/valid.extxyz \
  --test_file DFT_files/mace_ft/test.extxyz \
  --pt_train_file omat \
  --num_samples_pt 100000 \
  --filter_type_pt combinations \
  --subselect_pt fps \
  --weight_pt_head 1.0 \
  --atomic_numbers "[8, 11, 12, 13, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 38, 39, 40, 41, 42, 44, 45, 46, 49, 50]" \
  --multiheads_finetuning True \
  --E0s estimated \
  --lr 1e-4 \
  --ema \
  --ema_decay 0.99999 \
  --energy_weight 1.0 \
  --forces_weight 10.0 \
  --stress_weight 1.0 \
  --default_dtype float64 \
  --max_num_epochs 15 \
  --batch_size 4 \
  --device cuda \
  --seed 42
```

说明：

- 当前仓库内置与参考脚本默认都从 `--batch_size 4` 起跑，优先保守显存占用和训练稳定性；只有在显存和吞吐都确认宽松时再往上加。
- 如果你已经有 `e0s_estimated.json`，也可以把 `--E0s estimated` 改成固定文件
- 如果你还没有固定 `valid.extxyz` / `test.extxyz`，也可以先退回 `--valid_fraction 0.05`
- 官方文档建议多看初始误差；如果一开始 energy 误差非常高，先怀疑 `E0s`

### 7.3 LoRA fine-tuning

适合：

- 数据量不大
- 想减少过拟合风险
- 想降低训练内存和计算开销

官方文档明确说明：

- `LoRA` 会自动冻结 base model 权重
- 最终保存时会自动 merge 回普通 MACE 权重
- 更推荐 `LoRA + naive`，而不是 `LoRA + replay`

推荐起点：

```bash
mace_run_train \
  --name battery_mh1_lora \
  --foundation_model models/mace-mh-1.model \
  --foundation_head omat_pbe \
  --multiheads_finetuning=False \
  --train_file DFT_files/mace_ft/train.extxyz \
  --valid_fraction 0.05 \
  --lora True \
  --lora_rank 4 \
  --lora_alpha 1.0 \
  --E0s estimated \
  --energy_weight 1.0 \
  --forces_weight 10.0 \
  --stress_weight 1.0 \
  --lr 0.005 \
  --weight_decay 0.0 \
  --ema \
  --ema_decay 0.995 \
  --clip_grad 10.0 \
  --batch_size 2 \
  --max_num_epochs 20 \
  --default_dtype float64 \
  --device cuda \
  --seed 3
```

LoRA 常用调参建议：

- `rank=4` 先起步
- 数据更少时试 `rank=2`
- 明显欠拟合时试 `rank=8`

如果你已经固定数据集，同样建议把：

- `--valid_fraction 0.05`

换成：

- `--valid_file DFT_files/mace_ft/valid.extxyz`
- 可选 `--test_file DFT_files/mace_ft/test.extxyz`

## 8. 推荐的实际命令

### 8.1 抽取并过滤 ionic steps

```bash
source /home/chenhh/miniconda3/etc/profile.d/conda.sh
conda activate catmaster

python prep_dft_ft_data/extract_elec_converged_ase_db.py \
  --root DFT_files/v0 \
  --pattern vasprun.xml \
  --db-path DFT_files/ionic_steps_converged.db \
  --only-electronically-converged \
  --overwrite \
  --workers 32
```

### 8.2 导出总训练池

```bash
python prep_dft_ft_data/export_ase_db_to_mace_xyz.py \
  --db-path DFT_files/ionic_steps_converged.db \
  --out-path DFT_files/mace_ft/all.extxyz \
  --head omat_pbe \
  --config-type dft \
  --overwrite
```

### 8.3 固定 `train/valid/test` 划分

```bash
python - <<'PY'
from random import Random
from ase.io import read, write

seed = 3
valid_fraction = 0.05
test_fraction = 0.05

frames = read("DFT_files/mace_ft/all.extxyz", index=":")
idx = list(range(len(frames)))
Random(seed).shuffle(idx)

n_total = len(idx)
n_valid = int(round(n_total * valid_fraction))
n_test = int(round(n_total * test_fraction))
n_train = n_total - n_valid - n_test

train_idx = idx[:n_train]
valid_idx = idx[n_train:n_train + n_valid]
test_idx = idx[n_train + n_valid:]

write("DFT_files/mace_ft/train.extxyz", [frames[i] for i in train_idx])
write("DFT_files/mace_ft/valid.extxyz", [frames[i] for i in valid_idx])
write("DFT_files/mace_ft/test.extxyz", [frames[i] for i in test_idx])

print("total =", n_total)
print("train =", len(train_idx))
print("valid =", len(valid_idx))
print("test  =", len(test_idx))
PY
```

### 8.4 只做 `estimated E0s` 评估

```bash
python prep_dft_ft_data/estimate_mace_e0s.py \
  --train-file DFT_files/mace_ft/train.extxyz \
  --foundation-model models/mace-mh-1.model \
  --foundation-head omat_pbe \
  --out-path DFT_files/mace_ft/e0s_estimated.json \
  --device cuda \
  --overwrite
```

### 8.5 直接在线使用 `estimated E0s` 做 naive 微调

```bash
mace_run_train \
  --name battery_mh1_naive \
  --foundation_model models/mace-mh-1.model \
  --foundation_head omat_pbe \
  --train_file DFT_files/mace_ft/train.extxyz \
  --valid_file DFT_files/mace_ft/valid.extxyz \
  --test_file DFT_files/mace_ft/test.extxyz \
  --multiheads_finetuning False \
  --E0s estimated \
  --energy_weight 1.0 \
  --forces_weight 10.0 \
  --stress_weight 1.0 \
  --default_dtype float64 \
  --device cuda
```

### 8.6 用固定 `json` 做 naive 微调

```bash
mace_run_train \
  --name battery_mh1_naive_e0json \
  --foundation_model models/mace-mh-1.model \
  --foundation_head omat_pbe \
  --train_file DFT_files/mace_ft/train.extxyz \
  --valid_file DFT_files/mace_ft/valid.extxyz \
  --test_file DFT_files/mace_ft/test.extxyz \
  --multiheads_finetuning False \
  --E0s DFT_files/mace_ft/e0s_estimated.json \
  --energy_weight 1.0 \
  --forces_weight 10.0 \
  --stress_weight 1.0 \
  --default_dtype float64 \
  --device cuda
```

### 8.7 直接在线使用 `estimated E0s` 做 multihead replay 微调

```bash
mace_run_train \
  --name battery_mh1_replay \
  --foundation_model models/mace-mh-1.model \
  --foundation_head omat_pbe \
  --train_file DFT_files/mace_ft/train.extxyz \
  --valid_file DFT_files/mace_ft/valid.extxyz \
  --test_file DFT_files/mace_ft/test.extxyz \
  --pt_train_file path/to/replay_dataset.xyz \
  --num_samples_pt 100000 \
  --filter_type_pt combinations \
  --subselect_pt fps \
  --weight_pt_head 1.0 \
  --atomic_numbers "[8, 11, 12, 13, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 38, 39, 40, 41, 42, 44, 45, 46, 49, 50]" \
  --multiheads_finetuning True \
  --E0s estimated \
  --lr 1e-4 \
  --ema \
  --ema_decay 0.99999 \
  --default_dtype float64 \
  --device cuda
```

### 8.8 LoRA + naive 微调

```bash
mace_run_train \
  --name battery_mh1_lora \
  --foundation_model models/mace-mh-1.model \
  --foundation_head omat_pbe \
  --train_file DFT_files/mace_ft/train.extxyz \
  --valid_file DFT_files/mace_ft/valid.extxyz \
  --test_file DFT_files/mace_ft/test.extxyz \
  --multiheads_finetuning False \
  --lora True \
  --lora_rank 4 \
  --lora_alpha 1.0 \
  --E0s estimated \
  --default_dtype float64 \
  --device cuda
```

## 9. 当前建议总结

- VASP 抽库阶段：
  - 保留全部 ionic steps
  - 不在第一步永久丢帧
  - 后续用 `step_electronic_converged_guess=1` 做训练集过滤
- `estimated E0s`：
  - 当前结果说明它有效
  - 但因为 `rank deficient`，不要把各元素修正量过度物理解读
- 微调阶段：
  - `mh-1 + omat_pbe` 必须显式指定 head
  - 至少要明确 `valid_fraction` 或固定 `valid_file`
  - 正式实验建议再加 `test_file`
  - 第一轮基线可先做 `naive + estimated`
  - 正式实验优先考虑 `multihead replay`
  - 小数据时可试 `LoRA + naive`
  - 正式对比实验更推荐固定 `json`

## 10. 官方文档

本节基于以下 MACE 官方文档整理：

- Fine-tuning Foundation Models: https://mace-docs.readthedocs.io/en/latest/guide/finetuning.html
- Multihead Replay Finetuning: https://mace-docs.readthedocs.io/en/latest/guide/multihead_finetuning.html
- LoRA Fine-tuning: https://mace-docs.readthedocs.io/en/latest/guide/lora_finetuning.html
- Heterogeneous Data Training: https://mace-docs.readthedocs.io/en/latest/guide/heterogeneous_data.html
- Troubleshooting and Q&A Guide: https://mace-docs.readthedocs.io/en/latest/guide/troubleshooting.html
