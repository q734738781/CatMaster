# 5. Experiment Agent 与四类计算 Worker

[上一章](04-webui.zh.md) | [目录](README.zh.md) | [下一章](06-computational-workflows.zh.md)

Experiment 是 CatMaster 的计算入口。它负责理解科学目标、查看项目中已有输入和结果、决定应该交给哪一类 worker，并在 worker 返回后检查工作是否足以回答当前问题。真正的领域操作由 Materials、Dynamics、ML 和 ORCA/xTB 四类 worker 完成。

这种分工允许用户从科学问题开始，而不必自己拼接所有工具。例如"比较一组氧空位附近的 Pd 吸附构型"可能同时需要缺陷结构、吸附位、批量候选、结构审计和快速势能筛选。Experiment 会把主体交给 Materials worker；如果后续目标变成高温迁移，则可以把已经确认的结构交给 Dynamics worker。Worker 之间共享 workspace，但按顺序工作，所以每一步都能读取前一步留下的结构、表格和说明。

## Experiment 如何选择 worker

Experiment 会依据主要对象和交付物选择执行者：

| 主要问题 | 通常负责的 worker |
|---|---|
| 晶体、表面、缺陷、吸附、VASP/CP2K、MLFF 单点或优化、NEB、性质分析 | Materials worker |
| AIMD、LAMMPS、MLFF MD、restart、轨迹健康与扩散 | Dynamics worker |
| 训练数据、MACE 训练与评估、主动学习候选 | ML worker |
| 分子、构象、xTB、CREST、ORCA、TS、IRC、TDDFT、NMR | ORCA/xTB worker |

边界并非只由软件名决定。MLFF 几何优化属于材料筛选时，Materials worker 可以保持整条筛选链；MLFF MD 的重点是轨迹和动力学时，Dynamics worker 更合适；MACE 模型的训练和 benchmark 则属于 ML worker。Experiment 会结合你的目标判断，而不是机械匹配关键词。

四类 worker 还共享项目内的通用工具：`write_todos` 用于维护本轮计划，`ls`、`glob`、`grep` 和 `read_file` 用于检查文件，`write_file` 与 `edit_file` 用于保存产物，`execute` 用于边界清楚的本地脚本和命令。`export_builtin_tool_source` 可以把已注册内置 tool 的实现导出到 workspace 供核对。它们是通用工作能力，不代替下面的领域 tools；在 Review 模式下，`write_file` 和 `edit_file` 会等待审批。

## Materials worker：从晶体到表面、反应路径和性质

Materials worker 是功能最广的计算 worker。它可以从 Materials Project 或 workspace 中的结构开始，建立可靠的体相参考，再继续到表面、缺陷、吸附、反应路径和性质计算。它既能调用确定性的建模工具，也能读取领域 skills 来决定怎样组织候选、保留约束、检查几何和准备后续计算。

### 材料发现与体相基准

如果项目还没有可靠结构，Materials worker 可以按化学式、元素、稳定性或 Materials Project 条件搜索候选，并把选定结构下载为 POSCAR、CIF 或 pymatgen JSON。下载只是起点。它还会检查晶胞、化学组成、对称性和候选来源，必要时建立 conventional cell、超胞或标准化版本。

体相基准贯穿后续工作。表面能、缺陷、能带和吸附比较都依赖一致的体相结构和计算设置，因此 worker 可以先准备 bulk relax 与 static stage，记录对称不等价位点，并在同一参考上继续生成表面或性质任务。

参考 prompt：

```text
使用 Experiment 为 rutile TiO2 建立一个可用于表面研究的体相基准。
请让 Materials worker 先检查 workspace 中是否已有来源可靠的结构；如果没有，
从 Materials Project 搜索并比较候选。自主选择需要的结构与对称性工具，
说明最终选择依据，并保存原始结构、标准化结构和来源记录。

接着准备一致的 VASP bulk relax 与 static 输入，但本轮不提交计算。
所有会影响后续表面能比较的设置都写入 notes/tio2_bulk_reference.md。
```

### Slab、终止面、固定层和表面检查

给定体相结构和 Miller 指数后，`build_slab` 可以生成所有识别到的终止面，并对每个终止面应用统一的面内扩展。Materials worker 会结合 slab skills 判断厚度、真空、是否需要对称 slab、上下表面关系、极性风险和固定层策略。它可以按底部层数、高度区间或明确索引设置固定原子，并在写出新结构时保留继承的 Selective Dynamics。

单纯生成 POSCAR 不是终点。Worker 还可以检查周期边界下的异常短距、孤立碎片、表面配位、悬挂原子和化学计量，并用 VESTA 标准视角或结构图辅助人工审查。如果用户用"最高处的 O"或"配位数为 1 的原子"描述目标，worker 应把它转成可复核的几何或邻接条件，报告阈值和原子索引，而不是凭肉眼猜测。

参考 prompt：

```text
读取 structures/relaxed_ceo2.vasp，为 CeO2(111) 建立后续单原子吸附所需的 slab 集合。
请让 Materials worker 自主组合 slab construction、termination screening 和 visual inspection skills。

需要比较所有合理终止面；真空不少于 15 Å，面内尺寸应能容纳单个 Pd 且避免明显镜像相互作用。
保留输入中的 Selective Dynamics。如果重新定义固定层更合理，先展示候选方案，不要直接覆盖。
逐个检查上下表面、化学计量、配位、CN=1 原子、异常短键和孤立片段，
保存结构图和审计报告。本轮不准备 POTCAR，也不提交远程任务。
```

### 吸附物、吸附位和批量筛选

Worker 可以从 SMILES 或已有分子结构生成吸附物，标准化三维构型，再在 slab 上枚举去重后的 top、bridge、hollow 等代表性位点。`place_adsorbate` 会把一个吸附物放到指定 slab 和位点上，并继承 slab 的 Selective Dynamics；`generate_batch_adsorption_structures` 可以根据位点清单批量生成候选。

Agent 不会只依赖位点标签。Adsorption skills 要求它记录位点来源、吸附物锚点、初始高度、朝向、覆盖度和候选命名，并检查碰撞、跨周期距离和不合理断键。候选很多时，可以先用几何规则和 MLFF 单点或优化做预筛，再为小集合准备一致的 VASP 计算。MLFF 排名是筛选证据，不应被写成 DFT 吸附能。

参考 prompt：

```text
在 structures/ceo2_111_selected.vasp 上研究 CO 的初始吸附构型。
让 Materials worker 先识别对称不等价吸附位，再结合位点环境生成合理的 C-down 和必要的倾斜构型。
不要为了凑数量穷举明显重复或碰撞的结构。

保留 slab 约束，为每个候选记录位点类型、锚点原子、初始距离、朝向和生成来源，
输出结构图与候选清单。可以建议 MLFF 预筛方案，但任何远程运行都要等我确认。
```

### 缺陷、掺杂和位点枚举

Materials worker 可以列出对称不等价位点，生成指定空位、取代原子或显式坐标的间隙原子。`create_vacancy` 和 `substitute_species` 可以针对一个明确位点，也可以为每类对称代表位点建立候选；`insert_interstitial_at_coords` 处理用户给出的间隙位置。

Defect skill 会把"第一轮结构筛选"与"完整缺陷形成能研究"区分开。后者还需要化学势、电荷态、有限尺寸修正、费米能级和一致的体相基准。Agent 可以为这些后续步骤准备结构和计划，但不会把一组中性超胞能量直接包装成完整缺陷热力学。

### VASP、CP2K 与电子结构输入

`vasp_prepare` 可以按 relax、static、frequency、DOS 或 MD 等预设建立规范输入，`vasp_band_prepare` 会从已松弛体相生成带有明确 k-path 来源的能带目录。CP2K 由 `cp2k_prepare` 统一准备单点、固定晶胞优化、晶胞优化、频率和 DOS 类 stage。Worker 会结合输入 skills 检查元素顺序、POTCAR、k 点、泛函、色散、自旋、DFT+U、收敛与约束，而不是只确认四个文件存在。

计算结束后，Materials worker 可以继续处理能带和 DOS、有限位移声子、有限应变弹性、气相或吸附态热力学校正，以及 VASP MD 的扩散分析。VASPKIT、ASE 或项目脚本只是实现路径，报告中会保留使用的输入、单位和假设。

### NEB、Dimer 与反应路径

反应路径从端点质量开始。Worker 可以检查初态和终态的元素、顺序、约束和周期映射，用 `remap_neb_endpoint_atoms` 修正移动原子的对应关系，估计图像数并生成插值路径。`vasp_neb_prepare` 建立 VASP NEB 目录，`vasp_dimer_prepare` 和 mode 生成工具可以从 NEB 邻近图像或 MACE 频率构造 dimer 初始方向。

相关 skills 覆盖从普通 NEB 到 CI-NEB、频率或 dimer 精修，以及计算后的能垒提取和路径 QC。Agent 会检查图像是否跳胞、发生碰撞或出现不连续重排，也会提醒用户 NEB 收敛不等于过渡态已经通过频率或 IRC 验证。

参考 prompt：

```text
使用 structures/initial.vasp 和 structures/final.vasp 建立一个 VASP NEB 路径。
让 Materials worker 先验证端点是否是同一体系、原子映射和 Selective Dynamics 是否一致，
必要时修正移动原子的顺序，再根据最大位移建议图像数。

生成并可视化插值路径，检查跨胞跳跃、原子碰撞和不连续变化。
只有端点与路径 QC 通过后才准备 NEB stage。本轮不要提交；请在报告中说明
plain NEB、CI-NEB 和后续 TS 验证的建议顺序。
```

### MLFF 快速筛选与几何优化

Materials worker 可以查询当前部署启用的 MACE、FairChem UMA、MatterSim 或 ORB-v3 backend，并通过 `mlff_sp`、`mlff_relax` 或 `mlff_neb` 完成单点、批量优化和固定图像路径优化。Agent 会先读取 task schema，再根据结构类型、模型覆盖、精度要求和成本选择是否适合 MLFF。

对于候选表面、吸附构型和反应路径，MLFF 常用于发现明显不稳定或几何错误的候选，减少后续 DFT 数量。训练域外元素、异常配位、强磁性、带电体系和断键过程需要额外谨慎；worker 应报告模型与设置，并建议独立验证。

<details>
<summary>Materials worker 当前 tools 与 skills</summary>

结构与材料 tools：`mp_search_materials`、`mp_download_structure`、`supercell`、`enumerate_unique_sites`、`build_slab`、`fix_atoms_by_layers`、`fix_atoms_by_height`、`fix_atoms_by_indices`、`create_vacancy`、`substitute_species`、`insert_interstitial_at_coords`、`identify_structure_fragments` 和 `render_vesta_views`。

吸附与路径 tools：`create_molecule_from_smiles`、`enumerate_adsorption_sites`、`place_adsorbate`、`generate_batch_adsorption_structures`、`estimate_neb_image_count`、`remap_neb_endpoint_atoms`、`make_neb_geometry`、`vasp_neb_prepare`、`vasp_dimer_prepare`、`make_dimer_mode_from_neb`、`make_dimer_mode_from_mace` 和 `analyze_vasp_neb_results`。

计算准备与性质 tools：`vasp_prepare`、`vasp_band_prepare`、`cp2k_prepare`、`generate_kpath`、`generate_phonon_displacements`、`generate_strained_structures`、`mace_analyze_frequencies`、`analyze_trajectory`、`vaspkit_adsorbate_thermo_correction` 和 `vaspkit_gas_thermo_correction`。

可视化与实现核对 tools：`generate_nanobanana_figure` 可生成需要人工核对的概念图草稿，`export_builtin_tool_source` 可导出注册 tool 的源码。定量结构图仍应使用结构渲染或数据绘图，不用生成图像代替。

执行 tools：`get_avail_remote_task`、`get_remote_task_spec`、`get_avail_resources`、`remote_submission` 和 `remote_submission_batch`。

当前领域 skills 是 `materials-discovery-and-bulk-selection`、`bulk-relax-and-reference`、`slab-construction-and-surface-modeling`、`surface-and-termination-screening`、`adsorbate-and-intermediate-generation`、`adsorption-site-screening`、`adsorption-screening`、`defect-and-dopant-screening`、`vasp-input-preparation`、`vasp-batch-execution`、`cp2k-dft-preparation`、`cp2k-electronic-properties`、`cp2k-vibrational-analysis`、`cp2k-pathway-calculations`、`mlff-screening-and-relaxation`、`mlff-path-optimization`、`neb-prepare`、`neb-calculation`、`neb-analysis`、`band-and-dos-analysis`、`phonon-displacement-workflow`、`elastic-property-workup`、`md-diffusion-analysis`、`thermo-free-energy-and-reporting`、`structure-visual-inspection` 和 `literature-grounding`。

</details>

## Dynamics worker：原子动力学与轨迹

Dynamics worker 关注体系随时间怎样演化。它可以准备 CP2K AIMD、LAMMPS 和受管 MLFF MD，处理已有计算的 restart，并在计算后检查轨迹是否健康。它不会看到一条轨迹就立即拟合扩散系数，而是先判断温度、能量、体积、时间步、采样长度、异常力、原子重叠、键断裂、逸出和轨迹连续性。

### CP2K AIMD

`cp2k_aimd_prepare` 用于建立 AIMD stage，支持从新结构或已有结果继续。相关 skills 要求记录 ensemble、温度、压力、时间步、thermostat/barostat、速度来源、随机状态和 restart 链。`cp2k_output_summary` 可以在运行后提取一般健康信息；更具体的性质分析会根据目标使用轨迹工具或项目脚本。

### LAMMPS

Worker 可以先用 `lammps_forcefield_validate` 检查势文件和元素映射，再用 `lammps_prepare` 建立 minimization、NVE、NVT、NPT、annealing 或 restart stage。`lammps_log_summary` 用于读取日志和热力学量。LAMMPS skills 特别关注 units、atom style、质量、边界、neighbor、势函数适用范围和 restart 文件，避免脚本能启动但物理设置错误。

### MLFF MD 与轨迹分析

如果部署启用了相应 backend，Dynamics worker 可以运行 `mlff_md`，并保持 restart 安全和轨迹连续。MACE、UMA、MatterSim 或 ORB-v3 的可用模型与参数来自当前 task catalog，不能从旧 prompt 猜测。

`md_trajectory_summary` 和 `analyze_trajectory` 可以生成通用时间序列、MSD、扩散拟合、RDF 等产物。Agent 会根据体系选择移动原子组、平衡段、拟合窗口和维度，并把这些选择写入结果，而不是只报一个扩散系数。

参考 prompt：

```text
让 Dynamics worker 检查 calculations/mlff_md_1073K/，判断这条轨迹能否用于 Pd 迁移分析。
先读取输入、日志、restart 和轨迹，不要重新运行。

请检查温度与总能量、时间连续性、帧数、Pd 团簇连通性、异常短距、原子逸出和 restart 来源。
只有健康检查通过后，再选择合理的平衡段和移动原子集合计算 MSD、RDF 与扩散拟合。
把每个选择、单位和不确定性写入 analysis/md_quality_and_diffusion.md。
```

<details>
<summary>Dynamics worker 当前 tools 与 skills</summary>

Tools 包括 `cp2k_aimd_prepare`、`cp2k_output_summary`、`lammps_forcefield_validate`、`lammps_prepare`、`lammps_log_summary`、`md_trajectory_summary`、`analyze_trajectory`、`export_builtin_tool_source`，以及远程 task 查询和提交 tools。

当前领域 skills 是 `cp2k-aimd-preparation`、`cp2k-aimd-restart`、`cp2k-run-analysis`、`lammps-preparation`、`lammps-minimization`、`lammps-md-execution`、`lammps-restart`、`mlff-md-sampling` 和 `trajectory-analysis`。共享的 `remote-stage-layouts` 与 `dpdispatcher-remote-receipts` 负责远程 stage 合同和 receipt 恢复。

</details>

## ML worker：数据集、训练与主动学习

ML worker 负责机器学习势的数据和模型生命周期。它可以从 VASP 结果树提取能量、力和应力，建立带有固定 train/valid/test 划分的 extxyz 数据集；也可以把数据与参数组织成远程 MACE 训练或评估 stage，并在独立测试集上分析误差。

数据整理不是简单合并文件。Worker 会检查元素覆盖、单位、参考能、重复构型、异常值、标签完整性和数据泄漏。训练时会保存配置、随机种子、模型来源、checkpoint、日志和测试结果。主动学习中，`calculate_al_candidates` 可以根据多样性和可选的 committee disagreement 对候选结构排序，但最终选择还要考虑目标体系覆盖和标注成本。

参考 prompt：

```text
让 ML worker 从 calculations/reference_vasp/ 建立一个用于 MACE 微调的数据集。
先审计哪些 run 真正收敛并包含可用的能量、力和应力，再决定纳入范围。
统一单位与标签，检查重复结构和不同计算设置混用的问题，固定随机种子并写出
train/valid/test 划分清单。不要在审计通过前启动训练。

数据集写到 ml/datasets/pd_ceo2_v1/，同时生成一份说明数据来源、排除项、
元素与构型覆盖、潜在泄漏和适用范围的报告。
```

<details>
<summary>ML worker 当前 tools 与 skills</summary>

Tools 包括 `build_dataset_from_runs`、`calculate_al_candidates`、`export_builtin_tool_source`，以及远程 catalog、resource 和 submission tools。主要 skills 是 `mace-dataset-curation`、`mace-finetuning-and-benchmark` 和 `active-learning-relabel-loop`。

当前受管远程 tasks 包括 `mace_train` 和 `mace_eval`。模型训练依赖已配置的 GPU resource 和 MACE 环境，不会在 control plane 上静默运行。

</details>

## ORCA/xTB worker：分子与量子化学

ORCA/xTB worker 处理非周期分子、配合物和有限簇。它可以从 SMILES 生成三维结构，枚举构象并去重，先用 CREST 或 xTB 做低成本搜索和预优化，再为选定构象准备 ORCA 优化、频率、热化学、TDDFT 或 NMR。

对于反应路径，它可以准备 relaxed scan，从扫描峰附近提取 TS guess 并执行 OptTS；也可以在明确的反应物和产物之间准备 NEB-TS，再用 IRC 检查连接关系。柔性分子的 NMR 任务可以串联构象生成、xTB 清理、ORCA NMR 和后续 Boltzmann 汇总所需的证据。

Agent 会要求明确总电荷和自旋多重度，并区分"未配对电子数"与"多重度"。构象排名会记录方法、溶剂和能量窗口；频率结果会用于判断极小值或过渡态，而不是只把优化终止当作成功。

参考 prompt：

```text
让 ORCA/xTB worker 从分子 SMILES 建立一个用于 ORCA 热化学的构象集合。
总电荷为 0，自旋多重度为 1，溶剂按 acetonitrile 处理。

请自主选择构象生成、CREST/xTB 预筛和去重策略，保留构象来源和相对能。
在准备 ORCA opt+freq 前先给出候选数量、能量窗口、几何重复检查和计算成本建议。
本轮只做到可审阅的 ORCA stage，不提交远程任务。
```

<details>
<summary>ORCA/xTB worker 当前 tools 与 skills</summary>

分子与构象 tools：`create_molecule_from_smiles`、`enumerate_molecular_conformers`、`filter_conformer_ensemble`、`extract_optimized_molecules` 和 `identify_structure_fragments`。

ORCA 准备与分析 tools：`orca_prepare`、`orca_scan_prepare`、`orca_optts_prepare`、`orca_nebts_prepare`、`orca_irc_prepare`、`analyze_orca_results` 和 `analyze_xtb_results`。远程执行使用 catalog、resource 和 submission tools；`export_builtin_tool_source` 用于实现核对。

Skills 包括 `conformer-search-and-preopt`、`xtb-screen-and-prune`、`mlff-molecular-screening`、`orca-optfreq-thermochemistry`、`scan-to-ts`、`nebts-and-irc` 和 `nmr-ensemble-workup`。

</details>

## 跨 worker 的任务怎样继续

一个研究目标可以跨 worker，但每次交接都应留下清楚的产物。例如 Materials worker 先建立吸附候选并用 MLFF 缩小集合，后续可以把选定结构交给 Dynamics worker 做高温稳定性；Dynamics 发现的异常构型可以交给 ML worker 作为主动学习候选；ORCA/xTB worker 得到的气相分子热化学也可以与 Materials worker 的表面频率校正一起进入自由能分析。

Experiment 会按顺序安排这些工作。用户不需要在 prompt 中手工指定每一次委派，但可以明确主要目标、允许的近似和暂停点。下一章不再按 worker 列表，而是从几条完整的建模能力链出发，说明 CatMaster 可以把工作推进到哪里，以及每一阶段应留下哪些证据。
