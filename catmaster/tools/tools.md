## CatMaster Tools (Stage-oriented, under `catmaster/tools/`)

本表按阶段列出工具：名称、位置、功能、输入、输出、推荐设备与关键参数说明。

| 阶段 | 名称 | 位置(模块:函数) | 功能 | 主要输入 | 主要输出 | 推荐设备 | 关键参数说明 |
|---|---|---|---|---|---|---|---|
| Retrieval | lit_search | `catmaster.tools.retrieval:lit_search` | 学术搜索抓取标题/作者/摘要 | `query`, `site?`, `max_results`, `chromedriver_path?`, `headless`, `timeout` | 列表[EvidenceAnchor字典] | local | `headless`: 是否无头; `timeout`: 页面加载超时秒 |
| Retrieval | lit_capture | `catmaster.tools.retrieval:lit_capture` | 抓取指定网页的可读内容 | `target_url`, `wait_seconds`, `chromedriver_path?`, `headless`, `timeout` | EvidenceAnchor字典 | local | `wait_seconds`: 打开后等待时间 |
| Retrieval | matdb_query | `catmaster.tools.retrieval:matdb_query` | Materials Project 查询与结构下载 | `criteria`, `properties?`, `structures_dir`, `api_key?` | `{count,hits_path,structures,provider,api_version}` | local | `criteria`: 支持 `material_ids|formula|chemsys|elements` |
| Geometry & Inputs | slab_cut | `catmaster.tools.geometry_inputs:slab_cut` | 体相切割晶面并生成带SD的POSCAR | `structure_file`, `miller_list`, `min_slab_size`, `min_vacuum_size`, `relax_thickness`, `output_root`, `get_symmetry_slab`, `fix_bottom` | `{compound,facets,output_root,generated[]}` | local | `miller_list`: [[h,k,l],...]; `relax_thickness`: 选择性弛豫厚度 |
| Geometry & Inputs | slab_fix | `catmaster.tools.geometry_inputs:slab_fix` | 批量/单个修复晶面，添加SD/居中 | `input_path(file/dir)`, `output_dir`, `relax_thickness`, `fix_bottom`, `centralize` | `{source,output_dir,generated[]}` | local | `centralize`: 是否沿c轴中心化 |
| Geometry & Inputs | adsorbate_placements | `catmaster.tools.geometry_inputs:adsorbate_placements` | 生成吸附初始放置候选集合 | `config` | `{schema_version,output_dir,num_candidates,candidate_paths[]}` | local | `config` 直接透传到生成器脚本 |
| Geometry & Inputs | gas_fill | `catmaster.tools.geometry_inputs:gas_fill` | 在真空层填充H2分子（理想气体估算） | `input`, `output?`, `temperature`, `pressure_mpa`, `buffer_A`, `bond_A`, `replicate?`, `targetN?`, `max_rep`, `rounding`, `min_molecules`, `seed`, `summary_json?` | `{output,placed_molecules,placed_H_atoms,expected_molecules,gas_volume_A3,summary}` | CPU | `rounding`: round/ceil/floor/poisson；`replicate` 与 `targetN` 二选一 |
| Geometry & Inputs | vasp_prepare | `catmaster.tools.geometry_inputs:vasp_prepare` | 规范化生成 VASP 输入集 | `input_path(file/dir)`, `output_root`, `calc_type`, `k_product`, `calibration_temperature?`, `run_yaml_template?`, `user_incar_settings?` | `{calc_type,k_product,structures_processed,outputs[]}` | local | `calc_type`: gas/bulk/slab；`k_product`: k点密度乘积 |
| Execution | mace_relax | `catmaster.tools.execution:mace_relax` | 基于 MACE 的几何弛豫（批量预筛） | `structure_file`, `work_dir`, `device`, `use_d3`, `fmax`, `maxsteps`, `optimizer`, `model`, `mode`, `constraint_entities`, `bond_rt_scale`, `hooke_k`, `amp_mode`, `amp_dtype`, `tf32` | `{converged,final_energy,relaxed_vasp,relaxed_traj,output_dir,...}` | GPU | `device`: auto/cpu/cuda; `amp_mode`: on/auto/off；`fmax`: 收敛阈值 |
| Execution | vasp_execute | `catmaster.tools.execution:vasp_execute` | 提交 VASP 到 CPU worker（含 SFTP 上传） | `input_dir`, `project`, `worker`, `remote_tmp_base`, `vasp_command?`, `env?`, `wait`, `timeout_s`, `poll_s` | `{job_db_id,job_uuid,remote_staging,output?}` | CPU | 先在本地准备 VASP 输入；远端 pre_run 仅做拷贝与清理 |
| Analysis | vasp_summarize | `catmaster.tools.analysis:vasp_summarize` | 从 VASP 目录提取结构化结果 | `work_dir` | 汇总字典（能量/收敛/最终结构等） | CPU | `work_dir` 需包含 `vasprun.xml` |

说明：
- 所有路径参数均支持相对/绝对路径；输出路径如不存在会自动创建上游目录。
- 返回的“字典/列表”均为可 JSON 序列化的数据结构，便于持久化或被上层代理消费。
