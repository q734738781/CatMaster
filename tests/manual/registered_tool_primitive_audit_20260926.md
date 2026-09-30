# 注册工具 primitive 设计审查

2026-09-26，基于当前本地源码。

上一轮审查不足以回答“这些工具是否符合 primitive 的设计初衷”。它主要确认了注册/schema、已有回归及 test2 已出现的故障，对科学参数能否表达、完整输入输出能否到达、批量身份是否保留、重复执行是否可靠，检查得不够。尤其不能因为准备工具支持原生输入，就顺带判定对应分析工具或批量路径也合格。本报告按这些问题重新逐项判断。

审查范围是 `ToolRegistry` 的 **101 个正式注册名称，涉及 39 个实现文件**。注册别名不重复计数。检查包括最终 OpenAI/LangChain schema、注册函数及相关辅助实现、模型可见返回、specialist/worker 绑定。本次没有修改普通工具实现，也没有触发公共 demo、远程计算、文献下载或付费模型请求。

## 判断标准

普通工具应把高频、重复、边界明确的小操作做可靠，让 agent 能自由组合。逐项按下列四个维度检查，表中顺序固定为 **P / F / R / H**：

| 维度 | 本次实际检查的问题 |
| --- | --- |
| P：操作边界 | 是否是可组合的小操作；是否夹带方法选择、科学判断或下一步执行；较大接口是否确有授权/事务/执行边界 |
| F：灵活性 | 有意义的参数、原生输入、选择条件和目的路径能否表达；默认是否变成不可替换的科学选择 |
| R：可达性及结果真实性 | 有效输入、帧、属性和完整结果是否可达；是否误选文件/索引，是否吞错或把部分结果说成完整成功 |
| H：高通量与组合 | 输入输出是否一一对应；独立调用是否互相覆盖；批次失败、重跑和大数据处理是否可靠 |

符号：✓ 表示在声明范围内基本满足；△ 表示存在限制或接口摩擦；✗ 表示已有关键缺陷或明显违背要求；N/A 表示该工具属于有理由的系统/事务/模型接口，不以“小本地函数”衡量 P。

“基本符合”是本次源码审查中的设计判断，不是无缺陷证明，也不是远端全流程验收。“部分符合”表示能力主体有用，但需要补齐选择、恢复、说明或可达性。“不符合”表示当前至少一条关键路径会给错结果、丢输入输出，或其设计不能承担宣称的通用/批量操作。没有 batch 参数本身不扣分；显式路径的单项调用可以可靠组合。POSCAR、OUTCAR 等格式约定与任意猜测 job 文件名分别判断。

## 总体结果

| 判断 | 数量 | 含义 |
| --- | ---: | --- |
| 基本符合 | 33 | 在声明范围内是可组合能力；不等于远端验收 |
| 部分符合 | 52 | 需要补控制、来源选择、恢复或接口说明 |
| 不符合 | 16 | 关键路径有错误、丢失或明显的设计问题 |

普通 primitive 共 51 个，其中 基本符合 13 个、部分符合 23 个、不符合 15 个。其余工具按查询、计算/模型服务、授权获取、工作区状态或 Research Graph 事务分别评价；没有用 Graph 字段问题代替普通科学工具的审查。

## 优先处理的具体问题

### 1. 批次“处理成功”不等于每个输入都有独立结果

`orca_prepare` 把相对路径清洗成 stage 名：`a/b.xyz` 与 `a_b.xyz` 都落到 `a_b/`。本地给这两个文件不同 H₂ 键长，工具报告 `prepared_count=2`，两个清单记录却指向同一 stage。

`supercell`、`build_slab` 和三个 `fix_atoms_*` 工具去掉扩展名并展平路径。输入 `same.cif` 与 `same.vasp`，分别使用 5 Å 和 4 Å 晶格，工具都报告两项成功、零错误，却把输出写到同一个目标。批量吸附工具存在同样的路径映射，后者本次是静态确认，未单独运行批量吸附样例。

这类问题需要修输出身份与冲突处理，单纯说明“换个文件名”会把工具缺陷转嫁给 agent。证据：R04、R05、R16；[ORCA stage 名](../../catmaster/tools/geometry_inputs/orca_prepare.py#L174)、[结构批量名](../../catmaster/tools/geometry_inputs/crystal_tool.py#L53)、[slab 批量名](../../catmaster/tools/geometry_inputs/slab_tools.py#L176)、[批量吸附目录](../../catmaster/tools/geometry_inputs/adsorbate_tool.py#L774)。

### 2. 有些解析器会改变或漏掉科学结果

- 构象筛选把绝对能量 100、102 kcal/mol 当成相对能量，5 kcal/mol 窗口错误地保留 0 个。
- 优化结构提取漏掉 `job.runtime.xyz`，跳过传入的单个结果根目录；`job_trj.xyz` 有 0.7 Å、1.4 Å 两帧时取到了第一帧。
- CP2K 目录存在调度日志 `a_scheduler.out` 和真实输出 `z_science.out` 时，摘要器静默选调度日志。
- XYZ 末帧截断时，轨迹摘要仍返回 completed，把前一个完整帧称为 final frame，没有部分/损坏状态。

这些是已经复现的本地逻辑问题。普通 ORCA、自定义 LAMMPS log 和未带 CatMaster summary 的 xTB 输出不可被发现，也已复现，但它们与“读了错误帧”不同：前者首先是输入范围不足，应提供显式源文件入口，或将工具的声明准确收窄。证据：R02、R03、R06、R07、R08；[构象处理](../../catmaster/tools/geometry_inputs/molecular_qchem.py#L277)、[CP2K 来源选择](../../catmaster/tools/dynamics/cp2k_analysis.py#L84)、[XYZ 计数与末帧](../../catmaster/tools/dynamics/lammps_tools.py#L354)、[量化结果发现](../../catmaster/tools/analysis/qchem_analysis.py#L129)。

### 3. 方法切换和坐标变换没有完整交代

声子位移工具捕获所有 phonopy 异常，然后换成手工算法；生成结果里可查到 `manual_fallback`，但原始失败原因丢失，输入也没有选择算法的字段。更严重的是手工路径扩胞后沿用原胞的代表原子索引。Na/Cl 原胞扩成 `[2,1,1]` 后顺序为 Na、Na、Cl、Cl，原胞索引 1 对应 Cl，扩胞索引 1 却对应 Na。实测两个代表位点的位移全作用于 Na，没有移动 Cl。证据：R09、R10；[手工位移](../../catmaster/tools/geometry_inputs/crystal_tool.py#L175)、[自动回退](../../catmaster/tools/geometry_inputs/crystal_tool.py#L1068)。

`generate_kpath` 默认在标准原胞上生成 KPOINTS，却没有输出该结构或变换关系。调用方得到的原输入文件不能保证与输出路径使用同一基底。这是静态确认的衔接缺口，本次没有开展能带计算验证误差。[实现](../../catmaster/tools/geometry_inputs/crystal_tool.py#L995)

MP 下载强制把结构转成 conventional cell，实际 schema 没有说明或选择开关。模拟提供方返回一个 Fe 原胞，真实本地变换保存了两个原子，但返回的 natoms 还是 1。证据：R11；[下载实现](../../catmaster/tools/retrieval/matdb.py#L321)。

VASPKIT 热化学工具在程序不可用时改走 ASE，虽然结果会说明 backend/近似，入口却没有提前说明或允许选择。MACE dimer 工具声明差分，内部传的是 auto。它们与 phonopy 原子索引错误的严重程度不同，但共同说明：默认值、自动回退和科学方法选择需要明确界面。

### 4. 有原生参数入口，也不代表所有关键控制都已开放

CP2K、xTB、CREST、LAMMPS 准备工具保留完整输入文件或 argv，当前方向符合要求。VASP 几个准备工具的 INCAR 覆盖也有用，但共用 writer 固定 `PBE_54`，通用网格只由 `k_product` 生成。POTCAR 选择、KPOINTS 选择与 INCAR 是不同控制面；不能用“agent 之后可以自己改文件”证明准备接口已经完整。band 工具已有显式 KPOINTS 输入，因此单独判断。[共用 writer](../../catmaster/tools/geometry_inputs/vasp_inputs.py#L176)

`calculate_al_candidates` 固定特征、权重组合与贪心规则，构建 `N×N×特征维` 距离临时数组，还只导出选中项。对大候选池，这既是控制不足，也是明确的规模问题；本次未用大数据制造内存耗尽。[特征与选择算法](../../catmaster/tools/machine_learning/mace_ml.py#L136)

### 5. 注册、artifact 和模型能看到的内容是三件事

当前 101 个注册名称中，`apply_aider_edits`、`peer_review_pdf_manuscript`、`mace_analyze_frequencies`、`open_public_page`、`find_in_page` 未在 active specialist/worker 中绑定。前两项有兼容保留性质，不能直接称为运行故障；后面几项则需要明确实际能力由谁提供。[角色工具集合](../../catmaster/specialists/runtime.py#L186)、[搜索装配](../../catmaster/runtime/search_surface.py#L16)

Graph 新增 E/H/R 的若干返回正文只有标题和 revision，ID 放在 artifact。普通小返回不会因“artifact 里存在”自动呈现给模型；agent 仍能 SQL 查询找回，但这增加了不必要的下一步查询。重复建 E 还涉及每次调用生成新 UUID、无幂等创建语义，不能把责任全部归到 agent。应明确新增、query/focus 复用和原位更新的分工，并直接返回新句柄。证据：R12；[共用 mutation 返回](../../catmaster/tools/misc/research_graph.py#L425)、[新增 E 服务](../../catmaster/research/knowledge_graph/service.py#L1578)、[输出适配](../../catmaster/runtime/tool_output_adapter.py#L232)。

最终 schema 两条路径除去重复的顶层 description 后一致。14 个 Graph 工具仍有 51 个顶层字段缺 description。合法 schema 和可加载工具不能抵消这些语义缺口。另有两个已复现的小接口问题：网页查找的 Unicode casefold 坐标错位（R14），以及 acquisition 缓存忽略新传入的 expected_title（R15）。

## 全部 101 个工具的逐项判断

每行源码链接指向注册函数；上面的重点问题链接指向相关辅助实现。带 R 编号的是下面离线探针的直接观察，或明确标注的模拟依赖结果；其余结论为静态源码判断。表中 H 的通过意味着可用独立路径/请求组合，不表示进行过压力测试。


### 分子准备与分析

| # | 工具与实现 | 类型 | 判断 | P / F / R / H | 依据与处理 |
| ---: | --- | --- | --- | --- | --- |
| 1 | [create_molecule_from_smiles](../../catmaster/tools/geometry_inputs/molecule.py#L65) | primitive | 部分符合 | ✓ / △ / △ / △ | R01：name 声称控制文件名，实际未使用；输出只由 output_path 决定。3D 生成隐藏固定 MMFF/seed，重复目标覆盖；应修正无效参数并说明生成方法与覆盖行为。 |
| 2 | [enumerate_molecular_conformers](../../catmaster/tools/geometry_inputs/molecular_qchem.py#L166) | primitive | 不符合 | ✓ / ✓ / ✗ / △ | R13：指定 MMFF 而取不到力场时仍返回 completed，没有逐构象优化状态；Minimize 返回值也未检查。重复输出目录留下旧构象。保留现有数量、种子、优化方法控制，补真实结果状态。 |
| 3 | [filter_conformer_ensemble](../../catmaster/tools/geometry_inputs/molecular_qchem.py#L276) | primitive | 不符合 | ✓ / △ / ✗ / △ | R02：100/102 kcal/mol 被直接当相对能量，5 kcal/mol 窗口保留 0 个；摘要解析失败被吞掉，缺能量仍继续。多帧文件只读第 0 帧，读取失败静默跳过。需要正确能量基准、帧选择和逐项失败信息。 |
| 4 | [extract_optimized_molecules](../../catmaster/tools/geometry_inputs/molecular_qchem.py#L444) | primitive | 不符合 | ✓ / ✗ / ✗ / △ | R03：漏 job.runtime.xyz、跳过根目录、双帧 job_trj.xyz 取首帧；结束标志代替优化收敛，跳过原因不可见。应支持明确源文件/最终帧和实际托管布局，不能靠 skill 教 agent 猜名字。 |
| 5 | [orca_prepare](../../catmaster/tools/geometry_inputs/orca_prepare.py#L211) | primitive | 不符合 | ✓ / ✓ / ✗ / ✗ | R05：a/b.xyz 与 a_b.xyz 映射同一 stage，prepared_count=2 实际只有 1 个。原生关键词/blocks 与显式电荷、自旋方向正确；修复批量身份后再评估，另需说明每文件单结构和辅助文件边界。 |
| 6 | [orca_nebts_prepare](../../catmaster/tools/geometry_inputs/orca_prepare.py#L263) | primitive | 基本符合 | ✓ / ✓ / ✓ / ✓ | 显式两个端点、电荷/自旋、原生关键词/blocks，检查原子数与元素顺序，拒绝已有 stage，返回输入及清单。job.inp/端点名是生成的执行契约；原子对应关系仍由调用方提供，单任务可独立组合。 |
| 7 | [xtb_prepare](../../catmaster/tools/geometry_inputs/xtb_prepare.py#L57) | primitive | 基本符合 | ✓ / ✓ / ✓ / ✓ | argv 按 token 原样保留，asset_mappings 可保持子目录和点文件，拒绝既有 stage，返回 manifest 路径；不替 agent 选科学方法。现有回归覆盖原生参数传递及独立输出。 |
| 8 | [crest_prepare](../../catmaster/tools/geometry_inputs/crest_prepare.py#L57) | primitive | 基本符合 | ✓ / ✓ / ✓ / ✓ | 与 xTB 同样保留原生 argv、TOML 和辅助文件，输出 stage 独立；无需每个方法另造一个封闭工具。方法语法正确性由原生引擎检查，工具应保持该边界。 |
| 9 | [analyze_xtb_results](../../catmaster/tools/analysis/qchem_analysis.py#L520) | primitive | 部分符合 | ✓ / △ / △ / △ | R08：仅凭 xtb_summary.json/crest_summary.json 发现目录，普通原生输出不可直接解析；没有显式 log/结构输入，root 命中后不遍历子目录。状态分层和完整 JSON 是优点；_last_xyz_frame 还会写入源目录，需明确或移至输出目录。 |
| 10 | [analyze_orca_results](../../catmaster/tools/analysis/qchem_analysis.py#L546) | primitive | 部分符合 | ✓ / △ / △ / △ | R08：发现条件是 job.out 或 orca_summary.json，任意命名 ORCA 输出不可达；fallback 对多个 .out 取排序末尾，无消歧入口。能区分进程/SCF/优化和缺属性，且兼容 property JSON；应扩展显式输入并避免把 input.xyz 冒充优化结构。 |

### 动力学输入与摘要

| # | 工具与实现 | 类型 | 判断 | P / F / R / H | 依据与处理 |
| ---: | --- | --- | --- | --- | --- |
| 11 | [cp2k_prepare](../../catmaster/tools/geometry_inputs/cp2k_prepare.py#L49) | primitive | 基本符合 | ✓ / ✓ / ✓ / ✓ | 完整原生输入文件与显式资产映射，原样复制，不注入固定方法；fresh stage 防止误覆盖。重命名为 job.inp 属于明确托管契约，任意源文件名可达，单 stage 可高通量组合。 |
| 12 | [cp2k_output_summary](../../catmaster/tools/dynamics/cp2k_analysis.py#L310) | primitive | 不符合 | ✓ / △ / ✗ / △ | R07：a_scheduler.out 和 z_science.out 共存时静默选前者，遗漏真实 CP2K 能量；无 output_file 选择。批次根出现 .out 又会遮蔽子目录。保留现有状态解析，先修来源选择/消歧。 |
| 13 | [lammps_prepare](../../catmaster/tools/dynamics/lammps_tools.py#L80) | primitive | 基本符合 | ✓ / ✓ / ✓ / ✓ | 完整 native input 和显式资产映射，拒绝已有 stage；不把势函数、系综、观测量封装成隐藏配方。固定 stage 文件名属于生成接口，独立输出目录可并行使用。 |
| 14 | [lammps_log_summary](../../catmaster/tools/dynamics/lammps_tools.py#L285) | primitive | 部分符合 | ✓ / △ / △ / △ | R08：自定义 log anneal.log 不可被发现，缺少显式日志参数；root 命中会停止向下发现。能导出多 thermo 段及进程/任务状态，适合当前托管布局，不能据此声称覆盖一般 LAMMPS 输出。 |
| 15 | [md_trajectory_summary](../../catmaster/tools/dynamics/lammps_tools.py#L431) | primitive | 不符合 | ✓ / ✓ / ✗ / △ | R06：完整首帧加截断末帧仍报 completed，只给首帧计数/结构，没有截断状态。明确文件路径和多候选拒绝很好，但计数/最后帧需要真实读取状态；原生观测表仅识别固定名，长轨迹提取整体读入内存。 |

### 托管执行与目录

| # | 工具与实现 | 类型 | 判断 | P / F / R / H | 依据与处理 |
| ---: | --- | --- | --- | --- | --- |
| 16 | [remote_submission](../../catmaster/tools/execution/remote_submission.py#L1381) | 系统接口 | 基本符合 | N/A / ✓ / ✓ / ✓ | 显式单 stage，原生参数/模板覆盖，隔离 dispatch 工作目录，失败保留 receipt、remote context 和 attempt 输出。阻塞至终态已在说明中写明；这是执行与恢复边界，不应为追求小函数拆掉。未新提交远端任务。 |
| 17 | [remote_submission_batch](../../catmaster/tools/execution/remote_submission.py#L1431) | 系统接口 | 部分符合 | N/A / △ / ✓ / △ | 同 task/config 的一级目录批量及恢复输出明确；但不能直接传选定 stage 列表或异构参数，一项准备错误使整次准备失败，错误提示未必指出完整逐项状态。一级布局本身有契约理由；应完善选择与失败定位，不能误判为所有浅扫描都错。 |
| 18 | [get_avail_remote_task](../../catmaster/tools/execution/remote_submission.py#L1835) | 查询 | 基本符合 | ✓ / ✓ / ✓ / ✓ | 只读列出角色可见任务、默认参数、支持的覆盖与执行绑定，可继续查 task spec；未看到自动选择或启动科学任务。目录条目完整返回，资源细节按需。 |
| 19 | [get_remote_task_spec](../../catmaster/tools/execution/remote_submission.py#L1904) | 查询 | 基本符合 | ✓ / ✓ / ✓ / ✓ | 可指定任务及模板覆盖，compact/full 提供参数表与完整 schema；方法关键覆盖可以被发现并传回提交工具。授权/部署绑定约束有实际边界，保持当前能力接口。 |
| 20 | [get_avail_resources](../../catmaster/tools/execution/remote_submission.py#L1936) | 查询 | 基本符合 | ✓ / ✓ / ✓ / ✓ | 边界明确为 general_execute 的脚本环境目录，返回可选名称和软件描述，不混作机器调度管理接口。无文件写入，结果可直接组成后续调用。 |

### VASP 输入

| # | 工具与实现 | 类型 | 判断 | P / F / R / H | 依据与处理 |
| ---: | --- | --- | --- | --- | --- |
| 21 | [vasp_prepare](../../catmaster/tools/geometry_inputs/vasp_prepare.py#L174) | primitive | 部分符合 | ✓ / △ / ✓ / ✓ | 单文件边界、显式输出和 INCAR force 覆盖可组合；共用 StructWriter 硬编码 PBE_54，KPOINTS 仅 k_product 生成奇数 Gamma 网格，不能原生选择 POTCAR 变体或通用网格。INCAR patch 不能弥补这些独立控制面。 |
| 22 | [vasp_band_prepare](../../catmaster/tools/geometry_inputs/vasp_band_prepare.py#L156) | primitive | 部分符合 | ✓ / △ / ✓ / △ | 原生 line-mode KPOINTS、CHGCAR 可传入，INCAR 可覆盖；仍继承固定 POTCAR 选择，且写出 VASP 输入后才解析/复制部分输入，失败可能留下阻止原位重试的半成品。要清楚说明并保留可恢复输出。 |
| 23 | [vasp_neb_prepare](../../catmaster/tools/geometry_inputs/neb_tools.py#L1640) | primitive | 部分符合 | ✓ / △ / ✓ / △ | 端点/已有 image tree/批次入口、插值控制、force INCAR 覆盖可用；继承固定 POTCAR，批次某项失败会中止且最终 batch_summary 尚未写出。IOPT 便捷字段虽有限枚举，仍可用 force patch 表达其他值，不能误判为彻底不可达。 |
| 24 | [vasp_dimer_prepare](../../catmaster/tools/geometry_inputs/dimer_tools.py#L636) | primitive | 部分符合 | ✓ / △ / ✓ / ✓ | 显式 TS 结构、逐原子方向文件、INCAR 覆盖和输出目录，原子顺序及质量归一化处理可复用；共用 PBE_54/网格限制仍在。单任务本身无需新增 batch 参数。 |

### 晶体与表面操作

| # | 工具与实现 | 类型 | 判断 | P / F / R / H | 依据与处理 |
| ---: | --- | --- | --- | --- | --- |
| 25 | [build_slab](../../catmaster/tools/geometry_inputs/slab_tools.py#L279) | primitive | 不符合 | ✓ / ✓ / ✗ / ✗ | R16：same.cif 与 same.vasp 的 slab_id 相同，两个输入共用输出目录/终止面文件，报告 2 个成功实际 1 套。终止面/厚度/真空等控制有价值，必须修复身份映射；批次清单写失败还被吞掉。 |
| 26 | [fix_atoms_by_layers](../../catmaster/tools/geometry_inputs/slab_tools.py#L591) | primitive | 不符合 | ✓ / ✓ / ✗ / ✗ | R16：两个同 stem 输入写成一个 .vasp，却报告 processed=2、errors=0。层数、容差、反选是合适 primitive 控制；批量命名和可见失败必须修复。 |
| 27 | [fix_atoms_by_height](../../catmaster/tools/geometry_inputs/slab_tools.py#L784) | primitive | 不符合 | ✓ / ✓ / ✗ / ✗ | R16：同 stem 批量覆盖，统计仍报两项成功。显式 Cartesian z 区间和反选保留灵活性；修复批次身份，并在说明里保留坐标方向/centralize 的实际含义。 |
| 28 | [fix_atoms_by_indices](../../catmaster/tools/geometry_inputs/slab_tools.py#L983) | primitive | 不符合 | ✓ / ✓ / ✗ / ✗ | R16：明确索引的单结构操作合理，但批量同 stem 仍覆盖且虚报成功数。索引基准已明确，不需要新工具语言；需要无碰撞输出及真实逐项结果。 |
| 29 | [supercell](../../catmaster/tools/geometry_inputs/crystal_tool.py#L435) | primitive | 不符合 | ✓ / △ / ✗ / ✗ | R04：same.cif/same.vasp 两项成功对应同一输出。另只有三个对角复制数，不能表达一般整数变换矩阵；应分开处理核心批量缺陷和能力扩展。 |
| 30 | [enumerate_unique_sites](../../catmaster/tools/geometry_inputs/crystal_tool.py#L579) | primitive | 基本符合 | ✓ / ✓ / ✓ / ✓ | 显式结构与对称容差，全部等价组/索引存到可达 JSON，可自定输出；不替 agent 选择缺陷。当前单结构查询范围清楚，独立调用可组合。 |
| 31 | [create_vacancy](../../catmaster/tools/geometry_inputs/crystal_tool.py#L632) | primitive | 部分符合 | ✓ / ✓ / ✓ / △ | 显式原子/等价组或每组一个缺陷，保留全量批次 JSON，操作边界合理；既有输出目录直接复用，重跑组数改变时旧结构残留，无覆盖/清理语义。应完善重跑行为，而不是强制每次只生成一个。 |
| 32 | [substitute_species](../../catmaster/tools/geometry_inputs/crystal_tool.py#L729) | primitive | 部分符合 | ✓ / ✓ / ✓ / △ | 替换目标和新元素明确，单项/按等价组枚举可组合，输出与源结构有对应；复用目录会覆盖并遗留旧成员。需要明确重跑及部分写入行为，保持现有显式选择。 |
| 33 | [insert_interstitial_at_coords](../../catmaster/tools/geometry_inputs/crystal_tool.py#L840) | primitive | 部分符合 | ✓ / △ / ✓ / △ | 显式笛卡尔/分数坐标和输出可用；多个 coords 实际是多个独立单插入候选，未明确为“同一结构多原子”还是枚举，重跑亦残留旧文件。应先明确参数语义，勿让 agent 猜。 |
| 34 | [generate_strained_structures](../../catmaster/tools/geometry_inputs/crystal_tool.py#L926) | primitive | 部分符合 | ✓ / ✓ / ✓ / △ | 接受完整变形矩阵和 mode/value 网格，文件含序号因此数值标签舍入不会直接合并同次候选；全矩阵可读。复用输出目录会保留旧批次成员，应明确重跑边界。 |
| 35 | [generate_kpath](../../catmaster/tools/geometry_inputs/crystal_tool.py#L995) | primitive | 不符合 | ✓ / ✓ / ✗ / ✓ | 默认先变成 primitive standard cell，再输出该基底的 KPOINTS；未输出对应标准结构或基底变换，仅记原输入路径。下游直接配原胞/常规胞可能错配。需要让与路径配套的结构可达，不是补一句提醒就够。 |
| 36 | [generate_phonon_displacements](../../catmaster/tools/geometry_inputs/crystal_tool.py#L1052) | primitive | 不符合 | △ / ✗ / ✗ / △ | R09/R10：任意 phonopy 异常触发手工算法，原错误丢失；扩胞后用原胞代表索引，Na/Cl 样例只移动 Na。phonopy 路径也未传 schema 中的 symprec/angle_tolerance。算法、参数和输出须一致。 |

### 吸附结构

| # | 工具与实现 | 类型 | 判断 | P / F / R / H | 依据与处理 |
| ---: | --- | --- | --- | --- | --- |
| 37 | [enumerate_adsorption_sites](../../catmaster/tools/geometry_inputs/adsorbate_tool.py#L520) | primitive | 基本符合 | ✓ / ✓ / ✓ / ✓ | 描述明确是 ASF 去重代表位点，site kind/高度可控，全部结果存 JSON，没有截断；返回默认建议但不执行选择。适合该窄范围，不能把它当任意取向/覆盖度枚举器。 |
| 38 | [place_adsorbate](../../catmaster/tools/geometry_inputs/adsorbate_tool.py#L588) | primitive | 部分符合 | △ / ✓ / ✓ / △ | 显式位点/坐标与保持分子朝向的约定清楚；即使给坐标仍先做 ASF 枚举，额外引入不必要失败。不同输出文件还共同读改写父目录 ads_indices.json，无并发协调，存在丢索引风险；各自 sidecar 仍保留。 |
| 39 | [generate_batch_adsorption_structures](../../catmaster/tools/geometry_inputs/adsorbate_tool.py#L704) | primitive | 不符合 | ✓ / △ / ✗ / ✗ | 静态：slab_dir 同 stem/展平路径映射同一目录，后者覆盖前者；max_structures 实际对每 slab 生效而描述像总上限。完整 sites manifest/offset 的设计应保留，但批次身份和并发索引需修。 |

### 过渡态几何与振动

| # | 工具与实现 | 类型 | 判断 | P / F / R / H | 依据与处理 |
| ---: | --- | --- | --- | --- | --- |
| 40 | [estimate_neb_image_count](../../catmaster/tools/geometry_inputs/neb_tools.py#L625) | primitive | 基本符合 | ✓ / ✓ / ✓ / ✓ | 端点、MIC、目标间距明确，返回位移与建议数，未自动生成或执行路径。RSS/间距只是可见启发式；实际 n_images 由 agent 决定。小型只读操作可高通量组合。 |
| 41 | [remap_neb_endpoint_atoms](../../catmaster/tools/geometry_inputs/neb_tools.py#L497) | primitive | 部分符合 | ✓ / △ / △ / ✓ | 有显式端点、MIC、锁定阈值、输出与 overwrite；移动子集仅从结构约束/sidecar 推断，没有显式待重排索引。映射数组只在 artifact，普通小返回没有完整映射文件入口；需补选择与可达映射。 |
| 42 | [make_neb_geometry](../../catmaster/tools/geometry_inputs/neb_tools.py#L843) | primitive | 部分符合 | ✓ / ✓ / ✓ / △ | 单项端点、插值/MIC、overwrite 和完整 image summary 合理；批量强制一级 IS/FS、禁止任意额外根文件/嵌套，一项失败中止且没有逐项结果清单。应保留单项路径能力，并改善 batch 选择/部分失败。 |
| 43 | [make_dimer_mode_from_neb](../../catmaster/tools/geometry_inputs/dimer_tools.py#L775) | primitive | 基本符合 | ✓ / ✓ / ✓ / ✓ | 从明确 image tree 和可选 TS 索引导出邻差向量，MIC、输出、overwrite 明确；全部源路径与原始/归一化向量可读。中央 image 只是公开默认，没有替 agent 判断真正 TS。 |
| 44 | [make_dimer_mode_from_mace](../../catmaster/tools/geometry_inputs/dimer_tools.py#L850) | 计算服务 | 部分符合 | △ / △ / △ / △ | 描述声称 finite-difference，实际强制 method=auto，可用 Hessian；不像频率工具可选择 method。fallback 原因未写该工具 summary，结果仅导出选中模式；本地计算范围/潜在工作量也应明确。 |
| 45 | [mace_analyze_frequencies](../../catmaster/tools/geometry_inputs/dimer_tools.py#L960) | 计算服务 | 部分符合 | N/A / ✓ / △ / ✓ | 实现可选 auto/Hessian/差分、显式活跃原子、完整频率/模式文件与 fallback 记录，输出隔离较好；但 active specialist/worker 未绑定此注册工具，材料 worker 不能直接调用。需判定该能力应开放给谁，而非宣布已经可用。 |

### 材料检索与结果分析

| # | 工具与实现 | 类型 | 判断 | P / F / R / H | 依据与处理 |
| ---: | --- | --- | --- | --- | --- |
| 46 | [mp_search_materials](../../catmaster/tools/retrieval/matdb.py#L198) | 查询 | 部分符合 | ✓ / ✓ / △ / △ | 直接 criteria/fields 与全量所取记录 CSV 很好；limit 可增大，但没有 cursor/offset，截断后只能更大限额重取或改查询。大集合的稳定续查和 provider 计数失败行为需要明确，未做在线 MP 验收。 |
| 47 | [mp_download_structure](../../catmaster/tools/retrieval/matdb.py#L321) | primitive | 不符合 | △ / ✗ / △ / △ | R11（模拟远端返回、真实本地变换）：强制 conventional cell，schema 未声明且无选择；输入 1 原子保存 2 原子，metadata natoms 仍写 1。下载夹带不可选结构变换，影响后续科学输入。 |
| 48 | [identify_structure_fragments](../../catmaster/tools/analysis/fragment_probe.py#L207) | primitive | 基本符合 | ✓ / ✓ / ✓ / ✓ | 连接方法可选、显式参考索引/片段/组成条件、全量片段 JSON；描述限定 periodic structure，单结构可组合。注意它是启发式连通性探测，不承担化学键判定；无须另造“自动 adsorbate 判断”流程。 |
| 49 | [analyze_vasp_neb_results](../../catmaster/tools/analysis/results_analysis.py#L736) | primitive | 部分符合 | ✓ / △ / △ / ✓ | 编号目录/OUTCAR 是真实 VASP NEB 约定，缺能量时拒绝报势垒，完整 profile 可读；但“completed”仅由使用场景声称，解析器只取末次 TOTEN，没有给每 image 收敛/运行状态。需要区分当前路径能量与已收敛势垒。 |
| 50 | [analyze_trajectory](../../catmaster/tools/analysis/results_analysis.py#L834) | primitive | 部分符合 | ✓ / △ / ✓ / △ | 显式轨迹、歧义拒绝、物理帧间隔、包裹语义、拟合窗口和全量 CSV 已有；仍绑定 MSD+RDF+绘图一起做，无分析项/抽帧选择，全部帧入内存、RDF 逐帧全距离矩阵。适合中小轨迹，不能据此宣称长轨迹高通量适配。 |
| 51 | [vaspkit_adsorbate_thermo_correction](../../catmaster/tools/analysis/vaspkit_thermo.py#L605) | primitive | 部分符合 | ✓ / △ / ✓ / △ | T 与单位清楚、返回 backend/近似；输入描述只说 VASPKIT501，实际缺 VASPKIT 自动换 ASE，后者仅提取实频并固定 50 cm^-1 下限。缺少后端/频率处理选择，不能把环境依赖当科学选择。 |
| 52 | [vaspkit_gas_thermo_correction](../../catmaster/tools/analysis/vaspkit_thermo.py#L631) | primitive | 部分符合 | ✓ / △ / ✓ / △ | T/P/自旋可传、单位及近似可见；缺程序自动切 ASE，几何/旋转对称数自动推断且不能覆盖，对称分析异常降到 1。需要在入口写明和开放关键选择；不是 OUTCAR 固定名本身的问题。 |

### 写作与可视化

| # | 工具与实现 | 类型 | 判断 | P / F / R / H | 依据与处理 |
| ---: | --- | --- | --- | --- | --- |
| 53 | [generate_figure](../../catmaster/tools/analysis/generate_figure.py#L142) | 模型服务 | 基本符合 | N/A / ✓ / ✓ / ✓ | 明确外部模型生成/编辑单图，prompt 原样、model/reference_images/image_options 可控，返回保存文件；独立路径可组合。单图范围已声明，不应伪装成科学数据绘图 primitive；未产生付费调用。 |
| 54 | [compile_text](../../catmaster/tools/analysis/agentic_compile_tex.py#L304) | primitive | 部分符合 | ✓ / △ / △ / △ | 当前是本地编译器包装，无隐藏 LLM；却固定 pdflatex/BibTeX，不能选 XeLaTeX/LuaLaTeX/Biber/输出目录。正则静态诊断也参与 compiled_ok，前 8 条以外诊断可能只在 artifact；一般 TeX 编译能力不完整。 |
| 55 | [render_markdown_pdf](../../catmaster/tools/analysis/markdown_pdf.py#L162) | primitive | 部分符合 | ✓ / △ / ✓ / △ | 原文保留、唯一临时目录/浏览器 profile、显式 PDF 路径，基本转换可组合；字体仅两种且不能传 CSS。渲染前直接删除已有 PDF，后续失败会丢旧成品，覆盖行为应明确并可恢复。 |
| 56 | [peer_review_pdf_manuscript](../../catmaster/tools/analysis/peer_review_pdf_manuscript.py#L86) | 模型服务 | 部分符合 | N/A / △ / △ / ✓ | 单 PDF/外部模型/自然语言结果明确，model 可选；round_index/max_rounds 只是 prompt 上下文，名称易暗示工具执行多轮。当前未绑定 active specialist/worker，属保留接口，需明确用途/与另两种审阅接口关系。 |
| 57 | [peer_review_request](../../catmaster/tools/analysis/peer_review_request.py#L62) | 模型服务 | 部分符合 | N/A / △ / △ / △ | 说明明确会调用全部配置审稿模型；内部却固定 ACS-style 与三标题，无模型子集/单项恢复。串行第二模型失败会抛错并丢掉已获得 reviews 的工具返回，不满足批量模型结果保留。 |
| 58 | [review_pdf_manuscript](../../catmaster/tools/analysis/review_pdf_manuscript.py#L76) | 模型服务 | 基本符合 | N/A / ✓ / ✓ / ✓ | 明确一次多模态外审，PDF、focus、context、model 可选，完整正文返回；没有隐藏修改/多轮循环。作为模型服务可保留，不把其结果当确定性验证，在线质量未验收。 |
| 59 | [render_vesta_views](../../catmaster/tools/analysis/vesta_render.py#L325) | primitive | 基本符合 | ✓ / ✓ / ✓ / ✓ | 明确“standardized views”范围，top/side/iso、重复胞、尺寸、输出命名可控，独立临时 HOME，全部图片路径返回。不是任意美术渲染器；GUI/外部 VESTA 并发未在线验收。 |

### 文献与网页

| # | 工具与实现 | 类型 | 判断 | P / F / R / H | 依据与处理 |
| ---: | --- | --- | --- | --- | --- |
| 60 | [ingest_literature_files](../../catmaster/runtime/literature/corpus.py#L245) | primitive | 基本符合 | ✓ / ✓ / ✓ / ✓ | 显式多文件、逐文件事务与错误、同路径替换索引，原文仍可读；100 项上限可拆批，未静默裁掉后续项。适合本地 corpus 入库；多进程写入吞吐未压测。 |
| 61 | [query_literature_corpus](../../catmaster/runtime/literature/corpus.py#L342) | 查询 | 部分符合 | ✓ / △ / ✓ / ✓ | 分页、完整源路径/页码及 partial 标记保持可达；但 query 无条件分词后 OR 拼接，不能表达 AND/短语或文献限定，接口未说清。应保留自然语言便捷入口并明确检索语义/可选过滤。 |
| 62 | [acquire_literature_source](../../catmaster/runtime/literature/acquisition.py#L886) | 授权获取接口 | 部分符合 | N/A / ✓ / △ / ✓ | 授权/OA 回退、内容验证、可读本地输出及同请求并发合并有真实边界；R15 显示缓存键不含 expected_title，同 DOI 改标题约束不重新校验。该限制须修，不能通过“已缓存”掩盖新约束。 |
| 63 | [batch_acquire_literature_sources](../../catmaster/runtime/literature/acquisition.py#L1038) | 授权获取接口 | 部分符合 | N/A / △ / ✓ / △ | 列表/文件入口、去重、逐项结果和失败后继续合理；执行串行、50 项上限可拆批。预校验遇到无效 identifier 拒绝整批，且没有每条 expected_title；需明确批次边界及恢复成本。 |
| 64 | [finalize_citations](../../catmaster/runtime/literature/citations.py#L191) | primitive | 部分符合 | ✓ / △ / ✓ / △ | DOI-only 在 schema 已声明，去重、3 路并发解析、未解析项完整返回；输出锁在 notes/literature，stem 清洗可能合并不同请求，不能直接选目的路径/冲突策略。不能因未支持任意标识符而误称 DOI 解析错误。 |
| 65 | [search_openalex](../../catmaster/runtime/literature/tools.py#L442) | 查询 | 基本符合 | ✓ / ✓ / ✓ / ✓ | 查询、page size、cursor 与下一页均公开，整页元数据直接可见；失败与空结果分开。作为 paper lookup 而非全部 OpenAlex API 的边界清楚，provider 实时行为未验收。 |
| 66 | [search_semantic_scholar](../../catmaster/runtime/literature/tools.py#L478) | 查询 | 基本符合 | ✓ / ✓ / ✓ / ✓ | 查询、年份、offset、next_offset 明确，元数据整页返回，rate limit 单独表达；可独立分页与重试。不是引用网络批量导出接口，不要求在本工具塞入所有远端 API。 |
| 67 | [get_openalex_record](../../catmaster/runtime/literature/tools.py#L527) | 查询 | 基本符合 | ✓ / ✓ / ✓ / ✓ | 显式 DOI/work ID 取单条标准论文元数据，结果直接返回，失败清楚，无共享写入。单条 lookup 可重复组合，未验收在线字段完整性。 |
| 68 | [get_semantic_scholar_record](../../catmaster/runtime/literature/tools.py#L550) | 查询 | 基本符合 | ✓ / ✓ / ✓ / ✓ | 显式 DOI/paper ID 取记录，完整当前 PaperRecord 投影可见，限流单独返回；无无关工作流或批处理强制前提。外部 API 可用性未验收。 |
| 69 | [recommend_semantic_scholar](../../catmaster/runtime/literature/tools.py#L579) | 查询 | 基本符合 | ✓ / ✓ / ✓ / ✓ | seed/positive/negative 与 limit 公开，完整本次推荐返回；明确为推荐而非 exhaustive search，不应把 provider 推荐上限误认作隐蔽裁剪。未自动获取/执行推荐项。 |
| 70 | [web_search](../../catmaster/runtime/literature/tools.py#L745) | 查询 | 部分符合 | ✓ / △ / △ / ✓ | 公开搜索故障会标明 scholarly fallback，保持状态诚实；但 max_results≤20、无续页，模型正文裁短标题/摘要且 suppress offload，完整 hit 内容可能只留 artifact。适合发现线索，未满足广义搜索结果的完整可达要求。 |
| 71 | [open_public_page](../../catmaster/runtime/literature/tools.py#L885) | 查询 | 部分符合 | ✓ / ✓ / △ / ✓ | 实现完整稳定快照、source_path/offset 分页、歧义校验，能力形态良好；active specialist 的搜索装配只选 web_search，未绑定此注册读取工具。原生 provider 浏览与普通函数路径需分别判断，不能把“注册”写成“可调用”。 |
| 72 | [find_in_page](../../catmaster/runtime/literature/tools.py#L932) | 查询 | 部分符合 | ✓ / ✓ / △ / ✓ | 快照与 match_offset 续查合理，但当前未绑定 active specialist；R14 还复现 casefold 改变字符串长度时原文坐标错误：Straße target 的 target 起点应 7，返回 8。需修 Unicode 定位并明确可用角色。 |

### 工作区接口

| # | 工具与实现 | 类型 | 判断 | P / F / R / H | 依据与处理 |
| ---: | --- | --- | --- | --- | --- |
| 73 | [apply_aider_edits](../../catmaster/tools/misc/memory_patch_apply.py#L114) | 兼容编辑接口 | 部分符合 | ✓ / △ / △ / △ | 当前未绑定 active specialist/self-evo；强依赖 Aider SEARCH/REPLACE 语法，allowed_paths 强制追加 / 使精确文件名不能作为该前缀，emit_diff 只入 artifact。虽有预计算/写失败回滚，不宜作为普通文件编辑默认入口。 |
| 74 | [export_builtin_tool_source](../../catmaster/tools/misc/export_builtin_tool_source.py#L107) | primitive | 基本符合 | ✓ / ✓ / ✓ / ✓ | 按 tool/module 导出原始源码及静态依赖，保留路径/import，完整树可 grep/read；明确不含动态依赖/非 Python 资产，可追加 module 查询。已有文件冲突先检查，未看到截断源码或假装导出可执行独立程序。 |
| 75 | [manage_effective_skills](../../catmaster/tools/misc/effective_skills.py#L109) | 系统接口 | 基本符合 | N/A / ✓ / ✓ / ✓ | list/detail 分页与版本/历史游标明确；变更使用当前会话来源和先前所见状态，返回完整 JSON。enable/pin/mode/边界处理属于真实持久状态接口，不需要为了 primitive 拆掉其事务关系。 |
| 76 | [notify_progress](../../catmaster/tools/misc/progress.py#L28) | 系统接口 | 基本符合 | N/A / ✓ / ✓ / ✓ | 只上报一条进展与可选下一步，通过原生工具事件承载；没有改变科学状态/调度状态。不是运行存活或等待轮询工具，边界明确。 |

### Research Graph

| # | 工具与实现 | 类型 | 判断 | P / F / R / H | 依据与处理 |
| ---: | --- | --- | --- | --- | --- |
| 77 | [list_research_graphs](../../catmaster/tools/misc/research_graph.py#L505) | 查询 | 部分符合 | ✓ / △ / ✓ / △ | 目录全文可达且有 archive 开关；没有分页/过滤，大 workspace 一次输出全部 graph 问题与标题。当前没有静默截断，但缺少大目录的可控分页。 |
| 78 | [revise_research_claim](../../catmaster/tools/misc/research_graph.py#L1049) | 事务接口 | 部分符合 | N/A / △ / ✓ / ✓ | 保留旧新 claim、显式 action 与 revision 的事务方向正确；graph_id/old_node_id/new_node_id/action/expected_revision 顶层无说明，需解释修改对象与引用方向，保持普通 SQL 可查询。 |
| 79 | [record_research_disposition](../../catmaster/tools/misc/research_graph.py#L1062) | 事务接口 | 部分符合 | N/A / △ / ✓ / ✓ | 用宿主绑定 graph/thread/run 写 stopping decision，状态记录可查询；reason/basis_node_ids/exception 缺少字段说明，例外的含义不能靠名称猜。应补作用范围而非增加治理字段。 |
| 80 | [record_research_review](../../catmaster/tools/misc/research_graph.py#L1076) | 事务接口 | 部分符合 | N/A / △ / ✓ / ✓ | 仅独立 reasoner 可写 stopping review 是真实角色边界；decision_id 缺说明，工具一句话未讲清 reviewer 所针对的决定与权限。需在 tool/skill 说明当前绑定关系，不能直接向普通 agent 随意开放。 |
| 81 | [mark_research_planning_no_change](../../catmaster/tools/misc/research_graph.py#L980) | 事务接口 | 基本符合 | N/A / ✓ / ✓ / ✓ | 明确只对当前 planning pass，reason 必填；返回是否继续从已有 ready E 比较选择，无新分支不等于结束整个研究。宿主绑定和事务边界适当。 |
| 82 | [create_research_graph](../../catmaster/tools/misc/research_graph.py#L553) | 事务接口 | 部分符合 | N/A / △ / △ / ✓ | 可建 graph+初始假设并绑定当前线程；正文给 graph ID/revision，但初始节点 ID 只在 artifact，需再 SQL 找回。question/模式/初始假设等描述也不完整，需明确创建副作用与可复用句柄。 |
| 83 | [query_research_graph_sql](../../catmaster/tools/misc/research_graph.py#L595) | 查询 | 基本符合 | ✓ / ✓ / ✓ / ✓ | 公开逻辑表/JSON1 路径，支持 SELECT/WITH、自选 LIMIT/OFFSET，无隐藏行截断，结果正文可读。绑定 graph 及独立 comparison 隔离有明确权限原因；未用 preview/detail 替代通用查询。 |
| 84 | [set_research_graph_focus](../../catmaster/tools/misc/research_graph.py#L626) | 事务接口 | 基本符合 | N/A / ✓ / ✓ / ✓ | 明确选中/清空当前线程 focus，返回节点/邻接关系完整 JSON；状态以线程隔离，不同线程可独立复用。同一线程的 focus 是顺序状态操作，应按这个真实边界使用。 |
| 85 | [create_bound_research_experiment](../../catmaster/tools/misc/research_graph.py#L673) | 事务接口 | 部分符合 | N/A / △ / △ / △ | R12 所示共用返回只给标题/revision，未给新 E ID；缺少关键字段说明。服务每次生成新 UUID，无幂等键或同目标复用，此工具应明确“创建新 E”，重试/已有 E 应 query+focus。重复登记不能只归咎 agent。 |
| 86 | [update_bound_research_result](../../catmaster/tools/misc/research_graph.py#L721) | 事务接口 | 部分符合 | N/A / △ / ✓ / ✓ | 原位修正同一 Result，methods/conclusion 的省略保留已明确，避免再建结果；result_node_id/summary/refs 无说明，应写清聚焦 E 的约束及 refs 是追加。正文已有结果 ID，可继续查询。 |
| 87 | [resume_bound_research_experiment](../../catmaster/tools/misc/research_graph.py#L765) | 事务接口 | 部分符合 | N/A / △ / ✓ / ✓ | 恢复当前 focused blocked E 的既有状态，不新建实验；reason/refs 缺说明。须明确这是 Research Graph 实验状态恢复，与 DBOS/LangGraph 线程排队恢复不同。 |
| 88 | [retract_bound_research_result](../../catmaster/tools/misc/research_graph.py#L800) | 事务接口 | 部分符合 | N/A / △ / ✓ / ✓ | 仅同 run/thread 所有权、focused E 的误分类 Result 可撤回，权限边界合理；字段缺说明。应告诉 agent 能撤回什么、理由如何用，不能当任意历史结果删除器。 |
| 89 | [update_research_graph_scope](../../catmaster/tools/misc/research_graph.py#L846) | 事务接口 | 部分符合 | N/A / △ / ✓ / ✓ | 用户指示的 graph 范围修正，宿主绑定和角色限制合理；空字符串字段被过滤，不能用空值清除，schema 未讲清保留语义。补齐 title/question/criterion 的变更范围。 |
| 90 | [add_research_hypothesis](../../catmaster/tools/misc/research_graph.py#L885) | 事务接口 | 部分符合 | N/A / △ / △ / ✓ | 单次新增假设、证据引用和 revision 合理；新增 ID 只在 artifact，正文只给标题。suggested_by_result_ids/refs 等无描述，需把新增句柄和边方向直接给 agent。 |
| 91 | [add_research_experiment](../../catmaster/tools/misc/research_graph.py#L912) | 事务接口 | 部分符合 | N/A / △ / △ / △ | 每次新增 E，原子事务/revision 有保护但不是幂等重试；正文只给标题，tests/dependencies/state 等缺说明。应暴露 E ID，并明确新增、复用与更新的分工。 |
| 92 | [record_research_result](../../catmaster/tools/misc/research_graph.py#L1121) | 事务接口 | 部分符合 | N/A / △ / △ / ✓ | 记录观测+科学方法+解释+typed judgments，保留证据图关系；新增 R ID 仅 artifact，judgments/refs/revision 缺说明。完整对象仍可 SQL 查询，但额外查询摩擦没有必要。 |
| 93 | [set_research_result_judgment](../../catmaster/tools/misc/research_graph.py#L1159) | 事务接口 | 部分符合 | N/A / △ / ✓ / ✓ | 对明确 R/H 覆盖一条证据关系，revision 控制合理；relation/scope/rationale 缺顶层说明，容易与改 claim 混淆。保持单关系 primitive，补作用范围即可。 |
| 94 | [stage_research_plan](../../catmaster/tools/misc/research_graph.py#L940) | 事务接口 | 基本符合 | N/A / ✓ / ✓ / ✓ | 绑定当前 planning 事务，暂存多个假设/实验并保留临时标签对应，提交结果可从 graph 查询；批量原子边界有实际理由。不能仅因含数组/结构化而判坏，未把它当执行工具。 |
| 95 | [record_research_experiment_comparison](../../catmaster/tools/misc/research_graph.py#L1023) | 事务接口 | 部分符合 | N/A / △ / ✓ / ✓ | 针对宿主绑定 A/B comparison 记录结果，隔离有科学理由；outcome 顶层无说明且仅返回 recorded。应写清取值相对哪两个候选，不让模型猜枚举方向。 |
| 96 | [set_research_graph_completion](../../catmaster/tools/misc/research_graph.py#L1089) | 事务接口 | 基本符合 | N/A / ✓ / ✓ / ✓ | 明确 graph/revision/completed，要求已有实际 Result，false 可重开，返回新 revision；这是图状态写入，不能当取消线程/停止远端作业接口。 |
| 97 | [record_bound_research_result](../../catmaster/tools/misc/research_graph.py#L1238) | 事务接口 | 部分符合 | N/A / ✓ / △ / ✓ | 利用 focused E 与当前 thread/run 来源减少重复参数，schema 边界较清楚；新增 Result ID 仍只在 artifact，正文标题不足以直接后续修订。需返回稳定 R 句柄，保留宿主所有权。 |
| 98 | [mark_research_experiment_failed](../../catmaster/tools/misc/research_graph.py#L1201) | 事务接口 | 基本符合 | N/A / ✓ / ✓ / ✓ | 显式 graph/E/revision、具体 reason 与 refs，描述明确实际写 blocked；没有把失败伪造成成功结果。合适的状态 primitive，不能用于掩盖重复建 E。 |
| 99 | [mark_bound_research_experiment_failed](../../catmaster/tools/misc/research_graph.py#L1292) | 事务接口 | 基本符合 | N/A / ✓ / ✓ / ✓ | 当前 focused E、具体 blocker 与自动来源，缺 E 会明确报错；无自动新建/跨任务选择。保持绑定边界，blocked 的后续由现有 resume 工具处理。 |

### 机器学习数据

| # | 工具与实现 | 类型 | 判断 | P / F / R / H | 依据与处理 |
| ---: | --- | --- | --- | --- | --- |
| 100 | [build_dataset_from_runs](../../catmaster/tools/machine_learning/dataset_tools.py#L288) | primitive | 部分符合 | ✓ / ✓ / △ / △ | 范围已明确只读 VASP XML；final/all、收敛筛选、split unit/比例/seed 可控，保留完整数据及 skipped 清单。全数据入内存且强绑定切分输出；所有输入失败时仅说 No frames，逐项原因未写出，恢复摩擦仍在。 |
| 101 | [calculate_al_candidates](../../catmaster/tools/machine_learning/mace_ml.py#L770) | 科学选择服务 | 不符合 | △ / ✗ / △ / ✗ | 特征固定为成分/体积/80-bin 距离直方图，分数组合与贪心规则不可选，完整未选候选评分不返回；_greedy_select 构造 N×N×特征维数组，不适合大候选池。若保留应明确算法及可调控制/全量评分，避免把固定筛选配方当通用 primitive。 |

## 本次验证与边界

离线探针及观察结果随报告保存：

- [复现脚本](registered_tool_primitive_probes_20260926.py)
- [实际观察 JSON](registered_tool_primitive_probe_results_20260926.json)

脚本创建独立 `/tmp` workspace，不修改公共项目或实际计算输出。R00 是注册/schema 检查；R01 至 R16 是 16 组行为观察，其中 R16 覆盖四个工具。R09 调用实际手工位移辅助函数；R10 注入 phonopy 失败；R11 模拟 MP 网络返回后运行真实结构变换；R12 观察共用 Graph 返回函数；R13 注入力场不可用；R14 模拟网页快照；R15 模拟 acquisition 网络层。其余科学准备/解析探针直接运行本地工具。模拟依赖用于隔离本地逻辑，不能冒充提供方或引擎验收。

| 观察编号 | 直接观察 |
| --- | --- |
| R00 | 101 个注册工具；两类最终 schema 一致（忽略重复 description）；Graph 14 工具/51 顶层字段无说明；5 名称未在 active runtime 绑定 |
| R01 | name 改变不改变输出名，两个请求都写 name/mol.xyz |
| R02 | 100、102 kcal/mol 的能量窗筛选得到 0 个，应为 2 个 |
| R03 | 两个 run 只提取 1 个，双帧 XYZ 取 0.7 Å 首帧；直接传该 run 目录得到 0 个 |
| R04 | supercell processed=2，distinct output=1 |
| R05 | ORCA prepared_count=2，distinct stage=1 |
| R06 | 末帧截断仍 completed，nframes=1，输出前一完整帧，无截断提示 |
| R07 | CP2K 选中调度日志，真实能量未进入摘要 |
| R08 | 自定义 ORCA/LAMMPS 文件名及无 summary 的 xTB 目录均找不到 run |
| R09 | Na/Cl 扩胞后两个代表位点的六个位移均移动 Na |
| R10 | 注入 phonopy 失败后返回手工生成成功，未保留原错误 |
| R11 | MP 原始 1 原子，保存 2 原子，返回 natoms=1 |
| R12 | 新 E ID 在 artifact 中，正文没有该 ID |
| R13 | MMFF 不可用时生成 completed、能量 null，没有优化失败说明 |
| R14 | Straße target 的 target 原文起点为 7，工具返回 8 |
| R15 | 同 DOI 两次 title A/title B 请求只执行 title A 的校验路径 |
| R16 | build_slab 和三个 fix_atoms 工具均报告 2 项成功，输出落到同一目标 |

在仓库根目录复跑探针：

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/catmaster-audit-mpl \
  conda run -n catmaster python \
  tests/manual/registered_tool_primitive_probes_20260926.py
```

本次重新运行下列已有回归，结果 **65 passed，28 warnings，4.90 s**：`test_orca_xtb_tools.py`、`test_cp2k_lammps_tools.py`、`test_crystal_tool_expansion.py`、`test_tool_output_adapter.py`、`test_research_graph_query.py`。这些通过结果与上述缺陷并不矛盾：它们没有覆盖这里使用的重名、截断、不同布局和回退索引样例。没有把上一轮其他测试的通过数重复算成本次验证。

没有进行外部服务、GUI、实际科学引擎、GPU 内存或并发压力验收。正文已把未压测的并发写风险、基于源码的规模问题与已复现错误区分开；也没有据本地源码推断公共 demo 已使用这些代码。

DeepAgents 通用文件/shell 工具、动态异步任务/切档工具、self-evo 专用查询及终止工具、provider 原生 hosted search 不在这 101 项注册表内。这些不是本次逐项清单的遗漏，也没有在本报告里给它们新的“已通过全面审查”结论。历史函数如未注册的 `mace_train`/`mace_evaluate` 也不重复计入。

## 实施判断

修复顺序应先处理会给错科学输入/输出和丢掉批次结果的路径：构象筛选与提取、批量命名、声子位移、k-path 配套结构、MP 下载变换、解析来源与截断状态。随后处理原生控制不足、完整句柄/结果可达、失败后恢复及长数据处理。只涉及范围、坐标/单位、绑定关系、参数保留语义的问题，直接在工具描述中解释，并在对应 skill 放一两个例子。

工具不必统一成一张大 schema，也不需要一套新 patch 语言。普通文件修改沿用现有文件工具；方法语法由原生输入表达；科学选择留给 agent；有真实事务或授权边界的接口保留该边界。

上述原则已写入根目录 [AGENTS.md 的 Primitive Tool Design and Review](../../AGENTS.md#primitive-tool-design-and-review)。该文件在当前仓库由 `.gitignore` 忽略，本次修改已写入本地文件，未改变忽略规则。普通工具实现、部署及线上任务均未修改。
