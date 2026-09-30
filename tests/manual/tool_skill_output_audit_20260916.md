# 工具输出与 skill 脚本执行审查

日期：2026-09-16。范围是项目维护的工具返回、skill 脚本及调用指引；不是全部科学算法或外部服务的重新验收。

## 覆盖

- `catmaster/tools/` 全部 65 个 Python 文件：扫描输出构造、序列化、异常与返回节点，并检查命中的模型可见内容和明细去向。
- `skills/` 全部 22 个 Python 文件、71 个 skill 入口及其 Markdown 资源（共 141 个 `.md`）：检查执行路径、复制/通读要求、打印行为和批量输出。
- 补查 `runtime/literature/tools.py`、`manager_tools.py`、`tool_output_adapter.py`、注册器和 specialist backend：区分真正送给模型的 content、内部 artifact、文件及原生分页。
- 16 个使用 argparse 的 bundled CLI 从独立临时目录运行 `--help`；全部退出 0。其余 6 个为模块或示例，不将执行演示当作普通 CLI 验证。
- 没有启动生产研究任务、访问付费模型或重新跑远程科学计算；没有审计整个第三方 EasySlides 源码。

## 真实轨迹证据

时间为 UTC+8。文本 token 用 `o200k_base` 估算，仅衡量该次工具返回，不是账单或总轨迹消耗。

| 调用 | 时间 | 已保存的工具返回 | 文本 token |
|---|---|---|---:|
| `call_I7aMyTUDMmgGdwqsSeROj8h3`，test | 09-15 09:59:32 | 整份检查器源码 | 4,777 |
| `call_la9gSy1i1RbjquUrPxifAqYa`，Lithium | 09-14 11:13:59 | 自写组装 wrapper 抛出完整 metrics，含 19 个 clash、6 个 anchor | 2,361 |
| `call_ETRUgaYeBOaDzxEP0hMQkTRz`，Lithium | 09-14 11:15:54 | 自写 wrapper 抛出完整 metrics，含 2 个 clash、10 个 anchor | 1,466 |
| `call_TBkGO7f0ATIJl0zi21ysskCO`，Lithium | 09-13 10:40:03 | 原生 CLI 检查 6 个结构 | 491 |

来源：各工作区 `metadata/workspace.sqlite` 的 `thread_messages`，按 `tool_call_id` 去重。组装 thread 为 `0062ee73-0968-5c24-8f50-df6999e957b8`；CLI thread 为 `ff6c7369-aad3-5b76-b084-1aecac144760`。不把图片 base64 当作文本计费，不把完整磁盘 JSON 大小算成已经进入模型的输出。

结论：原 checker CLI 原本就没有打印整个距离表。实际膨胀来自源码通读、自写异常展开，以及执行路径不清带来的复制与重试。已有 PASS 报告中的正常原子对留在磁盘，不能据此认定整表都已消耗 token。

## 修正

### 原始要求与首次实现

2026-08-26 的原始用户消息要求“在skills里落几个实用脚本到ref里，不考虑增加tool了，比如说分块刚体优化模板，overlap检查”。可复用、可执行的脚本用于补足工具覆盖，是当时的明确方向。

Git 中两个 skill 首次出现于 `e5bbf5c`（2026-08-27 09:59:54，UTC+8）。该提交已经写入：

> Copy ... into a workspace `scripts/` path before executing or adapting it.

组装 skill 还以“mounted skill files are reference assets, not writable project files”解释该要求。脚本本身已有 CLI；同一提交的 `_make_backend` 只把快照接到虚拟文件路由，没有给本地 shell 提供 skill 执行映射，运行时还要求挂载内容只通过虚拟路径访问。实现把不能修改挂载原件扩大成执行前必须复制，而未补齐原地调用链路。这是首次实现中的偏差，不是后来 agent 自己才引入的约定。历史记录没有证据表明用户要求过常规执行前复制或包装。

### 原子检查与组装

- 两个 skill 原先明确要求先复制脚本；现使用 bundled CLI，普通检查无需复制、包装或通读源码。
- 原 checker 改为状态、数量、异常距离/归一化比例的极值、周期像、cell 问题、最大目标接触偏差及报告路径。PASS 不再打印一对正常最短距离。
- 组装不可行时返回各约束的最大违反量和完整报告路径；报告仍包含全部已记录的 clash、anchor、orientation、region 明细。
- `format_summary` / `format_metrics_summary` 可供确实需要定制的调用者使用；指引禁止把整份诊断字典放进异常消息。
- 几何阈值、优化目标、接受规则和完整 JSON 结构没有改变。正常最短对仍按 `--top-pairs` 保留，所有 flagged pairs 均保留。

### 直接执行路径

DeepAgents 的 CompositeBackend 路由只映射文件操作，LocalShellBackend 的 shell 不解释虚拟 mount。仅修改 skill 文字不能使 `/.deepagents/skills/...` 成为可执行的主机路径。

每个 backend 现在绑定 `CATMASTER_SKILLS_ROOT` 到自己的有效快照，命令例如：

```bash
python "$CATMASTER_SKILLS_ROOT/atomistic/atomic-structure-validation-and-recovery/scripts/check_atomic_structure.py" \
  structures/candidate.extxyz --output analysis/candidate_geometry.json
```

不创建共享可变软链接，稳定版与 canary 可并存；Python 不在快照写入 bytecode。资源仍可位于 `scripts/` 或 `references/`，无需为了执行移动目录。框架行为已核对安装源码和 [LocalShellBackend 官方文档](https://docs.langchain.com/oss/python/deepagents/backends#localshellbackend-local-shell)。

### 同类问题

- 文献资源中 72 处 `python scripts/...` 示例改为实际挂载路径，避免被误认为工作区已有脚本。
- 两份 `format-converter.py` 的逐篇下载/标题/保存进度改为 `--verbose`；默认保留成功、失败、错误和输出目录。导出文献内容不裁剪。
- `validate_citations.py` 的逐篇验证进度同样由 `--verbose` 控制；具体错误、重复记录和完整报告仍可用。
- scientific-visualization 的示例明确如何从 skill 资源导入 helper 和样式，避免为导入而复制整个目录。
- EasySlides 的“Read its complete scripts”改为资源可用性描述，按相关 workflow 和 help 使用。
- `fix_atoms_by_indices` 不再回显整张输入索引表，返回选择/固定/放松数量及输出路径；完整索引保留在 artifact，实际约束可从输出 POSCAR 的 selective-dynamics flags 读取。

### 继续审查 verbose 的结果

| 入口 | 默认输出 | 显式详细输出 |
|---|---|---|
| `doi_to_bibtex.py` | 完整 BibTeX/JSON 或输出路径、转换总数、错误 | `--verbose` 增加逐 DOI 进度 |
| `extract_metadata.py` | 完整导出或输出路径、总数、错误 | `--verbose` 增加逐条识别与处理进度 |
| `search_pubmed.py` | 完整所请求结果或输出路径、查询数量、错误 | `--verbose` 增加查询回显和获取批次进度 |
| `search_google_scholar.py` | 完整所请求结果或输出路径、总数、错误 | `--verbose` 增加逐条检索进度 |
| 两个原子脚本 | 摘要和已保存的完整报告路径 | `--verbose` 额外打印完整报告，退出码不变 |
| `render_structure_panel.py` | 使用 `--metadata-json` 时返回图片、视图数及报告路径；未保存时仍返回完整 JSON | `--verbose` 允许保存后再打印完整 JSON |
| `validate_citations.py` | 使用 `--report` 时返回统计和路径；不保存时保留错误/重复记录明细 | `--verbose` 增加进度并展开全部诊断 |

本轮新增 7 个 CLI flag，并扩展一个已有 flag 的行为。没有更改检索数量、候选筛选、原始科学结果、导出格式或诊断接受标准，也没有为所有注册工具添加一个通用 `verbose` 参数。

额外保留：`format_bibtex.py` 的重复条目删除通知属于数据修改结果，且默认可原地覆盖输入，不作为普通进度隐藏；其余解析/排序步骤仅为常数行输出。OpenAlex 的 `--compact` 是显式结果展示选择，完整 JSON 仍保留。样式/模板示例打印和模板查询输出本身是用户请求的内容；VASP/ASE 热化学工具已对底层计算传入 `verbose=False`。这些入口不再加一层 verbosity 控制。

## 保留的输出及原因

| 类别 | 审查结论 |
|---|---|
| 结构生成、NEB、MD、量化计算、数据集、MLFF 执行 | 大多已返回数量、关键科学量与实际文件路径；未把内部完整 artifact 当作默认 content 膨胀处理 |
| 文献检索、网页读取、Research Graph 查询 | 查询结果和 continuation/source handles 是原生语义，保留；没有施加统一摘要或候选数限制 |
| remote task specs、工具参数目录 | compact/full 已区分；显式请求的完整 schema 与约束保留 |
| 渲染 helper | metadata 以视图、相机和图片路径为主，不是原子对表；保留其可配置输出 |
| BibTeX/RIS/JSON 导出 | 完整作者、摘要和用户选择的数据保留；大批量可用 output 参数或重定向 |
| 审稿和图像分析 | 请求本身就是分析文本，不替换成统一状态摘要 |
| 原生文件/执行工具 | 没有新加截断、隐藏字段或替代性的 preview 工具 |

另发现一个既存的独立问题：两份 converter 的可选 `--preflight` 分支引用未随资源提供的 `preflight` 模块。普通转换和 `--help` 不经过这个分支；本次没有扩展网络预检功能。文献验证脚本的 BibTeX 解析也仍是既有的简化解析器，本次不改其文献解析语义。

## 验证与适用边界

测试覆盖：正常结构不打印距离表且 JSON 明细完整；跨周期碰撞、正常距离但预期接触失败、不可能满足的组装约束；两个 backend 分别直接执行各自快照资源且输出留在工作区；真实注册工具的最终 ToolMessage 包含可读结构路径；文献默认/verbose 输出和完整导出。

运行时、skill staging、有效版本选择、workspace Python 和工具输出适配器使用现有回归测试。相关用例共 150 项通过：综合运行 149 项通过，一项新测试误把 workspace 虚拟环境的 bytecode 纳入断言；将断言收窄到 skill 快照后，该文件的 2 项定向复测通过。16 个 CLI help smoke 均通过，`git diff --check` 通过。CLI smoke 不代表外部检索 API 或所有科学任务均已在线验收。

verbose 后续修改的针对性测试：43 项通过（`test_skill_verbosity.py`、`test_atomistic_skill_scripts.py`、`test_structure_render_skill_code.py`、`test_academic_search_metadata_completeness.py`）。PubMed 用 205 条模拟记录跨两个获取批次验证导出不丢数据；所有网络返回均为测试替身，没有调用实际外部接口。

修改在开发仓库。没有重启部署、覆盖已有运行快照或改写历史工作区包装脚本；历史包装脚本仍可能自行打印整份字典。对未来 agent 是否每次都选择直接 CLI，还需要新轨迹验证，不能从单元测试推断必然遵循指引。
