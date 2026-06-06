# 功能介绍与日常使用

本章介绍 WebUI 中每个任务模式适合做什么，以及日常使用时应该怎样组织项目空间和提示词。

## 1. 基本概念

### 项目空间

项目空间是一个普通目录，用来保存：

- 上传的结构、文稿、数据和计算结果。
- agent 新建的脚本、输入文件、报告和图表。
- 每次 run 的历史、日志和中间状态。

建议一个研究问题或一个项目使用一个 project space。

### Run

一次提交就是一个 run。中断或需要继续时，可以在 WebUI 中选择历史 run，然后使用 `resume_selected_run`。

### Files

WebUI 的文件视图可以浏览项目空间里的文件、上传文件、预览文本和下载 workspace zip。

## 2. 任务模式

### Experiment

适合边界明确的计算任务：

- 构建 slab、吸附结构、分子结构。
- 准备 VASP、CP2K、LAMMPS、ORCA、xTB 输入。
- 调用 MACE relax、single point、MD、NEB、训练或评估任务。
- 分析已有输出，例如 `OUTCAR`、`vasprun.xml`、CP2K output、LAMMPS log。
- 提交已经准备好的远程 stage。

示例：

```text
读取当前项目里的 Ni slab 和 CO 分子，生成 CO 在顶位、桥位、空位的吸附结构，并为每个结构准备 VASP static 输入。
```

### Research

适合开放研究问题：

- 根据目标反应提出候选催化剂。
- 结合文献和项目已有结果制定筛选计划。
- 把一个大问题拆成文献、结构准备、计算、结果整理等多轮任务。

示例：

```text
围绕 CO2 加氢到甲醇，制定一轮 Cu 基催化剂筛选计划。先结合文献给出候选体系，再安排可执行的计算任务。
```

### Literature Review

适合直接做文献综述和公开来源核查：

- 代表性论文列表。
- DOI、年份、期刊、作者等元数据确认。
- 按催化剂结构、活性指标、证据类型整理文献。

示例：

```text
综述近五年单原子 Ni 催化 CO2 电还原生成 CO 的代表性工作，按催化剂结构、活性指标、关键证据和 DOI 整理。
```

### Writing

适合基于项目空间里的证据写作：

- 写 Results and Discussion。
- 修改摘要、引言、讨论。
- 整理 TeX 草稿。
- 把计算结果转成报告或回复草稿。

示例：

```text
根据当前项目中的计算结果和图表，写一版 ACS 风格的 Results and Discussion，输出 TeX。
```

### Peer Review

适合审阅已有 PDF 文稿：

- 检查结论是否被数据支持。
- 检查方法是否可复现。
- 给出 reviewer 风格评论和修改建议。

示例：

```text
审阅 files/manuscript.pdf，重点检查催化机理证据是否充分、计算方法是否可复现、结论是否被数据支持。
```

## 3. 提示词写法

好的提示词通常包含：

- 目标：希望最终得到什么。
- 输入：项目空间里哪些文件可用。
- 约束：方法、模型、泛函、结构范围、队列或资源限制。
- 输出格式：报告、CSV、JSON、VASP 输入目录、TeX 等。
- 是否允许远程提交或只做准备。

示例：

```text
使用 files/slab.vasp 和 files/CO.xyz，生成 CO 在 Fe(110) 的 ontop、bridge、hollow 初始吸附结构。输出到 adsorption_structures/，并写一个 summary.csv，记录位点名、初始高度和文件路径。暂时不要提交远程计算。
```

远程示例：

```text
把 vasp_inputs/ 下每个子目录作为一个 VASP stage，用 remote_submission_batch 提交 task_name=vasp_execute。提交前检查每个子目录都有 INCAR、KPOINTS、POSCAR、POTCAR。
```

## 4. 外部程序

只在需要对应功能时安装：

- OVITO：结构渲染和结构视图导出。
- LaTeX / `pdflatex`：TeX 文稿编译。
- VASPKIT：吸附物和气相热力学校正。
- ORCA、xTB、CREST：量化、半经验和构象搜索。
- VASP、CP2K、LAMMPS：第一性原理和分子模拟。
- MACE：机器学习势 relax、single point、MD、NEB、训练和评估。

外部程序可以在本机或远程机器上。远程程序路径通常通过 [远程配置要点](02-remote.zh.md) 中的 resource 环境加载。

## 5. 日常工作流

### 准备结构

1. 上传 bulk、slab、分子或已有计算结果。
2. 用 Experiment lane 生成结构或输入文件。
3. 检查输出目录和 summary 文件。
4. 再决定是否提交远程计算。

### 提交计算

1. 确认 stage 目录完整。
2. 选择 `remote_submission` 或 `remote_submission_batch`。
3. 明确 `task_name`，例如 `vasp_execute` 或 `mace_relax_dir`。
4. 运行后记录 `remote_context_id`、`submission_hash`、`receipt_rel`。

### 分析结果

1. 上传或下载计算结果。
2. 让 Experiment lane 提取能量、结构、收敛状态和关键指标。
3. 输出 CSV/JSON/Markdown 报告。

### 写作整理

1. 把 figures、tables、CSV、计算摘要放进项目空间。
2. 用 Writing lane 写结果、讨论或回复。
3. 用 Peer Review lane 检查证据链和可复现性。

## 6. 继续历史任务

如果一次任务没有完成：

1. 在 WebUI 左侧选择历史 run。
2. 把 run mode 改成 `resume_selected_run`。
3. 补充新的指令，例如“继续上次未完成的 VASP 输入准备，先检查已有文件，不要重做已经完成的结构”。

## 7. 下一步

- WebUI 不能启动：回到 [本地配置要点](01-local.zh.md)。
- 要提交集群任务：阅读 [远程配置要点](02-remote.zh.md)。
- 只想了解项目能力：回到根目录 [README.md](../../README.md)。
