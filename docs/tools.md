## 工具目录

| 名称 | 功能描述 |
| --- | --- |
| create_molecule_from_smiles | 从 SMILES 自动生成三维分子（ETKDGv3+MMFF，回退 UFF），写出 XYZ 与带盒子的 POSCAR，显著降低 LLM 编坐标出错。 |
| relax_prepare | 从结构文件或目录生成 VASP 放松输入集（支持 gas/bulk/slab/lattice），产出可直接运行的计算目录。 |
| mace_relax | 通过资源路由器提交一次 MACE 结构弛豫任务，返回放松结构、轨迹与日志等产物。 |
| vasp_execute | 提交一次 VASP 计算，使用已准备好的输入目录，路由与作业提交自动处理。 |
| vasp_summarize | 解析 VASP 输出目录，提取能量、收敛标志及最终结构等信息形成结构化摘要。 |
| list_files | 递归列出工作区内指定相对路径下的文件和目录。 |
| read_file | 读取工作区相对路径的文件，返回指定最大字节的内容。 |
| write_file | 将文本内容写入工作区相对路径文件，必要时自动创建父目录。 |
| find_text | 在工作区指定目录下搜索包含给定子串的文件并返回匹配路径。 |
| python_exec | 执行简短、无副作用的 Python 代码片段，返回其打印输出。 |
| write_note | 将简短笔记（可含标签）写入代理的观察日志以便后续回忆。 |
| mp_search_materials | 公式/化学体系检索 Materials Project，返回候选 mp_id 与关键性质，支持带隙/能量过滤（需环境变量 MP_API_KEY）。 |
| mp_download_structure | 通过 mp_id 下载结构为 POSCAR/CIF/JSON，写入工作区并返回路径与元数据（需环境变量 MP_API_KEY）。 |
| build_slab | 从体相结构生成指定 Miller 面的 slab，支持厚度/真空/超胞和对称终止选择。 |
| set_selective_dynamics | 为 slab 设置 Selective Dynamics（冻结底层或指定弛豫厚度），写出新结构文件。 |
