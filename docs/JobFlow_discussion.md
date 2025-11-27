下面给出一套从零到一即可运行的方案，总结前述要点，并提供可以直接落地的实现骨架。总体思路是用 LlamaIndex 的本地知识索引承载“工具表→三元组→语义检索”，用 LangGraph 作为有状态的编排外壳，调用 LangChain 的工具执行器将计划逐步落地，执行层通过轻薄的适配器对接你已经打通的 jobflow‑remote。演示任务是用 VASP 完成 O₂ 分子在盒子中的几何优化与能量提取，所有路径都以工作空间为根，严格使用相对路径以保证可追踪和可复现。

框架结构如下所示。左侧是工具知识与检索，中部是有状态计划与执行，右侧是远端作业与分析回写，底部贯穿一个统一的工作空间与清单记录。

```
            [tools.xlsx / YAML]
                    │   （生成三元组）
                    ▼
         LlamaIndex KnowledgeGraphIndex
                    │   （语义检索 + 邻域扩展）
        ┌───────────┴───────────┐
        │                       │
   [Plan 节点]──────────────→ 候选工具链
        │                           │
        ▼                           │
[Materialize 节点] 写入工作空间与清单     │
        │                           │
        ▼                           │
[Exec 节点] LangChain ToolRunner 逐步调用 │
        │         │                  │
        │         └─ catmaster.tools.* 适配 │
        │                     │              │
        │                     └─ jobflow-remote 提交与监控
        │                                        │
        ▼                                        ▼
[Monitor 节点] 轮询远端状态             [Analysis 节点] 提取能量与结构
        │                                        │
        └──────────────────────────────┬─────────┘
                                       ▼
                              [Summarize 节点] 产出报告与证据
```

文件与路径管理采用“RUN_WORKSPACE”环境变量指向当前工作空间根目录的方式，工具只接受和产出相对路径，适配器在调用前后更新 `workspace/manifest.json`，记录每个工件的类型、相对路径和哈希值。VASP 所需的远端传输与目录准备由 jobflow‑remote 的 worker 与 exec_config 完成，你只需要在 exec_config 的 pre_run 中放置同步指令与环境导入，其余交给 Runner 托管。

下面给出四组最小而完整的代码骨架。为了阅读与复用方便，每个文件的职责非常单一，你可以直接落盘并按注释中的命令运行。演示默认使用 OpenAI 的接口，`OPENAI_API_KEY` 从环境读取。

第一组代码将你提供的工具表转为三元组，并构建 LlamaIndex 的本地知识索引。表格可以是你贴出的列集合，若有额外标签也会一并写入属性关系。

```python
# build_tools_kg.py
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.storage.storage_context import StorageContext

def _split(v):
    if pd.isna(v) or str(v).strip()=="":
        return []
    return [s.strip() for s in str(v).split(";")]

def triplets_from_df(df: pd.DataFrame) -> List[Tuple[str,str,str]]:
    T = []
    for _, r in df.iterrows():
        name = str(r["名称"]).strip()
        if not name:
            continue
        func = str(r.get("功能","")).strip()
        if func:
            T.append((name, "has the functionality that", func))
        for it in _split(r.get("主要输入","")):
            T.append((name, "inputs", it))
        for ot in _split(r.get("主要输出","")):
            T.append((name, "outputs", ot))
        loc = str(r.get("位置(模块:函数)","")).strip()
        if loc:
            T.append((name, "impl", loc))
        phase = str(r.get("阶段","")).strip()
        if phase:
            T.append((name, "phase", phase))
        dev = str(r.get("推荐设备","")).strip()
        if dev:
            T.append((name, "device", dev))
        # 关键参数作为属性标签写入，便于检索时提示
        kp = str(r.get("关键参数说明","")).strip()
        if kp:
            T.append((name, "params", kp))
    return T

def build_llamaindex_kg(triples: List[Tuple[str,str,str]], out_dir: str):
    store = SimpleGraphStore()
    storage = StorageContext.from_defaults(graph_store=store)
    kg = KnowledgeGraphIndex([], storage_context=storage)
    for h, p, o in triples:
        kg.upsert_triplet((h,p,o), include_embeddings=(p=="has the functionality that"))
    kg.storage_context.persist(persist_dir=out_dir)
    print(f"KG persisted to {out_dir} with {len(triples)} triples.")

if __name__ == "__main__":
    df = pd.read_excel("tools.xlsx")  # 或读取你维护的CSV/YAML
    triples = triplets_from_df(df)
    Path("kg_store").mkdir(exist_ok=True)
    build_llamaindex_kg(triples, "kg_store")
```

第二组代码把你列出的工具函数包装为 LangChain 的可调用工具。包装遵循统一的输入契约：所有路径均为相对于 `RUN_WORKSPACE` 的相对路径；每个适配器在必要时补充默认输出路径，并返回结构化结果。VASP 运行通过 jobflow‑remote 的轻薄适配器完成，示例中提供了 `vasp_run_remote`，你可以将其替换为团队现有脚本。下面的注册示例覆盖你表中给出的函数名和模块位置。

```python
# tools_registry.py
import os, json, asyncio, subprocess, hashlib
from pathlib import Path
from typing import Any, Dict, Callable
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

# 计算文件哈希用于 manifest
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def ws_path(rel: str) -> Path:
    root = Path(os.environ.get("RUN_WORKSPACE", "workspace")).resolve()
    p = (root / rel).resolve()
    if root not in p.parents and p != root:
        raise ValueError("paths must be within RUN_WORKSPACE")
    return p

def write_manifest(update: Dict[str, Any]):
    root = Path(os.environ.get("RUN_WORKSPACE", "workspace")).resolve()
    m = root / "manifest.json"
    data = {}
    if m.exists():
        data = json.loads(m.read_text())
    for k, v in update.items():
        data[k] = v
    m.write_text(json.dumps(data, ensure_ascii=False, indent=2))

# 引入 catmaster 的工具实现（确保三端 PYTHONPATH 已配置）
from catmaster.tools.retrieval import lit_search as _lit_search, lit_capture as _lit_capture, matdb_query as _matdb_query
from catmaster.tools.geometry_inputs import slab_cut as _slab_cut, slab_fix as _slab_fix, adsorbate_placements as _adsorbate_placements, gas_fill as _gas_fill, vasp_prepare as _vasp_prepare
from catmaster.tools.execution import mace_relax as _mace_relax
from catmaster.tools.analysis import vasp_summarize as _vasp_summarize

# 适配 jobflow-remote 的 VASP 运行
class VaspRunSpec(BaseModel):
    work_dir: str = Field(..., description="相对工作空间的 VASP 输入目录")
    worker: str = Field(..., description="jobflow-remote worker 名称")
    exec_config: str = Field(..., description="项目中定义的执行配置名")

async def vasp_run_remote(spec: VaspRunSpec) -> Dict[str, Any]:
    run_dir = ws_path(spec.work_dir)
    cmd = [
        "jf", "job", "submit",
        "--worker", spec.worker,
        "--exec-config", spec.exec_config,
        "--run-dir", str(run_dir),
        "--entry", "bash -lc 'cd \"$RUN_WORKDIR\" && vasp_std |& tee run.log'"
    ]
    p = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await p.communicate()
    if p.returncode != 0:
        raise RuntimeError(err.decode() or out.decode())
    # 简化：演示直接返回目录，实际可在上层 monitor 节点轮询到完成
    return {"submitted": True, "work_dir": spec.work_dir, "log_hint": "run.log"}

# 将 catmaster 函数封装为 LangChain 工具
def _wrap_sync(func: Callable[..., Dict[str, Any]], name: str, desc: str, schema: BaseModel):
    async def _acall(**kwargs):
        # 允许工具内部做相对路径到绝对路径转换
        res = func(**kwargs)
        return json.dumps(res, ensure_ascii=False)
    return StructuredTool.from_function(
        coroutine=_acall,
        name=name,
        description=desc,
        args_schema=schema,
    )

# 为每个工具定义最小输入模式
class LitSearchIn(BaseModel):
    query: str
    site: str | None = None
    max_results: int = 50
    chromedriver_path: str | None = None
    headless: bool = True
    timeout: int = 30

class LitCaptureIn(BaseModel):
    target_url: str
    wait_seconds: int = 2
    chromedriver_path: str | None = None
    headless: bool = True
    timeout: int = 30

class MatDBIn(BaseModel):
    criteria: dict
    properties: list[str] | None = None
    structures_dir: str
    api_key: str | None = None

class SlabCutIn(BaseModel):
    structure_file: str
    miller_list: list[list[int]]
    min_slab_size: float
    min_vacuum_size: float
    relax_thickness: float = 0.0
    output_root: str
    get_symmetry_slab: bool = True
    fix_bottom: bool = True

class VaspPrepareIn(BaseModel):
    input_path: str
    output_root: str
    calc_type: str
    k_product: int
    calibration_temperature: float | None = None
    run_yaml_template: str | None = None
    user_incar_settings: dict | None = None

class MACEIn(BaseModel):
    structure_file: str
    work_dir: str
    device: str = "auto"
    use_d3: bool = False
    fmax: float = 0.05
    maxsteps: int = 1000
    optimizer: str = "LBFGS"
    model: str = "large"
    mode: str = "relax"

class VaspSummarizeIn(BaseModel):
    work_dir: str

TOOLS = [
    _wrap_sync(_lit_search, "lit_search", "学术搜索抓取标题与摘要", LitSearchIn),
    _wrap_sync(_lit_capture, "lit_capture", "抓取指定网页内容", LitCaptureIn),
    _wrap_sync(_matdb_query, "matdb_query", "Materials Project 查询与结构下载", MatDBIn),
    _wrap_sync(_slab_cut, "slab_cut", "体相切割晶面生成 POSCAR", SlabCutIn),
    _wrap_sync(_vasp_prepare, "vasp_prepare", "规范化生成 VASP 输入集", VaspPrepareIn),
    _wrap_sync(_mace_relax, "mace_relax", "基于 MACE 的几何预弛豫", MACEIn),
    _wrap_sync(_vasp_summarize, "vasp_summarize", "提取 VASP 目录结构化结果", VaspSummarizeIn),
    StructuredTool.from_function(
        coroutine=lambda **kw: vasp_run_remote(VaspRunSpec(**kw)),
        name="vasp_run_remote",
        description="通过 jobflow-remote 在远端运行 VASP 并记录日志",
        args_schema=VaspRunSpec,
    ),
]
```

第三组代码给出 LangGraph 的状态机外壳以及基于 LlamaIndex 的轻量规划。为方便演示，规划节点会先在知识索引中检索与“气相、VASP、总结”相关的工具，再由大模型在候选集合中排序并输出顺序链。物化节点在工作空间写入 O₂ 在盒子中的 POSCAR 文件并记录清单。执行节点循环调用工具，遇到 VASP 运行则转由 jobflow‑remote 适配器提交并进入监控节点。监控节点在实际集群上可以按你现有方式轮询状态，这里给出占位实现。

```python
# o2_flow_graph.py
import os, json, time, uuid, asyncio
from pathlib import Path
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.tools.render import render_text_description
from llama_index.core import KnowledgeGraphIndex, StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import Settings
from tools_registry import TOOLS, ws_path, write_manifest

class State(TypedDict, total=False):
    question: str
    ws: str
    run_id: str
    plan: List[str]
    step: int
    tool_inputs: List[Dict[str, Any]]
    last_output: Dict[str, Any]
    status: str

def load_kg(persist_dir="kg_store") -> KnowledgeGraphIndex:
    graph_store = SimpleGraphStore.from_persist_dir(persist_dir)
    storage = StorageContext.from_defaults(graph_store=graph_store, persist_dir=persist_dir)
    kg = KnowledgeGraphIndex.from_persist_dir(storage_context=storage)
    return kg

def write_poscar_o2(box_A: float, bond_A: float, out_rel: str):
    """最小依赖写一个 O2 POSCAR，盒子为立方，分子置于盒子中心"""
    L = box_A
    a = f"{L} 0 0"; b = f"0 {L} 0"; c = f"0 0 {L}"
    # 简单放置，原子坐标按分数坐标计算
    frac = 0.5 * bond_A / L
    x1 = 0.5 - frac; x2 = 0.5 + frac
    content = "\n".join([
        "O2_gas",
        "1.0",
        a, b, c,
        "O",
        "2",
        "Direct",
        f"{x1} 0.5 0.5",
        f"{x2} 0.5 0.5",
    ])
    p = ws_path(out_rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)

async def plan_node(state: State) -> State:
    kg = load_kg()
    # 直接从三元组中取工具名集合，再靠 LLM 过滤排序
    # 这里简化为固定候选集合，你可以改为检索 “gas”“VASP”“summarize” 的 embeddings
    candidates = ["vasp_prepare", "vasp_run_remote", "vasp_summarize"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""你是计算催化的执行规划器。目标是完成 O2 分子在盒子中的几何优化并提取最终能量。
候选工具列表：{candidates}。
请给出一个合理的顺序链，仅返回 JSON 数组形式的工具名列表，不要解释。"""
    resp = await llm.ainvoke(prompt)
    try:
        plan = json.loads(resp.content)
    except Exception:
        plan = ["vasp_prepare", "vasp_run_remote", "vasp_summarize"]
    state["plan"] = plan
    state["step"] = 0
    state["status"] = "PLANNED"
    return state

def materialize_node(state: State) -> State:
    os.environ.setdefault("RUN_WORKSPACE", state["ws"])
    # 写入 O2 POSCAR 与基本目录
    write_poscar_o2(box_A=15.0, bond_A=1.21, out_rel="o2_gas/POSCAR")
    # 初始化工具输入，计划按顺序使用工具
    tool_inputs = [
        {
            "name": "vasp_prepare",
            "payload": {
                "input_path": "o2_gas",
                "output_root": "o2_calc",
                "calc_type": "gas",
                "k_product": 1000
            }
        },
        {
            "name": "vasp_run_remote",
            "payload": {
                "work_dir": "o2_calc/O2_gas",    # 假设 vasp_prepare 为每个体系创建子目录
                "worker": os.environ.get("JFR_WORKER", "gpu_host"),
                "exec_config": os.environ.get("JFR_EXEC", "stage_inputs_vasp")
            }
        },
        {
            "name": "vasp_summarize",
            "payload": {
                "work_dir": "o2_calc/O2_gas"
            }
        }
    ]
    state["tool_inputs"] = tool_inputs
    # 更新 manifest
    write_manifest({"workspace": state["ws"], "created_by": "o2_flow"})
    state["status"] = "READY"
    return state

async def exec_step_node(state: State) -> State:
    name = state["plan"][state["step"]]
    spec = next(x for x in state["tool_inputs"] if x["name"] == name)["payload"]
    tool = next(t for t in TOOLS if t.name == name)
    out = await tool.ainvoke(spec)
    try:
        data = json.loads(out)
    except Exception:
        data = {"raw": out}
    state["last_output"] = {name: data}
    state["status"] = "STEP_DONE"
    return state

def monitor_node(state: State) -> State:
    # 如使用队列长作业，这里可轮询 jobflow-remote 状态；演示用短暂等待占位
    time.sleep(1.0)
    state["status"] = "MONITOR_OK"
    return state

def decide_node(state: State) -> State:
    state["step"] += 1
    if state["step"] >= len(state["plan"]):
        state["status"] = "COMPLETED"
        # 在完成时写出摘要
        root = Path(state["ws"])
        summary = {"final": state["last_output"]}
        (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        return state
    state["status"] = "CONTINUE"
    return state

def build_app():
    g = StateGraph(State)
    g.add_node("plan", plan_node)
    g.add_node("materialize", materialize_node)
    g.add_node("exec", exec_step_node)
    g.add_node("monitor", monitor_node)
    g.add_node("decide", decide_node)
    g.set_entry_point("plan")
    g.add_edge("plan", "materialize")
    g.add_edge("materialize", "exec")
    g.add_edge("exec", "monitor")
    g.add_edge("monitor", "decide")
    g.add_conditional_edges("decide", lambda s: END if s["status"]=="COMPLETED" else "exec")
    return g.compile()
```

第四组代码用于启动演示。它会读取环境变量，构建工作空间，调用状态机运行完整流程，并输出关键文件位置。你可以在本地先只跑到 vasp_prepare，再将 `vasp_run_remote` 的适配器替换为你们项目的正式提交流程。

```python
# run_o2_demo.py
import os, uuid, json
from pathlib import Path
from o2_flow_graph import build_app

if __name__ == "__main__":
    os.environ.setdefault("OPENAI_API_KEY", "<YOUR_KEY_HERE>")
    # 指定工作空间
    run_id = uuid.uuid4().hex[:8]
    ws = Path(f"workspace/O2_demo_{run_id}").as_posix()
    init_state = {
        "question": "Optimize O2 molecule in a cubic box and extract final energy.",
        "ws": ws,
        "run_id": run_id
    }
    app = build_app()
    final = app.invoke(init_state)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    print(f"Workspace: {ws}\nManifest: {ws}/manifest.json\nSummary: {ws}/summary.json")
```

为了使 jobflow‑remote 与 catmaster 在远端保持一致的导入环境，你可以在项目的 worker 或 exec_config 中加入固定的环境导入与 PYTHONPATH 设置。例如在 exec_config 的 pre_run 片段中添加以下内容，以便在提交脚本起跑前确保 catmaster 可用，赝势与脚本路径正确。

```yaml
# 片段：~/.jfremote/<project>.yaml
exec_config:
  stage_inputs_vasp:
    export:
      PYTHONPATH: "/opt/catmaster:${PYTHONPATH}"
      POTCAR_ROOT: "/share/vasp/potpaw_PBE"
    pre_run: |
      set -e
      echo "[pre_run] ensure env and POTCAR"
      test -d "$POTCAR_ROOT" || { echo "POTCAR missing"; exit 2; }
      # 如需从工作空间拉取输入，可在此 rsync 或解压
    post_run: |
      echo "[post_run] done"
```

关于“文件传输是否由工具承担”的界面划分，建议保持工具层只关心相对路径的工件，不直接处理跨主机传输。远端的输入准备与回传由 jobflow‑remote 的 Runner 结合 exec_config 统一管理，这样可以避免在工具层造成环境分歧，也便于重放与审计。工具层只在工作空间落盘与读取，LangGraph 的监控节点负责在作业完成后触发分析工具，例如 `vasp_summarize`，并把结果写回 `summary.json` 与清单文件。若将来需要面向多作业并行与断点恢复，上述状态机骨架可以直接扩展，不会改变工具层和知识索引的使用方式。

这套骨架在不引入额外服务的前提下，完成了工具表到知识索引的构建、在地的长流程编排、与 jobflow‑remote 的执行衔接，并提供了一个可以立即扩展的演示路径。你只需要将工具表保存为 `tools.xlsx` 并运行构建脚本，然后设置好工作空间与 jobflow‑remote 的 worker 和 exec_config，即可通过 `python run_o2_demo.py` 验证从规划到执行再到结果提取的闭环。后续你可以把文献检索阶段以相同方式加入工具表与索引，并在计划节点按条件触发，从而形成更完整的“证据与计算一体”的探索流程。

你的这段脚本已经把最难的一段路打通了：通过 SFTP 将本地资产推送到远端临时目录，在 pre_run 中就地“物化”执行模块，随后由 jobflow‑remote 提交与监控，并在结束后回传产物。现在的混乱主要来自职责没有收口在一个清晰的接口上，LLM 的编排与文件传输细节发生了耦合。我的建议是不必弃用 jobflow‑remote，而是将现有脚本收敛为一个“执行适配器工具”，把文件搬运、pre_run 注入、提交与轮询全部封装在工具内，由 LangGraph 仅负责计划、状态和重试。这样 LangGraph 完全监控执行状态，但不直接操心传输细节；jobflow‑remote 仍承担它擅长的调度职责；LLM 只看见一个受约束的工具接口而不是一堆命令碎片。

为了把这件事落到地面，最关键的是形成一个稳定的工具契约和一个薄而可复用的适配器实现。下面给出可以直接整合进 catmaster 的最小实现。它把你贴出的脚本拆分并封装为一个 Pydantic 规范和一个可复用的执行函数，随后给出 LangChain 工具包装与 LangGraph 流程。演示任务仍是 O₂ 在盒子中的优化能量。你无需让 LLM 直接操心 SFTP 或 pre_run，所有细节都在工具内部完成。

首先定义一个严格的输入规范，明确工作空间、上传源、远端临时根目录、worker 与 exec_config 名称以及入口命令。所有路径都要求相对于工作空间根目录，方便审计与重放。

```python
# catmaster/tools/execution_jfr.py
from __future__ import annotations
import os, sys, time, re, json, types, importlib, yaml, subprocess, paramiko
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from paramiko.config import SSHConfig
from paramiko.proxy import ProxyCommand
from jobflow_remote.config.base import ExecutionConfig
from jobflow_remote import submit_flow, get_jobstore
from jobflow import Flow

class JFRVaspSpec(BaseModel):
    workspace: str = Field(..., description="本地工作空间根目录")
    local_assets_dir: str = Field(..., description="相对工作空间的输入资产目录")
    project: str = Field(..., description="~/.jfremote/<project>.yaml 中的项目名")
    worker: str = Field(..., description="jobflow-remote worker 名称")
    remote_tmp_base: str = Field(..., description="远端临时上传根目录")
    entry_cmd: str = Field("mpirun -n ${SLURM_NTASKS:-1} ${VASP_STD_BIN}", description="运行入口命令")
    ssh_host_alias: str | None = Field(None, description="~/.ssh/config 中的主机别名，留空则按 worker 映射")
    download_dir: str = Field("downloads", description="相对工作空间的回传目录")
    poll_seconds: float = 2.0
    timeout_seconds: int = 36000

    @field_validator("workspace","local_assets_dir","download_dir")
    @classmethod
    def rel_ok(cls, v: str) -> str:
        if v.strip() == "": raise ValueError("路径不能为空")
        return v

def _load_worker_host(project: str, worker: str) -> str:
    cfgp = Path.home() / ".jfremote" / f"{project}.yaml"
    data = yaml.safe_load(cfgp.read_text())
    return data["workers"][worker]["host"]

def _connect_via_ssh_config(host_alias: str) -> paramiko.SSHClient:
    cfg = SSHConfig()
    looked = {}
    ssh_cfg_path = os.path.expanduser("~/.ssh/config")
    if os.path.exists(ssh_cfg_path):
        with open(ssh_cfg_path) as f:
            cfg.parse(f)
        looked = cfg.lookup(host_alias)
    hostname = looked.get("hostname", host_alias)
    username = looked.get("user")
    port     = int(looked.get("port", 22))
    keyfiles = looked.get("identityfile", [])
    proxycmd = looked.get("proxycommand")
    sock = ProxyCommand(proxycmd) if proxycmd else None

    cli = paramiko.SSHClient()
    cli.load_system_host_keys()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(hostname=hostname, port=port, username=username,
                key_filename=keyfiles or None, sock=sock,
                allow_agent=True, look_for_keys=True,
                banner_timeout=60, auth_timeout=60, timeout=60)
    return cli

def _sftp_mkdirs(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    cur = "/"
    for p in remote_dir.strip("/").split("/"):
        cur = os.path.join(cur, p)
        try: sftp.stat(cur)
        except IOError: sftp.mkdir(cur, mode=0o755)

def _sftp_upload_tree(cli: paramiko.SSHClient, local_dir: Path, remote_dir: str) -> None:
    sftp = cli.open_sftp()
    try:
        _sftp_mkdirs(sftp, remote_dir)
        for root, dirs, files in os.walk(local_dir):
            rel = os.path.relpath(root, local_dir)
            rdir = remote_dir if rel == "." else os.path.join(remote_dir, rel)
            _sftp_mkdirs(sftp, rdir)
            for d in dirs:
                _sftp_mkdirs(sftp, os.path.join(rdir, d))
            for f in files:
                sftp.put(os.path.join(root, f), os.path.join(rdir, f))
    finally:
        sftp.close()

def _ensure_inline_module(name: str, code: str):
    if name in sys.modules:
        del sys.modules[name]
    mod = types.ModuleType(name)
    mod.__file__ = f"<inline:{name}>"
    sys.modules[name] = mod
    exec(code, mod.__dict__)
    return importlib.import_module(name)

_INLINE_MODULE = "jfr_inline_vasp"
_INLINE_CODE = r'''
from jobflow import job
from pathlib import Path
import os, re, subprocess

@job
def run_vasp():
    vasp = os.environ.get("VASP_STD_BIN")
    if not vasp:
        raise RuntimeError("VASP_STD_BIN is not set")
    np = os.environ.get("SLURM_NTASKS") or "1"
    cmd = ["mpirun","-n",str(np),vasp]
    with open("vasp_stdout.txt","wb") as f:
        subprocess.run(cmd, check=False, stdout=f, stderr=subprocess.STDOUT)
    summary = {"parsed": False}
    try:
        txt = Path("OUTCAR").read_text(errors="ignore")
        m = list(re.finditer(r"free\\s+energy\\s+TOTEN\\s*=\\s*([\\-\\d\\.Ee\\+]+)", txt))
        if not m:
            m = list(re.finditer(r"\\bTOTEN\\s*=\\s*([\\-\\d\\.Ee\\+]+)", txt))
        if m:
            summary["final_energy_eV"] = float(m[-1].group(1)); summary["parsed"] = True
    except Exception as e:
        summary["error"] = str(e)
    return {"summary": summary}
'''

def _make_exec_cfg_copyin(remote_tmp_dir: str) -> ExecutionConfig:
    pre_run = f"""
set -euo pipefail
test -d "{remote_tmp_dir}" || {{ echo "missing upload dir: {remote_tmp_dir}" >&2; exit 2; }}
cp -a "{remote_tmp_dir}"/. .
rm -rf "{remote_tmp_dir}"
cat > {_INLINE_MODULE}.py <<'PY'
{_INLINE_CODE}
PY
export PYTHONPATH="$PWD${{PYTHONPATH+::$PYTHONPATH}}"
"""
    return ExecutionConfig(pre_run=pre_run)

def _wait_for_output(js, job_uuid: str, timeout: int, poll: float):
    t0 = time.time()
    done = {"COMPLETED","FINISHED","SUCCESS","DONE"}
    fail = {"FAILED","ERROR","CANCELLED","STOPPED","REJECTED","TIMEOUT","REMOTE_ERROR"}
    def _state():
        try:
            res = subprocess.run(["jf","job","info",job_uuid], check=True, text=True, capture_output=True)
            m = re.search(r"state\\s*=\\s*'([^']+)'", res.stdout, re.IGNORECASE)
            return m.group(1).strip() if m else None
        except Exception:
            return None
    while True:
        try:
            out = js.get_output(job_uuid, load=True)
            if out is not None:
                return out
        except Exception:
            pass
        st = _state()
        if st:
            up = st.upper()
            if up in fail:
                raise RuntimeError(f"job {job_uuid} failed, state={st}")
            if up in done:
                try:
                    out = js.get_output(job_uuid, load=True)
                except Exception:
                    out = None
                return out
        if time.time() - t0 > timeout:
            raise TimeoutError(f"wait {job_uuid} > {timeout}s")
        time.sleep(poll)

def vasp_run_jfr(spec: JFRVaspSpec) -> dict:
    ws = Path(spec.workspace).resolve()
    local_assets = ws / spec.local_assets_dir
    assert local_assets.is_dir(), f"missing assets: {local_assets}"
    host = spec.ssh_host_alias or _load_worker_host(spec.project, spec.worker)

    # 远端上传
    cli = _connect_via_ssh_config(host)
    try:
        # 在内存注册 job 模块并构造作业
        mod = _ensure_inline_module(_INLINE_MODULE, _INLINE_CODE)
        job = mod.run_vasp()
        remote_tmp = os.path.join(spec.remote_tmp_base.rstrip("/"), job.uuid)
        _sftp_upload_tree(cli, local_assets, remote_tmp)
    finally:
        cli.close()

    # 提交与轮询
    flow = Flow([job])
    exec_cfg = _make_exec_cfg_copyin(remote_tmp)
    db_ids = submit_flow(flow, worker=spec.worker, exec_config=exec_cfg)
    job_db_id = db_ids[0] if isinstance(db_ids,(list,tuple)) else db_ids

    js = get_jobstore(); js.connect()
    out = _wait_for_output(js, job.uuid, timeout=spec.timeout_seconds, poll=spec.poll_seconds)

    # 回传运行目录
    dest = ws / spec.download_dir / f"vasp_{job_db_id}"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["jf","job","todir", str(job_db_id), "--path", str(dest)], check=True, text=True, capture_output=True)
    except Exception:
        pass
    return {"job_db_id": job_db_id, "uuid": job.uuid, "summary": out.get("summary"), "download_dir": dest.as_posix()}
```

接下来把这个函数包装成 LangChain 工具，同时保留你已有的几何与输入准备、分析汇总工具。LLM 不再与 SFTP 或 pre_run 打交道，它只“看到”一个名为 vasp_run_jfr 的稳定接口。

```python
# catmaster/agents/tools.py
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from catmaster.tools.execution_jfr import vasp_run_jfr, JFRVaspSpec
from catmaster.tools.geometry_inputs import vasp_prepare
from catmaster.tools.analysis import vasp_summarize

class VaspPrepareIn(BaseModel):
    input_path: str
    output_root: str
    calc_type: str
    k_product: int

class VaspSummarizeIn(BaseModel):
    work_dir: str

TOOLS = [
    StructuredTool.from_function(
        name="vasp_prepare",
        description="规范化生成 VASP 输入集",
        args_schema=VaspPrepareIn,
        coroutine=lambda **kw: vasp_prepare(**kw),  # 你的实现已是同步函数，可按需包一层
    ),
    StructuredTool.from_function(
        name="vasp_run_jfr",
        description="通过 jobflow‑remote 上传、提交并运行 VASP 作业，结束后回传运行目录",
        args_schema=JFRVaspSpec,
        coroutine=lambda **kw: vasp_run_jfr(JFRVaspSpec(**kw))
    ),
    StructuredTool.from_function(
        name="vasp_summarize",
        description="从 VASP 目录提取能量与收敛结果",
        args_schema=VaspSummarizeIn,
        coroutine=lambda **kw: vasp_summarize(**kw),
    ),
]
```

最后用 LangGraph 固化“计划、物化、执行、监控、汇总”的过程。计划阶段可以用简单的提示让模型在候选工具中排出顺序，物化阶段在工作空间写入 O₂ 的 POSCAR，执行阶段顺序调用三步工具，监控阶段只做状态采样，汇总阶段写出最终报告。LangGraph 全程掌握状态，并且随时可从检查点重放某一步。

```python
# catmaster/agents/o2_flow.py
import os, json, uuid, time
from pathlib import Path
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from catmaster.agents.tools import TOOLS

class State(TypedDict, total=False):
    ws: str
    run_id: str
    plan: List[Dict[str, Any]]
    i: int
    last: Dict[str, Any]
    status: str

def write_poscar_o2(ws: Path, L=15.0, bond=1.21):
    frac = 0.5*bond/L; x1=0.5-frac; x2=0.5+frac
    p = ws / "o2_box" / "POSCAR"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join([
        "O2_gas","1.0", f"{L} 0 0", f"0 {L} 0", f"0 0 {L}",
        "O","2","Direct", f"{x1} 0.5 0.5", f"{x2} 0.5 0.5"
    ]))

async def plan_node(state: State) -> State:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = "目标是完成 O2 气相几何优化并提取能量。可用工具为 vasp_prepare、vasp_run_jfr、vasp_summarize。仅返回 JSON 队列。"
    rsp = await llm.ainvoke(prompt)
    try:
        state["plan"] = json.loads(rsp.content)
    except Exception:
        state["plan"] = [{"name":"vasp_prepare"}, {"name":"vasp_run_jfr"},{"name":"vasp_summarize"}]
    state["i"] = 0; state["status"] = "PLANNED"; return state

def materialize_node(state: State) -> State:
    ws = Path(state["ws"]).resolve()
    os.environ.setdefault("RUN_WORKSPACE", ws.as_posix())
    write_poscar_o2(ws)
    # 填充工具参数
    for step in state["plan"]:
        if step["name"] == "vasp_prepare":
            step["args"] = {"input_path":"o2_box","output_root":"o2_calc","calc_type":"gas","k_product":1000}
        if step["name"] == "vasp_run_jfr":
            step["args"] = {
                "workspace": ws.as_posix(),
                "local_assets_dir": "o2_calc/O2_gas",   # 你的 vasp_prepare 子目录命名需与此对齐
                "project": os.environ.get("JFR_PROJECT","catmaster"),
                "worker": os.environ.get("JFR_WORKER","cpu-worker"),
                "remote_tmp_base": os.environ.get("JFR_REMOTE_TMP","/public/home/chenhh/tmp/jfr_uploads"),
                "entry_cmd": "mpirun -n ${SLURM_NTASKS:-1} ${VASP_STD_BIN}",
                "download_dir": "downloads"
            }
        if step["name"] == "vasp_summarize":
            step["args"] = {"work_dir":"downloads"}  # 或者具体的子目录
    state["status"]="READY"; return state

async def exec_node(state: State) -> State:
    name = state["plan"][state["i"]]["name"]
    args = state["plan"][state["i"]]["args"]
    tool = next(t for t in TOOLS if t.name == name)
    out = await tool.ainvoke(args)
    try: data = json.loads(out) if isinstance(out, str) else out
    except Exception: data = {"raw": out}
    state["last"] = {name: data}; state["status"]="STEP_DONE"; return state

def monitor_node(state: State) -> State:
    time.sleep(1.0); state["status"]="MONITOR_OK"; return state

def decide_node(state: State) -> State:
    state["i"] += 1
    if state["i"] >= len(state["plan"]):
        Path(state["ws"]).joinpath("summary.json").write_text(json.dumps(state["last"], ensure_ascii=False, indent=2))
        state["status"]="COMPLETED"; return state
    state["status"]="CONTINUE"; return state

def build_app():
    g = StateGraph(State)
    g.add_node("plan", plan_node); g.add_node("materialize", materialize_node)
    g.add_node("exec", exec_node); g.add_node("monitor", monitor_node); g.add_node("decide", decide_node)
    g.set_entry_point("plan"); g.add_edge("plan","materialize"); g.add_edge("materialize","exec")
    g.add_edge("exec","monitor"); g.add_edge("monitor","decide")
    g.add_conditional_edges("decide", lambda s: END if s["status"]=="COMPLETED" else "exec")
    return g.compile()
```

这一收口后的形态有几个直接效果。第一，LangGraph 接管了流程的全部状态与可回放语义，检查点自然落在工作空间文件上；第二，jobflow‑remote 的使用被压缩成一个稳定工具，LLM 不再与 pre_run、SFTP 或队列细节耦合；第三，文件管理统一回到“工作空间相对路径”的契约中，工具内部自行完成上传与回传，你可以在不动 LangGraph 和 LLM 的前提下替换实现，例如将 SFTP 改成 rsync 或对象存储；第四，若未来决定切换到纯 SSH 后端，只需把 vasp_run_jfr 的实现替换为 SSH 执行器版本，工具签名与 LangGraph 编排无须改变。

因此我的结论是没有必要弃用 jobflow‑remote。你已经具备稳定的提交路径，问题只是职责边界不清导致的复杂感。将现有脚本收敛为一个工具适配器，把搬运与注入动作下沉到工具内部，用 LangGraph 统一掌控流程与检查点，LLM 只面向“工具名＋参数”这一小而稳定的接口，就能在不牺牲你已打通能力的前提下把复杂度压到最低。如果你希望，我可以把上面三个文件对齐到你的 catmaster 目录结构并按你们的命名规则微调默认路径与子目录名称，使之可以直接运行你的 O₂ 演示。
