from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from catmaster.agents.graph import GraphRunPolicy, GraphRunner
from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.artifact_store import ArtifactStore
from catmaster.runtime.local_tool_backend import LocalToolBackend
from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.run_context import RunContext
from catmaster.runtime.run_control import RunControl
from catmaster.runtime.run_ledger.history_reader import HistoryReader
from catmaster.runtime.run_ledger.hybrid_search import HybridRunLedgerSearcher
from catmaster.runtime.run_ledger.openrouter_embeddings import OpenRouterEmbeddings
from catmaster.runtime.run_ledger.store import RunLedgerStore
from catmaster.runtime.run_ledger.vector_index import VectorIndex
from catmaster.runtime.skills import CatMasterSkillsRuntime, SkillCatalog
from catmaster.runtime.tool_executor import ToolExecutor
from catmaster.runtime.trace_store import TraceStore
from catmaster.tools.base import system_root
from catmaster.tools.registry import ToolRegistry, get_tool_registry
from catmaster.ui.reporters import NullReporter, Reporter


@dataclass(frozen=True)
class BuiltGraphRunner:
    runner: GraphRunner
    run_context: RunContext


def build_graph_runner(
    *,
    workspace: Path,
    llm_profile: LLMProfile,
    reporter: Reporter | None,
    run_control: RunControl | None,
    project_id: str,
    run_dir: Path | None = None,
    run_policy: GraphRunPolicy | None = None,
    bind_run_control_id: bool = True,
    stream_debug_console: bool = False,
) -> BuiltGraphRunner:
    registry: ToolRegistry = get_tool_registry()
    run_ctx = RunContext.create(
        workspace=workspace,
        run_dir=run_dir,
        project_id=project_id,
        model_name=llm_profile.main.model,
        provider=llm_profile.main.provider,
        base_url=llm_profile.main.base_url,
    )
    if run_control is not None and bind_run_control_id:
        run_control.run_id = run_ctx.run_id

    memory_store = MemoryStore.create_default(workspace=workspace)
    memory_store.ensure_exists()
    tool_backend = LocalToolBackend(
        registry=registry,
        tool_executor=ToolExecutor(registry),
        artifact_store=ArtifactStore(run_ctx.run_dir),
        trace_store=TraceStore(run_ctx.run_dir),
        role="langgraph",
        workspace=workspace,
    )
    run_ledger_store = RunLedgerStore.create_default(workspace=workspace)
    embeddings = OpenRouterEmbeddings(system_root=system_root(workspace=workspace))
    vector_index = VectorIndex.create_default(workspace=workspace)
    hybrid_searcher = HybridRunLedgerSearcher(
        run_ledger_store=run_ledger_store,
        vector_index=vector_index,
        embeddings=embeddings,
    )
    history_reader = HistoryReader(
        searcher=hybrid_searcher,
        run_ledger_store=run_ledger_store,
        system_root=system_root(workspace=workspace),
        rerank_model=build_chat_model(llm_profile.config_for_role("history_reader")),
    )
    repo_root = Path(__file__).resolve().parents[2]
    skills_runtime = CatMasterSkillsRuntime(
        catalog=SkillCatalog.create_default(repo_root=repo_root)
    )
    runner = GraphRunner(
        task_runner_model=build_chat_model(llm_profile.config_for_role("task_runner")),
        proposal_model=build_chat_model(llm_profile.config_for_role("proposal")),
        director_model=build_chat_model(llm_profile.config_for_role("director")),
        memory_patch_model=build_chat_model(llm_profile.config_for_role("memory_patch")),
        registry=registry,
        memory_store=memory_store,
        run_context=run_ctx,
        reporter=reporter or NullReporter(),
        tool_backend=tool_backend,
        run_control=run_control,
        mcp_config=llm_profile.mcp,
        max_task_steps=llm_profile.agent_runtime.max_tool_calls,
        max_plan_steps=llm_profile.agent_runtime.max_tool_calls,
        recursion_limit=llm_profile.agent_runtime.recursion_limit,
        stream_debug_console=stream_debug_console,
        print_state_messages=llm_profile.agent_runtime.print_state_messages,
        run_ledger_store=run_ledger_store,
        history_reader=history_reader,
        skills_runtime=skills_runtime,
        tool_selector_model=build_chat_model(llm_profile.config_for_role("tool_selector")),
        run_policy=run_policy,
    )
    return BuiltGraphRunner(runner=runner, run_context=run_ctx)


__all__ = ["BuiltGraphRunner", "build_graph_runner"]
