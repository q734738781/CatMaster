from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

from catmaster.agents.graph import GraphRunPolicy, GraphRunner
from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.artifact_store import ArtifactStore
from catmaster.runtime.local_tool_backend import LocalToolBackend
from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.run_context import RunContext
from catmaster.runtime.run_control import RunControl
from catmaster.runtime.skills import CatMasterSkillsRuntime, SkillCatalog
from catmaster.runtime.tool_executor import ToolExecutor
from catmaster.runtime.trace_store import TraceStore
from catmaster.tools.registry import ToolRegistry, get_tool_registry
from catmaster.ui.reporters import NullReporter, Reporter

LEGACY_GRAPH_DRIVER_KIND = "legacy_graph_deprecated"
LEGACY_GRAPH_DEPRECATION_NOTICE = (
    "build_graph_runner()/GraphRunner is legacy and deprecated. "
    "Prefer build_specialist_runner()/SpecialistRunner for active development."
)


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
    warnings.warn(
        LEGACY_GRAPH_DEPRECATION_NOTICE,
        DeprecationWarning,
        stacklevel=2,
    )
    registry: ToolRegistry = get_tool_registry()
    run_ctx = RunContext.create(
        workspace=workspace,
        run_dir=run_dir,
        project_id=project_id,
        model_name=llm_profile.main.model,
        provider=llm_profile.main.provider,
        base_url=llm_profile.main.base_url,
        driver_kind=LEGACY_GRAPH_DRIVER_KIND,
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
        max_task_steps=llm_profile.agent_runtime.max_tool_calls,
        max_plan_steps=llm_profile.agent_runtime.max_tool_calls,
        recursion_limit=llm_profile.agent_runtime.recursion_limit,
        stream_debug_console=stream_debug_console,
        print_state_messages=llm_profile.agent_runtime.print_state_messages,
        skills_runtime=skills_runtime,
        tool_selector_model=build_chat_model(llm_profile.config_for_role("tool_selector")),
        run_policy=run_policy,
    )
    return BuiltGraphRunner(runner=runner, run_context=run_ctx)


__all__ = [
    "BuiltGraphRunner",
    "build_graph_runner",
    "LEGACY_GRAPH_DRIVER_KIND",
    "LEGACY_GRAPH_DEPRECATION_NOTICE",
]
