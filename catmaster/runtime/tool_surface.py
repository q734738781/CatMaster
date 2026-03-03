from __future__ import annotations

"""Runtime tool-surface composition for proposal/director/task_runner roles."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.tools import BaseTool

from catmaster.runtime.mcp_filesystem import MCPFilesystemRuntime
from catmaster.runtime.run_context import RunContext
from catmaster.tools.registry import ToolRegistry

_GLOBAL_TOOL_DROP = {
    "read_file",
    "list_directory_with_sizes",
    "get_file_info",
}


@dataclass
class RuntimeToolSurface:
    proposal_tools: list[BaseTool]
    director_tools: list[BaseTool]
    task_tools: list[BaseTool]
    task_runner_capability_guide_full: str
    task_runner_capability_guide_short: str


def _dedupe_tools(tools: Iterable[BaseTool]) -> list[BaseTool]:
    seen: set[str] = set()
    out: list[BaseTool] = []
    for tool in tools:
        name = str(getattr(tool, "name", "") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(tool)
    return out


def _tool_doc(registry: ToolRegistry, name: str) -> str:
    info = registry.get_tool_info(name)
    model = info.get("input_model") if isinstance(info, dict) else None
    doc = str(getattr(model, "__doc__", "") or "").strip()
    if not doc:
        return "Local domain tool."
    return " ".join(doc.split())


def _build_capability_guides(
    *,
    registry: ToolRegistry,
    task_tools: Sequence[BaseTool],
    mcp_fs_runtime: MCPFilesystemRuntime | None,
) -> tuple[str, str]:
    names = [str(getattr(tool, "name", "") or "") for tool in task_tools]

    filesystem_full = (
        mcp_fs_runtime.render_capability_guide(mode="full")
        if mcp_fs_runtime is not None
        else "Filesystem tools: disabled."
    )
    filesystem_short = (
        mcp_fs_runtime.render_capability_guide(mode="short")
        if mcp_fs_runtime is not None
        else "Filesystem tools: disabled."
    )

    domain_names = [name for name in names if name and name not in {"bash_exec"}]
    domain_lines_full = [f"- {name}: {_tool_doc(registry, name)}" for name in domain_names]
    domain_lines_short = [f"- {name}" for name in domain_names]

    full = "\n".join(
        [
            "Task runner standard capabilities:",
            "",
            filesystem_full,
            "",
            "Shell / external command:",
            "- bash_exec: run focused shell commands, content grep, parser invocations, and scientific binaries.",
            "",
            "Domain tools:",
            *(domain_lines_full or ["- (none)"]),
        ]
    )
    short = "\n".join(
        [
            "Task runner capabilities (short):",
            filesystem_short,
            "Shell: bash_exec",
            "Domain tools:",
            *(domain_lines_short or ["- (none)"]),
        ]
    )
    return full, short


def build_runtime_tool_surface(
    *,
    registry: ToolRegistry,
    run_context: RunContext,
    run_dir: Path,
    mcp_fs_runtime: MCPFilesystemRuntime | None,
    task_runner_denylist: set[str] | None = None,
) -> RuntimeToolSurface:
    denylist = set(task_runner_denylist or set())
    denylist.update(_GLOBAL_TOOL_DROP)
    local_tools = registry.as_langchain_tools(
        run_dir=str(run_dir),
        workspace=str(run_context.workspace),
    )

    local_by_name = {str(getattr(tool, "name", "") or ""): tool for tool in local_tools}
    bash_tool = local_by_name.get("bash_exec")
    aider_tool = local_by_name.get("apply_aider_edits")

    local_task_tools = [tool for tool in local_tools if str(getattr(tool, "name", "") or "") not in denylist]

    if mcp_fs_runtime is None:
        proposal_mcp: list[BaseTool] = []
        director_mcp: list[BaseTool] = []
        task_mcp: list[BaseTool] = []
    else:
        proposal_mcp = [
            tool
            for tool in mcp_fs_runtime.role_filtered_tools(role="proposal")
            if str(getattr(tool, "name", "") or "") not in _GLOBAL_TOOL_DROP
        ]
        director_mcp = [
            tool
            for tool in mcp_fs_runtime.role_filtered_tools(role="director")
            if str(getattr(tool, "name", "") or "") not in _GLOBAL_TOOL_DROP
        ]
        task_mcp = [
            tool
            for tool in mcp_fs_runtime.role_filtered_tools(role="task_runner")
            if str(getattr(tool, "name", "") or "") not in _GLOBAL_TOOL_DROP
        ]

    proposal_tools = _dedupe_tools(
        ([bash_tool] if bash_tool is not None else [])
        + ([aider_tool] if aider_tool is not None else [])
        + proposal_mcp
    )
    director_tools = _dedupe_tools(
        ([bash_tool] if bash_tool is not None else [])
        + ([aider_tool] if aider_tool is not None else [])
        + director_mcp
    )
    task_tools = _dedupe_tools(local_task_tools + task_mcp)

    guide_full, guide_short = _build_capability_guides(
        registry=registry,
        task_tools=task_tools,
        mcp_fs_runtime=mcp_fs_runtime,
    )

    return RuntimeToolSurface(
        proposal_tools=proposal_tools,
        director_tools=director_tools,
        task_tools=task_tools,
        task_runner_capability_guide_full=guide_full,
        task_runner_capability_guide_short=guide_short,
    )


__all__ = ["RuntimeToolSurface", "build_runtime_tool_surface"]
