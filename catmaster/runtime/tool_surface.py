from __future__ import annotations

"""Runtime tool-surface composition for proposal/director/task_runner roles."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.tools import BaseTool
from langchain_core.tools import StructuredTool

from catmaster.runtime.run_context import RunContext
from catmaster.tools.registry import ToolRegistry

_GLOBAL_TOOL_DROP = {
    "read_file",
    "list_directory_with_sizes",
    "get_file_info",
    "write_note",
}

_PLANNING_LOCAL_TOOL_ALLOWLIST = {
    "run_literature_research",
}


@dataclass
class RuntimeToolSurface:
    proposal_tools: list[BaseTool]
    director_tools: list[BaseTool]
    fast_director_tools: list[BaseTool]
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


def _make_role_scoped_literature_tool(tool: BaseTool, *, role: str) -> BaseTool:
    if str(getattr(tool, "name", "") or "") != "run_literature_research":
        return tool

    func = getattr(tool, "func", None)
    coroutine = getattr(tool, "coroutine", None)
    if func is None and coroutine is None:
        return tool

    raw_schema = getattr(tool, "args_schema", None)
    args_schema = copy.deepcopy(raw_schema) if isinstance(raw_schema, dict) else {}
    properties = args_schema.get("properties")
    if isinstance(properties, dict):
        properties.pop("role", None)
    required = args_schema.get("required")
    if isinstance(required, list):
        args_schema["required"] = [item for item in required if item != "role"]
    if isinstance(args_schema, dict):
        args_schema.setdefault("additionalProperties", False)

    description = str(getattr(tool, "description", "") or "").strip()

    def _wrapper(runtime: object | None = None, **kwargs):
        forced_kwargs = dict(kwargs)
        forced_kwargs["role"] = role
        return func(runtime=runtime, **forced_kwargs)

    async def _awrapper(runtime: object | None = None, **kwargs):
        forced_kwargs = dict(kwargs)
        forced_kwargs["role"] = role
        return await coroutine(runtime=runtime, **forced_kwargs)

    return StructuredTool.from_function(
        func=_wrapper if func is not None else None,
        coroutine=_awrapper if coroutine is not None else None,
        name="run_literature_research",
        description=description,
        args_schema=args_schema,
        infer_schema=False,
        response_format="content_and_artifact",
    )


def _build_capability_guides(
    *,
    registry: ToolRegistry,
    task_tools: Sequence[BaseTool],
) -> tuple[str, str]:
    names = [str(getattr(tool, "name", "") or "") for tool in task_tools]
    domain_names = [name for name in names if name]
    domain_lines_full = [f"- {name}: {_tool_doc(registry, name)}" for name in domain_names]
    domain_lines_short = [f"- {name}" for name in domain_names]

    full = "\n".join(
        [
            "Task runner standard capabilities:",
            "",
            "Domain tools:",
            *(domain_lines_full or ["- (none)"]),
        ]
    )
    short = "\n".join(
        [
            "Task runner capabilities (short):",
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
    task_runner_denylist: set[str] | None = None,
    include_literature_tool: bool = True,
) -> RuntimeToolSurface:
    denylist = set(task_runner_denylist or set())
    denylist.update(_GLOBAL_TOOL_DROP)
    local_tools = registry.as_langchain_tools(
        run_dir=str(run_dir),
        workspace=str(run_context.workspace),
    )

    local_by_name = {str(getattr(tool, "name", "") or ""): tool for tool in local_tools}
    aider_tool = local_by_name.get("apply_aider_edits")
    planning_local_tools = [
        tool
        for tool in local_tools
        if include_literature_tool
        and str(getattr(tool, "name", "") or "") in _PLANNING_LOCAL_TOOL_ALLOWLIST
    ]
    proposal_local_tools = [_make_role_scoped_literature_tool(tool, role="proposal") for tool in planning_local_tools]
    director_local_tools = [_make_role_scoped_literature_tool(tool, role="director") for tool in planning_local_tools]
    fast_director_local_tools = [
        _make_role_scoped_literature_tool(tool, role="fast_director") for tool in planning_local_tools
    ]

    local_task_tools = [
        (
            _make_role_scoped_literature_tool(tool, role="task_runner")
            if include_literature_tool
            and str(getattr(tool, "name", "") or "") in _PLANNING_LOCAL_TOOL_ALLOWLIST
            else tool
        )
        for tool in local_tools
        if str(getattr(tool, "name", "") or "") not in denylist
        and (
            include_literature_tool
            or str(getattr(tool, "name", "") or "") not in _PLANNING_LOCAL_TOOL_ALLOWLIST
        )
    ]

    proposal_tools = _dedupe_tools(proposal_local_tools)
    director_tools = _dedupe_tools(
        ([aider_tool] if aider_tool is not None else [])
        + director_local_tools
    )
    fast_director_tools = _dedupe_tools(fast_director_local_tools)
    task_tools = _dedupe_tools(local_task_tools)

    guide_full, guide_short = _build_capability_guides(
        registry=registry,
        task_tools=task_tools,
    )

    return RuntimeToolSurface(
        proposal_tools=proposal_tools,
        director_tools=director_tools,
        fast_director_tools=fast_director_tools,
        task_tools=task_tools,
        task_runner_capability_guide_full=guide_full,
        task_runner_capability_guide_short=guide_short,
    )


__all__ = ["RuntimeToolSurface", "build_runtime_tool_surface"]
