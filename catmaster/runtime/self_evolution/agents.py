from __future__ import annotations

import inspect
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemPermission
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.tools.registry import get_tool_registry

from .models import ProposerResult, ReviewerResult, SKILL_GROUPS
from .storage import SelfEvolutionStore
from .trace import TurnTrace


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "self_evolution" / f"{name}.md"
    return path.read_text(encoding="utf-8").strip()


def _structured_response(result: Any, model_type: type[BaseModel]) -> BaseModel:
    value = result.get("structured_response") if isinstance(result, dict) else None
    if isinstance(value, model_type):
        return value
    if isinstance(value, dict):
        return model_type.model_validate(value)
    raise ValueError(f"agent did not return {model_type.__name__} structured output")


def _last_ai_text(result: Any) -> str:
    messages = result.get("messages") if isinstance(result, dict) else []
    for message in reversed(list(messages or [])):
        if getattr(message, "type", "") != "ai":
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text") or "") if isinstance(item, dict) else str(item)
                for item in content
            ).strip()
    return ""


def _recover_proposer_result_from_files(*, result: Any, candidate_root: Path) -> ProposerResult:
    current_memory = candidate_root / "current" / "AGENTS.md"
    proposed_memory = candidate_root / "memories" / "AGENTS.md"
    skill_files = sorted((candidate_root / "proposed").glob("*/*/SKILL.md"))
    rationale = _last_ai_text(result)[:1200] or "The proposer completed one candidate asset before returning control."
    current_text = current_memory.read_text(encoding="utf-8", errors="replace") if current_memory.is_file() else ""
    proposed_text = proposed_memory.read_text(encoding="utf-8", errors="replace") if proposed_memory.is_file() else ""
    memory_changed = proposed_memory.is_file() and proposed_text != current_text
    if memory_changed and proposed_text.strip() and not skill_files:
        return ProposerResult(action="memory", rationale=rationale)
    if len(skill_files) == 1 and not memory_changed:
        relative = skill_files[0].relative_to(candidate_root / "proposed")
        return ProposerResult(
            action="skill",
            group=relative.parts[0],
            name=relative.parts[1],
            rationale=rationale,
        )
    raise ValueError("proposer returned no structured decision and did not leave exactly one recoverable candidate asset")


def _recover_reviewer_result_from_text(result: Any) -> ReviewerResult:
    text = _last_ai_text(result)
    matches = re.findall(r"(?im)^DECISION:\s*(APPROVE|REJECT)\s*$", text)
    if len(matches) == 1:
        return ReviewerResult(decision=matches[0].lower(), rationale=text[:2000])
    return ReviewerResult(
        decision="reject",
        rationale="Reviewer did not return a unique validated DECISION line; promotion was denied.",
    )


def _usage_metadata(result: Any) -> dict[str, int]:
    totals: dict[str, int] = {}
    messages = result.get("messages") if isinstance(result, dict) else []
    for message in list(messages or []):
        metadata = getattr(message, "usage_metadata", None)
        if not isinstance(metadata, dict):
            continue
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = metadata.get(key)
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def prepare_candidate_workspace(
    *,
    store: SelfEvolutionStore,
    candidate_id: str,
    trace: TurnTrace,
    repo_root: Path,
) -> Path:
    root = store.reset_candidate_dir(candidate_id)
    (root / "trace.md").write_text(trace.to_markdown(), encoding="utf-8")
    current = root / "current"
    current.mkdir(parents=True, exist_ok=True)
    authoring = repo_root / "skills" / "AGENTS.MD"
    if authoring.is_file():
        shutil.copyfile(authoring, current / "skill_authoring.md")
    memory = store.read_memory_text()
    (current / "AGENTS.md").write_text(memory, encoding="utf-8")
    proposed_memory = root / "memories"
    proposed_memory.mkdir(parents=True, exist_ok=True)
    (proposed_memory / "AGENTS.md").write_text(memory, encoding="utf-8")
    catalog_rows = ["# Current Skill Catalog", "", "Choose by group and directory name, then call `prepare_skill_candidate`.", ""]
    effective: dict[tuple[str, str], tuple[str, Path]] = {}
    for group in SKILL_GROUPS:
        for layer, layer_root in (
            ("built_in", repo_root / "skills" / group),
            ("self_develop", store.self_develop_skills_dir / group),
        ):
            if not layer_root.is_dir():
                continue
            for skill_md in sorted(layer_root.glob("*/SKILL.md")):
                effective[(group, skill_md.parent.name)] = (layer, skill_md)
    for (group, name), (layer, skill_md) in sorted(effective.items()):
        description = ""
        for line in skill_md.read_text(encoding="utf-8", errors="replace").splitlines()[:30]:
            if line.startswith("description:"):
                description = line.split(":", 1)[1].strip().strip("'\"")
                break
        catalog_rows.append(f"- `{group}/{name}` ({layer}): {description}")
    (current / "catalog.md").write_text("\n".join(catalog_rows) + "\n", encoding="utf-8")
    (root / "proposed").mkdir(parents=True, exist_ok=True)
    return root


class PrepareSkillInput(BaseModel):
    """Create a writable full skill bundle from the current effective skill, or scaffold a new bundle."""

    group: str = Field(..., description="Exact CatMaster skill group from the current skill tree.")
    name: str = Field(..., description="Directory-matching skill name using letters, digits, dot, underscore, or hyphen.")


class InspectToolInput(BaseModel):
    """Inspect a registered CatMaster tool's final LLM schema and canonical Python source."""

    tool_name: str = Field(..., description="Exact registered builtin tool name to inspect.")


def _prepare_skill_tool(
    candidate_root: Path,
    *,
    repo_root: Path | None = None,
    self_develop_root: Path | None = None,
) -> StructuredTool:
    resolved_repo_root = Path(repo_root or Path(__file__).resolve().parents[3]).expanduser().resolve()
    resolved_self_develop_root = Path(
        self_develop_root or candidate_root / "current" / "self_develop"
    ).expanduser().resolve()

    def prepare_skill_candidate(group: str, name: str) -> str:
        group = str(group or "").strip()
        name = str(name or "").strip()
        if group not in SKILL_GROUPS:
            return "Unknown group. Choose one of: " + ", ".join(SKILL_GROUPS)
        if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in name):
            return "Invalid skill name. Use only letters, digits, dot, underscore, and hyphen."
        destination = candidate_root / "proposed" / group / name
        if destination.exists():
            return f"Candidate bundle already exists at /proposed/{group}/{name}."
        workspace_source = resolved_self_develop_root / group / name
        built_in_source = resolved_repo_root / "skills" / group / name
        source = workspace_source if workspace_source.is_dir() else built_in_source
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            current_target = candidate_root / "current" / "target"
            if current_target.exists():
                shutil.rmtree(current_target)
            shutil.copytree(source, current_target)
            shutil.copytree(source, destination)
            layer = "self_develop" if source == workspace_source else "built_in"
            return f"Copied the complete current {layer} bundle to /proposed/{group}/{name}. Inspect and edit it there."
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "SKILL.md").write_text(
            "\n".join(
                [
                    "---",
                    f"name: {name}",
                    "description: Replace this with what the skill is for and when to use it.",
                    "license: project-local",
                    "compatibility: local",
                    "---",
                    f"# {name}",
                    "",
                    "## Overview",
                    "",
                    "## Quick Start",
                    "",
                    "## Workflow",
                    "",
                    "## Method-critical defaults",
                    "",
                    "## Output Contract",
                    "",
                    "## References",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return f"Created a new skill scaffold at /proposed/{group}/{name}. Complete the full bundle before returning."

    return StructuredTool.from_function(
        func=prepare_skill_candidate,
        name="prepare_skill_candidate",
        description=PrepareSkillInput.__doc__ or "Prepare a skill candidate bundle.",
        args_schema=PrepareSkillInput,
        infer_schema=False,
    )


def _inspect_tool() -> StructuredTool:
    registry = get_tool_registry()

    def inspect_catmaster_tool(tool_name: str) -> str:
        requested = str(tool_name or "").strip()
        info = registry.get_tool_info(requested)
        if not info:
            nearby = [name for name in registry.list_tools() if requested.lower() in name.lower()][:20]
            suffix = f" Nearby names: {', '.join(nearby)}" if nearby else ""
            return f"Unknown registered tool: {requested}.{suffix}"
        schema = next(
            (item for item in registry.as_openai_tools(allowlist=[requested]) if item.get("name") == requested),
            {},
        )
        chunks = ["FINAL AGENT-VISIBLE SCHEMA", json.dumps(schema, ensure_ascii=False, separators=(",", ":"))]
        function = info.get("function") or info.get("coroutine")
        try:
            source = inspect.getsource(function)
        except Exception as exc:
            source = f"<source unavailable: {exc}>"
        chunks.extend(["TOOL FUNCTION SOURCE", source[:12_000]])
        return "\n\n".join(chunks)[:20_000]

    return StructuredTool.from_function(
        func=inspect_catmaster_tool,
        name="inspect_catmaster_tool",
        description=InspectToolInput.__doc__ or "Inspect a CatMaster tool.",
        args_schema=InspectToolInput,
        infer_schema=False,
    )


def _web_tools(*, workspace: Path) -> list[StructuredTool]:
    try:
        return get_tool_registry().as_langchain_tools(
            allowlist=["web_search"],
            workspace=str(workspace),
            audience="self_evolution",
        )
    except Exception:
        return []


class ProposerAgent:
    def __init__(self, *, model: Any, model_label: str, workspace: Path) -> None:
        self.model = model
        self.model_label = str(model_label or "").strip()
        self.workspace = Path(workspace).expanduser().resolve()

    def propose(self, *, candidate_root: Path) -> tuple[ProposerResult, dict[str, Any]]:
        backend = FilesystemBackend(root_dir=candidate_root, virtual_mode=True)
        permissions = [
            FilesystemPermission(operations=["write"], paths=["/proposed/**", "/memories/AGENTS.md"], mode="allow"),
            FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
        ]
        agent = create_agent(
            model=self.model,
            tools=[
                _prepare_skill_tool(
                    candidate_root,
                    repo_root=Path(__file__).resolve().parents[3],
                    self_develop_root=self.workspace / "metadata" / "self_evolution" / "self_develop_skills",
                ),
                _inspect_tool(),
                *_web_tools(workspace=self.workspace),
            ],
            system_prompt=_load_prompt("proposer"),
            middleware=[
                # DeepAgents 0.6.12 passes its documented `permissions=` value
                # to this same middleware parameter internally (graph.py).
                FilesystemMiddleware(
                    backend=backend,
                    _permissions=permissions,
                    tool_token_limit_before_evict=8_000,
                    human_message_token_limit_before_evict=24_000,
                ),
                ToolCallLimitMiddleware(tool_name="inspect_catmaster_tool", run_limit=3, exit_behavior="continue"),
                ToolCallLimitMiddleware(tool_name="web_search", run_limit=2, exit_behavior="continue"),
            ],
            response_format=ToolStrategy(ProposerResult),
            name="self_evolution_proposer",
        )
        started = time.monotonic()
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Analyze the completed interaction in /trace.md. Inspect /current/AGENTS.md and "
                            "/current/catalog.md only when relevant. Decide semantically whether one "
                            "durable change is justified. For memory, directly edit the candidate copy at "
                            "/memories/AGENTS.md. For a skill, call "
                            "prepare_skill_candidate once and edit the complete bundle under /proposed/<group>/<name>/. "
                            "Return ignore when durable learning is not clearly supported."
                        ),
                    }
                ]
            },
            config={"metadata": {"lc_agent_name": "self_evolution_proposer"}},
        )
        try:
            response = _structured_response(result, ProposerResult)
        except ValueError:
            response = _recover_proposer_result_from_files(result=result, candidate_root=candidate_root)
        return response, {
            "model_label": self.model_label,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "usage": _usage_metadata(result),
        }


class ReviewerAgent:
    def __init__(self, *, model: Any, model_label: str, workspace: Path) -> None:
        self.model = model
        self.model_label = str(model_label or "").strip()
        self.workspace = Path(workspace).expanduser().resolve()

    def review(
        self,
        *,
        candidate_root: Path,
        action: str,
        group: str,
        name: str,
        rationale: str,
        validation: dict[str, Any],
    ) -> tuple[ReviewerResult, dict[str, Any]]:
        backend = FilesystemBackend(root_dir=candidate_root, virtual_mode=True)
        permissions = [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]
        agent = create_agent(
            model=self.model,
            tools=[_inspect_tool(), *_web_tools(workspace=self.workspace)],
            system_prompt=_load_prompt("reviewer"),
            middleware=[
                FilesystemMiddleware(
                    backend=backend,
                    _permissions=permissions,
                    tool_token_limit_before_evict=8_000,
                    human_message_token_limit_before_evict=24_000,
                ),
                ToolCallLimitMiddleware(tool_name="inspect_catmaster_tool", run_limit=3, exit_behavior="continue"),
                ToolCallLimitMiddleware(tool_name="web_search", run_limit=2, exit_behavior="continue"),
            ],
            response_format=ToolStrategy(ReviewerResult),
            name="self_evolution_reviewer",
        )
        request = {
            "action": action,
            "group": group,
            "name": name,
            "proposer_rationale": rationale,
            "host_validation": validation,
            "instructions": (
                "Read /trace.md and inspect the exact candidate /memories/AGENTS.md or complete /proposed bundle. "
                "For memory, compare it with /current/AGENTS.md. "
                "Use /current and source/web tools when needed. Approve or reject the exact files without editing them. "
                "In addition to structured output, end any textual conclusion with exactly one line: "
                "DECISION: APPROVE or DECISION: REJECT."
            ),
        }
        started = time.monotonic()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": json.dumps(request, ensure_ascii=False, indent=2)}]},
            config={"metadata": {"lc_agent_name": "self_evolution_reviewer"}},
        )
        try:
            response = _structured_response(result, ReviewerResult)
        except ValueError:
            response = _recover_reviewer_result_from_text(result)
        return response, {
            "model_label": self.model_label,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "usage": _usage_metadata(result),
        }


def build_self_evolution_agents(
    profile: LLMProfile,
    *,
    workspace: Path,
) -> tuple[ProposerAgent, ReviewerAgent]:
    proposer_label = profile.label_for_role("self_evolution_proposer")
    reviewer_label = profile.label_for_role("self_evolution_reviewer")
    proposer = ProposerAgent(
        model=build_chat_model(profile.config_for_role("self_evolution_proposer")),
        model_label=proposer_label,
        workspace=workspace,
    )
    reviewer = ReviewerAgent(
        model=build_chat_model(profile.config_for_role("self_evolution_reviewer")),
        model_label=reviewer_label,
        workspace=workspace,
    )
    return proposer, reviewer


__all__ = [
    "ProposerAgent",
    "ReviewerAgent",
    "build_self_evolution_agents",
    "prepare_candidate_workspace",
]
