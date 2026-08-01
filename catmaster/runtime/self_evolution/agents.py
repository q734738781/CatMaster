from __future__ import annotations

import inspect
import json
import shutil
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemPermission
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.search_surface import search_tools_for_role
from catmaster.tools.registry import get_tool_registry

from .models import (
    ProposerResult,
    ProportionalityAssessment,
    ReflectionResult,
    ReviewerResult,
    SKILL_GROUPS,
)
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
    conclusion_lines = {
        "RECOMMENDATION: APPROVE": "approve",
        "RECOMMENDATION: REJECT": "reject",
        "RECOMMENDATION: NEEDS_REVISION": "needs_revision",
        "DECISION: APPROVE": "approve",
        "DECISION: REJECT": "reject",
        "DECISION: NEEDS_REVISION": "needs_revision",
    }
    matches: list[str] = []
    for raw_line in text.splitlines():
        conclusion = conclusion_lines.get(raw_line.strip().upper())
        if conclusion:
            matches.append(conclusion)
    if len(matches) == 1:
        return ReviewerResult(
            recommendation=matches[0],
            summary="Structured human-review details were unavailable from the reviewer response.",
            proportionality_assessment=ProportionalityAssessment(
                status="warning",
                explanation="The host recovered only a textual recommendation, so proportionality needs human review.",
            ),
            concerns=["Structured change points and evidence mapping were unavailable from the reviewer response."],
            human_checks=["Inspect the exact diff and verify every consequential change against the source trace."],
            rationale=text[:2000],
        )
    return ReviewerResult(
        recommendation="reject",
        summary="The reviewer response could not be validated.",
        proportionality_assessment=ProportionalityAssessment(
            status="fail",
            explanation="No unique validated recommendation was returned.",
        ),
        concerns=["Reviewer output did not contain one valid structured or textual recommendation."],
        human_checks=["Regenerate the review before considering this candidate."],
        rationale="Reviewer did not return a unique validated recommendation line; promotion was denied.",
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
    trace: TurnTrace | None = None,
    repo_root: Path,
    revision: int = 1,
    evidence_markdown: str = "",
    owner_group: str = "",
    owner_name: str = "",
) -> Path:
    root = store.create_revision_dir(candidate_id, revision)
    complete_evidence = str(evidence_markdown or "").strip()
    if not complete_evidence and trace is not None:
        complete_evidence = trace.to_markdown().strip()
    (root / "evidence.md").write_text(
        (complete_evidence or "# Trajectory evidence\n\nNo eligible evidence was supplied.") + "\n",
        encoding="utf-8",
    )
    current = root / "current"
    current.mkdir(parents=True, exist_ok=True)
    authoring = repo_root / "skills" / "AGENTS.MD"
    if authoring.is_file():
        shutil.copyfile(authoring, current / "skill_authoring.md")
    memory = store.read_memory_text()
    (current / "AGENTS.md").write_text(memory, encoding="utf-8")
    if owner_group and owner_name:
        workspace_owner = store.self_develop_skills_dir / owner_group / owner_name
        builtin_owner = repo_root / "skills" / owner_group / owner_name
        effective_owner = workspace_owner if workspace_owner.is_dir() else builtin_owner
        if effective_owner.is_dir():
            shutil.copytree(effective_owner, current / "target")
        if builtin_owner.is_dir():
            shutil.copytree(builtin_owner, current / "builtin_target")
        else:
            (current / "builtin_absent").write_text(
                "No built-in target existed when this revision was prepared.\n",
                encoding="utf-8",
            )
    proposed_memory = root / "memories"
    proposed_memory.mkdir(parents=True, exist_ok=True)
    (proposed_memory / "AGENTS.md").write_text(memory, encoding="utf-8")
    catalog_rows = [
        "# Current skill targets",
        "",
        "Workspace-stable skills override built-in skills with the same exact group/name.",
        "",
    ]
    effective_skills: dict[tuple[str, str], tuple[Path, str]] = {}
    for skill_root, layer in (
        (repo_root / "skills", "built in"),
        (store.self_develop_skills_dir, "workspace stable"),
    ):
        if not skill_root.is_dir():
            continue
        for skill_md in skill_root.glob("*/*/SKILL.md"):
            group = skill_md.parent.parent.name
            name = skill_md.parent.name
            if group in SKILL_GROUPS:
                effective_skills[(group, name)] = (skill_md, layer)
    for (group, name), (skill_md, layer) in sorted(effective_skills.items()):
        if not skill_md.is_file():
            continue
        description = ""
        for line in skill_md.read_text(encoding="utf-8", errors="replace").splitlines()[:30]:
            if line.startswith("description:"):
                description = line.split(":", 1)[1].strip().strip("'\"")
                break
        catalog_rows.append(f"- `{group}/{name}` ({layer}): {description}")
    if not effective_skills:
        catalog_rows.append(
            "- No skill is currently available."
        )
    (current / "catalog.md").write_text("\n".join(catalog_rows) + "\n", encoding="utf-8")
    (root / "proposed").mkdir(parents=True, exist_ok=True)
    return root


class PrepareSkillInput(BaseModel):
    """Copy an existing full skill bundle, or create a complete bounded new skill from proposal content."""

    group: str = Field(..., description="Exact CatMaster skill group from the current skill tree.")
    name: str = Field(..., description="Directory-matching skill name using letters, digits, dot, underscore, or hyphen.")
    description: str = Field(
        default="",
        description=(
            "Required when no current skill exists: a substantive one-sentence description of what the new "
            "skill does and when it should activate. Leave empty when copying an existing skill."
        ),
    )
    applicability: str = Field(
        default="",
        description=(
            "Required when no current skill exists: the concrete task boundary where this method applies. "
            "Leave empty when copying an existing skill."
        ),
    )
    non_applicability: str = Field(
        default="",
        description=(
            "Required when no current skill exists: a concrete nearby case where the method must not activate. "
            "Leave empty when copying an existing skill."
        ),
    )
    expected_step_change: str = Field(
        default="",
        description=(
            "Required when no current skill exists: the specific workflow step that changes because of the "
            "candidate evidence. Leave empty when copying an existing skill."
        ),
    )


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

    def prepare_skill_candidate(
        group: str,
        name: str,
        description: str = "",
        applicability: str = "",
        non_applicability: str = "",
        expected_step_change: str = "",
    ) -> str:
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
        if source.is_dir():
            destination.parent.mkdir(parents=True, exist_ok=True)
            current_target = candidate_root / "current" / "target"
            if current_target.exists():
                shutil.rmtree(current_target)
            shutil.copytree(source, current_target)
            builtin_snapshot = candidate_root / "current" / "builtin_target"
            builtin_absent = candidate_root / "current" / "builtin_absent"
            if built_in_source.is_dir():
                if builtin_snapshot.exists():
                    shutil.rmtree(builtin_snapshot)
                shutil.copytree(built_in_source, builtin_snapshot)
                if builtin_absent.exists():
                    builtin_absent.unlink()
            elif not builtin_absent.exists():
                builtin_absent.write_text(
                    "No built-in target existed when this revision was prepared.\n",
                    encoding="utf-8",
                )
            shutil.copytree(source, destination)
            layer = "self_develop" if source == workspace_source else "built_in"
            return f"Copied the complete current {layer} bundle to /proposed/{group}/{name}. Inspect and edit it there."

        proposal_content = {
            "description": " ".join(str(description or "").split()),
            "applicability": " ".join(str(applicability or "").split()),
            "non_applicability": " ".join(str(non_applicability or "").split()),
            "expected_step_change": " ".join(str(expected_step_change or "").split()),
        }
        placeholder_markers = (
            "todo",
            "tbd",
            "placeholder",
            "replace this",
            "fill this",
            "fill in",
            "to be added",
            "to be completed",
            "待补充",
            "待完善",
            "占位",
        )
        incomplete_fields = [
            field_name
            for field_name, value in proposal_content.items()
            if len(value) < 8
            or any(marker in value.casefold() for marker in placeholder_markers)
        ]
        if incomplete_fields:
            return (
                "Cannot create a new skill because substantive proposal content is missing or placeholder-like "
                f"for: {', '.join(incomplete_fields)}. No candidate files were written."
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir(parents=True, exist_ok=True)
        builtin_absent = candidate_root / "current" / "builtin_absent"
        if not built_in_source.is_dir() and not builtin_absent.exists():
            builtin_absent.write_text(
                "No built-in target existed when this revision was prepared.\n",
                encoding="utf-8",
            )
        (destination / "SKILL.md").write_text(
            "\n".join(
                [
                    "---",
                    f"name: {name}",
                    f"description: {json.dumps(proposal_content['description'], ensure_ascii=False)}",
                    "license: project-local",
                    "compatibility: local",
                    "---",
                    f"# {name}",
                    "",
                    "## Overview",
                    "",
                    proposal_content["description"],
                    "",
                    "## Quick Start",
                    "",
                    f"- Use this skill when: {proposal_content['applicability']}",
                    f"- Do not use this skill when: {proposal_content['non_applicability']}",
                    "",
                    "## Workflow",
                    "",
                    f"1. Confirm that the task matches this boundary: {proposal_content['applicability']}",
                    f"2. Apply the bounded change: {proposal_content['expected_step_change']}",
                    "3. Preserve the user's task constraints and verify the resulting task outcome.",
                    "",
                    "## Method-critical defaults",
                    "",
                    f"- Default change: {proposal_content['expected_step_change']}",
                    f"- Stop or use the ordinary workflow when: {proposal_content['non_applicability']}",
                    "- Keep this method bounded; do not turn it into a universal task requirement.",
                    "",
                    "## Output Contract",
                    "",
                    "- State whether the applicability boundary matched.",
                    "- Report the task action and its verified outcome, with the relevant evidence reference.",
                    "- If the boundary did not match, continue without imposing this method.",
                    "",
                    "## References",
                    "",
                    "- Treat the active CatMaster tool schemas and task contract as the authoritative runtime references.",
                    "- Keep candidate-specific evidence in the self-evolution revision record rather than copying raw traces here.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return f"Created a complete bounded new skill at /proposed/{group}/{name} from the supplied proposal content."

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


class ProposerAgent:
    def __init__(
        self,
        *,
        model: Any,
        model_label: str,
        workspace: Path,
        search_tools: list[Any] | None = None,
    ) -> None:
        self.model = model
        self.model_label = str(model_label or "").strip()
        self.workspace = Path(workspace).expanduser().resolve()
        self.search_tools = list(search_tools or [])

    def reflect(
        self,
        *,
        trajectory_markdown: str,
        skill_catalog: str,
        prior_targets: list[str],
    ) -> tuple[ReflectionResult, dict[str, Any]]:
        """Read one complete trajectory and decide whether skill learning exists."""

        agent = create_agent(
            model=self.model,
            tools=[],
            system_prompt=_load_prompt("reflector"),
            response_format=ToolStrategy(ReflectionResult),
            name="self_evolution_reflector",
        )
        request = "\n".join(
            [
                "# Current skill catalog",
                "",
                skill_catalog or "No current skill catalog was available.",
                "",
                "# Existing open evidence targets",
                "",
                *(
                    [f"- `{item}`" for item in prior_targets]
                    if prior_targets
                    else ["- None."]
                ),
                "",
                trajectory_markdown,
            ]
        )
        started = time.monotonic()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": request}]},
            config={"metadata": {"lc_agent_name": "self_evolution_reflector"}},
        )
        response = _structured_response(result, ReflectionResult)
        return response, {
            "model_label": self.model_label,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "usage": _usage_metadata(result),
        }

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
                *self.search_tools,
            ],
            system_prompt=_load_prompt("proposer"),
            middleware=[
                # DeepAgents 0.6.12 passes its documented `permissions=` value
                # to this same middleware parameter internally (graph.py).
                FilesystemMiddleware(
                    backend=backend,
                    _permissions=permissions,
                    tool_token_limit_before_evict=8_000,
                    human_message_token_limit_before_evict=None,
                ),
                ToolCallLimitMiddleware(tool_name="inspect_catmaster_tool", run_limit=3, exit_behavior="continue"),
                ToolCallLimitMiddleware(tool_name="web_search", run_limit=2, exit_behavior="continue"),
            ],
            response_format=ToolStrategy(ProposerResult),
            name="self_evolution_proposer",
        )
        evidence = (candidate_root / "evidence.md").read_text(
            encoding="utf-8",
            errors="replace",
        )
        started = time.monotonic()
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Analyze the complete episode trajectories below. Treat any model attribution "
                            "as a hypothesis unless the supplied outcomes or user correction verify it. Inspect "
                            "/current/AGENTS.md, /current/catalog.md, and /current/target when relevant. A prior semantic "
                            "reflection selected one exact target, but you must return ignore if the complete evidence "
                            "does not justify changing it. Prefer a bounded edit of an existing owner over a new skill. "
                            "For memory, directly edit the candidate copy at "
                            "/memories/AGENTS.md. For a skill, call "
                            "prepare_skill_candidate once and edit the complete bundle under /proposed/<group>/<name>/. "
                            "When creating a genuinely new skill, supply substantive description, applicability, "
                            "non_applicability, and expected_step_change arguments; missing or placeholder content "
                            "must leave no candidate bundle. "
                            "Do not add generic validation or recovery obligations that the task evidence and an "
                            "explicit contract do not require. State concrete applicability and non-applicability "
                            "boundaries. Return ignore when durable learning is not clearly supported.\n\n"
                            + evidence
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
    def __init__(
        self,
        *,
        model: Any,
        model_label: str,
        workspace: Path,
        search_tools: list[Any] | None = None,
    ) -> None:
        self.model = model
        self.model_label = str(model_label or "").strip()
        self.workspace = Path(workspace).expanduser().resolve()
        self.search_tools = list(search_tools or [])

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
            tools=[_inspect_tool(), *self.search_tools],
            system_prompt=_load_prompt("reviewer"),
            middleware=[
                FilesystemMiddleware(
                    backend=backend,
                    _permissions=permissions,
                    tool_token_limit_before_evict=8_000,
                    human_message_token_limit_before_evict=None,
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
                "Read /evidence.md and inspect the exact candidate /memories/AGENTS.md or complete /proposed bundle. "
                "For memory, compare it with /current/AGENTS.md. "
                "Use /current and source/web tools when needed. Check evidence sufficiency, "
                "counterexamples, applicability boundaries, ownership, cost, "
                "and proportionality. It is valid to conclude that no candidate should proceed. Recommend approve, "
                "reject, or needs_revision for the exact files without editing them. Your recommendation is advisory: "
                "it never creates a terminal rejection and never authorizes canary or stable promotion. "
                "In addition to structured output, end any textual conclusion with exactly one line: "
                "RECOMMENDATION: APPROVE, RECOMMENDATION: REJECT, or RECOMMENDATION: NEEDS_REVISION."
            ),
            "complete_trajectory_evidence": (
                candidate_root / "evidence.md"
            ).read_text(encoding="utf-8", errors="replace"),
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
    search_scope = f"self_evolution:{uuid4().hex}"
    proposer = ProposerAgent(
        model=build_chat_model(profile.config_for_role("self_evolution_proposer")),
        model_label=proposer_label,
        workspace=workspace,
        search_tools=search_tools_for_role(
            profile,
            "self_evolution_proposer",
            workspace=workspace,
            audience="self_evolution",
            runtime_context={"search_scope": search_scope},
        ),
    )
    reviewer = ReviewerAgent(
        model=build_chat_model(profile.config_for_role("self_evolution_reviewer")),
        model_label=reviewer_label,
        workspace=workspace,
        search_tools=search_tools_for_role(
            profile,
            "self_evolution_reviewer",
            workspace=workspace,
            audience="self_evolution",
            runtime_context={"search_scope": search_scope},
        ),
    )
    return proposer, reviewer


__all__ = [
    "ProposerAgent",
    "ReviewerAgent",
    "build_self_evolution_agents",
    "prepare_candidate_workspace",
]
