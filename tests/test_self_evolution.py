from __future__ import annotations

import json
import inspect
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from catmaster.runtime import run_context as run_context_module
from catmaster.runtime.observability_store import ObservabilityStore
from catmaster.runtime.run_context import RunContext
from catmaster.runtime.self_evolution import agents as self_evolution_agents
from catmaster.runtime.self_evolution import (
    CandidateGate,
    EvolutionTraceScope,
    LearningCandidate,
    Observation,
    PromotionManager,
    ProposerResult,
    ReflectionBatch,
    ReflectionResult,
    ReviewerResult,
    SelfEvolutionCoordinator,
    SelfEvolutionStore,
    SkillRun,
    normalize_candidate_status,
)
from catmaster.runtime.self_evolution.consolidation import ConsolidationService
from catmaster.runtime.self_evolution.models import TextResult
from catmaster.runtime.self_evolution.agents import build_self_evolution_agents
from catmaster.runtime.self_evolution.gate import read_skill_frontmatter
from catmaster.runtime.self_evolution.query import EvolutionHistoryScope
from catmaster.runtime.self_evolution.storage import (
    hash_text,
    hash_tree,
)
from catmaster.runtime.self_evolution.telemetry import (
    finalize_skill_run_telemetry,
    write_skill_version_manifest,
)
from catmaster.runtime.self_evolution.trace import collect_turn_trace
from catmaster.specialists.runtime import build_specialist_runner
from catmaster.tools.base import ensure_project_space_layout, system_root
from catmaster.webui.projections.self_evolution import project_self_evolution_job
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage, ToolMessage


def test_self_evolution_uses_public_filesystem_allowlists_and_path_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    candidate_root = workspace / "metadata" / "self_evolution" / "candidates" / "candidate" / "revisions" / "1"
    candidate_root.mkdir(parents=True)
    (candidate_root / "evidence.md").write_text("# Evidence\n\nNo durable change.\n", encoding="utf-8")
    captured: list[dict] = []

    class _FakeAgent:
        def __init__(self, response: object) -> None:
            self.response = response

        def invoke(self, payload, config=None):
            _ = (payload, config)
            return {"structured_response": self.response, "messages": []}

    reflection_response = ReflectionBatch(
        items=[ReflectionResult(kind="no_change", rationale="No durable evidence.")]
    )
    proposer_response = ProposerResult(action="ignore", rationale="No durable evidence.")
    reviewer_response = ReviewerResult(
        recommendation="reject",
        summary="No supported candidate.",
    )

    def fake_create_deep_agent(**kwargs):
        captured.append(kwargs)
        response = {
            "self_evolution_reflector": reflection_response,
            "self_evolution_proposer": proposer_response,
            "self_evolution_reviewer": reviewer_response,
        }[kwargs["name"]]
        return _FakeAgent(response)

    monkeypatch.setattr(
        self_evolution_agents,
        "create_deep_agent",
        fake_create_deep_agent,
    )
    model = FakeMessagesListChatModel(responses=[AIMessage(content="unused")])
    proposer = self_evolution_agents.ProposerAgent(
        model=model,
        model_label="proposer",
        workspace=workspace,
    )
    reviewer = self_evolution_agents.ReviewerAgent(
        model=model,
        model_label="reviewer",
        workspace=workspace,
    )

    class _TraceScope:
        @staticmethod
        def tools(*, result_writer=None) -> list[object]:
            assert callable(result_writer)
            return []

    proposer.reflect(
        trajectory_markdown="# Trace index",
        skill_catalog="catalog",
        prior_targets=[],
        trace_scope=_TraceScope(),
    )
    proposer.propose(candidate_root=candidate_root)
    reviewer.review(
        candidate_root=candidate_root,
        action="memory",
        group="",
        name="",
        rationale="No durable evidence.",
        validation={},
    )

    by_name = {kwargs["name"]: kwargs for kwargs in captured}
    reflector_middleware = by_name["self_evolution_reflector"]["middleware"]
    proposer_middleware = by_name["self_evolution_proposer"]["middleware"]
    reviewer_middleware = by_name["self_evolution_reviewer"]["middleware"]
    reflector_fs = next(item for item in reflector_middleware if type(item).__name__ == "FilesystemMiddleware")
    proposer_fs = next(item for item in proposer_middleware if type(item).__name__ == "FilesystemMiddleware")
    reviewer_fs = next(item for item in reviewer_middleware if type(item).__name__ == "FilesystemMiddleware")
    assert {tool.name for tool in reflector_fs.tools} == set(
        self_evolution_agents._SELF_EVOLUTION_REVIEWER_FILESYSTEM_TOOLS
    )
    assert {tool.name for tool in proposer_fs.tools} == set(
        self_evolution_agents._SELF_EVOLUTION_PROPOSER_FILESYSTEM_TOOLS
    )
    assert {tool.name for tool in reviewer_fs.tools} == set(
        self_evolution_agents._SELF_EVOLUTION_REVIEWER_FILESYSTEM_TOOLS
    )
    assert "replaced in its entirety" in next(
        tool.description for tool in proposer_fs.tools if tool.name == "write_file"
    )
    assert "below `/proposed/`" in next(
        tool.description for tool in proposer_fs.tools if tool.name == "delete"
    )
    assert sum(
        type(item).__name__ == "_SelfEvolutionFilesystemGuardMiddleware"
        for item in proposer_middleware
    ) == 1
    assert sum(
        type(item).__name__ == "_SelfEvolutionFilesystemGuardMiddleware"
        for item in reviewer_middleware
    ) == 1
    for name, kwargs in by_name.items():
        assert kwargs["backend"] is next(
            item.backend
            for item in kwargs["middleware"]
            if type(item).__name__ == "FilesystemMiddleware"
        )
        assert sum(
            item.name == "SummarizationMiddleware"
            for item in kwargs["middleware"]
        ) == 1
        assert kwargs["subagents"][0]["name"] == "general-purpose"
        assert kwargs["subagents"][0]["middleware"][0].backend is kwargs["backend"]
        assert name.startswith("self_evolution_")
    assert list(
        (workspace / "metadata" / "self_evolution" / "agent_context").iterdir()
    ) == []


def test_self_evolution_deepagent_final_readonly_surface_is_bounded(
    tmp_path: Path,
) -> None:
    class _BindableFakeModel(FakeMessagesListChatModel):
        bound_surfaces: ClassVar[list[set[str]]] = []

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            _ = (tool_choice, kwargs)
            self.bound_surfaces.append({tool.name for tool in tools})
            return self

    def query_probe(sql: str) -> str:
        return sql

    trace_tool = StructuredTool.from_function(
        func=query_probe,
        name="query_evolution_trace_sql",
        description="Read one authorized trace query.",
    )
    workspace = tmp_path / "workspace"
    model = _BindableFakeModel(responses=[AIMessage(content="ready")])
    _BindableFakeModel.bound_surfaces = []

    with self_evolution_agents._self_evolution_backend(
        workspace=workspace,
        role="surface-test",
    ) as backend:
        agent = self_evolution_agents._build_self_evolution_deep_agent(
            model=model,
            backend=backend,
            tools=[trace_tool],
            investigator_tools=[trace_tool],
            system_prompt="Inspect the trace.",
            response_schema=ReflectionBatch,
            name="self_evolution_surface_test",
            filesystem_tools=self_evolution_agents._SELF_EVOLUTION_REVIEWER_FILESYSTEM_TOOLS,
            allow_mutations=False,
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Inspect."}]},
            config=self_evolution_agents._self_evolution_invoke_config(
                "self_evolution_surface_test"
            ),
        )

    assert result["messages"][-1].content == "ready"
    main_surface = next(
        surface
        for surface in _BindableFakeModel.bound_surfaces
        if "task" in surface
    )
    assert main_surface == {
        "ReflectionBatch",
        "query_evolution_trace_sql",
        "ls",
        "read_file",
        "glob",
        "grep",
        "task",
    }
    assert list(
        (workspace / "metadata" / "self_evolution" / "agent_context").iterdir()
    ) == []


def test_self_evolution_path_guard_preserves_candidate_memory_and_readonly_review() -> None:
    proposer_guard = self_evolution_agents._SelfEvolutionFilesystemGuardMiddleware(
        allow_mutations=True
    )
    reviewer_guard = self_evolution_agents._SelfEvolutionFilesystemGuardMiddleware(
        allow_mutations=False
    )

    def handler(request: object) -> ToolMessage:
        tool_call = request.tool_call
        return ToolMessage(
            content="allowed",
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
            status="success",
        )

    def call(guard: object, tool: str, path: str) -> ToolMessage:
        return guard.wrap_tool_call(
            SimpleNamespace(
                tool_call={
                    "name": tool,
                    "args": {"file_path": path},
                    "id": f"{tool}:{path}",
                }
            ),
            handler,
        )

    assert call(proposer_guard, "write_file", "/proposed/group/skill/SKILL.md").status == "success"
    assert call(proposer_guard, "edit_file", "/memories/AGENTS.md").status == "success"
    assert call(proposer_guard, "delete", "/proposed/group/skill").status == "success"
    assert call(proposer_guard, "delete", "/memories/AGENTS.md").status == "error"
    assert call(proposer_guard, "delete", "/proposed").status == "error"
    assert call(proposer_guard, "write_file", "/current/AGENTS.md").status == "error"
    assert call(reviewer_guard, "write_file", "/proposed/group/skill/SKILL.md").status == "error"


@pytest.mark.parametrize(
    ("provider", "expects_native"),
    [("codex_oauth", True), ("openai", True), ("langchain", False)],
)
def test_self_evolution_search_uses_shared_provider_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    expects_native: bool,
) -> None:
    class _Profile:
        def config_for_role(self, role: str) -> SimpleNamespace:
            return SimpleNamespace(
                model=f"{role}-model",
                provider=provider,
                base_url=None,
            )

        def label_for_role(self, role: str) -> str:
            return f"{role}-label"

    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace)
    monkeypatch.setattr(
        "catmaster.runtime.self_evolution.agents.build_chat_model",
        lambda config: {"model": config.model},
    )

    proposer, reviewer = build_self_evolution_agents(
        _Profile(),
        workspace=workspace,
    )

    for agent in (proposer, reviewer):
        assert len(agent.search_tools) == 1
        if expects_native:
            assert agent.search_tools == [{"type": "web_search"}]
        else:
            assert isinstance(agent.search_tools[0], StructuredTool)
            assert agent.search_tools[0].name == "web_search"


def _skill_text(name: str, marker: str) -> str:
    return "\n".join(
        [
            "---",
            f"name: {name}",
            f"description: {marker} surface termination repair selection workflow.",
            "license: project-local",
            "compatibility: local",
            "---",
            f"# {name}",
            "",
            "## Overview",
            marker,
            "",
            "## Quick Start",
            "Use the explicit surface index only for a matching termination-selection failure.",
            "",
            "## Workflow",
            "Repair the bounded surface termination decision.",
            "",
            "## Method-critical defaults",
            "Do not add checksum or recovery-grade QC to ordinary work.",
            "",
            "## Output Contract",
            "Return the selected termination and evidence.",
            "",
            "## References",
            "No external references.",
            "",
        ]
    )


def _write_skill(root: Path, *, group: str, name: str, marker: str) -> Path:
    path = root / group / name if group else root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(_skill_text(name, marker), encoding="utf-8")
    return path


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "skills").mkdir(parents=True)
    (root / "skills" / "AGENTS.MD").write_text(
        "# Skill authoring\n\nKeep changes bounded.\n",
        encoding="utf-8",
    )
    return root


def _run_dir(
    workspace: Path,
    run_id: str,
    *,
    prompt: str = "Complete the task.",
    status: str = "done",
    final_answer: str = "",
    resume_guidance: str = "",
) -> Path:
    path = workspace / "metadata" / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "run_state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "thread_id": "",
                "entrypoint": "experiment",
                "status": status,
                "user_prompt": prompt,
                "summary": "Task finished." if status == "done" else "Task failed.",
                "final_answer": final_answer,
                "resume_guidance": resume_guidance,
            }
        ),
        encoding="utf-8",
    )
    (path / "meta.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    _record_run_event(
        path,
        name="RUN_START",
        ts=0.0,
        payload={"entrypoint": "experiment", "user_prompt": prompt},
    )
    return path


def _record_run_event(
    run_dir: Path,
    *,
    name: str,
    payload: dict,
    source: str = "test",
    ts: float = 1.0,
) -> None:
    ObservabilityStore(run_dir).record_event(
        source=source,
        channel="test",
        name=name,
        category="test",
        ts=ts,
        seq=None,
        run_id=run_dir.name,
        task_id="",
        step_id=None,
        payload=payload,
    )


class _MemoryProposer:
    def __init__(self, suffix: str = "- Prefer Chinese reports.\n") -> None:
        self.suffix = suffix

    def reflect(self, *, trace_scope, **_kwargs):
        return (
            ReflectionResult(
                kind="workspace_preference",
                name="report_language",
                change="Prefer Chinese for future workspace reports.",
                evidence_refs=_all_event_refs(trace_scope),
                rationale="The user explicitly established a durable preference.",
            ),
            {},
        )

    def propose(self, *, candidate_root: Path):
        path = candidate_root / "memories" / "AGENTS.md"
        current = path.read_text(encoding="utf-8")
        path.write_text(current.rstrip() + "\n" + self.suffix, encoding="utf-8")
        return (
            ProposerResult(
                action="memory",
                rationale="The user explicitly requested a durable workspace preference.",
                delta_operation="merge",
                applicability_boundary=["Future reports in this workspace."],
                non_applicability=["Other workspaces and verbatim source quotations."],
                expected_step_change="Use Chinese for future workspace reports.",
            ),
            {},
        )


class _SkillProposer:
    def __init__(
        self,
        *,
        group: str = "materials_worker",
        name: str = "surface-repair",
        marker: str = "candidate revision",
        reflection_kind: str = "skill_revision",
        proposed_group: str = "",
        proposed_name: str = "",
    ) -> None:
        self.group = group
        self.name = name
        self.marker = marker
        self.reflection_kind = reflection_kind
        self.proposed_group = proposed_group or group
        self.proposed_name = proposed_name or name

    def reflect(self, *, trace_scope, **_kwargs):
        return (
            ReflectionResult(
                kind=self.reflection_kind,
                group=self.group,
                name=self.name,
                change=(
                    "Repair the explicit surface-index decision only for a "
                    "matching termination-selection failure."
                ),
                evidence_refs=_all_event_refs(trace_scope),
                rationale="The complete trajectory demonstrates a missing bounded instruction.",
            ),
            {},
        )

    def propose(self, *, candidate_root: Path):
        _write_skill(
            candidate_root / "proposed",
            group=self.proposed_group,
            name=self.proposed_name,
            marker=self.marker,
        )
        return (
            ProposerResult(
                action="skill",
                group=self.proposed_group,
                name=self.proposed_name,
                rationale="Repeated verified failures and a counterexample support a narrow owner-skill amendment.",
                delta_operation="merge",
                applicability_boundary=["Verified surface termination-selection failures."],
                non_applicability=["Ordinary analysis without a termination-selection failure."],
                expected_step_change="Repair the explicit surface-index decision without adding general QC.",
            ),
            {},
        )


class _Reviewer:
    def __init__(self, recommendation: str = "approve") -> None:
        self.recommendation = recommendation

    def review(self, **_kwargs):
        return (
            ReviewerResult(
                recommendation=self.recommendation,
                summary="A bounded change with explicit non-applicability.",
                evidence_sufficiency="Independent supporting and counterexample evidence is present.",
                scope_assessment="The scope is limited to the observed decision.",
                counterexamples=["Ordinary analysis must not activate the skill."],
                concerns=[],
                human_checks=[],
                rationale="The evidence and bounded scope support ordinary activation.",
            ),
            {},
        )


def _all_event_refs(trace_scope) -> list[str]:
    result = trace_scope.execute(
        "SELECT run_id, id FROM trajectory_events ORDER BY run_id, id"
    )
    return [
        f"run:{row['run_id']}#event:{row['id']}"
        for row in result["rows"]
    ]


class _SequenceReviewer:
    def __init__(self, *recommendations: str) -> None:
        self.recommendations = list(recommendations)

    def review(self, **_kwargs):
        return _Reviewer(self.recommendations.pop(0)).review()


class _FailingReviewer:
    def review(self, **_kwargs):
        raise RuntimeError("review service unavailable")


class _NoChangeProposer:
    def __init__(self, kind: str = "no_change") -> None:
        self.kind = kind
        self.trajectories: list[str] = []

    def reflect(self, *, trajectory_markdown: str, **_kwargs):
        self.trajectories.append(trajectory_markdown)
        return ReflectionResult(
            kind=self.kind,
            rationale="The complete episode does not justify a durable SOP update.",
        ), {}

    def propose(self, **_kwargs):
        raise AssertionError("non-actionable reflection must not invoke proposal")


def _explicit_preference_candidate(
    workspace: Path,
    repo: Path,
    *,
    reviewer=None,
    proposer=None,
) -> tuple[SelfEvolutionCoordinator, LearningCandidate]:
    run_dir = _run_dir(
        workspace,
        "run-preference",
        prompt="Use Chinese for future workspace reports.",
    )
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        mode="observe",
        repo_root=repo,
        proposer=proposer or _MemoryProposer(),
        reviewer=reviewer or _Reviewer(),
    )
    coordinator.enqueue_explicit_learn(
        run_id="run-preference",
        run_dir=run_dir,
        thread_id="thread-a",
        note="From now on, always prefer Chinese for reports in this workspace.",
    )
    job = coordinator.process_pending_jobs()[0]
    candidate = coordinator.store.read_candidate(job.candidate_id)
    assert candidate is not None
    return coordinator, candidate


def _recurrent_skill_candidate(
    workspace: Path,
    repo: Path,
    *,
    reviewer=None,
    proposer=None,
) -> tuple[SelfEvolutionCoordinator, LearningCandidate]:
    _write_skill(
        repo / "skills",
        group="materials_worker",
        name="surface-repair",
        marker="surface termination repair selection failure explicit index",
    )
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        mode="observe",
        repo_root=repo,
        proposer=proposer or _SkillProposer(),
        reviewer=reviewer or _Reviewer(),
    )
    claim = "surface termination repair selection failure needs an explicit surface index"
    inputs = [
        ("run-fail-a", "thread-a", "error", "verifier:run-fail-a", "supporting"),
        ("run-fail-b", "thread-b", "error", "verifier:run-fail-b", "supporting"),
        ("run-success-c", "thread-b", "done", "outcome:surface-control", "counterexample"),
    ]
    for run_id, thread_id, status, outcome_ref, role in inputs:
        run_dir = _run_dir(workspace, run_id, prompt=claim, status=status)
        coordinator.enqueue_post_run(
            run_id=run_id,
            thread_id=thread_id,
            terminal_status=status,
            run_dir=run_dir,
            payload={
                "learning_claim": claim,
                "task_outcome": "failure" if status == "error" else "success",
                "outcome_ref": outcome_ref,
                "evidence_role": role,
            },
        )
        coordinator.process_pending_jobs()
    candidates = coordinator.store.list_candidates()
    assert len(candidates) == 1
    return coordinator, candidates[0]


def _actual_owner_candidate(
    workspace: Path,
    *,
    run_id: str,
    note: str,
    marker: str,
) -> tuple[SelfEvolutionCoordinator, LearningCandidate]:
    repo = Path(__file__).resolve().parents[1]
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        mode="observe",
        repo_root=repo,
        proposer=_SkillProposer(
            name="surface-and-termination-screening",
            marker=marker,
        ),
        reviewer=_Reviewer(),
    )
    run_dir = _run_dir(workspace, run_id, prompt=note)
    coordinator.enqueue_explicit_learn(
        run_id=run_id,
        run_dir=run_dir,
        thread_id=f"thread-{run_id}",
        note=note,
    )
    job = coordinator.process_pending_jobs()[0]
    candidate = coordinator.store.read_candidate(job.candidate_id)
    assert candidate is not None and candidate.status == "review"
    assert candidate.name == "surface-and-termination-screening"
    return coordinator, candidate


def test_store_uses_only_four_domain_tables_and_minimal_observation_columns(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = SelfEvolutionStore(workspace, project_id="demo")
    with sqlite3.connect(store.db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(observations)")
        }
        job_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(jobs)")
        }
        skill_run_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(skill_runs)")
        }
    assert tables == {"jobs", "observations", "candidates", "skill_runs"}
    assert columns == {
        "observation_id",
        "run_id",
        "thread_id",
        "job_id",
        "episode_id",
        "item_ref",
        "signal_kind",
        "target",
        "resolved_target",
        "claim",
        "evidence_refs_json",
        "outcome_ref",
        "status",
        "created_at",
    }
    assert not {"confidence", "importance", "model", "tokens", "checksum"} & columns
    assert {"payload_json", "owner", "lease_until", "updated_at"} <= job_columns
    assert not {"heartbeat_at", "input_ref"} & job_columns
    assert not {
        "source_event_count",
        "source_min_event_id",
        "source_max_event_id",
    } & skill_run_columns


def test_workspace_memory_decoder_does_not_silently_replace_non_object_values() -> None:
    with pytest.raises(TypeError, match="must decode to a JSON object"):
        SelfEvolutionStore._decode_memory_value('"not-a-memory-record"')


def test_corrupt_job_json_remains_an_exact_read_diagnostic(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(workspace, "run-corrupt-job-json")
    store = SelfEvolutionStore(workspace, project_id="demo")
    job = store.enqueue_job(
        trigger_kind="post_run",
        run_id="run-corrupt-job-json",
        run_dir=run_dir,
    )
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "UPDATE jobs SET outcome_json = '' WHERE job_id = ?",
            (job.job_id,),
        )
    reread = store.read_job(job.job_id)
    assert reread is not None
    assert reread.outcome["_read_error_type"] == "JSONDecodeError"
    assert "cannot decode" in reread.outcome["_read_error"]


def test_unreadable_memory_candidate_is_a_concrete_gate_error(tmp_path: Path) -> None:
    store = SelfEvolutionStore(tmp_path / "workspace", project_id="demo")
    candidate = LearningCandidate(
        candidate_id="candidate-invalid-memory-encoding",
        project_id="demo",
        run_id="run",
        thread_id="thread",
        action="memory",
    )
    root = store.reset_candidate_dir(candidate.candidate_id)
    memory = root / "memories" / "AGENTS.md"
    memory.parent.mkdir(parents=True)
    memory.write_bytes(b"\xff\xfe")

    report = CandidateGate(store).run(candidate)

    assert report.valid is False
    assert report.repair_required is True
    assert any("UnicodeDecodeError" in error for error in report.errors)


@pytest.mark.parametrize(
    "unsupported",
    ["proposed", "approved", "reviewed", "promoted", "rolled_back"],
)
def test_candidate_status_rejects_abandoned_lifecycle_values(
    unsupported: str,
) -> None:
    with pytest.raises(ValueError, match="unsupported candidate status"):
        normalize_candidate_status(unsupported)


def test_store_uses_filesystem_safe_sqlite_journal_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CATMASTER_WORKSPACE_SQLITE_JOURNAL_MODE", "DELETE")
    store = SelfEvolutionStore(tmp_path / "workspace", project_id="demo")
    with store._connect() as connection:
        mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    assert mode == "delete"


def test_skill_run_reports_bounded_actual_use_without_claiming_causal_credit() -> None:
    presented_only = SkillRun(
        run_id="run-presented",
        skill_name="materials_worker/surface-repair",
        skill_version="sec_test@r0001",
        presented=True,
    )
    read = SkillRun(
        run_id="run-read",
        skill_name="materials_worker/surface-repair",
        skill_version="sec_test@r0001",
        presented=True,
        read=True,
    )
    assert presented_only.to_dict()["used"] is False
    assert read.to_dict()["used"] is True


def test_jobs_keep_one_payload_copy_and_enforce_same_owner_finish(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(workspace, "run-one")
    store = SelfEvolutionStore(workspace, project_id="demo")
    job = store.enqueue_job(
        trigger_kind="explicit_learn",
        run_id="run-one",
        run_dir=run_dir,
        payload={"note": "durable preference"},
    )
    assert job.payload == {"note": "durable preference"}
    assert not (store.root / "job_inputs").exists()
    claimed = store.claim_jobs(owner="worker-a", lease_seconds=60)
    assert len(claimed) == 1 and claimed[0].owner == "worker-a"
    with pytest.raises(RuntimeError, match="lease is no longer owned"):
        store.finish_job(claimed[0], status="done", owner="worker-b")
    finished = store.finish_job(claimed[0], status="done", owner="worker-a")
    assert finished.status == "done"
    assert finished.payload == {"note": "durable preference"}


def test_job_error_count_tracks_only_unretried_terminal_leaves(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(workspace, "run-error-leaf")
    store = SelfEvolutionStore(workspace, project_id="demo")
    original = store.enqueue_job(
        trigger_kind="post_run",
        run_id="run-error-leaf",
        run_dir=run_dir,
    )
    claimed = store.claim_jobs(owner="worker-original", project_id="demo")[0]
    failed = store.finish_job(
        claimed,
        status="error",
        error="The first attempt failed.",
        owner="worker-original",
    )
    assert failed.job_id == original.job_id
    assert store.unresolved_job_error_count(project_id="demo") == 1

    store.enqueue_job(
        trigger_kind="selected_retry",
        run_id="run-error-leaf",
        run_dir=run_dir,
        payload={"retry": "whole-job"},
        predecessor_job_id=failed.job_id,
    )
    assert store.unresolved_job_error_count(project_id="demo") == 0
    retry = store.claim_jobs(owner="worker-retry", project_id="demo")[0]
    store.finish_job(retry, status="done", owner="worker-retry")

    assert store.unresolved_job_error_count(project_id="demo") == 0


def test_selected_retry_identity_tracks_each_immutable_predecessor(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(workspace, "run-retry-chain")
    store = SelfEvolutionStore(workspace, project_id="demo")
    original = store.enqueue_job(
        trigger_kind="post_run",
        run_id="run-retry-chain",
        run_dir=run_dir,
        episode_id="episode-retry-chain",
    )

    first_retry = store.enqueue_job(
        trigger_kind="selected_retry",
        run_id=original.run_id,
        run_dir=run_dir,
        episode_id=original.episode_id,
        predecessor_job_id=original.job_id,
    )
    duplicate_first_retry = store.enqueue_job(
        trigger_kind="selected_retry",
        run_id=original.run_id,
        run_dir=run_dir,
        episode_id=original.episode_id,
        predecessor_job_id=original.job_id,
    )
    second_retry = store.enqueue_job(
        trigger_kind="selected_retry",
        run_id=original.run_id,
        run_dir=run_dir,
        episode_id=original.episode_id,
        predecessor_job_id=first_retry.job_id,
    )

    assert duplicate_first_retry.job_id == first_retry.job_id
    assert second_retry.job_id != first_retry.job_id
    assert second_retry.predecessor_job_id == first_retry.job_id
    assert len(store.list_jobs()) == 3


def test_restart_requeues_only_expired_leases(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(workspace, "run-one")
    store = SelfEvolutionStore(workspace, project_id="demo")
    store.enqueue_job(trigger_kind="post_run", run_id="run-one", run_dir=run_dir)
    claimed = store.claim_jobs(owner="worker-a", lease_seconds=600)[0]
    assert store.requeue_expired_jobs() == 0
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "UPDATE jobs SET lease_until = '2000-01-01T00:00:00+00:00' WHERE job_id = ?",
            (claimed.job_id,),
        )
    assert store.requeue_expired_jobs() == 1
    assert store.list_jobs()[0].status == "queued"


def test_trace_preserves_complete_semantic_events_and_excludes_transport_duplicates(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(
        workspace,
        "run-one",
        status="error",
        final_answer="The final answer retained the repaired output.",
        resume_guidance="Continue from the approved repair boundary.",
    )
    for index in range(12):
        _record_run_event(
            run_dir,
            name="LLM_RAW_RESPONSE",
            ts=float(index + 1),
            payload={
                "callback_run_id": f"llm-{index}",
                "generations": [
                    {
                        "reasoning_text": f"reasoning-{index}",
                        "response_text": f"complete-model-result-{index}",
                        "parsed_tool_calls": [],
                        "response_content_raw": (
                            [{"type": "text", "text": "raw-content-block"}]
                            if index == 0
                            else []
                        ),
                        "invalid_tool_calls": (
                            [{"name": "broken_call", "error": "invalid arguments"}]
                            if index == 0
                            else []
                        ),
                    }
                ],
            },
        )
    _record_run_event(
        run_dir,
        name="LLM_RAW_REQUEST",
        ts=20,
        payload={"messages": ["TRANSPORT-DUPLICATE-SHOULD-NOT-APPEAR"]},
    )
    _record_run_event(
        run_dir,
        name="TASK_DECISION",
        ts=20.5,
        payload={"decision": "repair", "reason": "tool result contradicted the first branch"},
    )
    long_result = "full-tool-result-" + ("x" * 6_000)
    raw_tool_result = {
        "content": long_result,
        "artifact": {"complete_detail": "raw-artifact-content"},
    }
    _record_run_event(
        run_dir,
        name="TOOL_RAW_OUTPUT",
        ts=21,
        payload={
            "callback_run_id": "tool-1",
            "tool": "surface_select",
            "status": "success",
            "projection": {"content_text": "condensed projection"},
            "raw_output": raw_tool_result,
        },
    )
    trace = collect_turn_trace(run_dir=run_dir, include_events=False)
    scope = EvolutionTraceScope({"run-one": (run_dir, trace)})
    queried = scope.execute(
        "SELECT id, name, payload_json FROM trajectory_events ORDER BY id"
    )
    serialized = json.dumps(queried, ensure_ascii=False)
    markdown = trace.to_markdown()
    assert trace.events == []
    assert queried["row_count"] == 15
    assert "complete-model-result-0" in serialized
    assert "complete-model-result-11" in serialized
    assert "raw-content-block" in serialized
    assert "broken_call" in serialized
    assert long_result not in serialized
    assert "raw-artifact-content" not in serialized
    assert "condensed projection" in serialized
    assert "tool result contradicted the first branch" in serialized
    assert "TRANSPORT-DUPLICATE-SHOULD-NOT-APPEAR" not in serialized
    assert "Continue from the approved repair boundary." in markdown
    assert "The final answer retained the repaired output." in markdown
    assert trace.run_id == "run-one"


def test_execution_error_and_unreferenced_task_end_are_not_verified_outcomes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(workspace, "run-error", status="error")
    _record_run_event(
        run_dir,
        name="TASK_END",
        payload={"outcome": "failure", "summary": "The task failed."},
    )

    trace = collect_turn_trace(run_dir=run_dir)

    assert trace.status == "error"
    assert trace.task_outcome == ""
    assert trace.outcome_ref == ""

    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path),
        proposer=_NoChangeProposer(),
        reviewer=_Reviewer(),
    )
    coordinator.enqueue_post_run(
        run_id="run-error",
        thread_id="thread-a",
        terminal_status="error",
        run_dir=run_dir,
        payload={"learning_claim": "Do not infer a task verdict from process failure."},
    )
    job = coordinator.process_pending_jobs()[0]
    assert coordinator.store.list_observations() == []


def test_formal_task_end_requires_explicit_verifier_reference(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(workspace, "run-verified", status="done")
    _record_run_event(
        run_dir,
        name="TASK_END",
        source="host_verifier",
        payload={
            "task_outcome": "success",
            "outcome_ref": "verifier:run-verified",
            "summary": "Host verifier passed.",
        },
    )

    trace = collect_turn_trace(run_dir=run_dir)

    assert trace.task_outcome == "verified_success"
    assert trace.outcome_ref == "verifier:run-verified"


def test_complete_failure_and_repair_trajectory_reaches_reflector_and_candidate(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(
        workspace,
        "run-repaired",
        prompt="Repair the failed surface selection call.",
        status="done",
    )
    _record_run_event(
        run_dir,
        name="TOOL_RAW_INPUT",
        ts=1,
        payload={
            "callback_run_id": "call-failed",
            "tool": "surface_select",
            "params_compact": '{"surface_index": 4}',
        },
    )
    _record_run_event(
        run_dir,
        name="TOOL_CALL_END",
        ts=2,
        payload={
            "callback_run_id": "call-failed",
            "tool": "surface_select",
            "status": "error",
            "error": "Index 4 is outside the generated set.",
        },
    )
    _record_run_event(
        run_dir,
        name="TOOL_RAW_INPUT",
        ts=3,
        payload={
            "callback_run_id": "call-repair",
            "tool": "surface_select",
            "params_compact": '{"surface_index": 2}',
        },
    )
    _record_run_event(
        run_dir,
        name="TOOL_CALL_END",
        ts=4,
        payload={
            "callback_run_id": "call-repair",
            "tool": "surface_select",
            "status": "success",
            "projection": {"content_preview": "Selected surface 2."},
        },
    )
    proposer = _SkillProposer()
    repo = _repo(tmp_path)
    _write_skill(
        repo / "skills",
        group="materials_worker",
        name="surface-repair",
        marker="current surface repair workflow",
    )
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=repo,
        proposer=proposer,
        reviewer=_Reviewer(),
    )
    coordinator.enqueue_post_run(
        run_id="run-repaired",
        thread_id="thread-repair",
        terminal_status="done",
        run_dir=run_dir,
    )
    job = coordinator.process_pending_jobs()[0]

    observations = coordinator.store.list_observations()
    assert len(observations) == 1
    assert observations[0].signal_kind == "skill_revision"
    assert observations[0].target == "materials_worker/surface-repair"
    candidate = coordinator.store.list_candidates()[0]
    evidence = (
        coordinator.store.revision_dir(candidate.candidate_id, candidate.revision)
        / "evidence.md"
    ).read_text(encoding="utf-8")
    assert "Index 4 is outside the generated set." not in evidence
    assert "Selected surface 2." not in evidence
    assert "run:run-repaired#event:" in evidence
    assert "not copied into this file" in evidence


def test_successful_terminal_run_does_not_create_observation_or_candidate(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    repo = _repo(tmp_path)
    proposer = _NoChangeProposer()
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=repo,
        proposer=proposer,
        reviewer=_Reviewer(),
    )
    run_dir = _run_dir(workspace, "run-one", prompt="The agent voluntarily computed a checksum.")
    job = coordinator.enqueue_post_run(
        run_id="run-one",
        thread_id="thread-a",
        terminal_status="done",
        run_dir=run_dir,
    )
    assert job is not None
    processed = coordinator.process_pending_jobs()
    assert len(processed) == 1 and processed[0].status == "done"
    assert len(proposer.trajectories) == 1
    assert "voluntarily computed a checksum" in proposer.trajectories[0]
    assert len(coordinator.store.list_jobs()) == 1
    assert coordinator.store.list_observations() == []
    assert coordinator.store.list_candidates() == []


def test_new_skill_proposer_may_correct_the_reflected_owner_anchor(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    repo = _repo(tmp_path)
    proposer = _SkillProposer(
        group="materials_worker",
        name="new-surface-method",
        reflection_kind="skill_discovery",
        proposed_name="unrelated-method",
    )
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        mode="observe",
        repo_root=repo,
        proposer=proposer,
        reviewer=_Reviewer(),
    )
    run_dir = _run_dir(
        workspace,
        "run-discovery",
        prompt="The complete episode demonstrates one independent reusable surface method.",
    )
    coordinator.enqueue_post_run(
        run_id="run-discovery",
        thread_id="thread-discovery",
        terminal_status="done",
        run_dir=run_dir,
    )
    job = coordinator.process_pending_jobs()[0]

    observation = coordinator.store.list_observations()[0]
    candidate = coordinator.store.list_candidates()[0]
    assert observation.target == "materials_worker/new-surface-method"
    assert candidate.route == "new_skill"
    assert candidate.status == "review"
    assert candidate.group == "materials_worker"
    assert candidate.name == "unrelated-method"
    assert job.outcome["reflection_items"][0]["resolved_target"] == (
        "materials_worker/unrelated-method"
    )
    assert candidate.validation["valid"] is True
    assert candidate.validation["loadable"] is True


def test_two_reflection_anchors_converge_on_one_resolved_target_chain(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    repo = _repo(tmp_path)
    jobs = []
    for index, anchor_name in enumerate(("anchor-one", "anchor-two")):
        coordinator = SelfEvolutionCoordinator(
            workspace=workspace,
            project_id="demo",
            repo_root=repo,
            proposer=_SkillProposer(
                name=anchor_name,
                proposed_name="shared-owner",
                reflection_kind="skill_discovery",
                marker=f"shared revision {index + 1}",
            ),
            reviewer=_Reviewer(),
        )
        run_id = f"run-anchor-{index + 1}"
        run_dir = _run_dir(workspace, run_id)
        coordinator.enqueue_post_run(
            run_id=run_id,
            thread_id=f"thread-{index + 1}",
            terminal_status="done",
            run_dir=run_dir,
        )
        jobs.append(coordinator.process_pending_jobs()[0])

    candidates = coordinator.store.list_candidates()
    assert len(candidates) == 1
    assert candidates[0].name == "shared-owner"
    assert candidates[0].revision == 2
    assert {
        item["resolved_target"]
        for job in jobs
        for item in job.outcome["reflection_items"]
    } == {"materials_worker/shared-owner"}
    assert {
        observation.target
        for observation in coordinator.store.list_observations()
    } == {"materials_worker/anchor-one", "materials_worker/anchor-two"}
    assert {
        observation.resolved_target
        for observation in coordinator.store.list_observations()
    } == {"materials_worker/shared-owner"}


def test_concurrent_reroutes_share_the_resolved_target_lock_and_chain(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    repo = _repo(tmp_path)
    rendezvous = threading.Barrier(2)
    prepared: list[tuple[SelfEvolutionCoordinator, SelfEvolutionJob, Observation, object]] = []

    class _ConcurrentReroute(_SkillProposer):
        def __init__(self, *, name: str, marker: str) -> None:
            super().__init__(
                name=name,
                proposed_name="shared-concurrent-owner",
                reflection_kind="skill_discovery",
                marker=marker,
            )
            self.first_proposal = True

        def propose(self, *, candidate_root: Path):
            if self.first_proposal:
                self.first_proposal = False
                rendezvous.wait(timeout=10)
            return super().propose(candidate_root=candidate_root)

    for index, anchor_name in enumerate(("concurrent-anchor-one", "concurrent-anchor-two")):
        coordinator = SelfEvolutionCoordinator(
            workspace=workspace,
            project_id="demo",
            repo_root=repo,
            proposer=_ConcurrentReroute(
                name=anchor_name,
                marker=f"concurrent revision {index + 1}",
            ),
            reviewer=_Reviewer(),
            worker_id=f"worker-{index + 1}",
        )
        run_id = f"run-concurrent-reroute-{index + 1}"
        run_dir = _run_dir(workspace, run_id)
        job = coordinator.store.enqueue_job(
            trigger_kind="post_run",
            run_id=run_id,
            run_dir=run_dir,
        )
        trace = collect_turn_trace(run_dir=run_dir, include_events=False)
        observation = coordinator.store.write_observation(
                Observation(
                    observation_id=f"obs-concurrent-reroute-{index + 1}",
                    run_id=run_id,
                    thread_id=f"thread-concurrent-{index + 1}",
                    job_id=job.job_id,
                signal_kind="skill_discovery",
                target=f"materials_worker/{anchor_name}",
                claim=f"Concurrent finding {index + 1}.",
                evidence_refs=[{"source_ref": f"run:{run_id}#event:1"}],
                created_at=f"2026-08-22T00:00:0{index}+00:00",
            )
        )
        prepared.append((coordinator, job, observation, trace))

    def materialize(item):
        coordinator, job, observation, trace = item
        return coordinator._candidate_for_observation(
            job=job,
            observation=observation,
            current_trace=trace,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(materialize, prepared))

    store = prepared[0][0].store
    candidates = store.list_candidates()
    assert len(candidates) == 1
    assert candidates[0].name == "shared-concurrent-owner"
    assert candidates[0].revision == 2
    assert {result[3] for result in results} == {
        "materials_worker/shared-concurrent-owner"
    }
    assert len(store.list_observations(status="consolidated")) == 2


def test_same_episode_findings_remain_together_when_owner_is_rerouted(
    tmp_path: Path,
) -> None:
    class _TwoFindingProposer(_SkillProposer):
        def reflect(self, *, trace_scope, **_kwargs):
            refs = _all_event_refs(trace_scope)
            return (
                ReflectionBatch(
                    items=[
                        ReflectionResult(
                            kind="skill_discovery",
                            group="materials_worker",
                            name="initial-owner",
                            change="Preserve the successful method choice.",
                            evidence_refs=refs,
                            rationale="The episode exposes a reusable method invariant.",
                        ),
                        ReflectionResult(
                            kind="skill_discovery",
                            group="materials_worker",
                            name="initial-owner",
                            change="Avoid the unsupported recovery branch.",
                            evidence_refs=refs,
                            rationale="The same episode exposes a bounded counterexample.",
                        ),
                    ]
                ),
                {},
            )

    workspace = tmp_path / "workspace"
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path),
        proposer=_TwoFindingProposer(
            proposed_name="actual-owner",
            reflection_kind="skill_discovery",
        ),
        reviewer=_Reviewer(),
    )
    run_dir = _run_dir(workspace, "run-two-findings")
    coordinator.enqueue_post_run(
        run_id="run-two-findings",
        terminal_status="done",
        run_dir=run_dir,
    )
    job = coordinator.process_pending_jobs()[0]
    candidate = coordinator.store.read_candidate(job.candidate_id)

    assert candidate is not None
    assert len(candidate.evidence_ids) == 2
    assert len(coordinator.store.list_observations(status="consolidated")) == 2
    assert coordinator.store.list_observations(status="open") == []
    assert {
        item["resolved_target"] for item in job.outcome["reflection_items"]
    } == {"materials_worker/actual-owner"}


def test_new_skill_keeps_the_reflected_exact_target_through_review(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    repo = _repo(tmp_path)
    proposer = _SkillProposer(
        group="materials_worker",
        name="new-surface-method",
        reflection_kind="skill_discovery",
    )
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        mode="observe",
        repo_root=repo,
        proposer=proposer,
        reviewer=_Reviewer(),
    )
    run_dir = _run_dir(workspace, "run-discovery")
    coordinator.enqueue_post_run(
        run_id="run-discovery",
        thread_id="thread-discovery",
        terminal_status="done",
        run_dir=run_dir,
    )
    coordinator.process_pending_jobs()

    candidate = coordinator.store.list_candidates()[0]
    assert candidate.route == "new_skill"
    assert candidate.group == "materials_worker"
    assert candidate.name == "new-surface-method"
    assert candidate.status == "review"


def test_explicit_durable_preference_skips_recurrence_but_stays_human_controlled(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, candidate = _explicit_preference_candidate(workspace, _repo(tmp_path))
    assert candidate.route == "workspace_preference"
    assert candidate.action == "memory"
    assert candidate.status == "review"
    assert coordinator.store.read_memory_text() == ""
    assert (
        coordinator.store.revision_dir(candidate.candidate_id, 1)
        / "review.json"
    ).is_file()
    evidence = (
        coordinator.store.revision_dir(candidate.candidate_id, 1)
        / "evidence.md"
    ).read_text(encoding="utf-8")
    assert "From now on, always prefer Chinese for reports in this workspace." not in evidence
    assert "job:sej_" in evidence
    assert "run:run-preference#event:" in evidence
    assert candidate.candidate_id not in coordinator.store.read_active_skills()["skills"]


def test_exact_target_combines_complete_cross_thread_evidence_without_threshold(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, candidate = _recurrent_skill_candidate(workspace, _repo(tmp_path))
    assert candidate.route == "amend_existing_skill"
    assert candidate.group == "materials_worker"
    assert candidate.name == "surface-repair"
    assert candidate.status == "review"
    assert candidate.revision == 3
    observations = coordinator.store.list_observations(status="consolidated")
    assert len(observations) == 3
    assert len({item.thread_id for item in observations}) == 2
    proposal = json.loads(
        (
            coordinator.store.revision_dir(candidate.candidate_id, 3)
            / "proposal.json"
        ).read_text(encoding="utf-8")
    )
    assert proposal["evidence_ids"] == [observations[0].observation_id]
    assert "supporting_evidence_ids" not in proposal
    assert "counterexample_ids" not in proposal
    evidence = (
        coordinator.store.revision_dir(candidate.candidate_id, 3)
        / "evidence.md"
    ).read_text(encoding="utf-8")
    assert evidence.count("## Finding") == 1
    assert "# Complete episode trajectory" not in evidence


def test_human_rejection_does_not_create_a_regex_blocklist_for_future_evidence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    proposer = _SkillProposer(marker="rejected draft marker")
    coordinator, candidate = _recurrent_skill_candidate(
        workspace,
        _repo(tmp_path),
        proposer=proposer,
    )
    coordinator.promotion.reject(
        candidate,
        actor="alice",
        rationale="This proposal generalizes beyond the observed failure.",
    )
    rejected_revision = candidate.revision
    run_dir = _run_dir(
        workspace,
        "run-later",
        prompt="A later complete episode supplies new evidence for the same exact target.",
    )
    proposer.marker = "fresh evidence marker"
    coordinator.enqueue_post_run(
        run_id="run-later",
        thread_id="thread-later",
        terminal_status="done",
        run_dir=run_dir,
    )
    coordinator.process_pending_jobs()
    revised = coordinator.store.read_candidate(candidate.candidate_id)
    assert revised is not None
    assert revised.revision == rejected_revision + 1
    assert revised.status == "review"
    revised_root = coordinator.store.revision_dir(revised.candidate_id, revised.revision)
    proposal = json.loads((revised_root / "proposal.json").read_text(encoding="utf-8"))
    assert proposal["content_parent_version"] == "base"
    assert not (revised_root / "current" / "prior_revision").exists()
    revised_skill = (
        revised_root / "proposed" / revised.group / revised.name / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "fresh evidence marker" in revised_skill
    assert "rejected draft marker" not in revised_skill
    audit = coordinator.store.audit_log_path.read_text(encoding="utf-8")
    assert "rejection_signature" not in audit


@pytest.mark.parametrize(
    "reflection_kind",
    [
        "no_change",
        "execution_lapse",
    ],
)
def test_non_actionable_reflections_never_invoke_candidate_proposal(
    tmp_path: Path,
    reflection_kind: str,
) -> None:
    workspace = tmp_path / reflection_kind
    run_dir = _run_dir(workspace, "run-route")
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path / reflection_kind),
        proposer=_NoChangeProposer(reflection_kind),
        reviewer=_Reviewer(),
    )
    coordinator.enqueue_explicit_learn(
        run_id="run-route",
        run_dir=run_dir,
        note="Inspect the complete episode before deciding whether anything is reusable.",
    )
    job = coordinator.process_pending_jobs()[0]
    assert job.status == "done"
    projected = project_self_evolution_job(job, workspace=workspace)
    assert projected["outcome"]["reflection_items"][0]["reason"] == (
        "The complete episode does not justify a durable SOP update."
    )
    assert projected["summary"] == (
        "The complete episode does not justify a durable SOP update."
    )
    assert coordinator.store.list_candidates() == []
    assert coordinator.store.list_observations() == []


def test_proposer_decline_is_a_visible_no_candidate_result_with_reason(
    tmp_path: Path,
) -> None:
    class _DecliningProposer:
        def reflect(self, *, trace_scope, **_kwargs):
            return (
                ReflectionResult(
                    kind="skill_discovery",
                    group="materials_worker",
                    name="tentative-method",
                    change="Inspect whether this one-off action should become an SOP.",
                    evidence_refs=_all_event_refs(trace_scope),
                    rationale="The reflection identified a question worth deeper inspection.",
                ),
                {},
            )

        def propose(self, **_kwargs):
            return (
                ProposerResult(
                    action="ignore",
                    rationale="The complete evidence shows a one-off action, not a reusable SOP.",
                ),
                {},
            )

    workspace = tmp_path / "workspace"
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path),
        proposer=_DecliningProposer(),
        reviewer=_Reviewer(),
    )
    run_dir = _run_dir(workspace, "run-proposer-decline")
    coordinator.enqueue_post_run(
        run_id="run-proposer-decline",
        terminal_status="done",
        run_dir=run_dir,
    )

    job = coordinator.process_pending_jobs()[0]
    projected = project_self_evolution_job(job, workspace=workspace)

    assert job.status == "done"
    assert job.outcome["reflection_items"][0]["status"] == "ignored"
    assert job.outcome["reflection_items"][0]["reason"] == (
        "The complete evidence shows a one-off action, not a reusable SOP."
    )
    assert projected["result_kind"] == "ignored"
    assert projected["summary"] == (
        "The complete evidence shows a one-off action, not a reusable SOP."
    )
    assert coordinator.store.list_candidates() == []
    assert coordinator.store.list_observations(status="open") == []
    assert len(coordinator.store.list_observations(status="consolidated")) == 1


def test_rerouted_defer_keeps_open_delta_for_later_resolved_target_episode(
    tmp_path: Path,
) -> None:
    class _RerouteThenDefer(_SkillProposer):
        def __init__(self) -> None:
            super().__init__(
                name="initial-owner",
                proposed_name="actual-owner",
                reflection_kind="skill_discovery",
            )
            self.proposal_count = 0

        def propose(self, **kwargs):
            self.proposal_count += 1
            if self.proposal_count == 2:
                return (
                    ProposerResult(
                        action="defer",
                        rationale="One ordinary episode is not enough to distinguish a durable pattern.",
                    ),
                    {},
                )
            return super().propose(candidate_root=kwargs["candidate_root"])

    workspace = tmp_path / "workspace"
    proposer = _RerouteThenDefer()
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path),
        proposer=proposer,
        reviewer=_Reviewer(),
    )
    first_dir = _run_dir(workspace, "run-deferred-first")
    coordinator.enqueue_post_run(
        run_id="run-deferred-first",
        terminal_status="done",
        run_dir=first_dir,
    )
    first_job = coordinator.process_pending_jobs()[0]

    assert first_job.outcome["reflection_items"][0]["status"] == "deferred"
    assert first_job.outcome["reflection_items"][0]["resolved_target"] == (
        "materials_worker/actual-owner"
    )
    first_observation = coordinator.store.list_observations(status="open")[0]
    assert first_observation.target == "materials_worker/initial-owner"
    assert first_observation.resolved_target == "materials_worker/actual-owner"
    assert coordinator.store.list_candidates() == []
    assert coordinator.store.list_observations_for_target(
        "materials_worker/initial-owner",
        status="open",
    ) == []
    assert coordinator.store.list_observations_for_target(
        "materials_worker/actual-owner",
        status="open",
    ) == [first_observation]
    history = EvolutionHistoryScope(
        db_path=coordinator.store.db_path,
        target="materials_worker/actual-owner",
    ).execute(
        "SELECT target, resolved_target, status FROM evolution_observations"
    )
    assert history["rows"] == [
        {
            "target": "materials_worker/initial-owner",
            "resolved_target": "materials_worker/actual-owner",
            "status": "open",
        }
    ]
    projected = project_self_evolution_job(first_job, workspace=workspace)
    assert projected["result_kind"] == "deferred"
    assert projected["outcome"]["reflection_items"][0]["finding"]["change"]

    proposer.name = "actual-owner"
    second_dir = _run_dir(workspace, "run-deferred-second")
    coordinator.enqueue_post_run(
        run_id="run-deferred-second",
        terminal_status="done",
        run_dir=second_dir,
    )
    second_job = coordinator.process_pending_jobs()[0]
    candidate = coordinator.store.read_candidate(second_job.candidate_id)

    assert candidate is not None
    assert len(candidate.evidence_ids) == 2
    assert coordinator.store.list_observations(status="open") == []
    assert len(coordinator.store.list_observations(status="consolidated")) == 2


def test_reject_and_exhausted_needs_revision_stop_without_a_human_review_queue(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "reject"
    coordinator, candidate = _explicit_preference_candidate(
        workspace,
        _repo(tmp_path / "reject"),
        reviewer=_Reviewer("reject"),
    )
    assert candidate.status == "rejected"
    assert candidate.review["recommendation"] == "reject"
    assert coordinator.promotion.allowed_actions(candidate) == []

    workspace_revision = tmp_path / "needs-revision"
    coordinator2, candidate2 = _explicit_preference_candidate(
        workspace_revision,
        _repo(tmp_path / "needs-revision"),
        reviewer=_Reviewer("needs_revision"),
    )
    assert candidate2.status == "revision"
    assert candidate2.revision == 3
    assert candidate2.review["recommendation"] == "needs_revision"
    job = coordinator2.store.list_jobs(limit=10)[0]
    assert job.outcome["reflection_items"][0]["status"] == "needs_revision"
    assert "automatic repair budget" in job.outcome["reflection_items"][0]["reason"]


def test_needs_revision_automatically_creates_r0002_without_mutating_r0001(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    proposer = _MemoryProposer("- Prefer Chinese reports.\n")
    coordinator, candidate = _explicit_preference_candidate(
        workspace,
        _repo(tmp_path),
        reviewer=_SequenceReviewer("needs_revision", "approve"),
        proposer=proposer,
    )
    assert candidate.revision == 2
    assert candidate.review["recommendation"] == "approve"
    root1 = coordinator.store.revision_dir(candidate.candidate_id, 1)
    root2 = coordinator.store.revision_dir(candidate.candidate_id, 2)
    revision_one = {
        path.relative_to(root1).as_posix(): path.read_bytes()
        for path in root1.rglob("*")
        if path.is_file()
    }
    assert root2.is_dir()
    assert (root2 / "prior_review.json").is_file()
    assert json.loads((root1 / "review.json").read_text(encoding="utf-8"))[
        "recommendation"
    ] == "needs_revision"
    assert json.loads((root2 / "review.json").read_text(encoding="utf-8"))[
        "recommendation"
    ] == "approve"
    after_read = {
        path.relative_to(root1).as_posix(): path.read_bytes()
        for path in root1.rglob("*")
        if path.is_file()
    }
    assert after_read == revision_one
    audit = [
        json.loads(line)
        for line in coordinator.store.audit_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["event"] == "automatic_revision_requested" for event in audit)
    activation_events = [
        event["event"]
        for event in audit
        if event["event"]
        in {
            "review_approved_auto_head",
            "automatic_selection",
            "review_activation_result",
        }
    ]
    assert activation_events == ["review_approved_auto_head"]


def test_skill_revision_starts_from_complete_predecessor_candidate(tmp_path: Path) -> None:
    class _PredecessorEditingProposer(_SkillProposer):
        def __init__(self) -> None:
            super().__init__(marker="candidate revision one")
            self.proposal_calls = 0
            self.saw_prior_artifacts = False

        def propose(self, *, candidate_root: Path):
            self.proposal_calls += 1
            if self.proposal_calls == 1:
                return super().propose(candidate_root=candidate_root)
            skill_md = (
                candidate_root
                / "proposed/materials_worker/surface-repair/SKILL.md"
            )
            original = skill_md.read_text(encoding="utf-8")
            assert "candidate revision one" in original
            self.saw_prior_artifacts = (
                candidate_root / "current/prior_revision/proposal.json"
            ).is_file() and (
                candidate_root / "current/prior_revision/validation.json"
            ).is_file()
            skill_md.write_text(
                original.rstrip() + "\n\nCandidate revision two preserves revision one.\n",
                encoding="utf-8",
            )
            return (
                ProposerResult(
                    action="skill",
                    group="materials_worker",
                    name="surface-repair",
                    rationale="Edit the exact predecessor candidate without discarding its supported SOP.",
                ),
                {},
            )

    workspace = tmp_path / "workspace"
    repo = _repo(tmp_path)
    _write_skill(
        repo / "skills",
        group="materials_worker",
        name="surface-repair",
        marker="built-in owner",
    )
    proposer = _PredecessorEditingProposer()
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=repo,
        proposer=proposer,
        reviewer=_SequenceReviewer("needs_revision", "approve"),
    )
    run_dir = _run_dir(workspace, "run-predecessor-revision")
    coordinator.enqueue_explicit_learn(
        run_id="run-predecessor-revision",
        run_dir=run_dir,
        note="Preserve and refine this bounded surface repair SOP.",
    )
    job = coordinator.process_pending_jobs()[0]
    candidate = coordinator.store.list_candidates()[0]
    revision_one = coordinator.store.revision_dir(candidate.candidate_id, 1)
    before = {
        path.relative_to(revision_one).as_posix(): path.read_bytes()
        for path in revision_one.rglob("*")
        if path.is_file()
    }
    assert job.status == "done"
    assert candidate.revision == 2
    assert proposer.proposal_calls == 2
    assert proposer.saw_prior_artifacts is True
    revision_two_skill = (
        coordinator.store.revision_dir(candidate.candidate_id, 2)
        / "proposed/materials_worker/surface-repair/SKILL.md"
    ).read_text(encoding="utf-8")
    assert "candidate revision one" in revision_two_skill
    assert "Candidate revision two preserves revision one" in revision_two_skill
    assert {
        path.relative_to(revision_one).as_posix(): path.read_bytes()
        for path in revision_one.rglob("*")
        if path.is_file()
    } == before


def test_missing_predecessor_is_reported_before_creating_a_revision_workspace(
    tmp_path: Path,
) -> None:
    store = SelfEvolutionStore(tmp_path / "workspace", project_id="demo")
    candidate_id = "candidate-missing-predecessor"
    with pytest.raises(FileNotFoundError, match="predecessor revision workspace is missing"):
        self_evolution_agents.prepare_candidate_workspace(
            store=store,
            candidate_id=candidate_id,
            repo_root=_repo(tmp_path),
            revision=2,
            prior_revision_root=tmp_path / "missing-r0001",
        )
    assert not store.revision_dir(candidate_id, 2).exists()


def test_selected_retry_preserves_candidate_revision_execution_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path),
        proposer=_MemoryProposer(),
        reviewer=_Reviewer("approve"),
    )
    coordinator.store.enqueue_job(
        trigger_kind="selected_retry",
        run_id="revision-retry",
        run_dir=workspace,
        payload={"original_trigger_kind": "candidate_revision"},
    )
    claimed = coordinator.store.claim_jobs(
        project_id="demo",
        owner=coordinator.worker_id,
        limit=1,
    )
    assert len(claimed) == 1
    processed: list[str] = []

    def _process_revision(job):
        processed.append(job.job_id)
        return "sec_retried_revision"

    monkeypatch.setattr(coordinator, "_process_revision_job", _process_revision)
    finished = coordinator._process_job(claimed[0])

    assert processed == [claimed[0].job_id]
    assert finished.status == "done"
    assert finished.candidate_id == "sec_retried_revision"
    assert finished.outcome["revision_candidate_id"] == "sec_retried_revision"


def test_reviewer_failure_preserves_candidate_and_exact_error(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path),
        proposer=_MemoryProposer(),
        reviewer=_FailingReviewer(),
    )
    run_dir = _run_dir(workspace, "run-review")
    coordinator.enqueue_explicit_learn(
        run_id="run-review",
        run_dir=run_dir,
        note="From now on always prefer Chinese workspace reports.",
    )
    job = coordinator.process_pending_jobs()[0]
    assert job.status == "error"
    assert job.candidate_id
    item = job.outcome["reflection_items"][0]
    assert item["target"] == "memory/report_language"
    assert "review service unavailable" in item["error"]
    candidate = coordinator.store.read_candidate(job.candidate_id)
    assert candidate is not None
    assert candidate.status == "revision"
    assert candidate.review["error_type"] == "RuntimeError"
    assert candidate.review["error"] == "review service unavailable"
    assert coordinator.store.list_observations()[0].status == "consolidated"


def test_memory_requires_human_promotion_and_rollback_restores_exact_parent(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    store = SelfEvolutionStore(workspace, project_id="demo")
    before = "# Persistent Instruction Memory\n\n- Prefer English reports.\n"
    assert store.compare_and_swap_memory(
        expected_hash=store.memory_hash(),
        new_text=before,
    )[0]
    coordinator, candidate = _explicit_preference_candidate(workspace, _repo(tmp_path))
    assert coordinator.store.read_memory_text() == before
    report = coordinator.gate.run(candidate)
    promoted = coordinator.promotion.promote_stable(
        candidate,
        report,
        actor="alice",
        rationale="The explicit preference and exact memory diff were reviewed.",
    )
    assert promoted.status == "stable"
    assert "Chinese" in coordinator.store.read_memory_text()
    rolled_back = coordinator.promotion.rollback(
        promoted,
        actor="alice",
        rationale="The preference is no longer desired.",
    )
    assert rolled_back.status == "inactive"
    assert coordinator.store.read_memory_text() == before


def test_skill_rollback_restores_exact_previous_stable_revision(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, revision_one = _actual_owner_candidate(
        workspace,
        run_id="stable-revision-one",
        note=(
            "Surface termination screening must preserve the established slab "
            "generation and freezing policy."
        ),
        marker="stable revision one",
    )
    report_one = coordinator.gate.run(revision_one)
    canary_one = coordinator.promotion.start_canary(
        revision_one,
        report_one,
        actor="alice",
        run_ids=["canary-revision-one"],
    )
    version_one = f"{revision_one.candidate_id}@r{revision_one.revision:04d}"
    coordinator.store.upsert_skill_run(
        SkillRun(
            run_id="canary-revision-one",
            skill_name="materials_worker/surface-and-termination-screening",
            skill_version=version_one,
            presented=True,
            read=True,
            outcome="verified_success",
        )
    )
    coordinator.promotion.promote_stable(
        canary_one,
        report_one,
        actor="alice",
        rationale="The first exact revision passed its selected canary.",
    )

    coordinator_two, revision_two = _actual_owner_candidate(
        workspace,
        run_id="stable-revision-two",
        note=(
            "Surface termination screening must also preserve the bounded "
            "neighbor criterion in the same workflow."
        ),
        marker="stable revision two",
    )
    assert revision_two.candidate_id == revision_one.candidate_id
    assert revision_two.revision == revision_one.revision + 1
    report_two = coordinator_two.gate.run(revision_two)
    canary_two = coordinator_two.promotion.start_canary(
        revision_two,
        report_two,
        actor="alice",
        run_ids=["canary-revision-two"],
    )
    version_two = f"{revision_two.candidate_id}@r{revision_two.revision:04d}"
    coordinator_two.store.upsert_skill_run(
        SkillRun(
            run_id="canary-revision-two",
            skill_name="materials_worker/surface-and-termination-screening",
            skill_version=version_two,
            presented=True,
            read=True,
            outcome="verified_success",
        )
    )
    stable_two = coordinator_two.promotion.promote_stable(
        canary_two,
        report_two,
        actor="alice",
        rationale="The second exact revision passed its selected canary.",
    )
    materialized = (
        coordinator_two.store.self_develop_skills_dir
        / "materials_worker"
        / "surface-and-termination-screening"
        / "SKILL.md"
    )
    assert "stable revision two" in materialized.read_text(encoding="utf-8")

    rolled_back = coordinator_two.promotion.rollback(
        stable_two,
        actor="alice",
        rationale="Restore the previous exact stable revision.",
    )
    active = coordinator_two.store.read_active_skills()["skills"][
        "materials_worker/surface-and-termination-screening"
    ]
    assert rolled_back.status == "inactive"
    assert active == {
        "enabled": True,
        "selected_version": version_one,
        "update_policy": "pinned",
        "auto_head": version_two,
        "stable": version_one,
    }
    restored = materialized.read_text(encoding="utf-8")
    assert "stable revision one" in restored
    assert "stable revision two" not in restored


def test_skill_canary_scope_is_exact_and_release_evidence_is_human_judged(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, candidate = _recurrent_skill_candidate(workspace, _repo(tmp_path))
    report = coordinator.gate.run(candidate)
    assert coordinator.promotion.allowed_actions(candidate) == [
        "request_revision",
        "start_canary",
        "reject",
    ]
    with pytest.raises(ValueError, match="at least one explicit thread or run"):
        coordinator.promotion.start_canary(
            candidate,
            report,
            actor="alice",
        )
    canary = coordinator.promotion.start_canary(
        candidate,
        report,
        actor="alice",
        thread_ids=["thread-canary"],
        rationale="Low-risk canary thread.",
    )
    pointer = coordinator.store.read_active_skills()["skills"][
        "materials_worker/surface-repair"
    ]["canary"]
    exact_version = f"{candidate.candidate_id}@r{candidate.revision:04d}"
    assert pointer == {
        "version": exact_version,
        "thread_ids": ["thread-canary"],
        "run_ids": [],
    }
    assert "promote_stable" in coordinator.promotion.allowed_actions(canary)
    coordinator.store.upsert_skill_run(
        SkillRun(
            run_id="run-canary",
            skill_name="materials_worker/surface-repair",
            skill_version=exact_version,
            presented=True,
            read=True,
            outcome="verified_failure",
        )
    )
    assert "promote_stable" in coordinator.promotion.allowed_actions(canary)
    readiness = coordinator.promotion.promotion_readiness(canary)
    assert readiness["canary_actual_use"]["decision"] == "human"
    assert readiness["canary_actual_use"]["passed"] is None
    assert readiness["canary_actual_use"]["failed_runs"] == 1
    stable = coordinator.promotion.promote_stable(
        canary,
        report,
        actor="alice",
        rationale="The human reviewed the scoped evidence and accepts the exact revision.",
    )
    active = coordinator.store.read_active_skills()["skills"][
        "materials_worker/surface-repair"
    ]
    assert active == {
        "enabled": True,
        "selected_version": exact_version,
        "update_policy": "pinned",
        "auto_head": exact_version,
        "stable": exact_version,
    }
    assert stable.status == "stable"


def test_false_activation_is_observational_and_preserves_canary_and_stable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, candidate = _recurrent_skill_candidate(workspace, _repo(tmp_path))
    report = coordinator.gate.run(candidate)
    canary = coordinator.promotion.start_canary(
        candidate,
        report,
        actor="alice",
        run_ids=["run-canary"],
    )
    active = coordinator.store.read_active_skills()
    active["skills"]["materials_worker/surface-repair"]["stable"] = "older@r0001"
    coordinator.store.write_active_skills(active)
    exact_version = f"{candidate.candidate_id}@r{candidate.revision:04d}"
    coordinator.store.upsert_skill_run(
        SkillRun(
            run_id="run-canary",
            skill_name="materials_worker/surface-repair",
            skill_version=exact_version,
            presented=True,
            read=True,
            outcome="verified_failure",
            false_activation=True,
        )
    )
    evidence = coordinator.promotion.promotion_readiness(canary)["canary_actual_use"]
    assert evidence["decision"] == "human"
    assert evidence["failed_runs"] == 1
    pointers = coordinator.store.read_active_skills()["skills"][
        "materials_worker/surface-repair"
    ]
    assert pointers == {
        "enabled": True,
        "selected_version": "base",
        "update_policy": "follow_auto",
        "auto_head": exact_version,
        "stable": "older@r0001",
        "canary": {
            "version": exact_version,
            "thread_ids": [],
            "run_ids": ["run-canary"],
        },
    }
    assert coordinator.store.read_candidate(canary.candidate_id).status == "canary"


def test_host_verified_false_activation_is_visible_without_automatic_canary_stop(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, candidate = _recurrent_skill_candidate(workspace, _repo(tmp_path))
    report = coordinator.gate.run(candidate)
    coordinator.promotion.start_canary(
        candidate,
        report,
        actor="alice",
        run_ids=["run-observed-false-activation"],
    )
    active = coordinator.store.read_active_skills()
    active["skills"]["materials_worker/surface-repair"]["stable"] = "older@r0001"
    coordinator.store.write_active_skills(active)

    run_id = "run-observed-false-activation"
    run_dir = _run_dir(
        workspace,
        run_id,
        prompt="Perform an ordinary analysis with no surface-selection failure.",
        status="error",
    )
    exact_version = f"{candidate.candidate_id}@r{candidate.revision:04d}"
    virtual_path = "/.deepagents/skills/materials_worker/surface-repair"
    write_skill_version_manifest(
        run_dir=run_dir,
        run_id=run_id,
        entries=[
            {
                "skill_name": "materials_worker/surface-repair",
                "skill_version": exact_version,
                "virtual_path": virtual_path,
            }
        ],
    )
    _record_run_event(
        run_dir,
        name="TOOL_RAW_INPUT",
        ts=1,
        payload={
            "callback_run_id": "read-skill",
            "tool": "read_file",
            "params_compact": f'{{"path":"{virtual_path}/SKILL.md"}}',
        },
    )
    _record_run_event(
        run_dir,
        name="TOOL_CALL_END",
        ts=2,
        payload={
            "callback_run_id": "read-skill",
            "tool": "read_file",
            "status": "success",
        },
    )
    _record_run_event(
        run_dir,
        name="SKILL_OUTCOME",
        ts=3,
        source="host_verifier",
        payload={
            "skill_name": "materials_worker/surface-repair",
            "skill_version": exact_version,
            "outcome": "failure",
            "false_activation": True,
            "outcome_ref": "verifier:false-activation",
        },
    )
    coordinator._proposer = _NoChangeProposer()
    coordinator.enqueue_post_run(
        run_id=run_id,
        thread_id="thread-ordinary",
        terminal_status="error",
        run_dir=run_dir,
    )
    coordinator.process_pending_jobs()

    records = coordinator.store.list_skill_runs(
        skill_name="materials_worker/surface-repair",
        run_id=run_id,
    )
    assert len(records) == 1
    assert records[0].presented is True
    assert records[0].read is True
    assert records[0].helper_used is False
    assert records[0].outcome == "verified_failure"
    assert records[0].false_activation is True
    assert coordinator.store.read_active_skills()["skills"][
        "materials_worker/surface-repair"
    ] == {
        "enabled": True,
        "selected_version": "base",
        "update_policy": "follow_auto",
        "auto_head": exact_version,
        "canary": {
            "version": exact_version,
            "thread_ids": [],
            "run_ids": [run_id],
        },
    }
    assert coordinator.store.read_candidate(candidate.candidate_id).status == "canary"


def test_skill_telemetry_pages_past_five_thousand_events_without_inventory_fields(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    run_id = "run-long-telemetry"
    run_dir = _run_dir(workspace, run_id)
    store = SelfEvolutionStore(workspace, project_id="demo")
    virtual_path = "/.deepagents/skills/materials_worker/surface-repair"
    write_skill_version_manifest(
        run_dir=run_dir,
        run_id=run_id,
        entries=[
            {
                "skill_name": "materials_worker/surface-repair",
                "skill_version": "base@surface-repair-v1",
                "virtual_path": virtual_path,
            },
            {
                "skill_name": "materials_worker/unused-base-skill",
                "skill_version": "base@unused-v1",
                "virtual_path": "/.deepagents/skills/materials_worker/unused-base-skill",
            },
        ],
    )
    _record_run_event(
        run_dir,
        name="TOOL_RAW_INPUT",
        ts=1,
        payload={
            "callback_run_id": "old-read-skill",
            "tool": "read_file",
            "params_compact": f'{{"path":"{virtual_path}/SKILL.md"}}',
        },
    )
    _record_run_event(
        run_dir,
        name="TOOL_CALL_END",
        ts=2,
        payload={
            "callback_run_id": "old-read-skill",
            "tool": "read_file",
            "status": "success",
        },
    )
    observation_store = ObservabilityStore(run_dir)
    payload = json.dumps({"status": "done"}, sort_keys=True)
    with observation_store._connect() as conn:
        conn.executemany(
            """
            INSERT INTO observation_events (
                ts, seq, source, channel, category, name,
                run_id, task_id, step_id, thread_id, message_id, part_id,
                agent_name, callback_run_id, parent_callback_run_id,
                node, model, tool, status, duration_ms, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    float(index + 3), None, "test", "test", "test", "TASK_END",
                    run_id, "", None, "", "", "", "", "", "", "", "", "",
                    "done", None, payload,
                )
                for index in range(5_000)
            ],
        )

    records = finalize_skill_run_telemetry(
        store=store,
        run_id=run_id,
        run_dir=run_dir,
        task_outcome="verified_success",
        outcome_ref="verifier:long-run",
    )

    assert len(records) == 1
    record = records[0]
    assert record.read is True
    assert record.skill_version == "base@surface-repair-v1"
    assert record.outcome == "unknown"
    assert record.partial is False
    assert not hasattr(record, "source_event_count")
    assert not hasattr(record, "source_min_event_id")
    assert not hasattr(record, "source_max_event_id")
    history = EvolutionHistoryScope(
        db_path=store.db_path,
        target="materials_worker/surface-repair",
    ).execute(
        "SELECT run_id, skill_version, presented, read, helper_used, outcome "
        "FROM evolution_skill_runs"
    )
    assert history["rows"] == [
        {
            "run_id": run_id,
            "skill_version": "base@surface-repair-v1",
            "presented": 1,
            "read": 1,
            "helper_used": 0,
            "outcome": "unknown",
        }
    ]


def test_run_level_failure_is_not_guessed_as_skill_failure_or_automatic_stop(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, candidate = _recurrent_skill_candidate(workspace, _repo(tmp_path))
    report = coordinator.gate.run(candidate)
    run_id = "run-verified-canary-failure"
    coordinator.promotion.start_canary(
        candidate,
        report,
        actor="alice",
        run_ids=[run_id],
    )
    active = coordinator.store.read_active_skills()
    active["skills"]["materials_worker/surface-repair"]["stable"] = "older@r0001"
    coordinator.store.write_active_skills(active)

    run_dir = _run_dir(
        workspace,
        run_id,
        prompt="Use the surface repair procedure.",
        status="error",
    )
    exact_version = f"{candidate.candidate_id}@r{candidate.revision:04d}"
    virtual_path = "/.deepagents/skills/materials_worker/surface-repair"
    write_skill_version_manifest(
        run_dir=run_dir,
        run_id=run_id,
        entries=[
            {
                "skill_name": "materials_worker/surface-repair",
                "skill_version": exact_version,
                "virtual_path": virtual_path,
            }
        ],
    )
    _record_run_event(
        run_dir,
        name="TOOL_RAW_INPUT",
        ts=1,
        payload={
            "callback_run_id": "read-skill",
            "tool": "read_file",
            "params_compact": f'{{"path":"{virtual_path}/SKILL.md"}}',
        },
    )
    _record_run_event(
        run_dir,
        name="TOOL_CALL_END",
        ts=2,
        payload={
            "callback_run_id": "read-skill",
            "tool": "read_file",
            "status": "success",
        },
    )
    _record_run_event(
        run_dir,
        name="TASK_END",
        ts=3,
        source="host_verifier",
        payload={
            "task_outcome": "failure",
            "outcome_ref": "verifier:canary-task-failure",
            "summary": "The required surface selection remained wrong.",
        },
    )
    coordinator._proposer = _NoChangeProposer()
    coordinator.enqueue_post_run(
        run_id=run_id,
        thread_id="thread-canary-failure",
        terminal_status="error",
        run_dir=run_dir,
    )
    coordinator.process_pending_jobs()

    record = coordinator.store.list_skill_runs(
        skill_name="materials_worker/surface-repair",
        run_id=run_id,
    )[0]
    assert record.used is True
    assert record.outcome == "unknown"
    assert record.false_activation is False
    assert coordinator.store.read_active_skills()["skills"][
        "materials_worker/surface-repair"
    ] == {
        "enabled": True,
        "selected_version": "base",
        "update_policy": "follow_auto",
        "auto_head": exact_version,
        "canary": {
            "version": exact_version,
            "thread_ids": [],
            "run_ids": [run_id],
        },
    }
    assert coordinator.store.read_candidate(candidate.candidate_id).status == "canary"


def test_runtime_pins_exact_canary_and_stable_in_distinct_immutable_snapshots(tmp_path: Path) -> None:
    class _Profile:
        def config_for_role(self, role: str):
            return SimpleNamespace(model=f"{role}-model", provider="langchain", base_url=None)

    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, stable_candidate = _actual_owner_candidate(
        workspace,
        run_id="stable-source",
        note=(
            "Surface termination screening slab generation must preserve one fixed "
            "freezing policy and uniform VASP ranking."
        ),
        marker="stable revision",
    )
    stable_report = coordinator.gate.run(stable_candidate)
    stable_canary = coordinator.promotion.start_canary(
        stable_candidate,
        stable_report,
        actor="alice",
        run_ids=["stable-canary"],
    )
    coordinator.store.upsert_skill_run(
        SkillRun(
            run_id="stable-canary",
            skill_name="materials_worker/surface-and-termination-screening",
            skill_version=(
                f"{stable_candidate.candidate_id}@r{stable_candidate.revision:04d}"
            ),
            presented=True,
            read=True,
            outcome="verified_success",
        )
    )
    stable_candidate = coordinator.promotion.promote_stable(
        stable_canary,
        stable_report,
        actor="alice",
        rationale="Selected canary succeeded.",
    )
    coordinator2, canary_candidate = _actual_owner_candidate(
        workspace,
        run_id="canary-source",
        note=(
            "The controlled slab termination screening set must keep the neighbor "
            "criterion fixed alongside slab generation, freezing policy, lateral "
            "expansion, and standardized VASP ranking runs."
        ),
        marker="canary revision with fixed neighbor criterion",
    )
    canary_report = coordinator2.gate.run(canary_candidate)
    canary = coordinator2.promotion.start_canary(
        canary_candidate,
        canary_report,
        actor="alice",
        thread_ids=["thread-canary"],
    )

    canary_runner = build_specialist_runner(
        workspace=workspace,
        llm_profile=_Profile(),
        reporter=None,
        run_control=None,
        project_id="demo",
        preferred_entrypoint="experiment",
    ).runner
    stable_runner = build_specialist_runner(
        workspace=workspace,
        llm_profile=_Profile(),
        reporter=None,
        run_control=None,
        project_id="demo",
        preferred_entrypoint="experiment",
    ).runner
    canary_runner._stage_deepagent_assets(workspace / "files", thread_id="thread-canary")
    stable_runner._stage_deepagent_assets(workspace / "files", thread_id="thread-stable")
    assert canary_runner._skill_snapshot_root.is_dir()
    assert stable_runner._skill_snapshot_root.is_dir()
    assert (
        canary_runner._skill_snapshot_root
        / "skills/materials_worker/surface-and-termination-screening/SKILL.md"
    ).is_file()
    assert canary_runner._skill_snapshot_root != stable_runner._skill_snapshot_root
    assert canary_runner._skill_version_entries
    canary_selected_entries = [
        item
        for item in canary_runner._skill_version_entries
        if "@r" in item["skill_version"]
    ]
    stable_selected_entries = [
        item
        for item in stable_runner._skill_version_entries
        if "@r" in item["skill_version"]
    ]
    assert {
        item["skill_name"]
        for item in canary_selected_entries
    } == {"materials_worker/surface-and-termination-screening"}
    assert {
        item["skill_name"]
        for item in stable_selected_entries
    } == {"materials_worker/surface-and-termination-screening"}
    canary_version = next(
        item["skill_version"]
        for item in canary_runner._skill_version_entries
        if item["skill_name"] == "materials_worker/surface-and-termination-screening"
    )
    stable_version = next(
        item["skill_version"]
        for item in stable_runner._skill_version_entries
        if item["skill_name"] == "materials_worker/surface-and-termination-screening"
    )
    assert canary_version == (
        f"{canary_candidate.candidate_id}@r{canary_candidate.revision:04d}"
    )
    assert stable_version == (
        f"{stable_candidate.candidate_id}@r{stable_candidate.revision:04d}"
    )
    assert any(
        item["skill_version"].startswith("base@")
        for item in canary_runner._skill_version_entries
    )
    assert canary.status == "canary"


def test_released_bundle_drift_is_visible_without_silent_candidate_demotion(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, candidate = _actual_owner_candidate(
        workspace,
        run_id="drift-source",
        note=(
            "Surface termination screening slab generation must preserve one fixed "
            "freezing policy and uniform VASP ranking."
        ),
        marker="candidate before builtin drift",
    )
    report = coordinator.gate.run(candidate)
    coordinator.promotion.start_canary(
        candidate,
        report,
        actor="alice",
        thread_ids=["thread-drift"],
    )
    snapshot_skill = (
        coordinator.store.revision_dir(candidate.candidate_id, 1)
        / "proposed/materials_worker/surface-and-termination-screening/SKILL.md"
    )
    snapshot_skill.write_text(
        snapshot_skill.read_text(encoding="utf-8") + "\nLegacy snapshot marker.\n",
        encoding="utf-8",
    )
    runner = build_specialist_runner(
        workspace=workspace,
        llm_profile=SimpleNamespace(
            config_for_role=lambda role: SimpleNamespace(
                model=f"{role}-model",
                provider="langchain",
                base_url=None,
            )
        ),
        reporter=None,
        run_control=None,
        project_id="demo",
        preferred_entrypoint="experiment",
    ).runner
    with pytest.raises(RuntimeError, match="revision bytes changed"):
        runner._active_skill_sources(thread_id="thread-drift")
    assert coordinator.store.read_candidate(candidate.candidate_id).status == "canary"
    assert "canary" in coordinator.store.read_active_skills()["skills"][
        "materials_worker/surface-and-termination-screening"
    ]


def test_newest_first_cursor_pagination_for_candidates_and_observations(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = SelfEvolutionStore(workspace, project_id="demo")
    for index in range(4):
        observation = Observation(
            observation_id=f"obs-{index}",
            run_id=f"run-{index}",
            thread_id=f"thread-{index % 2}",
            signal_kind="workspace_preference",
            target=f"memory/preference-{index}",
            claim=f"failure {index}",
            evidence_refs=[],
            outcome_ref=f"run:{index}",
            created_at=f"2026-07-29T00:00:0{index}+00:00",
        )
        store.write_observation(observation)
        candidate = LearningCandidate(
            candidate_id=f"candidate-{index}",
            project_id="demo",
            run_id=f"run-{index}",
            thread_id="thread",
            action="memory",
            route="workspace_preference",
            evidence_ids=[observation.observation_id],
            created_at=f"2026-07-29T00:00:0{index}+00:00",
        )
        root = store.reset_candidate_dir(candidate.candidate_id)
        (root / "memories").mkdir()
        (root / "memories/AGENTS.md").write_text(f"rule {index}", encoding="utf-8")
        candidate.bundle_hash = hash_text(f"rule {index}")
        store.write_candidate(candidate)
    first = store.list_candidates(limit=2)
    second = store.list_candidates(limit=2, before=first[-1].candidate_id)
    assert [item.candidate_id for item in first] == ["candidate-3", "candidate-2"]
    assert [item.candidate_id for item in second] == ["candidate-1", "candidate-0"]
    observations = store.list_observations(limit=2)
    assert [item.observation_id for item in observations] == ["obs-3", "obs-2"]


def test_candidate_gate_does_not_reject_duplicate_loader_names_by_host_policy(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = SelfEvolutionStore(workspace, project_id="demo")
    existing = store.self_develop_skills_dir / "materials_worker/other-directory"
    _write_skill(
        existing.parent,
        group="",
        name="other-directory",
        marker="existing",
    )
    text = (existing / "SKILL.md").read_text(encoding="utf-8")
    (existing / "SKILL.md").write_text(
        text.replace("name: other-directory", "name: duplicate-name"),
        encoding="utf-8",
    )
    candidate = LearningCandidate(
        candidate_id="candidate-duplicate",
        project_id="demo",
        run_id="run",
        thread_id="thread",
        action="skill",
        group="materials_worker",
        name="duplicate-name",
    )
    root = store.reset_candidate_dir(candidate.candidate_id)
    proposed = _write_skill(
        root / "proposed",
        group="materials_worker",
        name="duplicate-name",
        marker="candidate",
    )
    candidate.bundle_hash = hash_tree(proposed)
    report = CandidateGate(store).run(candidate)
    assert report.valid is True
    assert report.loadable is True
    assert report.errors == []


def test_candidate_gate_uses_real_loader_and_ignores_allowed_tools_metadata(
    tmp_path: Path,
) -> None:
    store = SelfEvolutionStore(tmp_path / "workspace", project_id="demo")
    scaffold = LearningCandidate(
        candidate_id="candidate-scaffold",
        project_id="demo",
        run_id="run",
        thread_id="thread",
        action="skill",
        group="materials_worker",
        name="empty-scaffold",
    )
    scaffold_root = store.reset_candidate_dir(scaffold.candidate_id)
    skill_root = (
        scaffold_root
        / "proposed"
        / scaffold.group
        / scaffold.name
    )
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {scaffold.name}",
                "description: Replace this with what the skill is for and when to use it.",
                "---",
                "Follow the reusable SOP and consult [shared context](../shared/reference.md) when useful.",
            ]
        ),
        encoding="utf-8",
    )
    scaffold.bundle_hash = hash_tree(skill_root)
    scaffold_report = CandidateGate(store).run(scaffold)
    assert scaffold_report.valid is True
    assert scaffold_report.loadable is True
    assert scaffold_report.errors == []

    wrong_audience = LearningCandidate(
        candidate_id="candidate-wrong-audience",
        project_id="demo",
        run_id="run",
        thread_id="thread",
        action="skill",
        group="writing_specialist",
        name="vasp-writing",
    )
    wrong_root = store.reset_candidate_dir(wrong_audience.candidate_id)
    proposed = _write_skill(
        wrong_root / "proposed",
        group=wrong_audience.group,
        name=wrong_audience.name,
        marker="bounded manuscript editing",
    )
    skill_path = proposed / "SKILL.md"
    skill_path.write_text(
        skill_path.read_text(encoding="utf-8").replace(
            "compatibility: local",
            "compatibility: local\nallowed-tools: vasp_prepare",
        ),
        encoding="utf-8",
    )
    wrong_audience.bundle_hash = hash_tree(proposed)
    wrong_report = CandidateGate(store).run(wrong_audience)
    assert wrong_report.valid is True
    assert wrong_report.loadable is True
    assert wrong_report.errors == []
    assert not any("allowed-tools" in item for item in wrong_report.diagnostics)
    wrong_audience.status = "review"
    store.write_candidate(wrong_audience)
    canary = PromotionManager(store).start_canary(
        wrong_audience,
        wrong_report,
        actor="alice",
        run_ids=["run"],
        rationale="Exercise the exact candidate without treating metadata as authorization.",
    )
    assert canary.status == "canary"


def test_loader_diagnostic_returns_candidate_to_model_for_repair(tmp_path: Path) -> None:
    class _RepairingProposer:
        def __init__(self) -> None:
            self.calls = 0
            self.feedback: list[list[str]] = []

        def reflect(self, *, trace_scope, **_kwargs):
            return _SkillProposer(
                name="loader-repair",
                reflection_kind="skill_discovery",
            ).reflect(trace_scope=trace_scope)

        def propose(
            self,
            *,
            candidate_root: Path,
            correction_feedback: list[str] | None = None,
        ):
            self.calls += 1
            self.feedback.append(list(correction_feedback or []))
            target = candidate_root / "proposed/materials_worker/loader-repair"
            target.mkdir(parents=True, exist_ok=True)
            if self.calls == 1:
                content = "---\nname: loader-repair\n---\nFirst draft.\n"
            else:
                content = (
                    "---\n"
                    "name: loader-repair\n"
                    "description: Repair this exact loader problem and then use the SOP.\n"
                    "---\n"
                    "Follow the evidence-backed procedure and consult "
                    "[shared context](../shared/reference.md) when useful; no canonical heading order is required.\n"
                )
            (target / "SKILL.md").write_text(content, encoding="utf-8")
            return (
                ProposerResult(
                    action="skill",
                    group="materials_worker",
                    name="loader-repair",
                    rationale="The evidence supports one reusable procedure.",
                ),
                {"response_evidence_text": f"raw proposer response {self.calls}"},
            )

    workspace = tmp_path / "workspace"
    proposer = _RepairingProposer()
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        mode="observe",
        repo_root=_repo(tmp_path),
        proposer=proposer,
        reviewer=_Reviewer(),
    )
    run_dir = _run_dir(workspace, "run-loader-repair")
    coordinator.enqueue_post_run(
        run_id="run-loader-repair",
        thread_id="thread-loader-repair",
        terminal_status="done",
        run_dir=run_dir,
    )
    job = coordinator.process_pending_jobs()[0]
    candidate = coordinator.store.read_candidate(job.candidate_id)

    assert candidate is not None
    assert candidate.status == "review"
    assert candidate.validation["loadable"] is True
    assert proposer.calls == 2
    assert proposer.feedback[0] == []
    assert any("description" in item.casefold() for item in proposer.feedback[1])
    revision_root = coordinator.store.revision_dir(candidate.candidate_id, 1)
    proposal = json.loads((revision_root / "proposal.json").read_text(encoding="utf-8"))
    attempts = json.loads((revision_root / proposal["attempt_response_ref"]).read_text(encoding="utf-8"))
    feedback = json.loads((revision_root / proposal["validation_feedback_ref"]).read_text(encoding="utf-8"))
    assert [item["response_text"] for item in attempts["attempts"]] == [
        "raw proposer response 1",
        "raw proposer response 2",
    ]
    assert len(feedback["rounds"]) == 1
    assert (revision_root / "proposer_response.txt").read_text(encoding="utf-8") == (
        "raw proposer response 2"
    )


def test_exact_target_consolidation_does_not_group_by_wording(tmp_path: Path) -> None:
    store = SelfEvolutionStore(tmp_path / "workspace", project_id="demo")
    same_target_a = store.write_observation(
        Observation(
            observation_id="obs-surface-a",
            run_id="run-a",
            thread_id="thread-a",
            signal_kind="skill_revision",
            target="materials_worker/surface-repair",
            claim="Require the valid surface index after a selection failure.",
            evidence_refs=[{"source_ref": "run:run-a"}],
            created_at="2026-07-27T00:00:00+00:00",
        )
    )
    same_target_b = store.write_observation(
        Observation(
            observation_id="obs-surface-b",
            run_id="run-b",
            thread_id="thread-b",
            signal_kind="skill_revision",
            target="materials_worker/surface-repair",
            claim="表面选择报错后，只修正这一次的终止面编号。",
            evidence_refs=[{"source_ref": "run:run-b"}],
            created_at="2026-07-28T00:00:00+00:00",
        )
    )
    same_words_other_target = store.write_observation(
        Observation(
            observation_id="obs-other-target",
            run_id="run-c",
            thread_id="thread-c",
            signal_kind="skill_revision",
            target="materials_worker/adsorbate-placement",
            claim=same_target_a.claim,
            evidence_refs=[{"source_ref": "run:run-c"}],
            created_at="2026-07-29T00:00:00+00:00",
        )
    )

    batch = ConsolidationService(store).batch_for(same_target_b)

    assert batch.target == "materials_worker/surface-repair"
    assert batch.evidence_ids == (
        same_target_a.observation_id,
        same_target_b.observation_id,
    )
    assert same_words_other_target.observation_id not in batch.evidence_ids


def test_history_omits_active_uncommitted_candidate_revision(tmp_path: Path) -> None:
    store = SelfEvolutionStore(tmp_path / "workspace", project_id="demo")
    candidate = LearningCandidate(
        candidate_id="candidate-history-transaction",
        project_id="demo",
        run_id="run-history",
        thread_id="thread-history",
        action="skill",
        route="amend_existing_skill",
        group="dynamics_worker",
        name="lammps-md-execution",
        revision=1,
        status="review",
    )
    store.write_candidate(candidate)
    active_revision = store.revision_dir(candidate.candidate_id, 2)
    active_revision.mkdir(parents=True)
    (active_revision / "proposal.json").write_text("{}\n", encoding="utf-8")

    rows = EvolutionHistoryScope(db_path=store.db_path).execute(
        "SELECT revision, status FROM evolution_candidate_revisions "
        "ORDER BY revision"
    )["rows"]

    assert rows == [{"revision": 1, "status": "review"}]

    (store.revision_dir(candidate.candidate_id, 1) / "candidate.json").unlink()
    committed_error = EvolutionHistoryScope(db_path=store.db_path).execute(
        "SELECT revision, status FROM evolution_candidate_revisions"
    )["rows"]
    assert committed_error == [{"revision": 1, "status": "read_error"}]


def test_consolidation_exposes_no_similarity_or_recurrence_decision_api() -> None:
    public_methods = {
        name
        for name in dir(ConsolidationService)
        if not name.startswith("_")
        and callable(getattr(ConsolidationService, name))
    }

    assert public_methods == {"batch_for", "evidence_markdown"}
    assert not {
        "decide",
        "eligible",
        "cluster_for",
        "similarity",
        "route",
    } & public_methods


def test_new_same_target_evidence_creates_immutable_revision_without_fixed_gate(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    repo = _repo(tmp_path)
    _write_skill(
        repo / "skills",
        group="materials_worker",
        name="surface-repair",
        marker="current surface repair workflow",
    )
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=repo,
        proposer=_SkillProposer(),
        reviewer=_Reviewer(),
    )

    first_run = _run_dir(
        workspace,
        "run-first",
        prompt="First complete episode for the surface-repair target.",
    )
    coordinator.enqueue_post_run(
        run_id="run-first",
        thread_id="thread-a",
        terminal_status="done",
        run_dir=first_run,
    )
    first_job = coordinator.process_pending_jobs()[0]
    first = coordinator.store.read_candidate(first_job.candidate_id)
    assert first is not None and first.revision == 1
    revision_one = coordinator.store.revision_dir(first.candidate_id, 1)
    revision_one_before = {
        path.relative_to(revision_one).as_posix(): path.read_bytes()
        for path in revision_one.rglob("*")
        if path.is_file()
    }

    second_run = _run_dir(
        workspace,
        "run-second",
        prompt="A differently worded second episode maps to the same exact target.",
    )
    coordinator.enqueue_post_run(
        run_id="run-second",
        thread_id="thread-b",
        terminal_status="done",
        run_dir=second_run,
    )
    second_job = coordinator.process_pending_jobs()[0]
    revised = coordinator.store.read_candidate(first.candidate_id)

    assert second_job.candidate_id == first.candidate_id
    assert revised is not None and revised.revision == 2
    assert len(revised.evidence_ids) == 1
    proposal = json.loads(
        (
            coordinator.store.revision_dir(revised.candidate_id, 2)
            / "proposal.json"
        ).read_text(encoding="utf-8")
    )
    assert proposal["evidence_ids"] == revised.evidence_ids
    assert "supporting_evidence_ids" not in proposal
    assert "counterexample_ids" not in proposal
    revision_one_after = {
        path.relative_to(revision_one).as_posix(): path.read_bytes()
        for path in revision_one.rglob("*")
        if path.is_file()
    }
    assert revision_one_after == revision_one_before

    duplicate = coordinator.enqueue_post_run(
        run_id="run-second",
        thread_id="thread-b",
        terminal_status="done",
        run_dir=second_run,
    )
    assert duplicate.status == "done"
    assert coordinator.process_pending_jobs() == []
    assert coordinator.store.read_candidate(first.candidate_id).revision == 2


def test_later_revision_restores_each_explicit_correction_from_its_job(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(
        workspace,
        "run-shared",
        prompt="Prepare the workspace report.",
    )
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path),
        proposer=_MemoryProposer(),
        reviewer=_Reviewer(),
    )
    first_note = "Use Chinese for generated workspace reports."
    second_note = "Keep verbatim source quotations in their original language."
    for note in (first_note, second_note):
        coordinator.enqueue_explicit_learn(
            run_id="run-shared",
            run_dir=run_dir,
            thread_id="thread-a",
            note=note,
        )
        coordinator.process_pending_jobs()

    candidate = coordinator.store.list_candidates()[0]
    evidence = (
        coordinator.store.revision_dir(candidate.candidate_id, 2)
        / "evidence.md"
    ).read_text(encoding="utf-8")

    assert first_note not in evidence
    assert second_note not in evidence
    assert evidence.count("job:sej_") == 1
    assert evidence.count("run:run-shared#event:1") == 1


def test_legacy_running_job_without_lease_moves_to_recovery_review(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    root = workspace / "metadata" / "self_evolution"
    root.mkdir(parents=True, exist_ok=True)
    database = root / "jobs.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                run_dir TEXT NOT NULL,
                thread_id TEXT NOT NULL DEFAULT '',
                trigger_kind TEXT NOT NULL,
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                candidate_id TEXT NOT NULL DEFAULT '',
                model_config TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL DEFAULT '{}',
                error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO jobs(
                job_id, project_id, run_id, run_dir, trigger_kind, status,
                created_at, updated_at
            ) VALUES (
                'legacy-running', 'demo', 'run-old', '/tmp/run-old',
                'post_run', 'running', '2026-07-20T00:00:00+00:00',
                '2026-07-20T00:00:00+00:00'
            )
            """
        )
    store = SelfEvolutionStore(workspace, project_id="demo")
    recovered = store.list_jobs()[0]
    assert recovered.status == "recovery_review"
    assert recovered.owner == ""
    assert recovered.lease_until == ""
    assert "no verifiable lease" in recovered.error
    assert store.claim_jobs(owner="worker-a") == []


def test_candidate_cursor_pagination_promotes_updated_old_candidate(
    tmp_path: Path,
) -> None:
    store = SelfEvolutionStore(tmp_path / "workspace", project_id="demo")
    for index in range(30):
        candidate = LearningCandidate(
            candidate_id=f"candidate-{index:02d}",
            project_id="demo",
            run_id=f"run-{index:02d}",
            thread_id="thread",
            action="memory",
            route="workspace_preference",
            created_at=f"2020-01-{index + 1:02d}T00:00:00+00:00",
        )
        root = store.reset_candidate_dir(candidate.candidate_id)
        (root / "memories").mkdir()
        text = f"rule {index}"
        (root / "memories/AGENTS.md").write_text(text, encoding="utf-8")
        candidate.bundle_hash = hash_text(text)
        store.write_candidate(candidate)
    with sqlite3.connect(store.db_path) as connection:
        for index in range(30):
            connection.execute(
                "UPDATE candidates SET updated_at = ? WHERE candidate_id = ?",
                (
                    f"2020-01-{index + 1:02d}T00:00:00+00:00",
                    f"candidate-{index:02d}",
                ),
            )
    store.update_candidate_status("candidate-00", "review")

    observed: list[str] = []
    before = ""
    while True:
        page = store.list_candidates(limit=11, before=before)
        if not page:
            break
        observed.extend(item.candidate_id for item in page)
        before = page[-1].candidate_id
    assert len(observed) == 30
    assert len(set(observed)) == 30
    assert observed[0] == "candidate-00"


def test_queued_job_keeps_original_anchor_after_newer_runs_exist(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    proposer = _NoChangeProposer()
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path),
        proposer=proposer,
        reviewer=_Reviewer(),
    )
    anchor_dir = _run_dir(
        workspace,
        "run-anchor",
        prompt="Inspect the original anchor episode.",
    )
    queued = coordinator.enqueue_post_run(
        run_id="run-anchor",
        thread_id="thread-a",
        episode_id="episode-anchor",
        terminal_status="done",
        run_dir=anchor_dir,
    )
    assert queued is not None

    _run_dir(
        workspace,
        "run-newer-same-thread",
        prompt="This is a newer run in the same thread.",
    )
    _run_dir(
        workspace,
        "run-newer-other-thread",
        prompt="This is a newer run in another thread.",
    )

    finished = coordinator.process_pending_jobs()[0]
    assert finished.run_id == "run-anchor"
    assert finished.run_dir == str(anchor_dir.resolve())
    assert finished.status == "done"
    assert len(proposer.trajectories) == 1
    assert "`run-anchor`" in proposer.trajectories[0]
    assert "Inspect the original anchor episode." in proposer.trajectories[0]
    assert "run-newer-same-thread" not in proposer.trajectories[0]
    assert "run-newer-other-thread" not in proposer.trajectories[0]


def test_episode_terminal_enqueue_uses_only_final_resumed_physical_run(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    proposer = _NoChangeProposer()
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path),
        proposer=proposer,
        reviewer=_Reviewer(),
    )
    episode_id = "episode-user-message-1"
    episode_prompt = "Complete the ceramic analysis after any required approval."

    interrupted_dir = _run_dir(
        workspace,
        "run-interrupted",
        prompt=episode_prompt,
        status="interrupted",
    )
    interrupted_state = json.loads(
        (interrupted_dir / "run_state.json").read_text(encoding="utf-8")
    )
    interrupted_state.update(
        {
            "episode_id": episode_id,
            "episode_prompt": episode_prompt,
            "checkpoint_resume_available": True,
        }
    )
    (interrupted_dir / "run_state.json").write_text(
        json.dumps(interrupted_state),
        encoding="utf-8",
    )
    assert (
        coordinator.enqueue_post_run(
            run_id="run-interrupted",
            thread_id="thread-a",
            episode_id=episode_id,
            terminal_status="interrupted",
            run_dir=interrupted_dir,
        )
        is None
    )

    resumable_error_dir = _run_dir(
        workspace,
        "run-resumable-error",
        prompt="",
        status="error",
    )
    error_state = json.loads(
        (resumable_error_dir / "run_state.json").read_text(encoding="utf-8")
    )
    error_state.update(
        {
            "episode_id": episode_id,
            "episode_prompt": episode_prompt,
            "checkpoint_resume_available": True,
        }
    )
    (resumable_error_dir / "run_state.json").write_text(
        json.dumps(error_state),
        encoding="utf-8",
    )
    assert (
        coordinator.enqueue_post_run(
            run_id="run-resumable-error",
            thread_id="thread-a",
            episode_id=episode_id,
            terminal_status="error",
            run_dir=resumable_error_dir,
        )
        is None
    )

    final_dir = _run_dir(workspace, "run-final", prompt="", status="done")
    final_state = json.loads(
        (final_dir / "run_state.json").read_text(encoding="utf-8")
    )
    final_state.update(
        {
            "episode_id": episode_id,
            "episode_prompt": episode_prompt,
            "checkpoint_resume_available": False,
        }
    )
    (final_dir / "run_state.json").write_text(
        json.dumps(final_state),
        encoding="utf-8",
    )
    final_job = coordinator.enqueue_post_run(
        run_id="run-final",
        thread_id="thread-a",
        episode_id=episode_id,
        terminal_status="done",
        run_dir=final_dir,
    )
    assert final_job is not None
    assert final_job.run_id == "run-final"

    duplicate_dir = _run_dir(
        workspace,
        "run-duplicate-final",
        prompt=episode_prompt,
        status="done",
    )
    duplicate = coordinator.enqueue_post_run(
        run_id="run-duplicate-final",
        thread_id="thread-a",
        episode_id=episode_id,
        terminal_status="done",
        run_dir=duplicate_dir,
    )
    assert duplicate is not None
    assert duplicate.job_id == final_job.job_id
    assert duplicate.run_id == "run-final"
    assert len(coordinator.store.list_jobs()) == 1

    finished = coordinator.process_pending_jobs()[0]
    assert finished.run_id == "run-final"
    assert finished.episode_id == episode_id
    assert episode_prompt in proposer.trajectories[0]
    assert "`run-final`" in proposer.trajectories[0]
    assert "run-interrupted" not in proposer.trajectories[0]


def test_explicit_learn_from_older_selected_response_keeps_selected_run(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    old_dir = _run_dir(
        workspace,
        "run-selected-old",
        prompt="The selected older assistant response belongs to this run.",
    )
    _run_dir(
        workspace,
        "run-later",
        prompt="A later turn must not replace the selected evidence.",
    )
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path),
        proposer=_MemoryProposer(),
        reviewer=_Reviewer(),
    )
    queued = coordinator.enqueue_explicit_learn(
        run_id="run-selected-old",
        run_dir=old_dir,
        thread_id="thread-a",
        episode_id="episode-selected-old",
        note="Always prefer Chinese for generated reports in this workspace.",
    )
    finished = coordinator.process_pending_jobs()[0]
    candidate = coordinator.store.read_candidate(finished.candidate_id)

    assert finished.job_id == queued.job_id
    assert finished.run_id == "run-selected-old"
    assert candidate is not None
    assert candidate.run_id == "run-selected-old"
    evidence = (
        coordinator.store.revision_dir(candidate.candidate_id, candidate.revision)
        / "evidence.md"
    ).read_text(encoding="utf-8")
    assert "run:run-selected-old#event:1" in evidence
    assert "run-later" not in evidence


def test_exact_target_history_pages_twelve_runs_and_opens_each_run_individually(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path),
        proposer=_MemoryProposer(),
        reviewer=_Reviewer(),
    )
    expected_run_ids = [f"run-history-{index:02d}" for index in range(12)]
    for index, run_id in enumerate(expected_run_ids):
        run_dir = _run_dir(
            workspace,
            run_id,
            prompt=f"Historical exact-target episode {index}.",
        )
        coordinator.enqueue_explicit_learn(
            run_id=run_id,
            run_dir=run_dir,
            thread_id=f"thread-{index % 3}",
            episode_id=f"episode-history-{index:02d}",
            note=f"Keep the same report-language target, evidence occurrence {index}.",
        )
        assert coordinator.process_pending_jobs()[0].status == "done"

    candidate = coordinator.store.list_candidates()[0]
    assert candidate.revision == 12
    assert len(candidate.evidence_ids) == 1
    scope = coordinator._trace_scope_for_evidence_ids(
        list(candidate.evidence_ids),
        anchor_run_id=candidate.run_id,
    )
    assert scope is not None
    assert scope.run_ids == (candidate.run_id,)

    tools = {tool.name: tool for tool in scope.tools()}
    history_tool = tools["query_evolution_history_sql"]
    pages: list[dict] = []
    for offset in (0, 5, 10):
        pages.append(
            json.loads(
                history_tool.invoke(
                    {
                        "sql": (
                            "SELECT run_id, run_ref FROM evolution_observations "
                            f"ORDER BY created_at, observation_id LIMIT 5 OFFSET {offset}"
                        )
                    }
                )
            )
        )
    assert [page["row_count"] for page in pages] == [5, 5, 2]
    history_rows = [row for page in pages for row in page["rows"]]
    assert {row["run_id"] for row in history_rows} == set(expected_run_ids)

    revision_count = json.loads(
        history_tool.invoke(
            {"sql": "SELECT COUNT(*) AS total FROM evolution_candidate_revisions"}
        )
    )
    assert revision_count["rows"] == [{"total": 12}]

    trace_tool = tools["query_evolution_trace_sql"]
    for row in history_rows:
        selected = json.loads(
            trace_tool.invoke(
                {
                    "sql": "SELECT run_id, user_prompt FROM trajectory_runs",
                    "run_ref": row["run_ref"],
                }
            )
        )
        assert selected["row_count"] == 1
        assert selected["rows"][0]["run_id"] == row["run_id"]


def test_same_target_reflection_findings_materialize_one_candidate_revision(
    tmp_path: Path,
) -> None:
    changes = [
        "Reconstruct cumulative physical time across inherited LAMMPS stages.",
        "Stop dependent stages when the predecessor scientific gate fails.",
    ]

    class _SameTargetProposer(_SkillProposer):
        def __init__(self) -> None:
            super().__init__(
                group="dynamics_worker",
                name="lammps-md-execution",
                marker="batched same-target revision",
            )
            self.propose_calls = 0

        def reflect(self, *, trace_scope, **_kwargs):
            refs = _all_event_refs(trace_scope)
            return (
                ReflectionBatch(
                    items=[
                        ReflectionResult(
                            kind="skill_revision",
                            group=self.group,
                            name=self.name,
                            change=change,
                            evidence_refs=refs,
                            rationale="The complete episode supports this bounded change.",
                        )
                        for change in changes
                    ]
                ),
                {},
            )

        def propose(self, *, candidate_root: Path):
            self.propose_calls += 1
            return super().propose(candidate_root=candidate_root)

    class _CountingReviewer(_Reviewer):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def review(self, **kwargs):
            self.calls += 1
            return super().review(**kwargs)

    workspace = tmp_path / "workspace"
    repo = _repo(tmp_path)
    _write_skill(
        repo / "skills",
        group="dynamics_worker",
        name="lammps-md-execution",
        marker="current LAMMPS execution workflow",
    )
    proposer = _SameTargetProposer()
    reviewer = _CountingReviewer()
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        mode="observe",
        repo_root=repo,
        proposer=proposer,
        reviewer=reviewer,
    )
    run_dir = _run_dir(workspace, "run-same-target-batch")
    coordinator.enqueue_post_run(
        run_id="run-same-target-batch",
        thread_id="thread-a",
        terminal_status="done",
        run_dir=run_dir,
    )

    finished = coordinator.process_pending_jobs()[0]

    items = finished.outcome["reflection_items"]
    assert finished.status == "done"
    assert [item["status"] for item in items] == ["dormant", "dormant"]
    assert len({item["candidate_id"] for item in items}) == 1
    assert proposer.propose_calls == 1
    assert reviewer.calls == 1
    candidate = coordinator.store.read_candidate(items[0]["candidate_id"])
    assert candidate is not None
    assert candidate.revision == 1
    assert len(candidate.evidence_ids) == 2
    revision_root = coordinator.store.revision_dir(candidate.candidate_id, 1)
    assert not coordinator.store.revision_dir(candidate.candidate_id, 2).exists()
    evidence = (revision_root / "evidence.md").read_text(encoding="utf-8")
    assert all(change in evidence for change in changes)


def test_mixed_reflection_batch_projects_partial_and_retries_only_failed_item(
    tmp_path: Path,
) -> None:
    class _ThreeFindingProposer(_MemoryProposer):
        def __init__(self) -> None:
            super().__init__()
            self.reflect_calls = 0

        def reflect(self, *, trace_scope, **_kwargs):
            self.reflect_calls += 1
            refs = _all_event_refs(trace_scope)
            return (
                ReflectionBatch(
                    items=[
                        ReflectionResult(
                            kind="workspace_preference",
                            name=f"report_rule_{index}",
                            change=f"Preserve bounded report rule {index}.",
                            evidence_refs=refs,
                            rationale="The episode explicitly supports this exact rule.",
                        )
                        for index in range(3)
                    ]
                ),
                {},
            )

    class _FailThirdReviewOnce(_Reviewer):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def review(self, **kwargs):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("third target review failed")
            return super().review(**kwargs)

    workspace = tmp_path / "workspace"
    proposer = _ThreeFindingProposer()
    reviewer = _FailThirdReviewOnce()
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        mode="observe",
        repo_root=_repo(tmp_path),
        proposer=proposer,
        reviewer=reviewer,
    )
    run_dir = _run_dir(workspace, "run-mixed-findings")
    coordinator.enqueue_explicit_learn(
        run_id="run-mixed-findings",
        run_dir=run_dir,
        thread_id="thread-a",
        episode_id="episode-mixed-findings",
        note="Record the three independent durable rules from this episode.",
    )
    finished = coordinator.process_pending_jobs()[0]

    assert finished.status == "done"
    assert [item["status"] for item in finished.outcome["reflection_items"]] == [
        "dormant",
        "dormant",
        "error",
    ]
    projected = project_self_evolution_job(finished, workspace=workspace)
    assert projected["outcome"]["state"] == "partial"
    assert projected["result_kind"] == "partial"
    assert projected["outcome"]["completed_item_count"] == 2
    assert projected["outcome"]["failed_item_count"] == 1
    assert len(projected["outcome"]["reflection_items"]) == 3

    successful_ids = [
        item["candidate_id"]
        for item in finished.outcome["reflection_items"]
        if item["status"] == "completed"
    ]
    all_candidate_ids = [
        item["candidate_id"]
        for item in finished.outcome["reflection_items"]
        if item["candidate_id"]
    ]
    assert {
        path.name
        for path in coordinator.store.candidates_dir.iterdir()
        if path.is_dir()
    } == set(all_candidate_ids)
    successful_before = {
        candidate_id: {
            path.relative_to(
                coordinator.store.revision_dir(candidate_id, 1)
            ).as_posix(): path.read_bytes()
            for path in coordinator.store.revision_dir(candidate_id, 1).rglob("*")
            if path.is_file()
        }
        for candidate_id in successful_ids
    }
    failed_item = finished.outcome["reflection_items"][2]
    failed_candidate = coordinator.store.read_candidate(failed_item["candidate_id"])
    assert failed_candidate is not None
    assert failed_candidate.status == "revision"
    assert failed_candidate.review["error"] == "third target review failed"
    predecessor_memory = (
        coordinator.store.revision_dir(failed_candidate.candidate_id, 1)
        / "memories"
        / "AGENTS.md"
    ).read_text(encoding="utf-8")
    retry = coordinator.enqueue_selected_retry(
        job_id=finished.job_id,
        item_ref=failed_item["item_ref"],
        actor="alice",
    )
    retried = coordinator.process_pending_jobs()[0]

    assert retry.run_id == finished.run_id
    assert retry.selected_item_ref == failed_item["item_ref"]
    assert retried.status == "done"
    assert len(retried.outcome["reflection_items"]) == 1
    assert retried.outcome["reflection_items"][0]["item_ref"] == failed_item["item_ref"]
    assert retried.outcome["reflection_items"][0]["status"] == "dormant"
    repaired = coordinator.store.read_candidate(failed_item["candidate_id"])
    assert repaired is not None
    assert repaired.revision == 2
    assert repaired.status == "review"
    revised_memory = (
        coordinator.store.revision_dir(repaired.candidate_id, 2)
        / "memories"
        / "AGENTS.md"
    ).read_text(encoding="utf-8")
    assert predecessor_memory.rstrip() in revised_memory
    assert revised_memory.count("Prefer Chinese reports.") == (
        predecessor_memory.count("Prefer Chinese reports.") + 1
    )
    assert proposer.reflect_calls == 1
    assert len(coordinator.store.list_candidates()) == 3
    for candidate_id, before in successful_before.items():
        revision_root = coordinator.store.revision_dir(candidate_id, 1)
        after = {
            path.relative_to(revision_root).as_posix(): path.read_bytes()
            for path in revision_root.rglob("*")
            if path.is_file()
        }
        assert after == before


@pytest.mark.parametrize(
    ("outcome", "expected_kind", "expected_summary"),
    [
        (
            {
                "no_change": True,
                "reflection_items": [
                    {
                        "item_ref": "item-no-change",
                        "kind": "no_change",
                        "status": "no_change",
                    }
                ],
            },
            "no_change",
            "did not support a durable skill or workspace-memory change",
        ),
        (
            {
                "reflection_items": [
                    {
                        "item_ref": "item-lapse",
                        "kind": "execution_lapse",
                        "status": "execution_lapse",
                    }
                ],
            },
            "execution_lapse",
            "Existing guidance already covered the issue",
        ),
        (
            {
                "reflection_items": [
                    {
                        "item_ref": "item-candidate",
                        "kind": "skill_revision",
                        "target": "materials_worker/relaxation",
                        "status": "completed",
                        "candidate_id": "sec_candidate",
                    }
                ],
            },
            "candidate_created",
            "immutable learning candidate revision",
        ),
        (
            {
                "reflection_items": [
                    {
                        "item_ref": "item-ignored",
                        "kind": "skill_discovery",
                        "target": "materials_worker/new-method",
                        "status": "ignored",
                    }
                ],
            },
            "ignored",
            "recorded and consumed as non-durable evidence",
        ),
        ({}, "legacy_complete", "before detailed learning results were recorded"),
    ],
)
def test_completed_job_projection_returns_an_explicit_learning_result(
    outcome: dict,
    expected_kind: str,
    expected_summary: str,
) -> None:
    projected = project_self_evolution_job(
        {
            "job_id": "sej-result",
            "run_id": "run-result",
            "trigger_kind": "post_run",
            "status": "done",
            "attempt_count": 1,
            "outcome": outcome,
        }
    )

    assert projected["result_kind"] == expected_kind
    assert projected["outcome"]["result_kind"] == expected_kind
    assert expected_summary in projected["summary"]
    if outcome.get("reflection_items"):
        assert projected["outcome"]["reflection_items"][0]["status_label"]


def test_failed_job_projection_returns_an_explicit_error_result() -> None:
    projected = project_self_evolution_job(
        {
            "job_id": "sej-error",
            "run_id": "run-error",
            "trigger_kind": "post_run",
            "status": "error",
            "attempt_count": 1,
            "error": "Trace query rejected after the retry budget was exhausted.",
            "outcome": {},
        }
    )

    assert projected["result_kind"] == "error"
    assert projected["retry_allowed"] is True
    assert "Trace query rejected" in projected["summary"]


def test_job_projection_keeps_reflect_reason_owner_and_evidence_handles() -> None:
    projected = project_self_evolution_job(
        {
            "job_id": "sej-reflect-history",
            "run_id": "run-reflect-history",
            "trigger_kind": "post_run",
            "status": "done",
            "attempt_count": 1,
            "outcome": {
                "reflection_items": [
                    {
                        "item_ref": "finding-one",
                        "kind": "skill_revision",
                        "target": "materials_worker/initial-owner",
                        "resolved_target": "materials_worker/actual-owner",
                        "status": "deferred",
                        "reason": "Wait for another independent use.",
                        "finding": {
                            "change": "Keep the successful scientific method invariant.",
                            "rationale": "One episode does not separate method choice from runtime noise.",
                            "evidence_refs": [
                                {
                                    "source_ref": "run:run-reflect-history#event:7",
                                    "reason": "Observed method choice",
                                    "excerpt": "The method completed but the attribution remains uncertain.",
                                }
                            ],
                        },
                    }
                ]
            },
        }
    )

    item = projected["outcome"]["reflection_items"][0]
    assert item["target"] == "materials_worker/initial-owner"
    assert item["resolved_target"] == "materials_worker/actual-owner"
    assert item["finding"]["change"].startswith("Keep the successful")
    assert item["finding"]["rationale"].startswith("One episode")
    assert item["finding"]["evidence_refs"] == [
        {
            "source_ref": "run:run-reflect-history#event:7",
            "reason": "Observed method choice",
            "excerpt": "The method completed but the attribution remains uncertain.",
        }
    ]


def test_same_target_candidate_race_creates_immutable_followup_revision(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    repo = _repo(tmp_path)
    _write_skill(
        repo / "skills",
        group="materials_worker",
        name="surface-repair",
        marker="active revision",
    )
    first_proposal_started = threading.Event()
    second_observation_written = threading.Event()

    class _BlockingFirstProposer(_SkillProposer):
        def propose(self, *, candidate_root: Path):
            first_proposal_started.set()
            if not second_observation_written.wait(timeout=10):
                raise TimeoutError("second observation did not reach candidate materialization")
            return super().propose(candidate_root=candidate_root)

    coordinator_a = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=repo,
        proposer=_BlockingFirstProposer(marker="proposal-a"),
        reviewer=_Reviewer(),
        worker_id="worker-a",
    )
    coordinator_b = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=repo,
        proposer=_SkillProposer(marker="proposal-b"),
        reviewer=_Reviewer(),
        worker_id="worker-b",
    )
    for index, coordinator in enumerate((coordinator_a, coordinator_b)):
        run_id = f"run-race-{index}"
        run_dir = _run_dir(
            workspace,
            run_id,
            prompt="Repair the same exact surface-selection target.",
        )
        coordinator.enqueue_explicit_learn(
            run_id=run_id,
            run_dir=run_dir,
            thread_id=f"thread-{index}",
            episode_id=f"episode-race-{index}",
            note=f"Exact target evidence from concurrent episode {index}.",
        )

    original_candidate_b = coordinator_b._candidate_for_observation

    def _mark_second_observation(**kwargs):
        second_observation_written.set()
        return original_candidate_b(**kwargs)

    coordinator_b._candidate_for_observation = _mark_second_observation  # type: ignore[method-assign]
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(coordinator_a.process_pending_jobs, limit=1)
        assert first_proposal_started.wait(timeout=10)
        second_future = pool.submit(coordinator_b.process_pending_jobs, limit=1)
        first_result = first_future.result(timeout=20)
        second_result = second_future.result(timeout=20)

    assert first_result[0].status == "done"
    assert second_result[0].status == "done"
    observations = coordinator_a.store.list_observations(status="consolidated")
    assert len(observations) == 2
    candidates = coordinator_a.store.list_candidates()
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.revision == 2
    assert len(candidate.evidence_ids) == 1

    revision_one = coordinator_a.store.revision_dir(candidate.candidate_id, 1)
    revision_two = coordinator_a.store.revision_dir(candidate.candidate_id, 2)
    descriptor_one = json.loads(
        (revision_one / "candidate.json").read_text(encoding="utf-8")
    )
    descriptor_two = json.loads(
        (revision_two / "candidate.json").read_text(encoding="utf-8")
    )
    assert len(descriptor_one["evidence_ids"]) == 1
    assert len(descriptor_two["evidence_ids"]) == 1
    assert "proposal-a" in (
        revision_one / "proposed/materials_worker/surface-repair/SKILL.md"
    ).read_text(encoding="utf-8")
    assert "proposal-b" in (
        revision_two / "proposed/materials_worker/surface-repair/SKILL.md"
    ).read_text(encoding="utf-8")


def test_run_context_retries_exclusive_directory_after_generated_id_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    collision_dir = system_root(workspace=workspace) / "runs" / "run-collision"
    collision_dir.mkdir(parents=True)
    marker = collision_dir / "owner.txt"
    marker.write_text("existing-run", encoding="utf-8")
    generated_ids = iter(("run-collision", "run-collision", "run-fresh"))
    monkeypatch.setattr(
        run_context_module,
        "_default_run_id",
        lambda: next(generated_ids),
    )

    context = RunContext.create(workspace=workspace, model_name="test-model")

    assert context.run_id == "run-fresh"
    assert context.run_dir.name == "run-fresh"
    assert context.run_dir != collision_dir
    assert marker.read_text(encoding="utf-8") == "existing-run"
    assert (context.run_dir / "meta.json").is_file()


def test_tool_inspection_stages_complete_registered_schema_and_long_source(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    inspection_tool = self_evolution_agents._inspect_tool(candidate_root)
    result = inspection_tool.invoke(
        {"tool_name": "generate_batch_adsorption_structures"}
    )
    lines = {
        key.strip(): value.strip()
        for line in str(result).splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
    }
    source_ref = lines["source_ref"]
    schema_ref = lines["schema_ref"]
    source_path = candidate_root / source_ref.removeprefix("/")
    schema_path = candidate_root / schema_ref.removeprefix("/")
    registry = self_evolution_agents.get_tool_registry()
    info = registry.get_tool_info("generate_batch_adsorption_structures")
    function = info.get("function") or info.get("coroutine")
    expected_source = inspect.getsource(function)
    expected_schema = next(
        item
        for item in registry.as_openai_tools(
            allowlist=["generate_batch_adsorption_structures"]
        )
        if item["name"] == "generate_batch_adsorption_structures"
    )

    assert len(expected_source) > 12_000
    assert int(lines["source_chars"]) == len(expected_source)
    assert source_path.read_text(encoding="utf-8") == expected_source
    assert json.loads(schema_path.read_text(encoding="utf-8")) == expected_schema


def test_skill_frontmatter_description_is_read_beyond_old_line_prefix(
    tmp_path: Path,
) -> None:
    skill_md = tmp_path / "SKILL.md"
    decisive_description = (
        "Decisive description after the former line prefix for exact target routing."
    )
    skill_md.write_text(
        "\n".join(
            [
                "---",
                "name: long-frontmatter-skill",
                *[f"padding_{index:02d}: value-{index}" for index in range(55)],
                f"description: {decisive_description}",
                "---",
                "# Long frontmatter skill",
            ]
        ),
        encoding="utf-8",
    )

    parsed = read_skill_frontmatter(skill_md)

    assert parsed["description"] == decisive_description
    assert SelfEvolutionCoordinator._skill_description(skill_md) == decisive_description


def test_revision_worker_reads_complete_predecessor_review_beyond_twenty_concerns(
    tmp_path: Path,
) -> None:
    concerns = [f"Concern {index}: exact predecessor detail" for index in range(35)]

    class _ConcernReviewer(_Reviewer):
        def __init__(self) -> None:
            self.calls = 0

        def review(self, **_kwargs):
            self.calls += 1
            if self.calls > 1:
                return _Reviewer("approve").review()
            return (
                ReviewerResult(
                    recommendation="needs_revision",
                    summary="The exact candidate needs one bounded revision.",
                    concerns=concerns,
                    human_checks=["Inspect every predecessor concern."],
                ),
                {},
            )

    class _PriorReviewProposer(_MemoryProposer):
        def __init__(self) -> None:
            super().__init__("- Prefer Chinese only for generated reports.\n")
            self.seen_concerns: list[str] = []

        def propose(self, *, candidate_root: Path):
            prior_path = candidate_root / "prior_review.json"
            if prior_path.is_file():
                prior = json.loads(prior_path.read_text(encoding="utf-8"))
                self.seen_concerns = list(prior["concerns"])
            return super().propose(candidate_root=candidate_root)

    workspace = tmp_path / "workspace"
    revision_proposer = _PriorReviewProposer()
    reviewer = _ConcernReviewer()
    coordinator, candidate = _explicit_preference_candidate(
        workspace,
        _repo(tmp_path),
        reviewer=reviewer,
        proposer=revision_proposer,
    )
    assert candidate.revision == 2
    assert candidate.review["recommendation"] == "approve"
    assert reviewer.calls == 2
    assert revision_proposer.seen_concerns == concerns
    prior_review = json.loads(
        (
            coordinator.store.revision_dir(candidate.candidate_id, 2)
            / "prior_review.json"
        ).read_text(encoding="utf-8")
    )
    assert prior_review["concerns"] == concerns


def test_reflection_text_is_completed_and_visible_without_inventing_a_judgment(tmp_path):
    text = 'A narrower correction may help, but this is a textual conclusion.\n```json\n{"kind":"skill_revision"}\n```'

    class TextReflector(_MemoryProposer):
        calls = 0

        def reflect(self, **kwargs):
            self.calls += 1
            return TextResult(text=text), {}

        def propose(self, **kwargs):
            raise AssertionError("text must not imply a candidate")

    workspace = tmp_path / "workspace"
    proposer = TextReflector()
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace, project_id="demo", mode="auto", repo_root=_repo(tmp_path),
        proposer=proposer, reviewer=_FailingReviewer(),
    )
    coordinator.enqueue_post_run(run_id="text-run", run_dir=_run_dir(workspace, "text-run"), terminal_status="done")
    job = coordinator.process_pending_jobs()[0]
    assert job.status == "done" and not job.error and not job.candidate_id
    assert proposer.calls == 1
    assert not job.outcome["no_change"]
    assert job.outcome["reflection_items"] == []
    assert not coordinator.store.list_observations()
    projected = project_self_evolution_job(job, workspace=workspace)
    assert projected["result_kind"] == "text_completed"
    assert "textual conclusion" in projected["summary"]
    assert projected["outcome"]["text_responses"][0]["text"].endswith('```')


def test_proposal_text_keeps_edits_and_open_findings_without_occupying_a_revision(tmp_path):
    class TextProposer(_MemoryProposer):
        calls = 0

        def propose(self, *, candidate_root):
            self.calls += 1
            super().propose(candidate_root=candidate_root)
            return TextResult(text="Draft notes are retained; I have not submitted this proposal."), {}

    workspace = tmp_path / "workspace"
    proposer = TextProposer()
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace, project_id="demo", mode="auto", repo_root=_repo(tmp_path),
        proposer=proposer, reviewer=_FailingReviewer(),
    )
    coordinator.enqueue_explicit_learn(
        run_id="draft-run", run_dir=_run_dir(workspace, "draft-run"), thread_id="thread-a",
        note="From now on, always prefer Chinese for generated reports.",
    )
    job = coordinator.process_pending_jobs()[0]
    assert job.status == "done" and not job.candidate_id and not job.error
    assert proposer.calls == 1
    assert job.outcome["reflection_items"][0]["status"] == "text_completed"
    assert coordinator.store.list_observations()[0].status == "open"
    draft_path = workspace / job.outcome["text_responses"][0]["draft_path"]
    assert "Prefer Chinese" in (draft_path / "memories/AGENTS.md").read_text()
    projected = project_self_evolution_job(job, workspace=workspace)
    assert projected["result_kind"] == "text_completed"
    assert projected["outcome"]["completed_item_count"] == 1

    coordinator._proposer = _MemoryProposer()
    coordinator._reviewer = _Reviewer()
    coordinator.enqueue_explicit_learn(
        run_id="submitted-run", run_dir=_run_dir(workspace, "submitted-run"), thread_id="thread-a",
        note="Keep the Chinese-report preference as a durable convention.",
    )
    next_job = coordinator.process_pending_jobs()[0]
    candidate = coordinator.store.read_candidate(next_job.candidate_id)
    assert next_job.status == "done" and candidate is not None and candidate.revision == 1
    assert draft_path.is_dir()


def test_review_text_is_saved_without_approval_or_another_revision(tmp_path):
    class TextReviewer:
        calls = 0

        def review(self, **kwargs):
            self.calls += 1
            return TextResult(text="The word approve here is discussion, not a submitted decision."), {}

    reviewer = TextReviewer()
    coordinator, candidate = _explicit_preference_candidate(tmp_path / "workspace", _repo(tmp_path), reviewer=reviewer)
    job = coordinator.store.list_jobs()[0]
    assert job.status == "done" and not job.error
    assert candidate.status == "review" and candidate.revision == 1
    assert candidate.review["format"] == "text" and "recommendation" not in candidate.review
    assert reviewer.calls == 1
    assert job.outcome["reflection_items"][0]["status"] == "text_completed"
    assert project_self_evolution_job(job)["result_kind"] == "text_completed"
    assert not coordinator.effective.target_summary("/memories/AGENTS.md").get("auto_head")
    root = coordinator.store.revision_dir(candidate.candidate_id, 1)
    assert (root / "reviewer_response.txt").read_text() == candidate.review["text"]
    assert json.loads((root / "review.json").read_text())["text"] == candidate.review["text"]

    coordinator.review_candidate(candidate_id=candidate.candidate_id, expected_revision=1)
    assert reviewer.calls == 1
    assert coordinator.store.read_candidate(candidate.candidate_id).review["format"] == "text"


def test_requested_revision_can_end_with_text_and_preserve_the_previous_revision(tmp_path):
    class TextProposer(_MemoryProposer):
        def propose(self, *, candidate_root):
            super().propose(candidate_root=candidate_root)
            return TextResult(text="The previous concerns need further discussion."), {}

    coordinator, candidate = _explicit_preference_candidate(tmp_path / "workspace", _repo(tmp_path))
    coordinator.store.update_candidate_status(candidate.candidate_id, "revision")
    coordinator._proposer = TextProposer()
    coordinator._reviewer = _FailingReviewer()
    coordinator.enqueue_revision(candidate_id=candidate.candidate_id, expected_revision=1,
                                 guidance="Review the scope of this preference.", actor="user")
    job = coordinator.process_pending_jobs()[0]
    current = coordinator.store.read_candidate(candidate.candidate_id)
    assert job.status == "done" and not job.error
    assert current.revision == 1 and current.status == "revision"
    assert job.outcome["text_responses"][0]["stage"] == "proposal"
    assert project_self_evolution_job(job)["result_kind"] == "text_completed"
    assert not coordinator.store.revision_dir(candidate.candidate_id, 2).exists()


def test_candidate_persists_complete_model_visible_response_evidence(
    tmp_path: Path,
) -> None:
    proposer_evidence = "proposer-start\n" + ("complete proposer evidence\n" * 2_000) + "proposer-tail"
    reviewer_evidence = "reviewer-start\n" + ("complete reviewer evidence\n" * 2_000) + "reviewer-tail"

    class _EvidenceProposer(_MemoryProposer):
        def propose(self, *, candidate_root: Path):
            response, _metadata = super().propose(candidate_root=candidate_root)
            return response, {"response_evidence_text": proposer_evidence}

    class _EvidenceReviewer(_Reviewer):
        def review(self, **kwargs):
            response, _metadata = super().review(**kwargs)
            return response, {"response_evidence_text": reviewer_evidence}

    coordinator, candidate = _explicit_preference_candidate(
        tmp_path / "workspace",
        _repo(tmp_path),
        proposer=_EvidenceProposer(),
        reviewer=_EvidenceReviewer(),
    )
    revision_root = coordinator.store.revision_dir(candidate.candidate_id, 1)
    proposal = json.loads(
        (revision_root / "proposal.json").read_text(encoding="utf-8")
    )
    review = json.loads(
        (revision_root / "review.json").read_text(encoding="utf-8")
    )

    assert proposal["raw_response_ref"] == "proposer_response.txt"
    assert review["raw_response_ref"] == "reviewer_response.txt"
    assert (
        revision_root / proposal["raw_response_ref"]
    ).read_text(encoding="utf-8") == proposer_evidence
    assert (
        revision_root / review["raw_response_ref"]
    ).read_text(encoding="utf-8") == reviewer_evidence
    job_evidence = list(coordinator.store.job_evidence_dir.rglob("*.txt"))
    assert len(job_evidence) == 2
    assert {path.read_text(encoding="utf-8") for path in job_evidence} == {
        proposer_evidence,
        reviewer_evidence,
    }
