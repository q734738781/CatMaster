from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from catmaster.research.knowledge_graph.models import (
    ExperimentCreateRequest,
    GraphCreateRequest,
    GraphPatchRequest,
    HypothesisCreateRequest,
    ResearchGraphPlanningDraft,
    ResearchRefInput,
    ResultJudgmentInput,
    ResultCreateRequest,
    ResultJudgmentSetRequest,
)
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.research.knowledge_graph.store import ResearchGraphConflict
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.runtime.tool_runtime import current_tool_context
from catmaster.tools.base import project_space_root


class ListResearchGraphsInput(BaseModel):
    """[research/graph] List workspace Research Graphs before selecting one explicitly."""

    model_config = ConfigDict(extra="forbid")

    include_archived: bool = Field(
        False,
        description="Pass true only when an archived graph must be found.",
    )


class CreateResearchGraphInput(GraphCreateRequest):
    """[research/graph] Create a workspace Research Graph from a question and optional seed hypotheses."""

    completion_criterion: str = Field(
        "",
        max_length=4_000,
        description=(
            "Optional human-readable scientific completion criterion. Leave empty "
            "to use the default defensible-answer criterion."
        ),
    )


class InspectResearchGraphInput(BaseModel):
    """[research/graph] Read a bounded human-readable neighborhood from one explicit graph."""

    model_config = ConfigDict(extra="forbid")

    graph_id: str = Field(..., min_length=3, description="Explicit graph ID from list or bound context.")
    focus_node_id: str = Field(
        "",
        description="Optional node ID to center; leave empty to inspect the graph frontier.",
    )
    query: str = Field(
        "",
        description="Optional scientific phrase for FTS retrieval; leave empty when focus is sufficient.",
    )
    max_nodes: int = Field(
        24,
        ge=4,
        le=100,
        description="Hard node budget; use 24 unless a broader view is necessary.",
    )
    max_chars: int = Field(
        12_000,
        ge=2_000,
        le=40_000,
        description="Hard character budget for model-visible Markdown.",
    )


class AddResearchHypothesisInput(HypothesisCreateRequest):
    """[research/graph] Add one falsifiable hypothesis with graph revision CAS."""

    graph_id: str = Field(..., min_length=3, description="Explicit target graph ID.")


class AddResearchExperimentInput(ExperimentCreateRequest):
    """[research/graph] Add a draft or ready experiment proposal with graph revision CAS."""

    graph_id: str = Field(..., min_length=3, description="Explicit target graph ID.")


class StageResearchPlanInput(ResearchGraphPlanningDraft):
    """[research/graph] Publish science-first temporary branches from the bound planning turn."""


class SetResearchGraphCompletionInput(BaseModel):
    """[research/graph] Mark whether recorded Results satisfy the graph's completion criterion."""

    model_config = ConfigDict(extra="forbid")

    graph_id: str = Field(..., min_length=3, description="Explicit target graph ID.")
    expected_revision: int = Field(..., ge=1, description="Latest inspected graph revision.")
    completed: bool = Field(
        ...,
        description=(
            "True only when recorded Results and sources satisfy the explicit "
            "completion criterion; false reopens the graph."
        ),
    )


class RecordResearchResultInput(ResultCreateRequest):
    """[research/graph] Record one concise observation or result, source refs, and typed judgments."""

    graph_id: str = Field(..., min_length=3, description="Explicit target graph ID.")


class SetResearchResultJudgmentInput(ResultJudgmentSetRequest):
    """[research/graph] Replace one Result-to-Hypothesis evidence judgment."""

    graph_id: str = Field(..., min_length=3, description="Explicit target graph ID.")
    result_node_id: str = Field(
        ...,
        min_length=3,
        description="Result whose effect is being judged.",
    )
    hypothesis_node_id: str = Field(
        ...,
        min_length=3,
        description="Hypothesis affected by this Result.",
    )


class MarkResearchExperimentFailedInput(BaseModel):
    """[research/graph] Mark an experiment blocked with a concrete reason and optional source refs."""

    model_config = ConfigDict(extra="forbid")

    graph_id: str = Field(..., min_length=3, description="Explicit target graph ID.")
    expected_revision: int = Field(..., ge=1, description="Latest inspected graph revision.")
    experiment_node_id: str = Field(..., min_length=3, description="Experiment node that could not proceed.")
    reason: str = Field(..., min_length=1, max_length=2_000, description="Concrete execution blocker.")
    refs: list[ResearchRefInput] = Field(
        default_factory=list,
        max_length=100,
        description="Typed source refs; omit or pass [] when no durable source exists.",
    )


class RecordBoundResearchResultInput(BaseModel):
    """[research/graph] Finish this child thread's bound experiment with one concise Result."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field("", max_length=300, description="Short result title; leave empty to derive it from the summary.")
    summary: str = Field(
        ...,
        min_length=1,
        max_length=4_000,
        description=(
            "Concise observed or derived scientific outcome, including what the "
            "decision rule needs. Separate observation from causal interpretation "
            "and state material conditions or provenance when they affect meaning; "
            "do not assign a global evidence grade."
        ),
    )
    judgments: list[ResultJudgmentInput] = Field(
        default_factory=list,
        max_length=100,
        description="Typed effects on the hypothesis node IDs shown in the bound graph context; omit or pass [] when the result is not discriminating.",
    )
    refs: list[ResearchRefInput] = Field(
        default_factory=list,
        max_length=98,
        description=(
            "Durable result sources such as artifact, note, DOI, or URL refs. "
            "The host attaches the bound child thread and current run refs automatically."
        ),
    )


class MarkBoundResearchExperimentFailedInput(BaseModel):
    """[research/graph] Finish this child thread's bound experiment as blocked."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(
        ...,
        min_length=1,
        max_length=2_000,
        description="Concrete reason the bound experiment cannot produce a scientific result.",
    )
    refs: list[ResearchRefInput] = Field(
        default_factory=list,
        max_length=98,
        description=(
            "Optional durable sources for the blocker. The host attaches the "
            "bound child thread and current run refs automatically."
        ),
    )


def _service() -> ResearchGraphService:
    workspace = project_space_root()
    return ResearchGraphService(workspace=workspace, workspace_id=workspace.name)


def _artifact(tool_name: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "data": data,
        "suppress_content_offload_ref": True,
    }


def _error(tool_name: str, graph_id: str, exc: Exception) -> None:
    raise CatMasterToolExecutionError(
        tool_name=tool_name,
        public_message=f"{tool_name} failed: {exc}",
        artifact={
            "tool_name": tool_name,
            "data": {"graph_id": str(graph_id or "")},
        },
        error_code="research_graph_error",
    ) from exc


def _trusted_thread_id() -> str:
    return str(current_tool_context().get("thread_id") or "").strip()


def _bind_created_graph(
    service: ResearchGraphService,
    *,
    graph_id: str,
    focus_node_id: str,
) -> str:
    """Bind a new graph only when the host supplied a real current thread."""

    thread_id = _trusted_thread_id()
    if not thread_id:
        return ""
    try:
        service.bind_thread(
            thread_id,
            graph_id=graph_id,
            focus_node_id=focus_node_id,
        )
    except (KeyError, ValueError):
        # Direct CLI/library Specialist runs may use a checkpoint identity that
        # is not a formal WebUI ThreadRecord. Only formal threads are bindable.
        return ""
    return thread_id


def _bound_execution_target(
    service: ResearchGraphService,
) -> tuple[str, str, str]:
    runtime_context = current_tool_context()
    entrypoint = str(runtime_context.get("entrypoint") or "").strip()
    if entrypoint not in {"experiment", "literature_review"}:
        raise ValueError(
            "This bound writeback tool is available only to an Experiment or "
            "Literature Review execution child."
        )
    thread_id = str(runtime_context.get("thread_id") or "").strip()
    if not thread_id:
        raise ValueError("The execution child has no trusted runtime thread binding.")
    thread = service.thread_store.get_thread(thread_id)
    graph_id = str(thread.active_research_graph_id or "").strip()
    experiment_node_id = str(thread.research_focus_node_id or "").strip()
    if not graph_id or not experiment_node_id:
        raise ValueError(
            "The execution child is not bound to a Research Graph experiment."
        )
    node = service.store.get_node(graph_id, experiment_node_id)
    if node["kind"] != "experiment":
        raise ValueError("The bound Research Graph focus is not an experiment.")
    return thread_id, graph_id, experiment_node_id


def _refs_with_bound_sources(
    refs: list[ResearchRefInput],
    *,
    thread_id: str,
) -> list[ResearchRefInput]:
    values = list(refs)
    if not any(
        ref.ref_kind.value == "thread" and ref.ref_id == thread_id
        for ref in values
    ):
        values.append(
            ResearchRefInput(ref_kind="thread", ref_id=thread_id)
        )
    run_id = str(current_tool_context().get("run_id") or "").strip()
    if run_id and not any(
        ref.ref_kind.value == "run" and ref.ref_id == run_id
        for ref in values
    ):
        values.append(
            ResearchRefInput(ref_kind="run", ref_id=run_id)
        )
    return values


def _context_result(
    *,
    tool_name: str,
    graph_id: str,
    focus_node_id: str = "",
    prefix: str = "",
) -> tuple[str, dict[str, Any]]:
    context = _service().context_builder.build(
        graph_id,
        focus_node_id=focus_node_id,
        max_nodes=24,
        max_chars=12_000,
    )
    content = context["markdown"]
    if prefix:
        content = f"{prefix}\n\n{content}"
    presentation = context["presentation"]
    return content, _artifact(
        tool_name,
        {
            "graph": presentation["graph"],
            "focus_node_id": presentation["focus_node_id"],
            "frontier_node_ids": presentation["frontier_node_ids"],
            "shown_count": presentation["shown_count"],
            "total_count": presentation["total_count"],
            "omitted_count": presentation["omitted_count"],
        },
    )


def list_research_graphs(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "list_research_graphs"
    try:
        params = ListResearchGraphsInput.model_validate(payload)
        rows = _service().catalog(include_archived=params.include_archived)
        if not rows:
            content = "No Research Graph exists in this workspace."
        else:
            lines = [f"Workspace Research Graphs ({len(rows)}):"]
            for graph in rows:
                state = (
                    "archived"
                    if graph["archived"]
                    else "completed"
                    if graph["completed"]
                    else "active"
                )
                lines.append(
                    f"- {graph['title']} ({graph['graph_id']}, revision "
                    f"{graph['revision']}, {state}, {graph['orchestration_mode']}): "
                    f"{graph['question']}"
                )
            content = "\n".join(lines)
        return content, _artifact(
            tool_name,
            {
                "graph_count": len(rows),
                "graphs": [
                    {
                        "graph_id": graph["graph_id"],
                        "title": graph["title"],
                        "question": graph["question"],
                        "completion_criterion": graph["completion_criterion"],
                        "completed": graph["completed"],
                        "archived": graph["archived"],
                        "revision": graph["revision"],
                        "orchestration_mode": graph["orchestration_mode"],
                    }
                    for graph in rows
                ],
            },
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, "", exc)


def create_research_graph(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "create_research_graph"
    try:
        params = CreateResearchGraphInput.model_validate(payload)
        service = _service()
        result = service.create_graph(params)
        graph_id = result["graph"]["graph_id"]
        focus_node_id = (
            str(result["nodes"][0]["node_id"])
            if result.get("nodes")
            else ""
        )
        bound_thread_id = _bind_created_graph(
            service,
            graph_id=graph_id,
            focus_node_id=focus_node_id,
        )
        return _context_result(
            tool_name=tool_name,
            graph_id=graph_id,
            focus_node_id=focus_node_id,
            prefix=(
                f"Created Research Graph {graph_id} and attached it to the "
                f"current thread {bound_thread_id}."
                if bound_thread_id
                else f"Created Research Graph {graph_id}."
            ),
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, "", exc)


def inspect_research_graph(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "inspect_research_graph"
    graph_id = str(payload.get("graph_id") or "")
    try:
        params = InspectResearchGraphInput.model_validate(payload)
        context = _service().context_builder.build(
            params.graph_id,
            focus_node_id=params.focus_node_id,
            query=params.query,
            max_nodes=params.max_nodes,
            max_chars=params.max_chars,
        )
        presentation = context["presentation"]
        return context["markdown"], _artifact(
            tool_name,
            {
                "graph": presentation["graph"],
                "focus_node_id": presentation["focus_node_id"],
                "frontier_node_ids": presentation["frontier_node_ids"],
                "shown_count": presentation["shown_count"],
                "total_count": presentation["total_count"],
                "omitted_count": presentation["omitted_count"],
            },
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def add_research_hypothesis(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "add_research_hypothesis"
    graph_id = str(payload.get("graph_id") or "")
    try:
        params = AddResearchHypothesisInput.model_validate(payload)
        request = HypothesisCreateRequest.model_validate(
            params.model_dump(mode="json", exclude={"graph_id"})
        )
        result = _service().add_hypothesis(params.graph_id, request)
        return _context_result(
            tool_name=tool_name,
            graph_id=params.graph_id,
            focus_node_id=result["node"]["node_id"],
            prefix=f"Added hypothesis {result['node']['title']}.",
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def add_research_experiment(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "add_research_experiment"
    graph_id = str(payload.get("graph_id") or "")
    try:
        params = AddResearchExperimentInput.model_validate(payload)
        request = ExperimentCreateRequest.model_validate(
            params.model_dump(mode="json", exclude={"graph_id"})
        )
        result = _service().add_experiment(params.graph_id, request)
        return _context_result(
            tool_name=tool_name,
            graph_id=params.graph_id,
            focus_node_id=result["node"]["node_id"],
            prefix=f"Added experiment proposal {result['node']['title']}.",
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def stage_research_plan(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "stage_research_plan"
    graph_id = ""
    try:
        params = StageResearchPlanInput.model_validate(payload)
        planning_thread_id = _trusted_thread_id()
        if not planning_thread_id:
            raise ValueError("The planning turn has no trusted thread binding.")
        service = _service()
        planning = service.store.find_planning_by_thread(planning_thread_id)
        if planning is None:
            raise ValueError(
                "The current thread is not an active Research Graph planning turn."
            )
        graph_id = str(planning["graph_id"])
        result = service.stage_planning_draft(
            graph_id,
            expected_revision=int(planning["revision"]),
            planning_thread_id=planning_thread_id,
            draft=ResearchGraphPlanningDraft.model_validate(
                params.model_dump(mode="json")
            ),
        )
        summary = str(result.get("summary") or "Temporary plan published.")
        materialized = dict(result.get("materialized") or {})
        return summary, _artifact(
            tool_name,
            {
                "graph_id": graph_id,
                "planning_id": result["planning_id"],
                "recommended_target_id": result["recommended_target_id"],
                "materialized": {
                    key: materialized[key]
                    for key in (
                        "proposal_id",
                        "node_ids",
                        "next_experiment_node_id",
                    )
                    if key in materialized
                },
            },
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def set_research_graph_completion(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    tool_name = "set_research_graph_completion"
    graph_id = str(payload.get("graph_id") or "")
    try:
        params = SetResearchGraphCompletionInput.model_validate(payload)
        service = _service()
        service.patch_graph(
            params.graph_id,
            GraphPatchRequest(
                expected_revision=params.expected_revision,
                completed=params.completed,
            ),
        )
        return _context_result(
            tool_name=tool_name,
            graph_id=params.graph_id,
            prefix=(
                "Research Graph completion criterion marked satisfied."
                if params.completed
                else "Research Graph reopened."
            ),
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def record_research_result(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "record_research_result"
    graph_id = str(payload.get("graph_id") or "")
    try:
        params = RecordResearchResultInput.model_validate(payload)
        request = ResultCreateRequest.model_validate(
            params.model_dump(mode="json", exclude={"graph_id"})
        )
        result = _service().record_result(params.graph_id, request)
        return _context_result(
            tool_name=tool_name,
            graph_id=params.graph_id,
            focus_node_id=result["node"]["node_id"],
            prefix=f"Recorded result {result['node']['title']}.",
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def set_research_result_judgment(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    tool_name = "set_research_result_judgment"
    graph_id = str(payload.get("graph_id") or "")
    try:
        params = SetResearchResultJudgmentInput.model_validate(payload)
        request = ResultJudgmentSetRequest.model_validate(
            params.model_dump(
                mode="json",
                exclude={"graph_id", "result_node_id", "hypothesis_node_id"},
            )
        )
        _service().set_result_judgment(
            params.graph_id,
            params.result_node_id,
            params.hypothesis_node_id,
            request,
        )
        return _context_result(
            tool_name=tool_name,
            graph_id=params.graph_id,
            focus_node_id=params.result_node_id,
            prefix=(
                f"Result judgment set to {params.relation} for hypothesis "
                f"{params.hypothesis_node_id}."
            ),
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def mark_research_experiment_failed(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    tool_name = "mark_research_experiment_failed"
    graph_id = str(payload.get("graph_id") or "")
    try:
        params = MarkResearchExperimentFailedInput.model_validate(payload)
        service = _service()
        refs = [service.validate_ref(ref) for ref in params.refs]
        node, _event_id = service.store.mark_experiment_blocked(
            params.graph_id,
            params.experiment_node_id,
            expected_revision=params.expected_revision,
            reason=params.reason,
            refs=refs,
        )
        return _context_result(
            tool_name=tool_name,
            graph_id=params.graph_id,
            focus_node_id=node["node_id"],
            prefix=f"Marked experiment blocked: {params.reason}",
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def record_bound_research_result(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    tool_name = "record_bound_research_result"
    graph_id = ""
    try:
        params = RecordBoundResearchResultInput.model_validate(payload)
        service = _service()
        thread_id, graph_id, experiment_node_id = _bound_execution_target(service)
        refs = _refs_with_bound_sources(params.refs, thread_id=thread_id)
        for attempt in range(2):
            graph = service.store.get_graph(graph_id)
            try:
                result = service.record_result(
                    graph_id,
                    ResultCreateRequest(
                        expected_revision=int(graph["revision"]),
                        title=params.title,
                        summary=params.summary,
                        experiment_node_id=experiment_node_id,
                        judgments=params.judgments,
                        refs=refs,
                    ),
                )
                break
            except ResearchGraphConflict:
                if attempt:
                    raise
        return _context_result(
            tool_name=tool_name,
            graph_id=graph_id,
            focus_node_id=result["node"]["node_id"],
            prefix=(
                f"Recorded the bound experiment result {result['node']['title']} "
                f"with child thread source {thread_id}."
            ),
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def mark_bound_research_experiment_failed(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    tool_name = "mark_bound_research_experiment_failed"
    graph_id = ""
    try:
        params = MarkBoundResearchExperimentFailedInput.model_validate(payload)
        service = _service()
        thread_id, graph_id, experiment_node_id = _bound_execution_target(service)
        refs = [
            service.validate_ref(ref)
            for ref in _refs_with_bound_sources(params.refs, thread_id=thread_id)
        ]
        for attempt in range(2):
            graph = service.store.get_graph(graph_id)
            try:
                node, _event_id = service.store.mark_experiment_blocked(
                    graph_id,
                    experiment_node_id,
                    expected_revision=int(graph["revision"]),
                    reason=params.reason,
                    refs=refs,
                )
                break
            except ResearchGraphConflict:
                if attempt:
                    raise
        return _context_result(
            tool_name=tool_name,
            graph_id=graph_id,
            focus_node_id=node["node_id"],
            prefix=(
                f"Marked the bound experiment blocked and attached child "
                f"thread source {thread_id}: {params.reason}"
            ),
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


__all__ = [
    "AddResearchExperimentInput",
    "AddResearchHypothesisInput",
    "CreateResearchGraphInput",
    "InspectResearchGraphInput",
    "ListResearchGraphsInput",
    "MarkBoundResearchExperimentFailedInput",
    "MarkResearchExperimentFailedInput",
    "RecordBoundResearchResultInput",
    "RecordResearchResultInput",
    "SetResearchResultJudgmentInput",
    "SetResearchGraphCompletionInput",
    "StageResearchPlanInput",
    "add_research_experiment",
    "add_research_hypothesis",
    "create_research_graph",
    "inspect_research_graph",
    "list_research_graphs",
    "mark_bound_research_experiment_failed",
    "mark_research_experiment_failed",
    "record_bound_research_result",
    "record_research_result",
    "set_research_result_judgment",
    "set_research_graph_completion",
    "stage_research_plan",
]
