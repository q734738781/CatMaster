from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from catmaster.research.knowledge_graph.models import (
    ExperimentCreateRequest,
    GraphCreateRequest,
    GraphPatchRequest,
    HypothesisCreateRequest,
    ResearchExperimentEvaluationDraft,
    ResearchGraphPlanningDraft,
    ResearchRefInput,
    ResultJudgmentInput,
    ResultCreateRequest,
    ResultJudgmentSetRequest,
)
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.research.knowledge_graph.query import ResearchGraphSQLQuery
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
        description=(
            "Optional human-readable scientific completion criterion. Leave empty "
            "to use the default defensible-answer criterion."
        ),
    )


class QueryResearchGraphSQLInput(BaseModel):
    """[research/graph] Query the current thread-bound graph through read-only logical SQLite tables."""

    model_config = ConfigDict(extra="forbid")

    sql: str = Field(
        ...,
        min_length=1,
        description=(
            "One standard SELECT or WITH statement over research_graphs, "
            "research_nodes, research_edges, research_refs, research_launches, "
            "research_planning, workspace_artifacts, or thread_messages. Use "
            "ordinary LIMIT/OFFSET or keyset pagination when desired."
        ),
    )


class AddResearchHypothesisInput(HypothesisCreateRequest):
    """[research/graph] Add one falsifiable hypothesis with graph revision CAS."""

    graph_id: str = Field(..., min_length=3, description="Explicit target graph ID.")


class AddResearchExperimentInput(ExperimentCreateRequest):
    """[research/graph] Add a draft or ready experiment proposal with graph revision CAS."""

    graph_id: str = Field(..., min_length=3, description="Explicit target graph ID.")


class StageResearchPlanInput(ResearchGraphPlanningDraft):
    """[research/graph] Publish science-first temporary branches from the bound planning turn."""


class EvaluateResearchExperimentsInput(ResearchExperimentEvaluationDraft):
    """[research/graph] Publish flat dual scores for every current candidate Experiment."""


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
    reason: str = Field(..., min_length=1, description="Concrete execution blocker.")
    refs: list[ResearchRefInput] = Field(
        default_factory=list,
        description="Typed source refs; omit or pass [] when no durable source exists.",
    )


class RecordBoundResearchResultInput(BaseModel):
    """[research/graph] Record one concise Result in this turn's bound graph."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field("", description="Short result title; leave empty to derive it from the summary.")
    summary: str = Field(
        ...,
        min_length=1,
        description=(
            "Concise observed or derived scientific outcome, including what the "
            "decision rule needs. Separate observation from causal interpretation "
            "and state material conditions or provenance when they affect meaning; "
            "do not assign a global evidence grade."
        ),
    )
    judgments: list[ResultJudgmentInput] = Field(
        default_factory=list,
        description="Typed effects on the hypothesis node IDs shown in the bound graph context; omit or pass [] when the result is not discriminating.",
    )
    refs: list[ResearchRefInput] = Field(
        default_factory=list,
        description=(
            "Durable result sources such as artifact, note, DOI, or URL refs. "
            "The host attaches the current turn's thread and run refs automatically."
        ),
    )


class MarkBoundResearchExperimentFailedInput(BaseModel):
    """[research/graph] Mark this turn's focused graph Experiment as blocked."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(
        ...,
        min_length=1,
        description="Concrete reason the bound experiment cannot produce a scientific result.",
    )
    refs: list[ResearchRefInput] = Field(
        default_factory=list,
        description=(
            "Optional durable sources for the blocker. The host attaches the "
            "current turn's thread and run refs automatically."
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
) -> tuple[str, str, str, str]:
    runtime_context = current_tool_context()
    entrypoint = str(runtime_context.get("entrypoint") or "").strip()
    if entrypoint not in {"experiment", "literature_review"}:
        raise ValueError(
            "This bound writeback tool is available only to a top-level "
            "Experiment or Literature Review turn."
        )
    thread_id = str(runtime_context.get("thread_id") or "").strip()
    if not thread_id:
        raise ValueError("This turn has no trusted runtime thread binding.")
    graph_id = str(runtime_context.get("research_graph_id") or "").strip()
    focus_node_id = str(
        runtime_context.get("research_focus_node_id") or ""
    ).strip()
    launch_id = str(runtime_context.get("research_launch_id") or "").strip()
    if not graph_id:
        raise ValueError("This turn is not bound to a Research Graph.")
    service.store.get_graph(graph_id)
    experiment_node_id = ""
    if focus_node_id:
        node = service.store.get_node(graph_id, focus_node_id)
        if node["kind"] == "experiment":
            experiment_node_id = focus_node_id
    if launch_id:
        launch = service.store.get_launch(launch_id)
        if str(launch.get("status") or "") not in {
            "claimed",
            "submitting",
            "running",
            "unknown",
        }:
            raise ValueError("The turn-bound research launch is no longer active.")
        if (
            str(launch.get("thread_id") or "") != thread_id
            or str(launch.get("graph_id") or "") != graph_id
            or str(launch.get("experiment_node_id") or "")
            != experiment_node_id
        ):
            raise ValueError(
                "The turn-bound launch does not match this graph execution target."
            )
    return thread_id, graph_id, experiment_node_id, launch_id


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


def _mutation_result(
    *,
    service: ResearchGraphService,
    tool_name: str,
    graph_id: str,
    changed: dict[str, Any],
    message: str,
) -> tuple[str, dict[str, Any]]:
    revision = int(service.store.get_graph(graph_id)["revision"])
    data = {
        "graph_id": graph_id,
        "revision": revision,
        "changed": changed,
    }
    return f"{message}\nLatest graph revision: {revision}.", _artifact(
        tool_name,
        data,
    )


def _bound_graph(service: ResearchGraphService) -> tuple[str, int]:
    runtime_context = current_tool_context()
    if "research_graph_id" in runtime_context:
        graph_id = str(runtime_context.get("research_graph_id") or "").strip()
        if not graph_id:
            raise ValueError("The current turn is not bound to a Research Graph.")
        graph = service.store.get_graph(graph_id)
        return graph_id, int(graph["revision"])
    thread_id = _trusted_thread_id()
    if not thread_id:
        raise ValueError("The graph query has no trusted runtime thread binding.")
    planning = service.store.find_planning_by_thread(thread_id)
    if planning is not None:
        graph_id = str(planning["graph_id"])
    else:
        thread = service.thread_store.get_thread(thread_id)
        graph_id = str(thread.active_research_graph_id or "").strip()
    if not graph_id:
        raise ValueError("The current thread is not bound to a Research Graph.")
    graph = service.store.get_graph(graph_id)
    return graph_id, int(graph["revision"])


def _runtime_launch_for_target(
    *,
    graph_id: str,
    experiment_node_id: str,
) -> str | None:
    runtime_context = current_tool_context()
    if "research_launch_id" not in runtime_context:
        return None
    launch_id = str(runtime_context.get("research_launch_id") or "").strip()
    if not launch_id:
        return ""
    if (
        str(runtime_context.get("research_graph_id") or "").strip() != graph_id
        or str(runtime_context.get("research_focus_node_id") or "").strip()
        != experiment_node_id
    ):
        return ""
    return launch_id


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
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=graph_id,
            changed={
                "graph": result["graph"],
                "initial_nodes": result.get("nodes") or [],
                "bound_thread_id": bound_thread_id,
            },
            message=(
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


def query_research_graph_sql(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "query_research_graph_sql"
    graph_id = ""
    try:
        params = QueryResearchGraphSQLInput.model_validate(payload)
        service = _service()
        graph_id, revision = _bound_graph(service)
        result = ResearchGraphSQLQuery(service.workspace).execute(
            graph_id=graph_id,
            sql=params.sql,
        )
        data = {
            "graph_id": graph_id,
            "revision": revision,
            **result,
        }
        return json.dumps(data, ensure_ascii=False), _artifact(
            tool_name,
            data,
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
        service = _service()
        result = service.add_hypothesis(params.graph_id, request)
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=params.graph_id,
            changed={
                "node": result["node"],
                "refs": [ref.model_dump(mode="json") for ref in request.refs],
                "suggested_by_result_ids": request.suggested_by_result_ids,
            },
            message=f"Added hypothesis {result['node']['title']}.",
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
        service = _service()
        result = service.add_experiment(params.graph_id, request)
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=params.graph_id,
            changed={
                "node": result["node"],
                "refs": [ref.model_dump(mode="json") for ref in request.refs],
                "tests_hypothesis_ids": request.tests_hypothesis_ids,
                "depends_on_experiment_ids": request.depends_on_experiment_ids,
            },
            message=f"Added experiment proposal {result['node']['title']}.",
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
        return summary, _artifact(
            tool_name,
            {
                "graph_id": graph_id,
                "planning_id": result["planning_id"],
                "revision": result["revision"],
                "candidate_experiment_ids": result["candidate_experiment_ids"],
                "staged": result["staged"],
            },
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def evaluate_research_experiments(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    tool_name = "evaluate_research_experiments"
    graph_id = ""
    try:
        params = EvaluateResearchExperimentsInput.model_validate(payload)
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
        result = service.stage_planning_evaluation(
            graph_id,
            expected_revision=int(planning["revision"]),
            planning_thread_id=planning_thread_id,
            evaluation=ResearchExperimentEvaluationDraft.model_validate(
                params.model_dump(mode="json")
            ),
        )
        content = (
            "Published current-revision Experiment evaluation. "
            f"Innovation recommendation: {result['innovation_recommendation'] or 'none'}; "
            f"conservative recommendation: "
            f"{result['conservative_recommendation'] or 'none'}."
        )
        return content, _artifact(tool_name, {"graph_id": graph_id, **result})
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
        result = service.patch_graph(
            params.graph_id,
            GraphPatchRequest(
                expected_revision=params.expected_revision,
                completed=params.completed,
            ),
        )
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=params.graph_id,
            changed={"graph": result["graph"]},
            message=(
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
        service = _service()
        result = service.record_result(
            params.graph_id,
            request,
            launch_id=_runtime_launch_for_target(
                graph_id=params.graph_id,
                experiment_node_id=request.experiment_node_id,
            ),
            run_id=str(current_tool_context().get("run_id") or "").strip(),
        )
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=params.graph_id,
            changed={
                "node": result["node"],
                "experiment_node_id": request.experiment_node_id,
                "judgments": [
                    item.model_dump(mode="json") for item in request.judgments
                ],
                "refs": [ref.model_dump(mode="json") for ref in request.refs],
            },
            message=f"Recorded result {result['node']['title']}.",
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
        service = _service()
        service.set_result_judgment(
            params.graph_id,
            params.result_node_id,
            params.hypothesis_node_id,
            request,
        )
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=params.graph_id,
            changed={
                "result_node_id": params.result_node_id,
                "hypothesis_node_id": params.hypothesis_node_id,
                "relation": params.relation,
            },
            message=(
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
            launch_id=_runtime_launch_for_target(
                graph_id=params.graph_id,
                experiment_node_id=params.experiment_node_id,
            ),
            run_id=str(current_tool_context().get("run_id") or "").strip(),
        )
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=params.graph_id,
            changed={
                "node": service._public_node(node),
                "refs": [ref.model_dump(mode="json") for ref in params.refs],
            },
            message=f"Marked experiment blocked: {params.reason}",
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
        thread_id, graph_id, experiment_node_id, launch_id = (
            _bound_execution_target(service)
        )
        refs = _refs_with_bound_sources(params.refs, thread_id=thread_id)
        graph = service.store.get_graph(graph_id)
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
            launch_id=launch_id,
            run_id=str(current_tool_context().get("run_id") or "").strip(),
        )
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=graph_id,
            changed={
                "node": result["node"],
                "experiment_node_id": experiment_node_id,
                "judgments": [
                    item.model_dump(mode="json") for item in params.judgments
                ],
                "refs": [ref.model_dump(mode="json") for ref in refs],
            },
            message=(
                f"Recorded the bound research result {result['node']['title']} "
                f"with turn source {thread_id}."
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
        thread_id, graph_id, experiment_node_id, launch_id = (
            _bound_execution_target(service)
        )
        if not experiment_node_id:
            raise ValueError(
                "A bound blocker requires this turn to focus an Experiment."
            )
        refs = [
            service.validate_ref(ref)
            for ref in _refs_with_bound_sources(params.refs, thread_id=thread_id)
        ]
        graph = service.store.get_graph(graph_id)
        node, _event_id = service.store.mark_experiment_blocked(
            graph_id,
            experiment_node_id,
            expected_revision=int(graph["revision"]),
            reason=params.reason,
            refs=refs,
            launch_id=launch_id,
            run_id=str(current_tool_context().get("run_id") or "").strip(),
        )
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=graph_id,
            changed={
                "node": service._public_node(node),
                "refs": [ref.model_dump(mode="json") for ref in refs],
            },
            message=(
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
    "EvaluateResearchExperimentsInput",
    "ListResearchGraphsInput",
    "MarkBoundResearchExperimentFailedInput",
    "MarkResearchExperimentFailedInput",
    "QueryResearchGraphSQLInput",
    "RecordBoundResearchResultInput",
    "RecordResearchResultInput",
    "SetResearchResultJudgmentInput",
    "SetResearchGraphCompletionInput",
    "StageResearchPlanInput",
    "add_research_experiment",
    "add_research_hypothesis",
    "create_research_graph",
    "evaluate_research_experiments",
    "list_research_graphs",
    "mark_bound_research_experiment_failed",
    "mark_research_experiment_failed",
    "query_research_graph_sql",
    "record_bound_research_result",
    "record_research_result",
    "set_research_result_judgment",
    "set_research_graph_completion",
    "stage_research_plan",
]
