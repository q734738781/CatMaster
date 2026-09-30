from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from catmaster.research.knowledge_graph.models import (
    BoundExperimentCreateRequest,
    BoundExperimentResumeRequest,
    BoundGraphScopeUpdateRequest,
    BoundResultRetractRequest,
    BoundResultUpdateRequest,
    ExperimentCreateRequest,
    GraphCreateRequest,
    GraphPatchRequest,
    HypothesisCreateRequest,
    ResearchExperimentPairOutcomeDraft,
    ResearchGraphPlanningDraft,
    ResearchGraphFocusRequest,
    ResearchRefInput,
    ResultJudgmentInput,
    ResultCreateRequest,
    ResultJudgmentSetRequest,
    ScientificRevisionRequest,
    ResearchDispositionRequest,
    ResearchReviewRequest,
)
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.research.knowledge_graph.query import ResearchGraphSQLQuery
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.runtime.tool_runtime import current_tool_context, current_tool_audience
from catmaster.tools.base import project_space_root


class ListResearchGraphsInput(BaseModel):
    """[research/graph] Browse workspace Research Graphs and return bound_graph_id for the current invocation. Listing does not select or change the binding."""

    model_config = ConfigDict(extra="forbid")

    offset: int = Field(0, ge=0, description="Zero-based offset after filtering; use next_offset to continue.")
    limit: int = Field(0, ge=0, description="Number of graphs to return; 0 returns all remaining graphs.")
    query: str = Field("", description="Case-insensitive substring in graph title or question; empty matches all.")

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
    """[research/graph] Query the current invocation's bound graph through read-only logical SQLite tables; returns its graph_id. This does not select or rebind a graph."""

    model_config = ConfigDict(extra="forbid")

    sql: str = Field(
        ...,
        min_length=1,
        description=(
            "One read-only SELECT or WITH statement over these exact logical "
            "schemas: research_graphs(graph_id, title, question, "
            "completion_criterion, decision_preferences, completed, "
            "orchestration_mode, archived, revision, created_at, updated_at); "
            "research_nodes(graph_id, "
            "node_id, kind, title, state, body_json, revision, created_at, "
            "updated_at); research_edges(graph_id, source_node_id, "
            "target_node_id, relation, scope, rationale, action); research_refs(graph_id, node_id, "
            "ref_kind, ref_id); research_launches(launch_id, graph_id, "
            "experiment_node_id, idempotency_key, status, thread_id, run_id, "
            "lease_owner, lease_until, created_at, updated_at); "
            "research_planning(planning_id, graph_id, start_revision, status, "
            "thread_id, preview_json, lease_until, created_at, updated_at); "
            "research_decisions(decision_id, graph_id, thread_id, run_id, body_json, review_json, created_at, updated_at); "
            "research_discussions(seq, message_id, graph_id, discussion_id, title, body, author_thread_id, "
            "author_kind, target_task_id, node_id, reply_to, references_json, created_at); "
            "workspace_artifacts(artifact_id, thread_id, payload_json, "
            "created_at, updated_at); thread_messages(row_id, thread_id, "
            "message_id, created_at, updated_at, payload_json, message_role, "
            "message_run_id). Node details such as claim, summary, objective, "
            "or decision_rule are JSON keys in body_json, not columns; for "
            "example use json_extract(n.body_json, '$.claim'). Artifact path, "
            "mime_type, title, renderer, and summary are JSON keys in "
            "payload_json; for example use json_extract(a.payload_json, "
            "'$.path'). Message content is likewise in payload_json. Do not "
            "query sqlite_master or qualify tables with main. Queries use this "
            "invocation's graph binding supplied in the current turn. Titles, completion "
            "states and earlier bindings do not change the query target. Use ordinary LIMIT/OFFSET or keyset pagination when "
            "desired."
        ),
    )


class AddResearchHypothesisInput(HypothesisCreateRequest):
    """[research/graph] Add one falsifiable hypothesis to the bound graph using its latest inspected revision."""


class ReviseResearchClaimInput(ScientificRevisionRequest):
    """[research/graph] In the bound graph, link a new H/R to the older same-kind claim it replaces, qualifies or withdraws; retain both records."""


class RecordResearchDispositionInput(ResearchDispositionRequest):
    """[research/graph] Declare why an unfinished stage is stopping or continuing. Stalled persistent research receives one independent review before parking. Reuse an existing issue; do not repeat review on unchanged evidence."""


class RecordResearchReviewInput(ResearchReviewRequest):
    """[research/graph] Record an independent review of one stopping decision and optionally reference an authorized validation Experiment. This call records the assessment; it does not execute the Experiment or grant calculation authorization."""


class AddResearchExperimentInput(ExperimentCreateRequest):
    """[research/graph] Create a new draft or ready Experiment in the bound graph using its latest inspected revision. Each successful call creates a new ID; reuse an existing Experiment for continued work."""


class StageResearchPlanInput(ResearchGraphPlanningDraft):
    """[research/graph] Publish science-first temporary branches from the bound planning turn."""


class MarkResearchPlanningNoChangeInput(BaseModel):
    """[research/graph] Record that this planning pass adds no justified branch."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(
        ...,
        min_length=1,
        description=(
            "Concise evidence-grounded reason why this pass adds no justified "
            "hypothesis or experiment branch. Existing ready Experiments remain "
            "eligible for comparison and execution."
        ),
    )


class RecordResearchExperimentComparisonInput(ResearchExperimentPairOutcomeDraft):
    """[research/graph] Publish one clean A/B outcome from the bound pair turn."""


class SetResearchGraphCompletionInput(BaseModel):
    """[research/graph] Mark whether scientific state satisfies the bound graph's completion criterion."""

    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(..., ge=1, description="Latest inspected graph revision.")
    completed: bool = Field(
        ...,
        description=(
            "True only when recorded Results, sources, and any explicitly required "
            "deliverable or external handoff satisfy the completion criterion; "
            "at least one actual Result must already be recorded. False reopens the graph."
        ),
    )


class RecordResearchResultInput(ResultCreateRequest):
    """[research/graph] Record one concise observation or result, source refs, and typed judgments in the bound graph."""


class SetResearchResultJudgmentInput(ResultJudgmentSetRequest):
    """[research/graph] Replace one Result-to-Hypothesis evidence judgment in the bound graph."""

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
    """[research/graph] Mark an experiment in the bound graph blocked with a concrete reason and optional source refs."""

    model_config = ConfigDict(extra="forbid")

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

    methods: str = Field("", description=ResultCreateRequest.model_fields["methods"].description)
    conclusion: str = Field("", description=ResultCreateRequest.model_fields["conclusion"].description)

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
            "Current-conversation source refs are attached automatically; do not supply them manually."
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
            "Optional durable sources for the blocker. Current-conversation "
            "source refs are attached automatically; do not supply them manually."
        ),
    )


class SetResearchGraphFocusInput(ResearchGraphFocusRequest):
    """[research/graph] Select or clear the focus node in this conversation's graph."""


class CreateBoundResearchExperimentInput(BoundExperimentCreateRequest):
    """[research/graph] Create a NEW Experiment in the bound graph and focus this thread on it. Each successful call creates a new ID. Query and focus existing work; resume a blocked Experiment instead of registering it again."""


class UpdateBoundResearchResultInput(BoundResultUpdateRequest):
    """[research/graph] Correct the same scientific Result produced by the focused Experiment."""


class ResumeBoundResearchExperimentInput(BoundExperimentResumeRequest):
    """[research/graph] Resume the focused blocked Experiment after its blocker is removed."""


class RetractBoundResearchResultInput(BoundResultRetractRequest):
    """[research/graph] Retract one same-run category-error Result from the focused Experiment."""


class UpdateResearchGraphScopeInput(BoundGraphScopeUpdateRequest):
    """[research/graph] Apply an explicit user-directed scope correction to the bound graph."""


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
    *,
    allow_standalone_literature_result: bool = False,
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
    try:
        thread = service.thread_store.get_thread(thread_id)
    except KeyError:
        thread = None
    if thread is not None:
        if str(thread.active_research_graph_id or "") != graph_id:
            raise ValueError("The current thread no longer has this Research Graph binding.")
        focus_node_id = str(thread.research_focus_node_id or "").strip()
    experiment_node_id = ""
    if focus_node_id:
        node = service.store.get_node(graph_id, focus_node_id)
        if node["kind"] == "experiment":
            experiment_node_id = focus_node_id
    if not experiment_node_id:
        if allow_standalone_literature_result and entrypoint == "literature_review":
            if launch_id:
                raise ValueError(
                    "A turn-bound Experiment launch cannot write a standalone "
                    "literature Result."
                )
            return thread_id, graph_id, "", ""
        raise ValueError(
            "Bound Result writeback requires an explicit Experiment focus. "
            "Set graph focus or explicitly adopt one Experiment first."
        )
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
    handles: list[str] = []
    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("node_id"):
                handles.append(str(value["node_id"]))
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    collect(changed)
    handle_text = "\nnode_ids=" + json.dumps(list(dict.fromkeys(handles))) if handles else ""
    return f"{message}\ngraph_id={graph_id}\nLatest graph revision: {revision}.{handle_text}", _artifact(
        tool_name,
        data,
    )


def _bound_graph_id(service: ResearchGraphService) -> str:
    """Resolve the same accepted binding for catalog and query surfaces."""
    runtime_context = current_tool_context()
    if "research_graph_id" in runtime_context:
        return str(runtime_context.get("research_graph_id") or "").strip()
    thread_id = _trusted_thread_id()
    if not thread_id:
        return ""
    planning = service.store.find_planning_by_thread(thread_id)
    if planning is not None:
        return str(planning["graph_id"])
    try:
        thread = service.thread_store.get_thread(thread_id)
    except KeyError:
        # A CLI checkpoint name need not identify a WebUI thread. Catalog
        # browsing remains available without inventing a binding for it.
        return ""
    return str(thread.active_research_graph_id or "").strip()


def _bound_graph(service: ResearchGraphService) -> tuple[str, int]:
    graph_id = _bound_graph_id(service)
    if not graph_id:
        if "research_graph_id" in current_tool_context():
            raise ValueError("The current turn is not bound to a Research Graph.")
        if not _trusted_thread_id():
            raise ValueError("The graph query has no trusted runtime thread binding.")
        raise ValueError("The current thread is not bound to a Research Graph.")
    graph = service.store.get_graph(graph_id)
    return graph_id, int(graph["revision"])


def _bound_mutation_payload(
    service: ResearchGraphService, payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Use the accepted graph for agent writes; retain direct-call compatibility."""
    values = dict(payload)
    legacy_graph_id = str(values.pop("graph_id", "") or "").strip()
    if "research_graph_id" in current_tool_context() or _trusted_thread_id():
        graph_id, _revision = _bound_graph(service)
        if legacy_graph_id and legacy_graph_id != graph_id:
            raise ValueError("The requested graph differs from this turn's bound Research Graph.")
    else:
        # Historical host callers can still pass an explicit graph. Active
        # agent schemas omit it, and accepted turns always carry their binding.
        graph_id = legacy_graph_id
        if not graph_id:
            raise ValueError("This operation requires a bound Research Graph.")
        service.store.get_graph(graph_id)
    return graph_id, values


def _is_comparison_thread(service: ResearchGraphService) -> bool:
    thread_id = _trusted_thread_id()
    if not thread_id:
        return False
    from catmaster.webui.thread_models import ThreadRole
    try:
        thread = service.thread_store.get_thread(thread_id)
    except KeyError:
        return False
    if thread.thread_role != ThreadRole.RESEARCH_COMPARISON:
        return False
    # Keep the restricted scientific query subset for the whole isolated
    # invocation, including any model turn after the outcome tool has cleared
    # active_comparison. The planning record retains the thread handle.
    return service.store.find_planning_by_comparison_thread(thread_id) is not None


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
        service = _service()
        bound_graph_id = _bound_graph_id(service)
        rows = service.catalog(include_archived=params.include_archived)
        if params.query:
            needle = params.query.casefold()
            rows = [row for row in rows if needle in (row["title"] + " " + row["question"]).casefold()]
        rows.sort(key=lambda row: row["graph_id"])
        total = len(rows)
        rows = rows[params.offset:params.offset + params.limit if params.limit else None]
        next_offset = params.offset + len(rows) if params.offset + len(rows) < total else None
        if not rows:
            content = "No Research Graph matches this page."
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
        content = f"Current turn graph binding: {bound_graph_id or 'none'}\n" + content
        content += f"\nmatched_count={total} offset={params.offset} next_offset={next_offset}"
        return content, _artifact(
            tool_name,
            {
                "bound_graph_id": bound_graph_id,
                "matched_count": total, "offset": params.offset, "next_offset": next_offset,
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
        result = service.create_graph(
            params,
            session_root_thread_id=_trusted_thread_id(),
        )
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
            excluded_tables=(
                {"research_planning", "research_launches", "thread_messages"}
                if _is_comparison_thread(service)
                else set()
            ),
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


def set_research_graph_focus(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "set_research_graph_focus"
    graph_id = ""
    try:
        params = SetResearchGraphFocusInput.model_validate(payload)
        runtime_context = current_tool_context()
        entrypoint = str(runtime_context.get("entrypoint") or "").strip()
        if entrypoint not in {
            "research",
            "persistent_research",
            "experiment",
            "literature_review",
            "writing",
        }:
            raise ValueError("This role cannot change Research Graph focus.")
        thread_id = _trusted_thread_id()
        if not thread_id:
            raise ValueError("The current turn has no trusted thread binding.")
        service = _service()
        graph_id, revision = _bound_graph(service)
        result = service.set_thread_focus(
            thread_id=thread_id,
            graph_id=graph_id,
            node_id=params.node_id,
        )
        data = {
            "graph_id": graph_id,
            "revision": revision,
            "focus": result["focus"],
            "neighbors": result["neighbors"],
            "edges": result["edges"],
        }
        message = (
            f"Focused this thread on {params.node_id}."
            if params.node_id
            else "Cleared this thread's Research Graph focus."
        )
        return message + "\n" + json.dumps(data, ensure_ascii=False), _artifact(
            tool_name,
            data,
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def create_bound_research_experiment(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    tool_name = "create_bound_research_experiment"
    graph_id = ""
    try:
        params = CreateBoundResearchExperimentInput.model_validate(payload)
        runtime_context = current_tool_context()
        if str(runtime_context.get("entrypoint") or "").strip() != "experiment":
            raise ValueError(
                "Only a top-level Experiment turn may explicitly adopt new work into a bound graph."
            )
        thread_id = _trusted_thread_id()
        if not thread_id:
            raise ValueError("The current turn has no trusted thread binding.")
        service = _service()
        graph_id, revision = _bound_graph(service)
        refs = _refs_with_bound_sources(params.refs, thread_id=thread_id)
        request = BoundExperimentCreateRequest.model_validate(
            {
                **params.model_dump(mode="json", exclude={"refs"}),
                "refs": [ref.model_dump(mode="json") for ref in refs],
            }
        )
        result = service.add_bound_experiment(
            thread_id=thread_id,
            graph_id=graph_id,
            expected_revision=revision,
            request=request,
        )
        node = result["node"]
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=graph_id,
            changed={
                "node": node,
                "focus_node_id": node["node_id"],
                "refs": [ref.model_dump(mode="json") for ref in refs],
            },
            message=f"Created and focused Experiment {node['title']}.",
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def update_bound_research_result(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    tool_name = "update_bound_research_result"
    graph_id = ""
    try:
        params = UpdateBoundResearchResultInput.model_validate(payload)
        service = _service()
        thread_id, graph_id, experiment_node_id, _launch_id = (
            _bound_execution_target(
                service,
                allow_standalone_literature_result=True,
            )
        )
        # Updating an observation must preserve its creation ownership. Append
        # only the evidence refs the caller supplied; do not relabel an older
        # Result as created by the current run/thread.
        refs = list(params.refs)
        result = service.update_bound_result(
            graph_id=graph_id,
            experiment_node_id=experiment_node_id,
            result_node_id=params.result_node_id,
            summary=params.summary,
            methods=params.methods if "methods" in params.model_fields_set else None,
            conclusion=params.conclusion if "conclusion" in params.model_fields_set else None,
            title=params.title,
            refs=refs,
        )
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=graph_id,
            changed={
                "node": result["node"],
                "refs_appended": [ref.model_dump(mode="json") for ref in refs],
            },
            message=f"Updated Result {params.result_node_id} in place.",
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def resume_bound_research_experiment(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    tool_name = "resume_bound_research_experiment"
    graph_id = ""
    try:
        params = ResumeBoundResearchExperimentInput.model_validate(payload)
        service = _service()
        thread_id, graph_id, experiment_node_id, _launch_id = _bound_execution_target(
            service
        )
        refs = _refs_with_bound_sources(params.refs, thread_id=thread_id)
        result = service.resume_bound_experiment(
            graph_id=graph_id,
            experiment_node_id=experiment_node_id,
            reason=params.reason,
            refs=refs,
        )
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=graph_id,
            changed={
                "node": result["node"],
                "refs_appended": [ref.model_dump(mode="json") for ref in refs],
                "reason": params.reason,
            },
            message=f"Resumed Experiment {experiment_node_id}.",
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def retract_bound_research_result(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    tool_name = "retract_bound_research_result"
    graph_id = ""
    try:
        params = RetractBoundResearchResultInput.model_validate(payload)
        service = _service()
        thread_id, graph_id, experiment_node_id, _launch_id = (
            _bound_execution_target(
                service,
                allow_standalone_literature_result=True,
            )
        )
        run_id = str(current_tool_context().get("run_id") or "").strip()
        if not run_id:
            raise ValueError("The current turn has no trusted run ownership.")
        graph = service.store.get_graph(graph_id)
        result_node = service.store.get_node(graph_id, params.result_node_id)
        result = service.retract_result(
            graph_id=graph_id,
            result_node_id=params.result_node_id,
            expected_revision=int(graph["revision"]),
            expected_node_revision=int(result_node["revision"]),
            reason=params.reason,
            focused_experiment_node_id=experiment_node_id,
            owner_run_id=run_id,
            owner_thread_id=thread_id,
        )
        data = {
            "graph_id": graph_id,
            "revision": int(result["graph"]["revision"]),
            "deleted_result": result["deleted_result"],
            "changed_experiment_ids": result["changed_experiment_ids"],
            "event_id": result["event_id"],
        }
        return f"Retracted Result {params.result_node_id}.\nLatest graph revision: {data['revision']}.", _artifact(
            tool_name,
            data,
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def update_research_graph_scope(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    tool_name = "update_research_graph_scope"
    graph_id = ""
    try:
        params = UpdateResearchGraphScopeInput.model_validate(payload)
        if str(current_tool_context().get("entrypoint") or "").strip() not in {
            "research",
            "persistent_research",
        }:
            raise ValueError(
                "Only Research or Persistent Research may change a graph's "
                "scientific scope."
            )
        service = _service()
        graph_id, revision = _bound_graph(service)
        changes = {
            key: value
            for key, value in params.model_dump(mode="json").items()
            if str(value or "").strip()
        }
        request = GraphPatchRequest.model_validate(
            {"expected_revision": revision, **changes}
        )
        result = service.patch_graph(graph_id, request)
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=graph_id,
            changed={"graph": result["graph"]},
            message="Updated the explicit Research Graph scope.",
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def add_research_hypothesis(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "add_research_hypothesis"
    graph_id = str(payload.get("graph_id") or "")
    try:
        service = _service()
        graph_id, payload = _bound_mutation_payload(service, payload)
        params = AddResearchHypothesisInput.model_validate(payload)
        request = HypothesisCreateRequest.model_validate(
            params.model_dump(mode="json")
        )
        result = service.add_hypothesis(graph_id, request)
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=graph_id,
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
        service = _service()
        graph_id, payload = _bound_mutation_payload(service, payload)
        params = AddResearchExperimentInput.model_validate(payload)
        request = ExperimentCreateRequest.model_validate(
            params.model_dump(mode="json")
        )
        result = service.add_experiment(graph_id, request)
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=graph_id,
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


def mark_research_planning_no_change(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """[research/graph] End one planning pass without inventing a new branch."""

    tool_name = "mark_research_planning_no_change"
    graph_id = ""
    try:
        params = MarkResearchPlanningNoChangeInput.model_validate(payload)
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
        result = service.mark_planning_no_change(
            graph_id,
            expected_revision=int(planning["revision"]),
            planning_thread_id=planning_thread_id,
            reason=params.reason,
        )
        if result.get("selection_started"):
            summary = (
                "No new branch was added; comparator selection is continuing over "
                f"{len(result.get('candidate_experiment_ids') or [])} existing ready "
                f"Experiment(s): {result['reason']}"
            )
        else:
            summary = (
                "No new or existing ready branch is available in this revision: "
                f"{result['reason']}"
            )
        return summary, _artifact(tool_name, {"graph_id": graph_id, **result})
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def record_research_experiment_comparison(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    tool_name = "record_research_experiment_comparison"
    graph_id = ""
    try:
        params = RecordResearchExperimentComparisonInput.model_validate(payload)
        comparison_thread_id = _trusted_thread_id()
        if not comparison_thread_id:
            raise ValueError("The comparison turn has no trusted thread binding.")
        service = _service()
        result = service.record_pair_comparison(
            comparison_thread_id=comparison_thread_id,
            outcome=ResearchExperimentPairOutcomeDraft.model_validate(
                params.model_dump(mode="json")
            ),
        )
        graph_id = str(result["graph_id"])
        content = "Recorded this isolated Experiment comparison."
        return content, _artifact(tool_name, {"graph_id": graph_id, **result})
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(tool_name, graph_id, exc)


def revise_research_claim(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name = 'revise_research_claim'
    graph_id = str(payload.get('graph_id') or '')
    try:
        service = _service()
        graph_id, payload = _bound_mutation_payload(service, payload)
        params = ReviseResearchClaimInput.model_validate(payload)
        service.store.revise_claim(graph_id=graph_id, **params.model_dump(mode='json'))
        return _mutation_result(service=service, tool_name=name, graph_id=graph_id,
            changed=params.model_dump(mode='json'), message='Scientific revision recorded; both claims remain queryable.')
    except Exception as exc:
        _error(name, graph_id, exc)


def record_research_disposition(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name = 'record_research_disposition'
    context = current_tool_context()
    graph_id = str(context.get('research_graph_id') or '')
    try:
        if not graph_id or not _trusted_thread_id():
            raise ValueError('A bound research session is required.')
        record = _service().store.record_disposition(graph_id,
            thread_id=_trusted_thread_id(), run_id=str(context.get('run_id') or ''), body=payload)
        return 'Research stopping decision recorded.', _artifact(name, {'decision': record})
    except Exception as exc:
        _error(name, graph_id, exc)


def record_research_review(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name = 'record_research_review'
    graph_id = str(current_tool_context().get('research_graph_id') or '')
    try:
        # Keep historical planning/checkpoint callers readable and resumable.
        if current_tool_audience() not in {'research_challenger', 'hypothesis_proposer'}:
            raise ValueError('Only the independent scientific reasoner can record this review.')
        record = _service().store.record_review(graph_id, review=payload)
        return 'Independent stopping review recorded.', _artifact(name, {'decision': record})
    except Exception as exc:
        _error(name, graph_id, exc)


def set_research_graph_completion(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    tool_name = "set_research_graph_completion"
    graph_id = str(payload.get("graph_id") or "")
    try:
        service = _service()
        graph_id, payload = _bound_mutation_payload(service, payload)
        params = SetResearchGraphCompletionInput.model_validate(payload)
        result = service.patch_graph(
            graph_id,
            GraphPatchRequest(
                expected_revision=params.expected_revision,
                completed=params.completed,
            ),
        )
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=graph_id,
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
        service = _service()
        graph_id, payload = _bound_mutation_payload(service, payload)
        params = RecordResearchResultInput.model_validate(payload)
        request = ResultCreateRequest.model_validate(
            params.model_dump(mode="json")
        )
        result = service.record_result(
            graph_id,
            request,
            launch_id=_runtime_launch_for_target(
                graph_id=graph_id,
                experiment_node_id=request.experiment_node_id,
            ),
            run_id=str(current_tool_context().get("run_id") or "").strip(),
        )
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=graph_id,
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
        service = _service()
        graph_id, payload = _bound_mutation_payload(service, payload)
        params = SetResearchResultJudgmentInput.model_validate(payload)
        request = ResultJudgmentSetRequest.model_validate(
            params.model_dump(
                mode="json",
                exclude={"result_node_id", "hypothesis_node_id"},
            )
        )
        service.set_result_judgment(
            graph_id,
            params.result_node_id,
            params.hypothesis_node_id,
            request,
        )
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=graph_id,
            changed={
                "result_node_id": params.result_node_id,
                "hypothesis_node_id": params.hypothesis_node_id,
                "relation": params.relation,
                "scope": params.scope,
                "rationale": params.rationale,
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
        service = _service()
        graph_id, payload = _bound_mutation_payload(service, payload)
        params = MarkResearchExperimentFailedInput.model_validate(payload)
        refs = [service.validate_ref(ref) for ref in params.refs]
        node, _event_id = service.store.mark_experiment_blocked(
            graph_id,
            params.experiment_node_id,
            expected_revision=params.expected_revision,
            reason=params.reason,
            refs=refs,
            launch_id=_runtime_launch_for_target(
                graph_id=graph_id,
                experiment_node_id=params.experiment_node_id,
            ),
            run_id=str(current_tool_context().get("run_id") or "").strip(),
        )
        return _mutation_result(
            service=service,
            tool_name=tool_name,
            graph_id=graph_id,
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
            _bound_execution_target(
                service,
                allow_standalone_literature_result=True,
            )
        )
        refs = _refs_with_bound_sources(params.refs, thread_id=thread_id)
        graph = service.store.get_graph(graph_id)
        result = service.record_result(
            graph_id,
            ResultCreateRequest(
                expected_revision=int(graph["revision"]),
                title=params.title,
                summary=params.summary,
                methods=params.methods,
                conclusion=params.conclusion,
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
                "refs": list(refs),
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
    "CreateBoundResearchExperimentInput",
    "RecordResearchExperimentComparisonInput",
    "ListResearchGraphsInput",
    "MarkResearchPlanningNoChangeInput",
    "MarkBoundResearchExperimentFailedInput",
    "MarkResearchExperimentFailedInput",
    "QueryResearchGraphSQLInput",
    "ResumeBoundResearchExperimentInput",
    "RetractBoundResearchResultInput",
    "RecordBoundResearchResultInput",
    "RecordResearchResultInput",
    "SetResearchResultJudgmentInput",
    "ReviseResearchClaimInput",
    "RecordResearchDispositionInput",
    "RecordResearchReviewInput",
    "SetResearchGraphFocusInput",
    "SetResearchGraphCompletionInput",
    "StageResearchPlanInput",
    "UpdateBoundResearchResultInput",
    "UpdateResearchGraphScopeInput",
    "add_research_experiment",
    "add_research_hypothesis",
    "create_research_graph",
    "create_bound_research_experiment",
    "record_research_experiment_comparison",
    "list_research_graphs",
    "mark_research_planning_no_change",
    "mark_bound_research_experiment_failed",
    "mark_research_experiment_failed",
    "query_research_graph_sql",
    "resume_bound_research_experiment",
    "retract_bound_research_result",
    "record_bound_research_result",
    "record_research_result",
    "set_research_result_judgment",
    "revise_research_claim",
    "record_research_disposition",
    "record_research_review",
    "set_research_graph_completion",
    "set_research_graph_focus",
    "stage_research_plan",
    "update_bound_research_result",
    "update_research_graph_scope",
]
