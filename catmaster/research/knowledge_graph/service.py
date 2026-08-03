from __future__ import annotations

import asyncio
import re
import socket
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlparse

from catmaster.tools.base import system_root
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.thread_models import ThreadSubmitRequest, ThreadStatus
from catmaster.webui.thread_store import ThreadStore

from .context import ResearchGraphContextBuilder, runnable_frontier_ids
from .models import (
    EdgeRelation,
    ExperimentCreateRequest,
    ExperimentState,
    GraphCreateRequest,
    GraphPatchRequest,
    HypothesisCreateRequest,
    NodeKind,
    NodePatchRequest,
    RefKind,
    ResearchExperimentProposal,
    ResearchExperimentEvaluationDraft,
    ResearchGraphPlanningDraft,
    ResearchGraphPlanningProposal,
    ResearchHypothesisProposal,
    ResearchRefInput,
    ResultCreateRequest,
    ResultJudgmentSetRequest,
)
from .planning import build_planning_preview
from .store import ResearchGraphConflict, ResearchGraphStore

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_ACTIVE_LAUNCH_STATUSES = {"claimed", "submitting", "running", "unknown"}
_PLANNING_THREAD_KIND = "research_graph_planning"


class ResearchGraphService:
    """Workspace graph domain service plus child-thread orchestration."""

    def __init__(
        self,
        *,
        workspace: Path | str,
        workspace_id: str = "",
        agent_loop_factory: Callable[[Path, str], Any] | None = None,
        worker_id: str = "",
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace_id = str(workspace_id or self.workspace.name).strip() or self.workspace.name
        self.store = ResearchGraphStore(self.workspace)
        self.thread_store = ThreadStore(
            workspace=self.workspace,
            workspace_id=self.workspace_id,
        )
        self.artifact_registry = ArtifactRegistry(
            workspace=self.workspace,
            workspace_id=self.workspace_id,
        )
        self.context_builder = ResearchGraphContextBuilder(
            workspace=self.workspace,
            store=self.store,
        )
        self.agent_loop_factory = agent_loop_factory
        self.worker_id = str(worker_id or f"{socket.gethostname()}:{id(self)}")

    @staticmethod
    def _hypothesis_evidence_states(
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, str]:
        relations: dict[str, set[str]] = {}
        for edge in edges:
            if edge["relation"] in {"supports", "opposes", "inconclusive"}:
                relations.setdefault(str(edge["target_node_id"]), set()).add(
                    str(edge["relation"])
                )
        result: dict[str, str] = {}
        for node in nodes:
            if node["kind"] != "hypothesis":
                continue
            values = relations.get(str(node["node_id"]), set())
            if "supports" in values and "opposes" in values:
                state = "conflicting_evidence"
            elif "supports" in values:
                state = "supporting_evidence"
            elif "opposes" in values:
                state = "opposing_evidence"
            elif "inconclusive" in values:
                state = "not_distinguished"
            else:
                state = "no_results"
            result[str(node["node_id"])] = state
        return result

    @staticmethod
    def _frontier_ids(snapshot: dict[str, Any]) -> list[str]:
        return runnable_frontier_ids(snapshot["nodes"], snapshot["edges"])

    @staticmethod
    def _is_internal_planning_thread(thread: Any) -> bool:
        meta = thread.meta if isinstance(getattr(thread, "meta", None), dict) else {}
        if str(meta.get("internal_kind") or "") == _PLANNING_THREAD_KIND:
            return True
        return (
            str(getattr(thread, "thread_id", "")).startswith("thread_rg_")
            and str(getattr(thread, "title", "")).startswith("Plan next step:")
        )

    def _safe_note_path(self, ref_id: str) -> tuple[str, Path] | None:
        raw = str(ref_id or "").strip().replace("\\", "/").lstrip("/")
        if not raw or any(part in {"", ".", ".."} for part in Path(raw).parts):
            return None
        files_root = (self.workspace / "files").resolve()
        candidate = self.workspace.joinpath(*Path(raw).parts).resolve()
        if not candidate.exists() and not raw.startswith("files/"):
            candidate = files_root.joinpath(*Path(raw).parts).resolve()
        try:
            candidate.relative_to(files_root)
        except ValueError:
            return None
        if not candidate.is_file():
            return None
        relative = str(candidate.relative_to(self.workspace)).replace("\\", "/")
        return relative, candidate

    def _message_ref(self, ref_id: str) -> tuple[Any, Any] | None:
        raw = str(ref_id or "").strip()
        thread_hint = ""
        message_id = raw
        if ":" in raw:
            thread_hint, message_id = raw.split(":", 1)
        threads = self.thread_store.list_threads()
        if thread_hint:
            threads = [
                thread for thread in threads if thread.thread_id == thread_hint
            ]
        matches: list[tuple[Any, Any]] = []
        for thread in threads:
            message = self.thread_store.get_message(thread.thread_id, message_id)
            if message is not None:
                matches.append((thread, message))
        # Bare message IDs are accepted for legacy callers only when ownership
        # is unambiguous. Persisted refs are canonicalized below so later SQL
        # projection cannot accidentally expose a same-ID message in another
        # thread.
        return matches[0] if len(matches) == 1 else None

    def validate_ref(self, ref: ResearchRefInput | dict[str, Any]) -> dict[str, str]:
        model = (
            ref
            if isinstance(ref, ResearchRefInput)
            else ResearchRefInput.model_validate(ref)
        )
        kind = model.ref_kind
        ref_id = model.ref_id
        if kind is RefKind.THREAD:
            try:
                self.thread_store.get_thread(ref_id)
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    "Thread reference is not available in this workspace."
                ) from exc
        elif kind is RefKind.MESSAGE:
            found = self._message_ref(ref_id)
            if found is None:
                raise ValueError(
                    "Message reference is not available in this workspace. "
                    "Use thread_id:message_id when the message ID is ambiguous."
                )
            thread, message = found
            ref_id = f"{thread.thread_id}:{message.id}"
        elif kind is RefKind.ARTIFACT:
            if self.artifact_registry.get(ref_id) is None:
                raise ValueError(
                    "Artifact reference is not available in this workspace."
                )
        elif kind is RefKind.RUN:
            run_path = system_root(self.workspace) / "runs" / ref_id
            if not run_path.is_dir():
                raise ValueError(
                    "Run reference is not available in this workspace."
                )
        elif kind is RefKind.NOTE:
            if self._safe_note_path(ref_id) is None:
                raise ValueError(
                    "Note reference must point to an existing file under this "
                    "workspace's files directory."
                )
        elif kind is RefKind.DOI:
            normalized = re.sub(
                r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
                "",
                ref_id,
                flags=re.IGNORECASE,
            ).strip()
            if not _DOI_RE.fullmatch(normalized):
                raise ValueError("DOI reference is invalid.")
            ref_id = normalized
        elif kind is RefKind.URL:
            parsed = urlparse(ref_id)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("URL reference must be an http or https URL.")
        return {"ref_kind": kind.value, "ref_id": ref_id}

    def resolve_ref(self, ref: dict[str, str]) -> dict[str, Any]:
        kind = RefKind(str(ref["ref_kind"]))
        ref_id = str(ref["ref_id"])
        base = {
            "ref_kind": kind.value,
            "ref_id": ref_id,
            "label": "Source unavailable",
            "available": False,
            "href": "",
        }
        try:
            if kind is RefKind.THREAD:
                thread = self.thread_store.get_thread(ref_id)
                return {
                    **base,
                    "label": thread.title or "Untitled thread",
                    "available": True,
                    "thread_id": thread.thread_id,
                }
            if kind is RefKind.MESSAGE:
                found = self._message_ref(ref_id)
                if found is None:
                    return base
                thread, message = found
                return {
                    **base,
                    "label": f"{message.role.title()} message in {thread.title or 'thread'}",
                    "available": True,
                    "thread_id": thread.thread_id,
                    "message_id": message.id,
                }
            if kind is RefKind.ARTIFACT:
                artifact = self.artifact_registry.get(ref_id)
                if artifact is None:
                    return base
                return {
                    **base,
                    "label": artifact.title or Path(artifact.path).name,
                    "available": True,
                    "artifact_id": artifact.artifact_id,
                    "preview_url": artifact.preview_url,
                    "download_url": artifact.download_url,
                }
            if kind is RefKind.RUN:
                path = system_root(self.workspace) / "runs" / ref_id
                if not path.is_dir():
                    return base
                return {
                    **base,
                    "label": f"Run {ref_id}",
                    "available": True,
                    "run_id": ref_id,
                }
            if kind is RefKind.NOTE:
                found = self._safe_note_path(ref_id)
                if found is None:
                    return base
                relative, path = found
                return {
                    **base,
                    "label": path.name,
                    "available": True,
                    "path": relative,
                }
            if kind is RefKind.DOI:
                return {
                    **base,
                    "label": f"DOI {ref_id}",
                    "available": True,
                    "href": f"https://doi.org/{quote(ref_id, safe='/()')}",
                }
            if kind is RefKind.URL:
                return {
                    **base,
                    "label": urlparse(ref_id).netloc or ref_id,
                    "available": True,
                    "href": ref_id,
                }
        except (KeyError, ValueError, OSError):
            return base
        return base

    @staticmethod
    def _public_launch(launch: dict[str, Any]) -> dict[str, Any]:
        return {
            "launch_id": str(launch["launch_id"]),
            "experiment_node_id": str(launch["experiment_node_id"]),
            "status": str(launch["status"]),
            "thread_id": str(launch.get("thread_id") or ""),
            "run_id": str(launch.get("run_id") or ""),
        }

    @staticmethod
    def _public_graph(graph: dict[str, Any]) -> dict[str, Any]:
        return {
            "graph_id": str(graph["graph_id"]),
            "title": str(graph["title"]),
            "question": str(graph["question"]),
            "completion_criterion": str(graph["completion_criterion"]),
            "completed": bool(graph["completed"]),
            "orchestration_mode": str(graph["orchestration_mode"]),
            "archived": bool(graph["archived"]),
            "revision": int(graph["revision"]),
        }

    @staticmethod
    def _public_node(node: dict[str, Any]) -> dict[str, Any]:
        return {
            "node_id": str(node["node_id"]),
            "kind": str(node["kind"]),
            "title": str(node["title"]),
            "state": str(node.get("state") or ""),
            "body": dict(node.get("body") or {}),
            "revision": int(node["revision"]),
        }

    @staticmethod
    def _public_edge(edge: dict[str, Any]) -> dict[str, str]:
        return {
            "source_node_id": str(edge["source_node_id"]),
            "target_node_id": str(edge["target_node_id"]),
            "relation": str(edge["relation"]),
        }

    def presentation(
        self,
        graph_id: str,
        *,
        current_thread_id: str = "",
    ) -> dict[str, Any]:
        snapshot = self.store.get_snapshot(graph_id)
        evidence_states = self._hypothesis_evidence_states(
            snapshot["nodes"], snapshot["edges"]
        )
        refs_by_node: dict[str, list[dict[str, Any]]] = {}
        for ref in snapshot["refs"]:
            refs_by_node.setdefault(str(ref["node_id"]), []).append(
                self.resolve_ref(ref)
            )
        active_by_experiment = {
            str(launch["experiment_node_id"]): self._public_launch(launch)
            for launch in snapshot["launches"]
            if launch["status"] in _ACTIVE_LAUNCH_STATUSES
        }
        nodes: list[dict[str, Any]] = []
        for node in snapshot["nodes"]:
            node_id = str(node["node_id"])
            presented = {
                **self._public_node(node),
                "evidence_state": evidence_states.get(node_id, ""),
                "refs": refs_by_node.get(node_id, []),
            }
            if node_id in active_by_experiment:
                presented["active_launch"] = active_by_experiment[node_id]
            nodes.append(presented)
        frontier_ids = self._frontier_ids(snapshot)
        node_by_id = {str(node["node_id"]): node for node in nodes}
        bound_threads = [
            thread
            for thread in self.thread_store.list_threads()
            if thread.active_research_graph_id == graph_id
            and not self._is_internal_planning_thread(thread)
        ]
        graph = {
            **self._public_graph(snapshot["graph"]),
            "counts": {
                "hypotheses": sum(node["kind"] == "hypothesis" for node in nodes),
                "experiments": sum(node["kind"] == "experiment" for node in nodes),
                "results": sum(node["kind"] == "result" for node in nodes),
            },
            "frontier": [
                {
                    "node_id": node_id,
                    "title": str(node_by_id[node_id]["title"]),
                }
                for node_id in frontier_ids
                if node_id in node_by_id
            ],
            "bound_thread_count": len(bound_threads),
            "bound_to_current_thread": any(
                thread.thread_id == current_thread_id for thread in bound_threads
            ),
        }
        result = {
            "graph": graph,
            "nodes": nodes,
            "edges": [self._public_edge(edge) for edge in snapshot["edges"]],
        }
        planning = self.store.latest_planning_preview(graph_id)
        if planning is not None and not graph["completed"]:
            stored_preview = dict(planning.get("preview") or {})
            raw_proposal = stored_preview.get("proposal")
            if isinstance(raw_proposal, dict):
                proposal = ResearchGraphPlanningProposal.model_validate(raw_proposal)
                public_preview = build_planning_preview(
                    snapshot,
                    proposal,
                    focus_node_id=str(stored_preview.get("focus_node_id") or ""),
                    evaluation=(
                        ResearchExperimentEvaluationDraft.model_validate(
                            stored_preview["evaluation"]
                        )
                        if isinstance(stored_preview.get("evaluation"), dict)
                        else None
                    ),
                )
                for node in public_preview.get("nodes", []):
                    node["refs"] = [
                        self.resolve_ref(ref)
                        for ref in list(node.get("refs") or [])
                    ]
                result["planning_preview"] = {
                    "planning_id": str(planning["planning_id"]),
                    "revision": int(planning["revision"]),
                    **public_preview,
                }
        return result

    def catalog(
        self,
        *,
        include_archived: bool = True,
        current_thread_id: str = "",
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        threads = self.thread_store.list_threads()
        for graph in self.store.list_graphs(include_archived=include_archived):
            snapshot = self.store.get_snapshot(graph["graph_id"])
            frontier_ids = self._frontier_ids(snapshot)
            by_id = {
                str(node["node_id"]): node for node in snapshot["nodes"]
            }
            bound = [
                thread
                for thread in threads
                if thread.active_research_graph_id == graph["graph_id"]
                and not self._is_internal_planning_thread(thread)
            ]
            entries.append(
                {
                    **self._public_graph(graph),
                    # The catalog renders this as a human-readable update label.
                    # Creation time and the rest of the storage row stay internal.
                    "updated_at": float(graph["updated_at"]),
                    "counts": {
                        "hypotheses": sum(
                            node["kind"] == "hypothesis"
                            for node in snapshot["nodes"]
                        ),
                        "experiments": sum(
                            node["kind"] == "experiment"
                            for node in snapshot["nodes"]
                        ),
                        "results": sum(
                            node["kind"] == "result"
                            for node in snapshot["nodes"]
                        ),
                    },
                    "frontier": [
                        {
                            "node_id": node_id,
                            "title": str(by_id[node_id]["title"]),
                        }
                        for node_id in frontier_ids
                        if node_id in by_id
                    ],
                    "bound_thread_count": len(bound),
                    "bound_to_current_thread": any(
                        thread.thread_id == current_thread_id for thread in bound
                    ),
                }
            )
        return entries

    def create_graph(self, request: GraphCreateRequest) -> dict[str, Any]:
        seeds = []
        for seed in request.initial_hypotheses:
            payload = seed.model_dump(mode="json")
            payload["refs"] = [self.validate_ref(ref) for ref in seed.refs]
            seeds.append(payload)
        graph = self.store.create_graph(
            title=request.title,
            question=request.question,
            completion_criterion=request.completion_criterion,
            orchestration_mode=request.orchestration_mode,
            initial_hypotheses=seeds,
        )
        return self.presentation(graph["graph_id"])

    def patch_graph(
        self,
        graph_id: str,
        request: GraphPatchRequest,
    ) -> dict[str, Any]:
        changes = {
            key: getattr(request, key)
            for key in request.model_fields_set
            if key != "expected_revision"
        }
        if changes.get("completed"):
            snapshot = self.store.get_snapshot(graph_id)
            if not any(node["kind"] == "result" for node in snapshot["nodes"]):
                raise ValueError(
                    "A Research Graph cannot be completed before it records a Result."
                )
        if (
            {"question", "completion_criterion"} & set(changes)
            and "completed" not in changes
        ):
            changes["completed"] = False
        self.store.update_graph(
            graph_id,
            expected_revision=request.expected_revision,
            changes=changes,
        )
        return self.presentation(graph_id)

    def add_hypothesis(
        self,
        graph_id: str,
        request: HypothesisCreateRequest,
    ) -> dict[str, Any]:
        refs = [self.validate_ref(ref) for ref in request.refs]
        title = request.title or request.claim[:120]
        node_id = ""
        # Allocate the node ID in the store, then form suggests edges against it
        # by using one explicit stable ID for this atomic bundle.
        from uuid import uuid4

        node_id = f"hyp_{uuid4().hex[:16]}"
        edges = [
            {
                "source_node_id": result_id,
                "target_node_id": node_id,
                "relation": EdgeRelation.SUGGESTS.value,
            }
            for result_id in request.suggested_by_result_ids
        ]
        node, _event_id = self.store.add_node_bundle(
            graph_id,
            expected_revision=request.expected_revision,
            kind=NodeKind.HYPOTHESIS,
            title=title,
            body={
                "claim": request.claim,
                "rationale": request.rationale,
                "predictions": request.predictions,
                "importance": request.importance,
            },
            edges=edges,
            refs=refs,
            node_id=node_id,
            change="hypothesis.added",
        )
        return {"node": self._public_node(node), **self.presentation(graph_id)}

    def add_experiment(
        self,
        graph_id: str,
        request: ExperimentCreateRequest,
    ) -> dict[str, Any]:
        refs = [self.validate_ref(ref) for ref in request.refs]
        from uuid import uuid4

        node_id = f"exp_{uuid4().hex[:16]}"
        edges = [
            {
                "source_node_id": hypothesis_id,
                "target_node_id": node_id,
                "relation": EdgeRelation.TESTS.value,
            }
            for hypothesis_id in request.tests_hypothesis_ids
        ]
        edges.extend(
            {
                "source_node_id": node_id,
                "target_node_id": dependency_id,
                "relation": EdgeRelation.DEPENDS_ON.value,
            }
            for dependency_id in request.depends_on_experiment_ids
        )
        node, _event_id = self.store.add_node_bundle(
            graph_id,
            expected_revision=request.expected_revision,
            kind=NodeKind.EXPERIMENT,
            title=request.title or request.objective[:120],
            body={
                "objective": request.objective,
                "plan_summary": request.plan_summary,
                "decision_rule": request.decision_rule,
                "blocking_reason": request.blocking_reason,
                "execution_lane": request.execution_lane,
                "estimated_compute_cost": request.estimated_compute_cost,
            },
            state=request.state.value,
            edges=edges,
            refs=refs,
            node_id=node_id,
            change="experiment.added",
        )
        return {"node": self._public_node(node), **self.presentation(graph_id)}

    def record_result(
        self,
        graph_id: str,
        request: ResultCreateRequest,
    ) -> dict[str, Any]:
        refs = [self.validate_ref(ref) for ref in request.refs]
        node, _event_id = self.store.add_result_bundle(
            graph_id,
            expected_revision=request.expected_revision,
            title=request.title,
            body={"summary": request.summary},
            experiment_node_id=request.experiment_node_id,
            judgments=[
                judgment.model_dump(mode="json")
                for judgment in request.judgments
            ],
            refs=refs,
        )
        return {"node": self._public_node(node), **self.presentation(graph_id)}

    def set_result_judgment(
        self,
        graph_id: str,
        result_node_id: str,
        hypothesis_node_id: str,
        request: ResultJudgmentSetRequest,
    ) -> dict[str, Any]:
        self.store.set_result_judgment(
            graph_id,
            expected_revision=request.expected_revision,
            result_node_id=result_node_id,
            hypothesis_node_id=hypothesis_node_id,
            relation=request.relation,
        )
        node = self.store.get_node(graph_id, result_node_id)
        return {"node": self._public_node(node), **self.presentation(graph_id)}

    def update_node(
        self,
        graph_id: str,
        node_id: str,
        request: NodePatchRequest,
    ) -> dict[str, Any]:
        node, _event_id = self.store.update_node(
            graph_id,
            node_id,
            expected_revision=request.expected_revision,
            expected_node_revision=request.expected_node_revision,
            title=request.title,
            state=request.state,
            body=request.body,
        )
        return {"node": self._public_node(node), **self.presentation(graph_id)}

    def add_edge(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        source_node_id: str,
        target_node_id: str,
        relation: EdgeRelation,
    ) -> dict[str, Any]:
        self.store.add_edge(
            graph_id,
            expected_revision=expected_revision,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relation=relation,
        )
        return self.presentation(graph_id)

    def add_ref(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        node_id: str,
        ref: ResearchRefInput,
    ) -> dict[str, Any]:
        validated = self.validate_ref(ref)
        self.store.add_ref(
            graph_id,
            expected_revision=expected_revision,
            node_id=node_id,
            ref_kind=validated["ref_kind"],
            ref_id=validated["ref_id"],
        )
        return self.presentation(graph_id)

    def mark_experiment_blocked(
        self,
        graph_id: str,
        experiment_node_id: str,
        *,
        expected_revision: int,
        reason: str,
    ) -> dict[str, Any]:
        self.store.mark_experiment_blocked(
            graph_id,
            experiment_node_id,
            expected_revision=expected_revision,
            reason=reason,
        )
        return self.presentation(graph_id)

    def bind_thread(
        self,
        thread_id: str,
        *,
        graph_id: str,
        focus_node_id: str = "",
    ) -> Any:
        thread = self.thread_store.get_thread(thread_id)
        if not graph_id:
            return self.thread_store.update_thread(
                thread.thread_id,
                active_research_graph_id="",
                research_focus_node_id="",
            )
        self.store.get_graph(graph_id)
        if focus_node_id:
            self.store.get_node(graph_id, focus_node_id)
        return self.thread_store.update_thread(
            thread.thread_id,
            active_research_graph_id=graph_id,
            research_focus_node_id=focus_node_id,
        )

    def _experiment_launch_prompt(
        self,
        *,
        graph_id: str,
        experiment: dict[str, Any],
        replicate: bool,
    ) -> str:
        body = experiment["body"]
        attempt = "replicate this experiment" if replicate else "run this experiment"
        return (
            f"You are continuing workspace Research Graph {graph_id}. "
            f"Prepare and {attempt} for node {experiment['node_id']}.\n\n"
            f"Objective: {body['objective']}\n"
            f"Plan: {body['plan_summary']}\n"
            f"Decision rule: {body['decision_rule']}\n\n"
            "Keep detailed calculations, receipts, and reports in their normal "
            "workspace owners. When execution reaches a scientifically "
            "meaningful outcome, collect a concise Result and attach the source "
            "run/artifacts as refs. Before the atomic graph writeback, ask the "
            "shared evidence judge to assess only the hypotheses the evidence "
            "actually distinguishes; an empty judgment set is valid. If execution "
            "cannot proceed, record the concrete blocking reason."
        )

    async def launch_experiment(
        self,
        graph_id: str,
        experiment_node_id: str,
        *,
        expected_revision: int,
        replicate: bool = False,
    ) -> dict[str, Any]:
        launch, claimed = self.store.claim_launch(
            graph_id,
            experiment_node_id,
            expected_revision=expected_revision,
            replicate=replicate,
            lease_owner=self.worker_id,
        )
        if not claimed:
            return {
                "accepted": True,
                "deduplicated": True,
                "launch": self._public_launch(launch),
                **self.presentation(graph_id),
            }
        launch, child = await self._materialize_launch(
            launch,
            replicate=replicate,
        )
        return {
            "accepted": True,
            "deduplicated": False,
            "launch": self._public_launch(launch),
            "thread": child,
            **self.presentation(graph_id),
        }

    async def _materialize_launch(
        self,
        launch: dict[str, Any],
        *,
        replicate: bool,
    ) -> tuple[dict[str, Any], Any]:
        if self.agent_loop_factory is None:
            self.store.update_launch(
                launch["launch_id"],
                status="unknown",
                lease_owner="",
                lease_until=0,
            )
            raise RuntimeError("Research launch service is unavailable.")
        graph_id = str(launch["graph_id"])
        experiment_node_id = str(launch["experiment_node_id"])
        child_thread_id = f"thread_rg_{launch['launch_id'].removeprefix('launch_')}"
        experiment = self.store.get_node(graph_id, experiment_node_id)
        entrypoint = str(experiment["body"]["execution_lane"])
        if entrypoint not in {"research", "experiment", "literature_review"}:
            entrypoint = "experiment"
        try:
            self.store.update_launch(
                launch["launch_id"],
                status="submitting",
                lease_owner=self.worker_id,
                lease_until=launch["lease_until"],
            )
            child = self.thread_store.create_thread(
                thread_id=child_thread_id,
                title=(
                    f"Replicate: {experiment['title']}"
                    if replicate
                    else f"Run: {experiment['title']}"
                ),
                entrypoint=entrypoint,
            )
            child = self.thread_store.update_thread(
                child.thread_id,
                active_research_graph_id=graph_id,
                research_focus_node_id=experiment_node_id,
            )
            self.store.update_launch(
                launch["launch_id"],
                status="submitting",
                thread_id=child.thread_id,
                lease_owner=self.worker_id,
                lease_until=launch["lease_until"],
            )
            existing_messages = self.thread_store.list_messages(child.thread_id)
            if not existing_messages:
                await self.agent_loop_factory(
                    self.workspace, self.workspace_id
                ).submit(
                    thread_id=child.thread_id,
                    payload=ThreadSubmitRequest(
                        text=self._experiment_launch_prompt(
                            graph_id=graph_id,
                            experiment=experiment,
                            replicate=replicate,
                        ),
                        entrypoint=entrypoint,
                    ),
                )
            launch = self.store.update_launch(
                launch["launch_id"],
                status="running",
                thread_id=child.thread_id,
                lease_owner="",
                lease_until=0,
            )
        except Exception:
            # A child or remote submission may already exist. Preserve that
            # identity for reconciliation and never blindly resubmit.
            self.store.update_launch(
                launch["launch_id"],
                status="unknown",
                thread_id=child_thread_id,
                lease_owner="",
                lease_until=0,
            )
            raise
        return launch, child

    def reconcile_finished_child(
        self,
        *,
        child_thread_id: str,
        terminal_status: str,
        run_id: str = "",
    ) -> None:
        launch = self.store.find_launch_by_thread(child_thread_id)
        if launch is None:
            planning = self.store.find_planning_by_thread(child_thread_id)
            if planning is None:
                return
            graph = self.store.get_graph(planning["graph_id"])
            has_staged_proposal = bool(
                dict(planning.get("preview") or {}).get("proposal")
            )
            graph_revision = int(graph["revision"])
            planning_revision = int(planning["revision"])
            self.store.update_planning(
                planning["graph_id"],
                planning["planning_id"],
                start_revision=planning_revision,
                status=(
                    "stale"
                    if graph_revision != planning_revision
                    else ("finished" if has_staged_proposal else "no_change")
                ),
                thread_id=child_thread_id,
            )
            return
        if run_id:
            launch = self.store.update_launch(
                launch["launch_id"],
                status=launch["status"],
                run_id=run_id,
                thread_id=child_thread_id,
                lease_owner="",
                lease_until=0,
            )
        snapshot = self.store.get_snapshot(launch["graph_id"])
        has_result = any(
            edge["relation"] == "produces"
            and edge["source_node_id"] == launch["experiment_node_id"]
            for edge in snapshot["edges"]
        )
        if has_result:
            self.store.update_launch(
                launch["launch_id"],
                status="completed",
                run_id=run_id,
                thread_id=child_thread_id,
                lease_owner="",
                lease_until=0,
            )
            return
        normalized_status = str(terminal_status or "unknown").strip().lower()
        if normalized_status in {
            "interrupted",
            "paused",
            "awaiting_human_feedback",
            "queued",
            "running",
            "streaming",
            "steered",
        }:
            return
        # A stopped, failed, or simply incomplete child is operational state,
        # not a scientific Result. Make the experiment runnable again and keep
        # thread/run identity on the launch for diagnosis and late writeback.
        self.store.release_incomplete_launch(
            launch["launch_id"],
            run_id=run_id,
            thread_id=child_thread_id,
        )

    def _validated_planning_proposal(
        self,
        snapshot: dict[str, Any],
        proposal: ResearchGraphPlanningProposal,
    ) -> ResearchGraphPlanningProposal:
        payload = proposal.model_dump(mode="json")
        durable_by_id = {
            str(node["node_id"]): node for node in snapshot["nodes"]
        }
        proposed_hypothesis_ids = {
            str(item["proposal_id"]) for item in payload["hypotheses"]
        }
        proposed_experiment_ids = {
            str(item["proposal_id"]) for item in payload["experiments"]
        }
        valid_hypothesis_ids = {
            node_id
            for node_id, node in durable_by_id.items()
            if node["kind"] == "hypothesis"
        } | proposed_hypothesis_ids
        valid_experiment_ids = {
            node_id
            for node_id, node in durable_by_id.items()
            if node["kind"] == "experiment"
        } | proposed_experiment_ids
        for item in payload["hypotheses"]:
            item["refs"] = [self.validate_ref(ref) for ref in item["refs"]]
        dependencies: dict[str, list[str]] = {}
        for item in payload["experiments"]:
            item["refs"] = [self.validate_ref(ref) for ref in item["refs"]]
            unknown_hypotheses = sorted(
                set(item["tests_hypothesis_ids"]) - valid_hypothesis_ids
            )
            if unknown_hypotheses:
                raise ValueError(
                    "A planning experiment references unknown hypothesis IDs: "
                    + ", ".join(unknown_hypotheses)
                )
            unknown_dependencies = sorted(
                set(item["depends_on_experiment_ids"]) - valid_experiment_ids
            )
            if unknown_dependencies:
                raise ValueError(
                    "A planning experiment references unknown dependency IDs: "
                    + ", ".join(unknown_dependencies)
                )
            dependencies[item["proposal_id"]] = [
                dependency_id
                for dependency_id in item["depends_on_experiment_ids"]
                if dependency_id in proposed_experiment_ids
            ]

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(proposal_id: str) -> None:
            if proposal_id in visiting:
                raise ValueError(
                    "Planning experiment dependencies must remain acyclic."
                )
            if proposal_id in visited:
                return
            visiting.add(proposal_id)
            for dependency_id in dependencies.get(proposal_id, []):
                visit(dependency_id)
            visiting.remove(proposal_id)
            visited.add(proposal_id)

        for proposal_id in sorted(dependencies):
            visit(proposal_id)

        recommended_target_id = str(payload["recommended_target_id"] or "")
        ready_ids = set(self._frontier_ids(snapshot))
        if (
            recommended_target_id
            and recommended_target_id not in proposed_experiment_ids
            and recommended_target_id not in ready_ids
        ):
            raise ValueError(
                "The recommended route must be a proposed Experiment or an "
                "existing ready Experiment."
            )
        return ResearchGraphPlanningProposal.model_validate(payload)

    @staticmethod
    def _planning_semantic_key(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip()).casefold()

    @classmethod
    def _add_planning_alias(
        cls,
        aliases: dict[str, set[str]],
        value: str,
        target_id: str,
    ) -> None:
        key = cls._planning_semantic_key(value)
        if key:
            aliases.setdefault(key, set()).add(target_id)

    @classmethod
    def _resolve_planning_alias(
        cls,
        aliases: dict[str, set[str]],
        value: str,
        *,
        role: str,
    ) -> str:
        key = cls._planning_semantic_key(value)
        matches = aliases.get(key, set())
        if not matches:
            raise ValueError(
                f"The planning {role} '{value}' does not match an exact scientific "
                "title, claim, or objective in the bound graph or this draft."
            )
        if len(matches) > 1:
            raise ValueError(
                f"The planning {role} '{value}' is scientifically ambiguous. "
                "Use a unique exact title, claim, or objective."
            )
        return next(iter(matches))

    def _planning_source_refs(self, values: list[str]) -> list[ResearchRefInput]:
        refs: list[ResearchRefInput] = []
        for raw in values:
            value = str(raw or "").strip()
            if not value:
                continue
            normalized_doi = re.sub(
                r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
                "",
                value,
                flags=re.IGNORECASE,
            ).strip()
            if _DOI_RE.fullmatch(normalized_doi):
                kind = RefKind.DOI
                ref_id = normalized_doi
            else:
                parsed = urlparse(value)
                if parsed.scheme in {"http", "https"} and parsed.netloc:
                    kind = RefKind.URL
                    ref_id = value
                else:
                    kind = RefKind.NOTE
                    ref_id = value
            refs.append(ResearchRefInput(ref_kind=kind, ref_id=ref_id))
        return refs

    def compile_planning_draft(
        self,
        snapshot: dict[str, Any],
        draft: ResearchGraphPlanningDraft,
        *,
        planning_id: str,
    ) -> ResearchGraphPlanningProposal:
        """Resolve scientific labels and add internal IDs outside the model contract."""

        prefix = re.sub(
            r"[^A-Za-z0-9_.:-]+",
            "_",
            str(planning_id or "planning").strip(),
        )[:96]
        hypothesis_ids = [
            f"{prefix}_hypothesis_{index}"
            for index, _item in enumerate(draft.hypotheses, start=1)
        ]
        experiment_ids = [
            f"{prefix}_experiment_{index}"
            for index, _item in enumerate(draft.experiments, start=1)
        ]

        hypothesis_aliases: dict[str, set[str]] = {}
        experiment_aliases: dict[str, set[str]] = {}
        ready_route_aliases: dict[str, set[str]] = {}
        ready_ids = set(self._frontier_ids(snapshot))
        for node in list(snapshot.get("nodes") or []):
            node_id = str(node.get("node_id") or "")
            body = dict(node.get("body") or {})
            if node.get("kind") == "hypothesis":
                self._add_planning_alias(
                    hypothesis_aliases,
                    str(node.get("title") or ""),
                    node_id,
                )
                self._add_planning_alias(
                    hypothesis_aliases,
                    str(body.get("claim") or ""),
                    node_id,
                )
            elif node.get("kind") == "experiment":
                self._add_planning_alias(
                    experiment_aliases,
                    str(node.get("title") or ""),
                    node_id,
                )
                self._add_planning_alias(
                    experiment_aliases,
                    str(body.get("objective") or ""),
                    node_id,
                )
                if node_id in ready_ids:
                    self._add_planning_alias(
                        ready_route_aliases,
                        str(node.get("title") or ""),
                        node_id,
                    )
                    self._add_planning_alias(
                        ready_route_aliases,
                        str(body.get("objective") or ""),
                        node_id,
                    )

        for target_id, item in zip(hypothesis_ids, draft.hypotheses, strict=True):
            self._add_planning_alias(hypothesis_aliases, item.title, target_id)
            self._add_planning_alias(hypothesis_aliases, item.claim, target_id)
        for target_id, item in zip(experiment_ids, draft.experiments, strict=True):
            self._add_planning_alias(experiment_aliases, item.title, target_id)
            self._add_planning_alias(experiment_aliases, item.objective, target_id)
            self._add_planning_alias(ready_route_aliases, item.title, target_id)
            self._add_planning_alias(ready_route_aliases, item.objective, target_id)

        hypotheses = [
            ResearchHypothesisProposal(
                proposal_id=target_id,
                claim=item.claim,
                title=item.title,
                rationale=item.rationale,
                predictions=item.predictions,
                importance=item.importance,
                refs=self._planning_source_refs(item.sources),
            )
            for target_id, item in zip(
                hypothesis_ids,
                draft.hypotheses,
                strict=True,
            )
        ]
        experiments = [
            ResearchExperimentProposal(
                proposal_id=target_id,
                objective=item.objective,
                title=item.title,
                plan_summary=item.plan_summary,
                decision_rule=item.decision_rule,
                execution_lane=item.execution_lane,
                estimated_compute_cost=item.estimated_compute_cost,
                tests_hypothesis_ids=[
                    self._resolve_planning_alias(
                        hypothesis_aliases,
                        reference,
                        role="hypothesis reference",
                    )
                    for reference in item.tests_hypotheses
                ],
                depends_on_experiment_ids=[
                    self._resolve_planning_alias(
                        experiment_aliases,
                        reference,
                        role="experiment prerequisite",
                    )
                    for reference in item.depends_on_experiments
                ],
                refs=self._planning_source_refs(item.sources),
            )
            for target_id, item in zip(
                experiment_ids,
                draft.experiments,
                strict=True,
            )
        ]
        recommended_target_id = ""
        if draft.recommended_route:
            recommended_target_id = self._resolve_planning_alias(
                ready_route_aliases,
                draft.recommended_route,
                role="recommended route",
            )
        return ResearchGraphPlanningProposal(
            hypotheses=hypotheses,
            experiments=experiments,
            recommended_target_id=recommended_target_id,
            recommendation_reason=draft.recommendation_reason,
        )

    def stage_planning_draft(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        planning_thread_id: str,
        draft: ResearchGraphPlanningDraft,
    ) -> dict[str, Any]:
        """Compile science-first input and publish its temporary graph preview."""

        planning = self.store.find_planning_by_thread(planning_thread_id)
        if planning is None or str(planning["graph_id"]) != str(graph_id):
            raise ValueError(
                "This tool is available only inside the active bound Research "
                "Graph planning thread."
            )
        proposal = self.compile_planning_draft(
            self.store.get_snapshot(graph_id),
            draft,
            planning_id=str(planning["planning_id"]),
        )
        return self.stage_planning_proposal(
            graph_id,
            expected_revision=expected_revision,
            planning_thread_id=planning_thread_id,
            proposal=proposal,
        )

    def stage_planning_proposal(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        planning_thread_id: str,
        proposal: ResearchGraphPlanningProposal,
    ) -> dict[str, Any]:
        """Publish one temporary evidence-aware route plan from the bound child."""

        planning = self.store.find_planning_by_thread(planning_thread_id)
        if planning is None or str(planning["graph_id"]) != str(graph_id):
            raise ValueError(
                "This tool is available only inside the active bound Research "
                "Graph planning thread."
            )
        if int(planning["revision"]) != int(expected_revision):
            raise ResearchGraphConflict(
                expected_revision=int(expected_revision),
                current_revision=int(self.store.get_graph(graph_id)["revision"]),
            )
        snapshot = self.store.get_snapshot(graph_id)
        if int(snapshot["graph"]["revision"]) != int(expected_revision):
            raise ResearchGraphConflict(
                expected_revision=int(expected_revision),
                current_revision=int(snapshot["graph"]["revision"]),
            )
        proposal = self._validated_planning_proposal(snapshot, proposal)
        thread = self.thread_store.get_thread(planning_thread_id)
        focus_node_id = str(thread.research_focus_node_id or "")
        public_preview = build_planning_preview(
            snapshot,
            proposal,
            focus_node_id=focus_node_id,
        )
        stored_preview = {
            "focus_node_id": focus_node_id,
            "proposal": proposal.model_dump(mode="json"),
        }
        self.store.set_planning_preview(
            graph_id,
            planning["planning_id"],
            start_revision=expected_revision,
            preview=stored_preview,
        )
        return {
            "accepted": True,
            "planning_id": str(planning["planning_id"]),
            "revision": int(expected_revision),
            "summary": str(public_preview.get("summary") or ""),
            "candidate_experiment_ids": list(
                public_preview.get("candidate_experiment_ids") or []
            ),
            "staged": {
                "hypotheses": len(proposal.hypotheses),
                "experiments": len(proposal.experiments),
            },
        }

    def stage_planning_evaluation(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        planning_thread_id: str,
        evaluation: ResearchExperimentEvaluationDraft,
    ) -> dict[str, Any]:
        """Attach model-generated Experiment comparisons to this revision's preview."""

        planning = self.store.find_planning_by_thread(planning_thread_id)
        if planning is None or str(planning["graph_id"]) != str(graph_id):
            raise ValueError(
                "This tool is available only inside the active bound Research "
                "Graph planning thread."
            )
        graph = self.store.get_graph(graph_id)
        if (
            int(planning["revision"]) != int(expected_revision)
            or int(graph["revision"]) != int(expected_revision)
        ):
            raise ResearchGraphConflict(
                expected_revision=int(expected_revision),
                current_revision=int(graph["revision"]),
            )
        preview = dict(planning.get("preview") or {})
        raw_proposal = preview.get("proposal")
        if not isinstance(raw_proposal, dict):
            raise ValueError(
                "The planning proposer must publish the temporary plan before "
                "Experiments can be evaluated."
            )
        proposal = ResearchGraphPlanningProposal.model_validate(raw_proposal)
        projected = build_planning_preview(
            self.store.get_snapshot(graph_id),
            proposal,
            focus_node_id=str(preview.get("focus_node_id") or ""),
        )
        candidates = set(projected.get("candidate_experiment_ids") or [])
        supplied = set(evaluation.experiment_ids)
        if supplied != candidates:
            missing = sorted(candidates - supplied)
            unknown = sorted(supplied - candidates)
            details: list[str] = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unknown:
                details.append("unknown: " + ", ".join(unknown))
            raise ValueError(
                "The evaluation must cover every current candidate Experiment"
                + (" (" + "; ".join(details) + ")" if details else "")
                + "."
            )
        preview["evaluation"] = evaluation.model_dump(mode="json")
        self.store.set_planning_preview(
            graph_id,
            str(planning["planning_id"]),
            start_revision=expected_revision,
            preview=preview,
        )
        return {
            "accepted": True,
            "planning_id": str(planning["planning_id"]),
            "revision": int(expected_revision),
            "experiment_ids": list(evaluation.experiment_ids),
            "innovation_recommendation": evaluation.innovation_recommendation,
            "conservative_recommendation": evaluation.conservative_recommendation,
            "evaluation_memo": evaluation.evaluation_memo,
        }

    def materialize_planning_proposal(
        self,
        graph_id: str,
        planning_id: str,
        *,
        expected_revision: int,
        proposal_id: str,
        keep_preview_for_scheduler: bool = False,
    ) -> dict[str, Any]:
        """Atomically materialize one selected temporary route."""

        graph = self.store.get_graph(graph_id)
        if int(graph["revision"]) != int(expected_revision):
            raise ResearchGraphConflict(
                expected_revision=int(expected_revision),
                current_revision=int(graph["revision"]),
            )
        planning = self.store.get_planning(graph_id, planning_id)
        preview = dict(planning.get("preview") or {})
        if int(planning["revision"]) != int(expected_revision):
            raise ValueError(
                "This temporary plan is stale. Refresh and plan against the "
                "current graph revision."
            )
        raw_proposal = dict(preview.get("proposal") or {})
        proposal = ResearchGraphPlanningProposal.model_validate(raw_proposal)
        projected_preview = build_planning_preview(
            self.store.get_snapshot(graph_id),
            proposal,
            focus_node_id=str(preview.get("focus_node_id") or ""),
        )
        hypotheses = {
            item.proposal_id: item for item in proposal.hypotheses
        }
        experiments = {
            item.proposal_id: item for item in proposal.experiments
        }
        if proposal_id not in hypotheses and proposal_id not in experiments:
            raise ValueError("The selected provisional node is not in this plan.")

        selected_hypotheses: set[str] = set()
        selected_experiments: set[str] = set()

        def include_experiment(experiment_id: str) -> None:
            if experiment_id in selected_experiments:
                return
            item = experiments[experiment_id]
            for dependency_id in item.depends_on_experiment_ids:
                if dependency_id in experiments:
                    include_experiment(dependency_id)
            selected_experiments.add(experiment_id)
            selected_hypotheses.update(
                hypothesis_id
                for hypothesis_id in item.tests_hypothesis_ids
                if hypothesis_id in hypotheses
            )

        if proposal_id in hypotheses:
            selected_hypotheses.add(proposal_id)
        else:
            include_experiment(proposal_id)

        nodes: list[dict[str, Any]] = []
        for hypothesis_id in sorted(selected_hypotheses):
            item = hypotheses[hypothesis_id]
            nodes.append(
                {
                    "proposal_id": hypothesis_id,
                    "kind": "hypothesis",
                    "title": item.title or item.claim[:120],
                    "state": "",
                    "body": item.model_dump(
                        mode="json",
                        exclude={"proposal_id", "title", "refs"},
                    ),
                    "refs": [
                        self.validate_ref(ref) for ref in item.refs
                    ],
                }
            )
        for experiment_id in sorted(selected_experiments):
            item = experiments[experiment_id]
            experiment_state = (
                ExperimentState.READY.value
                if item.plan_summary and item.decision_rule
                else ExperimentState.DRAFT.value
            )
            nodes.append(
                {
                    "proposal_id": experiment_id,
                    "kind": "experiment",
                    "title": item.title or item.objective[:120],
                    "state": experiment_state,
                    "body": item.model_dump(
                        mode="json",
                        exclude={
                            "proposal_id",
                            "title",
                            "tests_hypothesis_ids",
                            "depends_on_experiment_ids",
                            "refs",
                        },
                    ),
                    "refs": [
                        self.validate_ref(ref) for ref in item.refs
                    ],
                }
            )

        selected_ids = selected_hypotheses | selected_experiments
        durable_ids = {
            str(node["node_id"])
            for node in self.store.get_snapshot(graph_id)["nodes"]
        }
        edges = [
            dict(edge)
            for edge in list(projected_preview.get("edges") or [])
            if (
                str(edge.get("source_node_id") or "") in selected_ids
                or str(edge.get("target_node_id") or "") in selected_ids
            )
            and (
                str(edge.get("source_node_id") or "") in selected_ids
                or str(edge.get("source_node_id") or "")
                in durable_ids
            )
            and (
                str(edge.get("target_node_id") or "") in selected_ids
                or str(edge.get("target_node_id") or "")
                in durable_ids
            )
        ]
        focus_node_id = str(preview.get("focus_node_id") or "")
        focus_node = (
            self.store.get_node(graph_id, focus_node_id)
            if focus_node_id
            else None
        )
        if focus_node is not None and focus_node["kind"] == NodeKind.RESULT.value:
            edges.extend(
                {
                    "source_node_id": focus_node_id,
                    "target_node_id": hypothesis_id,
                    "relation": EdgeRelation.SUGGESTS.value,
                }
                for hypothesis_id in sorted(selected_hypotheses)
            )
        mapping, _event_id = self.store.materialize_plan_bundle(
            graph_id,
            expected_revision=expected_revision,
            nodes=nodes,
            edges=edges,
        )
        current = self.store.get_snapshot(graph_id)
        frontier = self._frontier_ids(current)
        materialized_ids = set(mapping.values())
        selected_experiment_id = str(mapping.get(proposal_id) or "")
        next_experiment_id = (
            selected_experiment_id
            if selected_experiment_id in materialized_ids
            and selected_experiment_id in frontier
            else ""
        )
        if keep_preview_for_scheduler:
            # Scores belong only to their pre-materialization revision. Replace
            # them with a minimal recovery receipt that can authorize exactly
            # the evaluator-selected Experiment, never another runnable member
            # of the materialized route.
            scheduler_receipt = {
                "materialized_node_ids": mapping,
                "materialized_next_experiment_id": next_experiment_id,
                "materialized_revision": int(current["graph"]["revision"]),
            }
            self.store.set_planning_preview(
                graph_id,
                planning_id,
                start_revision=int(planning["revision"]),
                preview=scheduler_receipt,
            )
        return {
            "proposal_id": proposal_id,
            "node_ids": mapping,
            "next_experiment_node_id": next_experiment_id,
            **self.presentation(graph_id),
        }

    def _planning_focus(self, snapshot: dict[str, Any]) -> str:
        nodes = list(snapshot["nodes"])
        if not nodes:
            return ""
        # Mutation-driven planning follows the most recently changed scientific
        # node. An explicit user planning request still supplies its own focus.
        latest = max(
            enumerate(nodes),
            key=lambda item: (
                float(item[1].get("updated_at") or 0.0),
                float(item[1].get("created_at") or 0.0),
                item[0],
            ),
        )
        return str(latest[1]["node_id"])

    async def _launch_planning_child(
        self,
        graph_id: str,
        *,
        revision: int,
        focus_node_id: str = "",
        allow_same_revision_after_no_change: bool = False,
    ) -> tuple[bool, Any | None]:
        if self.agent_loop_factory is None:
            return False, None
        if focus_node_id:
            self.store.get_node(graph_id, focus_node_id)
        claim, claimed = self.store.claim_planning(
            graph_id,
            expected_revision=revision,
            allow_same_revision_after_no_change=allow_same_revision_after_no_change,
        )
        if not claimed:
            thread_id = str(claim.get("thread_id") or "")
            if thread_id:
                try:
                    return False, self.thread_store.get_thread(thread_id)
                except KeyError:
                    pass
            return False, None
        planning_id = str(claim["planning_id"])
        snapshot = self.store.get_snapshot(graph_id)
        focus_node_id = focus_node_id or self._planning_focus(snapshot)
        child_thread_id = f"thread_rg_{planning_id.removeprefix('planning_')}"
        try:
            child = self.thread_store.create_thread(
                thread_id=child_thread_id,
                title=f"Plan next step: {snapshot['graph']['title']}",
                entrypoint="research",
                meta={"internal_kind": _PLANNING_THREAD_KIND},
            )
            child = self.thread_store.update_thread(
                child.thread_id,
                active_research_graph_id=graph_id,
                research_focus_node_id=focus_node_id,
                meta={**child.meta, "internal_kind": _PLANNING_THREAD_KIND},
            )
            self.store.update_planning(
                graph_id,
                planning_id,
                start_revision=revision,
                status="attached",
                thread_id=child.thread_id,
            )
            if not self.thread_store.list_messages(child.thread_id):
                focus = next(
                    (
                        node
                        for node in snapshot["nodes"]
                        if str(node["node_id"]) == focus_node_id
                    ),
                    None,
                )
                focus_description = (
                    f"The bound focus is {focus['kind']} {focus['title']} "
                    f"({focus['node_id']})."
                    if focus is not None
                    else "No individual focus node is bound."
                )
                await self.agent_loop_factory(
                    self.workspace, self.workspace_id
                ).submit(
                    thread_id=child.thread_id,
                    payload=ThreadSubmitRequest(
                        text=(
                            "Run one result-focused scientific planning pass for the bound "
                            f"Research Graph. {focus_description} The supplied graph text "
                            "is an explicitly partial focus snippet; use the narrow graph "
                            "query and evidence-reconciliation skills to inspect canonical "
                            "state and decisive sources as needed. First ask "
                            "hypothesis_proposer to compare the focus, existing predictions, "
                            "related older Results, dependencies, and the complete runnable "
                            "frontier. Existing Hypotheses may already be sufficient; create "
                            "a new falsifiable explanation only when the Result warrants one. "
                            "The proposer may publish one temporary H/E plan and should return "
                            "a concise scientific memo, or return no-change without staging. "
                            "If a plan was staged, subsequently ask experiment_evaluator to "
                            "compare every current candidate Experiment under both innovation "
                            "and conservative policies and publish the current-revision "
                            "evaluation. Empty recommendations are valid when no route is "
                            "worth selecting or the alternatives cannot be distinguished. "
                            "Return the scientific memos in ordinary language; do not copy a "
                            "delegate payload into another tool call."
                        ),
                        entrypoint="research",
                    ),
                )
            return True, child
        except Exception:
            self.store.update_planning(
                graph_id,
                planning_id,
                start_revision=revision,
                status="stale",
                thread_id=child_thread_id,
            )
            raise

    async def plan_next_step(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        focus_node_id: str = "",
    ) -> dict[str, Any]:
        """Start one user-requested planning turn focused on an explicit node."""

        if self.agent_loop_factory is None:
            raise RuntimeError("Research planning service is unavailable.")
        started, child = await self._launch_planning_child(
            graph_id,
            revision=expected_revision,
            focus_node_id=focus_node_id,
            # Scheduler no-change suppression prevents loops. A deliberate
            # user click is allowed to ask for another independent planning
            # pass at the same scientific revision.
            allow_same_revision_after_no_change=True,
        )
        result: dict[str, Any] = {
            "accepted": True,
            "deduplicated": not started,
            **self.presentation(graph_id),
        }
        if child is not None:
            result["thread"] = child
        return result

    async def tick(self) -> None:
        """Recover active work, then advance each auto graph by one planning or execution child.

        A fresh planning pass follows each graph mutation, including a Result.
        Temporary model values may reorder many branches, while actual execution
        remains one experiment per graph until worker isolation is introduced.
        """

        for launch in self.store.active_launches():
            thread_id = str(launch.get("thread_id") or "")
            if not thread_id:
                if float(launch.get("lease_until") or 0) > time.time():
                    continue
                # Deterministic child IDs and the existing-message check make
                # recovery safe after a crash between claim and child binding.
                try:
                    await self._materialize_launch(
                        launch,
                        replicate=str(launch["idempotency_key"]).startswith(
                            "replicate_"
                        ),
                    )
                except (KeyError, ValueError, RuntimeError):
                    continue
            else:
                try:
                    thread = self.thread_store.get_thread(thread_id)
                except KeyError:
                    self.store.update_launch(
                        launch["launch_id"],
                        status="unknown",
                        lease_owner="",
                        lease_until=0,
                    )
                    continue
                if thread.status in {
                    ThreadStatus.ERROR,
                    ThreadStatus.STOPPED,
                    ThreadStatus.IDLE,
                }:
                    self.reconcile_finished_child(
                        child_thread_id=thread_id,
                        terminal_status=thread.status.value,
                        run_id=thread.active_run_id,
                    )

        for graph in self.store.list_graphs(include_archived=False):
            if (
                graph["orchestration_mode"] != "auto"
                or graph["completed"]
            ):
                continue
            snapshot = self.store.get_snapshot(graph["graph_id"])
            if any(
                launch["status"] in _ACTIVE_LAUNCH_STATUSES
                for launch in snapshot["launches"]
            ):
                continue
            frontier = set(self._frontier_ids(snapshot))
            planning = self.store.latest_planning_preview(
                graph["graph_id"],
                current_revision_only=False,
            )
            preview = dict(planning.get("preview") or {}) if planning else {}
            materialized_id = str(
                preview.get("materialized_next_experiment_id") or ""
            )
            try:
                materialized_revision = int(preview.get("materialized_revision") or 0)
            except (TypeError, ValueError):
                materialized_revision = 0
            if (
                materialized_id
                and materialized_revision == int(graph["revision"])
            ):
                if materialized_id not in frontier:
                    continue
                try:
                    await self.launch_experiment(
                        graph["graph_id"],
                        materialized_id,
                        expected_revision=graph["revision"],
                    )
                except (ResearchGraphConflict, ValueError):
                    continue
                continue
            if not self.store.planning_covers_current_graph(graph["graph_id"]):
                try:
                    await self._launch_planning_child(
                        graph["graph_id"],
                        revision=graph["revision"],
                    )
                except (ResearchGraphConflict, ValueError):
                    continue
                continue
            if planning is None:
                continue

            # Scores and choices are valid only against the exact graph
            # revision on which the evaluator produced them.
            if int(planning["revision"]) != int(graph["revision"]):
                continue
            raw_evaluation = preview.get("evaluation")
            raw_proposal = preview.get("proposal")
            if not isinstance(raw_evaluation, dict) or not isinstance(
                raw_proposal, dict
            ):
                continue
            try:
                evaluation = ResearchExperimentEvaluationDraft.model_validate(
                    raw_evaluation
                )
                proposal = ResearchGraphPlanningProposal.model_validate(raw_proposal)
            except ValueError:
                continue
            # Auto orchestration deliberately defaults to the conservative
            # current-revision policy. An empty or invalid choice means wait.
            selected_id = str(evaluation.conservative_recommendation or "")
            if not selected_id:
                continue
            if selected_id in frontier:
                try:
                    await self.launch_experiment(
                        graph["graph_id"],
                        selected_id,
                        expected_revision=graph["revision"],
                    )
                except (ResearchGraphConflict, ValueError):
                    continue
                continue

            proposed_experiments = {
                item.proposal_id: item for item in proposal.experiments
            }
            selected = proposed_experiments.get(selected_id)
            if (
                selected is None
                or not selected.plan_summary
                or not selected.decision_rule
            ):
                continue
            try:
                self.materialize_planning_proposal(
                    graph["graph_id"],
                    str(planning["planning_id"]),
                    expected_revision=graph["revision"],
                    proposal_id=selected_id,
                    keep_preview_for_scheduler=True,
                )
            except (ResearchGraphConflict, ValueError):
                continue


__all__ = ["ResearchGraphService"]
