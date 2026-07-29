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

from .context import ResearchGraphContextBuilder, ranked_frontier_ids
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
    ResearchRefInput,
    ResultCreateRequest,
)
from .store import ResearchGraphConflict, ResearchGraphStore

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_ACTIVE_LAUNCH_STATUSES = {"claimed", "submitting", "running", "unknown"}


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
        return ranked_frontier_ids(snapshot["nodes"], snapshot["edges"])

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
        for thread in threads:
            message = self.thread_store.get_message(thread.thread_id, message_id)
            if message is not None:
                return thread, message
        return None

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
            if self._message_ref(ref_id) is None:
                raise ValueError(
                    "Message reference is not available in this workspace. "
                    "Use thread_id:message_id when the message ID is ambiguous."
                )
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
        return {
            "graph": graph,
            "nodes": nodes,
            "edges": [self._public_edge(edge) for edge in snapshot["edges"]],
        }

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
                        for node_id in frontier_ids[:3]
                        if node_id in by_id
                    ],
                    "frontier_omitted_count": max(0, len(frontier_ids) - 3),
                    "bound_thread_count": len(bound),
                    "bound_to_current_thread": any(
                        thread.thread_id == current_thread_id for thread in bound
                    ),
                }
            )
        return entries

    def create_graph(self, request: GraphCreateRequest) -> dict[str, Any]:
        seeds = [
            seed.model_dump(mode="json") for seed in request.initial_hypotheses
        ]
        graph = self.store.create_graph(
            title=request.title,
            question=request.question,
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
                "execution_lane": request.execution_lane,
                "expected_value": request.expected_value,
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
            "meaningful outcome, record a concise Result in the bound graph, "
            "attach the source run/artifacts as refs, and judge each tested "
            "hypothesis as supporting, opposing, or inconclusive. If execution "
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
            self.store.update_planning(
                planning["graph_id"],
                planning["planning_id"],
                start_revision=int(planning["revision"]),
                status=(
                    "finished"
                    if int(graph["revision"]) > int(planning["revision"])
                    else "no_change"
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
        }:
            return
        if normalized_status in {"error", "stopped", "failure"}:
            reason = (
                f"Execution thread {child_thread_id} ended with status "
                f"{normalized_status} before recording a result."
            )
        else:
            reason = (
                f"Execution thread {child_thread_id} completed without "
                "recording the required Research Graph result."
            )
        for attempt in range(2):
            graph = self.store.get_graph(launch["graph_id"])
            try:
                self.store.mark_experiment_blocked(
                    launch["graph_id"],
                    launch["experiment_node_id"],
                    expected_revision=graph["revision"],
                    reason=reason,
                    refs=[
                        {
                            "ref_kind": RefKind.THREAD.value,
                            "ref_id": child_thread_id,
                        },
                        *(
                            [
                                {
                                    "ref_kind": RefKind.RUN.value,
                                    "ref_id": run_id,
                                }
                            ]
                            if run_id
                            else []
                        ),
                    ],
                )
                break
            except ResearchGraphConflict:
                if attempt:
                    return
        self.store.update_launch(
            launch["launch_id"],
            status="blocked",
            run_id=run_id,
            thread_id=child_thread_id,
            lease_owner="",
            lease_until=0,
        )
        return

    def _planning_focus(self, snapshot: dict[str, Any]) -> str:
        # Continue from the latest result first, then a hypothesis that has no
        # testing experiment. This is deterministic and avoids scoring metadata.
        for node in reversed(snapshot["nodes"]):
            if node["kind"] == "result":
                return str(node["node_id"])
        tested = {
            str(edge["source_node_id"])
            for edge in snapshot["edges"]
            if edge["relation"] == "tests"
        }
        for node in snapshot["nodes"]:
            if node["kind"] == "hypothesis" and node["node_id"] not in tested:
                return str(node["node_id"])
        return next(
            (
                str(node["node_id"])
                for node in snapshot["nodes"]
                if node["kind"] == "hypothesis"
            ),
            "",
        )

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
            lease_owner=self.worker_id,
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
            )
            child = self.thread_store.update_thread(
                child.thread_id,
                active_research_graph_id=graph_id,
                research_focus_node_id=focus_node_id,
            )
            self.store.update_planning(
                graph_id,
                planning_id,
                start_revision=revision,
                status="attached",
                thread_id=child.thread_id,
            )
            if not self.thread_store.list_messages(child.thread_id):
                await self.agent_loop_factory(
                    self.workspace, self.workspace_id
                ).submit(
                    thread_id=child.thread_id,
                    payload=ThreadSubmitRequest(
                        text=(
                            "Advance the bound Research Graph with a bounded "
                            "portfolio of scientifically distinct branches. "
                            "Inspect the focused result or hypothesis, preserve "
                            "plausible competing hypotheses, and propose complete "
                            "experiments with explicit decision rules. Mark only "
                            "coarse relative hypothesis importance, expected "
                            "decision value, and compute cost. Do not add a node "
                            "merely to report debugging or restate existing content."
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
        """Recover active work, then advance each auto graph by at most one child.

        A graph may retain many ready branches. Auto mode deliberately starts
        only the first ranked branch so concurrent execution remains one per
        graph until worker isolation is introduced.
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
            if graph["orchestration_mode"] != "auto":
                continue
            snapshot = self.store.get_snapshot(graph["graph_id"])
            if any(
                launch["status"] in _ACTIVE_LAUNCH_STATUSES
                for launch in snapshot["launches"]
            ):
                continue
            frontier = self._frontier_ids(snapshot)
            if frontier:
                try:
                    await self.launch_experiment(
                        graph["graph_id"],
                        frontier[0],
                        expected_revision=graph["revision"],
                    )
                except (ResearchGraphConflict, ValueError):
                    continue
            elif snapshot["nodes"]:
                try:
                    await self._launch_planning_child(
                        graph["graph_id"],
                        revision=graph["revision"],
                    )
                except (ResearchGraphConflict, ValueError):
                    continue


__all__ = ["ResearchGraphService"]
