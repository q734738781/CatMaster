from __future__ import annotations

import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from catmaster.research.hypothesis_engine.models import HypothesisEngineState
from catmaster.research.hypothesis_engine.storage import safe_thread_id
from catmaster.storage import connect_workspace_db
from catmaster.tools.base import system_root
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.thread_store import ThreadStore

from .models import RefKind
from .store import ResearchGraphStore


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class ResearchGraphMigrator:
    """One-way deterministic legacy import with an explicit archive manifest."""

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.files_root = self.workspace / "files"
        self.metadata_root = system_root(self.workspace)
        self.campaign_root = self.files_root / "research_hypothesis_engines"
        self.kernel_root = self.files_root / "research_kernels"
        self.plan_path = self.metadata_root / "research_graph_migration_plan.json"
        self.progress_path = (
            self.metadata_root / "research_graph_migration_in_progress.json"
        )
        self.archive_root = self.metadata_root / "legacy_research_state"
        self.store = ResearchGraphStore(self.workspace)
        self.thread_store = ThreadStore(
            workspace=self.workspace,
            workspace_id=self.workspace.name,
        )

    def _campaign_files(self) -> list[Path]:
        if not self.campaign_root.is_dir():
            return []
        return sorted(self.campaign_root.glob("*/state.json"))

    def _kernel_files(self) -> list[Path]:
        if not self.kernel_root.is_dir():
            return []
        return sorted(self.kernel_root.glob("*/kernel.json"))

    @staticmethod
    def _source_key(path: Path, workspace: Path) -> str:
        return str(path.resolve().relative_to(workspace.resolve())).replace(
            "\\", "/"
        )

    def dry_run(self) -> dict[str, Any]:
        campaigns: list[dict[str, Any]] = []
        kernels: list[dict[str, Any]] = []
        totals = {
            "graphs": 0,
            "nodes": 0,
            "edges": 0,
            "refs": 0,
            "launches": 0,
            "review_items": 0,
            "quarantined_files": 0,
        }
        for path in self._campaign_files():
            row: dict[str, Any] = {
                "source": self._source_key(path, self.workspace),
                "status": "ready",
                "issues": [],
            }
            try:
                payload = _read_json(path)
                version = int(payload.get("schema_version", 4))
                row["schema_version"] = version
                if version == 2:
                    row["status"] = "review_required"
                    row["issues"].append(
                        "v2 scientific rationale, predictions, or decision "
                        "rules may be missing; no content will be invented."
                    )
                    totals["review_items"] += 1
                elif version not in {3, 4}:
                    row["status"] = "quarantine"
                    row["issues"].append(
                        f"unsupported campaign schema version {version}"
                    )
                    totals["quarantined_files"] += 1
                else:
                    state = HypothesisEngineState.model_validate(payload)
                    active_child = self._valid_active_child(
                        path.parent.name,
                        state.active_action_id,
                    )
                    result_count = len(state.evidence) + sum(
                        bool(action.failure_reason)
                        for action in state.actions
                    )
                    edge_count = sum(
                        len(action.target_hypotheses)
                        + len(action.prerequisite_action_ids)
                        for action in state.actions
                    )
                    edge_count += len(state.evidence)
                    edge_count += sum(
                        len(judgment.effects) for judgment in state.evidence
                    )
                    row.update(
                        {
                            "question": state.question,
                            "hypotheses": len(state.hypotheses),
                            "experiments": len(state.actions),
                            "results": result_count,
                            "edges": edge_count,
                        }
                    )
                    totals["graphs"] += 1
                    totals["nodes"] += (
                        len(state.hypotheses)
                        + len(state.actions)
                        + result_count
                    )
                    totals["edges"] += edge_count
                    totals["launches"] += int(active_child is not None)
                    if state.active_action_id and active_child is None:
                        row["issues"].append(
                            "Legacy active action "
                            f"{state.active_action_id!r} has no non-terminal "
                            "child thread; migration will release the "
                            "Experiment to ready."
                        )
                        totals["review_items"] += 1
            except Exception as exc:
                row["status"] = "quarantine"
                row["issues"].append(f"invalid or truncated JSON: {exc}")
                totals["quarantined_files"] += 1
            campaigns.append(row)

        known_thread_keys = {
            safe_thread_id(thread.deepagent_thread_id or thread.thread_id)
            for thread in self.thread_store.list_threads()
        }
        for path in self._kernel_files():
            row = {
                "source": self._source_key(path, self.workspace),
                "status": "review_required",
                "issues": [],
            }
            try:
                payload = _read_json(path)
                thread_key = path.parent.name
                exact_thread = thread_key in known_thread_keys
                row.update(
                    {
                        "question": str(payload.get("question") or "").strip(),
                        "hypothesis_count": len(
                            list(payload.get("hypotheses") or [])
                        ),
                        "run_card_count": len(
                            list(payload.get("run_cards") or [])
                        ),
                        "exact_thread_match": exact_thread,
                    }
                )
                if not exact_thread:
                    row["issues"].append(
                        "Bare Kernel has no exact thread relationship and will "
                        "not be merged by question similarity."
                    )
                if payload.get("hypotheses"):
                    row["issues"].append(
                        "Kernel hypotheses lack structured rationale and "
                        "predictions and require review."
                    )
                if payload.get("run_cards"):
                    row["issues"].append(
                        "Run cards require an explicit producing experiment "
                        "and durable source before becoming Result nodes."
                    )
                totals["review_items"] += 1
            except Exception as exc:
                row["status"] = "quarantine"
                row["issues"].append(f"invalid or truncated JSON: {exc}")
                totals["quarantined_files"] += 1
            kernels.append(row)
        return {
            "workspace": str(self.workspace),
            "mode": "dry_run",
            "totals": totals,
            "campaigns": campaigns,
            "kernels": kernels,
        }

    def _load_or_create_plan(
        self,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        if self.plan_path.is_file():
            plan = _read_json(self.plan_path)
            if plan.get("workspace") != str(self.workspace):
                raise ValueError(
                    "Existing migration plan belongs to a different workspace."
                )
            return plan
        sources: dict[str, Any] = {}
        for campaign in report["campaigns"]:
            if campaign["status"] != "ready":
                continue
            payload = _read_json(
                self.workspace.joinpath(*Path(campaign["source"]).parts)
            )
            state = HypothesisEngineState.model_validate(payload)
            sources[campaign["source"]] = {
                "graph_id": _new_id("graph"),
                "hypotheses": {
                    item.id: _new_id("hyp") for item in state.hypotheses
                },
                "experiments": {
                    item.id: _new_id("exp") for item in state.actions
                },
                "results": {
                    item.action_id: _new_id("res")
                    for item in state.evidence
                },
                "blocked_results": {
                    item.id: _new_id("res")
                    for item in state.actions
                    if item.failure_reason
                },
            }
        plan = {
            "workspace": str(self.workspace),
            "created_at": time.time(),
            "sources": sources,
        }
        _atomic_json(self.plan_path, plan)
        return plan

    def _source_ref(
        self,
        source: str,
    ) -> dict[str, str] | None:
        value = str(source or "").strip()
        if not value:
            return None
        lower = value.lower()
        if lower.startswith(("http://", "https://")):
            if "doi.org/" in lower:
                return {
                    "ref_kind": RefKind.DOI.value,
                    "ref_id": value.split("doi.org/", 1)[1],
                }
            return {"ref_kind": RefKind.URL.value, "ref_id": value}
        if lower.startswith("doi:") or value.startswith("10."):
            return {
                "ref_kind": RefKind.DOI.value,
                "ref_id": value.removeprefix("doi:").strip(),
            }
        if (system_root(self.workspace) / "runs" / value).is_dir():
            return {"ref_kind": RefKind.RUN.value, "ref_id": value}
        if ArtifactRegistry(
            workspace=self.workspace,
            workspace_id=self.workspace.name,
        ).get(value) is not None:
            return {"ref_kind": RefKind.ARTIFACT.value, "ref_id": value}
        candidate = (self.workspace / value).resolve()
        files_root = self.files_root.resolve()
        try:
            candidate.relative_to(files_root)
        except ValueError:
            candidate = Path()
        if candidate.is_file():
            return {
                "ref_kind": RefKind.NOTE.value,
                "ref_id": str(candidate.relative_to(self.workspace)).replace(
                    "\\", "/"
                ),
            }
        return None

    def _matching_threads(self, campaign_key: str) -> list[Any]:
        rows = []
        for thread in self.thread_store.list_threads():
            meta = dict(thread.meta or {})
            identities = {
                safe_thread_id(thread.deepagent_thread_id or thread.thread_id)
            }
            legacy_campaign_id = str(
                meta.get("research_campaign_id") or ""
            ).strip()
            if legacy_campaign_id:
                identities.add(safe_thread_id(legacy_campaign_id))
            if campaign_key in identities:
                rows.append(thread)
        return rows

    def _valid_active_child(
        self,
        campaign_key: str,
        active_action_id: str,
    ) -> Any | None:
        action_id = str(active_action_id or "").strip()
        if not action_id:
            return None
        for thread in self._matching_threads(campaign_key):
            meta = dict(thread.meta or {})
            if str(meta.get("research_map_action_id") or "") != action_id:
                continue
            if str(thread.status.value) in {
                "running",
                "stopping",
                "interrupted",
            }:
                return thread
        return None

    def _bind_campaign_threads(
        self,
        *,
        campaign_key: str,
        graph_id: str,
        experiment_ids: dict[str, str],
    ) -> None:
        """Idempotently repair formal bindings after the graph commit."""

        for thread in self._matching_threads(campaign_key):
            action_id = str(
                dict(thread.meta or {}).get("research_map_action_id") or ""
            )
            focus_node_id = str(experiment_ids.get(action_id) or "")
            if (
                thread.active_research_graph_id == graph_id
                and thread.research_focus_node_id == focus_node_id
            ):
                continue
            self.thread_store.update_thread(
                thread.thread_id,
                active_research_graph_id=graph_id,
                research_focus_node_id=focus_node_id,
            )

    def _repair_released_active_action(
        self,
        *,
        graph_id: str,
        experiment_node_id: str,
    ) -> None:
        """Repair a graph committed by an interrupted older import."""

        now = time.time()
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            node = connection.execute(
                """
                SELECT state
                FROM research_nodes
                WHERE graph_id = ? AND node_id = ?
                """,
                (graph_id, experiment_node_id),
            ).fetchone()
            if node is None:
                return
            active_launches = connection.execute(
                """
                SELECT launch_id
                FROM research_launches
                WHERE graph_id = ? AND experiment_node_id = ?
                  AND status IN ('claimed', 'submitting', 'running', 'unknown')
                """,
                (graph_id, experiment_node_id),
            ).fetchall()
            needs_node_release = str(node["state"]) == "running"
            if not needs_node_release and not active_launches:
                return
            connection.execute(
                """
                UPDATE research_launches
                SET status = 'blocked', lease_owner = '', lease_until = 0,
                    updated_at = ?
                WHERE graph_id = ? AND experiment_node_id = ?
                  AND status IN ('claimed', 'submitting', 'running', 'unknown')
                """,
                (now, graph_id, experiment_node_id),
            )
            if needs_node_release:
                connection.execute(
                    """
                    UPDATE research_nodes
                    SET state = 'ready', revision = revision + 1,
                        updated_at = ?
                    WHERE graph_id = ? AND node_id = ?
                    """,
                    (now, graph_id, experiment_node_id),
                )
            connection.execute(
                """
                UPDATE research_graphs
                SET revision = revision + 1, updated_at = ?
                WHERE graph_id = ?
                """,
                (now, graph_id),
            )
            graph = connection.execute(
                """
                SELECT revision FROM research_graphs
                WHERE graph_id = ?
                """,
                (graph_id,),
            ).fetchone()
            self.store._write_event(
                connection,
                graph_id=graph_id,
                revision=int(graph["revision"]),
                change="legacy.active_action_released",
                node_ids=[experiment_node_id],
            )

    def _import_campaign(
        self,
        *,
        path: Path,
        mapping: dict[str, Any],
    ) -> dict[str, int]:
        state = HypothesisEngineState.model_validate(_read_json(path))
        graph_id = str(mapping["graph_id"])
        hypothesis_ids = dict(mapping["hypotheses"])
        experiment_ids = dict(mapping["experiments"])
        result_ids = dict(mapping["results"])
        blocked_ids = dict(mapping["blocked_results"])
        campaign_key = path.parent.name
        active_child = self._valid_active_child(
            campaign_key,
            state.active_action_id,
        )
        try:
            self.store.get_graph(graph_id)
            if state.active_action_id and active_child is None:
                self._repair_released_active_action(
                    graph_id=graph_id,
                    experiment_node_id=experiment_ids[state.active_action_id],
                )
            self._bind_campaign_threads(
                campaign_key=campaign_key,
                graph_id=graph_id,
                experiment_ids=experiment_ids,
            )
            return {
                "graphs": 0,
                "nodes": 0,
                "edges": 0,
                "refs": 0,
                "launches": 0,
            }
        except KeyError:
            pass
        now = time.time()
        refs_count = 0
        edge_count = 0
        launch_count = 0
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO research_graphs (
                    graph_id, title, question, orchestration_mode,
                    archived, revision, created_at, updated_at
                ) VALUES (?, ?, ?, 'manual', 0, 1, ?, ?)
                """,
                (
                    graph_id,
                    state.question[:300],
                    state.question,
                    now,
                    now,
                ),
            )
            for hypothesis in state.hypotheses:
                node_id = hypothesis_ids[hypothesis.id]
                body = {
                    "claim": hypothesis.claim,
                    "rationale": hypothesis.rationale,
                    "predictions": hypothesis.predictions,
                }
                connection.execute(
                    """
                    INSERT INTO research_nodes (
                        graph_id, node_id, kind, title, state, body_json,
                        revision, created_at, updated_at
                    ) VALUES (?, ?, 'hypothesis', ?, '', ?, 1, ?, ?)
                    """,
                    (
                        graph_id,
                        node_id,
                        hypothesis.claim[:300],
                        json.dumps(body, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
            action_by_id = {action.id: action for action in state.actions}
            evidence_by_action = {
                judgment.action_id: judgment for judgment in state.evidence
            }
            for action in state.actions:
                node_id = experiment_ids[action.id]
                state_value = {
                    "planned": "ready",
                    "running": "running",
                    "completed": "has_results",
                    "failed": "blocked",
                }[action.status.value]
                if (
                    action.id == state.active_action_id
                    and active_child is None
                ):
                    state_value = "ready"
                lane = {
                    "literature": "literature_review",
                    "experiment": "experiment",
                    "workspace": "research",
                    "human": "research",
                }[action.executor.value]
                body = {
                    "objective": action.question,
                    "plan_summary": action.task,
                    "decision_rule": action.decision_rule,
                    "execution_lane": lane,
                }
                connection.execute(
                    """
                    INSERT INTO research_nodes (
                        graph_id, node_id, kind, title, state, body_json,
                        revision, created_at, updated_at
                    ) VALUES (?, ?, 'experiment', ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        graph_id,
                        node_id,
                        action.question[:300],
                        state_value,
                        json.dumps(body, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                for hypothesis_id in action.target_hypotheses:
                    connection.execute(
                        """
                        INSERT INTO research_edges (
                            graph_id, source_node_id, target_node_id, relation
                        ) VALUES (?, ?, ?, 'tests')
                        """,
                        (
                            graph_id,
                            hypothesis_ids[hypothesis_id],
                            node_id,
                        ),
                    )
                    edge_count += 1
                for dependency_id in action.prerequisite_action_ids:
                    connection.execute(
                        """
                        INSERT INTO research_edges (
                            graph_id, source_node_id, target_node_id, relation
                        ) VALUES (?, ?, ?, 'depends_on')
                        """,
                        (
                            graph_id,
                            node_id,
                            experiment_ids[dependency_id],
                        ),
                    )
                    edge_count += 1
                if action.failure_reason:
                    blocked_id = blocked_ids[action.id]
                    connection.execute(
                        """
                        INSERT INTO research_nodes (
                            graph_id, node_id, kind, title, state, body_json,
                            revision, created_at, updated_at
                        ) VALUES (?, ?, 'result', 'Execution blocked', '', ?, 1, ?, ?)
                        """,
                        (
                            graph_id,
                            blocked_id,
                            json.dumps(
                                {"summary": action.failure_reason},
                                ensure_ascii=False,
                            ),
                            now,
                            now,
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO research_edges (
                            graph_id, source_node_id, target_node_id, relation
                        ) VALUES (?, ?, ?, 'produces')
                        """,
                        (graph_id, node_id, blocked_id),
                    )
                    edge_count += 1
            for action_id, judgment in evidence_by_action.items():
                result_id = result_ids[action_id]
                action_id_node = experiment_ids[action_id]
                connection.execute(
                    """
                    INSERT INTO research_nodes (
                        graph_id, node_id, kind, title, state, body_json,
                        revision, created_at, updated_at
                    ) VALUES (?, ?, 'result', ?, '', ?, 1, ?, ?)
                    """,
                    (
                        graph_id,
                        result_id,
                        f"Result: {action_by_id[action_id].question}"[:300],
                        json.dumps(
                            {"summary": judgment.summary},
                            ensure_ascii=False,
                        ),
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO research_edges (
                        graph_id, source_node_id, target_node_id, relation
                    ) VALUES (?, ?, ?, 'produces')
                    """,
                    (graph_id, action_id_node, result_id),
                )
                edge_count += 1
                for effect in judgment.effects:
                    connection.execute(
                        """
                        INSERT INTO research_edges (
                            graph_id, source_node_id, target_node_id, relation
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            graph_id,
                            result_id,
                            hypothesis_ids[effect.hypothesis_id],
                            effect.verdict.value,
                        ),
                    )
                    edge_count += 1
                source_ref = self._source_ref(judgment.source)
                if source_ref is not None:
                    connection.execute(
                        """
                        INSERT INTO research_refs (
                            graph_id, node_id, ref_kind, ref_id
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            graph_id,
                            result_id,
                            source_ref["ref_kind"],
                            source_ref["ref_id"],
                        ),
                    )
                    refs_count += 1

            if state.active_action_id:
                if active_child is not None:
                    launch_id = _new_id("launch")
                    connection.execute(
                        """
                        INSERT INTO research_launches (
                            launch_id, graph_id, experiment_node_id,
                            idempotency_key, status, thread_id, run_id,
                            lease_owner, lease_until, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, '', 0, ?, ?)
                        """,
                        (
                            launch_id,
                            graph_id,
                            experiment_ids[state.active_action_id],
                            _new_id("migration"),
                            (
                                "running"
                                if active_child.status.value
                                in {"running", "stopping", "interrupted"}
                                else "unknown"
                            ),
                            active_child.thread_id,
                            active_child.active_run_id,
                            now,
                            now,
                        ),
                    )
                    launch_count = 1
            connection.execute(
                """
                INSERT INTO ui_events (
                    event_type, thread_id, graph_id, payload_json, created_at
                ) VALUES ('research_graph.updated', '', ?, ?, ?)
                """,
                (
                    graph_id,
                    json.dumps(
                        {
                            "graph_id": graph_id,
                            "revision": 1,
                            "change": "legacy.imported",
                            "node_ids": [],
                        },
                        ensure_ascii=False,
                    ),
                    now,
                ),
            )
        self._bind_campaign_threads(
            campaign_key=campaign_key,
            graph_id=graph_id,
            experiment_ids=experiment_ids,
        )
        return {
            "graphs": 1,
            "nodes": (
                len(state.hypotheses)
                + len(state.actions)
                + len(state.evidence)
                + len(blocked_ids)
            ),
            "edges": edge_count,
            "refs": refs_count,
            "launches": launch_count,
        }

    def apply(self) -> dict[str, Any]:
        if self.progress_path.is_file():
            progress = _read_json(self.progress_path)
            if progress.get("workspace") != str(self.workspace):
                raise ValueError(
                    "In-progress Research Graph migration belongs to another "
                    "workspace."
                )
            manifest_path = self.workspace.joinpath(
                *Path(str(progress["manifest"])).parts
            )
            manifest = _read_json(manifest_path)
            plan = _read_json(self.plan_path)
        else:
            dry_run = self.dry_run()
            plan = self._load_or_create_plan(dry_run)
            # The random suffix keeps rollback manifests immutable when two
            # idempotency checks happen within the same second.
            stamp = (
                f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
                f"-{uuid.uuid4().hex[:8]}"
            )
            archive_dir = self.archive_root / stamp
            manifest_path = archive_dir / "rollback_manifest.json"
            report_by_source = {
                row["source"]: row
                for row in [*dry_run["campaigns"], *dry_run["kernels"]]
            }
            import_sources = [
                source
                for source in plan["sources"]
                if report_by_source.get(source, {}).get("status") == "ready"
                and self.workspace.joinpath(*Path(source).parts).is_file()
            ]
            planned_moves: list[dict[str, Any]] = []
            for source, row in report_by_source.items():
                source_path = self.workspace.joinpath(*Path(source).parts)
                if not source_path.is_file():
                    continue
                category = (
                    "quarantine"
                    if row["status"] == "quarantine"
                    else "review"
                    if row["status"] == "review_required"
                    else "imported"
                )
                destination = archive_dir / category / source
                planned_moves.append(
                    {
                        "source": source,
                        "archived": str(
                            destination.relative_to(self.workspace)
                        ).replace("\\", "/"),
                        "review": (
                            row
                            if row["status"] != "ready" or row.get("issues")
                            else {}
                        ),
                    }
                )
            thread_bindings_before: dict[str, dict[str, str]] = {}
            for source in import_sources:
                source_path = self.workspace.joinpath(*Path(source).parts)
                for thread in self._matching_threads(source_path.parent.name):
                    thread_bindings_before.setdefault(
                        thread.thread_id,
                        {
                            "active_research_graph_id": (
                                thread.active_research_graph_id
                            ),
                            "research_focus_node_id": (
                                thread.research_focus_node_id
                            ),
                        },
                    )
            manifest = {
                "workspace": str(self.workspace),
                "created_at": time.time(),
                "archive_dir": str(archive_dir.relative_to(self.workspace)),
                "graphs": [
                    str(plan["sources"][source]["graph_id"])
                    for source in import_sources
                ],
                "import_sources": import_sources,
                "planned_moves": planned_moves,
                "moves": [],
                "review_queue": [],
                "thread_bindings_before": thread_bindings_before,
                "counts": {
                    "graphs": 0,
                    "nodes": 0,
                    "edges": 0,
                    "refs": 0,
                    "launches": 0,
                },
                "completed": False,
            }
            # Persist recovery identity before the first database or file
            # mutation. Each later step updates this same immutable manifest.
            _atomic_json(manifest_path, manifest)
            _atomic_json(
                self.progress_path,
                {
                    "workspace": str(self.workspace),
                    "manifest": str(
                        manifest_path.relative_to(self.workspace)
                    ).replace("\\", "/"),
                },
            )

        if manifest.get("completed"):
            self.progress_path.unlink(missing_ok=True)
            return {
                "mode": "applied",
                "manifest": str(
                    manifest_path.relative_to(self.workspace)
                ).replace("\\", "/"),
                **manifest,
            }

        for source in list(manifest.get("import_sources") or []):
            mapping = dict(plan["sources"][source])
            source_path = self.workspace.joinpath(*Path(source).parts)
            if source_path.is_file():
                self._import_campaign(path=source_path, mapping=mapping)
                continue
            # A crash may happen after the database commit. The stable plan ID
            # lets the resumed process distinguish that case from lost input.
            self.store.get_graph(str(mapping["graph_id"]))
            self._bind_campaign_threads(
                campaign_key=source_path.parent.name,
                graph_id=str(mapping["graph_id"]),
                experiment_ids=dict(mapping["experiments"]),
            )

        completed_moves = {
            str(row["source"]) for row in list(manifest.get("moves") or [])
        }
        for planned in list(manifest.get("planned_moves") or []):
            source = str(planned["source"])
            if source in completed_moves:
                continue
            source_path = self.workspace.joinpath(*Path(source).parts)
            destination = self.workspace.joinpath(
                *Path(str(planned["archived"])).parts
            )
            if source_path.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_path), str(destination))
            elif not destination.is_file():
                raise FileNotFoundError(
                    f"Legacy Research state vanished during migration: {source}"
                )
            manifest["moves"].append(
                {
                    "source": source,
                    "archived": str(planned["archived"]),
                }
            )
            review = dict(planned.get("review") or {})
            if review and not any(
                row.get("source") == source
                for row in manifest["review_queue"]
            ):
                manifest["review_queue"].append(review)
            _atomic_json(manifest_path, manifest)

        counts = {
            "graphs": 0,
            "nodes": 0,
            "edges": 0,
            "refs": 0,
            "launches": 0,
        }
        for graph_id in list(manifest.get("graphs") or []):
            snapshot = self.store.get_snapshot(str(graph_id))
            counts["graphs"] += 1
            counts["nodes"] += len(snapshot["nodes"])
            counts["edges"] += len(snapshot["edges"])
            counts["refs"] += len(snapshot["refs"])
            counts["launches"] += len(snapshot["launches"])
        manifest["counts"] = counts
        manifest["completed"] = True
        _atomic_json(manifest_path, manifest)
        self.progress_path.unlink(missing_ok=True)
        return {
            "mode": "applied",
            "manifest": str(manifest_path.relative_to(self.workspace)).replace(
                "\\", "/"
            ),
            **manifest,
        }

    def rollback(self, manifest_path: Path | str) -> dict[str, Any]:
        path = Path(manifest_path)
        if not path.is_absolute():
            path = self.workspace / path
        path = path.resolve()
        try:
            path.relative_to(self.archive_root.resolve())
        except ValueError as exc:
            raise ValueError(
                "Rollback manifest must be under metadata/legacy_research_state."
            ) from exc
        manifest = _read_json(path)
        if manifest.get("workspace") != str(self.workspace):
            raise ValueError("Rollback manifest belongs to another workspace.")
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for graph_id in list(manifest.get("graphs") or []):
                connection.execute(
                    "DELETE FROM ui_events WHERE graph_id = ?",
                    (str(graph_id),),
                )
                connection.execute(
                    "DELETE FROM research_graphs WHERE graph_id = ?",
                    (str(graph_id),),
                )
        for thread_id, binding in dict(
            manifest.get("thread_bindings_before") or {}
        ).items():
            try:
                self.thread_store.update_thread(
                    str(thread_id),
                    active_research_graph_id=str(
                        binding.get("active_research_graph_id") or ""
                    ),
                    research_focus_node_id=str(
                        binding.get("research_focus_node_id") or ""
                    ),
                )
            except KeyError:
                continue
        restored: list[str] = []
        for move in reversed(list(manifest.get("moves") or [])):
            source = self.workspace.joinpath(
                *Path(str(move["source"])).parts
            )
            archived = self.workspace.joinpath(
                *Path(str(move["archived"])).parts
            )
            if not archived.is_file() or source.exists():
                continue
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(archived), str(source))
            restored.append(str(move["source"]))
        if self.progress_path.is_file():
            progress = _read_json(self.progress_path)
            active_manifest = self.workspace.joinpath(
                *Path(str(progress.get("manifest") or "")).parts
            ).resolve()
            if active_manifest == path:
                self.progress_path.unlink(missing_ok=True)
        return {
            "mode": "rolled_back",
            "deleted_graphs": list(manifest.get("graphs") or []),
            "restored_sources": restored,
        }


__all__ = ["ResearchGraphMigrator"]
