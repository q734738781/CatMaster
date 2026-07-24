from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from catmaster.research.hypothesis_engine import ExecutionLane, ExecutionPacket
from catmaster.research.hypothesis_engine.storage import (
    campaign_lock,
    load_engine,
    save_engine,
)

from .thread_models import ThreadStatus, ThreadSubmitRequest
from .thread_store import ThreadStore

logger = logging.getLogger(__name__)

AUTOPILOT_ENABLED_META = "research_map_autopilot_enabled"
CAMPAIGN_ID_META = "research_campaign_id"
SOURCE_THREAD_ID_META = "research_map_source_thread_id"
ACTION_ID_META = "research_map_action_id"
LAUNCH_MODE_META = "research_map_launch_mode"

LaunchMode = Literal["manual", "auto"]


def _permission_mode(thread: Any) -> str:
    return "hitl" if str(dict(thread.meta or {}).get("permission_mode") or "") == "hitl" else "auto"


def _thread_title(packet: ExecutionPacket, launch_mode: LaunchMode) -> str:
    prefix = "Auto Research" if launch_mode == "auto" else "Research check"
    question = " ".join(str(packet.question or "").split())
    if len(question) > 76:
        question = question[:73].rstrip() + "..."
    return f"{prefix}: {question or packet.action_id}"


def build_action_research_prompt(
    *,
    campaign_id: str,
    packet: ExecutionPacket,
    revision: int,
    launch_mode: LaunchMode,
) -> str:
    hypotheses: list[str] = []
    for hypothesis in packet.hypotheses:
        hypotheses.extend(
            [
                f"- {hypothesis.id}: {hypothesis.claim}",
                f"  Rationale: {hypothesis.rationale}",
                "  Predictions:",
                *[f"  - {prediction}" for prediction in hypothesis.predictions],
            ]
        )
    trigger = (
        "The user selected this node in Research Map."
        if launch_mode == "manual"
        else "The asynchronous Research Map worker selected this node from the current ranking."
    )
    human_note = (
        "\nThis is a human-owned check. Ask the user for the required evidence in this thread "
        "and keep the campaign action active until that evidence is supplied."
        if packet.executor is ExecutionLane.HUMAN
        else ""
    )
    return "\n".join(
        [
            "Run one ordinary CatMaster Research turn for the selected hypothesis-map check.",
            "",
            trigger,
            "This is not a reduced controller turn: use the normal Research workflow, delegation, "
            "managed execution, streaming, and approval behavior.",
            "",
            f"Source campaign id: {campaign_id}",
            f"Reserved campaign revision: {revision}",
            f"Action id: {packet.action_id}",
            f"Execution owner: {packet.delegate_to}",
            f"Scientific question: {packet.question}",
            "",
            "Target hypotheses:",
            *hypotheses,
            "",
            "Bounded scientific task:",
            packet.task,
            "",
            "Decision rule:",
            packet.decision_rule,
            "",
            (
                "The action is already reserved in the source campaign. Inspect that campaign id "
                "before writing the result. If execution succeeds, send the actual result, exact "
                "source, complete target hypotheses, predictions, and decision rule to "
                "evidence_judge, then record only its structured judgment in the source campaign. "
                "If execution fails, record a concise failure without treating it as evidence. "
                "Do not initialize a second campaign and do not substitute this execution "
                "thread's id for the source campaign id."
            ),
            human_note,
            "",
            "Finish this turn after this one action is recorded or a real approval/user-input boundary is reached.",
        ]
    ).strip()


class ResearchMapService:
    """Launch and supervise ordinary Research threads for one scientific map action."""

    def __init__(self, *, agent_loop_factory) -> None:
        self._agent_loop_factory = agent_loop_factory

    @staticmethod
    def campaign_id(source_thread: Any) -> str:
        return str(source_thread.deepagent_thread_id or source_thread.thread_id)

    @staticmethod
    def related_children(store: ThreadStore, source_thread_id: str) -> list[Any]:
        return [
            thread
            for thread in store.list_threads()
            if str(dict(thread.meta or {}).get(SOURCE_THREAD_ID_META) or "")
            == source_thread_id
        ]

    def automation_snapshot(
        self,
        *,
        store: ThreadStore,
        source_thread: Any,
        engine: Any,
    ) -> dict[str, Any]:
        enabled = bool(dict(source_thread.meta or {}).get(AUTOPILOT_ENABLED_META))
        active_action_id = str(engine.state.active_action_id or "")
        children = self.related_children(store, source_thread.thread_id)
        active_child = next(
            (
                child
                for child in children
                if str(dict(child.meta or {}).get(ACTION_ID_META) or "")
                == active_action_id
            ),
            None,
        )
        controller = engine.controller_snapshot()
        status = "off"
        if enabled:
            if active_child is not None:
                if active_child.status in {ThreadStatus.RUNNING, ThreadStatus.STOPPING}:
                    status = "running"
                elif active_child.status is ThreadStatus.INTERRUPTED:
                    status = "waiting_review"
                elif active_child.status in {ThreadStatus.ERROR, ThreadStatus.STOPPED}:
                    status = "recovering"
                else:
                    packet = engine.active_packet()
                    status = (
                        "waiting_user"
                        if packet is not None and packet.executor is ExecutionLane.HUMAN
                        else "synchronizing"
                    )
            elif active_action_id:
                status = (
                    "running_in_source"
                    if source_thread.status
                    in {ThreadStatus.RUNNING, ThreadStatus.STOPPING, ThreadStatus.INTERRUPTED}
                    else "active_elsewhere"
                )
            elif controller["status"] == "complete":
                status = "complete"
            elif controller["status"] == "action_available":
                actions_by_id = {
                    action.id: action for action in engine.state.actions
                }
                auto_candidate = next(
                    (
                        actions_by_id[item.action_id]
                        for item in engine.rank_actions()
                        if item.eligible
                        and actions_by_id[item.action_id].executor
                        is not ExecutionLane.HUMAN
                    ),
                    None,
                )
                status = "ready" if auto_candidate is not None else "waiting_user"
            else:
                status = str(controller["status"])
        elif (
            active_child is not None
            and str(dict(active_child.meta or {}).get(LAUNCH_MODE_META) or "") == "auto"
            and active_child.status
            in {ThreadStatus.RUNNING, ThreadStatus.STOPPING, ThreadStatus.INTERRUPTED}
        ):
            status = "finishing_current"

        return {
            "enabled": enabled,
            "status": status,
            "active_action_id": active_action_id,
            "child_thread_id": str(active_child.thread_id if active_child else ""),
            "child_thread": active_child.model_dump(mode="json") if active_child else None,
        }

    async def launch_action(
        self,
        *,
        workspace: Path,
        workspace_id: str,
        source_thread_id: str,
        action_id: str,
        expected_revision: int = -1,
        launch_mode: LaunchMode,
    ) -> dict[str, Any]:
        store = ThreadStore(workspace=workspace, workspace_id=workspace_id)
        source_thread = store.get_thread(source_thread_id)
        if source_thread.status in {
            ThreadStatus.RUNNING,
            ThreadStatus.STOPPING,
            ThreadStatus.INTERRUPTED,
        }:
            raise ValueError(
                "the source Research thread is still running or waiting for review; "
                "finish or stop that turn before launching a Map action"
            )
        campaign_id = self.campaign_id(source_thread)
        files_root = Path(workspace) / "files"
        requested_action_id = str(action_id or "").strip()
        if not requested_action_id:
            raise ValueError("action_id is required")

        with campaign_lock(files_root, campaign_id):
            engine = load_engine(files_root, campaign_id)
            if expected_revision >= 0 and engine.state.revision != expected_revision:
                raise ValueError(
                    "stale campaign revision: "
                    f"expected {expected_revision}, current {engine.state.revision}; refresh and retry"
                )
            packet = engine.advance(requested_action_id)
            save_engine(files_root, campaign_id, engine)
            reserved_revision = engine.state.revision

        child = store.create_thread(
            title=_thread_title(packet, launch_mode),
            entrypoint="research",
            meta={
                "permission_mode": _permission_mode(source_thread),
                CAMPAIGN_ID_META: campaign_id,
                SOURCE_THREAD_ID_META: source_thread.thread_id,
                ACTION_ID_META: packet.action_id,
                LAUNCH_MODE_META: launch_mode,
            },
        )
        loop = self._agent_loop_factory(Path(workspace), workspace_id)
        loop.broker.emit(
            child.thread_id,
            "thread.created",
            status=str(child.status.value),
            data={"thread": child.model_dump(mode="json")},
        )
        prompt = build_action_research_prompt(
            campaign_id=campaign_id,
            packet=packet,
            revision=reserved_revision,
            launch_mode=launch_mode,
        )
        try:
            result = await loop.submit(
                thread_id=child.thread_id,
                payload=ThreadSubmitRequest(
                    text=prompt,
                    entrypoint="research",
                    permission_mode=_permission_mode(source_thread),
                ),
            )
        except Exception:
            with campaign_lock(files_root, campaign_id):
                current = load_engine(files_root, campaign_id)
                if current.state.active_action_id == packet.action_id:
                    current.release_active(packet.action_id)
                    save_engine(files_root, campaign_id, current)
            store.update_thread(child.thread_id, status=ThreadStatus.ERROR)
            raise

        return {
            "accepted": True,
            "launch_mode": launch_mode,
            "campaign_thread_id": campaign_id,
            "action_id": packet.action_id,
            "thread": result["thread"].model_dump(mode="json"),
            "message": result["message"].model_dump(mode="json"),
        }

    def set_autopilot(
        self,
        *,
        workspace: Path,
        workspace_id: str,
        source_thread_id: str,
        enabled: bool,
    ) -> Any:
        store = ThreadStore(workspace=workspace, workspace_id=workspace_id)
        source = store.get_thread(source_thread_id)
        meta = dict(source.meta or {})
        meta[AUTOPILOT_ENABLED_META] = bool(enabled)
        return store.update_thread(source.thread_id, meta=meta)

    def reconcile_finished_child(
        self,
        *,
        workspace: Path,
        workspace_id: str,
        child_thread_id: str,
        terminal_status: str,
    ) -> bool:
        store = ThreadStore(workspace=workspace, workspace_id=workspace_id)
        try:
            child = store.get_thread(child_thread_id)
        except KeyError:
            return False
        meta = dict(child.meta or {})
        campaign_id = str(meta.get(CAMPAIGN_ID_META) or "")
        action_id = str(meta.get(ACTION_ID_META) or "")
        if not campaign_id or not action_id:
            return False

        normalized = str(terminal_status or "").strip().lower()
        files_root = Path(workspace) / "files"
        with campaign_lock(files_root, campaign_id):
            try:
                engine = load_engine(files_root, campaign_id)
            except FileNotFoundError:
                return False
            if engine.state.active_action_id != action_id:
                return False
            packet = engine.active_packet()
            if packet is not None and packet.executor is ExecutionLane.HUMAN and normalized in {
                "done",
                "idle",
            }:
                return False
            if normalized in {"interrupted", "running", "stopping"}:
                return False
            reason = (
                "Map-launched Research thread finished without recording an evidence judgment."
                if normalized in {"done", "idle", "complete", "completed"}
                else f"Map-launched Research thread ended with status {normalized or 'unknown'}."
            )
            engine.record_result(
                action_id,
                outcome="failed",
                failure_reason=reason,
            )
            save_engine(files_root, campaign_id, engine)
            return True

    async def tick_workspace(
        self,
        *,
        workspace: Path,
        workspace_id: str,
    ) -> int:
        store = ThreadStore(workspace=workspace, workspace_id=workspace_id)
        launched = 0
        for source in store.list_threads():
            if not bool(dict(source.meta or {}).get(AUTOPILOT_ENABLED_META)):
                continue
            campaign_id = self.campaign_id(source)
            try:
                engine = load_engine(Path(workspace) / "files", campaign_id)
            except (FileNotFoundError, ValueError):
                continue

            if engine.state.active_action_id:
                children = self.related_children(store, source.thread_id)
                child = next(
                    (
                        item
                        for item in children
                        if str(dict(item.meta or {}).get(ACTION_ID_META) or "")
                        == engine.state.active_action_id
                    ),
                    None,
                )
                if child is not None and child.status in {
                    ThreadStatus.ERROR,
                    ThreadStatus.STOPPED,
                }:
                    self.reconcile_finished_child(
                        workspace=workspace,
                        workspace_id=workspace_id,
                        child_thread_id=child.thread_id,
                        terminal_status=child.status.value,
                    )
                continue

            if source.status in {
                ThreadStatus.RUNNING,
                ThreadStatus.STOPPING,
                ThreadStatus.INTERRUPTED,
            }:
                continue

            actions_by_id = {
                action.id: action for action in engine.state.actions
            }
            selected = next(
                (
                    actions_by_id[item.action_id]
                    for item in engine.rank_actions()
                    if item.eligible
                    and actions_by_id[item.action_id].executor
                    is not ExecutionLane.HUMAN
                ),
                None,
            )
            if selected is None:
                continue
            try:
                await self.launch_action(
                    workspace=workspace,
                    workspace_id=workspace_id,
                    source_thread_id=source.thread_id,
                    action_id=selected.id,
                    expected_revision=engine.state.revision,
                    launch_mode="auto",
                )
                launched += 1
            except ValueError:
                logger.info(
                    "Research Map auto selection changed before launch for %s",
                    source.thread_id,
                )
            except Exception:
                logger.exception(
                    "Research Map auto launch failed for %s",
                    source.thread_id,
                )
        return launched


__all__ = [
    "ACTION_ID_META",
    "AUTOPILOT_ENABLED_META",
    "CAMPAIGN_ID_META",
    "LAUNCH_MODE_META",
    "ResearchMapService",
    "SOURCE_THREAD_ID_META",
    "build_action_research_prompt",
]
