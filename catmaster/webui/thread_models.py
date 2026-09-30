from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator

def utc_ts() -> float:
    return time.time()


class ThreadStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    INTERRUPTED = "interrupted"
    ERROR = "error"


class ThreadRole(str, Enum):
    PRIMARY = "primary"
    RESEARCH_ROOT = "research_root"
    RESEARCH_EXECUTION = "research_execution"
    RESEARCH_PLANNING = "research_planning"
    RESEARCH_COMPARISON = "research_comparison"
    # Read-only compatibility for thread rows written by the retired custom
    # AgentTask runtime. New background specialists use native LangGraph threads
    # and must not create this role.
    AGENT_TASK = "agent_task"


def is_persistent_research_root(thread: Any) -> bool:
    """Return whether a thread owns a Persistent Research orchestration session."""

    try:
        role = ThreadRole(getattr(thread, "thread_role", ThreadRole.PRIMARY))
    except (TypeError, ValueError):
        return False
    return (
        role is ThreadRole.RESEARCH_ROOT
        and str(getattr(thread, "entrypoint", "") or "").strip() == "persistent_research"
        and not str(getattr(thread, "parent_thread_id", "") or "").strip()
    )


def research_automation_paused(thread: Any, graph: dict[str, Any]) -> bool:
    """Read session pause, preserving a legacy Manual graph until user continuation.

    Explicit Persistent input adopts the graph into the session and clears the
    old mode in thread_support. New sessions use only automation_paused.
    """
    return bool((getattr(thread, "meta", None) or {}).get("automation_paused")) or (
        str(getattr(thread, "entrypoint", "")) == "persistent_research"
        and graph.get("orchestration_mode") == "manual"
    )


class MessagePart(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    # Persisted provider and legacy records may contain part kinds introduced
    # after this WebUI build. Keep the base record loadable so the public
    # projector can render its safe, human-readable unknown-part card. Typed
    # subclasses below retain their narrower literals for active writers.
    type: str
    text: str = ""
    status: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class ToolCallPart(MessagePart):
    type: Literal["tool-call"] = "tool-call"
    tool_call_id: str
    tool: str = ""
    input: dict[str, Any] = Field(default_factory=dict)
    output: Any = None


class ArtifactPart(MessagePart):
    type: Literal["artifact"] = "artifact"
    artifact_id: str
    renderer: str = "text"
    title: str = ""
    summary: str = ""
    path: str = ""


class InterruptRecord(BaseModel):
    interrupt_id: str
    thread_id: str
    message_id: str = ""
    part_id: str = ""
    status: Literal["pending", "resolved"] = "pending"
    kind: str = "approval"
    title: str = ""
    body: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=utc_ts)
    resolved_at: float | None = None
    resolution: dict[str, Any] | None = None


class ThreadMessage(BaseModel):
    id: str
    thread_id: str
    role: Literal["user", "assistant", "system", "tool"]
    status: Literal["created", "streaming", "completed", "failed", "interrupted"] = "created"
    created_at: float = Field(default_factory=utc_ts)
    updated_at: float = Field(default_factory=utc_ts)
    parts: list[SerializeAsAny[MessagePart]] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    structured_sidecar: dict[str, Any] = Field(default_factory=dict)


class ThreadRecord(BaseModel):
    thread_id: str
    workspace_id: str
    # Retired descriptor fields remain parseable to detect intermediate Server
    # records that need explicit offline conversion. No online importer uses them.
    agent_server_thread_id: str = ""
    deepagent_thread_id: str = ""
    legacy_import_status: Literal["", "pending", "submitting", "closed"] = ""
    title: str = ""
    status: ThreadStatus = ThreadStatus.IDLE
    entrypoint: str = "research"
    created_at: float = Field(default_factory=utc_ts)
    updated_at: float = Field(default_factory=utc_ts)
    active_message_id: str = ""
    active_run_id: str = ""
    active_research_graph_id: str = ""
    research_focus_node_id: str = ""
    parent_thread_id: str = ""
    thread_role: ThreadRole = ThreadRole.PRIMARY
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _classify_legacy_research_threads(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        role = str(payload.get("thread_role") or "").strip()
        if role:
            payload.setdefault("parent_thread_id", "")
            return payload
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        internal_kind = str(meta.get("internal_kind") or "").strip()
        if internal_kind == "research_graph_planning":
            payload["thread_role"] = ThreadRole.RESEARCH_PLANNING.value
        elif internal_kind == "research_graph_experiment_comparison":
            payload["thread_role"] = ThreadRole.RESEARCH_COMPARISON.value
        elif (
            str(payload.get("entrypoint") or "").strip() == "persistent_research"
            and not str(payload.get("parent_thread_id") or "").strip()
        ):
            payload["thread_role"] = ThreadRole.RESEARCH_ROOT.value
        else:
            payload["thread_role"] = ThreadRole.PRIMARY.value
        payload.setdefault("parent_thread_id", "")
        return payload


class ArtifactRecord(BaseModel):
    artifact_id: str
    thread_id: str = ""
    message_id: str = ""
    tool_call_id: str = ""
    run_id: str = ""
    workspace_id: str = ""
    path: str
    mime_type: str = ""
    renderer: str = "text"
    title: str = ""
    summary: str = ""
    created_at: float = Field(default_factory=utc_ts)
    updated_at: float = Field(default_factory=utc_ts)
    preview_url: str = ""
    download_url: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class ThreadEventEnvelope(BaseModel):
    seq: int
    event: str
    thread_id: str
    message_id: str = ""
    status: str = ""
    created_at: float = Field(default_factory=utc_ts)
    data: dict[str, Any] = Field(default_factory=dict)


class ThreadCreateRequest(BaseModel):
    title: str = ""
    entrypoint: str = "research"
    permission_mode: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThreadSubmitRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    entrypoint: str = Field(default="", description="Omit to retain this thread's current entrypoint.")
    llm_config: str = Field(default="", alias="model_config")
    permission_mode: str = ""
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    strategy: Literal["enqueue", "interrupt", "rollback", "reject"] = "enqueue"


class ThreadResumeAction(BaseModel):
    action_id: str
    decision: Literal["approve", "edit", "reject", "respond"]
    fields: dict[str, str | int | float | bool] = Field(default_factory=dict)
    reason: str = ""


class ThreadResumeRequest(BaseModel):
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[ThreadResumeAction] = Field(default_factory=list)
    text: str = ""


class ThreadCheckpointContinueRequest(BaseModel):
    message_id: str


class ThreadStopRequest(BaseModel):
    run_id: str = ""
    action: Literal["interrupt", "rollback"] = "interrupt"
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_emergency(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        # Compatibility for stale browser bundles only. Both old choices
        # map to the safe keep-progress operation; the active UI sends action.
        payload.pop("emergency", None)
        return payload


class AsyncSubagentUpdateRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)


class AsyncSubagentStopRequest(BaseModel):
    run_id: str = ""
    action: Literal["interrupt", "rollback"] = "interrupt"
    reason: str = ""


class ThreadPatchRequest(BaseModel):
    title: str = ""
    entrypoint: str = ""
    status: ThreadStatus = ThreadStatus.IDLE
    permission_mode: str = ""
    active_research_graph_id: str = ""
    research_focus_node_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_nulls(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {key: item for key, item in value.items() if item is not None}


__all__ = [
    "ArtifactPart",
    "ArtifactRecord",
    "AsyncSubagentStopRequest",
    "AsyncSubagentUpdateRequest",
    "InterruptRecord",
    "MessagePart",
    "ThreadCreateRequest",
    "ThreadCheckpointContinueRequest",
    "ThreadEventEnvelope",
    "ThreadMessage",
    "ThreadPatchRequest",
    "is_persistent_research_root",
    "ThreadRecord",
    "ThreadRole",
    "ThreadResumeRequest",
    "ThreadResumeAction",
    "ThreadStatus",
    "ThreadStopRequest",
    "ThreadSubmitRequest",
    "ToolCallPart",
    "utc_ts",
]
