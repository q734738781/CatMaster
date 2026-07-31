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
    deepagent_thread_id: str
    title: str = ""
    status: ThreadStatus = ThreadStatus.IDLE
    entrypoint: str = "research"
    created_at: float = Field(default_factory=utc_ts)
    updated_at: float = Field(default_factory=utc_ts)
    active_message_id: str = ""
    active_run_id: str = ""
    active_research_graph_id: str = ""
    research_focus_node_id: str = ""
    pending_steering: list[dict[str, Any]] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


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
    entrypoint: str = "research"
    llm_config: str = Field(default="", alias="model_config")
    permission_mode: str = ""
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class ThreadResumeAction(BaseModel):
    action_id: str
    decision: Literal["approve", "edit", "reject", "respond"]
    fields: dict[str, str | int | float | bool] = Field(default_factory=dict)
    reason: str = ""


class ThreadResumeRequest(BaseModel):
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[ThreadResumeAction] = Field(default_factory=list)
    text: str = ""


class ThreadStopRequest(BaseModel):
    emergency: bool = False
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
    "InterruptRecord",
    "MessagePart",
    "ThreadCreateRequest",
    "ThreadEventEnvelope",
    "ThreadMessage",
    "ThreadPatchRequest",
    "ThreadRecord",
    "ThreadResumeRequest",
    "ThreadResumeAction",
    "ThreadStatus",
    "ThreadStopRequest",
    "ThreadSubmitRequest",
    "ToolCallPart",
    "utc_ts",
]
