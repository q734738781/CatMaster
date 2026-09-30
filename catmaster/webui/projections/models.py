from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema


class PublicField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    value: str
    href: str = ""
    copy_value: str = ""


class PublicFormField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    label: str
    value: str = ""
    input_type: Literal["text", "textarea", "number", "boolean"] = "text"
    required: bool = False


class PublicAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    kind: Literal["primary", "secondary", "danger", "link", "copy"] = "secondary"
    href: str = ""
    endpoint: str = ""
    decision: str = ""
    requires_reason: bool = False
    confirmation: str = ""
    fields: list[PublicFormField] = Field(default_factory=list)


class PublicItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    status: str = ""
    summary: str = ""
    href: str = ""


class TruncationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shown_count: int = 0
    total_count: int = 0
    total_unknown: bool = False
    truncated: bool = False
    next_cursor: str = ""
    full_content_ref: str = ""
    unit: Literal["characters", "bytes", "rows", "items"] = "characters"
    range_start: int = 0
    range_end: int = 0


class PublicPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Async brief excerpts; the task detail endpoint serves the complete text.
    on_completion: str = "resume_parent"
    task_description: str = ""
    task_cost: str = ""
    capacity_state: str = ""
    task_followup: str = ""
    task_followup_status: str = ""
    progress_summary: str = ""
    progress_next_step: str = ""

    id: str
    type: Literal[
        "text",
        "reasoning",
        "artifact",
        "tool",
        "receipt",
        "interrupt",
        "progress",
        "subagent",
        "error",
        "citations",
        "attachment",
        "unknown",
    ]
    status: str = ""
    title: str = ""
    activity_group_id: str = ""
    activity_group_title: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    duration_seconds: float = 0.0
    summary: str = ""
    text: str = ""
    fields: list[PublicField] = Field(default_factory=list)
    actions: list[PublicAction] = Field(default_factory=list)
    items: list[PublicItem] = Field(default_factory=list)
    artifact_id: str = ""
    renderer: str = ""
    path: str = ""
    detail_ref: str = ""
    input_ref: str = ""
    output_ref: str = ""
    diagnostics_ref: str = ""
    truncation: TruncationInfo = Field(default_factory=TruncationInfo)


class PublicMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origin: str = ""
    notification_title: str = ""
    source: str = ""

    id: str
    role: Literal["user", "assistant", "system", "tool"]
    status: str
    created_at: float
    updated_at: float
    parts: list[PublicPart] = Field(default_factory=list)
    parts_page: TruncationInfo = Field(default_factory=lambda: TruncationInfo(unit="items"))


class PublicResearchActivity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal[
        "idle",
        "planning",
        "comparing",
        "reviewing",
        "validating",
        "parked",
        "running",
        "waiting",
        "waiting_continue",
        "waiting_review",
        "operationally_incomplete",
        "paused",
        "blocked",
        "completed",
    ] = "idle"
    current_title: str = ""
    latest_progress: str = ""
    decision_summary: str = ""
    latest_result_title: str = ""
    latest_result_summary: str = ""
    latest_report_artifact_id: str = ""
    latest_report_path: str = ""
    latest_report_renderer: str = ""
    action_required: bool = False
    automation_paused: bool = False
    active_child_thread_id: str = ""
    active_child_role: Literal[
        "",
        "primary",
        "agent_task",
        "research_execution",
        "research_planning",
        "research_comparison",
    ] = ""
    execution_count: int = 0
    decision_round_count: int = 0
    updated_at: float = 0.0
    active_parts: list[PublicPart] = Field(default_factory=list)


class PublicThread(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    title: str
    status: str
    entrypoint: str
    permission_mode: str = "auto"
    active_run_id: str = ""
    pending_run_count: int = 0
    active_research_graph_id: str = ""
    research_focus_node_id: str = ""
    parent_thread_id: str = ""
    # ``agent_task`` remains readable for stored pre-Agent-Server threads only.
    thread_role: Literal[
        "primary",
        "research_root",
        "research_execution",
        "research_planning",
        "research_comparison",
        "agent_task",
    ] = "primary"
    research_activity: PublicResearchActivity = Field(
        default_factory=PublicResearchActivity
    )
    created_at: float
    updated_at: float


class PublicEventData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str = ""
    run_id: str = ""
    part_id: str = ""
    delta: str = ""
    text_offset: int = -1
    message: PublicMessage | SkipJsonSchema[None] = None
    thread: PublicThread | SkipJsonSchema[None] = None
    part: PublicPart | SkipJsonSchema[None] = None
    todo_parts: list[PublicPart] = Field(default_factory=list)
    title: str = ""
    summary: str = ""
    status: str = ""
    pending_run_count: int = 0
    fields: list[PublicField] = Field(default_factory=list)
    actions: list[PublicAction] = Field(default_factory=list)
    items: list[PublicItem] = Field(default_factory=list)
    diagnostics_ref: str = ""
    usage: list[PublicField] = Field(default_factory=list)


class PublicEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq: int
    event: str
    thread_id: str
    message_id: str = ""
    status: str = ""
    created_at: float
    data: PublicEventData = Field(default_factory=PublicEventData)


class PublicThreadEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread: PublicThread


class PublicThreadListEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threads: list[PublicThread] = Field(default_factory=list)


class PublicMessagePageEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[PublicMessage] = Field(default_factory=list)
    todo_parts: list[PublicPart] = Field(default_factory=list)
    active_parts: list[PublicPart] = Field(default_factory=list)
    stream_cursor: int = 0
    page: TruncationInfo


class PublicPartPageEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parts: list[PublicPart] = Field(default_factory=list)
    page: TruncationInfo


class PublicItemPageEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PublicItem] = Field(default_factory=list)
    page: TruncationInfo


class PublicPartEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    part: PublicPart


class PublicTextPageEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    page: TruncationInfo


class DeveloperDiagnosticsPageEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warning: str
    content_type: Literal["application/json"] = "application/json"
    content: str
    page: TruncationInfo


class PublicSubmitEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    queued: bool
    run_id: str = ""
    strategy: Literal["enqueue", "interrupt", "rollback", "reject"] = "enqueue"
    thread: PublicThread
    message: PublicMessage
    assistant_message: PublicMessage | SkipJsonSchema[None] = None


class PublicStopEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    status: str
    thread: PublicThread


class PublicResumeEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    run_id: str = ""
    assistant_message: PublicMessage
    thread: PublicThread
