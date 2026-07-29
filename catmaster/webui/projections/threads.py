from __future__ import annotations

from typing import Any

from .models import PublicThread


def project_thread(thread: Any) -> PublicThread:
    raw = thread.model_dump(mode="json") if hasattr(thread, "model_dump") else dict(thread or {})
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    permission_mode = "hitl" if str(meta.get("permission_mode") or "").lower() == "hitl" else "auto"
    status = raw.get("status")
    if hasattr(status, "value"):
        status = status.value
    return PublicThread(
        thread_id=str(raw.get("thread_id") or ""),
        title=str(raw.get("title") or "Untitled thread"),
        status=str(status or "idle"),
        entrypoint=str(raw.get("entrypoint") or "research"),
        permission_mode=permission_mode,
        active_research_graph_id=str(raw.get("active_research_graph_id") or ""),
        research_focus_node_id=str(raw.get("research_focus_node_id") or ""),
        created_at=float(raw.get("created_at") or 0.0),
        updated_at=float(raw.get("updated_at") or 0.0),
    )
