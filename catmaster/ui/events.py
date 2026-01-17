from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class UIEvent:
    ts: float
    level: str
    category: str
    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None
    task_id: Optional[str] = None
    step_id: Optional[int] = None


def make_event(
    name: str,
    *,
    level: str = "info",
    category: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    task_id: Optional[str] = None,
    step_id: Optional[int] = None,
) -> UIEvent:
    if category is None:
        category = name.split("_", 1)[0].lower()
    return UIEvent(
        ts=time.time(),
        level=level,
        category=category,
        name=name,
        payload=payload or {},
        run_id=run_id,
        task_id=task_id,
        step_id=step_id,
    )


__all__ = ["UIEvent", "make_event"]
