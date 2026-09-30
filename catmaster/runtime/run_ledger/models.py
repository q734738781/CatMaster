from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class RunLedgerEntry:
    project_id: str
    run_id: str
    lane: str
    status: str
    request: str
    answer_summary: str
    search_blob_text: str
    run_export_relpath: str
    ts_start: str
    ts_end: str
    model_name: str
    provider: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunSearchBlob:
    run_id: str
    request: str
    answer_summary: str
    task_goals: List[str] = field(default_factory=list)
    tool_names: List[str] = field(default_factory=list)
    artifact_paths: List[str] = field(default_factory=list)
    search_blob_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunSearchHit:
    run_id: str
    project_id: str
    lane: str
    status: str
    score: float
    source: str
    request: str = ""
    answer_summary: str = ""
    run_export_relpath: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunEvidenceChunk:
    run_id: str
    path: str
    section: str
    line_range: List[int]
    text: str
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HistoricalRunsContextPack:
    query: str
    selected_runs: List[str]
    context_text: str
    citations: List[Dict[str, Any]]
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = [
    "RunLedgerEntry",
    "RunSearchBlob",
    "RunSearchHit",
    "RunEvidenceChunk",
    "HistoricalRunsContextPack",
]
