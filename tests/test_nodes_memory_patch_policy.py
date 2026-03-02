from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core.prompts")

from catmaster.agents.nodes import memory_patch_node
from catmaster.runtime.memory_store import MemoryStore


class _FailIfInvokedModel:
    def invoke(self, _messages):
        raise AssertionError("model.invoke should not be called for success outcome")


class _Model:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        return SimpleNamespace(text=self.text)


class _Backend:
    def __init__(self, *, status: str = "success", content: str = "") -> None:
        self.status = status
        self.content = content
        self.calls: list[tuple[str, str, str]] = []

    def call(self, name: str, arguments_json: str, *, toolcall_key: str):
        self.calls.append((name, arguments_json, toolcall_key))
        return SimpleNamespace(status=self.status, content=self.content)


def _store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore.create_default(workspace=tmp_path)
    store.ensure_exists()
    return store


def _state(*, outcome: str) -> dict:
    return {
        "current_task_id": "task_01",
        "current_task_packet": {"goal": "task goal"},
        "task_result": {
            "task_outcome": outcome,
            "task_summary": f"{outcome} summary",
            "key_artifacts": [{"path": "results/out.json", "description": "result", "kind": "report"}],
            "structured_result": {
                "summary": f"{outcome} summary",
                "facts": [],
                "files": [],
                "constraints": [],
                "open_questions": [],
                "decisions": [],
                "next_steps": [],
                "artifacts": [],
            },
        },
        "tasks": [{"task_id": "task_01", "status": "running"}],
        "observations": [],
    }


def test_memory_patch_node_defers_reconcile_for_success(tmp_path: Path) -> None:
    store = _store(tmp_path)
    backend = _Backend()
    result = memory_patch_node(
        _state(outcome="success"),
        model=_FailIfInvokedModel(),
        memory_store=store,
        run_id="run_01",
        patch_repair_attempts=1,
        tool_backend=backend,
    )

    assert backend.calls == []
    events = store.read_events_tail(limit=1)
    assert events
    assert events[-1]["task_id"] == "task_01"
    assert events[-1]["memory_reconcile_requested"] is False
    assert events[-1]["memory_patch_status"] == "deferred"
    assert result["tasks"][0]["status"] == "success"
    assert result["observations"] and result["observations"][-1]["task_id"] == "task_01"


def test_memory_patch_node_reconciles_for_failure(tmp_path: Path) -> None:
    store = _store(tmp_path)
    model = _Model(
        "\n".join(
            [
                "MEMORY/MEMORY.md",
                "<<<<<<< SEARCH",
                "- Current focus: (empty)",
                "=======",
                "- Current focus: failure context",
                ">>>>>>> REPLACE",
            ]
        )
    )
    backend = _Backend(status="success")
    result = memory_patch_node(
        _state(outcome="failure"),
        model=model,
        memory_store=store,
        run_id="run_01",
        patch_repair_attempts=1,
        tool_backend=backend,
    )

    assert model.calls == 1
    assert len(backend.calls) == 1
    assert backend.calls[0][0] == "memory_apply_aider_edits"
    events = store.read_events_tail(limit=1)
    assert events
    assert events[-1]["memory_reconcile_requested"] is True
    assert events[-1]["memory_patch_status"] == "success"
    assert result["tasks"][0]["status"] == "failure"
