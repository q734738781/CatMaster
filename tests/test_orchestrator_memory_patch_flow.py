from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core.prompts")

sys.modules.setdefault("langchain_openai", types.SimpleNamespace(ChatOpenAI=object))

from catmaster.agents.orchestrator import MemoryPatchApplyError, Orchestrator
from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.trace_store import TraceStore


class _Prompt:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def format_messages(self, **kwargs):
        self.calls.append(dict(kwargs))
        return [
            SimpleNamespace(type="system", content="memory patch editor"),
            SimpleNamespace(type="human", content=json.dumps(kwargs, ensure_ascii=False)),
        ]


class _LLM:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)

    def invoke(self, _messages):
        if not self.outputs:
            raise AssertionError("no LLM output prepared")
        return SimpleNamespace(text=self.outputs.pop(0))


class _Backend:
    def __init__(self, outputs: list[dict]):
        self.outputs = list(outputs)
        self.calls: list[tuple[str, dict, str]] = []

    def call(self, name: str, arguments_json: str, *, toolcall_key: str, call_id: str | None = None) -> dict:
        self.calls.append((name, json.loads(arguments_json), toolcall_key))
        if not self.outputs:
            raise AssertionError("no backend output prepared")
        return dict(self.outputs.pop(0))


def _make_orchestrator(
    tmp_path: Path,
    *,
    llm_outputs: list[str],
    backend_outputs: list[dict],
    patch_repair_attempts: int = 1,
) -> tuple[Orchestrator, _Backend]:
    store = MemoryStore.create_default(workspace=tmp_path)
    store.ensure_exists()
    backend = _Backend(backend_outputs)
    orch = Orchestrator.__new__(Orchestrator)
    orch.memory_store = store
    orch.run_context = SimpleNamespace(
        workspace=tmp_path,
        run_id="run_01",
        run_dir=tmp_path / "metadata" / "runs" / "run_01",
    )
    orch.patch_repair_attempts = patch_repair_attempts
    orch.memory_patch_prompt = _Prompt()
    orch.memory_patch_repair_prompt = _Prompt()
    orch.llm = _LLM(llm_outputs)
    orch.tool_backend = backend
    orch.trace_store = TraceStore(orch.run_context.run_dir)
    orch._emit = lambda *args, **kwargs: None
    orch._write_llm_log = lambda *args, **kwargs: None
    return orch, backend


def _memory_edits(path: str = "MEMORY/MEMORY.md") -> str:
    return "\n".join([
        path,
        "<<<<<<< SEARCH",
        "# old",
        "=======",
        "# new",
        ">>>>>>> REPLACE",
        "",
    ])


def test_merge_memory_via_git_apply_success(tmp_path: Path) -> None:
    orch, backend = _make_orchestrator(
        tmp_path,
        llm_outputs=[_memory_edits()],
        backend_outputs=[{"status": "success", "data": {"diff_text": "diff --git a/MEMORY/MEMORY.md b/MEMORY/MEMORY.md\n"}}],
    )

    result = Orchestrator._merge_memory_via_git_apply(
        orch,
        run_id="run_01",
        task_id="task_01",
        outcome="success",
        task_goal_short="short goal",
        structured_result={
            "summary": "done",
            "facts": ["f1"],
            "files": [{"path": "outputs/a.txt", "description": "a", "kind": "output"}],
            "constraints": [],
            "open_questions": [],
            "decisions": [],
            "next_steps": [],
            "artifacts": [],
        },
    )

    assert result["event_path"] == "memory/events.jsonl"
    assert result["attempts"] == 1
    assert backend.calls[0][0] == "memory_apply_aider_edits"
    assert backend.calls[0][1]["allowed_paths"] == ["MEMORY/"]
    prompt_kwargs = orch.memory_patch_prompt.calls[0]
    assert "topic_goal_text" in prompt_kwargs
    assert "topic_facts_text" in prompt_kwargs
    assert "topic_files_text" in prompt_kwargs
    assert "topic_constraints_text" in prompt_kwargs
    assert "topic_questions_text" in prompt_kwargs
    assert "topic_runbook_text" in prompt_kwargs
    assert "topic_tldrs_json" not in prompt_kwargs
    assert "event_path" not in prompt_kwargs
    patch_path = tmp_path / "files" / result["patch_path"]
    assert patch_path.exists()


def test_merge_memory_via_git_apply_repair_after_rejected_patch(tmp_path: Path) -> None:
    orch, backend = _make_orchestrator(
        tmp_path,
        llm_outputs=[
            _memory_edits("src/not_allowed.py"),
            _memory_edits("MEMORY/topics/FACTS.md"),
        ],
        backend_outputs=[
            {"status": "failed", "error": "path validation failed", "data": {"error_detail": "forbidden path"}},
            {"status": "success", "data": {"diff_text": ""}},
        ],
        patch_repair_attempts=1,
    )

    result = Orchestrator._merge_memory_via_git_apply(
        orch,
        run_id="run_01",
        task_id="task_01",
        outcome="success",
        task_goal_short="short goal",
        structured_result={"summary": "done", "facts": [], "files": [], "constraints": [], "open_questions": [], "decisions": [], "next_steps": [], "artifacts": []},
    )

    assert result["attempts"] == 2
    assert len(backend.calls) == 2
    assert backend.calls[0][0] == "memory_apply_aider_edits"
    assert orch.memory_patch_repair_prompt.calls
    repair_kwargs = orch.memory_patch_repair_prompt.calls[0]
    assert "apply_error_context_json" in repair_kwargs
    assert "topic_goal_text" in repair_kwargs
    assert "topic_facts_text" in repair_kwargs
    assert "topic_files_text" in repair_kwargs
    assert "topic_constraints_text" in repair_kwargs
    assert "topic_questions_text" in repair_kwargs
    assert "topic_runbook_text" in repair_kwargs
    assert "topic_tldrs_json" not in repair_kwargs


def test_merge_memory_via_git_apply_passes_structured_error_to_repair_prompt(tmp_path: Path) -> None:
    orch, _ = _make_orchestrator(
        tmp_path,
        llm_outputs=[_memory_edits(), _memory_edits()],
        backend_outputs=[
            {
                "status": "failed",
                "error": "apply failed",
                "data": {
                    "error_code": "replace_no_match",
                    "error_detail": "search text missing",
                    "failed_path": "MEMORY/MEMORY.md",
                    "failed_block_index": 2,
                },
            },
            {"status": "success", "data": {"diff_text": ""}},
        ],
        patch_repair_attempts=1,
    )

    result = Orchestrator._merge_memory_via_git_apply(
        orch,
        run_id="run_01",
        task_id="task_01",
        outcome="success",
        task_goal_short="short goal",
        structured_result={"summary": "done", "facts": [], "files": [], "constraints": [], "open_questions": [], "decisions": [], "next_steps": [], "artifacts": []},
    )

    assert result["attempts"] == 2
    repair_kwargs = orch.memory_patch_repair_prompt.calls[0]
    error_ctx = json.loads(repair_kwargs["apply_error_context_json"])
    assert error_ctx["error_code"] == "replace_no_match"
    assert error_ctx["failed_path"] == "MEMORY/MEMORY.md"
    assert error_ctx["failed_block_index"] == 2


def test_merge_memory_via_git_apply_raises_after_retries(tmp_path: Path) -> None:
    orch, _ = _make_orchestrator(
        tmp_path,
        llm_outputs=[_memory_edits(), _memory_edits()],
        backend_outputs=[
            {"status": "failed", "error": "apply failed", "data": {"error_detail": "bad replace"}},
            {"status": "failed", "error": "apply failed", "data": {"error_detail": "still bad"}},
        ],
        patch_repair_attempts=1,
    )

    with pytest.raises(MemoryPatchApplyError) as excinfo:
        Orchestrator._merge_memory_via_git_apply(
            orch,
            run_id="run_01",
            task_id="task_01",
            outcome="success",
            task_goal_short="short goal",
            structured_result={"summary": "done", "facts": [], "files": [], "constraints": [], "open_questions": [], "decisions": [], "next_steps": [], "artifacts": []},
        )

    exc = excinfo.value
    assert exc.event_path == "memory/events.jsonl"
    assert ".logs/memory_patches/" in exc.patch_path
    assert "apply failed" in str(exc)


def test_write_latest_run_readme(tmp_path: Path) -> None:
    orch = Orchestrator.__new__(Orchestrator)
    latest_run = tmp_path / "files" / "reports" / "latest_run"

    Orchestrator._write_latest_run_readme(orch, latest_run)

    readme = latest_run / "README.md"
    assert readme.exists()
    text = readme.read_text(encoding="utf-8")
    assert "audit/debug snapshot" in text
    assert "not canonical memory" in text
