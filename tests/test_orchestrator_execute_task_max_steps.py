from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core.prompts")

sys.modules.setdefault("langchain_openai", types.SimpleNamespace(ChatOpenAI=object))

from catmaster.agents.orchestrator import Orchestrator
from catmaster.tools.base import ensure_project_space_layout


class _FakeStepper:
    def __init__(self, **_kwargs):
        pass

    def run(self, **_kwargs):
        return {
            "finish_reason": "max_steps",
            "control_payload": None,
            "output_text": "",
            "local_observations": [
                {
                    "step": 0,
                    "method": "vasp_execute_batch",
                    "params": {
                        "input_root_rel": "jobs/inputs",
                        "output_root_rel": "jobs/outputs",
                        "cases": ["CO", "O2", "H2"],
                    },
                    "result": {
                        "status": "success",
                        "tool_name": "vasp_execute_batch",
                        "warnings": [],
                        "error": None,
                        "data": {
                            "batch_state_rel": "jobs/outputs/_BATCH_STATE.json",
                            "input_root_rel": "jobs/inputs",
                            "output_root_rel": "jobs/outputs",
                            "outputs": [
                                {
                                    "input_dir_rel": "jobs/inputs/CO",
                                    "output_dir_rel": "jobs/outputs/CO",
                                    "output_files": ["*"],
                                },
                                {
                                    "input_dir_rel": "jobs/inputs/O2",
                                    "output_dir_rel": "jobs/outputs/O2",
                                    "output_files": ["*"],
                                },
                            ],
                            "submission_dir": "jobs/outputs/vasp_batch_001",
                            "task_states": ["5", "5"],
                            "work_base": "vasp_batch_001",
                        },
                    },
                }
            ],
        }


def test_execute_task_max_steps_auto_replan_and_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ensure_project_space_layout(tmp_path, create=True)

    orch = Orchestrator.__new__(Orchestrator)
    orch.context_builder = SimpleNamespace(build=lambda *_args, **_kwargs: {"constraints": "", "memory_index_excerpt": "", "artifact_slice": []})
    orch._interrupt_context_note = ""
    orch._emit = lambda *_args, **_kwargs: None
    orch._filtered_function_tools = lambda: []
    orch._supports_builtin_tools_for = lambda _role: False
    orch.tool_policy = SimpleNamespace(max_tool_calls_per_task=60, parallel_tool_calls=False, builtin_tools=[])
    orch.max_steps = 200
    orch._role_tool_driver = lambda _role: object()
    orch.task_step_prompt = None
    orch.reporter = SimpleNamespace()
    orch.tool_backend = SimpleNamespace()
    orch._tool_driver_kwargs = lambda _role: {}
    orch.trace_store = SimpleNamespace()
    orch.checkpoint_store = SimpleNamespace()
    orch.run_context = SimpleNamespace(
        workspace=tmp_path,
        run_id="run_01",
        run_dir=tmp_path / "metadata" / "runs" / "run_01",
    )
    orch._interrupt_requested = lambda *_args, **_kwargs: False
    orch._ack_interrupt = lambda *_args, **_kwargs: None
    orch.memory_store = SimpleNamespace(index_path=tmp_path / "files" / "memory" / "index.md")

    captured: dict[str, object] = {}

    def _fake_merge_memory_via_git_apply(*, structured_result, **_kwargs):
        captured["structured_result"] = json.loads(json.dumps(structured_result))
        return {
            "event_path": "memory/events.jsonl",
            "memory_index": "memory/index.md",
            "patch_path": "audit/memory_patches/task_01.patch",
            "attempts": 1,
        }

    orch._merge_memory_via_git_apply = _fake_merge_memory_via_git_apply
    orch._write_observation = lambda **_kwargs: "observations/obs_task_01.md"

    monkeypatch.setattr("catmaster.agents.orchestrator.ToolCallingTaskStepper", _FakeStepper)

    result = Orchestrator._execute_task(
        orch,
        task_id="task_01",
        task_goal="Long-running task",
        task_goal_short="Long task",
        log_llm=False,
        resume_state=None,
    )

    assert result["outcome"] == "failure"
    assert result["auto_replan"] is True
    assert result["failure_kind"] == "max_steps"

    structured = captured["structured_result"]
    assert isinstance(structured, dict)
    assert structured["toolcall_context_count"] == 1
    assert isinstance(structured["toolcall_context"], list)
    compact_row = structured["toolcall_context"][0]
    assert compact_row["method"] == "vasp_execute_batch"
    assert "params_compact" in compact_row
    compact_data = compact_row["result"]["data"]
    assert "outputs" in compact_data["data_keys"]
    assert "outputs" not in compact_data.get("scalars", {})
    assert compact_data["paths"]["submission_dir"] == "jobs/outputs/vasp_batch_001"

    assert "toolcall_context_path" not in structured
    assert not result["key_artifacts"]
    audit_hits = sorted((tmp_path / "metadata" / "runs" / "run_01" / "audit" / "toolcall_context").glob("*.json"))
    assert audit_hits
