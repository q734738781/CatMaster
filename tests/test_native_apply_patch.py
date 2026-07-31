from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai.chat_models.base import _construct_responses_api_input

from catmaster.runtime.apply_diff import apply_diff
from catmaster.runtime.native_apply_patch import (
    APPLY_PATCH_LARK_GRAMMAR,
    NativeApplyPatchError,
    NativeApplyPatchExecutor,
    build_native_apply_patch_tool,
    parse_apply_patch,
)
from catmaster.specialists.runtime import (
    build_specialist_runner,
    default_thread_interrupt_on,
)
from catmaster.specialists.streaming_runner import CatMasterStreamTranslator


class _Profile:
    def __init__(self, provider: str) -> None:
        self.provider = provider

    def config_for_role(self, role: str) -> SimpleNamespace:
        return SimpleNamespace(
            model=f"{role}-model",
            provider=self.provider,
            base_url=None,
        )


def _patch(*lines: str) -> str:
    return "\n".join((_BEGIN, *lines, _END, ""))


_BEGIN = "*** Begin Patch"
_END = "*** End Patch"


def test_apply_diff_preserves_existing_crlf_newlines() -> None:
    updated = apply_diff(
        "alpha\r\nbeta\r\n",
        "@@\n-alpha\n+gamma\n beta",
    )

    assert updated == "gamma\r\nbeta\r\n"


def test_parse_apply_patch_accepts_complete_multi_file_envelope() -> None:
    hunks = parse_apply_patch(
        _patch(
            "*** Add File: notes/new.txt",
            "+alpha",
            "+beta",
            "*** Update File: src/old.py",
            "*** Move to: src/new.py",
            "@@",
            "-old",
            "+new",
            "*** Delete File: obsolete.txt",
        )
    )

    assert [(hunk.kind, hunk.path, hunk.move_to) for hunk in hunks] == [
        ("add", "notes/new.txt", ""),
        ("update", "src/old.py", "src/new.py"),
        ("delete", "obsolete.txt", ""),
    ]
    assert hunks[0].body == "alpha\nbeta\n"


def test_executor_applies_add_update_move_and_delete(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "old.py").write_text("old\n", encoding="utf-8")
    (tmp_path / "obsolete.txt").write_text("gone\n", encoding="utf-8")
    executor = NativeApplyPatchExecutor(files_root=tmp_path)

    output = executor.execute(
        _patch(
            "*** Add File: notes/new.txt",
            "+alpha",
            "+beta",
            "*** Update File: src/old.py",
            "*** Move to: src/new.py",
            "@@",
            "-old",
            "+new",
            "*** Delete File: obsolete.txt",
        )
    )

    assert output == (
        "Done!\n"
        "A /notes/new.txt\n"
        "R /src/old.py -> /src/new.py\n"
        "D /obsolete.txt"
    )
    assert (tmp_path / "notes" / "new.txt").read_bytes() == b"alpha\nbeta\n"
    assert (tmp_path / "src" / "new.py").read_text(encoding="utf-8") == "new\n"
    assert not (tmp_path / "src" / "old.py").exists()
    assert not (tmp_path / "obsolete.txt").exists()


def test_executor_accepts_live_gpt56_move_only_hunk(tmp_path: Path) -> None:
    source = tmp_path / "base" / "b.txt"
    source.parent.mkdir()
    source.write_text("beta\n", encoding="utf-8")

    output = NativeApplyPatchExecutor(files_root=tmp_path).execute(
        _patch(
            "*** Update File: base/b.txt",
            "*** Move to: moved/b.txt",
        )
    )

    assert output == "Done!\nR /base/b.txt -> /moved/b.txt"
    assert not source.exists()
    assert (tmp_path / "moved" / "b.txt").read_text(encoding="utf-8") == "beta\n"


def test_add_file_matches_codex_overwrite_semantics(tmp_path: Path) -> None:
    target = tmp_path / "result.txt"
    target.write_text("old\n", encoding="utf-8")

    NativeApplyPatchExecutor(files_root=tmp_path).execute(
        _patch("*** Add File: result.txt", "+replacement")
    )

    assert target.read_text(encoding="utf-8") == "replacement\n"


def test_failed_update_is_model_visible_and_leaves_file_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "result.txt"
    target.write_text("current\n", encoding="utf-8")
    tool = build_native_apply_patch_tool(files_root=tmp_path)

    result = tool.invoke(
        {
            "__arg1": _patch(
                "*** Update File: result.txt",
                "@@",
                "-missing",
                "+replacement",
            )
        }
    )

    assert result[0]["type"] == "custom_tool_call_output"
    assert result[0]["output"].startswith("Error applying patch: Invalid Context")
    assert target.read_text(encoding="utf-8") == "current\n"


@pytest.mark.parametrize("path", ["../outside.txt", "/memories/AGENTS.md"])
def test_executor_rejects_non_workspace_routes(tmp_path: Path, path: str) -> None:
    executor = NativeApplyPatchExecutor(files_root=tmp_path)

    with pytest.raises(NativeApplyPatchError):
        executor.execute(_patch(f"*** Add File: {path}", "+blocked"))

    assert not (tmp_path.parent / "outside.txt").exists()
    assert not (tmp_path / "memories" / "AGENTS.md").exists()


def test_custom_tool_schema_matches_codex_freeform_protocol(tmp_path: Path) -> None:
    tool = build_native_apply_patch_tool(files_root=tmp_path)

    assert convert_to_openai_tool(tool) == {
        "type": "custom",
        "name": "apply_patch",
        "description": (
            "Edit workspace files with one Codex V4A patch. Pass the raw patch, not JSON."
        ),
        "format": {
            "type": "grammar",
            "syntax": "lark",
            "definition": APPLY_PATCH_LARK_GRAMMAR,
        },
    }


def test_langchain_replays_custom_patch_call_and_output() -> None:
    call_id = "patch-replay"
    patch = _patch("*** Delete File: old.txt")
    messages = [
        AIMessage(
            content=[
                {
                    "type": "custom_tool_call",
                    "name": "apply_patch",
                    "input": patch,
                    "call_id": call_id,
                    "id": "custom-patch-item",
                    "status": "completed",
                }
            ],
            tool_calls=[
                {
                    "name": "apply_patch",
                    "args": {"__arg1": patch},
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=[
                {
                    "type": "custom_tool_call_output",
                    "output": "Done!\nD /old.txt",
                }
            ],
            tool_call_id=call_id,
            name="apply_patch",
        ),
    ]

    assert _construct_responses_api_input(messages, store=False) == [
        {
            "type": "custom_tool_call",
            "name": "apply_patch",
            "input": patch,
            "call_id": call_id,
            "id": "custom-patch-item",
            "status": "completed",
        },
        {
            "type": "custom_tool_call_output",
            "call_id": call_id,
            "output": "Done!\nD /old.txt",
        },
    ]


def test_specialist_runtime_enables_patch_only_for_codex_oauth(tmp_path: Path) -> None:
    names_by_provider: dict[str, list[str]] = {}
    for provider in ("codex_oauth", "openai"):
        built = build_specialist_runner(
            workspace=tmp_path / provider,
            llm_profile=_Profile(provider),
            reporter=None,
            run_control=None,
            project_id=f"project-{provider}",
            preferred_entrypoint="experiment",
        )
        tools = built.runner._augment_with_default_autonomous_tools(
            [],
            model_role="task_runner",
        )
        names_by_provider[provider] = [
            str(tool.get("type") if isinstance(tool, dict) else tool.name)
            for tool in tools
        ]

    assert names_by_provider["codex_oauth"].count("apply_patch") == 1
    assert "apply_patch" not in names_by_provider["openai"]


def test_review_mode_default_only_interrupts_remote_submission() -> None:
    assert default_thread_interrupt_on() == {
        "remote_submission": True,
        "remote_submission_batch": True,
    }


def test_stream_projection_maps_custom_patch_call() -> None:
    translator = object.__new__(CatMasterStreamTranslator)
    translator.completed_tool_call_ids = set()
    translator.historical_completed_tool_call_ids = set()
    translator.tool_parts_by_call_id = {}
    started: list[dict[str, object]] = []
    translator._handle_tool_call_payload = (
        lambda payload, metadata=None: started.append(
            {"payload": payload, "metadata": metadata}
        )
    )

    patch = _patch(
        "*** Update File: notes/a.md",
        "@@",
        "-old",
        "+new",
    )
    translator._handle_provider_content_blocks(
        [
            {
                "type": "custom_tool_call",
                "name": "apply_patch",
                "input": patch,
                "call_id": "patch-ui",
            }
        ],
        metadata={"lc_agent_name": "experiment_specialist"},
    )

    assert started == [
        {
            "payload": {
                "id": "patch-ui",
                "name": "apply_patch",
                "args": {"__arg1": patch},
            },
            "metadata": {"lc_agent_name": "experiment_specialist"},
        }
    ]


def test_executor_serializes_high_frequency_calls_per_workspace(tmp_path: Path) -> None:
    executor = NativeApplyPatchExecutor(files_root=tmp_path)

    def _write(index: int) -> str:
        return executor.execute(
            _patch(f"*** Add File: batch/{index:03d}.txt", f"+value-{index}")
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        outputs = list(pool.map(_write, range(128)))

    assert len(outputs) == 128
    assert len(list((tmp_path / "batch").glob("*.txt"))) == 128
    for index in range(128):
        assert (tmp_path / "batch" / f"{index:03d}.txt").read_text(
            encoding="utf-8"
        ) == f"value-{index}\n"
