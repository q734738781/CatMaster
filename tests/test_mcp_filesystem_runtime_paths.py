from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")

from catmaster.llm.config import MCPFilesystemConfig
from catmaster.runtime.mcp_filesystem import MCPFilesystemRuntime
from catmaster.runtime.run_context import RunContext
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError


def _runtime(tmp_path) -> MCPFilesystemRuntime:
    run_ctx = RunContext.create(workspace=tmp_path, model_name="dummy-model")
    cfg = MCPFilesystemConfig.from_dict(
        {
            "enabled": True,
            "transport": "stdio",
            "mode": "stateful",
            "server_name": "filesystem",
            "command": "npx",
            "args_prefix": ["-y", "@modelcontextprotocol/server-filesystem"],
        }
    )
    return MCPFilesystemRuntime(config=cfg, run_context=run_ctx)


def test_rewrite_request_args_converts_relative_to_abs(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    args = runtime._rewrite_request_args(
        tool_name="read_text_file",
        args={"path": "README.md"},
    )
    resolved = Path(str(args["path"]))
    assert resolved.is_absolute()
    assert str(resolved).startswith(str(runtime.files_root))


def test_rewrite_request_args_allows_absolute_inside_workspace(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    abs_path = runtime.files_root / "CALCS" / "a" / "OUTCAR"
    args = runtime._rewrite_request_args(
        tool_name="read_text_file",
        args={"path": str(abs_path)},
    )
    assert args["path"] == str(abs_path.resolve())


def test_rewrite_request_args_rejects_absolute_outside_workspace(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    outside = (runtime.files_root.parent / "outside.txt").resolve()
    with pytest.raises(CatMasterToolExecutionError):
        runtime._rewrite_request_args(
            tool_name="read_text_file",
            args={"path": str(outside)},
        )


def test_rewrite_request_args_rejects_escape(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    with pytest.raises(CatMasterToolExecutionError):
        runtime._rewrite_request_args(
            tool_name="read_text_file",
            args={"path": "../metadata/secret.txt"},
        )


def test_relativize_value_rewrites_files_root_paths(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    absolute_file = runtime.files_root / "CALCS" / "a" / "OUTCAR"
    payload = {
        "path": str(absolute_file),
        "paths": [str(absolute_file.parent)],
    }
    rel = runtime._relativize_value(payload)
    assert rel["path"] == "CALCS/a/OUTCAR"
    assert rel["paths"][0] == "CALCS/a"


def test_render_capability_guide_contains_abs_reference_hint(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    guide = runtime.render_capability_guide(mode="full")
    assert "reference absolute files root (orientation only)" in guide
    if runtime.skill_mounts:
        assert "skills are mounted read-only under `@skills`" in guide
    else:
        assert "skills mounts are unavailable in this invocation" in guide
    assert "prefer relative paths in filesystem tool arguments" in guide
    assert "if absolute paths are used, they must stay under project files root" in guide


def test_render_capability_guide_reports_missing_skills_mount(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    runtime.skill_mounts = {}
    runtime.skills_root = None
    guide = runtime.render_capability_guide(mode="full")
    assert "skills mounts are unavailable in this invocation" in guide
    assert "skills are mounted read-only under `@skills`" not in guide


def test_build_connection_injects_local_npm_cache_for_stdio(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    connection = runtime._build_connection()
    assert connection["transport"] == "stdio"
    env = connection.get("env") or {}
    cache = str(env.get("NPM_CONFIG_CACHE") or "")
    assert cache
    assert cache.startswith(str((tmp_path / "metadata").resolve()))
    assert env.get("npm_config_cache") == cache


def test_rewrite_request_args_supports_skills_token_for_read(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    if runtime.skills_root is None:
        pytest.skip("skills root unavailable in test environment")
    args = runtime._rewrite_request_args(
        tool_name="read_text_file",
        args={"path": "@skills/slab-construction-and-surface-modeling/SKILL.md"},
    )
    resolved = Path(str(args["path"]))
    assert resolved.is_absolute()
    assert str(resolved).startswith(str(runtime.skills_root))


def test_rewrite_request_args_rejects_write_under_skills_mount(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    if runtime.skills_root is None:
        pytest.skip("skills root unavailable in test environment")
    with pytest.raises(CatMasterToolExecutionError):
        runtime._rewrite_request_args(
            tool_name="write_file",
            args={"path": "@skills/slab-construction-and-surface-modeling/SKILL.md"},
        )


class _FakeCallToolResult:
    def __init__(self, *, content, structuredContent=None):
        self.content = content
        self.structuredContent = structuredContent

    def model_copy(self, *, update):
        content = update.get("content", self.content)
        structured = update.get("structuredContent", self.structuredContent)
        return _FakeCallToolResult(content=content, structuredContent=structured)


def test_rewrite_call_tool_result_rewrites_mapping_text_block_and_structured(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    absolute_file = runtime.files_root / "CALCS" / "a" / "OUTCAR"
    req = SimpleNamespace(name="read_text_file", args={"path": "CALCS/a/OUTCAR"})
    result = _FakeCallToolResult(
        content=[{"type": "text", "text": f"found: {absolute_file.as_posix()}"}],
        structuredContent={"path": str(absolute_file)},
    )
    rewritten = runtime._rewrite_call_tool_result(request=req, result=result)
    assert isinstance(rewritten.content, list)
    assert isinstance(rewritten.content[0], dict)
    assert rewritten.content[0]["text"] == "found: ./CALCS/a/OUTCAR"
    assert rewritten.structuredContent == {"path": "CALCS/a/OUTCAR"}


def test_rewrite_call_tool_result_hides_list_allowed_directories(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    req = SimpleNamespace(name="list_allowed_directories", args={})
    result = _FakeCallToolResult(
        content=[{"type": "text", "text": f"allowed: {runtime.files_root.as_posix()}"}],
        structuredContent={"allowed_directories": [runtime.files_root.as_posix()]},
    )
    rewritten = runtime._rewrite_call_tool_result(request=req, result=result)
    expected_allowed = ["."]
    expected_allowed.extend(sorted(runtime.skill_mounts))
    assert rewritten.structuredContent == {"allowed_directories": expected_allowed}
    assert isinstance(rewritten.content, list)
    assert rewritten.content
    first = rewritten.content[0]
    if isinstance(first, dict):
        text = str(first.get("text", ""))
    else:
        text = str(getattr(first, "text", ""))
    assert text == f"Allowed roots: {', '.join(expected_allowed)}"
