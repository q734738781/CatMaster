from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult, TextContent

from catmaster.runtime.literature.browser_mcp import (
    AGENT_BROWSER_TOOL_ALLOWLIST,
    BrowserToolGuard,
    _browser_session_name,
    _normalize_tab_label,
    _sanitize_tool,
    open_literature_browser_tools,
)


class _Request:
    def __init__(self, name: str, args: dict) -> None:
        self.name = name
        self.args = args

    def override(self, *, args: dict):
        return _Request(self.name, args)


def test_browser_schema_hides_host_control_fields() -> None:
    tool = SimpleNamespace(
        name="agent_browser_read",
        args_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "extraArgs": {"type": "array"},
                "session": {"type": "string"},
                "namespace": {"type": "string"},
                "restore": {"oneOf": [{"type": "boolean"}, {"type": "string"}]},
                "llms": {"enum": ["index", "full"]},
                "raw": {"type": "boolean"},
                "requireMd": {"type": "boolean"},
            },
        },
        description="Open a page.",
    )

    sanitized = _sanitize_tool(tool)

    assert set(sanitized.args_schema["properties"]) == {"url"}
    assert "untrusted evidence" in sanitized.description


def test_browser_session_name_stays_below_unix_socket_limit(tmp_path: Path) -> None:
    long_root = tmp_path / ("workspace-" + "x" * 180)
    session_name = _browser_session_name(files_root=long_root, run_id="run-" + "y" * 180)

    assert session_name.startswith("cm-")
    assert len(session_name) == 19


def test_browser_tab_label_is_normalized_for_cli_contract() -> None:
    assert _normalize_tab_label("Frontiers TMP HER") == "Frontiers-TMP-HER"
    assert _normalize_tab_label("123 results") == "tab-123-results"


def test_browser_guard_injects_session_contains_path_and_truncates(tmp_path: Path) -> None:
    guard = BrowserToolGuard(
        files_root=tmp_path,
        session_name="fixed-session",
        namespace="catmaster",
        max_output_chars=100,
    )
    observed = {}

    async def _handler(request):
        observed.update(request.args)
        return CallToolResult(content=[TextContent(type="text", text="x" * 500)])

    result = asyncio.run(
        guard(
            _Request("agent_browser_download", {"selector": "@e1", "path": "literature/paper.pdf"}),
            _handler,
        )
    )

    assert observed["session"] == "fixed-session"
    assert observed["namespace"] == "catmaster"
    assert Path(observed["path"]).is_relative_to(tmp_path.resolve())
    assert "truncated browser output" in result.content[0].text

    with pytest.raises(ValueError, match="inside the active workspace"):
        asyncio.run(
            guard(
                _Request("agent_browser_download", {"selector": "@e1", "path": "../escape.pdf"}),
                _handler,
            )
        )


def test_browser_guard_rejects_hidden_cli_controls(tmp_path: Path) -> None:
    guard = BrowserToolGuard(
        files_root=tmp_path,
        session_name="fixed-session",
        namespace="catmaster",
        max_output_chars=1000,
    )

    async def _handler(request):
        return request

    with pytest.raises(ValueError, match="host-control"):
        asyncio.run(
            guard(
                _Request("agent_browser_open", {"url": "https://example.com", "extraArgs": ["--eval"]}),
                _handler,
            )
        )


def test_browser_guard_serializes_shared_session(tmp_path: Path) -> None:
    guard = BrowserToolGuard(
        files_root=tmp_path,
        session_name="fixed-session",
        namespace="catmaster",
        max_output_chars=1000,
    )
    active = 0
    maximum = 0

    async def _handler(request):
        nonlocal active, maximum
        _ = request
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return CallToolResult(content=[TextContent(type="text", text="ok")])

    async def _run():
        await asyncio.gather(
            guard(_Request("agent_browser_get_url", {}), _handler),
            guard(_Request("agent_browser_get_title", {}), _handler),
        )

    asyncio.run(_run())
    assert maximum == 1


def test_browser_missing_executable_has_actionable_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("catmaster.runtime.literature.browser_mcp.shutil.which", lambda _: None)

    async def _run():
        async with open_literature_browser_tools(
            files_root=tmp_path,
            workspace_name="test",
            run_id="run",
        ):
            pass

    with pytest.raises(RuntimeError, match="npm install -g agent-browser@0.31.1"):
        asyncio.run(_run())


def test_browser_allowlist_excludes_sensitive_tool_classes() -> None:
    names = " ".join(sorted(AGENT_BROWSER_TOOL_ALLOWLIST))
    for forbidden in ("cookie", "storage", "auth", "eval", "network", "plugin", "clipboard", "debug"):
        assert forbidden not in names
