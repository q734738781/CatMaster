from __future__ import annotations

from pathlib import Path

from catmaster.webui.session_registry import SessionRegistry


def test_bootstrap_loads_workspace_from_query(tmp_path: Path) -> None:
    (tmp_path / "alpha").mkdir(parents=True, exist_ok=True)
    registry = SessionRegistry(default_project_space_root=tmp_path)

    state = registry.bootstrap(ctx="ctx_test_001", project_space="alpha", run="")

    assert state.ctx == "ctx_test_001"
    assert state.project_space_name == "alpha"
    assert state.project_space_path.endswith("alpha")


def test_same_ctx_reuses_same_session(tmp_path: Path) -> None:
    registry = SessionRegistry(default_project_space_root=tmp_path)
    a = registry.get_session("ctx_test_abc")
    b = registry.get_session("ctx_test_abc")
    assert a is b


def test_monitor_url_encodes_values(tmp_path: Path) -> None:
    registry = SessionRegistry(default_project_space_root=tmp_path)
    url = registry.monitor_url(ctx="ctx test", project_space="a/b", run="run 1")
    assert url == "/monitor/?ctx=ctx+test&project_space=a%2Fb&run=run+1"
