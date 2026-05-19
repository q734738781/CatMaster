from __future__ import annotations

from pathlib import Path

from catmaster.tools.base import ensure_project_space_layout
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


def test_different_ctx_keeps_workspace_selection_isolated(tmp_path: Path) -> None:
    (tmp_path / "alpha").mkdir(parents=True, exist_ok=True)
    (tmp_path / "beta").mkdir(parents=True, exist_ok=True)
    registry = SessionRegistry(default_project_space_root=tmp_path)

    alpha = registry.bootstrap(ctx="ctx_test_alpha", project_space="alpha", run="")
    beta = registry.bootstrap(ctx="ctx_test_beta", project_space="beta", run="")

    assert alpha.ctx == "ctx_test_alpha"
    assert beta.ctx == "ctx_test_beta"
    assert registry.get_session("ctx_test_alpha").current_workspace_path().endswith("alpha")
    assert registry.get_session("ctx_test_beta").current_workspace_path().endswith("beta")


def test_monitor_url_encodes_values(tmp_path: Path) -> None:
    registry = SessionRegistry(default_project_space_root=tmp_path)
    url = registry.monitor_url(ctx="ctx test", project_space="a/b", run="run 1")
    assert url == "/monitor/?ctx=ctx+test&project_space=a%2Fb&run=run+1"


def test_bootstrap_loads_root_project_space_when_root_itself_is_project(tmp_path: Path) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    registry = SessionRegistry(default_project_space_root=tmp_path)

    state = registry.bootstrap(ctx="ctx_test_root", project_space=tmp_path.name, run="")

    assert state.project_space_name == tmp_path.name
    assert state.project_space_path == str(tmp_path.resolve())


def test_bootstrap_missing_project_space_keeps_root_selection_empty(tmp_path: Path) -> None:
    registry = SessionRegistry(default_project_space_root=tmp_path)

    state = registry.bootstrap(ctx="ctx_test_stale", project_space="missing", run="")

    assert state.project_space_name == ""
    assert state.project_space_path == ""
    assert "Project space does not exist: missing" in state.status
