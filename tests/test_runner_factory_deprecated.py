from __future__ import annotations

from types import SimpleNamespace

import pytest

from catmaster.agents.graph import GraphRunner
import catmaster.agents.runner_factory as runner_factory_mod
from catmaster.agents.runner_factory import (
    LEGACY_GRAPH_DEPRECATION_NOTICE,
    LEGACY_GRAPH_DRIVER_KIND,
    build_graph_runner,
)


class _FakeProfile:
    main = SimpleNamespace(model="main-model", provider="langchain", base_url=None)

    class agent_runtime:
        max_tool_calls = 2
        recursion_limit = 3
        print_state_messages = False

    @staticmethod
    def config_for_role(role: str) -> SimpleNamespace:
        return SimpleNamespace(model=f"{role}-model")


def test_build_graph_runner_marks_legacy_driver_and_warns(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_factory_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    with pytest.deprecated_call(match="legacy and deprecated"):
        built = build_graph_runner(
            workspace=tmp_path,
            llm_profile=_FakeProfile(),
            reporter=None,
            run_control=None,
            project_id="proj",
        )

    assert built.run_context.driver_kind == LEGACY_GRAPH_DRIVER_KIND
    assert built.runner.is_deprecated is True
    assert built.runner.deprecation_notice == GraphRunner.DEPRECATION_NOTICE
    assert LEGACY_GRAPH_DEPRECATION_NOTICE.startswith("build_graph_runner()/GraphRunner is legacy")
