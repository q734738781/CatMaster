from __future__ import annotations

from pathlib import Path

import pytest

from catmaster.agents.runner_factory import (
    LEGACY_GRAPH_DEPRECATION_NOTICE,
    build_graph_runner,
)


def test_build_graph_runner_is_removed_entrypoint(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="legacy graph entrypoint has been removed"):
        build_graph_runner(
            workspace=tmp_path,
            llm_profile=object(),
            reporter=None,
            run_control=None,
            project_id="proj",
        )
    assert LEGACY_GRAPH_DEPRECATION_NOTICE.startswith("build_graph_runner() is legacy")
