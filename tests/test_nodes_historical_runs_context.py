from __future__ import annotations

from pathlib import Path

from catmaster.agents.nodes import (
    _build_director_context,
    _build_fast_director_context,
    _build_proposal_context,
)
from catmaster.runtime.memory_store import MemoryStore
from catmaster.tools.base import workspace_scope


def test_context_builders_include_historical_runs_section(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        store = MemoryStore.create_default(workspace=tmp_path)
        store.ensure_exists()
        state = {
            "user_request": "Find best adsorption site",
            "proposal_feedback": "",
            "proposal_review_enabled": True,
            "proposal_md": "proposal body",
            "work_packages": ["wp1"],
            "tasks": [],
            "observations": [],
            "historical_runs_context_text": "historical evidence summary",
        }
        tools = "- place_adsorbate\n- generate_batch_adsorption_structures"

        proposal_ctx = _build_proposal_context(state, store, tools)
        director_ctx = _build_director_context(state, store, tools)
        fast_ctx = _build_fast_director_context(state, store, tools)

        marker = "Relevant historical runs (auto-retrieved)"
        assert marker in proposal_ctx
        assert marker in director_ctx
        assert marker in fast_ctx
        assert "historical evidence summary" in proposal_ctx
