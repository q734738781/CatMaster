from __future__ import annotations

from pathlib import Path

from catmaster.agents.nodes import (
    _build_director_context,
    _build_fast_director_context,
    _build_proposal_context,
    _build_task_context,
)
from catmaster.runtime.memory_store import MemoryStore
from catmaster.tools.base import workspace_scope


def test_context_builders_include_chat_session_context_section(tmp_path: Path) -> None:
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
            "session_context_text": "<chat_session_context>\nUser: compare bridge and ontop.\n</chat_session_context>",
        }
        tools = "- place_adsorbate\n- generate_batch_adsorption_structures"

        proposal_ctx = _build_proposal_context(state, store, tools)
        director_ctx = _build_director_context(state, store, tools)
        fast_ctx = _build_fast_director_context(state, store, tools)

        marker = "Relevant chat session context"
        assert marker in proposal_ctx
        assert marker in director_ctx
        assert marker in fast_ctx
        assert "compare bridge and ontop" in proposal_ctx


def test_goal_still_flows_via_state_not_memory_topic(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        store = MemoryStore.create_default(workspace=tmp_path)
        store.ensure_exists()
        state = {
            "user_request": "Find the most stable adsorption geometry",
            "proposal_md": "proposal body",
            "work_packages": ["wp1"],
            "tasks": [],
            "observations": [],
            "current_task_packet": {"goal": "Compare bridge vs ontop on the bounded slab"},
        }
        tools = "- place_adsorbate"

        director_ctx = _build_director_context(state, store, tools)
        task_ctx = _build_task_context(state, store)

        assert "Find the most stable adsorption geometry" in director_ctx
        assert "Task goal:\nCompare bridge vs ontop on the bounded slab" in task_ctx
