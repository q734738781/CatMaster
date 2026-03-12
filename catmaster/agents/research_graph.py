from __future__ import annotations

from functools import partial
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .research_nodes import (
    build_dossier_node,
    execute_experiment_node,
    execute_literature_node,
    execute_writer_handoff_node,
    finalize_ask_human_node,
    initialize_research_state_node,
    init_campaign_node,
    persist_conclusion_node,
    plan_research_node,
    sync_research_state_node,
    summarize_research_node,
)


class ResearchState(TypedDict, total=False):
    request: dict
    board: Any
    lead_action: dict
    planner_context: Any
    latest_state_sync: Any
    latest_literature: Any
    latest_experiment: Any
    literature_packs: list[Any]
    experiment_packs: list[Any]
    conclusion: Any
    dossier: Any
    sync_reason: str
    resume_mode: bool
    resume_goto: str
    status: str
    summary: str
    final_answer: str


def build_research_graph(
    *,
    store,
    planner_agent,
    state_updater_agent,
    memory_store,
    literature_runner,
    experiment_runner,
    writer_runner=None,
    skills_runtime,
    progress_callback=None,
):
    graph = StateGraph(ResearchState)
    graph.add_node("init_campaign", partial(init_campaign_node, store=store, progress_callback=progress_callback))
    graph.add_node(
        "initialize_research_state",
        partial(
            initialize_research_state_node,
            store=store,
            state_updater_agent=state_updater_agent,
            memory_store=memory_store,
            skills_runtime=skills_runtime,
            progress_callback=progress_callback,
        ),
    )
    graph.add_node(
        "plan_research",
        partial(
            plan_research_node,
            store=store,
            planner_agent=planner_agent,
            memory_store=memory_store,
            skills_runtime=skills_runtime,
            progress_callback=progress_callback,
        ),
    )
    graph.add_node(
        "sync_research_state",
        partial(
            sync_research_state_node,
            store=store,
            state_updater_agent=state_updater_agent,
            memory_store=memory_store,
            skills_runtime=skills_runtime,
            progress_callback=progress_callback,
        ),
    )
    graph.add_node("RunLiterature", partial(execute_literature_node, store=store, literature_runner=literature_runner, progress_callback=progress_callback))
    graph.add_node("RunExperiment", partial(execute_experiment_node, store=store, experiment_runner=experiment_runner, progress_callback=progress_callback))
    graph.add_node(
        "RunWriter",
        partial(
            execute_writer_handoff_node,
            store=store,
            writer_runner=writer_runner,
            progress_callback=progress_callback,
        ),
    )
    graph.add_node("AskHuman", partial(finalize_ask_human_node, store=store, progress_callback=progress_callback))
    graph.add_node("Conclude", partial(persist_conclusion_node, store=store, progress_callback=progress_callback))
    graph.add_node("build_dossier", partial(build_dossier_node, store=store, progress_callback=progress_callback))
    graph.add_node("summarize_research", summarize_research_node)
    graph.set_entry_point("init_campaign")
    graph.add_edge("summarize_research", END)
    graph.add_edge("AskHuman", "summarize_research")
    return graph.compile()


__all__ = ["ResearchState", "build_research_graph"]
