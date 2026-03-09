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
    init_campaign_node,
    persist_conclusion_node,
    plan_research_node,
    summarize_research_node,
)


class ResearchState(TypedDict, total=False):
    request: dict
    board: Any
    lead_action: dict
    planner_context: Any
    latest_literature: Any
    latest_experiment: Any
    literature_packs: list[Any]
    experiment_packs: list[Any]
    conclusion: Any
    dossier: Any
    history_context_summary: str
    context_review: Any
    resume_mode: bool
    resume_goto: str
    status: str
    summary: str
    final_answer: str


def build_research_graph(
    *,
    store,
    planner_model,
    memory_store,
    literature_runner,
    experiment_runner,
    history_reader,
    project_id,
    skills_runtime,
):
    graph = StateGraph(ResearchState)
    graph.add_node("init_campaign", partial(init_campaign_node, store=store))
    graph.add_node(
        "plan_research",
        partial(
            plan_research_node,
            store=store,
            planner_model=planner_model,
            memory_store=memory_store,
            history_reader=history_reader,
            project_id=project_id,
            skills_runtime=skills_runtime,
        ),
    )
    graph.add_node("RunLiterature", partial(execute_literature_node, store=store, literature_runner=literature_runner))
    graph.add_node("RunExperiment", partial(execute_experiment_node, store=store, experiment_runner=experiment_runner))
    graph.add_node("RunWriter", partial(execute_writer_handoff_node, store=store))
    graph.add_node("AskHuman", partial(finalize_ask_human_node, store=store))
    graph.add_node("Conclude", partial(persist_conclusion_node, store=store))
    graph.add_node("build_dossier", partial(build_dossier_node, store=store))
    graph.add_node("summarize_research", summarize_research_node)
    graph.set_entry_point("init_campaign")
    graph.add_edge("summarize_research", END)
    graph.add_edge("AskHuman", "summarize_research")
    return graph.compile()


__all__ = ["ResearchState", "build_research_graph"]
