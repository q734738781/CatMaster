from __future__ import annotations

from functools import partial
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .writing_nodes import (
    assemble_manuscript_node,
    init_writing_node,
    plan_writing_node,
    review_section_node,
    summarize_writing_node,
    write_section_node,
)


class WritingState(TypedDict, total=False):
    request: dict[str, Any]
    board: Any
    plan: Any
    dossier: Any
    latest_draft: Any
    latest_review: Any
    section_drafts: list[Any]
    section_reviews: list[Any]
    resume_mode: bool
    resume_goto: str
    status: str
    summary: str
    final_answer: str


def build_writing_graph(
    *,
    writing_store,
    write_director_agent,
    section_writer_agent,
    write_reviewer_model,
    source_store,
    skills_runtime,
    writing_config,
):
    graph = StateGraph(WritingState)
    graph.add_node("init_writing", partial(init_writing_node, writing_store=writing_store, source_store=source_store))
    graph.add_node(
        "plan_writing",
        partial(
            plan_writing_node,
            writing_store=writing_store,
            source_store=source_store,
            write_director_agent=write_director_agent,
            skills_runtime=skills_runtime,
        ),
    )
    graph.add_node(
        "write_section",
        partial(
            write_section_node,
            writing_store=writing_store,
            source_store=source_store,
            section_writer_agent=section_writer_agent,
            skills_runtime=skills_runtime,
        ),
    )
    graph.add_node(
        "review_section",
        partial(
            review_section_node,
            writing_store=writing_store,
            write_reviewer_model=write_reviewer_model,
            skills_runtime=skills_runtime,
        ),
    )
    graph.add_node(
        "assemble_manuscript",
        partial(
            assemble_manuscript_node,
            writing_store=writing_store,
            source_store=source_store,
            writing_config=writing_config,
        ),
    )
    graph.add_node("summarize_writing", summarize_writing_node)
    graph.set_entry_point("init_writing")
    graph.add_edge("summarize_writing", END)
    return graph.compile()


__all__ = ["WritingState", "build_writing_graph"]
