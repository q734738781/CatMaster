from __future__ import annotations

from pathlib import Path

import pytest

from catmaster.agents.writing_nodes import (
    assemble_manuscript_node,
    init_writing_node,
    plan_writing_node,
    review_section_node,
    write_section_node,
)
from catmaster.agents.writing_prompts import build_section_writer_context
from catmaster.agents.writing_schemas import WritingRequest
from catmaster.runtime.research import ResearchBoard, ResearchDossier, ResearchStore
from catmaster.runtime.writing import SectionDraftModel, SectionReviewModel, WritingPlanModel, WritingSectionSpec, WritingStore
from catmaster.llm.config import WritingRuntimeConfig


class _DummyAgent:
    def __init__(self, payload):
        self.payload = payload

    async def ainvoke(self, payload):
        _ = payload
        return {"structured_response": self.payload}


class _DummyReviewerModel:
    def __init__(self, payload):
        self.payload = payload

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, payload):
        _ = payload
        return self.payload


@pytest.mark.anyio
async def test_writing_nodes_pipeline_builds_manuscript_and_updates_source_board(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_polish(payload):
        path = tmp_path / "files" / str(payload["source_path"])
        text = path.read_text(encoding="utf-8")
        path.write_text(text + "\n% polished\n", encoding="utf-8")
        return "polished", {"data": {"model_name": "fake-polisher"}}

    monkeypatch.setattr("catmaster.agents.writing_nodes.polish_academic_prose", _fake_polish)
    source_store = ResearchStore(workspace=tmp_path, campaign_id="camp_001")
    source_store.save_board(
        ResearchBoard(
            campaign_id="camp_001",
            question="What controls CO adsorption on Fe(110)?",
            exploration_policy="anchored",
            status="done",
            max_cycles=4,
            max_literature_queries=2,
            max_fast_runs=2,
            max_standard_runs=1,
        )
    )
    source_store.persist_dossier(
        ResearchDossier(
            campaign_id="camp_001",
            question="What controls CO adsorption on Fe(110)?",
            exploration_policy="anchored",
            final_answer_md="Bridge adsorption remains the leading hypothesis.",
            confidence="medium",
        )
    )
    store = WritingStore(workspace=tmp_path, run_id="write_001")
    request = WritingRequest(
        request="Write a short evidence-backed section from existing Fe adsorption results.",
        source_campaign_id="camp_001",
    )
    init_cmd = await init_writing_node({"request": request.model_dump()}, writing_store=store, source_store=source_store)
    assert init_cmd.goto == "plan_writing"

    plan_cmd = await plan_writing_node(
        {
            "request": request.model_dump(),
            "board": init_cmd.update["board"],
            "dossier": init_cmd.update["dossier"],
        },
        writing_store=store,
        source_store=source_store,
        write_director_agent=_DummyAgent(
            WritingPlanModel(
                title="CO adsorption on Fe(110)",
                writing_mode="section_draft",
                target_audience="internal",
                section_specs=[
                    WritingSectionSpec(
                        section_id="results_discussion",
                        heading="Results and Discussion",
                        purpose="Summarize the main evidence chain.",
                    )
                ],
            )
        ),
        skills_runtime=None,
    )
    assert plan_cmd.goto == "write_section"

    draft_cmd = await write_section_node(
        {
            "request": request.model_dump(),
            "board": plan_cmd.update["board"],
            "plan": plan_cmd.update["plan"],
            "dossier": plan_cmd.update["dossier"],
        },
        writing_store=store,
        source_store=source_store,
        section_writer_agent=_DummyAgent(
            {
                "section_id": "results_discussion",
                "heading": "Results and Discussion",
                "status": "drafted",
                "section_tex": "\\section{Results and Discussion}\nBridge adsorption remained preferred in the bounded calculations.",
                "citations": ["Fe(110) study (2024)"],
            }
        ),
        skills_runtime=None,
    )
    assert draft_cmd.goto == "review_section"

    review_cmd = await review_section_node(
        {
            "request": request.model_dump(),
            "board": draft_cmd.update["board"],
            "plan": draft_cmd.update["plan"],
            "latest_draft": draft_cmd.update["latest_draft"],
        },
        writing_store=store,
        write_reviewer_model=_DummyReviewerModel(
            {
                "section_id": "results_discussion",
                "status": "approved",
                "revision_notes": [],
                "unsupported_claims": [],
                "missing_citations": [],
            }
        ),
        skills_runtime=None,
    )
    assert review_cmd.goto == "write_section"

    advance_cmd = await write_section_node(
        {
            "request": request.model_dump(),
            "board": review_cmd.update["board"],
            "plan": review_cmd.update["plan"],
            "dossier": source_store.load_dossier(),
        },
        writing_store=store,
        source_store=source_store,
        section_writer_agent=_DummyAgent({}),
        skills_runtime=None,
    )
    assert advance_cmd.goto == "assemble_manuscript"

    result = await assemble_manuscript_node(
        {
            "request": request.model_dump(),
            "board": advance_cmd.update["board"],
            "plan": advance_cmd.update["plan"],
        },
        writing_store=store,
        source_store=source_store,
        writing_config=WritingRuntimeConfig(author_name="CatMaster"),
    )
    assert result["status"] == "done"
    latex_manuscript = tmp_path / "files" / "writing" / "write_001" / "manuscript" / "MANUSCRIPT.tex"
    assert latex_manuscript.exists()
    latex_text = latex_manuscript.read_text(encoding="utf-8")
    assert "\\documentclass[journal=jacsat,manuscript=article]{achemso}" in latex_text
    assert "\\author{CatMaster}" in latex_text
    assert "\\bibliography{references}" in latex_text
    assert "% polished" in latex_text
    updated_source_board = source_store.load_board()
    assert updated_source_board is not None
    assert updated_source_board.latest_writer_ref == "writing/write_001/manuscript/MANUSCRIPT.tex"
    assert updated_source_board.action_refs[-1].kind == "writer"


def test_section_writer_context_carries_plan_level_tex_signal() -> None:
    plan = WritingPlanModel(
        title="CO adsorption on Fe(110)",
        writing_mode="section_draft",
        target_audience="internal",
        preferred_output_format="tex",
        section_specs=[
            WritingSectionSpec(
                section_id="results_discussion",
                heading="Results and Discussion",
                purpose="Summarize the main evidence chain.",
            )
        ],
    )
    spec = plan.section_specs[0]
    context = build_section_writer_context(
        request={
            "request": "Write a short evidence-backed section from existing Fe adsorption results.",
            "writing_mode": "section_draft",
            "source_campaign_id": "camp_001",
        },
        plan=plan,
        spec=spec,
        dossier={"question": "What controls CO adsorption on Fe(110)?"},
        memory_index_excerpt="# MEMORY (AUTOLOADED INDEX)\n\n## Top Constraints\n1. Keep claims evidence-backed.",
        prior_draft=None,
        review_notes=[],
        skill_guide="template-aware skill available",
    )
    assert "Preferred output format: tex" in context
    assert '"preferred_output_format": "tex"' in context
    assert "Project memory index:" in context
    assert "Keep claims evidence-backed." in context


@pytest.mark.anyio
async def test_writing_nodes_allow_direct_request_without_dossier(tmp_path: Path) -> None:
    store = WritingStore(workspace=tmp_path, run_id="write_002")
    request = WritingRequest(
        request="Write an internal report from the existing workspace evidence without new calculations.",
        source_campaign_id=None,
    )
    init_cmd = await init_writing_node({"request": request.model_dump()}, writing_store=store, source_store=None)
    assert init_cmd.goto == "plan_writing"
    assert init_cmd.update["dossier"] is None

    plan_cmd = await plan_writing_node(
        {
            "request": request.model_dump(),
            "board": init_cmd.update["board"],
            "dossier": init_cmd.update["dossier"],
        },
        writing_store=store,
        source_store=None,
        write_director_agent=_DummyAgent(
            WritingPlanModel(
                title="Fe workspace summary",
                writing_mode="internal_report",
                target_audience="internal",
                section_specs=[
                    WritingSectionSpec(
                        section_id="results_discussion",
                        heading="Results and Discussion",
                        purpose="Summarize available workspace evidence.",
                    )
                ],
            )
        ),
        skills_runtime=None,
    )
    assert plan_cmd.goto == "write_section"


@pytest.mark.anyio
async def test_assemble_manuscript_always_writes_master_tex_and_copies_figures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_polish(payload):
        path = tmp_path / "files" / str(payload["source_path"])
        text = path.read_text(encoding="utf-8")
        path.write_text(text + "\n% polished\n", encoding="utf-8")
        return "polished", {"data": {"model_name": "fake-polisher"}}

    monkeypatch.setattr("catmaster.agents.writing_nodes.polish_academic_prose", _fake_polish)

    source_store = ResearchStore(workspace=tmp_path, campaign_id="camp_tex")
    source_store.save_board(
        ResearchBoard(
            campaign_id="camp_tex",
            question="Assemble TeX manuscript fragments.",
            exploration_policy="anchored",
            status="done",
            max_cycles=1,
            max_literature_queries=0,
            max_fast_runs=0,
            max_standard_runs=0,
        )
    )
    store = WritingStore(workspace=tmp_path, run_id="write_tex")
    request = WritingRequest(request="Write a TeX section from existing files.", source_campaign_id="camp_tex")
    store.write_request(request.model_dump())
    board = source_store.load_board()
    _ = board
    init_cmd = await init_writing_node({"request": request.model_dump()}, writing_store=store, source_store=source_store)
    board_model = init_cmd.update["board"]
    plan = WritingPlanModel(
        title="TeX assembly",
        writing_mode="section_draft",
        target_audience="internal",
        section_specs=[
            WritingSectionSpec(
                section_id="rd1",
                heading="Results and Discussion",
                purpose="Assemble from explicit tex artifact.",
            )
        ],
    )
    store.persist_plan(plan)
    tex_path = tmp_path / "files" / "research" / "camp_tex" / "manuscript" / "RD-1.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\\section{Results and Discussion}\n\\includegraphics{FIG-demo.png}\nBody.\n", encoding="utf-8")
    fig_path = tmp_path / "files" / "research" / "camp_tex" / "manuscript" / "FIG-demo.png"
    fig_path.write_bytes(b"fakepng")
    store.persist_section_draft(
        SectionDraftModel(
            section_id="rd1",
            heading="Results and Discussion",
            status="drafted",
            latex_artifact_refs=["research/camp_tex/manuscript/RD-1.tex"],
            figure_refs=["research/camp_tex/manuscript/FIG-demo.png"],
        )
    )
    store.persist_section_review(
        SectionReviewModel(
            section_id="rd1",
            status="approved",
            revision_notes=[],
            unsupported_claims=[],
            missing_citations=[],
        )
    )

    result = await assemble_manuscript_node(
        {
            "request": request.model_dump(),
            "board": board_model,
            "plan": plan,
        },
        writing_store=store,
        source_store=source_store,
        writing_config=WritingRuntimeConfig(author_name="CatMaster"),
    )

    assert result["status"] == "done"
    latex_manuscript = tmp_path / "files" / "writing" / "write_tex" / "manuscript" / "MANUSCRIPT.tex"
    copied_figure = tmp_path / "files" / "writing" / "write_tex" / "manuscript" / "FIG-demo.png"
    copied_bib = tmp_path / "files" / "writing" / "write_tex" / "manuscript" / "references.bib"
    copied_section = tmp_path / "files" / "writing" / "write_tex" / "manuscript" / "sections" / "rd1_results_and_discussion.tex"
    assert latex_manuscript.exists()
    assert copied_figure.exists()
    assert copied_bib.exists()
    assert copied_section.exists()
    text = latex_manuscript.read_text(encoding="utf-8")
    assert "\\documentclass[journal=jacsat,manuscript=article]{achemso}" in text
    assert "\\input{sections/rd1_results_and_discussion.tex}" in text
    assert "\\section{Results and Discussion}" in copied_section.read_text(encoding="utf-8")
    assert "% polished" in text
    bundle = store.load_bundle()
    assert bundle is not None
    assert bundle.final_latex_path == "writing/write_tex/manuscript/MANUSCRIPT.tex"
    assert bundle.final_manuscript_path == "writing/write_tex/manuscript/MANUSCRIPT.tex"
    assert any(item.get("copied_ref") == "writing/write_tex/manuscript/FIG-demo.png" for item in bundle.figure_manifest)
