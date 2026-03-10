from __future__ import annotations

from pathlib import Path

import pytest

from catmaster.agents.writing_nodes import (
    assemble_manuscript_node,
    finalize_writing_node,
    init_writing_node,
    plan_writing_node,
    review_section_node,
    write_section_node,
)
from catmaster.agents.writing_prompts import build_section_writer_context
from catmaster.agents.writing_schemas import WritingRequest
from catmaster.runtime.research import ResearchBoard, ResearchDossier, ResearchStore
from catmaster.runtime.writing import SectionDraftModel, SectionReviewModel, WritingBoard, WritingPlanModel, WritingSectionSpec, WritingStore
from catmaster.llm.config import WritingRuntimeConfig


class _DummyAgent:
    def __init__(self, payload):
        self.payload = payload

    async def ainvoke(self, payload):
        _ = payload
        return {"structured_response": self.payload}


class _FailingAgent:
    def __init__(self, exc: Exception):
        self.exc = exc

    async def ainvoke(self, payload):
        _ = payload
        raise self.exc


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
    store.ensure_exists()
    (store.files_root / "references.bib").write_text(
        "@article{fe110_2024,\n"
        "  title={Fe(110) adsorption benchmark},\n"
        "  author={Doe, Jane},\n"
        "  journal={J. Test},\n"
        "  year={2024}\n"
        "}\n",
        encoding="utf-8",
    )
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
                "section_tex": "\\section{Results and Discussion}\nBridge adsorption remained preferred in the bounded calculations~\\cite{fe110_2024}.",
                "citations": ["fe110_2024"],
                "planned_figure_ids": [],
                "realized_figure_refs": [],
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

    assemble_cmd = await assemble_manuscript_node(
        {
            "request": request.model_dump(),
            "board": advance_cmd.update["board"],
            "plan": advance_cmd.update["plan"],
        },
        writing_store=store,
        source_store=source_store,
        writing_config=WritingRuntimeConfig(author_name="CatMaster"),
    )
    assert assemble_cmd.goto == "finalize_writing"
    finalize_cmd = await finalize_writing_node(
        {
            "request": request.model_dump(),
            "board": assemble_cmd.update["board"],
            "plan": advance_cmd.update["plan"],
            "bundle": assemble_cmd.update["bundle"],
            "summary": assemble_cmd.update["summary"],
        },
        writing_store=store,
        source_store=source_store,
        write_finalizer_agent=_DummyAgent(
            {
                "summary": "Compile/fix pass completed.",
                "compile_notes": ["compiled in static-check mode"],
                "final_latex_path": "manuscript/MANUSCRIPT.tex",
            }
        ),
        skills_runtime=None,
    )
    assert finalize_cmd.goto == "summarize_writing"
    latex_manuscript = tmp_path / "files" / "manuscript" / "MANUSCRIPT.tex"
    assert latex_manuscript.exists()
    latex_text = latex_manuscript.read_text(encoding="utf-8")
    assert "\\documentclass[journal=jacsat,manuscript=article]{achemso}" in latex_text
    assert "\\author{CatMaster}" in latex_text
    assert "\\title{CO adsorption on Fe(110)}" in latex_text
    assert "\\bibliography{references}" in latex_text
    assert "\\begin{abstract}" not in latex_text
    assert "% polished" in latex_text
    updated_source_board = source_store.load_board()
    assert updated_source_board is not None
    assert updated_source_board.latest_writer_ref == "manuscript/MANUSCRIPT.tex"
    assert updated_source_board.action_refs[-1].kind == "writer"


@pytest.mark.anyio
async def test_review_section_node_blocks_missing_bibliography_keys(tmp_path: Path) -> None:
    store = WritingStore(workspace=tmp_path, run_id="write_001")
    store.ensure_exists()
    draft = SectionDraftModel(
        section_id="results_discussion",
        heading="Results and Discussion",
        status="drafted",
        section_tex="\\section{Results and Discussion}\nA key comparison remains informative~\\cite{missing_key}.",
        citations=["missing_key"],
        planned_figure_ids=[],
        realized_figure_refs=[],
    )
    store.persist_section_draft(draft)
    board = WritingBoard(run_id="write_001", status="reviewing")
    store.save_board(board)
    plan = WritingPlanModel(
        title="Test",
        writing_mode="section_draft",
        target_audience="internal",
        section_specs=[
            WritingSectionSpec(
                section_id="results_discussion",
                heading="Results and Discussion",
                purpose="Summarize evidence.",
            )
        ],
    )
    store.persist_plan(plan)
    review_cmd = await review_section_node(
        {
            "request": WritingRequest(request="Review this section.").model_dump(),
            "board": board.model_dump(),
            "plan": plan.model_dump(),
            "latest_draft": draft.model_dump(),
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
    review = store.load_section_reviews()[0]
    assert review.status == "needs_revision"
    assert review.missing_citations == ["missing_key"]


@pytest.mark.anyio
async def test_review_section_node_forces_progress_after_revision_limit(tmp_path: Path) -> None:
    store = WritingStore(workspace=tmp_path, run_id="write_001")
    store.ensure_exists()
    draft = SectionDraftModel(
        section_id="results_discussion",
        heading="Results and Discussion",
        status="drafted",
        section_tex="\\section{Results and Discussion}\nStill cites a missing source~\\cite{missing_key}.",
        citations=["missing_key"],
        planned_figure_ids=[],
        realized_figure_refs=[],
    )
    store.persist_section_draft(draft)
    board = WritingBoard(
        run_id="write_001",
        status="reviewing",
        revision_counts={"results_discussion": 4},
    )
    store.save_board(board)
    plan = WritingPlanModel(
        title="Test",
        writing_mode="section_draft",
        target_audience="internal",
        section_specs=[
            WritingSectionSpec(
                section_id="results_discussion",
                heading="Results and Discussion",
                purpose="Summarize evidence.",
            )
        ],
    )
    store.persist_plan(plan)
    review_cmd = await review_section_node(
        {
            "request": WritingRequest(request="Review this section.").model_dump(),
            "board": board.model_dump(),
            "plan": plan.model_dump(),
            "latest_draft": draft.model_dump(),
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
    review = store.load_section_reviews()[0]
    assert review.status == "approved"
    assert any("Revision limit reached" in note for note in review.revision_notes)
    updated_board = store.load_board()
    assert updated_board is not None
    assert updated_board.revision_counts["results_discussion"] == 5


def test_section_writer_context_carries_plan_level_tex_signal() -> None:
    plan = WritingPlanModel(
        title="CO adsorption on Fe(110)",
        writing_mode="section_draft",
        target_audience="internal",
        preferred_output_format="tex",
        figure_requests=[],
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
        working_manuscript_root="manuscript",
        working_sections_dir="manuscript/sections",
        working_figures_dir="manuscript/figures",
        prior_draft=None,
        review_notes=[],
        figure_registry=[],
        skill_guide="template-aware skill available",
    )
    assert "Preferred output format: tex" in context
    assert "Plan overview JSON:" in context
    assert '"preferred_output_format": "tex"' in context
    assert "Project memory index:" in context
    assert "Keep claims evidence-backed." in context
    assert "Current manuscript output root: manuscript" in context
    assert "Current manuscript sections output dir: manuscript/sections" in context
    assert "Current manuscript figures output dir: manuscript/figures" in context
    assert "Relevant figure requests JSON:" in context
    assert "Writing plan JSON:" not in context


@pytest.mark.anyio
async def test_write_section_failure_routes_back_to_director(tmp_path: Path) -> None:
    source_store = ResearchStore(workspace=tmp_path, campaign_id="camp_001")
    store = WritingStore(workspace=tmp_path, run_id="write_001")
    request = WritingRequest(
        request="Revise the manuscript and improve figures.",
        source_campaign_id="camp_001",
    )
    init_cmd = await init_writing_node({"request": request.model_dump()}, writing_store=store, source_store=source_store)
    plan = WritingPlanModel(
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
    board = init_cmd.update["board"].model_copy(update={"title": plan.title, "writing_mode": plan.writing_mode, "status": "drafting"})
    store.persist_plan(plan)
    store.save_board(board)

    cmd = await write_section_node(
        {
            "request": request.model_dump(),
            "board": board,
            "plan": plan,
            "dossier": {},
        },
        writing_store=store,
        source_store=source_store,
        section_writer_agent=_FailingAgent(ValueError("upstream 500")),
        skills_runtime=None,
    )
    assert cmd.goto == "plan_writing"
    assert "Section writing failed for `Results and Discussion`." in str(cmd.update["assembly_feedback"])
    assert "ValueError: upstream 500" in str(cmd.update["assembly_feedback"])
    persisted = store.load_board()
    assert persisted is not None
    assert persisted.status == "planning"


def test_coerce_draft_splits_legacy_figure_refs() -> None:
    from catmaster.agents.writing_nodes import _coerce_draft

    draft = _coerce_draft(
        {
            "section_id": "intro",
            "heading": "Introduction",
            "status": "drafted",
            "section_tex": "\\section{Introduction}\n",
            "figure_refs": ["fig_workflow_and_models", "manuscript/figures/fig_workflow_and_models.png"],
        }
    )
    assert draft.planned_figure_ids == ["fig_workflow_and_models"]
    assert draft.realized_figure_refs == ["manuscript/figures/fig_workflow_and_models.png"]


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
async def test_assemble_manuscript_routes_back_to_planning_on_duplicate_figure_inclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_polish(_payload):
        return "polished", {"data": {"model_name": "fake-polisher"}}

    monkeypatch.setattr("catmaster.agents.writing_nodes.polish_academic_prose", _fake_polish)

    source_store = ResearchStore(workspace=tmp_path, campaign_id="camp_dup")
    source_store.save_board(
        ResearchBoard(
            campaign_id="camp_dup",
            question="Check duplicate figures.",
            exploration_policy="anchored",
            status="done",
            max_cycles=1,
            max_literature_queries=0,
            max_fast_runs=0,
            max_standard_runs=0,
        )
    )
    source_store.persist_dossier(
        ResearchDossier(
            campaign_id="camp_dup",
            question="Check duplicate figures.",
            exploration_policy="anchored",
            final_answer_md="Existing results are enough.",
            confidence="medium",
        )
    )
    store = WritingStore(workspace=tmp_path, run_id="write_dup")
    request = WritingRequest(request="Write a short draft.", source_campaign_id="camp_dup")
    init_cmd = await init_writing_node({"request": request.model_dump()}, writing_store=store, source_store=source_store)
    board_model = init_cmd.update["board"]
    plan = WritingPlanModel(
        title="Duplicate figure test",
        writing_mode="full_draft",
        target_audience="internal",
        section_specs=[
            WritingSectionSpec(section_id="intro", heading="Introduction", purpose="Intro."),
            WritingSectionSpec(section_id="results", heading="Results", purpose="Results."),
        ],
    )
    store.persist_plan(plan)
    fig_path = tmp_path / "files" / "manuscript" / "figures" / "FIG-dup.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path.write_bytes(b"fakepng")
    store.persist_section_draft(
        SectionDraftModel(
            section_id="intro",
            heading="Introduction",
            status="drafted",
            section_tex="\\section{Introduction}\n\\begin{figure}\\includegraphics{FIG-dup.png}\\end{figure}\n",
            planned_figure_ids=["fig_dup"],
            realized_figure_refs=["manuscript/figures/FIG-dup.png"],
        )
    )
    store.persist_section_draft(
        SectionDraftModel(
            section_id="results",
            heading="Results",
            status="drafted",
            section_tex="\\section{Results}\n\\begin{figure}\\includegraphics{FIG-dup.png}\\end{figure}\n",
            planned_figure_ids=["fig_dup_reused_badly"],
            realized_figure_refs=["manuscript/figures/FIG-dup.png"],
        )
    )
    store.persist_section_review(
        SectionReviewModel(section_id="intro", status="approved", revision_notes=[], unsupported_claims=[], missing_citations=[])
    )
    store.persist_section_review(
        SectionReviewModel(section_id="results", status="approved", revision_notes=[], unsupported_claims=[], missing_citations=[])
    )

    assemble_cmd = await assemble_manuscript_node(
        {
            "request": request.model_dump(),
            "board": board_model,
            "plan": plan,
        },
        writing_store=store,
        source_store=source_store,
        writing_config=WritingRuntimeConfig(author_name="CatMaster"),
    )
    assert assemble_cmd.goto == "plan_writing"
    assert "same realized figure image was inserted" in str(assemble_cmd.update.get("assembly_feedback") or "")


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
            planned_figure_ids=["fig_demo"],
            realized_figure_refs=["research/camp_tex/manuscript/FIG-demo.png"],
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

    assemble_cmd = await assemble_manuscript_node(
        {
            "request": request.model_dump(),
            "board": board_model,
            "plan": plan,
        },
        writing_store=store,
        source_store=source_store,
        writing_config=WritingRuntimeConfig(author_name="CatMaster"),
    )
    assert assemble_cmd.goto == "finalize_writing"
    finalize_cmd = await finalize_writing_node(
        {
            "request": request.model_dump(),
            "board": assemble_cmd.update["board"],
            "plan": plan,
            "bundle": assemble_cmd.update["bundle"],
            "summary": assemble_cmd.update["summary"],
        },
        writing_store=store,
        source_store=source_store,
        write_finalizer_agent=_DummyAgent(
            {
                "summary": "Compile/fix pass completed.",
                "compile_notes": ["rewrote relative graphics refs"],
                "final_latex_path": "manuscript/MANUSCRIPT.tex",
            }
        ),
        skills_runtime=None,
    )
    assert finalize_cmd.goto == "summarize_writing"
    latex_manuscript = tmp_path / "files" / "manuscript" / "MANUSCRIPT.tex"
    copied_figure = tmp_path / "files" / "manuscript" / "figures" / "FIG-demo.png"
    copied_section = tmp_path / "files" / "manuscript" / "sections" / "rd1_results_and_discussion.tex"
    assert latex_manuscript.exists()
    assert copied_figure.exists()
    assert copied_section.exists()
    text = latex_manuscript.read_text(encoding="utf-8")
    assert "\\documentclass[journal=jacsat,manuscript=article]{achemso}" in text
    assert "\\title{TeX assembly}" in text
    assert "\\input{sections/rd1_results_and_discussion.tex}" in text
    assert "\\begin{abstract}" not in text
    assert "\\bibliography{references}" not in text
    copied_section_text = copied_section.read_text(encoding="utf-8")
    assert "\\section{Results and Discussion}" in copied_section_text
    assert "\\includegraphics{figures/FIG-demo.png}" in copied_section_text
    assert "research/camp_tex/manuscript/FIG-demo.png" not in copied_section_text
    assert "% polished" in text
    bundle = store.load_bundle()
    assert bundle is not None
    assert bundle.final_latex_path == "manuscript/MANUSCRIPT.tex"
    assert bundle.final_manuscript_path == "manuscript/MANUSCRIPT.tex"
    assert bundle.bibliography_shortlist == []
    assert any(item.get("copied_ref") == "manuscript/figures/FIG-demo.png" for item in bundle.figure_manifest)
    assert any(item.get("planned_figure_ids") == ["fig_demo"] for item in bundle.figure_manifest)


@pytest.mark.anyio
async def test_assemble_manuscript_omits_bibliography_when_no_citations_are_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_polish(payload):
        path = tmp_path / "files" / str(payload["source_path"])
        text = path.read_text(encoding="utf-8")
        path.write_text(text + "\n% polished\n", encoding="utf-8")
        return "polished", {"data": {"model_name": "fake-polisher"}}

    monkeypatch.setattr("catmaster.agents.writing_nodes.polish_academic_prose", _fake_polish)

    source_store = ResearchStore(workspace=tmp_path, campaign_id="camp_nobib")
    source_store.save_board(
        ResearchBoard(
            campaign_id="camp_nobib",
            question="Assemble manuscript without citations.",
            exploration_policy="anchored",
            status="done",
            max_cycles=1,
            max_literature_queries=0,
            max_fast_runs=0,
            max_standard_runs=0,
        )
    )
    store = WritingStore(workspace=tmp_path, run_id="write_nobib")
    request = WritingRequest(request="Write a compact manuscript section without references.", source_campaign_id="camp_nobib")
    init_cmd = await init_writing_node({"request": request.model_dump()}, writing_store=store, source_store=source_store)
    board_model = init_cmd.update["board"]
    plan = WritingPlanModel(
        title="No bibliography test",
        writing_mode="section_draft",
        target_audience="internal",
        section_specs=[
            WritingSectionSpec(
                section_id="results",
                heading="Results and Discussion",
                purpose="Summarize current results without citations.",
            )
        ],
    )
    store.persist_plan(plan)
    store.persist_section_draft(
        SectionDraftModel(
            section_id="results",
            heading="Results and Discussion",
            status="drafted",
            section_tex="\\section{Results and Discussion}\nA compact result paragraph without citation commands.\n",
            citations=["not_used_anywhere"],
        )
    )
    store.persist_section_review(
        SectionReviewModel(
            section_id="results",
            status="approved",
            revision_notes=[],
            unsupported_claims=[],
            missing_citations=[],
        )
    )

    assemble_cmd = await assemble_manuscript_node(
        {
            "request": request.model_dump(),
            "board": board_model,
            "plan": plan,
        },
        writing_store=store,
        source_store=source_store,
        writing_config=WritingRuntimeConfig(author_name="CatMaster"),
    )
    assert assemble_cmd.goto == "finalize_writing"
    latex_manuscript = tmp_path / "files" / "manuscript" / "MANUSCRIPT.tex"
    text = latex_manuscript.read_text(encoding="utf-8")
    assert "\\bibliography{references}" not in text
    assert not (tmp_path / "files" / "manuscript" / "references.bib").exists()
    bundle = store.load_bundle()
    assert bundle is not None
    assert bundle.bibliography_shortlist == []


@pytest.mark.anyio
async def test_assemble_manuscript_preserves_existing_references_bib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_polish(payload):
        path = tmp_path / "files" / str(payload["source_path"])
        text = path.read_text(encoding="utf-8")
        path.write_text(text + "\n% polished\n", encoding="utf-8")
        return "polished", {"data": {"model_name": "fake-polisher"}}

    monkeypatch.setattr("catmaster.agents.writing_nodes.polish_academic_prose", _fake_polish)

    source_store = ResearchStore(workspace=tmp_path, campaign_id="camp_keepbib")
    source_store.save_board(
        ResearchBoard(
            campaign_id="camp_keepbib",
            question="Keep active bibliography entries.",
            exploration_policy="anchored",
            status="done",
            max_cycles=1,
            max_literature_queries=0,
            max_fast_runs=0,
            max_standard_runs=0,
        )
    )
    store = WritingStore(workspace=tmp_path, run_id="write_keepbib")
    store.ensure_exists()
    existing_bib = (
        "@article{Batatia2022,\n"
        "  title={MACE foundation paper},\n"
        "  author={Batatia, Ilyes},\n"
        "  journal={Test Journal},\n"
        "  year={2022}\n"
        "}\n"
    )
    (store.files_root / "references.bib").write_text(existing_bib, encoding="utf-8")
    request = WritingRequest(request="Assemble manuscript with an existing bibliography.", source_campaign_id="camp_keepbib")
    init_cmd = await init_writing_node({"request": request.model_dump()}, writing_store=store, source_store=source_store)
    board_model = init_cmd.update["board"]
    plan = WritingPlanModel(
        title="Preserve bibliography",
        writing_mode="section_draft",
        target_audience="internal",
        section_specs=[
            WritingSectionSpec(
                section_id="results",
                heading="Results and Discussion",
                purpose="Use a grounded citation from the existing bibliography.",
            )
        ],
    )
    store.persist_plan(plan)
    store.persist_section_draft(
        SectionDraftModel(
            section_id="results",
            heading="Results and Discussion",
            status="drafted",
            section_tex="\\section{Results and Discussion}\nMACE is referenced here~\\cite{Batatia2022}.\n",
            citations=["Batatia2022"],
        )
    )
    store.persist_section_review(
        SectionReviewModel(
            section_id="results",
            status="approved",
            revision_notes=[],
            unsupported_claims=[],
            missing_citations=[],
        )
    )

    assemble_cmd = await assemble_manuscript_node(
        {
            "request": request.model_dump(),
            "board": board_model,
            "plan": plan,
        },
        writing_store=store,
        source_store=source_store,
        writing_config=WritingRuntimeConfig(author_name="CatMaster"),
    )
    assert assemble_cmd.goto == "finalize_writing"
    assert (store.files_root / "references.bib").read_text(encoding="utf-8") == existing_bib
    manuscript = (store.files_root / "MANUSCRIPT.tex").read_text(encoding="utf-8")
    assert "\\bibliography{references}" in manuscript
