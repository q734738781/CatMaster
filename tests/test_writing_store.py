from __future__ import annotations

from pathlib import Path

from catmaster.runtime.writing import (
    ManuscriptBundleModel,
    SectionDraftModel,
    SectionReviewModel,
    WritingBoard,
    WritingPlanModel,
    WritingSectionSpec,
    WritingStore,
)


def test_writing_store_round_trip(tmp_path: Path) -> None:
    store = WritingStore(workspace=tmp_path, run_id="write_001")
    store.prepare_new_run()
    req_path = store.write_request(
        {
            "request": "Write a full draft from current evidence.",
            "source_campaign_id": "camp_001",
        }
    )
    assert req_path == "manuscript/state/writing_request.json"

    board = WritingBoard(
        run_id="write_001",
        source_campaign_id="camp_001",
        writing_mode="full_draft",
        title="CO adsorption on Fe(110)",
    )
    board_path = store.save_board(board)
    assert board_path == "manuscript/state/board.json"
    assert store.load_board() is not None

    plan = WritingPlanModel(
        title="CO adsorption on Fe(110)",
        writing_mode="full_draft",
        target_audience="internal",
        section_specs=[
            WritingSectionSpec(
                section_id="sec_results",
                heading="Results and Discussion",
                purpose="Explain the leading evidence chain.",
            )
        ],
    )
    assert store.persist_plan(plan) == "manuscript/state/writing_plan.json"
    assert store.load_plan() is not None

    draft = SectionDraftModel(
        section_id="sec_results",
        heading="Results and Discussion",
        status="drafted",
        section_tex="\\section{Results and Discussion}\nBridge adsorption remained preferred.",
    )
    assert store.persist_section_draft(draft) == "manuscript/state/sections/sec_results.json"
    assert len(store.load_section_drafts()) == 1

    review = SectionReviewModel(section_id="sec_results", status="approved")
    assert store.persist_section_review(review) == "manuscript/state/reviews/sec_results.json"
    assert len(store.load_section_reviews()) == 1

    manuscript_path = store.write_manuscript("MANUSCRIPT.tex", "\\section{Test}\n")
    assert manuscript_path == "manuscript/MANUSCRIPT.tex"

    bundle = ManuscriptBundleModel(
        source_campaign_id="camp_001",
        writing_mode="full_draft",
        title="CO adsorption on Fe(110)",
        final_manuscript_path=manuscript_path,
        final_latex_path="manuscript/MANUSCRIPT.tex",
    )
    assert store.persist_bundle(bundle) == "manuscript/state/manuscript_bundle.json"
    assert store.load_bundle() is not None


def test_writing_store_archives_previous_manuscript_tree(tmp_path: Path) -> None:
    first = WritingStore(workspace=tmp_path, run_id="write_001")
    first.prepare_new_run()
    first.write_manuscript("MANUSCRIPT.tex", "\\section{Old}\n")
    first.persist_plan(
        WritingPlanModel(
            title="Old",
            writing_mode="section_draft",
            target_audience="internal",
        )
    )

    second = WritingStore(workspace=tmp_path, run_id="write_002")
    second.prepare_new_run()

    archived_tex = tmp_path / "files" / "manuscript_archive" / "write_002" / "MANUSCRIPT.tex"
    fresh_root = tmp_path / "files" / "manuscript"
    assert archived_tex.exists()
    assert fresh_root.exists()
    assert not (fresh_root / "MANUSCRIPT.tex").exists()
