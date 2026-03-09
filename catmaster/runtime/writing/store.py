from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from catmaster.tools.base import ensure_project_space_layout, workspace_root

from .models import ManuscriptBundleModel, SectionDraftModel, SectionReviewModel, WritingBoard, WritingPlanModel


class WritingStore:
    def __init__(self, *, workspace: Path, run_id: str) -> None:
        ensure_project_space_layout(workspace, create=True)
        self.workspace = Path(workspace).expanduser().resolve()
        self.run_id = str(run_id).strip()
        self.files_root = workspace_root(self.workspace) / "writing" / self.run_id
        self.state_root = self.files_root / "state"

    @property
    def sections_dir(self) -> Path:
        return self.state_root / "sections"

    @property
    def reviews_dir(self) -> Path:
        return self.state_root / "reviews"

    def ensure_exists(self) -> None:
        for path in (
            self.state_root,
            self.sections_dir,
            self.reviews_dir,
            self.files_root,
            self.files_root / "manuscript",
        ):
            path.mkdir(parents=True, exist_ok=True)
        for legacy in (self.files_root / "figures", self.files_root / "latex"):
            if legacy.exists():
                try:
                    legacy.rmdir()
                except OSError:
                    pass

    def write_request(self, payload: dict[str, Any]) -> str:
        self.ensure_exists()
        path = self.state_root / "writing_request.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._files_rel(path)

    def load_request(self) -> dict[str, Any] | None:
        path = self.state_root / "writing_request.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None

    def save_board(self, board: WritingBoard) -> str:
        self.ensure_exists()
        path = self.state_root / "board.json"
        path.write_text(json.dumps(board.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return self._files_rel(path)

    def load_board(self) -> WritingBoard | None:
        path = self.state_root / "board.json"
        if not path.exists():
            return None
        return WritingBoard.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def persist_plan(self, plan: WritingPlanModel) -> str:
        self.ensure_exists()
        path = self.state_root / "writing_plan.json"
        path.write_text(json.dumps(plan.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return self._files_rel(path)

    def load_plan(self) -> WritingPlanModel | None:
        path = self.state_root / "writing_plan.json"
        if not path.exists():
            return None
        return WritingPlanModel.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def persist_section_draft(self, draft: SectionDraftModel) -> str:
        self.ensure_exists()
        path = self.sections_dir / f"{draft.section_id}.json"
        path.write_text(json.dumps(draft.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return self._files_rel(path)

    def load_section_drafts(self) -> list[SectionDraftModel]:
        if not self.sections_dir.exists():
            return []
        out: list[SectionDraftModel] = []
        for path in sorted(self.sections_dir.glob("*.json")):
            out.append(SectionDraftModel.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        return out

    def persist_section_review(self, review: SectionReviewModel) -> str:
        self.ensure_exists()
        path = self.reviews_dir / f"{review.section_id}.json"
        path.write_text(json.dumps(review.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return self._files_rel(path)

    def load_section_reviews(self) -> list[SectionReviewModel]:
        if not self.reviews_dir.exists():
            return []
        out: list[SectionReviewModel] = []
        for path in sorted(self.reviews_dir.glob("*.json")):
            out.append(SectionReviewModel.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        return out

    def write_manuscript(self, filename: str, content_md: str) -> str:
        self.ensure_exists()
        path = self.files_root / "manuscript" / filename
        path.write_text(str(content_md or ""), encoding="utf-8")
        return self._files_rel(path)

    def persist_bundle(self, bundle: ManuscriptBundleModel) -> str:
        self.ensure_exists()
        path = self.state_root / "manuscript_bundle.json"
        path.write_text(json.dumps(bundle.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return self._files_rel(path)

    def load_bundle(self) -> ManuscriptBundleModel | None:
        path = self.state_root / "manuscript_bundle.json"
        if not path.exists():
            return None
        return ManuscriptBundleModel.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def _files_rel(self, path: Path) -> str:
        return str(path.resolve().relative_to(workspace_root(self.workspace).resolve())).replace("\\", "/")


__all__ = ["WritingStore"]
