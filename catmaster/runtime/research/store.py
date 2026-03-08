from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from catmaster.runtime.literature.models import LiteratureContextPack
from catmaster.tools.base import ensure_project_space_layout, system_root, workspace_root

from .models import ConclusionRecord, ExperimentRunPack, ResearchBoard, ResearchDossier


def _slug(text: str, *, fallback: str = "item") -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(text or "").strip())
    compact = "-".join(part for part in cleaned.split("-") if part)
    return compact or fallback


class ResearchStore:
    def __init__(self, *, workspace: Path, campaign_id: str) -> None:
        ensure_project_space_layout(workspace, create=True)
        self.workspace = Path(workspace).expanduser().resolve()
        self.campaign_id = str(campaign_id).strip()
        self.metadata_root = system_root(self.workspace) / "research_campaigns" / self.campaign_id
        self.files_root = workspace_root(self.workspace) / "research" / self.campaign_id

    @property
    def literature_dir(self) -> Path:
        return self.metadata_root / "literature"

    @property
    def experiments_dir(self) -> Path:
        return self.metadata_root / "experiments"

    @property
    def conclusion_dir(self) -> Path:
        return self.metadata_root / "conclusion"

    @property
    def dossier_dir(self) -> Path:
        return self.metadata_root / "dossier"

    def ensure_exists(self) -> None:
        for path in (
            self.metadata_root,
            self.literature_dir,
            self.experiments_dir,
            self.conclusion_dir,
            self.dossier_dir,
            self.files_root / "dossier",
            self.files_root / "manuscript",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def write_request(self, request) -> str:
        self.ensure_exists()
        path = self.metadata_root / "request.json"
        payload = request.model_dump() if hasattr(request, "model_dump") else dict(request)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._meta_rel(path)

    def load_request(self) -> dict[str, Any] | None:
        path = self.metadata_root / "request.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None

    def load_board(self) -> ResearchBoard:
        path = self.metadata_root / "board.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return ResearchBoard.model_validate(data)

    def load_action_log(self) -> list[dict[str, Any]]:
        path = self.metadata_root / "actions.jsonl"
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except Exception:
                continue
            if isinstance(item, dict):
                out.append(item)
        return out

    def save_board(self, board: ResearchBoard) -> str:
        self.ensure_exists()
        path = self.metadata_root / "board.json"
        path.write_text(json.dumps(board.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        (self.metadata_root / "board.md").write_text(self._render_board_md(board), encoding="utf-8")
        return self._meta_rel(path)

    def persist_literature_pack(self, pack: LiteratureContextPack, action_id: str) -> str:
        self.ensure_exists()
        path = self.literature_dir / f"{_slug(action_id, fallback='lit')}.json"
        path.write_text(json.dumps(pack.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return self._meta_rel(path)

    def load_literature_packs(self) -> list[LiteratureContextPack]:
        if not self.literature_dir.exists():
            return []
        packs: list[LiteratureContextPack] = []
        for path in sorted(self.literature_dir.glob("*.json")):
            try:
                packs.append(LiteratureContextPack.model_validate(json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return packs

    def persist_experiment_pack(self, pack: ExperimentRunPack, action_id: str) -> str:
        self.ensure_exists()
        path = self.experiments_dir / f"{_slug(action_id, fallback='exp')}.json"
        path.write_text(json.dumps(pack.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return self._meta_rel(path)

    def load_experiment_packs(self) -> list[ExperimentRunPack]:
        if not self.experiments_dir.exists():
            return []
        packs: list[ExperimentRunPack] = []
        for path in sorted(self.experiments_dir.glob("*.json")):
            try:
                packs.append(ExperimentRunPack.model_validate(json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return packs

    def persist_conclusion(self, payload) -> str:
        self.ensure_exists()
        record = ConclusionRecord.model_validate(
            payload.model_dump() if hasattr(payload, "model_dump") else payload
        )
        path = self.conclusion_dir / "conclusion.json"
        path.write_text(json.dumps(record.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return self._meta_rel(path)

    def load_conclusion(self) -> ConclusionRecord | None:
        path = self.conclusion_dir / "conclusion.json"
        if not path.exists():
            return None
        try:
            return ConclusionRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None

    def persist_dossier(self, dossier: ResearchDossier) -> tuple[str, str]:
        self.ensure_exists()
        meta_json = self.dossier_dir / "dossier.json"
        meta_md = self.dossier_dir / "dossier.md"
        file_md = self.files_root / "dossier" / "RESEARCH_DOSSIER.md"
        payload = dossier.model_dump()
        meta_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        rendered = self._render_dossier_md(dossier)
        meta_md.write_text(rendered, encoding="utf-8")
        file_md.write_text(rendered, encoding="utf-8")
        return self._meta_rel(meta_json), self._files_rel(file_md)

    def load_dossier(self) -> ResearchDossier | None:
        path = self.dossier_dir / "dossier.json"
        if not path.exists():
            return None
        try:
            return ResearchDossier.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None

    def write_manuscript(self, filename: str, content_md: str) -> str:
        self.ensure_exists()
        path = self.files_root / "manuscript" / filename
        path.write_text(str(content_md or ""), encoding="utf-8")
        return self._files_rel(path)

    def append_action_log(self, record: dict) -> str:
        self.ensure_exists()
        path = self.metadata_root / "actions.jsonl"
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        return self._meta_rel(path)

    def _render_board_md(self, board: ResearchBoard) -> str:
        lines = [
            f"# Research Board: {board.campaign_id}",
            "",
            f"- Question: {board.question}",
            f"- Policy: {board.exploration_policy}",
            f"- Status: {board.status}",
            f"- Cycle: {board.cycle_index}/{board.max_cycles}",
            (
                "- Budgets:"
                f" literature {board.used_literature_queries}/{board.max_literature_queries},"
                f" fast {board.used_fast_runs}/{board.max_fast_runs},"
                f" standard {board.used_standard_runs}/{board.max_standard_runs}"
            ),
            "",
            "## Hypotheses",
        ]
        if board.hypotheses:
            for item in board.hypotheses:
                lines.append(f"- {item.hypothesis_id} [{item.status}]: {item.text}")
                if item.note:
                    lines.append(f"  note: {item.note}")
        else:
            lines.append("- (none)")
        lines.extend(["", "## Current Best Answer", board.current_best_answer_md or "(empty)", "", "## Open Questions"])
        lines.extend([f"- {item}" for item in board.open_questions] or ["- (none)"])
        if board.latest_human_questions:
            lines.extend(["", "## Latest Ask-Human Questions"])
            lines.extend([f"- {item}" for item in board.latest_human_questions])
        if board.human_feedback_summary:
            lines.extend(["", "## Latest Human Feedback", board.human_feedback_summary])
        lines.extend(["", "## Actions"])
        lines.extend(
            [f"- {item.action_id} [{item.kind}/{item.status}]: {item.summary} -> {item.ref_path}" for item in board.action_refs]
            or ["- (none)"]
        )
        return "\n".join(lines).rstrip() + "\n"

    def _render_dossier_md(self, dossier: ResearchDossier) -> str:
        lines = [
            f"# Research Dossier: {dossier.campaign_id}",
            "",
            f"- Question: {dossier.question}",
            f"- Exploration policy: {dossier.exploration_policy}",
            f"- Confidence: {dossier.confidence}",
            "",
            "## Final Answer",
            dossier.final_answer_md,
            "",
            "## Hypotheses",
        ]
        lines.extend(
            [f"- {item.hypothesis_id} [{item.status}]: {item.text}" for item in dossier.hypotheses]
            or ["- (none)"]
        )
        lines.extend(["", "## Literature Summary", dossier.literature_summary or "(none)", "", "## Key Papers"])
        lines.extend(
            [
                f"- {paper.title} ({paper.year or 'n.d.'})"
                + (f" | {paper.doi}" if paper.doi else "")
                + (f" | {paper.url}" if paper.url else "")
                for paper in dossier.key_papers
            ]
            or ["- (none)"]
        )
        lines.extend(["", "## Experiments"])
        lines.extend(
            [
                f"- {row.experiment_id} [{row.lane}/{row.status}]: {row.goal}"
                f" | {row.summary}"
                for row in dossier.experiment_rows
            ]
            or ["- (none)"]
        )
        lines.extend(["", "## Supported Claims"])
        lines.extend([f"- {item}" for item in dossier.supported_claims] or ["- (none)"])
        lines.extend(["", "## Open Questions"])
        lines.extend([f"- {item}" for item in dossier.open_questions] or ["- (none)"])
        lines.extend(["", "## Recommended Next Steps"])
        lines.extend([f"- {item}" for item in dossier.recommended_next_steps] or ["- (none)"])
        lines.extend(["", "## Key Artifacts"])
        lines.extend([f"- {item.path}: {item.description}" for item in dossier.key_artifacts] or ["- (none)"])
        return "\n".join(lines).rstrip() + "\n"

    def _meta_rel(self, path: Path) -> str:
        return str(path.resolve().relative_to(system_root(self.workspace).resolve())).replace("\\", "/")

    def _files_rel(self, path: Path) -> str:
        return str(path.resolve().relative_to(workspace_root(self.workspace).resolve())).replace("\\", "/")


__all__ = ["ResearchStore"]
