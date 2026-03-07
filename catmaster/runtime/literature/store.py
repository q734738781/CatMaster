from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from catmaster.tools.base import ensure_project_space_layout, project_space_root, system_root

from .models import LiteratureContextPack, PaperRecord


class LiteratureStore:
    def __init__(self, *, workspace: Path | str | None = None) -> None:
        root = project_space_root(workspace)
        ensure_project_space_layout(root, create=True)
        self.metadata_root = system_root(root)
        self.records_dir = self.metadata_root / "literature_records"
        self.memos_dir = self.metadata_root / "literature_memos"
        self.queries_dir = self.metadata_root / "literature_queries"
        for path in (self.records_dir, self.memos_dir, self.queries_dir):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slug(text: str, *, max_len: int = 48) -> str:
        normalized = "-".join(part for part in str(text or "").lower().split() if part)
        cleaned = "".join(ch for ch in normalized if ch.isalnum() or ch == "-")
        return (cleaned[:max_len] or "query").strip("-") or "query"

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()[:12]

    def persist_records(self, papers: Iterable[PaperRecord]) -> list[str]:
        written: list[str] = []
        seen: set[str] = set()
        for paper in papers:
            key = paper.paper_id or paper.doi or self._hash(paper.title)
            key = self._slug(key, max_len=80)
            if not key or key in seen:
                continue
            seen.add(key)
            path = self.records_dir / f"{key}.json"
            path.write_text(json.dumps(paper.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
            written.append(str(path))
        return written

    def persist_query_cache(self, *, query: str, depth: str, candidate_paper_ids: list[str]) -> str:
        name = f"{self._slug(query)}-{self._hash(query + '|' + depth)}.json"
        path = self.queries_dir / name
        payload = {
            "query": query,
            "depth": depth,
            "candidate_paper_ids": list(candidate_paper_ids),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def persist_memo(self, pack: LiteratureContextPack) -> str:
        name = f"{self._slug(pack.query)}-{self._hash(pack.query + '|' + pack.depth)}.json"
        path = self.memos_dir / name
        path.write_text(json.dumps(pack.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)


__all__ = ["LiteratureStore"]
