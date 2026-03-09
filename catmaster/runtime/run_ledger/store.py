from __future__ import annotations

import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from catmaster.runtime.run_ledger.models import RunLedgerEntry, RunSearchHit
from catmaster.tools.base import system_root


class RunLedgerStore:
    """SQLite-backed run ledger storage with FTS5 sparse retrieval."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @classmethod
    def create_default(cls, *, workspace: Path | str | None = None) -> "RunLedgerStore":
        root = system_root(workspace=workspace)
        return cls(root / "run_ledger.sqlite")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS run_ledger (
                        run_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        lane TEXT NOT NULL,
                        status TEXT NOT NULL,
                        request TEXT NOT NULL,
                        answer_summary TEXT NOT NULL,
                        search_blob_text TEXT NOT NULL,
                        final_report_relpath TEXT NOT NULL,
                        run_export_relpath TEXT NOT NULL,
                        ts_start TEXT NOT NULL,
                        ts_end TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS run_ledger_fts USING fts5(
                        run_id UNINDEXED,
                        request,
                        answer_summary,
                        search_blob_text
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_run_ledger_project ON run_ledger(project_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_run_ledger_lane ON run_ledger(lane)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_run_ledger_status ON run_ledger(status)")
                conn.commit()

    @staticmethod
    def _fts_query(query: str) -> str:
        parts = re.findall(r"[A-Za-z0-9_./:+-]+", str(query or "").lower())
        if not parts:
            cleaned = " ".join(str(query or "").split()).strip()
            return f"\"{cleaned}\"" if cleaned else ""
        dedup: list[str] = []
        seen: set[str] = set()
        for token in parts:
            if token in seen:
                continue
            seen.add(token)
            dedup.append(token)
            if len(dedup) >= 24:
                break
        return " OR ".join(f"\"{token}\"" for token in dedup)

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> RunLedgerEntry:
        return RunLedgerEntry(
            project_id=str(row["project_id"] or ""),
            run_id=str(row["run_id"] or ""),
            lane=str(row["lane"] or ""),
            status=str(row["status"] or ""),
            request=str(row["request"] or ""),
            answer_summary=str(row["answer_summary"] or ""),
            search_blob_text=str(row["search_blob_text"] or ""),
            final_report_relpath=str(row["final_report_relpath"] or ""),
            run_export_relpath=str(row["run_export_relpath"] or ""),
            ts_start=str(row["ts_start"] or ""),
            ts_end=str(row["ts_end"] or ""),
            model_name=str(row["model_name"] or ""),
            provider=str(row["provider"] or ""),
        )

    @staticmethod
    def _row_to_hit(row: sqlite3.Row, *, source: str, score: float) -> RunSearchHit:
        return RunSearchHit(
            run_id=str(row["run_id"] or ""),
            project_id=str(row["project_id"] or ""),
            lane=str(row["lane"] or ""),
            status=str(row["status"] or ""),
            score=float(score),
            source=source,
            request=str(row["request"] or ""),
            answer_summary=str(row["answer_summary"] or ""),
            final_report_relpath=str(row["final_report_relpath"] or ""),
            run_export_relpath=str(row["run_export_relpath"] or ""),
        )

    def upsert_entry(self, entry: RunLedgerEntry) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO run_ledger (
                        run_id, project_id, lane, status,
                        request, answer_summary, search_blob_text,
                        final_report_relpath, run_export_relpath,
                        ts_start, ts_end, model_name, provider, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        project_id=excluded.project_id,
                        lane=excluded.lane,
                        status=excluded.status,
                        request=excluded.request,
                        answer_summary=excluded.answer_summary,
                        search_blob_text=excluded.search_blob_text,
                        final_report_relpath=excluded.final_report_relpath,
                        run_export_relpath=excluded.run_export_relpath,
                        ts_start=excluded.ts_start,
                        ts_end=excluded.ts_end,
                        model_name=excluded.model_name,
                        provider=excluded.provider,
                        updated_at=excluded.updated_at
                    """,
                    (
                        entry.run_id,
                        entry.project_id,
                        entry.lane,
                        entry.status,
                        entry.request,
                        entry.answer_summary,
                        entry.search_blob_text,
                        entry.final_report_relpath,
                        entry.run_export_relpath,
                        entry.ts_start,
                        entry.ts_end,
                        entry.model_name,
                        entry.provider,
                        now,
                    ),
                )
                conn.execute("DELETE FROM run_ledger_fts WHERE run_id = ?", (entry.run_id,))
                conn.execute(
                    """
                    INSERT INTO run_ledger_fts (run_id, request, answer_summary, search_blob_text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        entry.run_id,
                        entry.request,
                        entry.answer_summary,
                        entry.search_blob_text,
                    ),
                )
                conn.commit()

    def get_entry(self, run_id: str) -> RunLedgerEntry | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM run_ledger WHERE run_id = ?",
                    (str(run_id or "").strip(),),
                ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def get_entries(self, run_ids: List[str]) -> List[RunLedgerEntry]:
        cleaned = [str(item or "").strip() for item in run_ids if str(item or "").strip()]
        if not cleaned:
            return []
        placeholders = ",".join("?" for _ in cleaned)
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT * FROM run_ledger WHERE run_id IN ({placeholders})",
                    cleaned,
                ).fetchall()
        by_id = {str(row["run_id"]): self._row_to_entry(row) for row in rows}
        ordered: List[RunLedgerEntry] = []
        for rid in cleaned:
            hit = by_id.get(rid)
            if hit is not None:
                ordered.append(hit)
        return ordered

    def search_sparse(
        self,
        project_id: str,
        query: str,
        limit: int,
        lane: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[RunSearchHit]:
        q = self._fts_query(query)
        if not q:
            return []
        lim = max(1, int(limit))
        lane_v = str(lane or "").strip() or None
        status_v = str(status or "").strip() or None
        proj = str(project_id or "").strip()

        where_parts = ["rl.project_id = ?"]
        params: list[object] = [proj]
        if lane_v:
            where_parts.append("rl.lane = ?")
            params.append(lane_v)
        if status_v:
            where_parts.append("rl.status = ?")
            params.append(status_v)

        sql = (
            "SELECT rl.*, bm25(run_ledger_fts) AS rank "
            "FROM run_ledger_fts JOIN run_ledger rl ON rl.run_id = run_ledger_fts.run_id "
            f"WHERE run_ledger_fts MATCH ? AND {' AND '.join(where_parts)} "
            "ORDER BY rank ASC LIMIT ?"
        )
        params_fts = [q, *params, lim]

        with self._lock:
            with self._connect() as conn:
                try:
                    rows = conn.execute(sql, params_fts).fetchall()
                    hits: List[RunSearchHit] = []
                    for row in rows:
                        rank = float(row["rank"] if row["rank"] is not None else 0.0)
                        score = 1.0 / (1.0 + max(rank, 0.0))
                        hits.append(self._row_to_hit(row, source="sparse", score=score))
                    return hits
                except sqlite3.OperationalError:
                    pass

                # Fallback path for malformed FTS queries or environments with limited FTS behavior.
                like = f"%{str(query or '').strip()}%"
                where_like = list(where_parts)
                where_like.append("(rl.request LIKE ? OR rl.answer_summary LIKE ? OR rl.search_blob_text LIKE ?)")
                params_like = [*params, like, like, like, lim]
                rows = conn.execute(
                    "SELECT rl.* FROM run_ledger rl "
                    f"WHERE {' AND '.join(where_like)} "
                    "ORDER BY rl.ts_end DESC LIMIT ?",
                    params_like,
                ).fetchall()

        out: List[RunSearchHit] = []
        for idx, row in enumerate(rows):
            score = 1.0 / float(idx + 1)
            out.append(self._row_to_hit(row, source="sparse", score=score))
        return out


__all__ = ["RunLedgerStore"]
