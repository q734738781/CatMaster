from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from catmaster.tools.base import ensure_project_space_layout, system_root

from .models import LearningCandidate, SelfEvolutionJob, ValidationReport


SELF_EVOLUTION_DIR = "self_evolution"
MEMORY_STORE_FILE = "deepagent_memory.sqlite"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def stable_id(*parts: Any, length: int = 32) -> str:
    payload = "\x1f".join(str(part or "") for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def hash_tree(root: Path) -> str:
    path = Path(root)
    digest = hashlib.sha256()
    if not path.is_dir():
        return ""
    for item in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
        if not item.is_file() or item.is_symlink():
            continue
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_json_dumps(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _safe_component(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."}:
        raise ValueError(f"{label} is required")
    if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in text):
        raise ValueError(f"invalid {label}: {text!r}")
    return text


class SelfEvolutionStore:
    def __init__(self, workspace: Path | str, *, project_id: str = "") -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        ensure_project_space_layout(self.workspace, create=True)
        self.project_id = str(project_id or self.workspace.name).strip() or self.workspace.name
        self.root.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.self_develop_skills_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_jobs_schema()

    @property
    def root(self) -> Path:
        return system_root(self.workspace) / SELF_EVOLUTION_DIR

    @property
    def db_path(self) -> Path:
        return self.root / "jobs.sqlite"

    @property
    def candidates_dir(self) -> Path:
        return self.root / "candidates"

    @property
    def self_develop_skills_dir(self) -> Path:
        return self.root / "self_develop_skills"

    @property
    def audit_log_path(self) -> Path:
        return self.root / "audit.jsonl"

    @property
    def promotion_lock_path(self) -> Path:
        return self.root / "promotion.lock"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_jobs_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    run_dir TEXT NOT NULL,
                    thread_id TEXT NOT NULL DEFAULT '',
                    trigger_kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    candidate_id TEXT NOT NULL DEFAULT '',
                    model_config TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, created_at)")

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> SelfEvolutionJob:
        data = dict(row)
        try:
            data["payload"] = json.loads(data.pop("payload_json") or "{}")
        except Exception:
            data["payload"] = {}
        return SelfEvolutionJob.from_dict(data)

    def enqueue_job(
        self,
        *,
        trigger_kind: str,
        run_id: str,
        run_dir: Path | str,
        thread_id: str = "",
        payload: dict[str, Any] | None = None,
        model_config: str = "",
    ) -> SelfEvolutionJob:
        run_id = _safe_component(run_id, label="run_id")
        trigger = str(trigger_kind or "post_run").strip() or "post_run"
        job_id = "sej_" + stable_id(self.project_id, trigger, run_id, length=28)
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO jobs(
                    job_id, project_id, run_id, run_dir, thread_id, trigger_kind,
                    status, attempt_count, candidate_id, model_config, payload_json,
                    error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', 0, '', ?, ?, '', ?, ?)
                """,
                (
                    job_id,
                    self.project_id,
                    run_id,
                    str(Path(run_dir).expanduser().resolve()),
                    str(thread_id or "").strip(),
                    trigger,
                    str(model_config or "").strip(),
                    _json_dumps(dict(payload or {})),
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise RuntimeError(f"failed to enqueue self-evolution job {job_id}")
        return self._job_from_row(row)

    def claim_jobs(self, *, limit: int = 4) -> list[SelfEvolutionJob]:
        claimed: list[SelfEvolutionJob] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
            now = utc_now()
            for row in rows:
                conn.execute(
                    "UPDATE jobs SET status = 'running', attempt_count = attempt_count + 1, updated_at = ? WHERE job_id = ? AND status = 'queued'",
                    (now, row["job_id"]),
                )
            conn.commit()
            for row in rows:
                current = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
                if current is not None and current["status"] == "running":
                    claimed.append(self._job_from_row(current))
        return claimed

    def finish_job(
        self,
        job: SelfEvolutionJob,
        *,
        status: str,
        candidate_id: str = "",
        error: str = "",
    ) -> SelfEvolutionJob:
        if status not in {"done", "error"}:
            raise ValueError("finished job status must be done or error")
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, candidate_id = ?, error = ?, updated_at = ? WHERE job_id = ?",
                (status, str(candidate_id or ""), str(error or ""), utc_now(), job.job_id),
            )
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job.job_id,)).fetchone()
        if row is None:
            raise RuntimeError(f"self-evolution job disappeared: {job.job_id}")
        return self._job_from_row(row)

    def list_jobs(self) -> list[SelfEvolutionJob]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at").fetchall()
        return [self._job_from_row(row) for row in rows]

    def requeue_running_jobs(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = 'queued', error = '', updated_at = ? WHERE status = 'running'",
                (utc_now(),),
            )
        return max(0, int(cursor.rowcount or 0))

    def candidate_dir(self, candidate_id: str) -> Path:
        return self.candidates_dir / _safe_component(candidate_id, label="candidate_id")

    def reset_candidate_dir(self, candidate_id: str) -> Path:
        path = self.candidate_dir(candidate_id)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_candidate(self, candidate: LearningCandidate) -> Path:
        candidate.updated_at = utc_now()
        if not candidate.created_at:
            candidate.created_at = candidate.updated_at
        path = self.candidate_dir(candidate.candidate_id) / "candidate.json"
        _atomic_write_json(path, candidate.to_dict())
        return path

    def read_candidate(self, candidate_id: str) -> LearningCandidate | None:
        path = self.candidate_dir(candidate_id) / "candidate.json"
        if not path.is_file():
            return None
        data = _read_json(path)
        return LearningCandidate.from_dict(data) if data else None

    def list_candidates(self) -> list[LearningCandidate]:
        rows: list[LearningCandidate] = []
        for path in sorted(self.candidates_dir.glob("*/candidate.json")):
            data = _read_json(path)
            if data:
                rows.append(LearningCandidate.from_dict(data))
        return sorted(rows, key=lambda item: (item.created_at, item.candidate_id))

    def write_validation_report(self, report: ValidationReport) -> Path:
        path = self.candidate_dir(report.candidate_id) / "validation.json"
        _atomic_write_json(path, report.to_dict())
        return path

    @contextmanager
    def promotion_lock(self) -> Iterator[None]:
        self.promotion_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.promotion_lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def append_audit_event(self, payload: dict[str, Any]) -> None:
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"ts": utc_now(), **dict(payload)}
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(_json_dumps(row) + "\n")

    def read_memory_text(self) -> str:
        prefix = ".".join(("catmaster", self.project_id, "filesystem"))
        path = system_root(self.workspace) / MEMORY_STORE_FILE
        if not path.exists():
            return ""
        try:
            with sqlite3.connect(str(path)) as conn:
                row = conn.execute("SELECT value FROM store WHERE prefix = ? AND key = ?", (prefix, "/AGENTS.md")).fetchone()
        except sqlite3.Error:
            return ""
        return self._decode_memory_value(row[0]) if row else ""

    @staticmethod
    def _decode_memory_value(raw: Any) -> str:
        try:
            payload = json.loads(raw)
        except Exception:
            try:
                payload = json.loads(bytes(raw).decode("utf-8"))
            except Exception:
                return ""
        content = payload.get("content") if isinstance(payload, dict) else ""
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content or "")

    def memory_hash(self) -> str:
        return hash_text(self.read_memory_text())

    def compare_and_swap_memory(self, *, expected_hash: str, new_text: str) -> tuple[bool, str]:
        prefix = ".".join(("catmaster", self.project_id, "filesystem"))
        path = system_root(self.workspace) / MEMORY_STORE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        value_text = _json_dumps({"content": str(new_text or ""), "encoding": "utf-8"})
        with sqlite3.connect(str(path), timeout=30, isolation_level=None) as conn:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS store (
                    prefix TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (prefix, key)
                )
                """
            )
            row = conn.execute("SELECT value FROM store WHERE prefix = ? AND key = ?", (prefix, "/AGENTS.md")).fetchone()
            current = self._decode_memory_value(row[0]) if row else ""
            current_hash = hash_text(current)
            if current_hash != str(expected_hash or ""):
                conn.rollback()
                return False, current_hash
            columns = {str(item[1]) for item in conn.execute("PRAGMA table_info(store)").fetchall()}
            if {"created_at", "updated_at"}.issubset(columns):
                conn.execute(
                    """
                    INSERT INTO store(prefix, key, value, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(prefix, key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
                    """,
                    (prefix, "/AGENTS.md", value_text),
                )
            else:
                conn.execute("DELETE FROM store WHERE prefix = ? AND key = ?", (prefix, "/AGENTS.md"))
                conn.execute("INSERT INTO store(prefix, key, value) VALUES (?, ?, ?)", (prefix, "/AGENTS.md", value_text))
            conn.commit()
        return True, hash_text(str(new_text or ""))


__all__ = [
    "MEMORY_STORE_FILE",
    "SELF_EVOLUTION_DIR",
    "SelfEvolutionStore",
    "hash_text",
    "hash_tree",
    "stable_id",
    "utc_now",
]
