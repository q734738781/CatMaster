from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from catmaster.storage.workspace_db import workspace_journal_mode
from catmaster.tools.base import ensure_project_space_layout, system_root

from .models import (
    LearningCandidate,
    Observation,
    SelfEvolutionJob,
    SkillRun,
    ValidationReport,
    normalize_candidate_status,
)


SELF_EVOLUTION_DIR = "self_evolution"
MEMORY_STORE_FILE = "deepagent_memory.sqlite"
ACTIVE_SKILLS_FILE = "active_skills.json"
_DEFAULT_LEASE_SECONDS = 300


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _future_utc(seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=max(1, int(seconds)))).isoformat()


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


def _write_new_json(path: Path, value: Any) -> None:
    """Create an immutable JSON artifact and reject accidental replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_dumps(value) + "\n"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = path.read_text(encoding="utf-8", errors="replace")
        if existing != payload:
            raise FileExistsError(f"immutable self-evolution artifact already exists: {path}")
        return
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


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
    """Workspace-scoped storage for the four durable self-evolution entities."""

    def __init__(self, workspace: Path | str, *, project_id: str = "") -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        ensure_project_space_layout(self.workspace, create=True)
        self.project_id = str(project_id or self.workspace.name).strip() or self.workspace.name
        self.root.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.self_develop_skills_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @property
    def root(self) -> Path:
        return system_root(self.workspace) / SELF_EVOLUTION_DIR

    @property
    def db_path(self) -> Path:
        # Keep the historical filename so existing deployments upgrade in place.
        return self.root / "jobs.sqlite"

    @property
    def candidates_dir(self) -> Path:
        return self.root / "candidates"

    @property
    def self_develop_skills_dir(self) -> Path:
        # Compatibility materialization. Runtime resolution is pointer based.
        return self.root / "self_develop_skills"

    @property
    def active_skills_path(self) -> Path:
        return self.root / ACTIVE_SKILLS_FILE

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
        conn.execute(f"PRAGMA journal_mode={workspace_journal_mode(self.workspace)}")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}

    def _ensure_schema(self) -> None:
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
                    owner TEXT NOT NULL DEFAULT '',
                    lease_until TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            job_columns = self._columns(conn, "jobs")
            for column, declaration in (
                ("owner", "TEXT NOT NULL DEFAULT ''"),
                ("lease_until", "TEXT NOT NULL DEFAULT ''"),
            ):
                if column not in job_columns:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {declaration}")
            # A v1 ``running`` row has no lease provenance. Blindly executing it
            # again could duplicate an external action, while leaving it running
            # forever makes it undiscoverable. Quarantine it for explicit review.
            conn.execute(
                """
                UPDATE jobs
                SET status = 'recovery_review',
                    error = CASE
                        WHEN TRIM(error) = '' THEN
                            'Legacy running job has no verifiable lease; manual recovery review is required.'
                        ELSE error
                    END,
                    owner = '', updated_at = ?
                WHERE status = 'running' AND TRIM(lease_until) = ''
                """,
                (utc_now(),),
            )
            conn.execute("CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, lease_until, created_at)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    observation_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL DEFAULT '',
                    signal_kind TEXT NOT NULL,
                    target TEXT NOT NULL DEFAULT '',
                    claim TEXT NOT NULL,
                    evidence_refs_json TEXT NOT NULL DEFAULT '[]',
                    outcome_ref TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            observation_columns = self._columns(conn, "observations")
            if "target" not in observation_columns:
                conn.execute(
                    "ALTER TABLE observations ADD COLUMN target TEXT NOT NULL DEFAULT ''"
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS observations_status_idx "
                "ON observations(status, created_at DESC, observation_id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS observations_target_idx "
                "ON observations(target, created_at, observation_id)"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    route TEXT NOT NULL,
                    target_json TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    revision INTEGER NOT NULL,
                    bundle_hash TEXT NOT NULL DEFAULT '',
                    base_target_hash TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS candidates_status_idx "
                "ON candidates(status, created_at DESC, candidate_id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS candidates_updated_idx "
                "ON candidates(updated_at DESC, candidate_id DESC)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_runs (
                    run_id TEXT NOT NULL,
                    skill_name TEXT NOT NULL,
                    skill_version TEXT NOT NULL,
                    presented INTEGER NOT NULL DEFAULT 0,
                    read INTEGER NOT NULL DEFAULT 0,
                    helper_used INTEGER NOT NULL DEFAULT 0,
                    outcome TEXT NOT NULL DEFAULT 'unknown',
                    false_activation INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (run_id, skill_name, skill_version)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS skill_runs_skill_idx ON skill_runs(skill_name, skill_version)")

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
        payload_value = dict(payload or {})
        trigger_identity = _json_dumps(payload_value) if trigger != "post_run" else ""
        job_id = "sej_" + stable_id(
            self.project_id,
            trigger,
            run_id,
            trigger_identity,
            length=28,
        )
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO jobs(
                    job_id, project_id, run_id, run_dir, thread_id, trigger_kind,
                    status, attempt_count, candidate_id, model_config, payload_json,
                    error, owner, lease_until, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', 0, '', ?, ?, '', '', '', ?, ?)
                """,
                (
                    job_id,
                    self.project_id,
                    run_id,
                    str(Path(run_dir).expanduser().resolve()),
                    str(thread_id or "").strip(),
                    trigger,
                    str(model_config or "").strip(),
                    _json_dumps(payload_value),
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise RuntimeError(f"failed to enqueue self-evolution job {job_id}")
        return self._job_from_row(row)

    def claim_jobs(
        self,
        *,
        limit: int = 4,
        project_id: str = "",
        owner: str = "",
        lease_seconds: int = _DEFAULT_LEASE_SECONDS,
    ) -> list[SelfEvolutionJob]:
        claimed: list[SelfEvolutionJob] = []
        target_project_id = str(project_id or "").strip()
        worker = str(owner or f"pid-{os.getpid()}").strip()
        now = utc_now()
        lease_until = _future_utc(lease_seconds)
        where = "(status = 'queued' OR (status = 'running' AND lease_until != '' AND lease_until < ?))"
        params: list[Any] = [now]
        if target_project_id:
            where += " AND project_id = ?"
            params.append(target_project_id)
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                f"SELECT * FROM jobs WHERE {where} ORDER BY created_at LIMIT ?",
                tuple(params),
            ).fetchall()
            for row in rows:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'running', attempt_count = attempt_count + 1,
                        owner = ?, lease_until = ?, updated_at = ?
                    WHERE job_id = ? AND (
                        status = 'queued' OR
                        (status = 'running' AND lease_until != '' AND lease_until < ?)
                    )
                    """,
                    (worker, lease_until, now, row["job_id"], now),
                )
            conn.commit()
            for row in rows:
                current = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
                if current is not None and current["status"] == "running" and current["owner"] == worker:
                    claimed.append(self._job_from_row(current))
        return claimed

    def heartbeat_job(
        self,
        job_id: str,
        *,
        owner: str,
        lease_seconds: int = _DEFAULT_LEASE_SECONDS,
    ) -> bool:
        now = utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs SET lease_until = ?, updated_at = ?
                WHERE job_id = ? AND status = 'running' AND owner = ?
                """,
                (_future_utc(lease_seconds), now, job_id, str(owner or "").strip()),
            )
        return int(cursor.rowcount or 0) == 1

    def finish_job(
        self,
        job: SelfEvolutionJob,
        *,
        status: str,
        candidate_id: str = "",
        error: str = "",
        owner: str = "",
    ) -> SelfEvolutionJob:
        if status not in {"done", "error", "recovery_review"}:
            raise ValueError("finished job status must be done, error, or recovery_review")
        expected_owner = str(owner or job.owner or "").strip()
        if not expected_owner:
            raise ValueError("job owner is required to finish a claimed job")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = ?, candidate_id = ?, error = ?, owner = '',
                    lease_until = '', updated_at = ?
                WHERE job_id = ? AND status = 'running' AND owner = ?
                """,
                (
                    status,
                    str(candidate_id or ""),
                    str(error or ""),
                    utc_now(),
                    job.job_id,
                    expected_owner,
                ),
            )
            if int(cursor.rowcount or 0) != 1:
                raise RuntimeError(
                    f"self-evolution job lease is no longer owned by {expected_owner}: {job.job_id}"
                )
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job.job_id,)).fetchone()
        if row is None:
            raise RuntimeError(f"self-evolution job disappeared: {job.job_id}")
        return self._job_from_row(row)

    def list_jobs(
        self,
        *,
        limit: int = 100,
        before: str = "",
        project_id: str = "",
    ) -> list[SelfEvolutionJob]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        if before:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT created_at FROM jobs WHERE job_id = ?", (before,)
                ).fetchone()
            if row is not None:
                clauses.append("(created_at < ? OR (created_at = ? AND job_id < ?))")
                params.extend([row["created_at"], row["created_at"], before])
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(max(1, min(500, int(limit))))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs{where} ORDER BY created_at DESC, job_id DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def read_job(self, job_id: str) -> SelfEvolutionJob | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id or "").strip(),),
            ).fetchone()
        return self._job_from_row(row) if row is not None else None

    def queued_project_ids(self) -> list[str]:
        now = utc_now()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT project_id FROM jobs
                WHERE status = 'queued'
                   OR (status = 'running' AND lease_until != '' AND lease_until < ?)
                ORDER BY project_id
                """,
                (now,),
            ).fetchall()
        return [str(row["project_id"] or "").strip() for row in rows if str(row["project_id"] or "").strip()]

    def requeue_expired_jobs(self) -> int:
        now = utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = 'queued', error = '', owner = '', lease_until = '',
                    updated_at = ?
                WHERE status = 'running' AND lease_until != '' AND lease_until < ?
                """,
                (now, now),
            )
        return max(0, int(cursor.rowcount or 0))

    def requeue_running_jobs(self) -> int:
        """Compatibility alias with the corrected lease-expiry semantics."""

        return self.requeue_expired_jobs()

    # -- observations -----------------------------------------------------

    def write_observation(self, observation: Observation) -> Observation:
        if observation.signal_kind not in {
            "workspace_preference",
            "skill_revision",
            "skill_discovery",
        }:
            raise ValueError(f"unsupported observation signal: {observation.signal_kind}")
        if not observation.target.strip():
            raise ValueError("observation target is required")
        if not observation.claim.strip():
            raise ValueError("observation claim is required")
        if not observation.created_at:
            observation.created_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO observations(
                    observation_id, run_id, thread_id, signal_kind, target, claim,
                    evidence_refs_json, outcome_ref, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(observation_id) DO NOTHING
                """,
                (
                    observation.observation_id,
                    observation.run_id,
                    observation.thread_id,
                    observation.signal_kind,
                    observation.target.strip(),
                    observation.claim.strip(),
                    _json_dumps(list(observation.evidence_refs)),
                    observation.outcome_ref,
                    observation.status,
                    observation.created_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM observations WHERE observation_id = ?",
                (observation.observation_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"observation disappeared: {observation.observation_id}")
        return self._observation_from_row(row)

    @staticmethod
    def _observation_from_row(row: sqlite3.Row) -> Observation:
        data = dict(row)
        try:
            data["evidence_refs"] = json.loads(data.pop("evidence_refs_json") or "[]")
        except Exception:
            data["evidence_refs"] = []
        return Observation.from_dict(data)

    def read_observation(self, observation_id: str) -> Observation | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM observations WHERE observation_id = ?",
                (observation_id,),
            ).fetchone()
        return self._observation_from_row(row) if row is not None else None

    def list_observations(
        self,
        *,
        status: str = "",
        target: str = "",
        limit: int = 100,
        before: str = "",
    ) -> list[Observation]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if target:
            clauses.append("target = ?")
            params.append(str(target).strip())
        if before:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT created_at FROM observations WHERE observation_id = ?",
                    (before,),
                ).fetchone()
            if row is not None:
                clauses.append("(created_at < ? OR (created_at = ? AND observation_id < ?))")
                params.extend([row["created_at"], row["created_at"], before])
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(max(1, min(500, int(limit))))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM observations{where} "
                "ORDER BY created_at DESC, observation_id DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [self._observation_from_row(row) for row in rows]

    def list_observations_for_target(self, target: str) -> list[Observation]:
        """Return every signal for one exact semantic target.

        Signals are intentionally rare and already compressed by the model, so
        this query does not impose a wording, thread-count, or arbitrary top-k
        cutoff.
        """

        resolved = str(target or "").strip()
        if not resolved:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM observations
                WHERE target = ?
                ORDER BY created_at ASC, observation_id ASC
                """,
                (resolved,),
            ).fetchall()
        return [self._observation_from_row(row) for row in rows]

    def list_observation_targets(self) -> list[str]:
        """List exact targets already selected by semantic reflection."""

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT target FROM observations
                WHERE TRIM(target) != ''
                ORDER BY target
                """
            ).fetchall()
        return [
            str(row["target"]).strip()
            for row in rows
            if str(row["target"] or "").strip()
        ]

    def set_observation_status(self, observation_ids: list[str], status: str) -> int:
        if status not in {"open", "consolidated"}:
            raise ValueError(f"unsupported observation status: {status}")
        ids = [str(item).strip() for item in observation_ids if str(item).strip()]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE observations SET status = ? WHERE observation_id IN ({placeholders})",
                (status, *ids),
            )
        return max(0, int(cursor.rowcount or 0))

    def run_dir_for(self, run_id: str) -> Path | None:
        """Resolve the durable run directory already owned by a queued job."""

        resolved = str(run_id or "").strip()
        if not resolved:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_dir FROM jobs
                WHERE project_id = ? AND run_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (self.project_id, resolved),
            ).fetchone()
        if row is None:
            return None
        path = Path(str(row["run_dir"] or "")).expanduser().resolve()
        return path if path.is_dir() else None

    # -- immutable candidate revisions ----------------------------------

    def candidate_dir(self, candidate_id: str) -> Path:
        return self.candidates_dir / _safe_component(candidate_id, label="candidate_id")

    def revision_dir(self, candidate_id: str, revision: int) -> Path:
        return self.candidate_dir(candidate_id) / f"r{max(1, int(revision)):04d}"

    def reset_candidate_dir(self, candidate_id: str) -> Path:
        """Compatibility helper that creates a fresh immutable ``r0001``.

        It deliberately refuses an existing candidate and never performs the
        destructive v1 "reset" behavior.
        """

        path = self.candidate_dir(candidate_id)
        path.mkdir(parents=True, exist_ok=True)
        if any(path.iterdir()):
            raise FileExistsError(f"candidate evidence already exists: {candidate_id}")
        revision = path / "r0001"
        revision.mkdir()
        return revision

    def create_revision_dir(self, candidate_id: str, revision: int) -> Path:
        path = self.revision_dir(candidate_id, revision)
        path.mkdir(parents=True, exist_ok=False)
        return path

    def latest_revision(self, candidate_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT revision FROM candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return int(row["revision"]) if row is not None else 0

    @staticmethod
    def _candidate_descriptor(candidate: LearningCandidate) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "project_id": candidate.project_id,
            "run_id": candidate.run_id,
            "thread_id": candidate.thread_id,
            "action": candidate.action,
            "route": candidate.route,
            "group": candidate.group,
            "name": candidate.name,
            "rationale": candidate.rationale,
            "evidence_ids": list(candidate.evidence_ids),
            "revision": max(1, int(candidate.revision or 1)),
            "base_target_hash": candidate.base_target_hash,
            "bundle_hash": candidate.bundle_hash,
            "created_at": candidate.created_at,
        }

    def write_candidate(self, candidate: LearningCandidate) -> Path:
        """Persist immutable revision identity plus mutable lifecycle status."""

        candidate.status = normalize_candidate_status(candidate.status)
        candidate.updated_at = utc_now()
        if not candidate.created_at:
            candidate.created_at = candidate.updated_at
        revision = max(1, int(candidate.revision or 1))
        candidate.revision = revision
        revision_root = self.revision_dir(candidate.candidate_id, revision)
        revision_root.mkdir(parents=True, exist_ok=True)
        descriptor = revision_root / "candidate.json"
        _write_new_json(descriptor, self._candidate_descriptor(candidate))
        if candidate.validation:
            validation_path = revision_root / "validation.json"
            _write_new_json(validation_path, candidate.validation)
        if candidate.review and "recommendation" in candidate.review:
            review_path = revision_root / "review.json"
            _write_new_json(review_path, candidate.review)

        target = (
            {"path": "/memories/AGENTS.md"}
            if candidate.action == "memory"
            else {"group": candidate.group, "name": candidate.name}
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO candidates(
                    candidate_id, route, target_json, evidence_ids_json, revision,
                    bundle_hash, base_target_hash, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    route=excluded.route,
                    target_json=excluded.target_json,
                    evidence_ids_json=excluded.evidence_ids_json,
                    revision=excluded.revision,
                    bundle_hash=excluded.bundle_hash,
                    base_target_hash=excluded.base_target_hash,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (
                    candidate.candidate_id,
                    candidate.route,
                    _json_dumps(target),
                    _json_dumps(list(candidate.evidence_ids)),
                    revision,
                    candidate.bundle_hash,
                    candidate.base_target_hash,
                    candidate.status,
                    candidate.created_at,
                    candidate.updated_at,
                ),
            )
        return descriptor

    def update_candidate_status(
        self,
        candidate_id: str,
        status: str,
    ) -> LearningCandidate:
        normalized_status = normalize_candidate_status(status)
        updates = ["status = ?", "updated_at = ?"]
        params: list[Any] = [normalized_status, utc_now()]
        params.append(candidate_id)
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE candidates SET {', '.join(updates)} WHERE candidate_id = ?",
                tuple(params),
            )
        if int(cursor.rowcount or 0) != 1:
            raise ValueError(f"candidate not found: {candidate_id}")
        candidate = self.read_candidate(candidate_id)
        if candidate is None:
            raise RuntimeError(f"candidate disappeared: {candidate_id}")
        return candidate

    def write_revision_json(
        self,
        candidate_id: str,
        revision: int,
        name: str,
        value: dict[str, Any],
    ) -> Path:
        if name not in {
            "proposal.json",
            "review.json",
            "validation.json",
        }:
            raise ValueError(f"unsupported revision artifact: {name}")
        path = self.revision_dir(candidate_id, revision) / name
        _write_new_json(path, value)
        return path

    def read_candidate(self, candidate_id: str) -> LearningCandidate | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        try:
            target = json.loads(data.pop("target_json") or "{}")
        except Exception:
            target = {}
        try:
            evidence_ids = json.loads(data.pop("evidence_ids_json") or "[]")
        except Exception:
            evidence_ids = []
        revision = max(1, int(data.get("revision") or 1))
        descriptor = _read_json(self.revision_dir(candidate_id, revision) / "candidate.json")
        action = str(descriptor.get("action") or ("memory" if target.get("path") else "skill"))
        candidate = LearningCandidate(
            candidate_id=candidate_id,
            project_id=str(descriptor.get("project_id") or self.project_id),
            run_id=str(descriptor.get("run_id") or ""),
            thread_id=str(descriptor.get("thread_id") or ""),
            action=action,  # type: ignore[arg-type]
            status=normalize_candidate_status(data.get("status")),
            route=str(data.get("route") or descriptor.get("route") or "amend_existing_skill"),  # type: ignore[arg-type]
            group=str(target.get("group") or descriptor.get("group") or ""),
            name=str(target.get("name") or descriptor.get("name") or ""),
            rationale=str(descriptor.get("rationale") or ""),
            evidence_ids=[str(item) for item in evidence_ids if str(item).strip()],
            revision=revision,
            base_target_hash=str(data.get("base_target_hash") or descriptor.get("base_target_hash") or ""),
            bundle_hash=str(data.get("bundle_hash") or descriptor.get("bundle_hash") or ""),
            created_at=str(data.get("created_at") or descriptor.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )
        root = self.revision_dir(candidate_id, revision)
        candidate.review = _read_json(root / "review.json")
        candidate.validation = _read_json(root / "validation.json")
        return candidate

    def read_candidate_revision(
        self,
        candidate_id: str,
        revision: int,
    ) -> LearningCandidate | None:
        """Read one immutable revision without consulting the mutable latest row."""

        resolved_revision = max(1, int(revision))
        root = self.revision_dir(candidate_id, resolved_revision)
        descriptor = _read_json(root / "candidate.json")
        if not descriptor:
            return None
        action = str(descriptor.get("action") or "").strip()
        route = str(descriptor.get("route") or "").strip()
        if action not in {"memory", "skill"} or not route:
            return None
        candidate = LearningCandidate(
            candidate_id=str(descriptor.get("candidate_id") or candidate_id),
            project_id=str(descriptor.get("project_id") or self.project_id),
            run_id=str(descriptor.get("run_id") or ""),
            thread_id=str(descriptor.get("thread_id") or ""),
            action=action,  # type: ignore[arg-type]
            status="stable",
            route=route,  # type: ignore[arg-type]
            group=str(descriptor.get("group") or ""),
            name=str(descriptor.get("name") or ""),
            rationale=str(descriptor.get("rationale") or ""),
            evidence_ids=[
                str(item)
                for item in list(descriptor.get("evidence_ids") or [])
                if str(item).strip()
            ],
            revision=resolved_revision,
            base_target_hash=str(descriptor.get("base_target_hash") or ""),
            bundle_hash=str(descriptor.get("bundle_hash") or ""),
            created_at=str(descriptor.get("created_at") or ""),
        )
        candidate.review = _read_json(root / "review.json")
        candidate.validation = _read_json(root / "validation.json")
        return candidate

    def list_candidates(
        self,
        *,
        status: str = "",
        limit: int = 100,
        before: str = "",
    ) -> list[LearningCandidate]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(normalize_candidate_status(status))
        if before:
            with self._connect() as conn:
                cursor_row = conn.execute(
                    "SELECT updated_at FROM candidates WHERE candidate_id = ?",
                    (before,),
                ).fetchone()
            if cursor_row is not None:
                clauses.append(
                    "(updated_at < ? OR (updated_at = ? AND candidate_id < ?))"
                )
                params.extend(
                    [cursor_row["updated_at"], cursor_row["updated_at"], before]
                )
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(max(1, min(500, int(limit))))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT candidate_id FROM candidates{where} "
                "ORDER BY updated_at DESC, candidate_id DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        candidates = [self.read_candidate(str(row["candidate_id"])) for row in rows]
        return [item for item in candidates if item is not None]

    def candidate_status_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS total FROM candidates GROUP BY status"
            ).fetchall()
        return {
            str(row["status"]): int(row["total"])
            for row in rows
            if str(row["status"] or "").strip()
        }

    def observation_status_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS total FROM observations GROUP BY status"
            ).fetchall()
        return {
            str(row["status"]): int(row["total"])
            for row in rows
            if str(row["status"] or "").strip()
        }

    def write_validation_report(self, report: ValidationReport, *, revision: int | None = None) -> Path:
        target_revision = revision or self.latest_revision(report.candidate_id) or 1
        return self.write_revision_json(
            report.candidate_id,
            target_revision,
            "validation.json",
            report.to_dict(),
        )

    # -- actual-use telemetry -------------------------------------------

    def upsert_skill_run(self, record: SkillRun) -> SkillRun:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO skill_runs(
                    run_id, skill_name, skill_version, presented, read,
                    helper_used, outcome, false_activation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, skill_name, skill_version) DO UPDATE SET
                    presented = MAX(skill_runs.presented, excluded.presented),
                    read = MAX(skill_runs.read, excluded.read),
                    helper_used = MAX(skill_runs.helper_used, excluded.helper_used),
                    outcome = CASE
                        WHEN excluded.outcome != 'unknown' THEN excluded.outcome
                        ELSE skill_runs.outcome
                    END,
                    false_activation = MAX(skill_runs.false_activation, excluded.false_activation)
                """,
                (
                    record.run_id,
                    record.skill_name,
                    record.skill_version,
                    int(record.presented),
                    int(record.read),
                    int(record.helper_used),
                    record.outcome or "unknown",
                    int(record.false_activation),
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM skill_runs
                WHERE run_id = ? AND skill_name = ? AND skill_version = ?
                """,
                (record.run_id, record.skill_name, record.skill_version),
            ).fetchone()
        if row is None:
            raise RuntimeError("skill telemetry row disappeared")
        return SkillRun.from_dict(
            {
                **dict(row),
                "presented": bool(row["presented"]),
                "read": bool(row["read"]),
                "helper_used": bool(row["helper_used"]),
                "false_activation": bool(row["false_activation"]),
            }
        )

    def list_skill_runs(
        self,
        *,
        skill_name: str = "",
        run_id: str = "",
        limit: int = 500,
    ) -> list[SkillRun]:
        clauses: list[str] = []
        params: list[Any] = []
        if skill_name:
            clauses.append("skill_name = ?")
            params.append(skill_name)
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(max(1, min(2_000, int(limit))))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM skill_runs{where} ORDER BY run_id DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [
            SkillRun.from_dict(
                {
                    **dict(row),
                    "presented": bool(row["presented"]),
                    "read": bool(row["read"]),
                    "helper_used": bool(row["helper_used"]),
                    "false_activation": bool(row["false_activation"]),
                }
            )
            for row in rows
        ]

    # -- stable/canary pointers -----------------------------------------

    def read_active_skills(self) -> dict[str, Any]:
        value = _read_json(self.active_skills_path)
        skills = value.get("skills") if isinstance(value.get("skills"), dict) else {}
        return {"skills": dict(skills)}

    def write_active_skills(self, value: dict[str, Any]) -> None:
        skills = value.get("skills") if isinstance(value.get("skills"), dict) else {}
        _atomic_write_json(self.active_skills_path, {"skills": dict(skills)})

    # -- locks, audit, and memory ---------------------------------------

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
                row = conn.execute(
                    "SELECT value FROM store WHERE prefix = ? AND key = ?",
                    (prefix, "/AGENTS.md"),
                ).fetchone()
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
            row = conn.execute(
                "SELECT value FROM store WHERE prefix = ? AND key = ?",
                (prefix, "/AGENTS.md"),
            ).fetchone()
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
                    ON CONFLICT(prefix, key) DO UPDATE
                    SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
                    """,
                    (prefix, "/AGENTS.md", value_text),
                )
            else:
                conn.execute("DELETE FROM store WHERE prefix = ? AND key = ?", (prefix, "/AGENTS.md"))
                conn.execute(
                    "INSERT INTO store(prefix, key, value) VALUES (?, ?, ?)",
                    (prefix, "/AGENTS.md", value_text),
                )
            conn.commit()
        return True, hash_text(str(new_text or ""))

__all__ = [
    "ACTIVE_SKILLS_FILE",
    "MEMORY_STORE_FILE",
    "SELF_EVOLUTION_DIR",
    "SelfEvolutionStore",
    "hash_text",
    "hash_tree",
    "stable_id",
    "utc_now",
]
