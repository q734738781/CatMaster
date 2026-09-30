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


def _write_new_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = str(value or "")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.read_text(encoding="utf-8", errors="replace") != payload:
            raise FileExistsError(
                f"immutable self-evolution artifact already exists: {path}"
            )
        return
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"cannot read self-evolution JSON {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise TypeError(
            f"self-evolution JSON {path} must contain an object, found "
            f"{type(value).__name__}"
        )
    return value


def _json_object_or_diagnostic(raw: Any, *, label: str) -> dict[str, Any]:
    raw_text = "{}" if raw is None else str(raw)
    try:
        value = json.loads(raw_text)
    except Exception as exc:
        return {
            "_read_error_type": type(exc).__name__,
            "_read_error": f"cannot decode {label}: {exc}",
        }
    if not isinstance(value, dict):
        return {
            "_read_error_type": "TypeError",
            "_read_error": (
                f"{label} must contain a JSON object, found {type(value).__name__}"
            ),
        }
    return value


def _safe_component(value: str, *, label: str, allow_colon: bool = False) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."}:
        raise ValueError(f"{label} is required")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" + (":" if allow_colon else "")
    if any(char not in allowed for char in text):
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

    @property
    def target_locks_dir(self) -> Path:
        return self.root / "target_locks"

    @property
    def job_evidence_dir(self) -> Path:
        return self.root / "job_evidence"

    def write_job_evidence(
        self,
        job_id: str,
        name: str,
        text: str,
    ) -> Path:
        resolved_job = _safe_component(job_id, label="job_id")
        resolved_name = _safe_component(name, label="evidence name")
        path = self.job_evidence_dir / resolved_job / resolved_name
        _write_new_text(path, text)
        return path

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
                    episode_id TEXT NOT NULL DEFAULT '',
                    selected_item_ref TEXT NOT NULL DEFAULT '',
                    trigger_kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    candidate_id TEXT NOT NULL DEFAULT '',
                    outcome_json TEXT NOT NULL DEFAULT '{}',
                    predecessor_job_id TEXT NOT NULL DEFAULT '',
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
                ("outcome_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("predecessor_job_id", "TEXT NOT NULL DEFAULT ''"),
                ("episode_id", "TEXT NOT NULL DEFAULT ''"),
                ("selected_item_ref", "TEXT NOT NULL DEFAULT ''"),
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
                    job_id TEXT NOT NULL DEFAULT '',
                    episode_id TEXT NOT NULL DEFAULT '',
                    item_ref TEXT NOT NULL DEFAULT '',
                    signal_kind TEXT NOT NULL,
                    target TEXT NOT NULL DEFAULT '',
                    resolved_target TEXT NOT NULL DEFAULT '',
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
            if "resolved_target" not in observation_columns:
                conn.execute(
                    "ALTER TABLE observations ADD COLUMN "
                    "resolved_target TEXT NOT NULL DEFAULT ''"
                )
            for column in ("job_id", "episode_id", "item_ref"):
                if column not in observation_columns:
                    conn.execute(
                        f"ALTER TABLE observations ADD COLUMN {column} TEXT NOT NULL DEFAULT ''"
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
                "CREATE INDEX IF NOT EXISTS observations_resolved_target_idx "
                "ON observations(resolved_target, created_at, observation_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS observations_item_idx "
                "ON observations(item_ref, observation_id)"
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
                    partial INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (run_id, skill_name, skill_version)
                )
                """
            )
            skill_run_columns = self._columns(conn, "skill_runs")
            for column in ("partial",):
                if column not in skill_run_columns:
                    conn.execute(
                        f"ALTER TABLE skill_runs ADD COLUMN {column} "
                        "INTEGER NOT NULL DEFAULT 0"
                    )
            conn.execute("CREATE INDEX IF NOT EXISTS skill_runs_skill_idx ON skill_runs(skill_name, skill_version)")

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> SelfEvolutionJob:
        data = dict(row)
        data["payload"] = _json_object_or_diagnostic(
            data.pop("payload_json"),
            label=f"job {data.get('job_id') or '<unknown>'} payload_json",
        )
        data["outcome"] = _json_object_or_diagnostic(
            data.pop("outcome_json"),
            label=f"job {data.get('job_id') or '<unknown>'} outcome_json",
        )
        return SelfEvolutionJob.from_dict(data)

    def enqueue_job(
        self,
        *,
        trigger_kind: str,
        run_id: str,
        run_dir: Path | str,
        thread_id: str = "",
        episode_id: str = "",
        selected_item_ref: str = "",
        payload: dict[str, Any] | None = None,
        model_config: str = "",
        predecessor_job_id: str = "",
    ) -> SelfEvolutionJob:
        # Local DBOS identities are <thread key>:<turn key>; retain that identity.
        run_id = _safe_component(run_id, label="run_id", allow_colon=True)
        trigger = str(trigger_kind or "post_run").strip() or "post_run"
        episode = str(episode_id or "").strip()
        selected_item = str(selected_item_ref or "").strip()
        predecessor = str(predecessor_job_id or "").strip()
        payload_value = dict(payload or {})
        if trigger == "post_run" and episode:
            trigger_identity = f"episode:{episode}\x1eitem:{selected_item}"
        elif trigger == "selected_retry" and predecessor:
            trigger_identity = (
                f"predecessor:{predecessor}\x1eitem:{selected_item}"
            )
        elif trigger == "selected_retry" and episode:
            trigger_identity = f"episode:{episode}\x1eitem:{selected_item}"
        else:
            trigger_identity = _json_dumps(payload_value)
        job_id = "sej_" + stable_id(
            self.project_id,
            trigger,
            episode or run_id,
            trigger_identity,
            length=28,
        )
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO jobs(
                    job_id, project_id, run_id, run_dir, thread_id, episode_id,
                    selected_item_ref, trigger_kind,
                    status, attempt_count, candidate_id, outcome_json,
                    predecessor_job_id, model_config, payload_json,
                    error, owner, lease_until, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0, '', '{}', ?, ?, ?, '', '', '', ?, ?)
                """,
                (
                    job_id,
                    self.project_id,
                    run_id,
                    str(Path(run_dir).expanduser().resolve()),
                    str(thread_id or "").strip(),
                    episode,
                    selected_item,
                    trigger,
                    predecessor,
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
        outcome: dict[str, Any] | None = None,
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
                SET status = ?, candidate_id = ?, outcome_json = ?, error = ?, owner = '',
                    lease_until = '', updated_at = ?
                WHERE job_id = ? AND status = 'running' AND owner = ?
                """,
                (
                    status,
                    str(candidate_id or ""),
                    _json_dumps(dict(outcome if outcome is not None else job.outcome)),
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

    def unresolved_job_error_count(self, *, project_id: str = "") -> int:
        """Count terminal error leaves that do not yet have a retry successor."""

        clauses = ["j.status IN ('error', 'recovery_review')"]
        params: list[Any] = []
        if project_id:
            clauses.append("j.project_id = ?")
            params.append(str(project_id))
        clauses.append(
            "NOT EXISTS ("
            "SELECT 1 FROM jobs AS retry "
            "WHERE retry.predecessor_job_id = j.job_id"
            ")"
        )
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total FROM jobs AS j WHERE "
                + " AND ".join(clauses),
                tuple(params),
            ).fetchone()
        return int(row["total"] or 0) if row is not None else 0

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
        if not observation.created_at:
            observation.created_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO observations(
                    observation_id, run_id, thread_id, job_id, episode_id,
                    item_ref, signal_kind, target, resolved_target, claim,
                    evidence_refs_json, outcome_ref, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(observation_id) DO NOTHING
                """,
                (
                    observation.observation_id,
                    observation.run_id,
                    observation.thread_id,
                    observation.job_id,
                    observation.episode_id,
                    observation.item_ref,
                    observation.signal_kind,
                    observation.target.strip(),
                    observation.resolved_target.strip(),
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
            evidence_refs = json.loads(data.pop("evidence_refs_json") or "[]")
        except Exception as exc:
            raise ValueError(
                "cannot decode evidence_refs_json for observation "
                f"{data.get('observation_id') or '<unknown>'}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(evidence_refs, list):
            raise TypeError(
                "evidence_refs_json for observation "
                f"{data.get('observation_id') or '<unknown>'} must contain a JSON array"
            )
        data["evidence_refs"] = evidence_refs
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
            clauses.append("COALESCE(NULLIF(resolved_target, ''), target) = ?")
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

    def list_observations_for_target(
        self,
        target: str,
        *,
        status: str = "",
    ) -> list[Observation]:
        """Return signals for one exact semantic target.

        Signals are intentionally rare and already compressed by the model, so
        this query does not impose a wording, thread-count, or arbitrary top-k
        cutoff.
        """

        resolved = str(target or "").strip()
        if not resolved:
            return []
        resolved_status = str(status or "").strip()
        if resolved_status and resolved_status not in {"open", "consolidated"}:
            raise ValueError(f"unsupported observation status: {resolved_status}")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM observations "
                "WHERE COALESCE(NULLIF(resolved_target, ''), target) = ?"
                + (" AND status = ?" if resolved_status else "")
                + " ORDER BY created_at ASC, observation_id ASC",
                (resolved, resolved_status) if resolved_status else (resolved,),
            ).fetchall()
        return [self._observation_from_row(row) for row in rows]

    def read_candidate_for_target(self, target: str) -> LearningCandidate | None:
        """Return the latest candidate chain already bound to one exact target."""

        resolved = str(target or "").strip()
        if not resolved or resolved.startswith("memory/"):
            return None
        group, separator, name = resolved.partition("/")
        if not separator or not group or not name:
            return None
        encoded_target = _json_dumps({"group": group, "name": name})
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT candidate_id FROM candidates
                WHERE target_json = ?
                ORDER BY updated_at DESC, candidate_id DESC
                LIMIT 1
                """,
                (encoded_target,),
            ).fetchone()
        if row is None:
            return None
        return self.read_candidate(str(row["candidate_id"]))

    def list_observation_targets(self) -> list[str]:
        """List the exact semantic owners used for later consolidation."""

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT COALESCE(NULLIF(resolved_target, ''), target) AS target
                FROM observations
                WHERE TRIM(COALESCE(NULLIF(resolved_target, ''), target)) != ''
                ORDER BY target
                """
            ).fetchall()
        return [
            str(row["target"]).strip()
            for row in rows
            if str(row["target"] or "").strip()
        ]

    def set_observation_resolved_target(
        self,
        observation_ids: list[str],
        target: str,
    ) -> int:
        """Bind observations to the final model-selected semantic owner."""

        ids = [str(item).strip() for item in observation_ids if str(item).strip()]
        resolved = str(target or "").strip()
        if not ids or not resolved:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE observations SET resolved_target = ? "
                f"WHERE observation_id IN ({placeholders})",
                (resolved, *ids),
            )
        return max(0, int(cursor.rowcount or 0))

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
            "episode_id": candidate.episode_id,
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
            "prior_review.json",
            "validation.json",
            "proposer_attempts.json",
            "validation_feedback.json",
        }:
            raise ValueError(f"unsupported revision artifact: {name}")
        path = self.revision_dir(candidate_id, revision) / name
        _write_new_json(path, value)
        return path

    def write_revision_text(
        self,
        candidate_id: str,
        revision: int,
        name: str,
        value: str,
    ) -> Path:
        if name not in {
            "proposer_response.txt",
            "reviewer_response.txt",
        }:
            raise ValueError(f"unsupported revision artifact: {name}")
        path = self.revision_dir(candidate_id, revision) / name
        _write_new_text(path, value)
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
        data.pop("target_json", None)
        data.pop("evidence_ids_json", None)
        revision = max(1, int(data.get("revision") or 1))
        descriptor_path = self.revision_dir(candidate_id, revision) / "candidate.json"
        if not descriptor_path.is_file():
            raise FileNotFoundError(
                f"candidate index {candidate_id} references missing descriptor {descriptor_path}"
            )
        descriptor = _read_json(descriptor_path)
        action = str(descriptor.get("action") or "").strip()
        if action not in {"memory", "skill"}:
            raise ValueError(
                f"candidate descriptor {descriptor_path} has unsupported action {action!r}"
            )
        candidate = LearningCandidate(
            candidate_id=candidate_id,
            project_id=str(descriptor.get("project_id") or self.project_id),
            run_id=str(descriptor.get("run_id") or ""),
            thread_id=str(descriptor.get("thread_id") or ""),
            action=action,  # type: ignore[arg-type]
            episode_id=str(descriptor.get("episode_id") or ""),
            status=normalize_candidate_status(data.get("status")),
            route=str(descriptor.get("route") or data.get("route") or "amend_existing_skill"),  # type: ignore[arg-type]
            group=str(descriptor.get("group") or ""),
            name=str(descriptor.get("name") or ""),
            rationale=str(descriptor.get("rationale") or ""),
            evidence_ids=[
                str(item)
                for item in list(descriptor.get("evidence_ids") or [])
                if str(item).strip()
            ],
            revision=revision,
            base_target_hash=str(descriptor.get("base_target_hash") or ""),
            bundle_hash=str(descriptor.get("bundle_hash") or ""),
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
            raise ValueError(
                f"candidate descriptor {root / 'candidate.json'} has invalid "
                f"action/route: {action!r}/{route!r}"
            )
        candidate = LearningCandidate(
            candidate_id=str(descriptor.get("candidate_id") or candidate_id),
            project_id=str(descriptor.get("project_id") or self.project_id),
            run_id=str(descriptor.get("run_id") or ""),
            thread_id=str(descriptor.get("thread_id") or ""),
            action=action,  # type: ignore[arg-type]
            episode_id=str(descriptor.get("episode_id") or ""),
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
                    helper_used, outcome, false_activation, partial
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, skill_name, skill_version) DO UPDATE SET
                    presented = MAX(skill_runs.presented, excluded.presented),
                    read = MAX(skill_runs.read, excluded.read),
                    helper_used = MAX(skill_runs.helper_used, excluded.helper_used),
                    outcome = CASE
                        WHEN excluded.outcome != 'unknown' THEN excluded.outcome
                        ELSE skill_runs.outcome
                    END,
                    false_activation = MAX(skill_runs.false_activation, excluded.false_activation),
                    partial = MAX(skill_runs.partial, excluded.partial)
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
                    int(record.partial),
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
                "partial": bool(row["partial"]),
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
                    "partial": bool(row["partial"]),
                }
            )
            for row in rows
        ]

    # -- stable/canary pointers -----------------------------------------

    def read_active_skills(self) -> dict[str, Any]:
        value = _read_json(self.active_skills_path)
        if not value:
            return {"skills": {}}
        skills = value.get("skills")
        if not isinstance(skills, dict):
            raise TypeError(
                f"active skill pointer file {self.active_skills_path} must contain "
                "an object-valued 'skills' field"
            )
        payload: dict[str, Any] = {"skills": dict(skills)}
        mode = str(value.get("mode") or "").strip()
        if mode:
            payload["mode"] = mode
        return payload

    def write_active_skills(self, value: dict[str, Any]) -> None:
        skills = value.get("skills")
        if not isinstance(skills, dict):
            raise TypeError("active skill pointer payload must contain an object-valued 'skills' field")
        payload: dict[str, Any] = {"skills": dict(skills)}
        mode = str(value.get("mode") or "").strip()
        if mode:
            if mode not in {"off", "observe", "auto"}:
                raise ValueError("active skill workspace mode must be off, observe, or auto")
            payload["mode"] = mode
        _atomic_write_json(self.active_skills_path, payload)

    # -- locks, audit, and memory ---------------------------------------

    @contextmanager
    def candidate_lock(self, candidate_id: str) -> Iterator[None]:
        """Serialize materialization for one exact semantic target candidate."""

        resolved = _safe_component(candidate_id, label="candidate_id")
        self.target_locks_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.target_locks_dir / f"{resolved}.lock"
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

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
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"cannot read workspace memory database {path}: {type(exc).__name__}: {exc}"
            ) from exc
        return self._decode_memory_value(row[0]) if row else ""

    @staticmethod
    def _decode_memory_value(raw: Any) -> str:
        try:
            payload = json.loads(raw)
        except Exception as first_exc:
            try:
                payload = json.loads(bytes(raw).decode("utf-8"))
            except Exception as second_exc:
                raise ValueError(
                    "cannot decode workspace memory value: "
                    f"{type(first_exc).__name__}: {first_exc}; "
                    f"bytes fallback {type(second_exc).__name__}: {second_exc}"
                ) from second_exc
        if not isinstance(payload, dict):
            raise TypeError(
                "workspace memory value must decode to a JSON object, found "
                f"{type(payload).__name__}"
            )
        content = payload.get("content")
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
