from __future__ import annotations

import json
import logging
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import quote

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from catmaster.runtime.observability_store import (
    LEGACY_TRACE_SOURCES,
    OBSERVABILITY_DB_NAME,
)

from .trace import (
    TurnTrace,
    _TRAJECTORY_EVENT_NAMES,
    collect_turn_trace,
    semantic_event_name,
    semantic_event_payload,
)


logger = logging.getLogger(__name__)


_EVENT_HANDLE_RE = re.compile(r"^run:([^#]+)#event:([1-9][0-9]*)(\?view=raw)?$")
_RESULT_MAX_BYTES = 100_000
_READ_MAX_CHARS = 100_000

_TRACE_QUERY_SCHEMA: dict[str, tuple[str, ...]] = {
    "trajectory_runs": (
        "run_id",
        "thread_id",
        "entrypoint",
        "status",
        "user_prompt",
        "final_answer",
        "summary",
        "task_outcome",
        "outcome_ref",
        "explicit_correction",
        "resume_guidance",
    ),
    "trajectory_events": (
        "id",
        "run_id",
        "handle",
        "ts",
        "name",
        "agent_name",
        "node",
        "callback_run_id",
        "parent_callback_run_id",
        "model",
        "tool",
        "status",
        "duration_ms",
        "payload_json",
        "payload_chars",
    ),
    "trajectory_raw_events": (
        "id",
        "run_id",
        "handle",
        "ts",
        "name",
        "agent_name",
        "node",
        "callback_run_id",
        "parent_callback_run_id",
        "model",
        "tool",
        "status",
        "duration_ms",
        "payload_json",
        "payload_chars",
    ),
}
_HISTORY_QUERY_SCHEMA: dict[str, tuple[str, ...]] = {
    "evolution_observations": (
        "observation_id",
        "run_id",
        "run_ref",
        "thread_id",
        "job_id",
        "episode_id",
        "item_ref",
        "signal_kind",
        "target",
        "resolved_target",
        "claim",
        "evidence_refs_json",
        "outcome_ref",
        "status",
        "created_at",
    ),
    "evolution_candidate_revisions": (
        "candidate_id",
        "revision",
        "target",
        "route",
        "run_id",
        "run_ref",
        "thread_id",
        "episode_id",
        "status",
        "evidence_ids_json",
        "proposal_json",
        "review_json",
        "validation_json",
        "created_at",
        "updated_at",
    ),
    "evolution_jobs": (
        "job_id",
        "project_id",
        "run_id",
        "run_ref",
        "thread_id",
        "episode_id",
        "selected_item_ref",
        "trigger_kind",
        "status",
        "attempt_count",
        "candidate_id",
        "outcome_json",
        "predecessor_job_id",
        "error",
        "created_at",
        "updated_at",
    ),
    "evolution_skill_runs": (
        "run_id",
        "run_ref",
        "skill_name",
        "skill_version",
        "presented",
        "read",
        "helper_used",
        "outcome",
        "false_activation",
        "partial",
    ),
}


def _schema_text(schema: Mapping[str, tuple[str, ...]]) -> str:
    return "; ".join(
        f"{table}({', '.join(columns)})" for table, columns in schema.items()
    )


class EvolutionQueryError(ValueError):
    """A recoverable input/read failure with an operation-specific next step."""

    def __init__(self, message: str, *, recovery: str) -> None:
        super().__init__(message)
        self.recovery = recovery


@dataclass(frozen=True)
class _FileQueryResult:
    content: dict[str, Any]
    artifact: dict[str, Any]


def _agent_tool_error(
    error: Any,
    *,
    operation: str = "tool_call",
) -> str:
    message = str(error).strip() or type(error).__name__
    payload: dict[str, Any] = {
        "ok": False,
        "operation": str(operation or "tool_call"),
        "error": message,
    }
    if isinstance(error, EvolutionQueryError):
        payload["recovery"] = error.recovery
    elif isinstance(error, (ValueError, sqlite3.DatabaseError)):
        payload["recovery"] = "Correct this input using the fields in the tool description, then retry."
    else:
        payload["recovery"] = (
            "Use the exact error to choose whether to retry, "
            "inspect another source, or report the runtime failure."
        )
    return _json_dump(payload)


def _agent_tool_result(
    operation: Any,
    *,
    operation_name: str = "tool_call",
    with_artifact: bool = False,
) -> Any:
    """Return tool failures as observations so the ReAct loop can self-correct."""

    try:
        result = operation()
    except Exception as exc:
        if not isinstance(exc, (ValueError, sqlite3.DatabaseError)):
            logger.exception("Self-evolution agent tool failed during %s", operation_name)
        # Native ToolException handling keeps the agent running and sets the
        # resulting ToolMessage.status to error (including callback records).
        raise ToolException(_agent_tool_error(exc, operation=operation_name)) from exc
    artifact = None
    if isinstance(result, _FileQueryResult):
        artifact, result = result.artifact, result.content
    content = _json_dump(
        {"ok": True, **result} if isinstance(result, dict) else {"ok": True, "result": result}
    )
    return (content, artifact) if with_artifact else content


class QueryEvolutionTraceSQLInput(BaseModel):
    """[self-evolution/trace] Query one exact authorized run through read-only SQLite views."""

    model_config = ConfigDict(extra="forbid")

    sql: str = Field(
        ...,
        min_length=1,
        description=(
            "One read-only SELECT or WITH query over the tables in the tool description. "
            "Use JSON1, joins, aggregation, ordering, and LIMIT/OFFSET as needed. "
            "Select handle to obtain references ready for reading or citation; "
            "select payload_json to read multiple event bodies in the same query."
        ),
    )
    run_ref: str = Field(
        "",
        description=(
            "Omit or leave empty for the current job anchor run. For another authorized "
            "run, copy its complete run:<run_id> handle from query_evolution_history_sql, "
            "including suffixes such as :initial or :completion-... . For example, "
            "run:abc:initial; neither abc:initial nor run:abc selects that run. "
            "An event handle with #event:... belongs in read_evolution_event instead."
        ),
    )


class QueryEvolutionHistorySQLInput(BaseModel):
    """[self-evolution/history] Query the authorized workspace self-evolution history."""

    model_config = ConfigDict(extra="forbid")

    sql: str = Field(
        ...,
        min_length=1,
        description=(
            "One read-only SELECT or WITH query over the tables in the tool description. "
            "Use ordinary ordering and "
            "LIMIT/OFFSET pagination."
        ),
    )


class ReadEvolutionEventInput(BaseModel):
    """[self-evolution/trace] Continue reading an event returned by a trace query.

    Copy the query's handle unchanged; the default reads that reference's body.
    SQL can read multiple bodies at once, so use this tool for a selected field
    or a long body. Continue with next_offset until it is null. A returned SQL
    result_path can also be opened with the ordinary file tools.
    """

    model_config = ConfigDict(extra="forbid")

    handle: str = Field(
        ...,
        min_length=12,
        description="Copy an actual handle returned by a trace query or supplied evidence. Wait for that query before reading its results.",
    )
    field: str = Field(
        "payload_json",
        description=(
            "Omit to read the referenced event body. May also be an event column "
            "such as name, status, or tool, or a JSON path beginning with $ into "
            "that body. raw_payload_json explicitly reads the original recorded body."
        ),
    )
    offset: int = Field(0, ge=0, description="Zero-based character offset in the exact field text.")
    limit: int = Field(
        16_000,
        ge=1,
        le=_READ_MAX_CHARS,
        description="Maximum characters to return in this segment.",
    )


@dataclass(frozen=True)
class EvolutionTraceRun:
    run_id: str
    run_dir: Path
    trace: TurnTrace


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _semantic_payload_json(name: str, payload_json: str | None) -> str:
    raw_text = "" if payload_json is None else str(payload_json)
    try:
        raw = json.loads(raw_text)
    except Exception as exc:
        return _json_dump(
            {
                "_payload_parse_error": {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                        "raw_payload": raw_text,
                }
            }
        )
    payload = raw if isinstance(raw, dict) else {"value": raw}
    return _json_dump(semantic_event_payload(str(name or ""), payload))


def _bounded_query_result(
    conn: sqlite3.Connection,
    sql: str,
    *,
    label: str,
    result_writer: Callable[[str], str] | None = None,
) -> dict[str, Any] | _FileQueryResult:
    statement = str(sql or "").strip()
    if not statement:
        raise ValueError("SQL is required")
    try:
        cursor = conn.execute(statement)
        if cursor.description is None:
            raise ValueError("Only SELECT or WITH queries are allowed.")
        columns = [str(item[0]) for item in cursor.description]
        rows: list[dict[str, Any]] = []
        estimated_bytes = len(
            _json_dump({"columns": columns, "rows": [], "row_count": 0}).encode(
                "utf-8"
            )
        )
        for raw_row in cursor:
            row = dict(raw_row)
            estimated_bytes += len(_json_dump(row).encode("utf-8")) + 1
            if estimated_bytes > _RESULT_MAX_BYTES and result_writer is None:
                raise EvolutionQueryError(
                    f"{label} query result is above the {_RESULT_MAX_BYTES}-byte "
                    "result boundary; no rows were returned or truncated.",
                    recovery="Use WHERE or LIMIT/OFFSET for fewer rows. For a large event body, query handle and use read_evolution_event with next_offset.",
                )
            rows.append(row)
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"{label} query rejected: {exc}") from exc
    result = {"columns": columns, "rows": rows, "row_count": len(rows)}
    encoded = _json_dump(result).encode("utf-8")
    if len(encoded) > _RESULT_MAX_BYTES:
        if result_writer is not None:
            path = result_writer(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return _FileQueryResult(content={
                "columns": columns,
                "row_count": len(rows),
                "result_path": path,
                "read_with": "Read the complete JSON result with read_file; grep can locate relevant rows. Event handles also support field continuation with read_evolution_event.",
            }, artifact=result)
        raise EvolutionQueryError(
            f"{label} query result is above the {_RESULT_MAX_BYTES}-byte result boundary; no rows were returned or truncated.",
            recovery="Use WHERE or LIMIT/OFFSET. Read a large event field through its handle and next_offset.",
        )
    return result


class EvolutionHistoryScope:
    """Read-only catalog over the authorized workspace evolution database."""

    _VIEWS = {
        "evolution_observations",
        "evolution_candidate_revisions",
        "evolution_jobs",
        "evolution_skill_runs",
    }

    def __init__(self, *, db_path: Path | str, target: str = "") -> None:
        path = Path(db_path).expanduser().resolve()
        resolved_target = str(target or "").strip()
        if not path.is_file():
            raise ValueError("self-evolution history database is unavailable")
        self.db_path = path
        self.target = resolved_target
        self.candidates_dir = path.parent / "candidates"

    def _candidate_revision_rows(
        self,
        conn: sqlite3.Connection,
    ) -> list[tuple[Any, ...]]:
        current = {
            str(row["candidate_id"]): {
                "revision": int(row["revision"]),
                "status": str(row["status"]),
                "updated_at": str(row["updated_at"]),
            }
            for row in conn.execute(
                "SELECT candidate_id, revision, status, updated_at FROM main.candidates"
            ).fetchall()
        }
        observation_targets = {
            str(row["observation_id"]): str(
                row["resolved_target"] or row["target"]
            )
            for row in conn.execute(
                "SELECT observation_id, target, resolved_target FROM main.observations"
            ).fetchall()
        }
        rows: list[tuple[Any, ...]] = []
        for candidate_root in sorted(self.candidates_dir.glob("*")):
            if not candidate_root.is_dir():
                continue
            candidate_id = candidate_root.name
            current_row = current.get(candidate_id, {})
            for revision_root in sorted(
                candidate_root.glob("r[0-9][0-9][0-9][0-9]")
            ):
                revision = int(revision_root.name.removeprefix("r"))
                # Candidate construction creates its revision directory before
                # semantic review, then commits the immutable descriptor and DB
                # pointer together. Do not expose that active transaction as a
                # corrupt historical revision to the reviewer inspecting it.
                # A descriptor missing from an already committed revision still
                # falls through to the exact read_error row below.
                if revision > int(current_row.get("revision") or 0):
                    continue
                descriptor_path = revision_root / "candidate.json"
                try:
                    descriptor = json.loads(
                        descriptor_path.read_text(encoding="utf-8")
                    )
                except Exception as exc:
                    diagnostic = {
                        "_read_error_type": type(exc).__name__,
                        "_read_error": str(exc),
                        "_path": str(descriptor_path),
                    }
                    rows.append(
                        (
                            candidate_id,
                            revision,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "read_error",
                            "[]",
                            _json_dump(diagnostic),
                            "{}",
                            "{}",
                            "",
                            str(current_row.get("updated_at") or ""),
                        )
                    )
                    continue
                if not isinstance(descriptor, dict):
                    diagnostic = {
                        "_read_error_type": "TypeError",
                        "_read_error": "candidate.json must contain a JSON object",
                        "_path": str(descriptor_path),
                    }
                    rows.append(
                        (
                            candidate_id,
                            revision,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "read_error",
                            "[]",
                            _json_dump(diagnostic),
                            "{}",
                            "{}",
                            "",
                            str(current_row.get("updated_at") or ""),
                        )
                    )
                    continue
                evidence_ids = [
                    str(item)
                    for item in list(descriptor.get("evidence_ids") or [])
                    if str(item).strip()
                ]
                if str(descriptor.get("action") or "") == "memory":
                    target = next(
                        (
                            observation_targets[item]
                            for item in evidence_ids
                            if observation_targets.get(item, "").startswith("memory/")
                        ),
                        "memory",
                    )
                else:
                    target = (
                        f"{descriptor.get('group') or ''}/"
                        f"{descriptor.get('name') or ''}"
                    ).strip("/")
                def artifact(name: str) -> dict[str, Any]:
                    try:
                        value = json.loads(
                            (revision_root / name).read_text(encoding="utf-8")
                        )
                    except Exception as exc:
                        return {
                            "_read_error_type": type(exc).__name__,
                            "_read_error": str(exc),
                            "_path": str(revision_root / name),
                        }
                    if not isinstance(value, dict):
                        return {
                            "_read_error_type": "TypeError",
                            "_read_error": "expected a JSON object",
                            "_path": str(revision_root / name),
                        }
                    return value

                created_at = str(descriptor.get("created_at") or "")
                is_current = int(current_row.get("revision") or 0) == revision
                rows.append(
                    (
                        candidate_id,
                        revision,
                        target,
                        str(descriptor.get("route") or ""),
                        str(descriptor.get("run_id") or ""),
                        str(descriptor.get("thread_id") or ""),
                        str(descriptor.get("episode_id") or ""),
                        str(current_row.get("status") or "unknown")
                        if is_current
                        else "superseded",
                        _json_dump(evidence_ids),
                        _json_dump(artifact("proposal.json")),
                        _json_dump(artifact("review.json")),
                        _json_dump(artifact("validation.json")),
                        created_at,
                        str(current_row.get("updated_at") or created_at)
                        if is_current
                        else created_at,
                    )
                )
        return rows

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        uri = "file:" + quote(str(self.db_path), safe="/:._-") + "?mode=ro"
        conn = sqlite3.connect(uri, timeout=30, isolation_level=None, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(
                "CREATE TEMP TABLE _evolution_target_job_scope "
                "(job_id TEXT PRIMARY KEY)"
            )
            conn.execute(
                "CREATE TEMP TABLE _evolution_target_candidate_scope "
                "(candidate_id TEXT PRIMARY KEY)"
            )
            conn.execute(
                "CREATE TEMP TABLE _evolution_target_observation_scope "
                "(observation_id TEXT PRIMARY KEY)"
            )
            if self.target:
                conn.execute(
                    "INSERT OR IGNORE INTO _evolution_target_observation_scope(observation_id) "
                    "SELECT observation_id FROM main.observations "
                    "WHERE COALESCE(NULLIF(resolved_target, ''), target) = ?",
                    (self.target,),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO _evolution_target_job_scope(job_id) "
                    "SELECT job_id FROM main.observations "
                    "WHERE COALESCE(NULLIF(resolved_target, ''), target) = ?",
                    (self.target,),
                )
            conn.execute(
                """
                CREATE TEMP TABLE evolution_observations AS
                SELECT observation_id, run_id, 'run:' || run_id AS run_ref,
                       thread_id, job_id, episode_id, item_ref, signal_kind,
                       target, resolved_target, claim, evidence_refs_json,
                       outcome_ref, status,
                       created_at
                FROM main.observations
                WHERE ? = ''
                   OR COALESCE(NULLIF(resolved_target, ''), target) = ?
                """,
                (self.target, self.target),
            )
            conn.execute(
                """
                CREATE TEMP TABLE _evolution_candidate_revision_scope (
                    candidate_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    target TEXT NOT NULL,
                    route TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    proposal_json TEXT NOT NULL,
                    review_json TEXT NOT NULL,
                    validation_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            candidate_rows = self._candidate_revision_rows(conn)
            if self.target:
                candidate_rows = [
                    row for row in candidate_rows if str(row[2] or "") == self.target
                ]
            if candidate_rows:
                conn.executemany(
                    "INSERT INTO _evolution_candidate_revision_scope VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    candidate_rows,
                )
                conn.executemany(
                    "INSERT OR IGNORE INTO _evolution_target_candidate_scope(candidate_id) "
                    "VALUES (?)",
                    [(str(row[0]),) for row in candidate_rows],
                )
                if self.target:
                    linked_observation_ids: list[str] = []
                    for row in candidate_rows:
                        try:
                            values = json.loads(str(row[8] or "[]"))
                        except Exception:
                            values = []
                        linked_observation_ids.extend(
                            str(item)
                            for item in values
                            if str(item).strip()
                        )
                    if linked_observation_ids:
                        conn.executemany(
                            "INSERT OR IGNORE INTO _evolution_target_observation_scope(observation_id) "
                            "VALUES (?)",
                            [(item,) for item in dict.fromkeys(linked_observation_ids)],
                        )
                        conn.execute(
                            """
                            INSERT INTO evolution_observations
                            SELECT observation_id, run_id, 'run:' || run_id AS run_ref,
                                   thread_id, job_id, episode_id, item_ref, signal_kind,
                                   target, resolved_target, claim, evidence_refs_json,
                                   outcome_ref, status,
                                   created_at
                            FROM main.observations AS source
                            WHERE source.observation_id IN (
                                SELECT observation_id FROM _evolution_target_observation_scope
                            )
                              AND NOT EXISTS (
                                SELECT 1 FROM evolution_observations AS visible
                                WHERE visible.observation_id = source.observation_id
                              )
                            """
                        )
                        conn.execute(
                            "INSERT OR IGNORE INTO _evolution_target_job_scope(job_id) "
                            "SELECT job_id FROM main.observations WHERE observation_id IN "
                            "(SELECT observation_id FROM _evolution_target_observation_scope)",
                        )
            conn.execute(
                """
                CREATE TEMP VIEW evolution_candidate_revisions AS
                SELECT candidate_id, revision, target, route, run_id,
                       CASE WHEN TRIM(run_id) = '' THEN '' ELSE 'run:' || run_id END AS run_ref,
                       thread_id, episode_id, status, evidence_ids_json,
                       proposal_json, review_json, validation_json,
                       created_at, updated_at
                FROM _evolution_candidate_revision_scope
                """
            )
            conn.execute(
                """
                CREATE TEMP TABLE evolution_jobs AS
                SELECT job_id, project_id, run_id, 'run:' || run_id AS run_ref,
                       thread_id, episode_id, selected_item_ref, trigger_kind,
                       status, attempt_count, candidate_id, outcome_json,
                       predecessor_job_id, error, created_at, updated_at
                FROM main.jobs
                WHERE ? = ''
                   OR job_id IN (SELECT job_id FROM _evolution_target_job_scope)
                   OR candidate_id IN (SELECT candidate_id FROM _evolution_target_candidate_scope)
                """,
                (self.target,),
            )
            conn.execute(
                """
                CREATE TEMP TABLE evolution_skill_runs AS
                SELECT run_id, 'run:' || run_id AS run_ref, skill_name,
                       skill_version, presented, read, helper_used, outcome,
                       false_activation, partial
                FROM main.skill_runs
                WHERE ? = '' OR skill_name = ?
                """,
                (self.target, self.target),
            )
            conn.execute("PRAGMA query_only=ON")
            conn.set_authorizer(self._authorize)
            yield conn
        finally:
            conn.close()

    @classmethod
    def _authorize(
        cls,
        action: int,
        arg1: str | None,
        arg2: str | None,
        database: str | None,
        source: str | None,
    ) -> int:
        _ = database
        if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE}:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            table = str(arg1 or "")
            column = str(arg2 or "")
            origin = str(source or "")
            if table.startswith("sqlite_"):
                return sqlite3.SQLITE_DENY
            if table in cls._VIEWS or origin in cls._VIEWS:
                return sqlite3.SQLITE_OK
            if table == "_evolution_candidate_revision_scope" and not column:
                # SQLite emits one source-less pseudo-column read for COUNT(*)
                # over the public candidate-revision view. Permit only that
                # aggregate read; direct reads of internal table columns remain
                # denied by the branch above.
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            function_name = str(arg2 or arg1 or "").casefold()
            if function_name in {
                "edit",
                "eval",
                "fts3_tokenizer",
                "fsdir",
                "getenv",
                "load_extension",
                "readfile",
                "shell",
                "system",
                "writefile",
            } or function_name.startswith("pragma_"):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def execute(
        self, sql: str, *, result_writer: Callable[[str], str] | None = None,
    ) -> dict[str, Any] | _FileQueryResult:
        with self._connection() as conn:
            return _bounded_query_result(conn, sql, label="History", result_writer=result_writer)

    def open_run(self, run_id: str) -> tuple[Path, TurnTrace] | None:
        """Open one run already reachable through this exact history scope."""

        resolved = str(run_id or "").strip()
        if not resolved:
            return None
        with self._connection() as conn:
            authorized = conn.execute(
                """
                SELECT 1 FROM (
                    SELECT run_id FROM evolution_observations
                    UNION SELECT run_id FROM evolution_candidate_revisions
                    UNION SELECT run_id FROM evolution_jobs
                    UNION SELECT run_id FROM evolution_skill_runs
                ) WHERE run_id = ? LIMIT 1
                """,
                (resolved,),
            ).fetchone()
        if authorized is None:
            return None
        uri = "file:" + quote(str(self.db_path), safe="/:._-") + "?mode=ro"
        with sqlite3.connect(uri, timeout=30, isolation_level=None, uri=True) as conn:
            row = conn.execute(
                """
                SELECT run_dir, thread_id, payload_json FROM jobs
                WHERE run_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (resolved,),
            ).fetchone()
        if row is None:
            return None
        run_dir = Path(str(row[0] or "")).expanduser().resolve()
        if not run_dir.is_dir():
            return None
        trace = collect_turn_trace(
            run_dir=run_dir,
            fallback={**json.loads(row[2] or "{}"), "run_id": resolved, "thread_id": str(row[1] or "")},
            include_events=False,
        )
        return run_dir, trace

    def tool(self, *, result_writer: Callable[[str], str] | None = None) -> StructuredTool:
        def query_evolution_history_sql(sql: str) -> tuple[str, Any]:
            return _agent_tool_result(
                lambda: self.execute(sql, result_writer=result_writer),
                operation_name="query_evolution_history_sql",
                with_artifact=True,
            )

        return StructuredTool.from_function(
            func=query_evolution_history_sql,
            name="query_evolution_history_sql",
            description=(
                (QueryEvolutionHistorySQLInput.__doc__ or "Query workspace evolution history.")
                + " Exact available schema: "
                + _schema_text(_HISTORY_QUERY_SCHEMA)
                + ". Copy returned run_ref values unchanged to query a selected run. "
                "Large results return a complete result_path readable with read_file. "
                "Errors return ok=false and a recovery action."
            ),
            args_schema=QueryEvolutionHistorySQLInput,
            infer_schema=False,
            response_format="content_and_artifact",
            handle_tool_error=str,
            handle_validation_error=lambda exc: _agent_tool_error(
                exc,
                operation="query_evolution_history_sql",
            ),
        )


class EvolutionTraceScope:
    """Host-bound trace reader that opens one authorized run per invocation."""

    def __init__(
        self,
        runs: Mapping[str, tuple[Path | str, TurnTrace]],
        *,
        anchor_run_id: str = "",
        history_scope: EvolutionHistoryScope | None = None,
    ) -> None:
        normalized: dict[str, EvolutionTraceRun] = {}
        for raw_run_id, (raw_dir, trace) in runs.items():
            run_id = str(raw_run_id or "").strip()
            run_dir = Path(raw_dir).expanduser().resolve()
            if not run_id or "#" in run_id:
                raise ValueError("trace scope run IDs must be non-empty stable identifiers")
            if not run_dir.is_dir():
                raise ValueError(f"trace scope run directory is unavailable: {run_id}")
            if str(trace.run_id or "").strip() != run_id:
                raise ValueError(f"trace metadata does not match host run ID: {run_id}")
            normalized[run_id] = EvolutionTraceRun(
                run_id=run_id,
                run_dir=run_dir,
                trace=trace,
            )
        if not normalized:
            raise ValueError("an evolution trace scope requires at least one run")
        self._runs = dict(sorted(normalized.items()))
        self._anchor_run_id = str(anchor_run_id or next(iter(self._runs))).strip()
        if self._anchor_run_id not in self._runs:
            raise ValueError("the evolution trace anchor is outside the authorized run set")
        self._history_scope = history_scope

    @property
    def anchor_run_id(self) -> str:
        return self._anchor_run_id

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(self._runs)

    @property
    def traces(self) -> tuple[TurnTrace, ...]:
        return tuple(item.trace for item in self._runs.values())

    def _resolve_run_ref(self, run_ref: str = "") -> str:
        value = str(run_ref or "").strip()
        if not value:
            return self._anchor_run_id
        if not value.startswith("run:"):
            raise ValueError("run_ref must be empty or an authorized run:<run_id> handle")
        run_id = value.removeprefix("run:")
        if run_id not in self._runs and self._history_scope is not None:
            opened = self._history_scope.open_run(run_id)
            if opened is not None:
                run_dir, trace = opened
                self._runs[run_id] = EvolutionTraceRun(
                    run_id=run_id,
                    run_dir=run_dir,
                    trace=trace,
                )
        if run_id not in self._runs:
            raise ValueError("run_ref is outside the authorized history scope")
        return run_id

    @contextmanager
    def _connection(self, run_id: str = "") -> Iterator[sqlite3.Connection]:
        selected_run_id = self._resolve_run_ref(
            f"run:{run_id}" if run_id else ""
        )
        run = self._runs[selected_run_id]
        conn = sqlite3.connect(":memory:", timeout=30, isolation_level=None, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            conn.create_function(
                "semantic_payload",
                2,
                _semantic_payload_json,
                deterministic=True,
            )
            conn.create_function(
                "semantic_name",
                1,
                semantic_event_name,
                deterministic=True,
            )
            conn.execute(
                """
                CREATE TEMP TABLE _trajectory_run_scope (
                    run_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    entrypoint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    user_prompt TEXT NOT NULL,
                    final_answer TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    task_outcome TEXT NOT NULL,
                    outcome_ref TEXT NOT NULL,
                    explicit_correction TEXT NOT NULL,
                    resume_guidance TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO _trajectory_run_scope VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.trace.thread_id,
                    run.trace.entrypoint,
                    run.trace.status,
                    run.trace.user_prompt,
                    run.trace.final_answer,
                    run.trace.summary,
                    run.trace.task_outcome,
                    run.trace.outcome_ref,
                    run.trace.explicit_correction,
                    run.trace.resume_guidance,
                ),
            )
            db_path = run.run_dir / OBSERVABILITY_DB_NAME
            event_sql = ""
            raw_event_sql = ""
            if db_path.is_file():
                alias = "trace_selected"
                uri = "file:" + quote(str(db_path), safe="/:._-") + "?mode=ro"
                conn.execute(f"ATTACH DATABASE ? AS {alias}", (uri,))
                run_literal = _sql_literal(run.run_id)
                names = ", ".join(_sql_literal(item) for item in _TRAJECTORY_EVENT_NAMES)
                legacy = ", ".join(_sql_literal(item) for item in sorted(LEGACY_TRACE_SOURCES))
                event_sql = f"""
                    SELECT
                        e.id AS id,
                        {run_literal} AS run_id,
                        'run:' || {run_literal} || '#event:' || e.id AS handle,
                        e.ts AS ts,
                        semantic_name(e.name) AS name,
                        COALESCE(e.agent_name, '') AS agent_name,
                        COALESCE(e.node, '') AS node,
                        COALESCE(e.callback_run_id, '') AS callback_run_id,
                        COALESCE(e.parent_callback_run_id, '') AS parent_callback_run_id,
                        COALESCE(e.model, '') AS model,
                        COALESCE(e.tool, '') AS tool,
                        COALESCE(e.status, '') AS status,
                        e.duration_ms AS duration_ms,
                        semantic_payload(e.name, e.payload_json) AS payload_json,
                        length(semantic_payload(e.name, e.payload_json)) AS payload_chars
                    FROM {alias}.observation_events AS e
                    WHERE e.name IN ({names})
                      AND e.source NOT IN ({legacy})
                      AND NOT (
                        e.name = 'LLM_CALL_END'
                        AND COALESCE(e.callback_run_id, '') != ''
                        AND EXISTS (
                            SELECT 1
                            FROM {alias}.observation_events AS raw_model
                            WHERE raw_model.name = 'LLM_RAW_RESPONSE'
                              AND raw_model.callback_run_id = e.callback_run_id
                        )
                      )
                      AND NOT (
                        e.name = 'TOOL_CALL_END'
                        AND COALESCE(e.callback_run_id, '') != ''
                        AND EXISTS (
                            SELECT 1
                            FROM {alias}.observation_events AS raw_tool
                            WHERE raw_tool.name = 'TOOL_RAW_OUTPUT'
                              AND raw_tool.callback_run_id = e.callback_run_id
                        )
                      )
                    """
                raw_event_sql = f"""
                    SELECT
                        e.id AS id,
                        {run_literal} AS run_id,
                        'run:' || {run_literal} || '#event:' || e.id || '?view=raw' AS handle,
                        e.ts AS ts,
                        e.name AS name,
                        COALESCE(e.agent_name, '') AS agent_name,
                        COALESCE(e.node, '') AS node,
                        COALESCE(e.callback_run_id, '') AS callback_run_id,
                        COALESCE(e.parent_callback_run_id, '') AS parent_callback_run_id,
                        COALESCE(e.model, '') AS model,
                        COALESCE(e.tool, '') AS tool,
                        COALESCE(e.status, '') AS status,
                        e.duration_ms AS duration_ms,
                        e.payload_json AS payload_json,
                        length(e.payload_json) AS payload_chars
                    FROM {alias}.observation_events AS e
                    """
            conn.execute(
                """
                CREATE TEMP VIEW trajectory_runs AS
                SELECT run_id, thread_id, entrypoint, status, user_prompt,
                       final_answer, summary, task_outcome, outcome_ref,
                       explicit_correction, resume_guidance
                FROM _trajectory_run_scope
                """
            )
            event_sql = event_sql or (
                "SELECT CAST(NULL AS INTEGER) AS id, CAST(NULL AS TEXT) AS run_id, "
                "CAST(NULL AS TEXT) AS handle, "
                "CAST(NULL AS REAL) AS ts, CAST(NULL AS TEXT) AS name, "
                "CAST(NULL AS TEXT) AS agent_name, CAST(NULL AS TEXT) AS node, "
                "CAST(NULL AS TEXT) AS callback_run_id, "
                "CAST(NULL AS TEXT) AS parent_callback_run_id, "
                "CAST(NULL AS TEXT) AS model, CAST(NULL AS TEXT) AS tool, "
                "CAST(NULL AS TEXT) AS status, CAST(NULL AS INTEGER) AS duration_ms, "
                "CAST(NULL AS TEXT) AS payload_json, CAST(NULL AS INTEGER) AS payload_chars "
                "WHERE 0"
            )
            raw_event_sql = raw_event_sql or (
                "SELECT CAST(NULL AS INTEGER) AS id, CAST(NULL AS TEXT) AS run_id, "
                "CAST(NULL AS TEXT) AS handle, "
                "CAST(NULL AS REAL) AS ts, CAST(NULL AS TEXT) AS name, "
                "CAST(NULL AS TEXT) AS agent_name, CAST(NULL AS TEXT) AS node, "
                "CAST(NULL AS TEXT) AS callback_run_id, "
                "CAST(NULL AS TEXT) AS parent_callback_run_id, "
                "CAST(NULL AS TEXT) AS model, CAST(NULL AS TEXT) AS tool, "
                "CAST(NULL AS TEXT) AS status, CAST(NULL AS INTEGER) AS duration_ms, "
                "CAST(NULL AS TEXT) AS payload_json, CAST(NULL AS INTEGER) AS payload_chars "
                "WHERE 0"
            )
            conn.execute(f"CREATE TEMP VIEW trajectory_events AS {event_sql}")
            conn.execute(f"CREATE TEMP VIEW trajectory_raw_events AS {raw_event_sql}")
            conn.execute("PRAGMA query_only=ON")
            conn.set_authorizer(self._authorize)
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _authorize(
        action: int,
        arg1: str | None,
        arg2: str | None,
        database: str | None,
        source: str | None,
    ) -> int:
        if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE}:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            table = str(arg1 or "")
            origin = str(source or "")
            if table.startswith("sqlite_"):
                return sqlite3.SQLITE_DENY
            if table in {"trajectory_runs", "trajectory_events", "trajectory_raw_events"}:
                return sqlite3.SQLITE_OK
            if origin in {"trajectory_runs", "trajectory_events", "trajectory_raw_events"}:
                return sqlite3.SQLITE_OK
            # SQLite can report a source-less, empty-column read for COUNT(*)
            # after flattening a view (sqlite.org/c3ref/set_authorizer.html).
            # This exposes no columns; the selected run's raw count is public.
            if not arg2 and (
                (database == "trace_selected" and table == "observation_events")
                or (database in {None, "temp"} and table == "_trajectory_run_scope")
            ):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            function_name = str(arg2 or arg1 or "").casefold()
            if function_name in {
                "edit",
                "eval",
                "fts3_tokenizer",
                "fsdir",
                "getenv",
                "load_extension",
                "readfile",
                "shell",
                "system",
                "writefile",
            } or function_name.startswith("pragma_"):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    @staticmethod
    def _rows(conn: sqlite3.Connection, sql: str) -> tuple[list[str], list[dict[str, Any]]]:
        statement = str(sql or "").strip()
        if not statement:
            raise ValueError("SQL is required")
        try:
            cursor = conn.execute(statement)
            description = cursor.description
            if description is None:
                raise ValueError("Only SELECT or WITH queries are allowed.")
            columns = [str(item[0]) for item in description]
            rows = [dict(row) for row in cursor.fetchall()]
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"Trace query rejected: {exc}") from exc
        return columns, rows

    def execute(
        self, sql: str, *, run_ref: str = "",
        result_writer: Callable[[str], str] | None = None,
    ) -> dict[str, Any] | _FileQueryResult:
        run_id = self._resolve_run_ref(run_ref)
        with self._connection(run_id) as conn:
            return _bounded_query_result(conn, sql, label="Trace", result_writer=result_writer)

    def _event_row(self, handle: str) -> dict[str, Any]:
        match = _EVENT_HANDLE_RE.fullmatch(str(handle or "").strip())
        if match is None:
            raise EvolutionQueryError(
                "Invalid event handle.",
                recovery="Copy a complete handle returned by query_evolution_trace_sql or the supplied evidence.",
            )
        run_id, event_id_text, raw_view = match.groups()
        try:
            self._resolve_run_ref(f"run:{run_id}")
        except ValueError as exc:
            raise EvolutionQueryError(
                "The event's run is outside the accessible run scope.",
                recovery="Use a run_ref from the supplied evidence or query_evolution_history_sql.",
            ) from exc
        event_id = int(event_id_text)
        view = "trajectory_raw_events" if raw_view else "trajectory_events"
        with self._connection(run_id) as conn:
            def select(source: str):
                return conn.execute(
                    f"""
                    SELECT id, run_id, handle, ts, name, agent_name, node,
                           callback_run_id, parent_callback_run_id, model, tool,
                           status, duration_ms, payload_chars
                    FROM {source} WHERE id = ?
                    """,
                    (event_id,),
                ).fetchone()

            row = select(view)
            if row is None and not raw_view:
                # Legacy evidence may cite a raw-only event with the old bare
                # handle. Return its original body and explicit raw reference.
                view = "trajectory_raw_events"
                row = select(view)
        if row is None:
            raise EvolutionQueryError(
                f"Event {event_id} was not found in run {run_id}.",
                recovery=(
                    "Query existing handles first: SELECT handle, name, tool FROM "
                    "trajectory_events ORDER BY id LIMIT 20. Use trajectory_raw_events "
                    "for original records and copy a returned handle."
                ),
            )
        return {**dict(row), "_view": view}

    def contains_event(self, handle: str) -> bool:
        try:
            self._event_row(handle)
        except ValueError:
            return False
        return True

    def read_event(
        self,
        *,
        handle: str,
        field: str = "payload_json",
        offset: int = 0,
        limit: int = 16_000,
    ) -> dict[str, Any]:
        row = self._event_row(handle)
        requested = str(field or "payload_json").strip() or "payload_json"
        start = max(0, int(offset))
        length = min(_READ_MAX_CHARS, max(1, int(limit)))
        allowed_fields = set(_TRACE_QUERY_SCHEMA["trajectory_events"]) | {"raw_payload_json"}
        if not requested.startswith("$") and requested not in allowed_fields:
            raise EvolutionQueryError(
                f"Unknown event field: {requested}.",
                recovery="Use one of: " + ", ".join(sorted(allowed_fields)) + "; or a JSON path beginning with $.",
            )
        if requested in {"payload_json", "raw_payload_json"} or requested.startswith("$"):
            run_id, event_id = row["run_id"], row["id"]
            source_view = (
                "trajectory_raw_events" if requested == "raw_payload_json" else row["_view"]
            )
            try:
                with self._connection(run_id) as conn:
                    if requested in {"payload_json", "raw_payload_json"}:
                        value_row = conn.execute(
                            f"""
                            SELECT length(payload_json) AS total_chars,
                                   substr(payload_json, ?, ?) AS segment
                            FROM {source_view} WHERE run_id = ? AND id = ?
                            """,
                            (start + 1, length, run_id, event_id),
                        ).fetchone()
                    else:
                        value_row = conn.execute(
                            f"""
                            WITH selected AS (
                                SELECT json_type(payload_json, ?) AS kind,
                                       json_extract(payload_json, ?) AS value
                                FROM {source_view} WHERE run_id = ? AND id = ?
                            ), rendered AS (
                                SELECT kind, CASE
                                    WHEN kind = 'true' THEN 'true'
                                    WHEN kind = 'false' THEN 'false'
                                    WHEN kind IS NULL OR kind = 'null' THEN 'null'
                                    ELSE CAST(value AS TEXT)
                                END AS text
                                FROM selected
                            )
                            SELECT kind, length(text) AS total_chars,
                                   substr(text, ?, ?) AS segment
                            FROM rendered
                            """,
                            (
                                requested,
                                requested,
                                run_id,
                                event_id,
                                start + 1,
                                length,
                            ),
                        ).fetchone()
            except sqlite3.DatabaseError as exc:
                raise EvolutionQueryError(
                    f"Cannot read {requested}: {exc}",
                    recovery="Read payload_json to inspect the body, then select an existing field using SQLite JSON path syntax.",
                ) from exc
            if value_row is None:
                raise EvolutionQueryError(
                    "The event is no longer available.",
                    recovery="Query the event again to obtain a current handle.",
                )
            if requested.startswith("$") and value_row["kind"] is None:
                raise EvolutionQueryError(
                    f"JSON field {requested} does not exist in this event body.",
                    recovery="Read payload_json to see this event's available fields, then select an existing JSON path.",
                )
            total_chars = int(value_row["total_chars"] or 0)
            segment = str(value_row["segment"] or "")
        else:
            value = row.get(requested)
            text = value if isinstance(value, str) else _json_dump(value)
            total_chars = len(text)
            segment = text[start : start + length]
        next_offset = start + len(segment)
        return {
            "handle": row["handle"],
            "field": requested,
            "offset": start,
            "limit": length,
            "total_chars": total_chars,
            "segment": segment,
            "next_offset": next_offset if next_offset < total_chars else None,
        }

    def tools(self, *, result_writer: Callable[[str], str] | None = None) -> list[StructuredTool]:
        def query_evolution_trace_sql(sql: str, run_ref: str = "") -> tuple[str, Any]:
            return _agent_tool_result(
                lambda: self.execute(sql, run_ref=run_ref, result_writer=result_writer),
                operation_name="query_evolution_trace_sql",
                with_artifact=True,
            )

        def read_evolution_event(
            handle: str,
            field: str = "payload_json",
            offset: int = 0,
            limit: int = 16_000,
        ) -> str:
            return _agent_tool_result(
                lambda: self.read_event(
                    handle=handle,
                    field=field,
                    offset=offset,
                    limit=limit,
                ),
                operation_name="read_evolution_event",
            )

        tools = [
            StructuredTool.from_function(
                func=query_evolution_trace_sql,
                name="query_evolution_trace_sql",
                description=(
                    (QueryEvolutionTraceSQLInput.__doc__ or "Query an evolution trace.")
                    + " Exact available schema: "
                    + _schema_text(_TRACE_QUERY_SCHEMA)
                    + ". Read the run's final_answer and relevant events before deciding what to inspect further. "
                    "Example: SELECT handle, name, tool, payload_json FROM trajectory_events ORDER BY id LIMIT 20. "
                    "For a long body, SELECT handle, name, payload_chars FROM trajectory_events ORDER BY id LIMIT 20, "
                    "then pass an actual returned handle to read_evolution_event. Wait for a query's results "
                    "before issuing reads that depend on them; independent queries may run in parallel. "
                    "trajectory_raw_events exposes original records through equally readable handles. "
                    "Large results return a complete result_path readable with read_file. "
                    "Errors return ok=false and a recovery action."
                ),
                args_schema=QueryEvolutionTraceSQLInput,
                infer_schema=False,
                response_format="content_and_artifact",
                handle_tool_error=str,
                handle_validation_error=lambda exc: _agent_tool_error(
                    exc,
                    operation="query_evolution_trace_sql",
                ),
            ),
            StructuredTool.from_function(
                func=read_evolution_event,
                name="read_evolution_event",
                description=ReadEvolutionEventInput.__doc__ or "Read an evolution event field.",
                args_schema=ReadEvolutionEventInput,
                infer_schema=False,
                handle_tool_error=str,
                handle_validation_error=lambda exc: _agent_tool_error(
                    exc,
                    operation="read_evolution_event",
                ),
            ),
        ]
        if self._history_scope is not None:
            tools.append(self._history_scope.tool(result_writer=result_writer))
        return tools

    def index(self, *, context_label: str, skill_context: str) -> str:
        """Build a deterministic scope index without embedding semantic event bodies."""

        lines = [
            "# Evolution trace scope index",
            "",
            "This is a deterministic index, not a semantic summary. The full authorized semantic event set remains queryable.",
            "",
            f"Context: {str(context_label or 'self-evolution inspection').strip()}",
            "",
            "## Anchor run",
            "",
        ]
        for run in (self._runs[self._anchor_run_id],):
            lines.extend(
                [
                    f"### `{run.run_id}`",
                    "",
                    f"- Thread: `{run.trace.thread_id or 'not recorded'}`",
                    f"- Entrypoint: `{run.trace.entrypoint or 'not recorded'}`",
                    f"- Run status: `{run.trace.status or 'unknown'}` (this turn; delegated work may still be running)",
                    f"- Verified outcome: `{run.trace.task_outcome or 'not recorded'}`",
                    f"- Outcome handle: `{run.trace.outcome_ref or 'not recorded'}`",
                    "- Task:",
                    run.trace.user_prompt or "No initial task was recovered.",
                ]
            )
            if run.trace.resume_guidance:
                lines.extend(["- Resume guidance:", run.trace.resume_guidance])
            if run.trace.explicit_correction:
                lines.extend(["- Explicit correction:", run.trace.explicit_correction])
            if run.trace.diagnostics:
                lines.extend(
                    [
                        "- Trace read diagnostics:",
                        *[f"  - {item}" for item in run.trace.diagnostics],
                    ]
                )
            lines.append("")
        lines.extend(
            [
                "## Current skill context",
                "",
                str(skill_context or "No skill context was supplied."),
                "",
                (
                    "Read the run's final_answer and relevant event bodies using the trace query tool. "
                    "Its description contains the available tables and two query examples. "
                    "Select handle for references ready to read or cite; wait for the returned "
                    "references before issuing dependent reads. SQL can read multiple bodies "
                    "at once. Follow result_path with the file tools or next_offset for a long "
                    "event field. Consult workspace history when another episode is relevant."
                ),
            ]
        )
        return "\n".join(lines).strip() + "\n"


__all__ = [
    "EvolutionHistoryScope",
    "EvolutionTraceRun",
    "EvolutionTraceScope",
    "QueryEvolutionHistorySQLInput",
    "QueryEvolutionTraceSQLInput",
    "ReadEvolutionEventInput",
]
