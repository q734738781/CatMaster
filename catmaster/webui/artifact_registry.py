from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any

from catmaster.storage import connect_workspace_db
from catmaster.tools.base import system_root, workspace_root

from .thread_models import ArtifactRecord, utc_ts

STRUCTURE_SUFFIXES = {".cif", ".cssr", ".cube", ".gro", ".mol", ".mol2", ".pdb", ".sdf", ".traj", ".vasp", ".xsf", ".xyz"}
STRUCTURE_NAMES = {"POSCAR", "CONTCAR", "OUTCAR", "XDATCAR"}
MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx", ".rst"}
CSV_SUFFIXES = {".csv", ".tsv"}
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".out", ".patch", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z"}
_SCHEMA_COMPONENT = "artifact_registry"
_SCHEMA_VERSION = 1


def infer_renderer(path: str, mime_type: str = "") -> str:
    candidate = Path(str(path or ""))
    suffix = candidate.suffix.lower()
    name = candidate.name.upper()
    mime = str(mime_type or "").lower()
    if name in STRUCTURE_NAMES or suffix in STRUCTURE_SUFFIXES:
        return "structure"
    if mime.startswith("image/") or suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
        return "image"
    if suffix in CSV_SUFFIXES:
        return "csv"
    if suffix in MARKDOWN_SUFFIXES:
        return "markdown"
    if mime == "application/pdf" or suffix == ".pdf":
        return "pdf"
    if suffix in ARCHIVE_SUFFIXES:
        return "archive"
    if suffix in TEXT_SUFFIXES or mime.startswith("text/"):
        return "text"
    return "text"


def _dump_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class ArtifactRegistry:
    def __init__(self, *, workspace: Path | str, workspace_id: str = "") -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace_id = str(workspace_id or self.workspace.name).strip() or self.workspace.name
        self.root = system_root(self.workspace) / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self._init_storage()

    def list_artifacts(self, *, thread_id: str = "") -> list[ArtifactRecord]:
        self.migrate_legacy_run_artifacts()
        # Older streaming adapters could register schema field names such as
        # ``task_config.fmax`` as paths. Keep any migrated legacy index intact,
        # but never expose records whose target is not an actual workspace file.
        records = [
            record
            for record in self._read_all(thread_id=thread_id)
            if self._record_file_exists(record)
        ]
        records.sort(key=lambda item: float(item.created_at or 0.0))
        return records

    def get(self, artifact_id: str) -> ArtifactRecord | None:
        with connect_workspace_db(self.workspace) as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM workspace_artifacts
                WHERE artifact_id = ?
                """,
                (str(artifact_id or ""),),
            ).fetchone()
        if row is None:
            return None
        try:
            record = ArtifactRecord.model_validate(json.loads(str(row["payload_json"])))
        except Exception:
            return None
        return record if self._record_file_exists(record) else None

    def register_path(
        self,
        path: str,
        *,
        thread_id: str = "",
        message_id: str = "",
        tool_call_id: str = "",
        run_id: str = "",
        title: str = "",
        summary: str = "",
        mime_type: str = "",
        renderer: str = "",
        meta: dict[str, Any] | None = None,
        artifact_id: str = "",
    ) -> ArtifactRecord:
        normalized = self.normalize_workspace_path(path)
        candidate = self.workspace.joinpath(*Path(normalized).parts)
        if not candidate.is_file():
            raise ValueError("Artifact path must reference an existing workspace file.")
        guessed_mime = mime_type or mimetypes.guess_type(Path(normalized).name)[0] or ""
        renderer_value = renderer or infer_renderer(normalized, guessed_mime)
        aid = artifact_id or self._artifact_id(
            normalized,
            thread_id=thread_id,
            message_id=message_id,
            tool_call_id=tool_call_id,
            run_id=run_id,
        )
        now = utc_ts()
        existing = self.get(aid)
        created_at = existing.created_at if existing else now
        record = ArtifactRecord(
            artifact_id=aid,
            thread_id=str(thread_id or ""),
            message_id=str(message_id or ""),
            tool_call_id=str(tool_call_id or ""),
            run_id=str(run_id or ""),
            workspace_id=self.workspace_id,
            path=normalized,
            mime_type=guessed_mime,
            renderer=renderer_value,
            title=str(title or Path(normalized).name or normalized),
            summary=str(summary or ""),
            created_at=created_at,
            updated_at=now,
            preview_url=f"/api/artifacts/{aid}/preview",
            download_url=f"/api/artifacts/{aid}/download",
            meta=dict(meta or {}),
        )
        self._upsert(record)
        return record

    def register_from_run_state(
        self,
        run_state: dict[str, Any],
        *,
        thread_id: str = "",
        message_id: str = "",
        run_id: str = "",
    ) -> list[ArtifactRecord]:
        rows = run_state.get("artifacts")
        if not isinstance(rows, list):
            return []
        out: list[ArtifactRecord] = []
        for item in rows:
            if isinstance(item, str):
                path = item
                summary = ""
            elif isinstance(item, dict):
                path = str(item.get("path") or item.get("file") or item.get("output_path") or "").strip()
                summary = str(item.get("description") or item.get("summary") or "").strip()
            else:
                continue
            if not path:
                continue
            try:
                normalized = self.normalize_workspace_path(path)
                if not self.workspace.joinpath(*Path(normalized).parts).exists():
                    continue
                out.append(
                    self.register_path(
                        normalized,
                        thread_id=thread_id or str(run_state.get("thread_id") or ""),
                        message_id=message_id,
                        run_id=run_id or str(run_state.get("run_id") or ""),
                        summary=summary,
                        meta={"source": "run_state"},
                    )
                )
            except ValueError:
                continue
        return out

    def migrate_legacy_run_artifacts(self) -> list[ArtifactRecord]:
        runs_root = system_root(self.workspace) / "runs"
        if not runs_root.is_dir():
            return []
        out: list[ArtifactRecord] = []
        for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            state_path = run_dir / "run_state.json"
            if not state_path.exists():
                continue
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("artifact_ids"):
                continue
            if not isinstance(payload.get("artifacts"), list):
                continue
            thread_id = str(payload.get("webui_thread_id") or payload.get("thread_id") or "").strip()
            out.extend(
                self.register_from_run_state(
                    payload,
                    thread_id=thread_id,
                    run_id=run_dir.name,
                )
            )
        return out

    def normalize_workspace_path(self, path: str) -> str:
        raw = str(path or "").strip().replace("\\", "/").lstrip("/")
        if not raw:
            raise ValueError("Artifact path is required.")
        parts = Path(raw).parts
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("Artifact path is invalid.")
        if parts and parts[0] == "metadata":
            raise ValueError("Metadata paths are not user-facing artifacts.")
        candidate = self.workspace.joinpath(*parts).resolve()
        files_root = workspace_root(self.workspace).resolve()
        workspace_resolved = self.workspace.resolve()
        if not candidate.exists() and parts and parts[0] != "files":
            files_candidate = files_root.joinpath(*parts).resolve()
            try:
                files_candidate.relative_to(files_root)
            except ValueError as exc:
                raise ValueError("Artifact path escapes files root.") from exc
            candidate = files_candidate
        try:
            candidate.relative_to(workspace_resolved)
        except ValueError as exc:
            raise ValueError("Artifact path escapes workspace.") from exc
        return str(candidate.relative_to(workspace_resolved)).replace("\\", "/")

    def resolve_path(self, record: ArtifactRecord) -> Path:
        candidate = self.workspace.joinpath(*Path(record.path).parts).resolve()
        try:
            candidate.relative_to(self.workspace.resolve())
        except ValueError as exc:
            raise ValueError("Artifact path escapes workspace.") from exc
        return candidate

    def _record_file_exists(self, record: ArtifactRecord) -> bool:
        try:
            return self.resolve_path(record).is_file()
        except (OSError, ValueError):
            return False

    @staticmethod
    def find_in_project_root(project_root: Path | str, artifact_id: str) -> tuple[Path, ArtifactRecord] | None:
        root = Path(project_root).expanduser().resolve()
        if not root.exists():
            return None
        candidates: list[Path]
        if (root / "files").is_dir() and (root / "metadata").is_dir():
            candidates = [root]
        else:
            workspaces: list[Path] = []
            for database in root.rglob("metadata/workspace.sqlite"):
                if database.is_file():
                    workspaces.append(database.parents[1])
            for index in root.rglob("metadata/artifacts/index.jsonl"):
                if not index.is_file():
                    continue
                try:
                    workspaces.append(index.parents[2])
                except Exception:
                    continue
            candidates = list(dict.fromkeys(workspaces))
        for workspace in candidates:
            registry = ArtifactRegistry(workspace=workspace)
            record = registry.get(artifact_id)
            if record is not None:
                return workspace, record
        return None

    def _artifact_id(self, path: str, *, thread_id: str, message_id: str, tool_call_id: str, run_id: str) -> str:
        key = "\n".join([self.workspace_id, thread_id, message_id, tool_call_id, run_id, path])
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return f"art_{digest}"

    def _init_storage(self) -> None:
        legacy_records = self._read_legacy_index()
        with connect_workspace_db(self.workspace) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS workspace_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS workspace_artifacts_thread_created
                    ON workspace_artifacts(thread_id, created_at);
                """
            )
            connection.commit()
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT version FROM schema_migrations WHERE component = ?",
                (_SCHEMA_COMPONENT,),
            ).fetchone()
            version = int(row["version"]) if row is not None else 0
            if version >= _SCHEMA_VERSION:
                return
            for record in legacy_records:
                self._write_record(connection, record, keep_existing=True)
            connection.execute(
                """
                INSERT INTO schema_migrations(component, version)
                VALUES (?, ?)
                ON CONFLICT(component) DO UPDATE SET version=excluded.version
                """,
                (_SCHEMA_COMPONENT, _SCHEMA_VERSION),
            )

    def _read_legacy_index(self) -> list[ArtifactRecord]:
        records: list[ArtifactRecord] = []
        if not self.index_path.exists():
            return records
        with self.index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    records.append(ArtifactRecord.model_validate(json.loads(text)))
                except Exception:
                    continue
        return records

    def _read_all(self, *, thread_id: str = "") -> list[ArtifactRecord]:
        query = "SELECT payload_json FROM workspace_artifacts"
        params: tuple[Any, ...] = ()
        if thread_id:
            query += " WHERE thread_id = ?"
            params = (str(thread_id),)
        query += " ORDER BY created_at ASC, artifact_id ASC"
        with connect_workspace_db(self.workspace) as connection:
            rows = connection.execute(query, params).fetchall()
        records: list[ArtifactRecord] = []
        for row in rows:
            try:
                records.append(
                    ArtifactRecord.model_validate(json.loads(str(row["payload_json"])))
                )
            except Exception:
                continue
        return records

    @staticmethod
    def _write_record(
        connection: Any,
        record: ArtifactRecord,
        *,
        keep_existing: bool = False,
    ) -> None:
        conflict = "DO NOTHING" if keep_existing else (
            "DO UPDATE SET "
            "thread_id=excluded.thread_id, "
            "payload_json=excluded.payload_json, "
            "created_at=excluded.created_at, "
            "updated_at=excluded.updated_at"
        )
        connection.execute(
            f"""
            INSERT INTO workspace_artifacts (
                artifact_id, thread_id, payload_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(artifact_id) {conflict}
            """,
            (
                record.artifact_id,
                record.thread_id,
                _dump_json(record.model_dump(mode="json")),
                float(record.created_at),
                float(record.updated_at),
            ),
        )

    def _upsert(self, record: ArtifactRecord) -> None:
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._write_record(connection, record)


__all__ = ["ArtifactRegistry", "infer_renderer"]
