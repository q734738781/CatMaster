from __future__ import annotations

import asyncio
import csv
import difflib
import json
import logging
import mimetypes
import os
import re
import shutil
import sqlite3
import tempfile
import zipfile
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
import uvicorn

from catmaster.specialists import build_specialist_runner, default_thread_interrupt_on
from catmaster.specialists.streaming_runner import StreamingSpecialistRunner
from catmaster.runtime.observability_store import ObservabilityStore
from catmaster.research.knowledge_graph.models import (
    EdgeCreateRequest,
    ExperimentBlockedRequest,
    ExperimentCreateRequest,
    ExperimentLaunchRequest,
    GraphContextRequest,
    GraphCreateRequest,
    GraphPatchRequest,
    GraphPlanningRequest,
    HypothesisCreateRequest,
    NodePatchRequest,
    RefCreateRequest,
    ResultCreateRequest,
    ThreadGraphBindingRequest,
)
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.research.knowledge_graph.store import ResearchGraphConflict
from catmaster.runtime.self_evolution import SelfEvolutionCoordinator, SelfEvolutionStore
from catmaster.runtime.self_evolution.models import LearningCandidate, SKILL_GROUPS
from catmaster.runtime.self_evolution.promotion import PromotionConflict, PromotionManager
from catmaster.runtime.self_evolution.settings import resolve_self_evolution_mode
from catmaster.structures.models import (
    SaveStructureRequest,
    StructureOpenRequest,
    TRANSFORM_REQUEST_ADAPTER,
)
from catmaster.tools.base import ensure_project_space_layout, system_root

from .agent_loop import ThreadAgentLoopService
from .artifact_registry import ArtifactRegistry
from .auth import AuthIdentity, AuthManager, SESSION_COOKIE_NAME, SESSION_TTL_SECONDS
from .session_registry import SessionRegistry
from .thread_events import ThreadEventBroker, format_sse
from .projections import project_event, project_message, project_messages, project_thread
from .projections.files import project_artifact
from .projections.common import decode_public_cursor, encode_public_cursor, redact_internal_text
from .projections.monitor import project_monitor_snapshot
from .projections.messages import (
    project_citation_items,
    project_part,
    project_todo_items,
)
from .projections.models import (
    DeveloperDiagnosticsPageEnvelope,
    PublicEvent,
    PublicItemPageEnvelope,
    PublicMessagePageEnvelope,
    PublicPartEnvelope,
    PublicPartPageEnvelope,
    PublicResumeEnvelope,
    PublicStopEnvelope,
    PublicSubmitEnvelope,
    PublicTextPageEnvelope,
    PublicThreadEnvelope,
    PublicThreadListEnvelope,
)
from .projections.self_evolution import (
    project_self_evolution_candidate,
    project_self_evolution_job,
    project_self_evolution_payload,
)
from .thread_models import (
    ThreadCreateRequest,
    ThreadPatchRequest,
    ThreadResumeRequest,
    ThreadStopRequest,
    ThreadSubmitRequest,
)
from .thread_store import ThreadStore
from .structure_api import (
    StructureFormatLossError,
    StructureSerializationError,
    StructureVersionConflict,
    apply_transform,
    get_trajectory_frame,
    get_trajectory_meta,
    open_structure,
    save_structure,
)

TEXT_PREVIEW_LIMIT_BYTES = 64_000
DIAGNOSTICS_PAGE_LIMIT_CHARS = 64_000
SELF_EVOLUTION_DIFF_LIMIT_CHARS = 60_000
logger = logging.getLogger(__name__)
TEXT_KIND_PROBE_BYTES = 8_192
AUTO_TEXT_KIND_MAX_BYTES = 8 * 1024 * 1024
DIRECTORY_PREVIEW_LIMIT = 40
UPLOAD_LIMIT_BYTES = 512 * 1024 * 1024
ARCHIVE_ENTRY_LIMIT = 20_000
ARCHIVE_TOTAL_BYTES_LIMIT = 2 * 1024 * 1024 * 1024
UNZIP_ENTRY_LIMIT = 20_000
UNZIP_TOTAL_BYTES_LIMIT = 2 * 1024 * 1024 * 1024
STRUCTURE_FILE_SUFFIXES = {
    ".cif",
    ".cssr",
    ".extxyz",
    ".gro",
    ".mol",
    ".mol2",
    ".pdb",
    ".sdf",
    ".traj",
    ".vasp",
    ".xyz",
}
STRUCTURE_FILE_NAMES = {"POSCAR", "CONTCAR", "OUTCAR", "XDATCAR"}
VOLUME_FILE_SUFFIXES = {".cube", ".xsf"}
VOLUME_FILE_NAMES = {"CHGCAR", "LOCPOT", "ELFCAR"}
MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx", ".rst"}
JSON_SUFFIXES = {".json", ".jsonl", ".geojson"}
PDF_SUFFIXES = {".pdf"}
TEXTLIKE_SUFFIXES = {
    ".csv",
    ".log",
    ".out",
    ".patch",
    ".py",
    ".sh",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
} | MARKDOWN_SUFFIXES | JSON_SUFFIXES
_TEXT_ALLOWED_CONTROL_BYTES = {7, 8, 9, 10, 12, 13, 27}
_THREAD_LANE_ALIASES = {
    "litreview": "literature_review",
    "literature": "literature_review",
}
_HIDDEN_USER_DIRECTORY_NAMES = {
    ".deepagents",
    ".git",
    ".pip-cache",
    ".venv",
    "__pycache__",
    "metadata",
}


def _public_dump(value: Any) -> dict[str, Any]:
    return value.model_dump(mode="json", exclude_none=True)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return ""


def _diagnostics_json_page(
    value: Any,
    *,
    cursor: str,
    limit: int,
    cursor_identity: str,
    full_content_ref: str,
) -> dict[str, Any]:
    content = _canonical_json(value)
    start = 0
    if cursor:
        position = decode_public_cursor(
            cursor,
            kind="diagnostics_json",
            identity=cursor_identity,
        )
        if not isinstance(position, int) or position < 0 or position > len(content):
            raise ValueError("Diagnostics cursor is invalid or stale.")
        start = position
    capped_limit = min(
        256_000,
        max(1_000, int(limit or DIAGNOSTICS_PAGE_LIMIT_CHARS)),
    )
    visible = content[start:start + capped_limit]
    end = start + len(visible)
    has_more = end < len(content)
    return {
        "warning": (
            "Internal diagnostics may contain paths, provider data, "
            "and complete tool payloads."
        ),
        "content_type": "application/json",
        "content": visible,
        "page": {
            "shown_count": len(visible),
            "total_count": len(content),
            "total_unknown": False,
            "truncated": has_more,
            "next_cursor": (
                encode_public_cursor(
                    "diagnostics_json",
                    cursor_identity,
                    end,
                )
                if has_more
                else ""
            ),
            "full_content_ref": full_content_ref,
            "unit": "characters",
            "range_start": start,
            "range_end": end,
        },
    }


def _observed_tool_input(payload: Dict[str, Any]) -> Any:
    for key in ("params_full", "raw_params", "input", "args"):
        value = payload.get(key)
        if value not in (None, ""):
            return value
    compact = payload.get("params_compact")
    if isinstance(compact, str) and compact.strip():
        try:
            return json.loads(compact)
        except Exception:
            return compact
    return {}


def _tool_source_index(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        events = ObservabilityStore(run_dir).read_events_page(
            names=["TOOL_CALL_START", "TOOL_RAW_INPUT"],
            channel="callback",
            limit=2000,
        ).get("events") or []
    except Exception:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        tool = str(payload.get("tool") or payload.get("tool_name") or event.get("tool") or "").strip()
        source = str(payload.get("agent_name") or event.get("agent_name") or "").strip()
        if not tool or not source:
            continue
        out.setdefault(tool, []).append(
            {
                "agent_name": source,
                "subagent_source": source,
                "input_key": _canonical_json(_observed_tool_input(payload)),
            }
        )
    return out


def _enrich_thread_message_tool_sources(messages: list[dict[str, Any]], *, workspace: Path) -> list[dict[str, Any]]:
    indexes: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for message in messages:
        meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
        run_id = str(meta.get("run_id") or "").strip()
        if not run_id:
            continue
        if run_id not in indexes:
            indexes[run_id] = _tool_source_index(system_root(workspace) / "runs" / run_id)
        index = indexes.get(run_id) or {}
        for part in list(message.get("parts") or []):
            if not isinstance(part, dict) or part.get("type") != "tool-call":
                continue
            part_meta = part.get("meta") if isinstance(part.get("meta"), dict) else {}
            if part_meta.get("agent_name") or part_meta.get("subagent_source"):
                continue
            tool = str(part.get("tool") or part_meta.get("tool") or "").strip()
            if not tool:
                continue
            candidates = list(index.get(tool) or [])
            if not candidates:
                continue
            input_key = _canonical_json(part_meta.get("input") if "input" in part_meta else part.get("input") or {})
            selected = None
            for candidate in reversed(candidates):
                if input_key and candidate.get("input_key") == input_key:
                    selected = candidate
                    break
            selected = selected or candidates[-1]
            enriched_meta = {**part_meta, "agent_name": selected["agent_name"], "subagent_source": selected["subagent_source"]}
            part["meta"] = enriched_meta
    return messages


def _filter_unavailable_artifact_parts(
    messages: list[dict[str, Any]],
    *,
    available_artifact_ids: set[str],
) -> list[dict[str, Any]]:
    """Hide historical artifact cards whose workspace files do not exist."""
    for message in messages:
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        message["parts"] = [
            part
            for part in parts
            if not (
                isinstance(part, dict)
                and part.get("type") == "artifact"
                and str(part.get("artifact_id") or "") not in available_artifact_ids
            )
        ]
    return messages


_THREAD_ENTRYPOINTS = [
    {
        "id": "research",
        "label": "Research",
        "summary": "Research coordinator with delegation to experiment, writing, peer review, and literature specialists.",
    },
    {
        "id": "experiment",
        "label": "Experiment",
        "summary": "Computation and managed-execution specialist entry for bounded calculations and file-producing workflows.",
    },
    {
        "id": "writing",
        "label": "Writing",
        "summary": "Manuscript, report, response, and author-facing scientific writing specialist.",
    },
    {
        "id": "peer_review",
        "label": "Peer Review",
        "summary": "Reviewer-style critique and manuscript risk assessment specialist.",
    },
    {
        "id": "literature_review",
        "label": "Literature Review",
        "summary": "Focused literature synthesis entry backed by the literature/deep-research lane.",
    },
]
_SUPPORTED_THREAD_ENTRYPOINTS = {str(item["id"]) for item in _THREAD_ENTRYPOINTS}
_THREAD_PERMISSION_ALIASES = {
    "ask": "hitl",
    "manual": "hitl",
    "review": "hitl",
    "hitl": "hitl",
    "human": "hitl",
    "auto": "auto",
    "auto_approve": "auto",
    "autoapprove": "auto",
    "automatic": "auto",
}


def _normalize_thread_permission_mode(value: Any, *, default: str = "auto") -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return default
    mode = _THREAD_PERMISSION_ALIASES.get(raw)
    if mode is None:
        raise ValueError("permission_mode must be 'hitl' or 'auto'.")
    return mode


def _thread_event_run_id(data: Dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return ""
    for key in ("run_id", "active_run_id"):
        text = str(data.get(key) or "").strip()
        if text:
            return text
    receipt = data.get("receipt") if isinstance(data.get("receipt"), dict) else {}
    text = str(receipt.get("run_id") or "").strip()
    if text:
        return text
    message = data.get("message") if isinstance(data.get("message"), dict) else {}
    meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
    return str(meta.get("run_id") or "").strip()


def _thread_permission_mode(thread: Any, override: Any = "") -> str:
    if str(override or "").strip():
        return _normalize_thread_permission_mode(override)
    meta = getattr(thread, "meta", None)
    if not isinstance(meta, dict):
        meta = {}
    return _normalize_thread_permission_mode(meta.get("permission_mode"), default="auto")


def _interrupt_on_for_permission_mode(permission_mode: Any) -> dict[str, Any]:
    mode = _normalize_thread_permission_mode(permission_mode)
    if mode == "auto":
        return {}
    return default_thread_interrupt_on()

_AUTH_IDENTITY: ContextVar[AuthIdentity | None] = ContextVar("catmaster_webui_auth_identity", default=None)


def _serialize_choices(choices: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"label": str(label), "value": str(value)} for label, value in choices]


def _serialize_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for card in cards:
        out.append(
            {
                "run_name": str(card.get("run_name") or ""),
                "headline": str(card.get("headline") or ""),
                "summary": str(card.get("summary") or ""),
                "next_actions": [str(item) for item in list(card.get("next_actions") or []) if str(item).strip()],
                "status": str(card.get("status") or ""),
                "source": str(card.get("source") or ""),
                "model_name": str(card.get("model_name") or ""),
                "start_time": str(card.get("start_time") or ""),
                "project_space": str(card.get("project_space") or ""),
            }
        )
    return out


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    text = str(value).strip()
    if not text:
        return int(default)
    try:
        return int(text)
    except Exception:
        return int(default)


def _split_csv(value: str = "") -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _workspace_for_request(registry: SessionRegistry, session, project_space: str = "", *, create: bool = False) -> tuple[Optional[Path], str]:
    value = str(project_space or "").strip()
    if value:
        target = session.resolve_workspace_by_name(value)
        if target is None:
            raise HTTPException(status_code=404, detail=f"Project space not found: {value}")
        ok, message = session.open_workspace(str(target), create=create, set_current=False)
        if not ok:
            raise HTTPException(status_code=404 if not create else 400, detail=message)
        return target.expanduser().resolve(), registry._project_space_name_from_path(str(target), root=session.workspace_root) or value
    current = str(session.current_workspace_path() or "").strip()
    if current:
        target = Path(current).expanduser().resolve()
        return target, registry._project_space_name_from_path(str(target), root=session.workspace_root) or target.name
    return None, ""


def _workspace_root_for_session(session, *, workspace: Optional[Path] = None) -> Path:
    if isinstance(workspace, Path):
        workspace_root = workspace.expanduser().resolve()
    else:
        workspace_path = str(session.current_workspace_path() or "").strip()
        if not workspace_path:
            raise HTTPException(status_code=400, detail="Open a project space first.")
        workspace_root = Path(workspace_path).expanduser().resolve()
    if not workspace_root:
        raise HTTPException(status_code=400, detail="Open a project space first.")
    if not workspace_root.exists() or not workspace_root.is_dir():
        raise HTTPException(status_code=404, detail="Project space not found.")
    return workspace_root


def _resolve_workspace_entry(session, rel_path: str = "", *, workspace: Optional[Path] = None) -> tuple[Path, Path, str]:
    workspace_root = _workspace_root_for_session(session, workspace=workspace)
    requested = str(rel_path or "").strip().strip("/")
    candidate = workspace_root if not requested else (workspace_root / requested).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Requested path escapes the project space.") from exc
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Requested path was not found.")
    rel_text = "" if candidate == workspace_root else str(candidate.relative_to(workspace_root)).replace("\\", "/")
    if rel_text and rel_text.split("/", 1)[0] != "files":
        raise HTTPException(status_code=404, detail="Requested user file was not found.")
    if any(
        part.startswith(".") or part in _HIDDEN_USER_DIRECTORY_NAMES
        for part in Path(rel_text).parts[1:]
    ):
        raise HTTPException(status_code=404, detail="Requested user file was not found.")
    return workspace_root, candidate, rel_text


def _resolve_workspace_destination(session, rel_path: str = "", *, workspace: Optional[Path] = None) -> tuple[Path, Path, str]:
    workspace_root = _workspace_root_for_session(session, workspace=workspace)
    requested = str(rel_path or "").strip().strip("/")
    candidate = workspace_root if not requested else (workspace_root / requested).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Requested path escapes the project space.") from exc
    rel_text = "" if candidate == workspace_root else str(candidate.relative_to(workspace_root)).replace("\\", "/")
    if rel_text and rel_text.split("/", 1)[0] != "files":
        raise HTTPException(status_code=400, detail="Uploads are limited to user files.")
    if any(
        part.startswith(".") or part in _HIDDEN_USER_DIRECTORY_NAMES
        for part in Path(rel_text).parts[1:]
    ):
        raise HTTPException(status_code=400, detail="Uploads are limited to user files.")
    return workspace_root, candidate, rel_text


def _resolve_workspace_mutation_entry(session, rel_path: str, *, workspace: Optional[Path] = None) -> tuple[Path, Path, str]:
    workspace_root = _workspace_root_for_session(session, workspace=workspace)
    requested = str(rel_path or "").strip().strip("/")
    if not requested:
        raise HTTPException(status_code=400, detail="Refusing to modify the project-space root.")
    if "\x00" in requested:
        raise HTTPException(status_code=400, detail="Requested path is invalid.")
    parts = Path(requested.replace("\\", "/")).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise HTTPException(status_code=400, detail="Requested path is invalid.")
    candidate = workspace_root.joinpath(*parts)
    try:
        candidate.parent.resolve().relative_to(workspace_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Requested path escapes the project space.") from exc
    if not candidate.exists() and not candidate.is_symlink():
        raise HTTPException(status_code=404, detail="Requested path was not found.")
    if not candidate.is_symlink():
        try:
            candidate.resolve().relative_to(workspace_root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Requested path escapes the project space.") from exc
    rel_text = str(candidate.relative_to(workspace_root)).replace("\\", "/")
    if rel_text.split("/", 1)[0] != "files":
        raise HTTPException(status_code=400, detail="Only user files can be modified.")
    if any(
        part.startswith(".") or part in _HIDDEN_USER_DIRECTORY_NAMES
        for part in Path(rel_text).parts[1:]
    ):
        raise HTTPException(status_code=400, detail="Only user files can be modified.")
    return workspace_root, candidate, rel_text


def _safe_upload_filename(filename: str) -> str:
    name = Path(str(filename or "").replace("\\", "/")).name.strip()
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Upload filename is required.")
    if "/" in name or "\\" in name or "\x00" in name:
        raise HTTPException(status_code=400, detail="Upload filename is invalid.")
    if name.startswith(".") or name in _HIDDEN_USER_DIRECTORY_NAMES:
        raise HTTPException(status_code=400, detail="Hidden or internal filenames are not accepted.")
    return name


def _zip_info_is_symlink(info: zipfile.ZipInfo) -> bool:
    return ((int(info.external_attr) >> 16) & 0o170000) == 0o120000


def _safe_zip_member_path(root: Path, member_name: str) -> Path:
    raw_name = str(member_name or "").replace("\\", "/")
    if not raw_name or raw_name.startswith("/"):
        raise HTTPException(status_code=400, detail="Zip archive contains an invalid path.")
    parts = Path(raw_name).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise HTTPException(status_code=400, detail="Zip archive contains an unsafe path.")
    if any(part.startswith(".") or part in _HIDDEN_USER_DIRECTORY_NAMES for part in parts):
        raise HTTPException(status_code=400, detail="Zip archive contains a hidden or internal path.")
    target = root.joinpath(*parts).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Zip archive contains a path that escapes the target directory.") from exc
    return target


def _extract_zip_to_workspace(
    *,
    zip_path: Path,
    target_dir: Path,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    target_root = target_dir.resolve()
    extracted: list[dict[str, Any]] = []
    total_bytes = 0
    entry_count = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for info in archive.infolist():
                if not info.filename or info.filename.endswith("/"):
                    target_directory = _safe_zip_member_path(target_root, info.filename.rstrip("/"))
                    target_directory.mkdir(parents=True, exist_ok=True)
                    continue
                if _zip_info_is_symlink(info):
                    continue
                entry_count += 1
                if entry_count > UNZIP_ENTRY_LIMIT:
                    raise HTTPException(status_code=413, detail="Zip archive contains too many files.")
                total_bytes += int(info.file_size)
                if total_bytes > UNZIP_TOTAL_BYTES_LIMIT:
                    raise HTTPException(status_code=413, detail="Zip archive expands beyond the maximum allowed size.")

                target = _safe_zip_member_path(target_root, info.filename)
                if target.exists() and not overwrite:
                    raise HTTPException(status_code=409, detail=f"Zip entry already exists: {info.filename}")
                if target.exists() and target.is_dir():
                    raise HTTPException(status_code=409, detail=f"Zip entry conflicts with a directory: {info.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                extracted.append({"path": str(target.relative_to(target_root)).replace("\\", "/"), "size": int(info.file_size)})
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid zip archive.") from exc
    return extracted


def _looks_like_text_file(path: Path, *, file_size: int | None = None) -> bool:
    try:
        size = int(path.stat().st_size if file_size is None else file_size)
    except Exception:
        size = 0
    if size <= 0:
        return True
    if size > AUTO_TEXT_KIND_MAX_BYTES:
        return False
    try:
        with path.open("rb") as handle:
            sample = handle.read(min(TEXT_KIND_PROBE_BYTES, size))
    except Exception:
        return False
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    control_count = sum(1 for byte in sample if byte < 32 and byte not in _TEXT_ALLOWED_CONTROL_BYTES)
    if control_count / max(1, len(sample)) > 0.02:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return control_count == 0
    return True


def _entry_preview_kind(path: Path, *, mime_type: str = "", file_size: int | None = None) -> str:
    suffix = path.suffix.lower()
    if path.name.upper() in VOLUME_FILE_NAMES or suffix in VOLUME_FILE_SUFFIXES:
        return "volume"
    if path.name.upper() in STRUCTURE_FILE_NAMES or suffix in STRUCTURE_FILE_SUFFIXES:
        return "structure"
    if mime_type.startswith("image/"):
        return "image"
    if suffix in MARKDOWN_SUFFIXES:
        return "markdown"
    if suffix in JSON_SUFFIXES:
        return "json"
    if suffix in {".csv", ".tsv"}:
        return "csv"
    if suffix in PDF_SUFFIXES or mime_type == "application/pdf":
        return "pdf"
    if mime_type.startswith("text/") or suffix in TEXTLIKE_SUFFIXES:
        return "text"
    if _looks_like_text_file(path, file_size=file_size):
        return "text"
    return "binary"


def _directory_has_children(path: Path) -> bool:
    try:
        next(
            item
            for item in path.iterdir()
            if (
                item.name not in _HIDDEN_USER_DIRECTORY_NAMES
                and not item.name.startswith(".")
                and not item.is_symlink()
            )
        )
    except StopIteration:
        return False
    except Exception:
        return False
    return True


def _serialize_tree_entry(path: Path, *, workspace_root: Path) -> dict[str, Any]:
    stat = path.stat()
    rel_path = "" if path == workspace_root else str(path.relative_to(workspace_root)).replace("\\", "/")
    mime_type = mimetypes.guess_type(path.name)[0] or ""
    node_type = "directory" if path.is_dir() else "file"
    return {
        "name": path.name if rel_path else workspace_root.name,
        "path": rel_path,
        "node_type": node_type,
        "size": int(stat.st_size),
        "modified_ts": float(stat.st_mtime),
        "has_children": _directory_has_children(path) if node_type == "directory" else False,
        "preview_kind": "directory" if node_type == "directory" else _entry_preview_kind(path, mime_type=mime_type, file_size=int(stat.st_size)),
    }


def _list_directory_entries(
    directory: Path,
    *,
    workspace_root: Path,
    limit: int | None = 500,
) -> list[dict[str, Any]]:
    visible_directory = workspace_root / "files" if directory == workspace_root else directory
    if not visible_directory.is_dir():
        return []
    children = [
        child
        for child in visible_directory.iterdir()
        if (
            child.name not in _HIDDEN_USER_DIRECTORY_NAMES
            and not child.name.startswith(".")
            and not child.is_symlink()
        )
    ]
    children.sort(key=lambda item: (0 if item.is_dir() else 1, item.name.lower()))
    selected = children if limit is None else children[: max(0, int(limit))]
    return [_serialize_tree_entry(child, workspace_root=workspace_root) for child in selected]


def _read_text_preview(
    path: Path,
    *,
    cursor: int = 0,
    limit: int = TEXT_PREVIEW_LIMIT_BYTES,
) -> tuple[str, dict[str, Any]]:
    total_bytes = int(path.stat().st_size)
    start = min(total_bytes, max(0, int(cursor or 0)))
    capped_limit = min(256_000, max(1_000, int(limit or TEXT_PREVIEW_LIMIT_BYTES)))
    with path.open("rb") as handle:
        handle.seek(start)
        raw = handle.read(capped_limit)
    next_cursor = start + len(raw)
    truncated = next_cursor < total_bytes
    return raw.decode("utf-8", errors="replace"), {
        "shown_count": len(raw),
        "total_count": total_bytes,
        "total_unknown": False,
        "truncated": truncated,
        "next_cursor": str(next_cursor) if truncated else "",
        "full_content_ref": "",
        "unit": "bytes",
        "range_start": start,
    }


def _json_human_view(text: str, *, truncated: bool) -> dict[str, Any]:
    if truncated:
        return {}
    try:
        value = json.loads(text)
    except Exception:
        return {}
    if isinstance(value, dict):
        fields: list[dict[str, str]] = []
        collections: list[dict[str, Any]] = []
        for key, item in value.items():
            label = str(key).replace("_", " ").strip().capitalize() or "Value"
            if isinstance(item, bool):
                fields.append({"label": label, "value": "Yes" if item else "No"})
            elif isinstance(item, (str, int, float)) or item is None:
                fields.append({"label": label, "value": "" if item is None else str(item)})
            elif isinstance(item, list):
                scalar_items = [
                    str(row)
                    for row in item[:100]
                    if isinstance(row, (str, int, float, bool))
                ]
                if scalar_items:
                    collections.append(
                        {
                            "label": label,
                            "items": scalar_items,
                            "shown_count": len(scalar_items),
                            "total_count": len(item),
                            "truncated": len(scalar_items) < len(item),
                        }
                    )
        return {
            "kind": "record",
            "fields": fields[:100],
            "collections": collections[:20],
        }
    if isinstance(value, list):
        scalar_items = [
            str(row)
            for row in value[:200]
            if isinstance(row, (str, int, float, bool))
        ]
        return {
            "kind": "list",
            "items": scalar_items,
            "shown_count": len(scalar_items),
            "total_count": len(value),
            "truncated": len(scalar_items) < len(value),
        }
    return {"kind": "value", "value": str(value)}


def _csv_human_view(text: str, *, suffix: str, source_truncated: bool) -> dict[str, Any]:
    delimiter = "\t" if suffix.lower() == ".tsv" else ","
    try:
        rows = list(csv.reader(StringIO(text), delimiter=delimiter))
    except Exception:
        return {}
    if source_truncated and rows:
        rows = rows[:-1]
    if not rows:
        return {"kind": "table", "columns": [], "rows": [], "shown_count": 0, "total_unknown": source_truncated}
    columns = [str(value or f"Column {index + 1}") for index, value in enumerate(rows[0])]
    body = [
        [str(row[index]) if index < len(row) else "" for index in range(len(columns))]
        for row in rows[1:251]
    ]
    return {
        "kind": "table",
        "columns": columns,
        "rows": body,
        "shown_count": len(body),
        "total_count": len(body) if not source_truncated else 0,
        "total_unknown": source_truncated,
        "truncated": source_truncated or len(rows) > 251,
    }


def _read_structure_frames(path: Path, *, limit: int | None = None) -> Optional[tuple[list[Any], int, bool]]:
    try:
        from ase.io import iread as ase_iread
    except Exception:
        return None
    selected: list[Any] = []
    total = 0
    try:
        for frame in ase_iread(str(path), index=":"):
            total += 1
            if limit is None or len(selected) < max(0, int(limit)):
                selected.append(frame)
            if limit is not None and len(selected) >= max(0, int(limit)):
                break
    except Exception:
        return None
    if not total:
        return None
    return selected, total, total > len(selected)


def _parse_outcar_vibration_modes(path: Path, *, atom_count: int) -> list[dict[str, Any]]:
    if path.name.upper() != "OUTCAR" and path.suffix.lower() != ".outcar":
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []

    header_pattern = re.compile(
        r"^\s*(\d+)\s+f(?P<imag>/i)?(?:\s*=)?\s*([\-+\d.Ee]+)\s+THz\s+([\-+\d.Ee]+)\s+2PiTHz\s+([\-+\d.Ee]+)\s+cm-1\s+([\-+\d.Ee]+)\s+meV"
    )
    coord_header_pattern = re.compile(r"^\s*X\s+Y\s+Z\s+dx\s+dy\s+dz\s*$", re.IGNORECASE)
    modes: list[dict[str, Any]] = []

    index = 0
    while index < len(lines):
        match = header_pattern.match(lines[index])
        if not match:
            index += 1
            continue

        mode_number = int(match.group(1))
        imaginary = bool(match.group("imag"))
        freq_thz = float(match.group(3))
        freq_2pithz = float(match.group(4))
        freq_cm1 = float(match.group(5))
        freq_mev = float(match.group(6))

        cursor = index + 1
        while cursor < len(lines) and not coord_header_pattern.match(lines[cursor]):
            cursor += 1
        if cursor >= len(lines):
            index += 1
            continue

        displacements: list[list[float]] = []
        cursor += 1
        while cursor < len(lines) and len(displacements) < atom_count:
            tokens = lines[cursor].split()
            cursor += 1
            if len(tokens) < 6:
                if displacements:
                    break
                continue
            try:
                dx, dy, dz = (float(tokens[-3]), float(tokens[-2]), float(tokens[-1]))
            except Exception:
                break
            displacements.append([dx, dy, dz])

        if len(displacements) == atom_count:
            sign = "imag" if imaginary else "real"
            modes.append(
                {
                    "mode_index": len(modes),
                    "mode_number": mode_number,
                    "imaginary": imaginary,
                    "frequency_thz": freq_thz,
                    "frequency_2pithz": freq_2pithz,
                    "frequency_cm1": freq_cm1,
                    "frequency_mev": freq_mev,
                    "label": f"{mode_number}: {sign} {freq_cm1:.2f} cm-1",
                    "displacements": displacements,
                }
            )
        index = cursor

    return modes


def _structure_file_type(path: Path) -> str:
    upper_name = path.name.upper()
    suffix = path.suffix.lower()
    if upper_name == "OUTCAR" or suffix == ".outcar":
        return "VaspOutcar"
    if upper_name in {"POSCAR", "CONTCAR"} or suffix in {".poscar", ".contcar", ".vasp"}:
        return "VaspPoscar"
    return {
        ".cif": "Cif",
        ".mol": "Mol",
        ".mol2": "Mol2",
        ".pdb": "Pdb",
        ".sdf": "Mol",
        ".xyz": "Xyz",
    }.get(suffix, "")


def _build_structure_payload(path: Path, *, ctx: str, rel_path: str, project_space: str = "") -> Optional[dict[str, Any]]:
    try:
        from ase.io import write as ase_write
    except Exception:
        return None
    frame_bundle = _read_structure_frames(path, limit=1)
    if frame_bundle is None:
        return None
    frames, total_frames, _frames_omitted = frame_bundle
    atoms = frames[0]
    if (
        path.name.upper() == "XDATCAR"
        or path.suffix.lower() in {".traj", ".xdatcar", ".extxyz", ".xyz"}
    ):
        try:
            from catmaster.structures.trajectory import trajectory_frame_count

            total_frames = trajectory_frame_count(path)
        except Exception:
            total_frames = max(1, total_frames)
    periodic = bool(any(bool(value) for value in atoms.pbc))
    viewer_format = "cif" if periodic else "xyz"
    buffer: StringIO | BytesIO = StringIO() if viewer_format == "xyz" else BytesIO()
    try:
        ase_write(buffer, atoms, format=viewer_format)
    except Exception:
        return None
    cell_lengths = [float(value) for value in getattr(atoms.cell, "lengths", lambda: [])()]
    cell_angles = [float(value) for value in getattr(atoms.cell, "angles", lambda: [])()]
    cell_vectors = [[float(component) for component in vector] for vector in getattr(atoms.cell, "array", [])]
    symbols = [str(symbol) for symbol in atoms.get_chemical_symbols()]
    unique_elements = sorted(set(symbols))
    element_counts: Dict[str, int] = {}
    for symbol in symbols:
        element_counts[symbol] = int(element_counts.get(symbol, 0)) + 1
    is_vibration_source = bool(path.name.upper() == "OUTCAR" or path.suffix.lower() == ".outcar")
    vibration_modes = _parse_outcar_vibration_modes(path, atom_count=len(atoms)) if is_vibration_source else []
    is_trajectory_source = bool(
        path.name.upper() == "XDATCAR"
        or path.suffix.lower() in {".xdatcar", ".traj"}
        or (total_frames > 1 and not is_vibration_source)
    )
    viewer_source_mode = "inline"
    viewer_source_url = ""
    viewer_source_file_type = ""
    project_param = f"&project_space={quote(project_space)}" if project_space else ""
    if is_vibration_source and vibration_modes:
        viewer_source_mode = "url"
        viewer_source_url = f"/api/session/{ctx}/files/structure-vibration?path={quote(rel_path)}{project_param}"
        viewer_source_file_type = "Xyz"
    elif is_vibration_source:
        viewer_source_mode = "url"
        viewer_source_url = f"/api/session/{ctx}/files/view?path={quote(rel_path)}{project_param}"
        viewer_source_file_type = "VaspOutcar"
    elif is_trajectory_source:
        viewer_source_mode = "url"
        viewer_source_url = f"/api/session/{ctx}/files/structure-animation?path={quote(rel_path)}{project_param}"
        viewer_source_file_type = "Xyz"
    else:
        viewer_source_mode = "url"
        viewer_source_url = f"/api/session/{ctx}/files/view?path={quote(rel_path)}{project_param}"
        viewer_source_file_type = _structure_file_type(path)
    return {
        "formula": str(atoms.get_chemical_formula() or ""),
        "atom_count": int(len(atoms)),
        "periodic": periodic,
        "viewer_format": viewer_format,
        "viewer_text": (
            buffer.getvalue()
            if isinstance(buffer, StringIO)
            else buffer.getvalue().decode("utf-8", errors="replace")
        ),
        "cell_lengths": cell_lengths,
        "cell_angles": cell_angles,
        "cell_vectors": cell_vectors,
        "pbc": [bool(value) for value in atoms.pbc],
        "elements": unique_elements,
        "element_counts": element_counts,
        "viewer_source_mode": viewer_source_mode,
        "viewer_source_url": viewer_source_url,
        "viewer_source_file_type": viewer_source_file_type,
        "supports_animation": bool(is_trajectory_source),
        "supports_vibration": bool(vibration_modes),
        "frame_count": int(total_frames),
        "frames_truncated": False,
        "vibration_modes": [
            {
                "mode_index": int(mode["mode_index"]),
                "mode_number": int(mode["mode_number"]),
                "imaginary": bool(mode["imaginary"]),
                "frequency_thz": float(mode["frequency_thz"]),
                "frequency_2pithz": float(mode["frequency_2pithz"]),
                "frequency_cm1": float(mode["frequency_cm1"]),
                "frequency_mev": float(mode["frequency_mev"]),
                "label": str(mode["label"]),
            }
            for mode in vibration_modes
        ],
        "vibration_source_url": viewer_source_url if vibration_modes else "",
    }


def _structure_animation_response(*, session, rel_path: str, workspace: Optional[Path] = None) -> StreamingResponse:
    _workspace_root, candidate, _normalized_path = _resolve_workspace_entry(session, rel_path, workspace=workspace)
    if not candidate.is_file():
        raise HTTPException(status_code=400, detail="Only files can be viewed.")

    def _stream_frames():
        from ase.io import iread as ase_iread
        from ase.io import write as ase_write

        yielded = False
        for atoms in ase_iread(str(candidate), index=":"):
            payload = StringIO()
            ase_write(payload, atoms, format="extxyz")
            yielded = True
            yield payload.getvalue()
        if not yielded:
            raise ValueError("Structure trajectory contains no frames.")

    return StreamingResponse(_stream_frames(), media_type="chemical/x-xyz")


def _structure_view_response(*, session, rel_path: str, workspace: Optional[Path] = None) -> FileResponse:
    _workspace_root, candidate, _normalized_path = _resolve_workspace_entry(session, rel_path, workspace=workspace)
    if not candidate.is_file():
        raise HTTPException(status_code=400, detail="Only files can be viewed.")
    upper_name = candidate.name.upper()
    if upper_name in {"OUTCAR", "XDATCAR", "POSCAR", "CONTCAR"} or candidate.suffix.lower() in {".outcar", ".xdatcar", ".poscar", ".contcar", ".vasp"}:
        media_type = "text/plain; charset=utf-8"
    else:
        media_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
    return FileResponse(candidate, media_type=media_type)


def _structure_vibration_response(*, session, rel_path: str, mode_index: int, workspace: Optional[Path] = None) -> Response:
    _workspace_root, candidate, _normalized_path = _resolve_workspace_entry(session, rel_path, workspace=workspace)
    if not candidate.is_file():
        raise HTTPException(status_code=400, detail="Only files can be viewed.")
    frame_bundle = _read_structure_frames(candidate, limit=1)
    if frame_bundle is None:
        raise HTTPException(status_code=400, detail="Structure could not be parsed.")
    frames, _total_frames, _frames_truncated = frame_bundle
    atoms = frames[0]
    modes = _parse_outcar_vibration_modes(candidate, atom_count=len(atoms))
    if not modes:
        raise HTTPException(status_code=400, detail="No vibration modes were found in this OUTCAR.")
    import numpy as np

    if mode_index >= 0:
        if mode_index >= len(modes):
            raise HTTPException(status_code=400, detail="Requested vibration mode is out of range.")
        selected_modes = [modes[mode_index]]
    else:
        selected_modes = modes

    vibration_models = []
    for mode in selected_modes:
        image = atoms.copy()
        frequency_cm1 = float(mode["frequency_cm1"])
        if bool(mode["imaginary"]):
            frequency_cm1 = -abs(frequency_cm1)
        image.info["mode#"] = str(mode["mode_number"])
        image.info["frequency_cm-1"] = frequency_cm1
        image.arrays["mode"] = np.array(mode["displacements"], dtype=float)
        if image.has("masses"):
            del image.arrays["masses"]
        vibration_models.append(image)

    payload = StringIO()
    try:
        from ase.io import write as ase_write

        ase_write(payload, vibration_models, format="extxyz")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to serialize vibration models: {exc}") from exc
    return Response(payload.getvalue(), media_type="chemical/x-xyz")


def _file_content_payload(
    *,
    ctx: str,
    session,
    rel_path: str,
    workspace: Optional[Path] = None,
    cursor: int = 0,
    limit: int = TEXT_PREVIEW_LIMIT_BYTES,
) -> dict[str, Any]:
    workspace_root, candidate, normalized_path = _resolve_workspace_entry(session, rel_path, workspace=workspace)
    project_space = Path(workspace_root).name
    stat = candidate.stat()
    mime_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
    payload: dict[str, Any] = {
        "path": normalized_path,
        "name": candidate.name,
        "node_type": "directory" if candidate.is_dir() else "file",
        "size": int(stat.st_size),
        "modified_ts": float(stat.st_mtime),
        "mime_type": mime_type,
        "download_url": f"/api/session/{ctx}/files/download?path={quote(normalized_path)}&project_space={quote(project_space)}",
    }
    if candidate.is_dir():
        payload["kind"] = "directory"
        directory_entries = _list_directory_entries(
            candidate,
            workspace_root=workspace_root,
            limit=None,
        )
        payload["children"] = directory_entries[:DIRECTORY_PREVIEW_LIMIT]
        directory_total = len(directory_entries)
        payload["page"] = {
            "shown_count": len(payload["children"]),
            "total_count": directory_total,
            "total_unknown": False,
            "truncated": directory_total > len(payload["children"]),
            "next_cursor": "",
            "full_content_ref": (
                f"/api/session/{ctx}/files/tree?path={quote(normalized_path)}"
                f"&project_space={quote(project_space)}"
            ),
            "unit": "items",
        }
        return payload

    kind = _entry_preview_kind(candidate, mime_type=mime_type, file_size=int(stat.st_size))
    payload["kind"] = kind
    if kind == "image":
        payload["page"] = {
            "shown_count": 1,
            "total_count": 1,
            "total_unknown": False,
            "truncated": False,
            "next_cursor": "",
            "full_content_ref": payload["download_url"],
            "unit": "items",
        }
        return payload
    if kind == "volume":
        payload["volume"] = {
            "format": candidate.name.lower() if candidate.name.upper() in VOLUME_FILE_NAMES else candidate.suffix.lower().lstrip("."),
            "source_url": (
                f"/api/session/{ctx}/files/view?path={quote(normalized_path)}"
                f"&project_space={quote(project_space)}"
            ),
            "file_size": int(stat.st_size),
        }
        payload["preview_text"] = ""
        payload["page"] = {
            "shown_count": 1,
            "total_count": 1,
            "total_unknown": False,
            "truncated": False,
            "next_cursor": "",
            "full_content_ref": payload["download_url"],
            "unit": "items",
        }
        return payload
    if kind == "structure":
        payload["structure"] = _build_structure_payload(candidate, ctx=ctx, rel_path=normalized_path, project_space=project_space)
        preview_text, page = _read_text_preview(candidate, cursor=cursor, limit=limit)
        page["full_content_ref"] = (
            f"/api/session/{ctx}/files/content?path={quote(normalized_path)}"
            f"&project_space={quote(project_space)}"
        )
        payload["preview_text"] = preview_text
        payload["page"] = page
        payload["truncated"] = page["truncated"]
        return payload
    if kind == "pdf":
        payload["preview_text"] = ""
        payload["truncated"] = False
        payload["page"] = {
            "shown_count": 1,
            "total_count": 1,
            "total_unknown": False,
            "truncated": False,
            "next_cursor": "",
            "full_content_ref": payload["download_url"],
            "unit": "items",
        }
        return payload
    if kind in {"text", "markdown", "json", "csv"} or mime_type.startswith("text/"):
        preview_text, page = _read_text_preview(candidate, cursor=cursor, limit=limit)
        page["full_content_ref"] = (
            f"/api/session/{ctx}/files/content?path={quote(normalized_path)}"
            f"&project_space={quote(project_space)}"
        )
        payload["preview_text"] = preview_text
        payload["page"] = page
        payload["truncated"] = page["truncated"]
        if cursor <= 0 and kind == "json":
            payload["human_view"] = _json_human_view(
                preview_text,
                truncated=bool(page["truncated"]),
            )
        elif cursor <= 0 and kind == "csv":
            payload["human_view"] = _csv_human_view(
                preview_text,
                suffix=candidate.suffix,
                source_truncated=bool(page["truncated"]),
            )
        return payload
    payload["preview_text"] = ""
    payload["truncated"] = False
    payload["page"] = {
        "shown_count": 0,
        "total_count": int(stat.st_size),
        "total_unknown": False,
        "truncated": False,
        "next_cursor": "",
        "full_content_ref": payload["download_url"],
        "unit": "bytes",
    }
    return payload


async def _upload_workspace_file(
    *,
    session,
    rel_path: str,
    filename: str,
    request: Request,
    overwrite: bool = False,
    unzip: bool = False,
    workspace: Optional[Path] = None,
) -> dict[str, Any]:
    workspace_root, directory, normalized_dir = _resolve_workspace_destination(session, rel_path, workspace=workspace)
    if directory.exists() and not directory.is_dir():
        raise HTTPException(status_code=400, detail="Upload target is not a directory.")
    directory.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_upload_filename(filename)
    if unzip and not safe_name.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Unzip upload only supports .zip files.")
    destination = (directory / safe_name).resolve()
    try:
        destination.relative_to(workspace_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Upload path escapes the project space.") from exc
    if not unzip and destination.exists() and not overwrite:
        raise HTTPException(status_code=409, detail="A file with this name already exists.")
    if not unzip and destination.exists() and destination.is_dir():
        raise HTTPException(status_code=409, detail="A directory with this name already exists.")

    tmp_path = directory / f".{safe_name}.uploading"
    if tmp_path.exists():
        tmp_path.unlink()

    bytes_written = 0
    try:
        with tmp_path.open("wb") as handle:
            async for chunk in request.stream():
                if not chunk:
                    continue
                bytes_written += len(chunk)
                if bytes_written > UPLOAD_LIMIT_BYTES:
                    raise HTTPException(status_code=413, detail="Upload exceeds the maximum allowed size.")
                handle.write(chunk)
        if unzip:
            extracted = _extract_zip_to_workspace(zip_path=tmp_path, target_dir=directory, overwrite=overwrite)
            tmp_path.unlink(missing_ok=True)
            return {
                "ok": True,
                "path": normalized_dir,
                "directory": normalized_dir,
                "unzipped": True,
                "extracted_count": len(extracted),
                "extracted": extracted[:200],
            }
        tmp_path.replace(destination)
    except HTTPException:
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    rel_file = str(destination.relative_to(workspace_root)).replace("\\", "/")
    return {
        "ok": True,
        "path": rel_file,
        "directory": normalized_dir,
        "unzipped": False,
        "file": _serialize_tree_entry(destination, workspace_root=workspace_root),
    }


def _delete_workspace_entry(*, session, rel_path: str, workspace: Optional[Path] = None) -> dict[str, Any]:
    _workspace_root, candidate, normalized_path = _resolve_workspace_mutation_entry(session, rel_path, workspace=workspace)
    if candidate.is_symlink() or candidate.is_file():
        candidate.unlink()
        node_type = "file"
    elif candidate.is_dir():
        shutil.rmtree(candidate)
        node_type = "directory"
    else:
        raise HTTPException(status_code=400, detail="Requested path cannot be deleted.")
    return {"ok": True, "path": normalized_path, "node_type": node_type}


def _archive_workspace_entry(*, session, rel_path: str, workspace: Optional[Path] = None) -> FileResponse:
    workspace_root, candidate, normalized_path = _resolve_workspace_entry(session, rel_path, workspace=workspace)
    if not normalized_path:
        candidate = workspace_root / "files"
        if not candidate.is_dir():
            raise HTTPException(status_code=404, detail="No user files are available to archive.")
    archive_base = candidate.name if normalized_path else f"{workspace_root.name}-files"
    archive_name = f"{archive_base or 'workspace'}.zip"

    temp = tempfile.NamedTemporaryFile(prefix="catmaster-files-", suffix=".zip", delete=False)
    temp_path = Path(temp.name)
    temp.close()

    total_bytes = 0
    entry_count = 0
    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            if candidate.is_file():
                total_bytes = candidate.stat().st_size
                if total_bytes > ARCHIVE_TOTAL_BYTES_LIMIT:
                    raise HTTPException(status_code=413, detail="Archive exceeds the maximum allowed size.")
                archive.write(candidate, arcname=candidate.name)
                entry_count = 1
            else:
                for item in candidate.rglob("*"):
                    relative_parts = item.relative_to(candidate).parts
                    if any(part.startswith(".") or part in _HIDDEN_USER_DIRECTORY_NAMES for part in relative_parts):
                        continue
                    if item.is_symlink() or not item.is_file():
                        continue
                    try:
                        item.resolve().relative_to(workspace_root)
                    except ValueError:
                        continue
                    entry_count += 1
                    if entry_count > ARCHIVE_ENTRY_LIMIT:
                        raise HTTPException(status_code=413, detail="Archive contains too many files.")
                    total_bytes += item.stat().st_size
                    if total_bytes > ARCHIVE_TOTAL_BYTES_LIMIT:
                        raise HTTPException(status_code=413, detail="Archive exceeds the maximum allowed size.")
                    archive.write(item, arcname=str(item.relative_to(candidate.parent)).replace("\\", "/"))
    except HTTPException:
        temp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Archive failed: {exc}") from exc

    return FileResponse(
        temp_path,
        media_type="application/zip",
        filename=archive_name,
        background=BackgroundTask(lambda: temp_path.unlink(missing_ok=True)),
    )


def _runtime_snapshot(session, *, workspace: Optional[Path] = None) -> dict[str, Any]:
    if hasattr(session, "_lock") and hasattr(session, "_runtime_for_workspace_unlocked"):
        with session._lock:
            runtime_state = session._runtime_for_workspace_unlocked(workspace, create=True)
            reporter = runtime_state.reporter
    else:
        reporter = getattr(session, "reporter", None)
    if reporter is None:
        return {
            "active": False,
            "run_name": "",
            "seq": 0,
            "live_state": {},
            "llm": {},
            "graph": {},
            "usage_totals": {},
            "recent_events": [],
        }
    snapshot = reporter.get_snapshot()
    return {
        "active": True,
        "run_name": str(snapshot.get("run_name") or ""),
        "seq": int(snapshot.get("seq") or 0),
        "live_state": snapshot.get("live_state") if isinstance(snapshot.get("live_state"), dict) else {},
        "llm": snapshot.get("llm") if isinstance(snapshot.get("llm"), dict) else {},
        "graph": snapshot.get("graph") if isinstance(snapshot.get("graph"), dict) else {},
        "usage_totals": snapshot.get("usage_totals") if isinstance(snapshot.get("usage_totals"), dict) else {},
        "recent_events": snapshot.get("recent_events") if isinstance(snapshot.get("recent_events"), list) else [],
    }


def _active_run_name(session, runtime: dict[str, Any] | None = None, *, workspace: Optional[Path] = None) -> str:
    runtime_dict = runtime if isinstance(runtime, dict) else {}
    runtime_run = str(runtime_dict.get("run_name") or "").strip()
    if runtime_run:
        return runtime_run
    if hasattr(session, "_lock") and hasattr(session, "_runtime_for_workspace_unlocked"):
        with session._lock:
            runtime_state = session._runtime_for_workspace_unlocked(workspace, create=True)
            info = dict(runtime_state.run_info or {})
    else:
        info = getattr(session, "run_info", None)
    if isinstance(info, dict):
        run_id = str(info.get("run_id") or "").strip()
        if run_id:
            return run_id
    try:
        run_dir = session.get_selected_run_dir(workspace=workspace)
    except TypeError:
        run_dir = session.get_selected_run_dir()
    try:
        run_status = session._load_task_state_status(run_dir)
    except Exception:
        run_status = ""
    if run_dir is not None and run_status in {"running", "starting"}:
        return run_dir.name
    return ""


def _display_run_status(session, run_dir, *, workspace: Optional[Path] = None) -> str:
    try:
        with session._lock:
            status = session._runtime_for_workspace_unlocked(workspace, create=True).run_status
        return str(session._display_status(status, run_dir) or "idle")
    except Exception:
        return "idle"


def _merge_usage_summary(runtime_usage: dict[str, Any] | None, persisted_usage: dict[str, Any] | None) -> dict[str, Any]:
    runtime = dict(runtime_usage or {})
    persisted = dict(persisted_usage or {})
    if not runtime:
        return persisted
    if not persisted:
        return runtime

    merged = dict(persisted)
    count_keys = {
        "calls",
        "input_tokens",
        "input_uncached_tokens",
        "output_tokens",
        "total_tokens",
        "input_cached_tokens",
        "input_cache_read_tokens",
        "input_cache_write_tokens",
        "reasoning_tokens",
        "exact_cost_calls",
        "estimated_cost_calls",
    }
    cost_keys = {
        "cost_usd",
        "exact_cost_usd",
        "estimated_cost_usd",
    }
    for key in count_keys:
        runtime_value = runtime.get(key)
        persisted_value = persisted.get(key)
        if isinstance(runtime_value, (int, float)) and isinstance(persisted_value, (int, float)):
            merged[key] = max(runtime_value, persisted_value)
        elif runtime_value is not None:
            merged[key] = runtime_value

    runtime_cost = runtime.get("cost_usd")
    persisted_cost = persisted.get("cost_usd")
    runtime_has_cost = isinstance(runtime_cost, (int, float))
    persisted_has_cost = isinstance(persisted_cost, (int, float))
    for key in cost_keys:
        runtime_value = runtime.get(key)
        persisted_value = persisted.get(key)
        if isinstance(runtime_value, (int, float)) and isinstance(persisted_value, (int, float)):
            merged[key] = max(runtime_value, persisted_value)
        elif runtime_value is not None:
            merged[key] = runtime_value

    if runtime_has_cost and persisted_has_cost:
        if float(runtime_cost) >= float(persisted_cost):
            if runtime.get("cost_source") not in (None, ""):
                merged["cost_source"] = runtime.get("cost_source")
            if runtime.get("breakdown_usd") not in (None, "", [], {}):
                merged["breakdown_usd"] = runtime.get("breakdown_usd")
            if runtime.get("exact_breakdown_usd") not in (None, "", [], {}):
                merged["exact_breakdown_usd"] = runtime.get("exact_breakdown_usd")
            if runtime.get("by_model") not in (None, "", [], {}):
                merged["by_model"] = runtime.get("by_model")
            if runtime.get("by_role") not in (None, "", [], {}):
                merged["by_role"] = runtime.get("by_role")
    elif runtime_has_cost:
        for key in ("cost_source", "breakdown_usd", "exact_breakdown_usd", "by_model", "by_role"):
            if runtime.get(key) not in (None, "", [], {}):
                merged[key] = runtime.get(key)

    for key, value in runtime.items():
        if key in count_keys or key in cost_keys:
            continue
        if key not in merged or merged.get(key) in (None, "", [], {}):
            merged[key] = value
    return merged


def _stream_patch(session, runtime: dict[str, Any], *, workspace: Optional[Path] = None) -> dict[str, Any]:
    run_name = str(runtime.get("run_name") or "").strip()
    active_run = _active_run_name(session, runtime, workspace=workspace)
    selected_run = run_name
    run_dir = None
    workspace_path = str(workspace or session.current_workspace_path() or "").strip()
    if run_name and workspace_path:
        resolved = session._resolve_run_dir_by_name(run_name, workspace=Path(workspace_path))
        if resolved is not None:
            run_dir = resolved
    if run_dir is None:
        run_dir = session.get_selected_run_dir(workspace=workspace)
        selected_run = run_dir.name if run_dir is not None else selected_run
    persisted_usage = session.read_usage_summary(run_dir) if run_dir is not None else {}
    usage_summary = _merge_usage_summary(runtime.get("usage_totals"), persisted_usage)
    machine_time_summary = session.read_machine_time_summary(run_dir) if run_dir is not None else {}
    return {
        "active_run": active_run,
        "selected_run": selected_run,
        "chat_messages": session.get_chat_messages(workspace=workspace),
        "cards": _serialize_cards(session.list_run_cards(workspace=workspace)),
        "usage_summary": usage_summary,
        "machine_time_summary": machine_time_summary,
        "proposal": session.read_proposal(run_dir, workspace=workspace),
        "todo_items": session.read_todo_items(run_dir),
        "result_text": session.read_result_text(run_dir),
    }


def _pick_selected_run(session, requested_run: str = "", *, lane: str = "", workspace: Optional[Path] = None) -> str:
    selected = str(requested_run or "").strip()
    runs = session.list_runs(workspace=workspace)
    run_names = {value for _, value in runs}
    if selected and selected in run_names:
        session.select_run(selected, workspace=workspace)
        return selected
    current = session.get_selected_run_dir(workspace=workspace)
    current_name = current.name if current is not None else ""
    if current_name and current_name in run_names:
        return current_name
    lane_name = str(lane or "").strip()
    workspace_path = str(workspace or session.current_workspace_path() or "").strip()
    if lane_name and workspace_path:
        active_run = session._resolve_resume_dir(lane_name, workspace=Path(workspace_path))
        active_name = Path(active_run).name if active_run else ""
        if active_name and active_name in run_names:
            session.select_run(active_name, workspace=workspace)
            return active_name
    if runs:
        fallback = runs[0][1]
        session.select_run(fallback, workspace=workspace)
        return fallback
    return ""


def _run_dir_for_name(session, run_name: str, *, workspace: Optional[Path] = None):
    selected = _pick_selected_run(session, run_name, workspace=workspace)
    if not selected:
        return None, ""
    return session.get_selected_run_dir(workspace=workspace), selected


def _build_snapshot(
    *,
    registry: SessionRegistry,
    ctx: str,
    username: str = "admin",
    lane: str = "research",
    run_name: str = "",
    project_space: str = "",
) -> dict[str, Any]:
    session = registry.get_session(ctx, username=username)
    workspace, workspace_name = _workspace_for_request(registry, session, project_space)
    selected_run = _pick_selected_run(session, run_name, lane=lane, workspace=workspace)
    run_dir = session.get_selected_run_dir(workspace=workspace)
    runtime = _runtime_snapshot(session, workspace=workspace)
    active_run = _active_run_name(session, runtime, workspace=workspace)
    runtime_matches_selection = bool(selected_run) and selected_run == str(runtime.get("run_name") or "")

    if runtime_matches_selection:
        live_state = dict(runtime.get("live_state") or {})
        event_page = session.read_events(run_dir, limit=200)
        events = list(event_page.get("events") or []) or list(runtime.get("recent_events") or [])
        usage_summary = _merge_usage_summary(runtime.get("usage_totals"), session.read_usage_summary(run_dir))
        machine_time_summary = session.read_machine_time_summary(run_dir)
        llm = dict(runtime.get("llm") or {})
        graph = dict(runtime.get("graph") or {})
    else:
        live_state = session.snapshot_live_state(run_dir, workspace=workspace)
        event_page = session.read_events(run_dir, limit=200)
        events = list(event_page.get("events") or [])
        usage_summary = session.read_usage_summary(run_dir)
        machine_time_summary = session.read_machine_time_summary(run_dir)
        llm = live_state.get("llm") if isinstance(live_state.get("llm"), dict) else {}
        graph = {"node": str(live_state.get("current_node") or ""), "message_count": 0, "tool_calls": [], "text_preview": ""}

    cards = _serialize_cards(session.list_run_cards(workspace=workspace))
    return {
        "ctx": ctx,
        "workspace_root": str(session.workspace_root or registry.default_project_space_root),
        "workspace_root_locked": True,
        "workspace_path": str(workspace or ""),
        "workspace_name": workspace_name,
        "workspaces": _serialize_choices(session.list_workspaces()),
        "chat_sessions": _serialize_choices(session.list_chat_sessions(workspace=workspace)),
        "current_chat_session": session.current_chat_session_id(workspace=workspace),
        "runs": _serialize_choices(session.list_runs(workspace=workspace)),
        "active_run": active_run,
        "selected_run": selected_run,
        "cards": cards,
        "run_status": _display_run_status(session, run_dir, workspace=workspace),
        "run_status_text": session.run_status_text(workspace=workspace),
        "run_info": dict(session._runtime_for_workspace(workspace).run_info or {}),
        "live_state": live_state,
        "llm": llm,
        "graph": graph,
        "prompt": None,
        "events": events,
        "events_page": event_page if isinstance(event_page, dict) else {},
        "usage_summary": usage_summary,
        "machine_time_summary": machine_time_summary,
        "proposal": session.read_proposal(run_dir, workspace=workspace),
        "todo_items": session.read_todo_items(run_dir),
        "result_text": session.read_result_text(run_dir),
        "chat_messages": session.get_chat_messages(workspace=workspace),
        "entry_context_status": session.entry_context_status_text(lane=lane, workspace=workspace),
        "runtime": runtime,
        "can_submit_prompt": False,
    }


def _apply_chat_session_view(snapshot: dict[str, Any], *, active_run: str = "") -> dict[str, Any]:
    active = str(active_run or snapshot.get("active_run") or "").strip()
    if active:
        snapshot["selected_run"] = active
        return snapshot
    snapshot["selected_run"] = ""
    snapshot["live_state"] = {}
    snapshot["llm"] = {}
    snapshot["graph"] = {"node": "", "message_count": 0, "tool_calls": [], "text_preview": ""}
    snapshot["prompt"] = None
    snapshot["events"] = []
    snapshot["usage_summary"] = {}
    snapshot["machine_time_summary"] = {}
    snapshot["proposal"] = ""
    snapshot["todo_items"] = []
    snapshot["result_text"] = ""
    snapshot["can_submit_prompt"] = False
    return snapshot


def _build_details(
    *,
    registry: SessionRegistry,
    ctx: str,
    username: str = "admin",
    run_name: str,
    project_space: str = "",
    include_legacy_traces: bool = False,
    include_self_evolution: bool = True,
) -> dict[str, Any]:
    session = registry.get_session(ctx, username=username)
    workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
    run_dir, selected_run = _run_dir_for_name(session, run_name, workspace=workspace)
    artifacts = session.read_artifacts(workspace=workspace)
    if hasattr(artifacts, "to_dict"):
        artifact_rows = artifacts.to_dict(orient="records")
    else:
        artifact_rows = list(artifacts or [])
    payload = {
        "selected_run": selected_run,
        "memory": session.read_memory_index(workspace=workspace),
        "artifacts": artifact_rows,
        "proposal": session.read_proposal(run_dir, workspace=workspace),
        "task_state": session.read_task_state(run_dir, workspace=workspace),
        "self_evolution": (
            _self_evolution_for_run(workspace=workspace, workspace_id=_workspace_name, run_id=selected_run)
            if include_self_evolution
            else {"enabled": False, "disabled_reason": "Self-evolution is disabled in no-login mode."}
        ),
    }
    if include_legacy_traces:
        payload.update(
            {
                "trace_event": session.read_trace(run_dir, "event_trace.jsonl", workspace=workspace),
                "trace_tool": session.read_trace(run_dir, "tool_trace.jsonl", workspace=workspace),
                "trace_patch": session.read_trace(run_dir, "patch_trace.jsonl", workspace=workspace),
            }
        )
    return payload


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _candidate_mentions_run(candidate: dict[str, Any], run_id: str) -> bool:
    target = str(run_id or "").strip()
    return not target or str(candidate.get("run_id") or "").strip() == target


def _candidate_change_preview(
    store: SelfEvolutionStore,
    candidate: Any,
    *,
    limit_chars: int | None = SELF_EVOLUTION_DIFF_LIMIT_CHARS,
) -> tuple[str, bool]:
    candidate_dir = store.revision_dir(candidate.candidate_id, candidate.revision)
    if candidate.action == "memory":
        file_pairs = [
            (
                Path("AGENTS.md"),
                candidate_dir / "current" / "AGENTS.md",
                candidate_dir / "memories" / "AGENTS.md",
            )
        ]
        before_label = "current"
        after_label = "proposed"
    else:
        group = str(candidate.group or "").strip()
        name = str(candidate.name or "").strip()
        if group not in SKILL_GROUPS or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            return "", False
        before_root = candidate_dir / "current" / "target"
        after_root = candidate_dir / "proposed" / group / name
        relative_files = {
            path.relative_to(before_root)
            for path in before_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        } | {
            path.relative_to(after_root)
            for path in after_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        file_pairs = [(relative, before_root / relative, after_root / relative) for relative in sorted(relative_files)]
        before_label = f"current/{group}/{name}"
        after_label = f"proposed/{group}/{name}"

    chunks: list[str] = []
    truncated = False
    for relative, before_path, after_path in file_pairs:
        before_bytes = before_path.read_bytes() if before_path.is_file() else b""
        after_bytes = after_path.read_bytes() if after_path.is_file() else b""
        if before_bytes == after_bytes:
            continue
        if b"\0" in before_bytes or b"\0" in after_bytes:
            chunks.append(f"Binary file changed: {relative.as_posix()}\n")
            continue
        before_text = before_bytes.decode("utf-8", errors="replace")
        after_text = after_bytes.decode("utf-8", errors="replace")
        diff = "\n".join(
            difflib.unified_diff(
                before_text.splitlines(),
                after_text.splitlines(),
                fromfile=f"{before_label}/{relative.as_posix()}",
                tofile=f"{after_label}/{relative.as_posix()}",
                lineterm="",
            )
        )
        if diff:
            chunks.append(diff + "\n")
        if limit_chars is not None and sum(len(chunk) for chunk in chunks) > limit_chars:
            truncated = True
            break
    preview = "\n".join(chunks)
    if limit_chars is not None and len(preview) > limit_chars:
        preview = preview[:limit_chars].rstrip()
        truncated = True
    if truncated:
        preview += "\n...[diff truncated by host]"
    return preview, truncated


def _self_evolution_candidate_payload(
    *,
    store: SelfEvolutionStore,
    promotion: PromotionManager,
    candidate: Any,
) -> dict[str, Any]:
    revision_root = store.revision_dir(candidate.candidate_id, candidate.revision)
    proposal = _read_json_file(revision_root / "proposal.json")
    validation = _read_json_file(revision_root / "validation.json")
    proposal_value = proposal if isinstance(proposal, dict) else {}
    evidence = [
        item.to_dict()
        for observation_id in candidate.evidence_ids
        if (item := store.read_observation(observation_id)) is not None
    ]
    return {
        **candidate.to_dict(),
        "proposal": proposal_value,
        "validation": validation if isinstance(validation, dict) else {},
        "review": dict(candidate.review or {}),
        "evidence": evidence,
        "promotion_readiness": promotion.promotion_readiness(candidate),
        "allowed_actions": promotion.allowed_actions(candidate),
    }


def _self_evolution_for_run(
    *,
    workspace: Optional[Path],
    workspace_id: str,
    run_id: str,
    status: str = "",
    cursor: str = "",
    limit: int = 25,
    observation_status: str = "",
    observation_cursor: str = "",
    observation_limit: int = 25,
) -> dict[str, Any]:
    if workspace is None:
        return {
            "enabled": True,
            "candidates": [],
            "observations": [],
            "jobs": [],
        }
    store = SelfEvolutionStore(workspace, project_id=workspace_id)
    promotion = PromotionManager(store)
    capped_limit = max(1, min(100, int(limit or 25)))
    rows = store.list_candidates(
        status=str(status or "").strip(),
        limit=capped_limit + 1,
        before=str(cursor or "").strip(),
    )
    if str(run_id or "").strip():
        rows = [item for item in rows if item.run_id == str(run_id).strip()]
    has_more = len(rows) > capped_limit
    visible_rows = rows[:capped_limit]
    candidates = [
        _self_evolution_candidate_payload(
            store=store,
            promotion=promotion,
            candidate=item,
        )
        for item in visible_rows
    ]

    capped_observation_limit = max(1, min(100, int(observation_limit or 25)))
    observation_rows = store.list_observations(
        status=str(observation_status or "").strip(),
        limit=capped_observation_limit + 1,
        before=str(observation_cursor or "").strip(),
    )
    observation_has_more = len(observation_rows) > capped_observation_limit
    visible_observations = observation_rows[:capped_observation_limit]
    jobs = [
        item.to_dict()
        for item in store.list_jobs(limit=25, project_id=workspace_id)
        if not str(run_id or "").strip() or item.run_id == str(run_id).strip()
    ]
    active = store.read_active_skills().get("skills") or {}
    effective_skill_count = sum(
        bool(isinstance(pointer, dict) and pointer.get("stable"))
        for pointer in active.values()
    )
    status_counts = store.candidate_status_counts()
    observation_status_counts = store.observation_status_counts()
    return {
        "enabled": True,
        "mode": resolve_self_evolution_mode(),
        "workspace_id": workspace_id,
        "scope": "workspace",
        "activation": "next_selected_run",
        "candidates": candidates,
        "candidate_count": (
            len(candidates)
            if str(run_id or "").strip()
            else sum(status_counts.values())
        ),
        "next_cursor": (
            visible_rows[-1].candidate_id
            if has_more and visible_rows
            else ""
        ),
        "status_counts": status_counts,
        "observations": [item.to_dict() for item in visible_observations],
        "observation_count": sum(observation_status_counts.values()),
        "observation_next_cursor": (
            visible_observations[-1].observation_id
            if observation_has_more and visible_observations
            else ""
        ),
        "observation_status_counts": observation_status_counts,
        "effective_skill_count": effective_skill_count,
        "jobs": jobs,
        "job_count": len(jobs),
        "error_count": sum(
            str(item.get("status") or "") in {"error", "recovery_review"}
            for item in jobs
        ),
    }


def _read_self_evolution_candidate(store: SelfEvolutionStore, candidate_id: str):
    try:
        candidate = store.read_candidate(candidate_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid learning candidate id.") from exc
    if candidate is None:
        raise HTTPException(status_code=404, detail="Learning candidate not found.")
    return candidate


def _read_self_evolution_candidate_revision(
    store: SelfEvolutionStore,
    candidate_id: str,
    revision: int,
) -> tuple[LearningCandidate, LearningCandidate]:
    current = _read_self_evolution_candidate(store, candidate_id)
    requested = int(revision)
    if requested < 1 or requested > current.revision:
        raise HTTPException(status_code=404, detail="Learning candidate revision not found.")
    if requested == current.revision:
        return current, current
    revision_root = store.revision_dir(candidate_id, requested)
    descriptor = _read_json_file(revision_root / "candidate.json")
    if (
        not isinstance(descriptor, dict)
        or str(descriptor.get("candidate_id") or "") != candidate_id
    ):
        raise HTTPException(status_code=404, detail="Learning candidate revision not found.")
    validation = _read_json_file(revision_root / "validation.json")
    review = _read_json_file(revision_root / "review.json")
    if isinstance(validation, dict) and validation and validation.get("valid") is False:
        status = "revision"
    elif isinstance(review, dict) and review:
        status = "review"
    else:
        status = "pending"
    historical = LearningCandidate.from_dict(
        {
            **descriptor,
            "status": status,
            "revision": requested,
            "updated_at": "",
        }
    )
    historical.review = review if isinstance(review, dict) else {}
    historical.validation = validation if isinstance(validation, dict) else {}
    return historical, current


def _self_evolution_candidate_detail(*, workspace: Path, workspace_id: str, candidate_id: str) -> dict[str, Any]:
    store = SelfEvolutionStore(workspace, project_id=workspace_id)
    candidate = _read_self_evolution_candidate(store, candidate_id)
    candidate_dir = store.revision_dir(candidate.candidate_id, candidate.revision)
    validation = _read_json_file(candidate_dir / "validation.json")
    patch_text = ""
    patch_path: Path | None = candidate_dir / "memories" / "AGENTS.md" if candidate.action == "memory" else None
    group = str(candidate.group or "").strip()
    name = str(candidate.name or "").strip()
    if candidate.action == "skill" and group in SKILL_GROUPS and re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        patch_path = candidate_dir / "proposed" / group / name / "SKILL.md"
    if patch_path is not None and patch_path.is_file() and patch_path.stat().st_size <= TEXT_PREVIEW_LIMIT_BYTES:
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
    bundle_files = [
        str(path.relative_to(candidate_dir))
        for path in sorted(candidate_dir.rglob("*"))
        if path.is_file()
    ]
    change_preview, change_preview_truncated = _candidate_change_preview(store, candidate)
    return {
        "candidate": _self_evolution_candidate_payload(
            store=store,
            promotion=PromotionManager(store),
            candidate=candidate,
        ),
        "validation_report": validation,
        "patch_text": patch_text,
        "change_preview": change_preview,
        "change_preview_truncated": change_preview_truncated,
        "bundle_files": bundle_files,
    }


def _build_events(
    *,
    registry: SessionRegistry,
    ctx: str,
    username: str = "admin",
    run_name: str,
    project_space: str = "",
    limit: int = 200,
    before_id: int = 0,
    after_id: int = 0,
    before_seq: int = 0,
    after_seq: int = 0,
    channel: str = "",
    category: str = "",
    names: Optional[list[str]] = None,
    run_id: str = "",
    thread_id: str = "",
    agent_name: str = "",
    tool: str = "",
    include_legacy_trace_records: bool = False,
) -> dict[str, Any]:
    session = registry.get_session(ctx, username=username)
    workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
    run_dir, selected_run = _run_dir_for_name(session, run_name, workspace=workspace)
    if before_seq or after_seq:
        page = session.read_ui_events(run_dir, limit=limit, before_seq=before_seq, after_seq=after_seq)
    else:
        page = session.read_events(
            run_dir,
            limit=limit,
            before_id=before_id,
            after_id=after_id,
            channel=channel,
            category=category,
            names=names,
            run_id=run_id,
            thread_id=thread_id,
            agent_name=agent_name,
            tool=tool,
            include_legacy_trace_records=include_legacy_trace_records,
        )
    page["selected_run"] = selected_run
    return page


def _build_observability(
    *,
    registry: SessionRegistry,
    ctx: str,
    username: str = "admin",
    lane: str = "research",
    run_name: str = "",
    project_space: str = "",
    limit: int = 400,
) -> dict[str, Any]:
    session = registry.get_session(ctx, username=username)
    workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
    selected_run = _pick_selected_run(session, run_name, lane=lane, workspace=workspace)
    run_dir = session.get_selected_run_dir(workspace=workspace) if selected_run else None
    runtime = _runtime_snapshot(session, workspace=workspace)
    active_run = _active_run_name(session, runtime, workspace=workspace)
    runtime_matches_selection = bool(selected_run) and selected_run == str(runtime.get("run_name") or "")
    payload = session.read_observability(run_dir, workspace=workspace, limit=limit)
    if not isinstance(payload, dict):
        payload = {}
    if runtime_matches_selection:
        live_state = dict(runtime.get("live_state") or {})
        graph = dict(runtime.get("graph") or {})
    else:
        event_page = session.read_events(run_dir, limit=200)
        live_state = session.update_live_state(
            run_dir,
            list(event_page.get("events") or []),
            live_llm_enabled=False,
            workspace=workspace,
        )
        graph = {"node": str(live_state.get("current_node") or ""), "message_count": 0, "tool_calls": [], "text_preview": ""}
    payload["selected_run"] = selected_run
    payload["active_run"] = active_run
    payload["run_status"] = _display_run_status(session, run_dir, workspace=workspace)
    payload["run_status_text"] = session.run_status_text(workspace=workspace)
    payload["live_state"] = live_state
    payload["graph"] = graph
    payload["todo_items"] = session.read_todo_items(run_dir)
    payload["usage_summary"] = session.read_usage_summary(run_dir)
    payload["machine_time_summary"] = session.read_machine_time_summary(run_dir)
    payload["chat_messages"] = session.get_chat_messages(limit=120, workspace=workspace)
    artifacts = session.read_artifacts(workspace=workspace)
    if hasattr(artifacts, "to_dict"):
        payload["_artifacts"] = artifacts.to_dict(orient="records")
    else:
        payload["_artifacts"] = list(artifacts or [])
    return payload


def _build_memory(*, registry: SessionRegistry, ctx: str, username: str = "admin", run_name: str = "", source: str = "all", project_space: str = "") -> dict[str, Any]:
    session = registry.get_session(ctx, username=username)
    workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
    _run_dir, selected_run = _run_dir_for_name(session, run_name, workspace=workspace)
    return {
        "selected_run": selected_run,
        "source": str(source or "all").strip().lower() or "all",
        "memory": session.read_memory_index(source=source, workspace=workspace),
    }


def _nav_html(view: str) -> str:
    home_class = "nav-link active" if view == "home" else "nav-link"
    monitor_class = "nav-link active" if view == "monitor" else "nav-link"
    files_class = "nav-link active" if view == "files" else "nav-link"
    return (
        '<nav class="nav-bar">'
        f'<a class="{home_class}" href="/">Home</a>'
        f'<a class="{monitor_class}" href="/monitor/">Monitor</a>'
        f'<a class="{files_class}" href="/files/">Files</a>'
        "</nav>"
    )


def _page_html(*, view: str) -> str:
    title_map = {
        "home": "CatMaster",
        "monitor": "CatMaster Monitor",
        "files": "CatMaster Files",
    }
    title = title_map.get(view, "CatMaster")
    favicon_svg = (
        "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E"
        "%3Crect width='64' height='64' rx='14' fill='%23111827'/%3E"
        "%3Ctext x='32' y='41' text-anchor='middle' font-family='Arial,sans-serif' "
        "font-size='30' font-weight='700' fill='%23f8fafc'%3EC%3C/text%3E%3C/svg%3E"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <link rel="icon" href="{favicon_svg}" />
  <link rel="stylesheet" href="/static/app.css" />
</head>
<body data-catmaster-view="{view}">
  <div id="app"></div>
  <script type="module" src="/static/app.js"></script>
</body>
</html>"""


def _legacy_page_routes_enabled() -> bool:
    return str(os.environ.get("CATMASTER_WEBUI_LEGACY_ROUTES") or "").strip().lower() in {"1", "true", "yes", "on"}


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _static_file_response(*, static_dir: Path, file_name: str) -> FileResponse:
    candidate = (static_dir / file_name).resolve()
    static_root = static_dir.resolve()
    if candidate.parent != static_root or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Static asset not found")
    return FileResponse(candidate)


def _discover_project_spaces(project_space_root: Path | str) -> list[Path]:
    root = Path(project_space_root).expanduser().resolve()
    candidates = [root]
    for pattern in ("*", "users/*", "users/*/*"):
        candidates.extend(root.glob(pattern))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.is_dir() or not (candidate / "files").is_dir() or not (candidate / "metadata").is_dir():
            continue
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def create_app(
    *,
    project_space_root: str,
    no_login: bool = False,
    disable_registration: bool = False,
) -> FastAPI:
    default_project_space_root = str(Path(project_space_root).expanduser().resolve())
    registry = SessionRegistry(default_project_space_root=default_project_space_root)
    auth = AuthManager(
        auth_root=Path(default_project_space_root) / ".webui_auth",
        enabled=not no_login,
        registration_enabled=not disable_registration,
    )
    developer_diagnostics_enabled = str(
        os.getenv("CATMASTER_WEBUI_DEVELOPER_DIAGNOSTICS", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    legacy_routes_enabled = _legacy_page_routes_enabled()
    self_evolution_wakeup: asyncio.Event | None = None
    self_evolution_worker_task: asyncio.Task[Any] | None = None
    research_graph_wakeup: asyncio.Event | None = None
    research_graph_worker_task: asyncio.Task[Any] | None = None

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        nonlocal self_evolution_wakeup, self_evolution_worker_task
        nonlocal research_graph_wakeup, research_graph_worker_task
        research_graph_wakeup = asyncio.Event()
        research_graph_worker_task = asyncio.create_task(_research_graph_worker_loop())
        if auth.enabled:
            self_evolution_wakeup = asyncio.Event()
            self_evolution_worker_task = asyncio.create_task(_self_evolution_worker_loop())
        try:
            yield
        finally:
            if self_evolution_worker_task is not None:
                self_evolution_worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self_evolution_worker_task
            self_evolution_worker_task = None
            self_evolution_wakeup = None
            if research_graph_worker_task is not None:
                research_graph_worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await research_graph_worker_task
            research_graph_worker_task = None
            research_graph_wakeup = None

    app = FastAPI(title="CatMaster WebUI", lifespan=_lifespan)
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    thread_brokers: dict[str, ThreadEventBroker] = {}
    thread_tasks: dict[str, asyncio.Task[Any]] = {}
    thread_stop_flags: set[str] = set()

    def _identity_or_401() -> AuthIdentity:
        identity = _AUTH_IDENTITY.get()
        if identity is None or not identity.authenticated:
            raise HTTPException(status_code=401, detail="Authentication required.")
        return identity

    def _locked_user_root(identity: AuthIdentity) -> Path:
        if not auth.enabled:
            return registry.default_project_space_root
        return auth.user_root(identity.username, base_project_space_root=registry.default_project_space_root)

    def _bound_session(ctx: str):
        identity = _identity_or_401()
        session = registry.get_session(ctx, username=identity.username)
        ok, message, _choices = session.set_workspace_root(str(_locked_user_root(identity)))
        if not ok:
            raise HTTPException(status_code=500, detail=message)
        return identity, session

    def _with_auth(snapshot: dict[str, Any], identity: AuthIdentity) -> dict[str, Any]:
        snapshot["auth"] = auth.public_status(identity)
        snapshot["workspace_root_locked"] = True
        return snapshot

    def _require_self_evolution_enabled() -> None:
        if not auth.enabled:
            raise HTTPException(status_code=403, detail="Self-evolution is disabled in no-login mode.")

    def _require_developer_diagnostics() -> AuthIdentity:
        identity = _identity_or_401()
        if not developer_diagnostics_enabled:
            raise HTTPException(status_code=404, detail="Developer diagnostics are not enabled.")
        return identity

    def _require_legacy_routes() -> None:
        if not legacy_routes_enabled:
            raise HTTPException(status_code=404, detail="Legacy WebUI routes are not enabled.")

    def _public_workspace_result(
        *,
        session,
        identity: AuthIdentity,
        ok: bool,
        status_message: str,
        workspace_name: str = "",
    ) -> dict[str, Any]:
        return _with_auth(
            {
                "ok": bool(ok),
                "status_message": str(status_message or ""),
                "workspace_name": str(workspace_name or ""),
                "workspaces": _serialize_choices(session.list_workspaces()),
            },
            identity,
        )

    async def _validated_body(model_cls, request: Request):
        try:
            return model_cls.model_validate(await _json_body(request))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _workspace_from_id(workspace_id: str, identity: AuthIdentity) -> tuple[Path, str]:
        raw_workspace_id = str(workspace_id or "").strip().strip("/")
        if not raw_workspace_id:
            raise HTTPException(status_code=400, detail="Workspace id is required.")
        root = _locked_user_root(identity)
        target, resolved_name = registry._resolve_project_space_target(raw_workspace_id, root=root)
        if target is None:
            raise HTTPException(status_code=404, detail=f"Project space not found: {raw_workspace_id}")
        try:
            ensure_project_space_layout(target, create=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid project space layout: {exc}") from exc
        workspace_name = registry._project_space_name_from_path(str(target), root=root) or resolved_name or target.name
        return target.expanduser().resolve(), workspace_name

    def _workspace_for_thread(thread_id: str, identity: AuthIdentity) -> tuple[Path, str]:
        tid = str(thread_id or "").strip()
        if not re.match(r"^[A-Za-z0-9_.:-]{3,120}$", tid):
            raise HTTPException(status_code=400, detail="Invalid thread id.")
        root = _locked_user_root(identity).expanduser().resolve()
        workspace_candidates: list[Path] = []
        if (root / "files").is_dir() and (root / "metadata").is_dir():
            workspace_candidates.append(root)
        try:
            for thread_file in root.rglob(f"metadata/threads/{tid}/thread.json"):
                if thread_file.is_file():
                    try:
                        workspace_candidates.append(thread_file.parents[3])
                    except Exception:
                        continue
        except Exception:
            pass
        seen: set[str] = set()
        for workspace in workspace_candidates:
            key = str(workspace)
            if key in seen:
                continue
            seen.add(key)
            try:
                store = ThreadStore(workspace=workspace, workspace_id=registry._project_space_name_from_path(str(workspace), root=root) or workspace.name)
                store.get_thread(tid)
            except Exception:
                continue
            return workspace.resolve(), registry._project_space_name_from_path(str(workspace), root=root) or workspace.name
        raise HTTPException(status_code=404, detail=f"Thread not found: {tid}")

    def _broker_for_workspace(workspace: Path) -> ThreadEventBroker:
        key = str(workspace.expanduser().resolve())
        broker = thread_brokers.get(key)
        if broker is None:
            def _run_dir_for_thread_event(thread_id: str, data: Dict[str, Any]) -> Optional[Path]:
                run_id = _thread_event_run_id(data)
                if not run_id:
                    try:
                        ws_id = registry._project_space_name_from_path(str(workspace), root=registry.root) or workspace.name
                        run_id = ThreadStore(workspace=workspace, workspace_id=ws_id).get_thread(thread_id).active_run_id
                    except Exception:
                        run_id = ""
                run_id = str(run_id or "").strip()
                if not run_id or not re.match(r"^[A-Za-z0-9_.:-]{3,160}$", run_id):
                    return None
                return system_root(workspace) / "runs" / run_id

            broker = ThreadEventBroker(workspace=workspace, run_dir_resolver=_run_dir_for_thread_event)
            thread_brokers[key] = broker
        return broker

    def _thread_store(workspace: Path, workspace_id: str) -> ThreadStore:
        return ThreadStore(workspace=workspace, workspace_id=workspace_id)

    def _artifact_registry(workspace: Path, workspace_id: str) -> ArtifactRegistry:
        return ArtifactRegistry(workspace=workspace, workspace_id=workspace_id)

    def _thread_should_stop(thread_id: str) -> bool:
        return str(thread_id or "") in thread_stop_flags

    def _enqueue_self_evolution_post_run(**kwargs: Any) -> None:
        if not auth.enabled:
            return
        try:
            workspace = Path(kwargs.get("workspace") or "").expanduser().resolve()
            workspace_id = str(kwargs.get("workspace_id") or workspace.name)
            run_id = str(kwargs.get("run_id") or "")
            thread_id = str(kwargs.get("thread_id") or "")
            terminal_status = str(kwargs.get("terminal_status") or "")
            run_dir = kwargs.get("run_dir") or ""
            coordinator = SelfEvolutionCoordinator(workspace=workspace, project_id=workspace_id)
            job = coordinator.enqueue_post_run(
                run_id=run_id,
                thread_id=thread_id,
                message_id=str(kwargs.get("message_id") or ""),
                entrypoint=str(kwargs.get("entrypoint") or ""),
                terminal_status=terminal_status,
                run_dir=run_dir,
                payload={
                    "prior_assistant_message_id": str(
                        kwargs.get("prior_assistant_message_id") or ""
                    ),
                    "assistant_message_id": str(
                        kwargs.get("assistant_message_id") or ""
                    ),
                },
                model_config=str(kwargs.get("model_config") or ""),
            )
            if job is not None and self_evolution_wakeup is not None:
                self_evolution_wakeup.set()
        except Exception:
            logger.exception("Failed to enqueue self-evolution post-run job")

    def _known_project_spaces() -> list[Path]:
        return _discover_project_spaces(registry.default_project_space_root)

    async def _self_evolution_worker_loop() -> None:
        configured_recovery = (
            os.getenv("CATMASTER_SELF_EVOLUTION_RECOVERY_SEC") or "300"
        )
        try:
            recovery_interval = max(60, int(configured_recovery))
        except ValueError:
            recovery_interval = 300
        recovered_workspaces: set[str] = set()
        while True:
            if self_evolution_wakeup is not None:
                self_evolution_wakeup.clear()
            try:
                for workspace in _known_project_spaces():
                    bootstrap_store = SelfEvolutionStore(workspace, project_id=workspace.name)
                    workspace_key = str(workspace)
                    if workspace_key not in recovered_workspaces:
                        await asyncio.to_thread(bootstrap_store.requeue_expired_jobs)
                        recovered_workspaces.add(workspace_key)
                    for workspace_id in bootstrap_store.queued_project_ids():
                        coordinator = SelfEvolutionCoordinator(workspace=workspace, project_id=workspace_id)
                        await asyncio.to_thread(coordinator.process_pending_jobs, limit=4)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Self-evolution durable worker pass failed")
            if self_evolution_wakeup is None:
                await asyncio.sleep(recovery_interval)
                continue
            try:
                await asyncio.wait_for(
                    self_evolution_wakeup.wait(),
                    timeout=recovery_interval,
                )
            except TimeoutError:
                pass

    async def _research_graph_worker_loop() -> None:
        configured_recovery = (
            os.getenv("CATMASTER_RESEARCH_GRAPH_RECOVERY_SEC") or "300"
        )
        try:
            recovery_interval = max(60, int(configured_recovery))
        except ValueError:
            recovery_interval = 300
            logger.warning(
                "Invalid CATMASTER_RESEARCH_GRAPH_RECOVERY_SEC=%r; using 300",
                configured_recovery,
            )
        while True:
            # Graph mutations and child completion wake this worker immediately.
            # The timeout is only a low-frequency crash/lease recovery sweep.
            if research_graph_wakeup is not None:
                research_graph_wakeup.clear()
            try:
                for workspace in _known_project_spaces():
                    workspace_id = (
                        registry._project_space_name_from_path(
                            str(workspace),
                            root=registry.default_project_space_root,
                        )
                        or workspace.name
                    )
                    await ResearchGraphService(
                        workspace=workspace,
                        workspace_id=workspace_id,
                        agent_loop_factory=_agent_loop,
                    ).tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Research Graph worker pass failed")
            if research_graph_wakeup is None:
                await asyncio.sleep(recovery_interval)
                continue
            try:
                await asyncio.wait_for(
                    research_graph_wakeup.wait(),
                    timeout=recovery_interval,
                )
            except TimeoutError:
                pass

    def _on_thread_turn_finished(**kwargs: Any) -> None:
        try:
            ResearchGraphService(
                workspace=Path(kwargs.get("workspace") or ""),
                workspace_id=str(kwargs.get("workspace_id") or ""),
                agent_loop_factory=_agent_loop,
            ).reconcile_finished_child(
                child_thread_id=str(kwargs.get("thread_id") or ""),
                terminal_status=str(kwargs.get("terminal_status") or ""),
                run_id=str(kwargs.get("run_id") or ""),
            )
        except Exception:
            logger.exception(
                "Failed to reconcile Research Graph child thread %s",
                kwargs.get("thread_id"),
            )
        if research_graph_wakeup is not None:
            research_graph_wakeup.set()
        if str(kwargs.get("run_id") or "").strip():
            _enqueue_self_evolution_post_run(**kwargs)

    def _agent_loop(workspace: Path, workspace_id: str) -> ThreadAgentLoopService:
        return ThreadAgentLoopService(
            workspace=workspace,
            workspace_id=workspace_id,
            store=_thread_store(workspace, workspace_id),
            broker=_broker_for_workspace(workspace),
            artifact_registry=_artifact_registry(workspace, workspace_id),
            thread_tasks=thread_tasks,
            thread_stop_flags=thread_stop_flags,
            build_runner=build_specialist_runner,
            streaming_runner_cls=StreamingSpecialistRunner,
            permission_mode_for_thread=_thread_permission_mode,
            interrupt_on_for_permission_mode=_interrupt_on_for_permission_mode,
            normalize_entrypoint=_entrypoint,
            should_stop=_thread_should_stop,
            on_turn_finished=_on_thread_turn_finished,
        )

    def _entrypoint(value: str) -> str:
        raw = str(value or "research").strip().lower().replace("-", "_").replace(" ", "_") or "research"
        candidate = _THREAD_LANE_ALIASES.get(raw, raw)
        if candidate in _SUPPORTED_THREAD_ENTRYPOINTS:
            return candidate
        return "research"

    def _request_entrypoint(value: Any, *, default: str = "research") -> str:
        raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not raw:
            return default
        candidate = _THREAD_LANE_ALIASES.get(raw, raw)
        if candidate not in _SUPPORTED_THREAD_ENTRYPOINTS:
            raise HTTPException(status_code=400, detail=f"Invalid entrypoint: {value}")
        return candidate

    def _request_permission_mode(value: Any, *, default: str = "auto") -> str:
        try:
            return _normalize_thread_permission_mode(value, default=default)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _session_cookie_response(payload: dict[str, Any], token: str) -> JSONResponse:
        response = JSONResponse(payload)
        response.set_cookie(
            SESSION_COOKIE_NAME,
            token,
            httponly=True,
            samesite="lax",
            secure=False,
            max_age=SESSION_TTL_SECONDS,
        )
        return response

    @app.middleware("http")
    async def _browser_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-eval' blob:; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "font-src 'self' data:; "
            "connect-src 'self' ws: wss:; "
            "worker-src 'self' blob:; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "frame-ancestors 'self'; "
            "form-action 'self'",
        )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        return response

    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next):
        path = str(request.url.path or "")
        token = None
        identity: AuthIdentity | None = None
        if path.startswith("/api/") and not path.startswith("/api/auth/"):
            if auth.enabled:
                identity = auth.identity_for_token(str(request.cookies.get(SESSION_COOKIE_NAME) or ""))
                if identity is None:
                    return JSONResponse({"detail": "Authentication required."}, status_code=401)
            else:
                identity = auth.default_identity()
            token = _AUTH_IDENTITY.set(identity)
            request.state.auth_identity = identity
        try:
            return await call_next(request)
        finally:
            if token is not None:
                _AUTH_IDENTITY.reset(token)

    @app.get("/api/auth/status")
    def _auth_status(request: Request):
        if not auth.enabled:
            return JSONResponse(auth.public_status(auth.default_identity()))
        identity = auth.identity_for_token(str(request.cookies.get(SESSION_COOKIE_NAME) or ""))
        return JSONResponse(auth.public_status(identity))

    @app.get("/api/auth/captcha")
    def _auth_captcha():
        if not auth.enabled:
            return JSONResponse({"captcha_id": "", "question": ""})
        if not auth.registration_enabled:
            raise HTTPException(status_code=403, detail="Registration is disabled.")
        return JSONResponse(auth.create_captcha())

    @app.post("/api/auth/register")
    async def _auth_register(request: Request):
        if not auth.enabled:
            return JSONResponse(auth.public_status(auth.default_identity()))
        if not auth.registration_enabled:
            raise HTTPException(status_code=403, detail="Registration is disabled.")
        payload = await _json_body(request)
        try:
            username = auth.register_user(
                username=str(payload.get("username") or ""),
                password=str(payload.get("password") or ""),
                captcha_id=str(payload.get("captcha_id") or ""),
                captcha_answer=str(payload.get("captcha_answer") or ""),
            )
            token_value = auth.create_session(username)
            identity = AuthIdentity(username=username, authenticated=True, auth_enabled=True)
            _locked_user_root(identity).mkdir(parents=True, exist_ok=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _session_cookie_response(auth.public_status(identity), token_value)

    @app.post("/api/auth/login")
    async def _auth_login(request: Request):
        if not auth.enabled:
            return JSONResponse(auth.public_status(auth.default_identity()))
        payload = await _json_body(request)
        try:
            username = auth.authenticate_user(
                username=str(payload.get("username") or ""),
                password=str(payload.get("password") or ""),
            )
            token_value = auth.create_session(username)
            identity = AuthIdentity(username=username, authenticated=True, auth_enabled=True)
            _locked_user_root(identity).mkdir(parents=True, exist_ok=True)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return _session_cookie_response(auth.public_status(identity), token_value)

    @app.post("/api/auth/logout")
    async def _auth_logout(request: Request):
        if auth.enabled:
            auth.revoke_session(str(request.cookies.get(SESSION_COOKIE_NAME) or ""))
        response = JSONResponse(auth.public_status(None))
        response.delete_cookie(SESSION_COOKIE_NAME)
        return response

    @app.get("/asset-{asset_path:path}", include_in_schema=False)
    def _root_asset(asset_path: str):
        return _static_file_response(static_dir=static_dir, file_name=f"asset-{asset_path}")

    @app.get("/favicon.ico", include_in_schema=False)
    def _favicon():
        favicon_path = static_dir / "favicon.ico"
        if favicon_path.is_file():
            return FileResponse(favicon_path)
        return Response(status_code=204)

    @app.get("/monitor", include_in_schema=False)
    def _monitor_redirect(request: Request):
        if not _legacy_page_routes_enabled():
            return RedirectResponse(url="/#tab=monitor", status_code=307)
        query = request.url.query
        target = "/monitor/"
        if query:
            target = f"{target}?{query}"
        return RedirectResponse(url=target, status_code=307)

    @app.get("/files", include_in_schema=False)
    def _files_redirect(request: Request):
        if not _legacy_page_routes_enabled():
            return RedirectResponse(url="/#tab=files", status_code=307)
        query = request.url.query
        target = "/files/"
        if query:
            target = f"{target}?{query}"
        return RedirectResponse(url=target, status_code=307)

    @app.get("/", response_class=HTMLResponse)
    def _home_page() -> str:
        return _page_html(view="home")

    @app.get("/monitor/", response_class=HTMLResponse)
    def _monitor_page():
        if not _legacy_page_routes_enabled():
            return RedirectResponse(url="/#tab=monitor", status_code=307)
        return _page_html(view="monitor")

    @app.get("/files/", response_class=HTMLResponse)
    def _files_page():
        if not _legacy_page_routes_enabled():
            return RedirectResponse(url="/#tab=files", status_code=307)
        return _page_html(view="files")

    @app.get("/api/bootstrap")
    def _bootstrap(
        ctx: Optional[str] = None,
        project_space: Optional[str] = None,
        run: Optional[str] = None,
        lane: str = "research",
    ):
        identity = _identity_or_401()
        state = registry.bootstrap(
            ctx=ctx,
            project_space=project_space,
            run=run,
            username=identity.username,
            project_space_root=_locked_user_root(identity),
            default_project_space="admin" if not auth.enabled else "default",
            auto_open_default=True,
        )
        snapshot = _build_snapshot(
            registry=registry,
            ctx=state.ctx,
            username=identity.username,
            lane=lane,
            run_name=state.run_name,
            project_space=state.project_space_name,
        )
        snapshot["status_message"] = state.status
        public_snapshot = {
            "ctx": snapshot.get("ctx") or state.ctx,
            "workspace_name": snapshot.get("workspace_name") or state.project_space_name,
            "workspaces": snapshot.get("workspaces") or [],
            "status_message": state.status,
            "entrypoints": _THREAD_ENTRYPOINTS,
            "default_entrypoint": "research",
            "developer_diagnostics_enabled": developer_diagnostics_enabled,
        }
        return JSONResponse(_with_auth(public_snapshot, identity))

    @app.get("/api/entrypoints")
    def _entrypoints():
        _identity_or_401()
        return JSONResponse({"entrypoints": _THREAD_ENTRYPOINTS, "default_entrypoint": "research"})

    @app.post("/api/workspaces/{workspace_id}/threads", response_model=PublicThreadEnvelope)
    def _threads_create(workspace_id: str, payload: ThreadCreateRequest):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_from_id(workspace_id, identity)
        store = _thread_store(workspace, workspace_name)
        metadata = dict(payload.metadata or {})
        metadata["permission_mode"] = _request_permission_mode(payload.permission_mode or metadata.get("permission_mode"))
        thread = store.create_thread(
            title=payload.title,
            entrypoint=_request_entrypoint(payload.entrypoint),
            meta=metadata,
        )
        broker = _broker_for_workspace(workspace)
        broker.emit(thread.thread_id, "thread.created", status=str(thread.status.value), data={"thread": thread.model_dump(mode="json")})
        return {"thread": project_thread(thread)}

    @app.get("/api/workspaces/{workspace_id}/threads", response_model=PublicThreadListEnvelope)
    def _threads_list(workspace_id: str):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_from_id(workspace_id, identity)
        threads = _thread_store(workspace, workspace_name).list_threads()
        return {"threads": [project_thread(thread) for thread in threads]}

    @app.get("/api/threads/{thread_id}", response_model=PublicThreadEnvelope)
    def _thread_get(thread_id: str):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_for_thread(thread_id, identity)
        store = _thread_store(workspace, workspace_name)
        thread = store.get_thread(thread_id)
        return {"thread": project_thread(thread)}

    @app.patch("/api/threads/{thread_id}", response_model=PublicThreadEnvelope)
    def _thread_patch(thread_id: str, payload: ThreadPatchRequest):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_for_thread(thread_id, identity)
        updates: dict[str, Any] = {}
        supplied = payload.model_fields_set
        if "title" in supplied:
            updates["title"] = payload.title
        if "entrypoint" in supplied:
            updates["entrypoint"] = _request_entrypoint(payload.entrypoint)
        if "status" in supplied:
            updates["status"] = payload.status
        store = _thread_store(workspace, workspace_name)
        thread = store.get_thread(thread_id)
        if "metadata" in supplied or "permission_mode" in supplied:
            meta = {**dict(thread.meta or {})}
            if "metadata" in supplied:
                meta.update(dict(payload.metadata or {}))
            if "permission_mode" in supplied:
                meta["permission_mode"] = _request_permission_mode(payload.permission_mode)
            elif "permission_mode" in meta:
                meta["permission_mode"] = _request_permission_mode(meta.get("permission_mode"))
            updates["meta"] = meta
        thread = store.update_thread(thread_id, **updates)
        _broker_for_workspace(workspace).emit(thread_id, "thread.updated", status=str(thread.status.value), data={"thread": thread.model_dump(mode="json")})
        return {"thread": project_thread(thread)}

    @app.get("/api/threads/{thread_id}/messages", response_model=PublicMessagePageEnvelope)
    def _thread_messages(thread_id: str, before: str = "", limit: int = 50):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_for_thread(thread_id, identity)
        store = _thread_store(workspace, workspace_name)
        try:
            page = store.list_messages_page(thread_id, before=before, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        rows = [message.model_dump(mode="json") for message in page.messages]
        available_artifact_ids = {
            record.artifact_id
            for record in _artifact_registry(workspace, workspace_name).list_artifacts(thread_id=thread_id)
        }
        rows = _filter_unavailable_artifact_parts(
            rows,
            available_artifact_ids=available_artifact_ids,
        )
        rows = _enrich_thread_message_tool_sources(rows, workspace=workspace)
        projected = project_messages(rows, workspace=workspace)
        return {
            "messages": projected,
            "page": {
                    "shown_count": len(projected),
                    "total_count": page.total_count,
                    "total_unknown": False,
                    "truncated": page.has_more,
                    "next_cursor": page.next_cursor,
                    "full_content_ref": "",
                    "unit": "items",
                },
        }

    @app.get(
        "/api/threads/{thread_id}/messages/{message_id}/parts",
        response_model=PublicPartPageEnvelope,
    )
    def _thread_message_parts(
        thread_id: str,
        message_id: str,
        cursor: str = "",
        limit: int = 20,
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_for_thread(thread_id, identity)
        message = _thread_store(workspace, workspace_name).get_message(thread_id, message_id)
        if message is None:
            raise HTTPException(status_code=404, detail="Message not found.")
        start = 0
        capped_limit = min(100, max(1, int(limit or 20)))
        raw_parts = list(message.parts)
        if cursor:
            try:
                after_part_id = decode_public_cursor(
                    cursor,
                    kind="message_parts",
                    identity=message_id,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if not isinstance(after_part_id, str):
                raise HTTPException(status_code=400, detail="Message-parts cursor is invalid.")
            after_index = next(
                (index for index, item in enumerate(raw_parts) if item.id == after_part_id),
                None,
            )
            if after_index is None:
                raise HTTPException(status_code=400, detail="Message-parts cursor is stale.")
            start = after_index + 1
        visible = raw_parts[start:start + capped_limit]
        projected = [
            project_part(
                part,
                workspace=workspace,
                thread_id=thread_id,
                message_id=message_id,
            )
            for part in visible
        ]
        next_index = start + len(visible)
        has_more = next_index < len(raw_parts)
        return {
                "parts": projected,
                "page": {
                    "shown_count": next_index,
                    "total_count": len(raw_parts),
                    "total_unknown": False,
                    "truncated": has_more,
                    "next_cursor": (
                        encode_public_cursor("message_parts", message_id, visible[-1].id)
                        if has_more and visible
                        else ""
                    ),
                    "full_content_ref": "",
                    "unit": "items",
                },
        }

    @app.get(
        "/api/threads/{thread_id}/messages/{message_id}/parts/{part_id}",
        response_model=PublicPartEnvelope,
    )
    def _thread_message_part(thread_id: str, message_id: str, part_id: str):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_for_thread(thread_id, identity)
        message = _thread_store(workspace, workspace_name).get_message(thread_id, message_id)
        if message is None:
            raise HTTPException(status_code=404, detail="Message not found.")
        part = next((item for item in message.parts if item.id == part_id), None)
        if part is None:
            raise HTTPException(status_code=404, detail="Message part not found.")
        return {
                "part": project_part(
                        part,
                        workspace=workspace,
                        thread_id=thread_id,
                        message_id=message_id,
                    )
        }

    @app.get(
        "/api/threads/{thread_id}/messages/{message_id}/parts/{part_id}/content",
        response_model=PublicTextPageEnvelope,
    )
    def _thread_message_part_content(
        thread_id: str,
        message_id: str,
        part_id: str,
        cursor: str = "",
        limit: int = 64_000,
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_for_thread(thread_id, identity)
        message = _thread_store(workspace, workspace_name).get_message(thread_id, message_id)
        if message is None:
            raise HTTPException(status_code=404, detail="Message not found.")
        part = next((item for item in message.parts if item.id == part_id), None)
        if part is None:
            raise HTTPException(status_code=404, detail="Message part not found.")
        if part.type not in {"text", "reasoning", "subagent"}:
            raise HTTPException(
                status_code=404,
                detail="This part has no ordinary full-text representation.",
            )
        text = str(part.text or "")
        start = 0
        if cursor:
            try:
                position = decode_public_cursor(
                    cursor,
                    kind="part_content",
                    identity=part_id,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if not isinstance(position, int) or position < 0:
                raise HTTPException(status_code=400, detail="Part-content cursor is invalid.")
            start = min(len(text), position)
        capped_limit = min(256_000, max(1_000, int(limit or 64_000)))
        visible = text[start:start + capped_limit]
        next_cursor = start + len(visible)
        has_more = next_cursor < len(text)
        return {
                "text": visible,
                "page": {
                    "shown_count": next_cursor,
                    "total_count": len(text),
                    "total_unknown": False,
                    "truncated": has_more,
                    "next_cursor": (
                        encode_public_cursor("part_content", part_id, next_cursor)
                        if has_more
                        else ""
                    ),
                    "full_content_ref": (
                        f"/api/threads/{thread_id}/messages/{message_id}/parts/{part_id}/content"
                    ),
                    "unit": "characters",
                },
        }

    @app.get(
        "/api/threads/{thread_id}/messages/{message_id}/parts/{part_id}/items",
        response_model=PublicItemPageEnvelope,
    )
    def _thread_message_part_items(
        thread_id: str,
        message_id: str,
        part_id: str,
        cursor: str = "",
        limit: int = 100,
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_for_thread(thread_id, identity)
        message = _thread_store(workspace, workspace_name).get_message(thread_id, message_id)
        if message is None:
            raise HTTPException(status_code=404, detail="Message not found.")
        citation_part_id = f"part_citations_{message_id}"
        if part_id == citation_part_id:
            all_items = project_citation_items(message, workspace=workspace)
        else:
            part = next((item for item in message.parts if item.id == part_id), None)
            if part is None:
                raise HTTPException(status_code=404, detail="Message part not found.")
            all_items = project_todo_items(part, workspace=workspace)
        if not all_items:
            raise HTTPException(
                status_code=404,
                detail="This part has no pageable item representation.",
            )
        start = 0
        if cursor:
            try:
                position = decode_public_cursor(
                    cursor,
                    kind="part_items",
                    identity=part_id,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if not isinstance(position, int) or position < 0 or position > len(all_items):
                raise HTTPException(status_code=400, detail="Part-items cursor is invalid or stale.")
            start = position
        capped_limit = min(200, max(1, int(limit or 100)))
        visible = all_items[start:start + capped_limit]
        end = start + len(visible)
        has_more = end < len(all_items)
        full_ref = f"/api/threads/{thread_id}/messages/{message_id}/parts/{part_id}/items"
        return {
            "items": visible,
            "page": {
                "shown_count": len(visible),
                "total_count": len(all_items),
                "total_unknown": False,
                "truncated": has_more,
                "next_cursor": (
                    encode_public_cursor("part_items", part_id, end)
                    if has_more
                    else ""
                ),
                "full_content_ref": full_ref,
                "unit": "items",
                "range_start": start,
                "range_end": end,
            },
        }

    @app.get("/api/threads/{thread_id}/artifacts")
    def _thread_artifacts(thread_id: str):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_for_thread(thread_id, identity)
        records = _artifact_registry(workspace, workspace_name).list_artifacts(thread_id=thread_id)
        return JSONResponse(
            {"artifacts": [project_artifact(record, workspace=workspace) for record in records]}
        )

    def _research_graph_service(
        workspace: Path,
        workspace_name: str,
    ) -> ResearchGraphService:
        return ResearchGraphService(
            workspace=workspace,
            workspace_id=workspace_name,
            agent_loop_factory=_agent_loop,
        )

    def _raise_research_graph_http(exc: Exception) -> None:
        if isinstance(exc, KeyError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if isinstance(exc, ResearchGraphConflict):
            raise HTTPException(
                status_code=409,
                detail={
                    "message": str(exc),
                    "expected_revision": exc.expected_revision,
                    "current_revision": exc.current_revision,
                },
            ) from exc
        if isinstance(exc, (ValueError, RuntimeError)):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if isinstance(exc, sqlite3.IntegrityError):
            raise HTTPException(
                status_code=409,
                detail="That Research Graph relationship already exists. Refresh the graph before editing again.",
            ) from exc
        raise exc

    @app.get("/api/workspaces/{workspace_id}/research-graphs")
    def _research_graph_catalog(
        workspace_id: str,
        include_archived: bool = True,
        thread_id: str = "",
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_from_id(workspace_id, identity)
        if thread_id:
            try:
                thread = _thread_store(workspace, workspace_name).get_thread(thread_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if thread.workspace_id != workspace_name:
                raise HTTPException(status_code=404, detail="Thread not found.")
        service = _research_graph_service(workspace, workspace_name)
        return JSONResponse(
            {
                "graphs": service.catalog(
                    include_archived=include_archived,
                    current_thread_id=thread_id,
                )
            }
        )

    @app.post("/api/workspaces/{workspace_id}/research-graphs")
    def _research_graph_create(workspace_id: str, payload: GraphCreateRequest):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_from_id(workspace_id, identity)
        try:
            result = _research_graph_service(
                workspace, workspace_name
            ).create_graph(payload)
        except Exception as exc:
            _raise_research_graph_http(exc)
        if research_graph_wakeup is not None:
            research_graph_wakeup.set()
        return JSONResponse(result)

    @app.get("/api/workspaces/{workspace_id}/research-graphs/{graph_id}")
    def _research_graph_get(
        workspace_id: str,
        graph_id: str,
        thread_id: str = "",
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_from_id(workspace_id, identity)
        try:
            result = _research_graph_service(
                workspace, workspace_name
            ).presentation(graph_id, current_thread_id=thread_id)
        except Exception as exc:
            _raise_research_graph_http(exc)
        return JSONResponse(result)

    @app.patch("/api/workspaces/{workspace_id}/research-graphs/{graph_id}")
    def _research_graph_patch(
        workspace_id: str,
        graph_id: str,
        payload: GraphPatchRequest,
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_from_id(workspace_id, identity)
        try:
            result = _research_graph_service(
                workspace, workspace_name
            ).patch_graph(graph_id, payload)
        except Exception as exc:
            _raise_research_graph_http(exc)
        if research_graph_wakeup is not None:
            research_graph_wakeup.set()
        return JSONResponse(result)

    @app.post(
        "/api/workspaces/{workspace_id}/research-graphs/{graph_id}/hypotheses"
    )
    def _research_graph_add_hypothesis(
        workspace_id: str,
        graph_id: str,
        payload: HypothesisCreateRequest,
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_from_id(workspace_id, identity)
        try:
            result = _research_graph_service(
                workspace, workspace_name
            ).add_hypothesis(graph_id, payload)
        except Exception as exc:
            _raise_research_graph_http(exc)
        if research_graph_wakeup is not None:
            research_graph_wakeup.set()
        return JSONResponse(result)

    @app.post(
        "/api/workspaces/{workspace_id}/research-graphs/{graph_id}/experiments"
    )
    def _research_graph_add_experiment(
        workspace_id: str,
        graph_id: str,
        payload: ExperimentCreateRequest,
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_from_id(workspace_id, identity)
        try:
            result = _research_graph_service(
                workspace, workspace_name
            ).add_experiment(graph_id, payload)
        except Exception as exc:
            _raise_research_graph_http(exc)
        if research_graph_wakeup is not None:
            research_graph_wakeup.set()
        return JSONResponse(result)

    @app.post(
        "/api/workspaces/{workspace_id}/research-graphs/{graph_id}/results"
    )
    def _research_graph_add_result(
        workspace_id: str,
        graph_id: str,
        payload: ResultCreateRequest,
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_from_id(workspace_id, identity)
        try:
            result = _research_graph_service(
                workspace, workspace_name
            ).record_result(graph_id, payload)
        except Exception as exc:
            _raise_research_graph_http(exc)
        if research_graph_wakeup is not None:
            research_graph_wakeup.set()
        return JSONResponse(result)

    @app.patch(
        "/api/workspaces/{workspace_id}/research-graphs/{graph_id}/nodes/{node_id}"
    )
    def _research_graph_update_node(
        workspace_id: str,
        graph_id: str,
        node_id: str,
        payload: NodePatchRequest,
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_from_id(workspace_id, identity)
        try:
            result = _research_graph_service(
                workspace, workspace_name
            ).update_node(graph_id, node_id, payload)
        except Exception as exc:
            _raise_research_graph_http(exc)
        if research_graph_wakeup is not None:
            research_graph_wakeup.set()
        return JSONResponse(result)

    @app.post(
        "/api/workspaces/{workspace_id}/research-graphs/{graph_id}/edges"
    )
    def _research_graph_add_edge(
        workspace_id: str,
        graph_id: str,
        payload: EdgeCreateRequest,
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_from_id(workspace_id, identity)
        try:
            result = _research_graph_service(
                workspace, workspace_name
            ).add_edge(
                graph_id,
                expected_revision=payload.expected_revision,
                source_node_id=payload.source_node_id,
                target_node_id=payload.target_node_id,
                relation=payload.relation,
            )
        except Exception as exc:
            _raise_research_graph_http(exc)
        if research_graph_wakeup is not None:
            research_graph_wakeup.set()
        return JSONResponse(result)

    @app.post(
        "/api/workspaces/{workspace_id}/research-graphs/{graph_id}/refs"
    )
    def _research_graph_add_ref(
        workspace_id: str,
        graph_id: str,
        payload: RefCreateRequest,
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_from_id(workspace_id, identity)
        try:
            result = _research_graph_service(
                workspace, workspace_name
            ).add_ref(
                graph_id,
                expected_revision=payload.expected_revision,
                node_id=payload.node_id,
                ref=payload,
            )
        except Exception as exc:
            _raise_research_graph_http(exc)
        if research_graph_wakeup is not None:
            research_graph_wakeup.set()
        return JSONResponse(result)

    @app.post(
        "/api/workspaces/{workspace_id}/research-graphs/{graph_id}"
        "/experiments/{node_id}/blocked"
    )
    def _research_graph_mark_blocked(
        workspace_id: str,
        graph_id: str,
        node_id: str,
        payload: ExperimentBlockedRequest,
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_from_id(workspace_id, identity)
        try:
            result = _research_graph_service(
                workspace, workspace_name
            ).mark_experiment_blocked(
                graph_id,
                node_id,
                expected_revision=payload.expected_revision,
                reason=payload.reason,
            )
        except Exception as exc:
            _raise_research_graph_http(exc)
        if research_graph_wakeup is not None:
            research_graph_wakeup.set()
        return JSONResponse(result)

    @app.post(
        "/api/workspaces/{workspace_id}/research-graphs/{graph_id}"
        "/experiments/{node_id}/launch"
    )
    async def _research_graph_launch(
        workspace_id: str,
        graph_id: str,
        node_id: str,
        payload: ExperimentLaunchRequest,
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_from_id(workspace_id, identity)
        try:
            result = await _research_graph_service(
                workspace, workspace_name
            ).launch_experiment(
                graph_id,
                node_id,
                expected_revision=payload.expected_revision,
                replicate=payload.replicate,
            )
        except Exception as exc:
            _raise_research_graph_http(exc)
        if research_graph_wakeup is not None:
            research_graph_wakeup.set()
        if hasattr(result.get("thread"), "model_dump"):
            result["thread"] = project_thread(result["thread"])
        return JSONResponse(result)

    @app.post(
        "/api/workspaces/{workspace_id}/research-graphs/{graph_id}/context"
    )
    def _research_graph_context(
        workspace_id: str,
        graph_id: str,
        payload: GraphContextRequest,
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_from_id(workspace_id, identity)
        try:
            result = _research_graph_service(
                workspace, workspace_name
            ).context_builder.build(
                graph_id,
                focus_node_id=payload.focus_node_id,
                query=payload.query,
                max_nodes=payload.max_nodes,
                max_chars=payload.max_chars,
            )
        except Exception as exc:
            _raise_research_graph_http(exc)
        return JSONResponse(result)

    @app.post(
        "/api/workspaces/{workspace_id}/research-graphs/{graph_id}/plan"
    )
    async def _research_graph_plan(
        workspace_id: str,
        graph_id: str,
        payload: GraphPlanningRequest,
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_from_id(workspace_id, identity)
        try:
            result = await _research_graph_service(
                workspace, workspace_name
            ).plan_next_step(
                graph_id,
                expected_revision=payload.expected_revision,
                focus_node_id=payload.focus_node_id,
            )
        except Exception as exc:
            _raise_research_graph_http(exc)
        if research_graph_wakeup is not None:
            research_graph_wakeup.set()
        if hasattr(result.get("thread"), "model_dump"):
            result["thread"] = project_thread(result["thread"])
        return JSONResponse(result)

    @app.get(
        "/api/workspaces/{workspace_id}/research-graphs/{graph_id}/stream"
    )
    async def _research_graph_stream(
        workspace_id: str,
        graph_id: str,
        request: Request,
        after_event_id: int = 0,
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_from_id(workspace_id, identity)
        service = _research_graph_service(workspace, workspace_name)
        try:
            service.store.get_graph(graph_id)
        except Exception as exc:
            _raise_research_graph_http(exc)
        header_cursor = str(request.headers.get("last-event-id") or "").strip()
        cursor = max(0, int(after_event_id or 0))
        if header_cursor:
            try:
                cursor = max(cursor, int(header_cursor))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Last-Event-ID must be an integer.",
                )

        async def _event_stream():
            current = cursor
            idle_ticks = 0
            while not await request.is_disconnected():
                rows = service.store.list_events(
                    graph_id=graph_id,
                    after_event_id=current,
                    limit=100,
                )
                if rows:
                    idle_ticks = 0
                    for row in rows:
                        current = max(current, int(row["event_id"]))
                        yield (
                            f"id: {row['event_id']}\n"
                            f"event: {row['event_type']}\n"
                            f"data: {json.dumps(row, ensure_ascii=False, separators=(',', ':'))}\n\n"
                        )
                else:
                    idle_ticks += 1
                    if idle_ticks >= 15:
                        idle_ticks = 0
                        yield ": keep-alive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.put("/api/threads/{thread_id}/active-research-graph")
    def _thread_bind_research_graph(
        thread_id: str,
        payload: ThreadGraphBindingRequest,
    ):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_for_thread(thread_id, identity)
        try:
            thread = _research_graph_service(
                workspace, workspace_name
            ).bind_thread(
                thread_id,
                graph_id=payload.graph_id,
                focus_node_id=payload.focus_node_id,
            )
        except Exception as exc:
            _raise_research_graph_http(exc)
        _broker_for_workspace(workspace).emit(
            thread_id,
            "thread.updated",
            status=str(thread.status.value),
            data={"thread": thread.model_dump(mode="json")},
        )
        return {"thread": project_thread(thread)}

    @app.post("/api/threads/{thread_id}/self-evolution/learn")
    async def _thread_explicit_learn(thread_id: str, request: Request):
        """Queue a user-authored durable correction for this thread's latest run."""

        _require_self_evolution_enabled()
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_for_thread(thread_id, identity)
        payload = await _json_body(request)
        note = str(payload.get("note") or "").strip()
        if not note:
            raise HTTPException(
                status_code=400,
                detail="Describe the durable correction you want CatMaster to learn.",
            )
        store = _thread_store(workspace, workspace_name)
        selected_message = None
        selected_run_id = ""
        message_id = str(payload.get("message_id") or "").strip()
        if message_id:
            try:
                selected_message = store.get_message(thread_id, message_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid message reference.") from exc
            if selected_message is None or selected_message.role != "assistant":
                raise HTTPException(
                    status_code=404,
                    detail="The selected assistant result was not found in this thread.",
                )
            selected_run_id = str(
                (selected_message.meta or {}).get("run_id") or ""
            ).strip()
        else:
            message_id, selected_run_id = store.latest_assistant_run(thread_id)
        run_id = selected_run_id
        if not run_id:
            raise HTTPException(
                status_code=409,
                detail="This thread has no completed run to attach the correction to.",
            )
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{3,160}", run_id):
            raise HTTPException(status_code=409, detail="The selected run reference is invalid.")
        run_dir = system_root(workspace) / "runs" / run_id
        if not run_dir.is_dir():
            raise HTTPException(
                status_code=404,
                detail="The selected run is no longer available for evidence review.",
            )
        coordinator = SelfEvolutionCoordinator(
            workspace=workspace,
            project_id=workspace_name,
        )
        job = coordinator.enqueue_explicit_learn(
            run_id=run_id,
            run_dir=run_dir,
            thread_id=thread_id,
            note=note,
            actor=identity.username,
        )
        if self_evolution_wakeup is not None:
            self_evolution_wakeup.set()
        return JSONResponse(
            {
                "queued": True,
                "job": project_self_evolution_job(job, workspace=workspace),
            }
        )

    @app.post("/api/threads/{thread_id}/submit", response_model=PublicSubmitEnvelope)
    async def _thread_submit(thread_id: str, payload: ThreadSubmitRequest):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_for_thread(thread_id, identity)
        payload = payload.model_copy(update={"entrypoint": _request_entrypoint(payload.entrypoint)})
        result = await _agent_loop(workspace, workspace_name).submit(thread_id=thread_id, payload=payload)
        return {
                "accepted": True,
                "queued": bool(result.get("queued")),
                "thread": project_thread(result["thread"]),
                "message": project_message(result["message"], workspace=workspace),
                **(
                    {
                        "assistant_message": project_message(
                            result["assistant_message"],
                            workspace=workspace,
                        )
                    }
                    if result.get("assistant_message")
                    else {}
                ),
        }

    @app.post("/api/threads/{thread_id}/stop", response_model=PublicStopEnvelope)
    async def _thread_stop(thread_id: str, payload: ThreadStopRequest):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_for_thread(thread_id, identity)
        result = await _agent_loop(workspace, workspace_name).stop(thread_id=thread_id, payload=payload)
        return {
                "accepted": True,
                "status": result["status"],
                "thread": project_thread(result["thread"]),
        }

    @app.post("/api/threads/{thread_id}/resume", response_model=PublicResumeEnvelope)
    async def _thread_resume(thread_id: str, payload: ThreadResumeRequest):
        identity = _identity_or_401()
        workspace, workspace_name = _workspace_for_thread(thread_id, identity)
        result = await _agent_loop(workspace, workspace_name).resume(
            thread_id=thread_id,
            payload=payload,
            validate_decisions=StreamingSpecialistRunner._validate_decisions,
        )
        return {
                "accepted": True,
                "assistant_message": project_message(
                    result["assistant_message"],
                    workspace=workspace,
                ),
                "thread": project_thread(result["thread"]),
        }

    @app.get("/api/threads/{thread_id}/stream", response_model=PublicEvent)
    async def _thread_stream(thread_id: str, request: Request, last_seq: str | None = None, once: bool = False):
        identity = _identity_or_401()
        workspace, _workspace_name = _workspace_for_thread(thread_id, identity)
        broker = _broker_for_workspace(workspace)

        async def _event_stream():
            replay_cursor = last_seq if last_seq is not None else request.headers.get("last-event-id")
            seq = _coerce_int(replay_cursor, broker.latest_seq(thread_id)) if replay_cursor is not None else broker.latest_seq(thread_id)
            seen_failures: set[str] = set()

            def _track_failure(public_event: Any) -> bool:
                public_status = str(
                    public_event.status or public_event.data.status or ""
                ).lower()
                message_key = (
                    f"message:{public_event.message_id}"
                    if public_event.message_id
                    else ""
                )
                thread_key = f"thread:{public_event.thread_id}"
                if public_status in {"running", "queued", "streaming"}:
                    seen_failures.discard(thread_key)
                    if message_key:
                        seen_failures.discard(message_key)
                if public_event.event != "run.failed":
                    return False
                failure_key = message_key or thread_key
                duplicate = failure_key in seen_failures
                seen_failures.add(failure_key)
                return duplicate

            # Rebuild the small deduplication state from the durable outbox.
            # Otherwise a Last-Event-ID reconnect can display the second raw
            # envelope for the same failure as a new user-facing failure.
            for historical_event in broker.replay_through(
                thread_id,
                through_seq=seq,
            ):
                _track_failure(project_event(historical_event, workspace=workspace))

            while True:
                if await request.is_disconnected():
                    break
                events, seq = await asyncio.to_thread(broker.wait_for_events, thread_id, last_seq=seq, timeout_s=10.0)
                if not events:
                    yield ": keepalive\n\n"
                    if once:
                        break
                    continue
                for event in events:
                    public_event = project_event(event, workspace=workspace)
                    if _track_failure(public_event):
                        continue
                    yield format_sse(public_event)
                if once:
                    break

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    @app.get(
        "/api/diagnostics/threads/{thread_id}/messages/{message_id}",
        response_model=DeveloperDiagnosticsPageEnvelope,
    )
    def _diagnostics_thread_message(
        thread_id: str,
        message_id: str,
        cursor: str = "",
        limit: int = DIAGNOSTICS_PAGE_LIMIT_CHARS,
    ):
        identity = _require_developer_diagnostics()
        workspace, workspace_name = _workspace_for_thread(thread_id, identity)
        message = _thread_store(workspace, workspace_name).get_message(thread_id, message_id)
        if message is None:
            raise HTTPException(status_code=404, detail="Message not found.")
        full_ref = f"/api/diagnostics/threads/{thread_id}/messages/{message_id}"
        cursor_identity = f"message:{thread_id}:{message_id}:{message.updated_at}"
        try:
            return _diagnostics_json_page(
                message.model_dump(mode="json"),
                cursor=cursor,
                limit=limit,
                cursor_identity=cursor_identity,
                full_content_ref=full_ref,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get(
        "/api/diagnostics/threads/{thread_id}/messages/{message_id}/parts/{part_id}",
        response_model=DeveloperDiagnosticsPageEnvelope,
    )
    def _diagnostics_thread_message_part(
        thread_id: str,
        message_id: str,
        part_id: str,
        cursor: str = "",
        limit: int = DIAGNOSTICS_PAGE_LIMIT_CHARS,
    ):
        identity = _require_developer_diagnostics()
        workspace, workspace_name = _workspace_for_thread(thread_id, identity)
        message = _thread_store(workspace, workspace_name).get_message(thread_id, message_id)
        if message is None:
            raise HTTPException(status_code=404, detail="Message not found.")
        part = next((item for item in message.parts if item.id == part_id), None)
        if part is None:
            raise HTTPException(status_code=404, detail="Message part not found.")
        full_ref = (
            f"/api/diagnostics/threads/{thread_id}/messages/{message_id}/parts/{part_id}"
        )
        cursor_identity = (
            f"part:{thread_id}:{message_id}:{part_id}:{message.updated_at}"
        )
        try:
            return _diagnostics_json_page(
                part.model_dump(mode="json"),
                cursor=cursor,
                limit=limit,
                cursor_identity=cursor_identity,
                full_content_ref=full_ref,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/diagnostics/threads/{thread_id}/events")
    def _diagnostics_thread_events(
        thread_id: str,
        after_seq: int = 0,
        limit: int = 200,
    ):
        identity = _require_developer_diagnostics()
        workspace, _workspace_name = _workspace_for_thread(thread_id, identity)
        events = _broker_for_workspace(workspace).replay(
            thread_id,
            last_seq=max(0, int(after_seq or 0)),
            limit=min(1000, max(1, int(limit or 200))),
        )
        return JSONResponse(
            {
                "warning": "Internal diagnostics may contain paths, provider data, and complete tool payloads.",
                "events": [event.model_dump(mode="json") for event in events],
                "next_cursor": str(events[-1].seq) if events else str(max(0, int(after_seq or 0))),
            }
        )

    @app.get("/api/artifacts/{artifact_id}/preview")
    def _artifact_preview(artifact_id: str):
        identity = _identity_or_401()
        found = ArtifactRegistry.find_in_project_root(_locked_user_root(identity), artifact_id)
        if found is None:
            raise HTTPException(status_code=404, detail="Artifact not found.")
        workspace, record = found
        payload = _file_content_payload(ctx="artifact", session=None, rel_path=record.path, workspace=workspace)
        payload["artifact"] = project_artifact(record, workspace=workspace)
        payload["download_url"] = record.download_url
        payload["content_url"] = f"/api/artifacts/{artifact_id}/content"
        if isinstance(payload.get("page"), dict):
            payload["page"]["full_content_ref"] = f"/api/artifacts/{artifact_id}/content"
        if isinstance(payload.get("structure"), dict):
            payload["structure"]["viewer_source_mode"] = "url"
            payload["structure"]["viewer_source_url"] = f"/api/artifacts/{artifact_id}/content"
        return JSONResponse(payload)

    @app.get("/api/artifacts/{artifact_id}/content")
    def _artifact_content(artifact_id: str):
        identity = _identity_or_401()
        found = ArtifactRegistry.find_in_project_root(_locked_user_root(identity), artifact_id)
        if found is None:
            raise HTTPException(status_code=404, detail="Artifact not found.")
        workspace, record = found
        registry_for_artifact = ArtifactRegistry(workspace=workspace)
        candidate = registry_for_artifact.resolve_path(record)
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Artifact file not found.")
        return FileResponse(candidate, media_type=mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")

    @app.get("/api/artifacts/{artifact_id}/download")
    def _artifact_download(artifact_id: str):
        identity = _identity_or_401()
        found = ArtifactRegistry.find_in_project_root(_locked_user_root(identity), artifact_id)
        if found is None:
            raise HTTPException(status_code=404, detail="Artifact not found.")
        workspace, record = found
        registry_for_artifact = ArtifactRegistry(workspace=workspace)
        candidate = registry_for_artifact.resolve_path(record)
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Artifact file not found.")
        return FileResponse(candidate, filename=candidate.name)

    @app.get("/api/session/{ctx}/snapshot")
    def _session_snapshot(ctx: str, lane: str = "research", run: str = "", project_space: str = ""):
        _require_legacy_routes()
        identity, _session = _bound_session(ctx)
        return JSONResponse(
            _with_auth(
                _build_snapshot(registry=registry, ctx=ctx, username=identity.username, lane=lane, run_name=run, project_space=project_space),
                identity,
            )
        )

    @app.get("/api/diagnostics/session/{ctx}/details")
    def _session_details(ctx: str, run: str = "", project_space: str = "", include_legacy_traces: bool = False):
        _require_developer_diagnostics()
        identity, _session = _bound_session(ctx)
        return JSONResponse(
            _build_details(
                registry=registry,
                ctx=ctx,
                username=identity.username,
                run_name=run,
                project_space=project_space,
                include_legacy_traces=include_legacy_traces,
                include_self_evolution=auth.enabled,
            )
        )

    @app.get("/api/diagnostics/session/{ctx}/events")
    def _session_events(
        ctx: str,
        run: str = "",
        project_space: str = "",
        limit: int = 200,
        before_id: int = 0,
        after_id: int = 0,
        before_seq: int = 0,
        after_seq: int = 0,
        channel: str = "",
        category: str = "",
        name: str = "",
        names: str = "",
        run_id: str = "",
        thread_id: str = "",
        agent: str = "",
        agent_name: str = "",
        tool: str = "",
        include_legacy_trace_records: bool = False,
    ):
        _require_developer_diagnostics()
        identity, _session = _bound_session(ctx)
        event_names = _split_csv(names) or _split_csv(name)
        return JSONResponse(
            _build_events(
                registry=registry,
                ctx=ctx,
                username=identity.username,
                run_name=run,
                project_space=project_space,
                limit=limit,
                before_id=before_id,
                after_id=after_id,
                before_seq=before_seq,
                after_seq=after_seq,
                channel=channel,
                category=category,
                names=event_names,
                run_id=run_id,
                thread_id=thread_id,
                agent_name=agent_name or agent,
                tool=tool,
                include_legacy_trace_records=include_legacy_trace_records,
            )
        )

    @app.get("/api/diagnostics/session/{ctx}/observability")
    def _session_observability(
        ctx: str,
        lane: str = "research",
        run: str = "",
        project_space: str = "",
        limit: int = 400,
    ):
        _require_developer_diagnostics()
        identity, _session = _bound_session(ctx)
        return JSONResponse(
            _build_observability(
                registry=registry,
                ctx=ctx,
                username=identity.username,
                lane=lane,
                run_name=run,
                project_space=project_space,
                limit=limit,
            )
        )

    @app.get("/api/session/{ctx}/monitor")
    def _session_monitor(
        ctx: str,
        lane: str = "research",
        run: str = "",
        project_space: str = "",
        limit: int = 400,
        cursor: str = "",
        timeline_limit: int = 200,
    ):
        identity, session = _bound_session(ctx)
        workspace, workspace_name = _workspace_for_request(registry, session, project_space)
        raw = _build_observability(
            registry=registry,
            ctx=ctx,
            username=identity.username,
            lane=lane,
            run_name=run,
            project_space=project_space,
            limit=min(1000, max(1, int(limit or 400))),
        )
        monitor_ref = (
            f"/api/session/{ctx}/monitor"
            f"?lane={quote(lane)}&run={quote(run)}"
            f"&project_space={quote(project_space)}&limit={min(1000, max(1, int(limit or 400)))}"
        )
        details_ref = (
            f"/api/diagnostics/session/{ctx}/details"
            f"?run={quote(run)}&project_space={quote(project_space)}"
            if developer_diagnostics_enabled
            else ""
        )
        try:
            projected = project_monitor_snapshot(
                raw,
                workspace=workspace,
                diagnostics_available=developer_diagnostics_enabled,
                timeline_cursor=cursor,
                timeline_limit=timeline_limit,
                timeline_identity=(
                    f"{workspace_name}:{raw.get('selected_run') or run or 'monitor'}"
                ),
                timeline_ref=monitor_ref,
                details_ref=details_ref,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(projected)

    @app.get("/api/diagnostics/session/{ctx}/memory")
    def _session_memory(ctx: str, run: str = "", source: str = "all", project_space: str = ""):
        _require_developer_diagnostics()
        identity, _session = _bound_session(ctx)
        return JSONResponse(_build_memory(registry=registry, ctx=ctx, username=identity.username, run_name=run, source=source, project_space=project_space))

    @app.get("/api/session/{ctx}/self-evolution/candidates")
    def _session_self_evolution_candidates(
        ctx: str,
        project_space: str = "",
        run: str = "",
        status: str = "",
        cursor: str = "",
        limit: int = 25,
        observation_status: str = "",
        observation_cursor: str = "",
        observation_limit: int = 25,
    ):
        _require_self_evolution_enabled()
        _identity, session = _bound_session(ctx)
        workspace, workspace_name = _workspace_for_request(registry, session, project_space)
        if workspace is None:
            raise HTTPException(status_code=400, detail="Open a project space first.")
        raw = _self_evolution_for_run(
            workspace=workspace,
            workspace_id=workspace_name,
            run_id=run,
            status=status,
            cursor=cursor,
            limit=limit,
            observation_status=observation_status,
            observation_cursor=observation_cursor,
            observation_limit=observation_limit,
        )
        return JSONResponse(
            project_self_evolution_payload(
                raw,
                workspace=workspace,
                ctx=ctx,
                workspace_name=workspace_name,
            )
        )

    @app.get("/api/session/{ctx}/self-evolution/candidates/{candidate_id}")
    def _session_self_evolution_candidate_detail(ctx: str, candidate_id: str, project_space: str = ""):
        """Compatibility alias for opening the latest immutable revision."""

        _require_self_evolution_enabled()
        _identity, session = _bound_session(ctx)
        workspace, workspace_name = _workspace_for_request(registry, session, project_space)
        if workspace is None:
            raise HTTPException(status_code=400, detail="Open a project space first.")
        store = SelfEvolutionStore(workspace, project_id=workspace_name)
        candidate = _read_self_evolution_candidate(store, candidate_id)
        return JSONResponse(
            {
                "candidate": project_self_evolution_candidate(
                    _self_evolution_candidate_payload(
                        store=store,
                        promotion=PromotionManager(store),
                        candidate=candidate,
                    ),
                    workspace=workspace,
                    ctx=ctx,
                    workspace_name=workspace_name,
                )
            }
        )

    @app.get(
        "/api/session/{ctx}/self-evolution/candidates/{candidate_id}"
        "/revisions/{revision}"
    )
    def _session_self_evolution_candidate_revision(
        ctx: str,
        candidate_id: str,
        revision: int,
        project_space: str = "",
    ):
        _require_self_evolution_enabled()
        _identity, session = _bound_session(ctx)
        workspace, workspace_name = _workspace_for_request(registry, session, project_space)
        if workspace is None:
            raise HTTPException(status_code=400, detail="Open a project space first.")
        store = SelfEvolutionStore(workspace, project_id=workspace_name)
        candidate, current = _read_self_evolution_candidate_revision(
            store,
            candidate_id,
            revision,
        )
        raw = _self_evolution_candidate_payload(
            store=store,
            promotion=PromotionManager(store),
            candidate=candidate,
        )
        read_only = candidate.revision != current.revision
        if read_only:
            raw["allowed_actions"] = []
            raw["promotion_readiness"] = {
                "ready": False,
                "canary_ready": False,
                "reason": "Historical revisions are read-only.",
            }
        return JSONResponse(
            {
                "candidate": project_self_evolution_candidate(
                    raw,
                    workspace=workspace,
                    ctx=ctx,
                    workspace_name=workspace_name,
                ),
                "read_only": read_only,
                "current_revision": current.revision,
            }
        )

    @app.get(
        "/api/session/{ctx}/self-evolution/candidates/{candidate_id}"
        "/revisions/{revision}/diff"
    )
    def _session_self_evolution_candidate_diff(
        ctx: str,
        candidate_id: str,
        revision: int,
        project_space: str = "",
        cursor: str = "",
        limit: int = 64_000,
    ):
        _require_self_evolution_enabled()
        _identity, session = _bound_session(ctx)
        workspace, workspace_name = _workspace_for_request(registry, session, project_space)
        if workspace is None:
            raise HTTPException(status_code=400, detail="Open a project space first.")
        store = SelfEvolutionStore(workspace, project_id=workspace_name)
        candidate, current = _read_self_evolution_candidate_revision(
            store,
            candidate_id,
            revision,
        )
        diff_text, _preview_truncated = _candidate_change_preview(store, candidate, limit_chars=None)
        start = 0
        cursor_identity = f"{candidate_id}@r{revision:04d}"
        if cursor:
            try:
                position = decode_public_cursor(
                    cursor,
                    kind="self_evolution_diff",
                    identity=cursor_identity,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if not isinstance(position, int) or position < 0:
                raise HTTPException(status_code=400, detail="Candidate-diff cursor is invalid.")
            start = min(len(diff_text), position)
        capped_limit = min(256_000, max(1_000, int(limit or 64_000)))
        visible = diff_text[start:start + capped_limit]
        next_position = start + len(visible)
        has_more = next_position < len(diff_text)
        full_ref = (
            f"/api/session/{ctx}/self-evolution/candidates/{candidate_id}"
            f"/revisions/{revision}/diff"
            f"?project_space={quote(workspace_name)}"
        )
        return JSONResponse(
            {
                "diff": visible,
                "read_only": candidate.revision != current.revision,
                "current_revision": current.revision,
                "page": {
                    "shown_count": next_position,
                    "total_count": len(diff_text),
                    "total_unknown": False,
                    "truncated": has_more,
                    "next_cursor": (
                        encode_public_cursor(
                            "self_evolution_diff",
                            cursor_identity,
                            next_position,
                        )
                        if has_more
                        else ""
                    ),
                    "full_content_ref": full_ref,
                    "unit": "characters",
                },
            }
        )

    @app.post("/api/session/{ctx}/self-evolution/process")
    async def _session_self_evolution_process(ctx: str, request: Request):
        _require_self_evolution_enabled()
        _require_developer_diagnostics()
        _identity, session = _bound_session(ctx)
        payload = await _json_body(request)
        workspace, workspace_name = _workspace_for_request(registry, session, str(payload.get("project_space") or ""))
        if workspace is None:
            raise HTTPException(status_code=400, detail="Open a project space first.")
        limit = _coerce_int(payload.get("limit"), 20)
        coordinator = SelfEvolutionCoordinator(workspace=workspace, project_id=workspace_name)
        jobs = await asyncio.to_thread(coordinator.process_pending_jobs, limit=limit)
        return JSONResponse(
            {
                "processed": [
                    project_self_evolution_job(job, workspace=workspace)
                    for job in jobs
                ]
            }
        )

    @app.post("/api/session/{ctx}/self-evolution/learn")
    async def _session_self_evolution_learn(ctx: str, request: Request):
        _require_self_evolution_enabled()
        identity, session = _bound_session(ctx)
        payload = await _json_body(request)
        workspace, workspace_name = _workspace_for_request(registry, session, str(payload.get("project_space") or ""))
        if workspace is None:
            raise HTTPException(status_code=400, detail="Open a project space first.")
        run_name = str(payload.get("run") or payload.get("run_id") or "").strip()
        if not run_name:
            raise HTTPException(status_code=400, detail="run or run_id is required.")
        run_dir, selected_run = _run_dir_for_name(session, run_name, workspace=workspace)
        coordinator = SelfEvolutionCoordinator(workspace=workspace, project_id=workspace_name)
        job = coordinator.enqueue_explicit_learn(
            run_id=selected_run,
            run_dir=run_dir,
            thread_id=str(payload.get("thread_id") or ""),
            note=str(payload.get("note") or ""),
            model_config=str(payload.get("model_config") or ""),
            actor=identity.username,
        )
        if self_evolution_wakeup is not None:
            self_evolution_wakeup.set()
        return JSONResponse(
            {
                "queued": True,
                "job": project_self_evolution_job(job, workspace=workspace),
            }
        )

    @app.post(
        "/api/session/{ctx}/self-evolution/candidates/{candidate_id}"
        "/revisions/{revision}/{action}"
    )
    async def _session_self_evolution_candidate_action(
        ctx: str,
        candidate_id: str,
        revision: int,
        action: str,
        request: Request,
    ):
        _require_self_evolution_enabled()
        identity, session = _bound_session(ctx)
        payload = await _json_body(request)
        workspace, workspace_name = _workspace_for_request(registry, session, str(payload.get("project_space") or ""))
        if workspace is None:
            raise HTTPException(status_code=400, detail="Open a project space first.")
        coordinator = SelfEvolutionCoordinator(workspace=workspace, project_id=workspace_name)
        candidate = _read_self_evolution_candidate(coordinator.store, candidate_id)
        if candidate.revision != int(revision):
            raise HTTPException(
                status_code=409,
                detail="This candidate has a newer revision. Reopen it before deciding.",
            )
        action_name = str(action or "").strip().lower()
        allowed = {
            "run-review",
            "request-revision",
            "start-canary",
            "promote-stable",
            "reject",
            "quarantine",
            "retire",
            "rollback",
        }
        if action_name not in allowed:
            raise HTTPException(status_code=404, detail="Unknown candidate action.")
        rationale = str(payload.get("rationale") or payload.get("guidance") or "").strip()
        queued_job = None
        try:
            if action_name == "run-review":
                updated = await asyncio.to_thread(
                    coordinator.review_candidate,
                    candidate_id=candidate_id,
                    expected_revision=revision,
                    model_config=str(payload.get("model_config") or ""),
                )
            elif action_name == "request-revision":
                guidance = str(payload.get("guidance") or rationale).strip()
                updated = coordinator.promotion.request_revision(
                    candidate,
                    actor=identity.username,
                    rationale=guidance,
                )
                queued_job = coordinator.enqueue_revision(
                    candidate_id=candidate_id,
                    expected_revision=revision,
                    guidance=guidance,
                    actor=identity.username,
                    model_config=str(payload.get("model_config") or ""),
                )
                if self_evolution_wakeup is not None:
                    self_evolution_wakeup.set()
            elif action_name == "start-canary":
                thread_ids = [
                    str(item)
                    for item in payload.get("thread_ids", [])
                    if str(item).strip()
                ] if isinstance(payload.get("thread_ids"), list) else []
                run_ids = [
                    str(item)
                    for item in payload.get("run_ids", [])
                    if str(item).strip()
                ] if isinstance(payload.get("run_ids"), list) else []
                scope_kind = str(payload.get("scope_kind") or "").strip()
                scope_id = str(payload.get("scope_id") or "").strip()
                if scope_kind == "thread" and scope_id:
                    thread_ids.append(scope_id)
                elif scope_kind == "run" and scope_id:
                    run_ids.append(scope_id)
                report = coordinator.gate.run(candidate)
                updated = coordinator.promotion.start_canary(
                    candidate,
                    report,
                    actor=identity.username,
                    thread_ids=thread_ids,
                    run_ids=run_ids,
                    rationale=rationale,
                )
            elif action_name == "promote-stable":
                report = coordinator.gate.run(candidate)
                updated = coordinator.promotion.promote_stable(
                    candidate,
                    report,
                    actor=identity.username,
                    rationale=rationale,
                )
            elif action_name == "reject":
                updated = coordinator.promotion.reject(
                    candidate,
                    actor=identity.username,
                    rationale=rationale,
                )
            elif action_name == "quarantine":
                updated = coordinator.promotion.quarantine(
                    candidate,
                    actor=identity.username,
                    rationale=rationale,
                )
            elif action_name == "retire":
                updated = coordinator.promotion.retire(
                    candidate,
                    actor=identity.username,
                    rationale=rationale,
                )
            else:
                updated = coordinator.promotion.rollback(
                    candidate,
                    actor=identity.username,
                    rationale=rationale,
                )
        except PromotionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (FileExistsError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        projected = project_self_evolution_candidate(
            _self_evolution_candidate_payload(
                store=coordinator.store,
                promotion=coordinator.promotion,
                candidate=updated,
            ),
            workspace=workspace,
            ctx=ctx,
            workspace_name=workspace_name,
        )
        return JSONResponse(
            {
                "updated": True,
                "candidate": projected,
                "revision_job": (
                    project_self_evolution_job(queued_job, workspace=workspace)
                    if queued_job is not None
                    else None
                ),
            }
        )

    @app.get("/api/session/{ctx}/files/tree")
    def _session_files_tree(
        ctx: str,
        path: str = "",
        project_space: str = "",
        cursor: str = "",
        limit: int = 200,
    ):
        _identity, session = _bound_session(ctx)
        workspace, workspace_name = _workspace_for_request(registry, session, project_space)
        workspace_root, directory, normalized_path = _resolve_workspace_entry(session, path, workspace=workspace)
        if not directory.is_dir():
            raise HTTPException(status_code=400, detail="Requested path is not a directory.")
        all_entries = _list_directory_entries(directory, workspace_root=workspace_root, limit=None)
        start = 0
        if cursor:
            try:
                after_path = decode_public_cursor(
                    cursor,
                    kind="files_tree",
                    identity=normalized_path,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if not isinstance(after_path, str):
                raise HTTPException(status_code=400, detail="Files cursor is invalid.")
            after_index = next(
                (index for index, item in enumerate(all_entries) if item["path"] == after_path),
                None,
            )
            if after_index is None:
                raise HTTPException(status_code=400, detail="Files cursor is stale.")
            start = after_index + 1
        capped_limit = min(500, max(1, int(limit or 200)))
        children = all_entries[start:start + capped_limit]
        next_index = start + len(children)
        has_more = next_index < len(all_entries)
        return JSONResponse(
            {
                "path": normalized_path,
                "workspace_name": workspace_name,
                "children": children,
                "page": {
                    "shown_count": next_index,
                    "total_count": len(all_entries),
                    "total_unknown": False,
                    "truncated": has_more,
                    "next_cursor": (
                        encode_public_cursor("files_tree", normalized_path, children[-1]["path"])
                        if has_more and children
                        else ""
                    ),
                    "full_content_ref": "",
                    "unit": "items",
                },
            }
        )

    @app.get("/api/session/{ctx}/files/content")
    def _session_file_content(
        ctx: str,
        path: str,
        project_space: str = "",
        cursor: int = 0,
        limit: int = TEXT_PREVIEW_LIMIT_BYTES,
    ):
        _identity, session = _bound_session(ctx)
        workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
        return JSONResponse(
            _file_content_payload(
                ctx=ctx,
                session=session,
                rel_path=path,
                workspace=workspace,
                cursor=cursor,
                limit=limit,
            )
        )

    @app.post("/api/structures/open")
    async def _open_structure_document(request: Request):
        identity = _identity_or_401()
        body = await _validated_body(StructureOpenRequest, request)
        workspace, _workspace_name = _workspace_from_id(body.workspace, identity)
        try:
            return JSONResponse(open_structure(workspace, body))
        except StructureSerializationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/structures/transform")
    async def _transform_structure_document(request: Request):
        _identity_or_401()
        try:
            body = TRANSFORM_REQUEST_ADAPTER.validate_python(await _json_body(request))
            return JSONResponse(apply_transform(body))
        except (StructureSerializationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/structures/save")
    async def _save_structure_document(request: Request):
        identity = _identity_or_401()
        body = await _validated_body(SaveStructureRequest, request)
        workspace, _workspace_name = _workspace_from_id(body.workspace, identity)
        try:
            return JSONResponse(save_structure(workspace, body))
        except StructureVersionConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "source_changed", "message": str(exc)},
            ) from exc
        except StructureFormatLossError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "format_loss",
                    "message": "This format cannot preserve all scientific information.",
                    "warnings": exc.warnings,
                },
            ) from exc
        except StructureSerializationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/trajectories/meta")
    def _trajectory_metadata(workspace: str, path: str):
        identity = _identity_or_401()
        workspace_path, _workspace_name = _workspace_from_id(workspace, identity)
        try:
            return JSONResponse(get_trajectory_meta(workspace_path, path))
        except (StructureSerializationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/trajectories/frame")
    def _trajectory_frame(workspace: str, path: str, index: int):
        identity = _identity_or_401()
        workspace_path, _workspace_name = _workspace_from_id(workspace, identity)
        try:
            return JSONResponse(get_trajectory_frame(workspace_path, path, index))
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (StructureSerializationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/session/{ctx}/files/structure-animation")
    def _session_structure_animation(ctx: str, path: str, project_space: str = ""):
        _identity, session = _bound_session(ctx)
        workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
        return _structure_animation_response(session=session, rel_path=path, workspace=workspace)

    @app.get("/api/session/{ctx}/files/view")
    def _session_structure_view(ctx: str, path: str, project_space: str = ""):
        _identity, session = _bound_session(ctx)
        workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
        return _structure_view_response(session=session, rel_path=path, workspace=workspace)

    @app.get("/api/session/{ctx}/files/structure-vibration")
    def _session_structure_vibration(ctx: str, path: str, mode: int = -1, project_space: str = ""):
        _identity, session = _bound_session(ctx)
        workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
        return _structure_vibration_response(session=session, rel_path=path, mode_index=mode, workspace=workspace)

    @app.get("/api/session/{ctx}/files/download")
    def _session_file_download(ctx: str, path: str, project_space: str = ""):
        _identity, session = _bound_session(ctx)
        workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
        _workspace_root, candidate, _normalized_path = _resolve_workspace_entry(session, path, workspace=workspace)
        if not candidate.is_file():
            raise HTTPException(status_code=400, detail="Only files can be downloaded.")
        return FileResponse(candidate, filename=candidate.name)

    @app.get("/api/session/{ctx}/files/archive")
    def _session_file_archive(ctx: str, path: str = "", project_space: str = ""):
        _identity, session = _bound_session(ctx)
        workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
        return _archive_workspace_entry(session=session, rel_path=path, workspace=workspace)

    @app.post("/api/session/{ctx}/files/upload")
    async def _session_file_upload(
        ctx: str,
        request: Request,
        path: str = "files",
        filename: str = "",
        overwrite: bool = False,
        unzip: bool = False,
        project_space: str = "",
    ):
        _identity, session = _bound_session(ctx)
        workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
        return JSONResponse(
            await _upload_workspace_file(
                session=session,
                rel_path=path,
                filename=filename,
                request=request,
                overwrite=overwrite,
                unzip=unzip,
                workspace=workspace,
            )
        )

    @app.delete("/api/session/{ctx}/files/delete")
    def _session_file_delete(ctx: str, path: str, project_space: str = ""):
        _identity, session = _bound_session(ctx)
        workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
        return JSONResponse(_delete_workspace_entry(session=session, rel_path=path, workspace=workspace))

    @app.post("/api/session/{ctx}/workspace/open")
    async def _workspace_open(ctx: str, request: Request):
        payload = await _json_body(request)
        identity, session = _bound_session(ctx)
        project_space = str(payload.get("workspace") or "")
        ok, message = session.open_workspace_by_name(project_space, set_current=False)
        return JSONResponse(
            _public_workspace_result(
                session=session,
                identity=identity,
                ok=ok,
                status_message=message,
                workspace_name=project_space if ok else "",
            )
        )

    @app.post("/api/session/{ctx}/workspace/refresh")
    async def _workspace_refresh(ctx: str, request: Request):
        payload = await _json_body(request)
        identity, session = _bound_session(ctx)
        project_space = str(payload.get("workspace") or payload.get("project_space") or "")
        workspace, workspace_name = _workspace_for_request(registry, session, project_space)
        return JSONResponse(
            _public_workspace_result(
                session=session,
                identity=identity,
                ok=workspace is not None,
                status_message=(
                    f"Workspace {workspace_name} is available."
                    if workspace is not None
                    else "No workspace is open."
                ),
                workspace_name=workspace_name,
            )
        )

    @app.post("/api/session/{ctx}/workspace/create")
    async def _workspace_create(ctx: str, request: Request):
        payload = await _json_body(request)
        identity, session = _bound_session(ctx)
        project_space = str(payload.get("workspace") or "")
        ok, message = session.create_workspace(project_space, set_current=False)
        return JSONResponse(
            _public_workspace_result(
                session=session,
                identity=identity,
                ok=ok,
                status_message=message,
                workspace_name=project_space if ok else "",
            )
        )

    @app.delete("/api/session/{ctx}/workspace/delete")
    async def _workspace_delete(ctx: str, request: Request):
        payload = await _json_body(request)
        identity, session = _bound_session(ctx)
        project_space = str(payload.get("workspace") or "").strip()
        confirm_name = str(payload.get("confirm_name") or "").strip()
        active_workspace = str(payload.get("active_workspace") or "").strip()
        if not project_space:
            raise HTTPException(status_code=400, detail="Workspace name is required.")
        if confirm_name != project_space:
            raise HTTPException(status_code=400, detail="Workspace delete requires confirm_name to match workspace.")
        if active_workspace and active_workspace == project_space:
            raise HTTPException(status_code=400, detail="Switch away from the active workspace before deleting it.")
        root = _locked_user_root(identity).expanduser().resolve()
        target = session.resolve_workspace_by_name(project_space)
        if target is None:
            raise HTTPException(status_code=404, detail=f"Project space not found: {project_space}")
        target = target.expanduser().resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Workspace path escapes the locked root.") from exc
        current = Path(str(session.current_workspace_path() or "")).expanduser().resolve()
        if target == current:
            raise HTTPException(status_code=400, detail="Switch away from the active workspace before deleting it.")
        if target == root:
            raise HTTPException(status_code=400, detail="Refusing to delete the workspace root.")
        shutil.rmtree(target)
        return JSONResponse(
            _public_workspace_result(
                session=session,
                identity=identity,
                ok=True,
                status_message=f"Deleted workspace {project_space}.",
            )
        )

    @app.post("/api/session/{ctx}/run/select")
    async def _run_select(ctx: str, request: Request):
        _require_legacy_routes()
        payload = await _json_body(request)
        identity, session = _bound_session(ctx)
        project_space = str(payload.get("project_space") or payload.get("workspace") or "")
        workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
        message = session.select_run(str(payload.get("run_name") or ""), workspace=workspace)
        snapshot = _build_snapshot(
            registry=registry,
            ctx=ctx,
            username=identity.username,
            lane=str(payload.get("lane") or "research"),
            run_name=str(payload.get("run_name") or ""),
            project_space=project_space,
        )
        snapshot["status_message"] = message
        return JSONResponse(_with_auth(snapshot, identity))

    @app.post("/api/session/{ctx}/chat/create")
    async def _chat_create(ctx: str, request: Request):
        _require_legacy_routes()
        payload = await _json_body(request)
        identity, session = _bound_session(ctx)
        project_space = str(payload.get("project_space") or payload.get("workspace") or "")
        workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
        session_id = session.create_chat_session(workspace=workspace)
        snapshot = _build_snapshot(
            registry=registry,
            ctx=ctx,
            username=identity.username,
            lane=str(payload.get("lane") or "research"),
            project_space=project_space,
        )
        snapshot = _apply_chat_session_view(snapshot)
        snapshot["status_message"] = f"Started new chat session: {session_id}"
        return JSONResponse(_with_auth(snapshot, identity))

    @app.post("/api/session/{ctx}/chat/select")
    async def _chat_select(ctx: str, request: Request):
        _require_legacy_routes()
        payload = await _json_body(request)
        identity, session = _bound_session(ctx)
        project_space = str(payload.get("project_space") or payload.get("workspace") or "")
        workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
        session_id = session.select_chat_session(str(payload.get("session_id") or ""), workspace=workspace)
        snapshot = _build_snapshot(
            registry=registry,
            ctx=ctx,
            username=identity.username,
            lane=str(payload.get("lane") or "research"),
            project_space=project_space,
        )
        snapshot = _apply_chat_session_view(snapshot)
        snapshot["status_message"] = f"Switched to chat session: {session_id}" if session_id else "Chat session not found."
        return JSONResponse(_with_auth(snapshot, identity))

    @app.post("/api/session/{ctx}/run/start")
    async def _run_start(ctx: str, request: Request):
        _require_legacy_routes()
        payload = await _json_body(request)
        identity, session = _bound_session(ctx)
        project_space = str(payload.get("project_space") or payload.get("workspace") or "")
        workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
        message = session.start_run(
            prompt=str(payload.get("prompt") or ""),
            lane=str(payload.get("lane") or "research"),
            run_mode=str(payload.get("run_mode") or "new_run"),
            resume_run_name=str(payload.get("resume_run_name") or ""),
            proposal_review=bool(payload.get("proposal_review", False)),
            log_llm=bool(payload.get("log_llm", False)),
            full_auto_major=bool(payload.get("full_auto_major", False)),
            seed_hypotheses=str(payload.get("seed_hypotheses") or ""),
            exploration_policy=str(payload.get("exploration_policy") or "anchored"),
            writing_mode=str(payload.get("writing_mode") or "none"),
            output_format=str(payload.get("output_format") or "md"),
            target_section=str(payload.get("target_section") or ""),
            max_cycles=int(payload.get("max_cycles") or 6),
            max_literature_queries=int(payload.get("max_literature_queries") or 4),
            max_fast_runs=int(payload.get("max_fast_runs") or 3),
            max_standard_runs=int(payload.get("max_standard_runs") or 2),
            allow_deep_report=bool(payload.get("allow_deep_report", False)),
            workspace=workspace,
        )
        snapshot = _build_snapshot(
            registry=registry,
            ctx=ctx,
            username=identity.username,
            lane=str(payload.get("lane") or "research"),
            project_space=project_space,
        )
        snapshot["status_message"] = message
        return JSONResponse(_with_auth(snapshot, identity))

    @app.post("/api/session/{ctx}/run/interrupt")
    async def _run_interrupt(ctx: str, request: Request):
        _require_legacy_routes()
        payload = await _json_body(request)
        identity, session = _bound_session(ctx)
        project_space = str(payload.get("project_space") or payload.get("workspace") or "")
        workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
        message = session.request_interrupt_current_run(note=str(payload.get("note") or ""), workspace=workspace)
        snapshot = _build_snapshot(
            registry=registry,
            ctx=ctx,
            username=identity.username,
            lane=str(payload.get("lane") or "research"),
            project_space=project_space,
        )
        snapshot["status_message"] = message
        return JSONResponse(_with_auth(snapshot, identity))

    @app.post("/api/session/{ctx}/prompt/respond")
    async def _prompt_respond(ctx: str, request: Request):
        _ = (ctx, request)
        raise HTTPException(status_code=410, detail="Prompt-response HITL endpoint has been removed.")

    @app.get("/api/session/{ctx}/stream")
    async def _session_stream(ctx: str, request: Request, last_seq: str = "0", project_space: str = ""):
        _require_legacy_routes()
        identity, _session = _bound_session(ctx)
        username = identity.username

        async def _event_stream():
            seq = _coerce_int(last_seq, 0)
            while True:
                if await request.is_disconnected():
                    break
                session = registry.get_session(ctx, username=username)
                session.set_workspace_root(str(_locked_user_root(identity)))
                try:
                    workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
                except HTTPException:
                    workspace = None
                with session._lock:
                    runtime_state = session._runtime_for_workspace_unlocked(workspace, create=True)
                    reporter = runtime_state.reporter
                if reporter is None:
                    yield ": keepalive\n\n"
                    await asyncio.sleep(1.0)
                    continue
                events, seq = await asyncio.to_thread(reporter.wait_for_events_since, seq, timeout_s=10.0)
                if not events:
                    yield ": keepalive\n\n"
                    continue
                runtime = _runtime_snapshot(session, workspace=workspace)
                for event in events:
                    envelope = {
                        "event": event,
                        "runtime": runtime,
                        "run_status": str(runtime_state.run_status or "idle"),
                        "run_status_text": session.run_status_text(workspace=workspace),
                    }
                    envelope.update(_stream_patch(session, runtime, workspace=workspace))
                    yield f"id: {int(event.get('seq') or seq)}\ndata: {json.dumps(envelope, ensure_ascii=False)}\n\n"

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    return app


def launch(
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    project_space_root: Optional[str] = None,
    no_login: bool = False,
    disable_registration: bool = False,
    timeout_keep_alive: int = 0,
    timeout_graceful_shutdown: int = 0,
) -> None:
    if project_space_root is None:
        project_space_root = str(Path.cwd() / "project_space")
    app = create_app(
        project_space_root=project_space_root,
        no_login=no_login,
        disable_registration=disable_registration,
    )
    run_kwargs = {
        "host": host,
        "port": port,
        "log_level": "info",
        "timeout_keep_alive": max(0, int(timeout_keep_alive)),
        "timeout_graceful_shutdown": max(0, int(timeout_graceful_shutdown)),
    }
    try:
        uvicorn.run(app, **run_kwargs)
    except TypeError:
        run_kwargs.pop("timeout_graceful_shutdown", None)
        uvicorn.run(app, **run_kwargs)


__all__ = ["create_app", "launch"]
