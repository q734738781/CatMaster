from __future__ import annotations

import asyncio
import json
import mimetypes
import re
import shutil
import tempfile
import zipfile
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

from .auth import AuthIdentity, AuthManager, SESSION_COOKIE_NAME, SESSION_TTL_SECONDS
from .session_registry import SessionRegistry

TEXT_PREVIEW_LIMIT_BYTES = 160_000
DIRECTORY_PREVIEW_LIMIT = 40
STRUCTURE_ANIMATION_FRAME_LIMIT = 240
UPLOAD_LIMIT_BYTES = 512 * 1024 * 1024
ARCHIVE_ENTRY_LIMIT = 20_000
ARCHIVE_TOTAL_BYTES_LIMIT = 2 * 1024 * 1024 * 1024
UNZIP_ENTRY_LIMIT = 20_000
UNZIP_TOTAL_BYTES_LIMIT = 2 * 1024 * 1024 * 1024
STRUCTURE_FILE_SUFFIXES = {
    ".cif",
    ".cssr",
    ".cube",
    ".gro",
    ".mol",
    ".mol2",
    ".pdb",
    ".sdf",
    ".traj",
    ".vasp",
    ".xsf",
    ".xyz",
}
STRUCTURE_FILE_NAMES = {"POSCAR", "CONTCAR", "OUTCAR", "XDATCAR"}
MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx", ".rst"}
JSON_SUFFIXES = {".json", ".jsonl", ".geojson"}
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
    return workspace_root, candidate, rel_text


def _safe_upload_filename(filename: str) -> str:
    name = Path(str(filename or "").replace("\\", "/")).name.strip()
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Upload filename is required.")
    if "/" in name or "\\" in name or "\x00" in name:
        raise HTTPException(status_code=400, detail="Upload filename is invalid.")
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


def _entry_preview_kind(path: Path, *, mime_type: str = "") -> str:
    suffix = path.suffix.lower()
    if path.name.upper() in STRUCTURE_FILE_NAMES or suffix in STRUCTURE_FILE_SUFFIXES:
        return "structure"
    if mime_type.startswith("image/"):
        return "image"
    if suffix in MARKDOWN_SUFFIXES:
        return "markdown"
    if suffix in JSON_SUFFIXES:
        return "json"
    if mime_type.startswith("text/") or suffix in TEXTLIKE_SUFFIXES:
        return "text"
    return "binary"


def _directory_has_children(path: Path) -> bool:
    try:
        next(path.iterdir())
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
        "preview_kind": "directory" if node_type == "directory" else _entry_preview_kind(path, mime_type=mime_type),
    }


def _list_directory_entries(directory: Path, *, workspace_root: Path, limit: int = 500) -> list[dict[str, Any]]:
    children = [child for child in directory.iterdir()]
    children.sort(key=lambda item: (0 if item.is_dir() else 1, item.name.lower()))
    if directory == workspace_root:
        preferred = {"files": 0, "metadata": 1}
        children.sort(key=lambda item: (preferred.get(item.name, 10), 0 if item.is_dir() else 1, item.name.lower()))
    return [_serialize_tree_entry(child, workspace_root=workspace_root) for child in children[:limit]]


def _read_text_preview(path: Path) -> tuple[str, bool]:
    with path.open("rb") as handle:
        raw = handle.read(TEXT_PREVIEW_LIMIT_BYTES + 1)
    truncated = len(raw) > TEXT_PREVIEW_LIMIT_BYTES
    if truncated:
        raw = raw[:TEXT_PREVIEW_LIMIT_BYTES]
    return raw.decode("utf-8", errors="replace"), truncated


def _read_structure_frames(path: Path, *, limit: int = STRUCTURE_ANIMATION_FRAME_LIMIT) -> Optional[tuple[list[Any], int, bool]]:
    try:
        from ase.io import read as ase_read
    except Exception:
        return None
    try:
        frames = ase_read(str(path), index=":")
    except Exception:
        return None
    if frames is None:
        return None
    if not isinstance(frames, list):
        frames = [frames]
    total = len(frames)
    if not total:
        return None
    truncated = total > limit
    return frames[:limit], total, truncated


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
    frame_bundle = _read_structure_frames(path)
    if frame_bundle is None:
        return None
    frames, total_frames, frames_truncated = frame_bundle
    atoms = frames[0]
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
        "frames_truncated": bool(frames_truncated),
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


def _structure_animation_response(*, session, rel_path: str, workspace: Optional[Path] = None) -> Response:
    _workspace_root, candidate, _normalized_path = _resolve_workspace_entry(session, rel_path, workspace=workspace)
    if not candidate.is_file():
        raise HTTPException(status_code=400, detail="Only files can be viewed.")
    frame_bundle = _read_structure_frames(candidate)
    if frame_bundle is None:
        raise HTTPException(status_code=400, detail="Structure trajectory could not be parsed.")
    frames, _total_frames, _frames_truncated = frame_bundle
    payload = StringIO()
    try:
        from ase.io import write as ase_write

        ase_write(payload, frames, format="xyz")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to serialize trajectory: {exc}") from exc
    return Response(payload.getvalue(), media_type="chemical/x-xyz")


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


def _file_content_payload(*, ctx: str, session, rel_path: str, workspace: Optional[Path] = None) -> dict[str, Any]:
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
        payload["children"] = _list_directory_entries(candidate, workspace_root=workspace_root, limit=DIRECTORY_PREVIEW_LIMIT)
        return payload

    kind = _entry_preview_kind(candidate, mime_type=mime_type)
    payload["kind"] = kind
    if kind == "image":
        return payload
    if kind == "structure":
        payload["structure"] = _build_structure_payload(candidate, ctx=ctx, rel_path=normalized_path, project_space=project_space)
        preview_text, truncated = _read_text_preview(candidate)
        payload["preview_text"] = preview_text
        payload["truncated"] = truncated
        return payload
    if kind in {"text", "markdown", "json"} or mime_type.startswith("text/"):
        preview_text, truncated = _read_text_preview(candidate)
        payload["preview_text"] = preview_text
        payload["truncated"] = truncated
        return payload
    payload["preview_text"] = ""
    payload["truncated"] = False
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
    archive_base = candidate.name if normalized_path else workspace_root.name
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
                    if not item.is_file():
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
        "output_tokens",
        "total_tokens",
        "input_cached_tokens",
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
    return {
        "active_run": active_run,
        "selected_run": selected_run,
        "chat_messages": session.get_chat_messages(workspace=workspace),
        "cards": _serialize_cards(session.list_run_cards(workspace=workspace)),
        "usage_summary": usage_summary,
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
        event_page = session.read_ui_events(run_dir, limit=200)
        events = list(event_page.get("events") or []) or list(runtime.get("recent_events") or [])
        usage_summary = _merge_usage_summary(runtime.get("usage_totals"), session.read_usage_summary(run_dir))
        llm = dict(runtime.get("llm") or {})
        graph = dict(runtime.get("graph") or {})
    else:
        live_state = session.snapshot_live_state(run_dir, workspace=workspace)
        event_page = session.read_ui_events(run_dir, limit=200)
        events = list(event_page.get("events") or [])
        usage_summary = session.read_usage_summary(run_dir)
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
    snapshot["proposal"] = ""
    snapshot["todo_items"] = []
    snapshot["result_text"] = ""
    snapshot["can_submit_prompt"] = False
    return snapshot


def _build_details(*, registry: SessionRegistry, ctx: str, username: str = "admin", run_name: str, project_space: str = "") -> dict[str, Any]:
    session = registry.get_session(ctx, username=username)
    workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
    run_dir, selected_run = _run_dir_for_name(session, run_name, workspace=workspace)
    artifacts = session.read_artifacts(workspace=workspace)
    if hasattr(artifacts, "to_dict"):
        artifact_rows = artifacts.to_dict(orient="records")
    else:
        artifact_rows = list(artifacts or [])
    return {
        "selected_run": selected_run,
        "memory": session.read_memory_index(workspace=workspace),
        "artifacts": artifact_rows,
        "proposal": session.read_proposal(run_dir, workspace=workspace),
        "task_state": session.read_task_state(run_dir, workspace=workspace),
        "trace_event": session.read_trace(run_dir, "event_trace.jsonl", workspace=workspace),
        "trace_tool": session.read_trace(run_dir, "tool_trace.jsonl", workspace=workspace),
        "trace_patch": session.read_trace(run_dir, "patch_trace.jsonl", workspace=workspace),
    }


def _build_events(
    *,
    registry: SessionRegistry,
    ctx: str,
    username: str = "admin",
    run_name: str,
    project_space: str = "",
    limit: int = 200,
    before_seq: int = 0,
    after_seq: int = 0,
) -> dict[str, Any]:
    session = registry.get_session(ctx, username=username)
    workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
    run_dir, selected_run = _run_dir_for_name(session, run_name, workspace=workspace)
    page = session.read_ui_events(run_dir, limit=limit, before_seq=before_seq, after_seq=after_seq)
    page["selected_run"] = selected_run
    return page


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
    boot = json.dumps({"view": view}, ensure_ascii=False)
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
<body>
  <div id="app"></div>
  <script>window.CATMASTER_BOOT = {boot};</script>
  <script type="module" src="/static/app.js"></script>
</body>
</html>"""


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


def create_app(*, project_space_root: str, no_login: bool = False) -> FastAPI:
    default_project_space_root = str(Path(project_space_root).expanduser().resolve())
    registry = SessionRegistry(default_project_space_root=default_project_space_root)
    auth = AuthManager(auth_root=Path(default_project_space_root) / ".webui_auth", enabled=not no_login)
    app = FastAPI(title="CatMaster WebUI")
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

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
        return JSONResponse(auth.create_captcha())

    @app.post("/api/auth/register")
    async def _auth_register(request: Request):
        if not auth.enabled:
            return JSONResponse(auth.public_status(auth.default_identity()))
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
        query = request.url.query
        target = "/monitor/"
        if query:
            target = f"{target}?{query}"
        return RedirectResponse(url=target, status_code=307)

    @app.get("/files", include_in_schema=False)
    def _files_redirect(request: Request):
        query = request.url.query
        target = "/files/"
        if query:
            target = f"{target}?{query}"
        return RedirectResponse(url=target, status_code=307)

    @app.get("/", response_class=HTMLResponse)
    def _home_page() -> str:
        return _page_html(view="home")

    @app.get("/monitor/", response_class=HTMLResponse)
    def _monitor_page() -> str:
        return _page_html(view="monitor")

    @app.get("/files/", response_class=HTMLResponse)
    def _files_page() -> str:
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
        snapshot["workspace_root"] = state.project_space_root
        return JSONResponse(_with_auth(snapshot, identity))

    @app.get("/api/session/{ctx}/snapshot")
    def _session_snapshot(ctx: str, lane: str = "research", run: str = "", project_space: str = ""):
        identity, _session = _bound_session(ctx)
        return JSONResponse(
            _with_auth(
                _build_snapshot(registry=registry, ctx=ctx, username=identity.username, lane=lane, run_name=run, project_space=project_space),
                identity,
            )
        )

    @app.get("/api/session/{ctx}/details")
    def _session_details(ctx: str, run: str = "", project_space: str = ""):
        identity, _session = _bound_session(ctx)
        return JSONResponse(_build_details(registry=registry, ctx=ctx, username=identity.username, run_name=run, project_space=project_space))

    @app.get("/api/session/{ctx}/events")
    def _session_events(
        ctx: str,
        run: str = "",
        project_space: str = "",
        limit: int = 200,
        before_seq: int = 0,
        after_seq: int = 0,
    ):
        identity, _session = _bound_session(ctx)
        return JSONResponse(
            _build_events(
                registry=registry,
                ctx=ctx,
                username=identity.username,
                run_name=run,
                project_space=project_space,
                limit=limit,
                before_seq=before_seq,
                after_seq=after_seq,
            )
        )

    @app.get("/api/session/{ctx}/memory")
    def _session_memory(ctx: str, run: str = "", source: str = "all", project_space: str = ""):
        identity, _session = _bound_session(ctx)
        return JSONResponse(_build_memory(registry=registry, ctx=ctx, username=identity.username, run_name=run, source=source, project_space=project_space))

    @app.get("/api/session/{ctx}/files/tree")
    def _session_files_tree(ctx: str, path: str = "", project_space: str = ""):
        _identity, session = _bound_session(ctx)
        workspace, workspace_name = _workspace_for_request(registry, session, project_space)
        workspace_root, directory, normalized_path = _resolve_workspace_entry(session, path, workspace=workspace)
        if not directory.is_dir():
            raise HTTPException(status_code=400, detail="Requested path is not a directory.")
        return JSONResponse(
            {
                "path": normalized_path,
                "workspace_path": str(workspace_root),
                "workspace_name": workspace_name,
                "children": _list_directory_entries(directory, workspace_root=workspace_root),
            }
        )

    @app.get("/api/session/{ctx}/files/content")
    def _session_file_content(ctx: str, path: str, project_space: str = ""):
        _identity, session = _bound_session(ctx)
        workspace, _workspace_name = _workspace_for_request(registry, session, project_space)
        return JSONResponse(_file_content_payload(ctx=ctx, session=session, rel_path=path, workspace=workspace))

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
        snapshot = _build_snapshot(
            registry=registry,
            ctx=ctx,
            username=identity.username,
            lane=str(payload.get("lane") or "research"),
            project_space=project_space if ok else "",
        )
        snapshot["ok"] = ok
        snapshot["status_message"] = message
        return JSONResponse(_with_auth(snapshot, identity))

    @app.post("/api/session/{ctx}/workspace/refresh")
    async def _workspace_refresh(ctx: str, request: Request):
        payload = await _json_body(request)
        identity, _session = _bound_session(ctx)
        snapshot = _build_snapshot(
            registry=registry,
            ctx=ctx,
            username=identity.username,
            lane=str(payload.get("lane") or "research"),
            project_space=str(payload.get("workspace") or payload.get("project_space") or ""),
        )
        snapshot["ok"] = True
        snapshot["status_message"] = f"Project-space root is locked to {snapshot.get('workspace_root') or _locked_user_root(identity)}"
        return JSONResponse(_with_auth(snapshot, identity))

    @app.post("/api/session/{ctx}/workspace/create")
    async def _workspace_create(ctx: str, request: Request):
        payload = await _json_body(request)
        identity, session = _bound_session(ctx)
        project_space = str(payload.get("workspace") or "")
        ok, message = session.create_workspace(project_space, set_current=False)
        snapshot = _build_snapshot(
            registry=registry,
            ctx=ctx,
            username=identity.username,
            lane=str(payload.get("lane") or "research"),
            project_space=project_space if ok else "",
        )
        snapshot["ok"] = ok
        snapshot["status_message"] = message
        return JSONResponse(_with_auth(snapshot, identity))

    @app.post("/api/session/{ctx}/run/select")
    async def _run_select(ctx: str, request: Request):
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
    timeout_keep_alive: int = 0,
    timeout_graceful_shutdown: int = 0,
) -> None:
    if project_space_root is None:
        project_space_root = str(Path.cwd() / "project_space")
    app = create_app(project_space_root=project_space_root, no_login=no_login)
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
