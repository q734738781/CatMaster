from __future__ import annotations

import asyncio
import json
import mimetypes
import re
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .session_registry import SessionRegistry

TEXT_PREVIEW_LIMIT_BYTES = 160_000
DIRECTORY_PREVIEW_LIMIT = 40
STRUCTURE_ANIMATION_FRAME_LIMIT = 240
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


def _workspace_root_for_session(session) -> Path:
    workspace_path = str(session.current_workspace_path() or "").strip()
    if not workspace_path:
        raise HTTPException(status_code=400, detail="Open a project space first.")
    workspace_root = Path(workspace_path).expanduser().resolve()
    if not workspace_root.exists() or not workspace_root.is_dir():
        raise HTTPException(status_code=404, detail="Project space not found.")
    return workspace_root


def _resolve_workspace_entry(session, rel_path: str = "") -> tuple[Path, Path, str]:
    workspace_root = _workspace_root_for_session(session)
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


def _build_structure_payload(path: Path, *, ctx: str, rel_path: str) -> Optional[dict[str, Any]]:
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
    if is_vibration_source and vibration_modes:
        viewer_source_mode = "url"
        viewer_source_url = f"/api/session/{ctx}/files/structure-vibration?path={quote(rel_path)}"
        viewer_source_file_type = "Xyz"
    elif is_vibration_source:
        viewer_source_mode = "url"
        viewer_source_url = f"/api/session/{ctx}/files/view?path={quote(rel_path)}"
        viewer_source_file_type = "VaspOutcar"
    elif is_trajectory_source:
        viewer_source_mode = "url"
        viewer_source_url = f"/api/session/{ctx}/files/structure-animation?path={quote(rel_path)}"
        viewer_source_file_type = "Xyz"
    else:
        viewer_source_mode = "url"
        viewer_source_url = f"/api/session/{ctx}/files/view?path={quote(rel_path)}"
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


def _structure_animation_response(*, session, rel_path: str) -> Response:
    _workspace_root, candidate, _normalized_path = _resolve_workspace_entry(session, rel_path)
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


def _structure_view_response(*, session, rel_path: str) -> FileResponse:
    _workspace_root, candidate, _normalized_path = _resolve_workspace_entry(session, rel_path)
    if not candidate.is_file():
        raise HTTPException(status_code=400, detail="Only files can be viewed.")
    upper_name = candidate.name.upper()
    if upper_name in {"OUTCAR", "XDATCAR", "POSCAR", "CONTCAR"} or candidate.suffix.lower() in {".outcar", ".xdatcar", ".poscar", ".contcar", ".vasp"}:
        media_type = "text/plain; charset=utf-8"
    else:
        media_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
    return FileResponse(candidate, media_type=media_type)


def _structure_vibration_response(*, session, rel_path: str, mode_index: int) -> Response:
    _workspace_root, candidate, _normalized_path = _resolve_workspace_entry(session, rel_path)
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


def _file_content_payload(*, ctx: str, session, rel_path: str) -> dict[str, Any]:
    workspace_root, candidate, normalized_path = _resolve_workspace_entry(session, rel_path)
    stat = candidate.stat()
    mime_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
    payload: dict[str, Any] = {
        "path": normalized_path,
        "name": candidate.name,
        "node_type": "directory" if candidate.is_dir() else "file",
        "size": int(stat.st_size),
        "modified_ts": float(stat.st_mtime),
        "mime_type": mime_type,
        "download_url": f"/api/session/{ctx}/files/download?path={quote(normalized_path)}",
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
        payload["structure"] = _build_structure_payload(candidate, ctx=ctx, rel_path=normalized_path)
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


def _runtime_snapshot(session) -> dict[str, Any]:
    reporter = session.reporter
    if reporter is None:
        return {
            "active": False,
            "run_name": "",
            "seq": 0,
            "live_state": {},
            "llm": {},
            "graph": {},
            "prompt": None,
            "usage_totals": {},
            "recent_events": [],
        }
    snapshot = reporter.get_snapshot()
    prompt = snapshot.get("prompt") if isinstance(snapshot.get("prompt"), dict) else None
    if prompt is not None:
        try:
            prompt = session._annotate_prompt_payload(session.get_selected_run_dir(), prompt)
        except Exception:
            prompt = prompt
    return {
        "active": True,
        "run_name": str(snapshot.get("run_name") or ""),
        "seq": int(snapshot.get("seq") or 0),
        "live_state": snapshot.get("live_state") if isinstance(snapshot.get("live_state"), dict) else {},
        "llm": snapshot.get("llm") if isinstance(snapshot.get("llm"), dict) else {},
        "graph": snapshot.get("graph") if isinstance(snapshot.get("graph"), dict) else {},
        "prompt": prompt,
        "usage_totals": snapshot.get("usage_totals") if isinstance(snapshot.get("usage_totals"), dict) else {},
        "recent_events": snapshot.get("recent_events") if isinstance(snapshot.get("recent_events"), list) else [],
    }


def _active_run_name(session, runtime: dict[str, Any] | None = None) -> str:
    runtime_dict = runtime if isinstance(runtime, dict) else {}
    runtime_run = str(runtime_dict.get("run_name") or "").strip()
    if runtime_run:
        return runtime_run
    info = getattr(session, "run_info", None)
    if isinstance(info, dict):
        run_id = str(info.get("run_id") or "").strip()
        if run_id:
            return run_id
    return ""


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
    }
    for key in count_keys:
        runtime_value = runtime.get(key)
        persisted_value = persisted.get(key)
        if isinstance(runtime_value, (int, float)) and isinstance(persisted_value, (int, float)):
            merged[key] = max(runtime_value, persisted_value)
        elif runtime_value is not None:
            merged[key] = runtime_value

    for key, value in runtime.items():
        if key in count_keys:
            continue
        if key not in merged or merged.get(key) in (None, "", [], {}):
            merged[key] = value
    return merged


def _stream_patch(session, runtime: dict[str, Any]) -> dict[str, Any]:
    run_name = str(runtime.get("run_name") or "").strip()
    active_run = _active_run_name(session, runtime)
    selected_run = run_name
    run_dir = None
    workspace_path = str(session.current_workspace_path() or "").strip()
    if run_name and workspace_path:
        resolved = session._resolve_run_dir_by_name(run_name, workspace=Path(workspace_path))
        if resolved is not None:
            run_dir = resolved
    if run_dir is None:
        run_dir = session.get_selected_run_dir()
        selected_run = run_dir.name if run_dir is not None else selected_run
    persisted_usage = session.read_usage_summary(run_dir) if run_dir is not None else {}
    usage_summary = _merge_usage_summary(runtime.get("usage_totals"), persisted_usage)
    return {
        "active_run": active_run,
        "selected_run": selected_run,
        "chat_messages": session.get_chat_messages(limit=40),
        "cards": _serialize_cards(session.list_run_cards()),
        "usage_summary": usage_summary,
        "proposal": session.read_proposal(run_dir),
        "todo_items": session.read_todo_items(run_dir),
        "result_text": session.read_result_text(run_dir),
    }


def _pick_selected_run(session, requested_run: str = "") -> str:
    selected = str(requested_run or "").strip()
    runs = session.list_runs()
    run_names = {value for _, value in runs}
    if selected and selected in run_names:
        session.select_run(selected)
        return selected
    current = session.get_selected_run_dir()
    current_name = current.name if current is not None else ""
    if current_name and current_name in run_names:
        return current_name
    if runs:
        fallback = runs[0][1]
        session.select_run(fallback)
        return fallback
    return ""


def _run_dir_for_name(session, run_name: str):
    selected = _pick_selected_run(session, run_name)
    if not selected:
        return None, ""
    return session.get_selected_run_dir(), selected


def _build_snapshot(*, registry: SessionRegistry, ctx: str, lane: str = "research", run_name: str = "") -> dict[str, Any]:
    session = registry.get_session(ctx)
    selected_run = _pick_selected_run(session, run_name)
    run_dir = session.get_selected_run_dir()
    runtime = _runtime_snapshot(session)
    active_run = _active_run_name(session, runtime)
    runtime_matches_selection = bool(selected_run) and selected_run == str(runtime.get("run_name") or "")

    if runtime_matches_selection:
        live_state = dict(runtime.get("live_state") or {})
        prompt = runtime.get("prompt")
        events = list(runtime.get("recent_events") or [])
        usage_summary = _merge_usage_summary(runtime.get("usage_totals"), session.read_usage_summary(run_dir))
        llm = dict(runtime.get("llm") or {})
        graph = dict(runtime.get("graph") or {})
    else:
        live_state = session.snapshot_live_state(run_dir)
        prompt = session.get_prompt() if str(session._load_task_state_status(run_dir) or "") == "awaiting_human_feedback" else None
        events = []
        usage_summary = session.read_usage_summary(run_dir)
        llm = live_state.get("llm") if isinstance(live_state.get("llm"), dict) else {}
        graph = {"node": str(live_state.get("current_node") or ""), "message_count": 0, "tool_calls": [], "text_preview": ""}

    cards = _serialize_cards(session.list_run_cards())
    prompt_payload = prompt if isinstance(prompt, dict) else None
    return {
        "ctx": ctx,
        "workspace_root": str(registry.default_project_space_root),
        "workspace_path": session.current_workspace_path(),
        "workspace_name": registry.project_space_name_for_session(session),
        "workspaces": _serialize_choices(session.list_workspaces()),
        "chat_sessions": _serialize_choices(session.list_chat_sessions()),
        "current_chat_session": session.current_chat_session_id(),
        "runs": _serialize_choices(session.list_runs()),
        "active_run": active_run,
        "selected_run": selected_run,
        "cards": cards,
        "run_status": str(session.run_status or "idle"),
        "run_status_text": session.run_status_text(),
        "run_info": dict(session.run_info or {}),
        "live_state": live_state,
        "llm": llm,
        "graph": graph,
        "prompt": prompt_payload,
        "events": events[-120:],
        "usage_summary": usage_summary,
        "proposal": session.read_proposal(run_dir),
        "todo_items": session.read_todo_items(run_dir),
        "result_text": session.read_result_text(run_dir),
        "chat_messages": session.get_chat_messages(limit=40),
        "entry_context_status": session.entry_context_status_text(lane=lane),
        "runtime": runtime,
        "can_submit_prompt": bool(runtime_matches_selection and prompt_payload),
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


def _build_details(*, registry: SessionRegistry, ctx: str, run_name: str) -> dict[str, Any]:
    session = registry.get_session(ctx)
    run_dir, selected_run = _run_dir_for_name(session, run_name)
    artifacts = session.read_artifacts()
    if hasattr(artifacts, "to_dict"):
        artifact_rows = artifacts.to_dict(orient="records")
    else:
        artifact_rows = list(artifacts or [])
    return {
        "selected_run": selected_run,
        "memory": session.read_memory_index(),
        "artifacts": artifact_rows,
        "proposal": session.read_proposal(run_dir),
        "task_state": session.read_task_state(run_dir),
        "trace_event": session.read_trace(run_dir, "event_trace.jsonl"),
        "trace_tool": session.read_trace(run_dir, "tool_trace.jsonl"),
        "trace_patch": session.read_trace(run_dir, "patch_trace.jsonl"),
    }


def _build_memory(*, registry: SessionRegistry, ctx: str, run_name: str = "") -> dict[str, Any]:
    session = registry.get_session(ctx)
    _run_dir, selected_run = _run_dir_for_name(session, run_name)
    return {
        "selected_run": selected_run,
        "memory": session.read_memory_index(),
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


def create_app(*, project_space_root: str) -> FastAPI:
    default_project_space_root = str(Path(project_space_root).expanduser().resolve())
    registry = SessionRegistry(default_project_space_root=default_project_space_root)
    app = FastAPI(title="CatMaster WebUI")
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

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
        state = registry.bootstrap(ctx=ctx, project_space=project_space, run=run)
        snapshot = _build_snapshot(registry=registry, ctx=state.ctx, lane=lane, run_name=state.run_name)
        snapshot["status_message"] = state.status
        snapshot["workspace_root"] = state.project_space_root
        return JSONResponse(snapshot)

    @app.get("/api/session/{ctx}/snapshot")
    def _session_snapshot(ctx: str, lane: str = "research", run: str = ""):
        return JSONResponse(_build_snapshot(registry=registry, ctx=ctx, lane=lane, run_name=run))

    @app.get("/api/session/{ctx}/details")
    def _session_details(ctx: str, run: str = ""):
        return JSONResponse(_build_details(registry=registry, ctx=ctx, run_name=run))

    @app.get("/api/session/{ctx}/memory")
    def _session_memory(ctx: str, run: str = ""):
        return JSONResponse(_build_memory(registry=registry, ctx=ctx, run_name=run))

    @app.get("/api/session/{ctx}/files/tree")
    def _session_files_tree(ctx: str, path: str = ""):
        session = registry.get_session(ctx)
        workspace_root, directory, normalized_path = _resolve_workspace_entry(session, path)
        if not directory.is_dir():
            raise HTTPException(status_code=400, detail="Requested path is not a directory.")
        return JSONResponse(
            {
                "path": normalized_path,
                "workspace_path": str(workspace_root),
                "workspace_name": registry.project_space_name_for_session(session),
                "children": _list_directory_entries(directory, workspace_root=workspace_root),
            }
        )

    @app.get("/api/session/{ctx}/files/content")
    def _session_file_content(ctx: str, path: str):
        session = registry.get_session(ctx)
        return JSONResponse(_file_content_payload(ctx=ctx, session=session, rel_path=path))

    @app.get("/api/session/{ctx}/files/structure-animation")
    def _session_structure_animation(ctx: str, path: str):
        session = registry.get_session(ctx)
        return _structure_animation_response(session=session, rel_path=path)

    @app.get("/api/session/{ctx}/files/view")
    def _session_structure_view(ctx: str, path: str):
        session = registry.get_session(ctx)
        return _structure_view_response(session=session, rel_path=path)

    @app.get("/api/session/{ctx}/files/structure-vibration")
    def _session_structure_vibration(ctx: str, path: str, mode: int = -1):
        session = registry.get_session(ctx)
        return _structure_vibration_response(session=session, rel_path=path, mode_index=mode)

    @app.get("/api/session/{ctx}/files/download")
    def _session_file_download(ctx: str, path: str):
        session = registry.get_session(ctx)
        _workspace_root, candidate, _normalized_path = _resolve_workspace_entry(session, path)
        if not candidate.is_file():
            raise HTTPException(status_code=400, detail="Only files can be downloaded.")
        return FileResponse(candidate, filename=candidate.name)

    @app.post("/api/session/{ctx}/workspace/open")
    async def _workspace_open(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
        root_path = str(payload.get("root_path") or registry.default_project_space_root)
        session.set_workspace_root(root_path)
        ok, message = session.open_workspace_by_name(str(payload.get("workspace") or ""))
        snapshot = _build_snapshot(registry=registry, ctx=ctx, lane=str(payload.get("lane") or "research"))
        snapshot["ok"] = ok
        snapshot["status_message"] = message
        return JSONResponse(snapshot)

    @app.post("/api/session/{ctx}/workspace/refresh")
    async def _workspace_refresh(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
        ok, message, _choices = session.set_workspace_root(str(payload.get("root_path") or registry.default_project_space_root))
        snapshot = _build_snapshot(registry=registry, ctx=ctx, lane=str(payload.get("lane") or "research"))
        snapshot["ok"] = ok
        snapshot["status_message"] = message
        return JSONResponse(snapshot)

    @app.post("/api/session/{ctx}/workspace/create")
    async def _workspace_create(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
        root_path = str(payload.get("root_path") or registry.default_project_space_root)
        session.set_workspace_root(root_path)
        ok, message = session.create_workspace(str(payload.get("workspace") or ""))
        snapshot = _build_snapshot(registry=registry, ctx=ctx, lane=str(payload.get("lane") or "research"))
        snapshot["ok"] = ok
        snapshot["status_message"] = message
        return JSONResponse(snapshot)

    @app.post("/api/session/{ctx}/run/select")
    async def _run_select(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
        message = session.select_run(str(payload.get("run_name") or ""))
        snapshot = _build_snapshot(
            registry=registry,
            ctx=ctx,
            lane=str(payload.get("lane") or "research"),
            run_name=str(payload.get("run_name") or ""),
        )
        snapshot["status_message"] = message
        return JSONResponse(snapshot)

    @app.post("/api/session/{ctx}/chat/create")
    async def _chat_create(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
        session_id = session.create_chat_session()
        snapshot = _build_snapshot(registry=registry, ctx=ctx, lane=str(payload.get("lane") or "research"))
        snapshot = _apply_chat_session_view(snapshot)
        snapshot["status_message"] = f"Started new chat session: {session_id}"
        return JSONResponse(snapshot)

    @app.post("/api/session/{ctx}/chat/select")
    async def _chat_select(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
        session_id = session.select_chat_session(str(payload.get("session_id") or ""))
        snapshot = _build_snapshot(registry=registry, ctx=ctx, lane=str(payload.get("lane") or "research"))
        snapshot = _apply_chat_session_view(snapshot)
        snapshot["status_message"] = f"Switched to chat session: {session_id}" if session_id else "Chat session not found."
        return JSONResponse(snapshot)

    @app.post("/api/session/{ctx}/run/start")
    async def _run_start(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
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
        )
        snapshot = _build_snapshot(registry=registry, ctx=ctx, lane=str(payload.get("lane") or "research"))
        snapshot["status_message"] = message
        return JSONResponse(snapshot)

    @app.post("/api/session/{ctx}/run/interrupt")
    async def _run_interrupt(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
        message = session.request_interrupt_current_run(note=str(payload.get("note") or ""))
        snapshot = _build_snapshot(registry=registry, ctx=ctx, lane=str(payload.get("lane") or "research"))
        snapshot["status_message"] = message
        return JSONResponse(snapshot)

    @app.post("/api/session/{ctx}/prompt/respond")
    async def _prompt_respond(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
        message = session.submit_prompt(str(payload.get("prompt_id") or ""), str(payload.get("text") or ""))
        snapshot = _build_snapshot(
            registry=registry,
            ctx=ctx,
            lane=str(payload.get("lane") or "research"),
            run_name=str(payload.get("run_name") or ""),
        )
        snapshot["status_message"] = message
        return JSONResponse(snapshot)

    @app.get("/api/session/{ctx}/stream")
    async def _session_stream(ctx: str, request: Request, last_seq: str = "0"):
        async def _event_stream():
            seq = _coerce_int(last_seq, 0)
            while True:
                if await request.is_disconnected():
                    break
                session = registry.get_session(ctx)
                reporter = session.reporter
                if reporter is None:
                    yield ": keepalive\n\n"
                    await asyncio.sleep(1.0)
                    continue
                events, seq = await asyncio.to_thread(reporter.wait_for_events_since, seq, timeout_s=10.0)
                if not events:
                    yield ": keepalive\n\n"
                    continue
                runtime = _runtime_snapshot(session)
                for event in events:
                    envelope = {
                        "event": event,
                        "runtime": runtime,
                        "run_status": str(session.run_status or "idle"),
                        "run_status_text": session.run_status_text(),
                    }
                    envelope.update(_stream_patch(session, runtime))
                    yield f"id: {int(event.get('seq') or seq)}\ndata: {json.dumps(envelope, ensure_ascii=False)}\n\n"

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    return app


def launch(
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    project_space_root: Optional[str] = None,
    timeout_keep_alive: int = 0,
    timeout_graceful_shutdown: int = 0,
) -> None:
    if project_space_root is None:
        project_space_root = str(Path.cwd() / "project_space")
    app = create_app(project_space_root=project_space_root)
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
