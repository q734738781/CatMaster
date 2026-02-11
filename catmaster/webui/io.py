from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from catmaster.tools.base import resolve_workspace_path, system_root, workspace_root

PathScope = Literal["files", "metadata"]


def _safe_resolve(path: Path, *, root: Path, must_exist: bool = False) -> Path:
    root = root.resolve()
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes {root}: {resolved}") from exc
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def resolve_path(
    path: str | Path,
    *,
    scope: PathScope = "metadata",
    project_space: Path | str | None = None,
    must_exist: bool = False,
) -> Path:
    p = Path(path)
    if scope == "files":
        if p.is_absolute():
            return _safe_resolve(p, root=workspace_root(project_space), must_exist=must_exist)
        return resolve_workspace_path(str(p), workspace=project_space, must_exist=must_exist)
    if scope == "metadata":
        root = system_root(project_space)
        if p.is_absolute():
            return _safe_resolve(p, root=root, must_exist=must_exist)
        return _safe_resolve(root / p, root=root, must_exist=must_exist)
    raise ValueError(f"Invalid scope: {scope}")


def read_text(
    path: str | Path,
    *,
    scope: PathScope = "metadata",
    project_space: Path | str | None = None,
    max_chars: Optional[int] = None,
) -> str:
    try:
        p = resolve_path(path, scope=scope, project_space=project_space, must_exist=True)
    except Exception as exc:
        return f"(unavailable) {exc}"
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as exc:
        return f"(failed to read) {exc}"
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars] + "\n... (truncated)"
    return text


def read_json_pretty(
    path: str | Path,
    *,
    scope: PathScope = "metadata",
    project_space: Path | str | None = None,
    max_chars: Optional[int] = None,
) -> str:
    raw = read_text(path, scope=scope, project_space=project_space, max_chars=max_chars)
    try:
        data = json.loads(raw)
    except Exception:
        return raw
    return json.dumps(data, ensure_ascii=False, indent=2)


def read_artifacts_csv(path: str | Path, *, project_space: Path | str | None = None) -> pd.DataFrame:
    try:
        p = resolve_path(path, scope="metadata", project_space=project_space, must_exist=True)
    except Exception:
        return pd.DataFrame(columns=["path", "description", "type", "updated_time"])
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame(columns=["path", "description", "type", "updated_time"])
    return df


def tail_jsonl(path: str | Path, *, project_space: Path | str | None = None, max_lines: int = 200) -> str:
    try:
        p = resolve_path(path, scope="metadata", project_space=project_space, must_exist=True)
    except Exception as exc:
        return f"(unavailable) {exc}"
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return f"(failed to read) {exc}"
    if not lines:
        return ""
    tail = lines[-max_lines:]
    return "\n".join(tail)
