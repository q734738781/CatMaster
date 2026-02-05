from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Optional

import pandas as pd

from catmaster.tools.base import resolve_view_path, system_root, workspace_root


def _safe_resolve(path: Path, *, view: str, must_exist: bool = False) -> Path:
    root = workspace_root() if view == "user" else system_root()
    root = root.resolve()
    resolved = path.expanduser().resolve()
    if not str(resolved).startswith(str(root)):
        raise ValueError(f"Path escapes {view} root: {resolved}")
    if view == "user":
        sys_root = system_root().resolve()
        if str(resolved).startswith(str(sys_root)):
            raise ValueError(f"Path under system root is not allowed in user view: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def resolve_path(path: str | Path, *, view: str, must_exist: bool = False) -> Path:
    p = Path(path)
    if p.is_absolute():
        return _safe_resolve(p, view=view, must_exist=must_exist)
    return resolve_view_path(str(p), view, must_exist=must_exist)


def read_text(path: str | Path, *, view: str, max_chars: Optional[int] = None) -> str:
    try:
        p = resolve_path(path, view=view, must_exist=True)
    except Exception as exc:
        return f"(unavailable) {exc}"
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as exc:
        return f"(failed to read) {exc}"
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars] + "\n... (truncated)"
    return text


def read_json_pretty(path: str | Path, *, view: str, max_chars: Optional[int] = None) -> str:
    raw = read_text(path, view=view, max_chars=max_chars)
    try:
        data = json.loads(raw)
    except Exception:
        return raw
    return json.dumps(data, ensure_ascii=False, indent=2)


def read_artifacts_csv(path: str | Path) -> pd.DataFrame:
    try:
        p = resolve_path(path, view="system", must_exist=True)
    except Exception:
        return pd.DataFrame(columns=["path", "description", "type", "updated_time"])
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame(columns=["path", "description", "type", "updated_time"])
    return df


def tail_jsonl(path: str | Path, max_lines: int = 200) -> str:
    try:
        p = resolve_path(path, view="system", must_exist=True)
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

