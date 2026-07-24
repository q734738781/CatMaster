from __future__ import annotations

import fcntl
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .engine import HypothesisEngine
from .models import HypothesisEngineState


HYPOTHESIS_ENGINE_DIR = "research_hypothesis_engines"


def safe_thread_id(thread_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(thread_id or "").strip())
    normalized = normalized.strip("._") or "default"
    return normalized[:80]


def engine_relpath(thread_id: str) -> str:
    return f"{HYPOTHESIS_ENGINE_DIR}/{safe_thread_id(thread_id)}/state.json"


def engine_path(files_root: str | Path, thread_id: str) -> Path:
    return Path(files_root) / engine_relpath(thread_id)


def engine_lock_path(files_root: str | Path, thread_id: str) -> Path:
    return engine_path(files_root, thread_id).with_name(".state.lock")


@contextmanager
def campaign_lock(files_root: str | Path, thread_id: str) -> Iterator[None]:
    """Serialize campaign mutations while keeping state reads atomic."""

    lock_path = engine_lock_path(files_root, thread_id)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_engine(
    files_root: str | Path,
    thread_id: str,
) -> HypothesisEngine:
    path = engine_path(files_root, thread_id)
    if not path.exists():
        raise FileNotFoundError(f"hypothesis engine state does not exist: {engine_relpath(thread_id)}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = HypothesisEngineState.model_validate(payload)
    return HypothesisEngine(state)


def save_engine(files_root: str | Path, thread_id: str, engine: HypothesisEngine) -> Path:
    path = engine_path(files_root, thread_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(engine.state.model_dump_json(indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


__all__ = [
    "HYPOTHESIS_ENGINE_DIR",
    "campaign_lock",
    "engine_lock_path",
    "engine_path",
    "engine_relpath",
    "load_engine",
    "safe_thread_id",
    "save_engine",
]
