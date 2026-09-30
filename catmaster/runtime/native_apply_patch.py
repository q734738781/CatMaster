"""Codex-compatible freeform ``apply_patch`` execution for project files."""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Literal

from langchain_openai import custom_tool

from .apply_diff import apply_diff

APPLY_PATCH_LARK_GRAMMAR = r"""start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change?

filename: /(.+)/
add_line: "+" /(.*)/ LF -> line

change_move: "*** Move to: " filename LF
change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF

%import common.LF
"""

APPLY_PATCH_TOOL_FORMAT = {
    "type": "grammar",
    "syntax": "lark",
    "definition": APPLY_PATCH_LARK_GRAMMAR,
}

_BEGIN_PATCH = "*** Begin Patch"
_END_PATCH = "*** End Patch"
_ADD_FILE = "*** Add File: "
_DELETE_FILE = "*** Delete File: "
_UPDATE_FILE = "*** Update File: "
_MOVE_TO = "*** Move to: "
_HUNK_PREFIXES = (_ADD_FILE, _DELETE_FILE, _UPDATE_FILE)
_ROOT_LOCKS: dict[str, threading.RLock] = {}
_ROOT_LOCKS_GUARD = threading.Lock()


class NativeApplyPatchError(ValueError):
    """A freeform patch could not be safely applied."""


@dataclass(frozen=True)
class PatchHunk:
    kind: Literal["add", "delete", "update"]
    path: str
    body: str = ""
    move_to: str = ""


def parse_apply_patch(patch: str) -> list[PatchHunk]:
    """Parse the Codex V4A patch envelope into ordered file hunks."""

    if not isinstance(patch, str) or not patch.strip():
        raise NativeApplyPatchError("patch input is empty")
    normalized = patch.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = normalized.split("\n")
    if not lines or lines[0].strip() != _BEGIN_PATCH:
        raise NativeApplyPatchError("the first line must be '*** Begin Patch'")
    if lines[-1].strip() != _END_PATCH:
        raise NativeApplyPatchError("the last line must be '*** End Patch'")

    hunks: list[PatchHunk] = []
    index = 1
    last = len(lines) - 1
    while index < last:
        line = lines[index]
        if line.startswith(_ADD_FILE):
            path = _header_path(line, _ADD_FILE, index)
            body_lines, index = _collect_hunk_body(lines, index + 1, last)
            if not body_lines:
                raise NativeApplyPatchError(f"Add File hunk for {path!r} is empty")
            invalid = next((value for value in body_lines if not value.startswith("+")), "")
            if invalid:
                raise NativeApplyPatchError(
                    f"invalid Add File line for {path!r}: {invalid!r}"
                )
            body = "\n".join(value[1:] for value in body_lines) + "\n"
            hunks.append(PatchHunk(kind="add", path=path, body=body))
            continue
        if line.startswith(_DELETE_FILE):
            path = _header_path(line, _DELETE_FILE, index)
            hunks.append(PatchHunk(kind="delete", path=path))
            index += 1
            continue
        if line.startswith(_UPDATE_FILE):
            path = _header_path(line, _UPDATE_FILE, index)
            index += 1
            move_to = ""
            if index < last and lines[index].startswith(_MOVE_TO):
                move_to = _header_path(lines[index], _MOVE_TO, index)
                index += 1
            body_lines, index = _collect_hunk_body(lines, index, last)
            if not body_lines and not move_to:
                raise NativeApplyPatchError(f"Update File hunk for {path!r} is empty")
            # The published Codex grammar declares the update body optional,
            # and GPT-5.6 emits move-only hunks in live Codex OAuth traffic.
            # Codex CLI's current Rust parser is stricter, so accept the grammar
            # and live-model form here as a compatibility extension.
            body = "\n".join(body_lines)
            hunks.append(PatchHunk(kind="update", path=path, body=body, move_to=move_to))
            continue
        if not line.strip():
            raise NativeApplyPatchError(f"unexpected blank line at patch line {index + 1}")
        raise NativeApplyPatchError(
            f"invalid patch hunk header at line {index + 1}: {line!r}"
        )

    if not hunks:
        raise NativeApplyPatchError("patch does not contain any file hunks")
    return hunks


def _header_path(line: str, prefix: str, index: int) -> str:
    path = line[len(prefix) :]
    if not path:
        raise NativeApplyPatchError(f"path is missing at patch line {index + 1}")
    return path


def _collect_hunk_body(
    lines: list[str],
    start: int,
    end: int,
) -> tuple[list[str], int]:
    index = start
    while index < end and not lines[index].startswith(_HUNK_PREFIXES):
        index += 1
    return lines[start:index], index


class NativeApplyPatchExecutor:
    """Apply a complete Codex patch below one physical files root."""

    def __init__(self, *, files_root: Path) -> None:
        self.files_root = Path(files_root).expanduser().resolve()
        self.files_root.mkdir(parents=True, exist_ok=True)

    def execute(self, patch: str) -> str:
        hunks = parse_apply_patch(patch)
        summaries: list[str] = []
        with self._exclusive_root_lock():
            for hunk in hunks:
                target, display_path = self._resolve_path(hunk.path)
                if hunk.kind == "add":
                    self._write_file(target, hunk.body, overwrite=True)
                    summaries.append(f"A {display_path}")
                    continue
                if hunk.kind == "delete":
                    self._delete_existing(target)
                    summaries.append(f"D {display_path}")
                    continue

                updated, mode = self._derive_update(target, hunk.body)
                if hunk.move_to:
                    destination, destination_display = self._resolve_path(hunk.move_to)
                    if destination == target:
                        self._write_file(target, updated, overwrite=True, mode=mode)
                        summaries.append(f"M {display_path}")
                    else:
                        self._write_file(destination, updated, overwrite=True, mode=mode)
                        self._delete_existing(target)
                        summaries.append(f"R {display_path} -> {destination_display}")
                else:
                    self._write_file(target, updated, overwrite=True, mode=mode)
                    summaries.append(f"M {display_path}")
        return "Done!\n" + "\n".join(summaries)

    def _resolve_path(self, raw_path: str) -> tuple[Path, str]:
        if not raw_path:
            raise NativeApplyPatchError("path is required")
        if "\x00" in raw_path or "\\" in raw_path:
            raise NativeApplyPatchError("path must be a POSIX workspace path")

        virtual_path = raw_path[1:] if raw_path.startswith("/") else raw_path
        pure = PurePosixPath(virtual_path)
        parts = pure.parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise NativeApplyPatchError("path traversal and empty path segments are not allowed")
        if parts[0] == "memories":
            raise NativeApplyPatchError(
                "/memories is a DeepAgents store route; use edit_file for persistent memory"
            )

        target = self.files_root.joinpath(*parts)
        try:
            target.parent.resolve(strict=False).relative_to(self.files_root)
        except ValueError as exc:
            raise NativeApplyPatchError("path resolves outside the workspace files root") from exc

        current = self.files_root
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise NativeApplyPatchError("symbolic-link paths are not supported")
        return target, "/" + pure.as_posix()

    def _derive_update(self, target: Path, diff: str) -> tuple[str, int]:
        if not target.exists():
            raise NativeApplyPatchError("Update File target does not exist")
        if not target.is_file():
            raise NativeApplyPatchError("Update File target is not a regular file")
        before = target.stat()
        try:
            current = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise NativeApplyPatchError("Update File target is not UTF-8 text") from exc
        if diff:
            try:
                updated = apply_diff(current, diff)
            except ValueError as exc:
                raise NativeApplyPatchError(str(exc)) from exc
        else:
            updated = current
        current_stat = target.stat()
        signature = (before.st_ino, before.st_size, before.st_mtime_ns)
        current_signature = (
            current_stat.st_ino,
            current_stat.st_size,
            current_stat.st_mtime_ns,
        )
        if signature != current_signature:
            raise NativeApplyPatchError(
                "Update File target changed while the patch was being applied"
            )
        return updated, stat.S_IMODE(before.st_mode)

    @staticmethod
    def _write_temp(parent: Path, name: str, content: str, mode: int) -> Path:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{name}.",
            suffix=".patch",
            dir=parent,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content.encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temp_path, mode)
            return temp_path
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def _write_file(
        self,
        target: Path,
        content: str,
        *,
        overwrite: bool,
        mode: int = 0o644,
    ) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not target.is_file():
            raise NativeApplyPatchError("patch target is not a regular file")
        if target.exists() and not overwrite:
            raise NativeApplyPatchError("patch target already exists")
        temp_path = self._write_temp(target.parent, target.name, content, mode)
        try:
            os.replace(temp_path, target)
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _delete_existing(target: Path) -> None:
        if not target.exists():
            raise NativeApplyPatchError("Delete File target does not exist")
        if not target.is_file():
            raise NativeApplyPatchError("Delete File target is not a regular file")
        target.unlink()

    @contextmanager
    def _exclusive_root_lock(self) -> Iterator[None]:
        root_key = str(self.files_root)
        with _ROOT_LOCKS_GUARD:
            thread_lock = _ROOT_LOCKS.setdefault(root_key, threading.RLock())
        digest = hashlib.sha256(root_key.encode("utf-8")).hexdigest()[:24]
        lock_path = Path(tempfile.gettempdir()) / f"catmaster-apply-patch-{digest}.lock"
        with thread_lock:
            with lock_path.open("a+b") as lock_stream:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def build_native_apply_patch_tool(*, files_root: Path) -> Any:
    """Build the LangChain custom tool used by Codex OAuth models."""

    executor = NativeApplyPatchExecutor(files_root=files_root)

    @custom_tool(format=dict(APPLY_PATCH_TOOL_FORMAT))
    def apply_patch(patch: str) -> str:
        """Edit workspace files with one Codex V4A patch. Pass the raw patch, not JSON."""

        try:
            return executor.execute(patch)
        except Exception as exc:
            return f"Error applying patch: {exc}"

    return apply_patch


__all__ = [
    "APPLY_PATCH_LARK_GRAMMAR",
    "APPLY_PATCH_TOOL_FORMAT",
    "NativeApplyPatchError",
    "NativeApplyPatchExecutor",
    "PatchHunk",
    "build_native_apply_patch_tool",
    "parse_apply_patch",
]
