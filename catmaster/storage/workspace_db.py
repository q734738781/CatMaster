from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import os
from pathlib import Path
from typing import Iterator
import sqlite3

from catmaster.tools.base import system_root


WORKSPACE_DATABASE_NAME = "workspace.sqlite"
_LOCAL_WAL_FILESYSTEMS = {
    "apfs",
    "btrfs",
    "ext2",
    "ext3",
    "ext4",
    "f2fs",
    "overlay",
    "tmpfs",
    "xfs",
    "zfs",
}
_NETWORK_FILESYSTEMS = {
    "9p",
    "afs",
    "ceph",
    "cifs",
    "fuse.sshfs",
    "glusterfs",
    "lustre",
    "nfs",
    "nfs4",
    "smb3",
    "smbfs",
}


def workspace_database_path(workspace: Path | str) -> Path:
    """Return the internal database shared only at the connection layer."""

    path = system_root(workspace) / WORKSPACE_DATABASE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _unescape_mount_path(value: str) -> str:
    return (
        str(value or "")
        .replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


@lru_cache(maxsize=128)
def filesystem_type(path_text: str) -> str:
    """Return the longest matching Linux mount type, or an empty string."""

    candidate = Path(path_text).expanduser().resolve()
    mounts_path = Path("/proc/mounts")
    if not mounts_path.is_file():
        return ""
    best_length = -1
    best_type = ""
    try:
        rows = mounts_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for row in rows:
        fields = row.split()
        if len(fields) < 3:
            continue
        mount_path = Path(_unescape_mount_path(fields[1]))
        try:
            candidate.relative_to(mount_path)
        except ValueError:
            continue
        length = len(str(mount_path))
        if length > best_length:
            best_length = length
            best_type = str(fields[2]).strip().lower()
    return best_type


def workspace_journal_mode(workspace: Path | str) -> str:
    """Choose WAL only for a verified local filesystem.

    Network and unknown mounts use the rollback journal. Deployments that have
    independently verified their storage can opt in explicitly with
    ``CATMASTER_WORKSPACE_SQLITE_JOURNAL_MODE=WAL``.
    """

    configured = str(os.getenv("CATMASTER_WORKSPACE_SQLITE_JOURNAL_MODE") or "").strip().upper()
    if configured:
        if configured not in {"WAL", "DELETE"}:
            raise ValueError("CATMASTER_WORKSPACE_SQLITE_JOURNAL_MODE must be WAL or DELETE.")
        return configured
    fs_type = filesystem_type(str(Path(workspace).expanduser().resolve()))
    if fs_type in _NETWORK_FILESYSTEMS:
        return "DELETE"
    if fs_type in _LOCAL_WAL_FILESYSTEMS:
        return "WAL"
    return "DELETE"


@contextmanager
def connect_workspace_db(workspace: Path | str) -> Iterator[sqlite3.Connection]:
    """Open a short-lived connection with filesystem-safe journal settings."""

    connection = sqlite3.connect(
        str(workspace_database_path(workspace)),
        timeout=30.0,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute(f"PRAGMA journal_mode={workspace_journal_mode(workspace)}")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                component TEXT PRIMARY KEY,
                version INTEGER NOT NULL
            )
            """
        )
        yield connection
        connection.commit()
    finally:
        connection.close()


def ensure_workspace_ui_events(connection: sqlite3.Connection) -> None:
    """Create the bounded workspace outbox shared by graph and thread streams."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS ui_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            thread_id TEXT NOT NULL DEFAULT '',
            graph_id TEXT NOT NULL DEFAULT '',
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS ui_events_thread_cursor
            ON ui_events(thread_id, event_id);
        CREATE INDEX IF NOT EXISTS ui_events_graph_cursor
            ON ui_events(graph_id, event_id);
        """
    )
