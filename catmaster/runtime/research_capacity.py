"""Atomic task admission, alongside (not inside) DBOS's private tables.

DBOS owns execution, durable waits and recovery. Its queue API has one fixed
per-partition limit, so it cannot express a shared total with unequal cost caps.
This table supplies only that admission predicate; it never launches work.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ResearchPoolConfig:
    agent_pool_size: int = 16
    agent_task_concurrency: dict[str, int] = field(default_factory=lambda: {"low": 8, "medium": 4, "high": 2})
    control_concurrency: int = 4

    def __post_init__(self):
        if set(self.agent_task_concurrency) != {"low", "medium", "high"}:
            raise ValueError("agent_task_concurrency must contain low, medium and high")
        for value in [self.agent_pool_size, self.control_concurrency, *self.agent_task_concurrency.values()]:
            if type(value) is not int or value < 1:
                raise ValueError("Research concurrency limits must be positive integers")

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(**{**data, "agent_task_concurrency": {
            **cls().agent_task_concurrency, **data.get("agent_task_concurrency", {})}})


class ResearchCapacity:
    def __init__(self, path: Path, config: ResearchPoolConfig):
        self.path, self.config = path, config

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def setup(self):
        with self.connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS catmaster_research_admission (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT UNIQUE NOT NULL,
                workspace TEXT NOT NULL, thread_id TEXT NOT NULL,
                tier TEXT NOT NULL, epoch INTEGER NOT NULL, state TEXT NOT NULL)""")

    def _admit(self, conn):
        rows = conn.execute("SELECT * FROM catmaster_research_admission ORDER BY seq").fetchall()
        active = [r for r in rows if r["state"] == "active"]
        counts = {tier: sum(r["tier"] == tier for r in active) for tier in self.config.agent_task_concurrency}
        total, admitted = len(active), []
        for row in rows:
            if row["state"] != "waiting":
                continue
            tier = row["tier"]
            # FIFO among eligible tasks; a full high tier does not block low.
            if total >= self.config.agent_pool_size or counts[tier] >= self.config.agent_task_concurrency[tier]:
                continue
            conn.execute("UPDATE catmaster_research_admission SET state='active' WHERE run_id=?", (row["run_id"],))
            counts[tier] += 1
            total += 1
            admitted.append(row["run_id"])
        return admitted

    def acquire(self, packet, tier, epoch):
        if tier not in self.config.agent_task_concurrency:
            raise ValueError("Unknown research task cost: " + str(tier))
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            # A steer resumes the same research task after the old writer has
            # unwound. Transfer its reservation instead of admitting an extra
            # expensive route during the hand-off between turns.
            previous = conn.execute("""SELECT * FROM catmaster_research_admission
                WHERE workspace=? AND thread_id=? AND run_id<>?""",
                (packet["workspace"], packet["thread_id"], packet["run_id"])).fetchone()
            if previous:
                conn.execute("UPDATE catmaster_research_admission SET run_id=?, epoch=? WHERE run_id=?",
                             (packet["run_id"], epoch, previous["run_id"]))
            current = conn.execute("SELECT * FROM catmaster_research_admission WHERE run_id=?", (packet["run_id"],)).fetchone()
            if current and (current["epoch"] != epoch or current["tier"] != tier):
                # Called only after the graph has checkpointed a capacity pause.
                conn.execute("DELETE FROM catmaster_research_admission WHERE run_id=?", (packet["run_id"],))
            conn.execute("""INSERT OR IGNORE INTO catmaster_research_admission
                (run_id, workspace, thread_id, tier, epoch, state) VALUES (?, ?, ?, ?, ?, 'waiting')""",
                (packet["run_id"], packet["workspace"], packet["thread_id"], tier, epoch))
            admitted = self._admit(conn)
            row = conn.execute("SELECT state FROM catmaster_research_admission WHERE run_id=?", (packet["run_id"],)).fetchone()
            return row["state"] == "active", admitted

    def release(self, run_id):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM catmaster_research_admission WHERE run_id=?", (run_id,))
            return self._admit(conn)

    def rows(self):
        with self.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM catmaster_research_admission ORDER BY seq")]

    def snapshot(self):
        rows = self.rows()
        return {"agent_pool_size": self.config.agent_pool_size,
                "active": sum(r["state"] == "active" for r in rows),
                "tiers": {tier: {"limit": limit,
                    "active": sum(r["tier"] == tier and r["state"] == "active" for r in rows),
                    "waiting": sum(r["tier"] == tier and r["state"] == "waiting" for r in rows)}
                    for tier, limit in self.config.agent_task_concurrency.items()}}
