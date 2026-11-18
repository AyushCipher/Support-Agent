"""Human-in-the-loop approvals persisted in SQLite. Async (UI) or sync (auto-decider for evals/tests) modes."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Callable

SCHEMA = """CREATE TABLE IF NOT EXISTS approvals(
  id TEXT PRIMARY KEY, session_id TEXT, tool TEXT, args TEXT, reason TEXT, summary TEXT,
  status TEXT, reviewer TEXT, note TEXT, created_at REAL, decided_at REAL)"""


class ApprovalService:
    def __init__(self, conn: sqlite3.Connection, auto_decider: Callable[[dict], tuple[bool, str]] | None = None):
        self.conn, self.auto = conn, auto_decider
        self._lock = threading.Lock()
        with self._lock:
            self.conn.execute(SCHEMA)

    def request(self, session_id: str, tool: str, args: dict, reason: str, summary: str) -> dict:
        aid = f"APR-{uuid.uuid4().hex[:8]}"
        with self._lock:
            self.conn.execute("INSERT INTO approvals VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                              (aid, session_id, tool, json.dumps(args), reason, summary, "pending", None, None, time.time(), None))
        rec = self.get(aid)
        if self.auto:                                     # sync mode: reviewer simulated in-process
            ok, note = self.auto(rec)
            rec = self.decide(aid, ok, "auto-reviewer", note)
        return rec

    def get(self, aid: str) -> dict:
        with self._lock:
            r = self.conn.execute("SELECT * FROM approvals WHERE id=?", (aid,)).fetchone()
        d = dict(r) if hasattr(r, "keys") else dict(zip([c[0] for c in self.conn.execute("SELECT * FROM approvals LIMIT 0").description], r))
        d["args"] = json.loads(d["args"])
        return d

    def decide(self, aid: str, approved: bool, reviewer: str, note: str = "") -> dict:
        with self._lock:
            self.conn.execute("UPDATE approvals SET status=?, reviewer=?, note=?, decided_at=? WHERE id=? AND status='pending'",
                              ("approved" if approved else "denied", reviewer, note, time.time(), aid))
        return self.get(aid)

    def pending(self) -> list[dict]:
        with self._lock:
            ids = [r[0] for r in self.conn.execute("SELECT id FROM approvals WHERE status='pending' ORDER BY created_at").fetchall()]
        return [self.get(i) for i in ids]

    def all(self, limit: int = 50) -> list[dict]:
        with self._lock:
            ids = [r[0] for r in self.conn.execute("SELECT id FROM approvals ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()]
        return [self.get(i) for i in ids]
