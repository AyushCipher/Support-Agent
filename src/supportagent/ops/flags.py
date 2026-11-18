"""Operational controls: feature flags, kill switches, prompt rollout (canary), and shadow-mode comparison results.
All state is in SQLite and read on every use, so flipping a flag takes effect mid-session. Every change is audit-chained."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time

from ..security.audit import AuditLog

SCHEMA = [
    "CREATE TABLE IF NOT EXISTS flags(key TEXT PRIMARY KEY, value TEXT, updated_at REAL, updated_by TEXT)",
    """CREATE TABLE IF NOT EXISTS shadow_results(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, session TEXT, live_version TEXT,
       shadow_version TEXT, same_calls INTEGER, same_outcome INTEGER, live_calls TEXT, shadow_calls TEXT, live_reply TEXT,
       shadow_reply TEXT, shadow_cost REAL)""",
]


class FlagStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        for s in SCHEMA:
            conn.execute(s)

    def get(self, key: str, default=None):
        r = self.conn.execute("SELECT value FROM flags WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set(self, key: str, value, by: str = "system") -> None:
        self.conn.execute("INSERT INTO flags VALUES(?,?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at, updated_by=excluded.updated_by",
                          (key, json.dumps(value), time.time(), by))
        AuditLog(self.conn).append("flag_set", agent=f"human:{by}", detail={"key": key, "value": value})

    def all(self) -> dict:
        return {r[0]: json.loads(r[1]) for r in self.conn.execute("SELECT key,value FROM flags ORDER BY key")}

    # ---- kill switches -----------------------------------------------------------------------------
    def killed(self, scope: str = "global") -> bool:
        """scope: 'global' (everything off) or 'money' (money/cancel tools off; read-only help continues)."""
        return bool(self.get("kill_switch.global", False) or (scope != "global" and self.get(f"kill_switch.{scope}", False)))

    def set_kill(self, scope: str, on: bool, by: str = "operator") -> None:
        self.set(f"kill_switch.{scope}", bool(on), by)

    # ---- prompt rollout ------------------------------------------------------------------------------
    def assign_prompt(self, session_id: str, default_stable: str) -> str:
        """Sticky, deterministic canary assignment: same session id always lands in the same arm."""
        stable = self.get("prompt.stable", default_stable)
        canary, pct = self.get("prompt.canary_version"), int(self.get("prompt.canary_percent", 0) or 0)
        if canary and pct > 0 and int(hashlib.sha256(session_id.encode()).hexdigest()[:8], 16) % 100 < pct:
            return canary
        return stable

    # ---- shadow mode -----------------------------------------------------------------------------------
    def record_shadow(self, **r) -> None:
        self.conn.execute("INSERT INTO shadow_results(ts,session,live_version,shadow_version,same_calls,same_outcome,live_calls,shadow_calls,live_reply,shadow_reply,shadow_cost) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                          (time.time(), r["session"], r["live_version"], r["shadow_version"], int(r["same_calls"]), int(r["same_outcome"]),
                           json.dumps(r["live_calls"]), json.dumps(r["shadow_calls"]), r["live_reply"][:500], r["shadow_reply"][:500], r.get("shadow_cost", 0.0)))

    def shadow_report(self) -> dict:
        rows = self.conn.execute("SELECT same_calls, same_outcome, shadow_cost, live_version, shadow_version FROM shadow_results").fetchall()
        n = len(rows)
        return {"n": n, "tool_call_agreement": (sum(r[0] for r in rows) / n) if n else None, "outcome_agreement": (sum(r[1] for r in rows) / n) if n else None,
                "shadow_cost_usd": round(sum(r[2] for r in rows), 5), "versions": sorted({(r[3], r[4]) for r in rows})}
