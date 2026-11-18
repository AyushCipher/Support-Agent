"""Tamper-evident audit log: an append-only hash chain in SQLite.

Each entry's hash = SHA-256(prev_hash | canonical JSON of the entry). Editing or deleting any historical row breaks every
later link, which `verify()` detects and localizes. Deleting the *tail* is only detectable against an externally anchored
head hash: `supportagent audit anchor` writes signed heads to an append-only anchor log (security/anchors.py) and
`audit verify --anchors` compares the chain against them; ship the anchors to write-once storage (docs/compliance.md, ADR-0011).
Entries carry agent identity (`sub` from the JWT) and session id, so every action is attributable.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time

GENESIS = "0" * 64
SCHEMA = """CREATE TABLE IF NOT EXISTS audit_chain(
  seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, agent TEXT, session TEXT, customer TEXT, action TEXT, detail TEXT,
  prev_hash TEXT, hash TEXT)"""


def _digest(prev: str, ts: float, agent: str, session: str, customer: str, action: str, detail: str) -> str:
    body = json.dumps([ts, agent, session, customer, action, detail], separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(f"{prev}|{body}".encode()).hexdigest()


class AuditLog:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        conn.execute(SCHEMA)

    def append(self, action: str, agent: str = "", session: str = "", customer: str = "", detail: dict | None = None, ts: float | None = None) -> str:
        own_tx = not self.conn.in_transaction
        if own_tx:
            self.conn.execute("BEGIN IMMEDIATE")                  # serialize writers across processes (orders + tickets servers)
        try:
            row = self.conn.execute("SELECT hash FROM audit_chain ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row[0] if row else GENESIS
            ts = ts if ts is not None else round(time.time(), 6)
            d = json.dumps(detail or {}, sort_keys=True, default=str)
            h = _digest(prev, ts, agent, session, customer, action, d)
            self.conn.execute("INSERT INTO audit_chain(ts,agent,session,customer,action,detail,prev_hash,hash) VALUES(?,?,?,?,?,?,?,?)",
                              (ts, agent, session, customer, action, d, prev, h))
            if own_tx:
                self.conn.execute("COMMIT")
            return h
        except Exception:
            if own_tx and self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise

    def head(self) -> str:
        row = self.conn.execute("SELECT hash FROM audit_chain ORDER BY seq DESC LIMIT 1").fetchone()
        return row[0] if row else GENESIS

    def tip(self) -> tuple[int, int, str]:
        """(last seq, entry count, head hash) - what an anchor commits to."""
        row = self.conn.execute("SELECT seq, hash FROM audit_chain ORDER BY seq DESC LIMIT 1").fetchone()
        n = self.conn.execute("SELECT COUNT(*) FROM audit_chain").fetchone()[0]
        return (row[0], n, row[1]) if row else (0, 0, GENESIS)

    def hash_at(self, seq: int) -> str | None:
        row = self.conn.execute("SELECT hash FROM audit_chain WHERE seq=?", (seq,)).fetchone()
        return row[0] if row else None

    def verify(self) -> tuple[bool, int | None, int]:
        """Returns (ok, first_bad_seq, entries_checked)."""
        prev, n = GENESIS, 0
        for r in self.conn.execute("SELECT seq,ts,agent,session,customer,action,detail,prev_hash,hash FROM audit_chain ORDER BY seq"):
            seq, ts, agent, session, customer, action, detail, ph, h = tuple(r)
            n += 1
            if ph != prev or _digest(prev, ts, agent, session, customer, action, detail) != h:
                return False, seq, n
            prev = h
        return True, None, n

    def entries(self, session: str | None = None, limit: int = 1000) -> list[dict]:
        q, a = "SELECT seq,ts,agent,session,customer,action,detail,hash FROM audit_chain", ()
        if session:
            q, a = q + " WHERE session=?", (session,)
        rows = self.conn.execute(q + " ORDER BY seq LIMIT ?", a + (limit,)).fetchall()
        return [{"seq": r[0], "ts": r[1], "agent": r[2], "session": r[3], "customer": r[4], "action": r[5], "detail": json.loads(r[6]), "hash": r[7][:12]} for r in rows]

    def replay(self, session: str) -> str:
        """Human-readable timeline of everything that happened in one session, with agent attribution."""
        lines = []
        for e in self.entries(session):
            d = {k: v for k, v in e["detail"].items() if v not in (None, "", [])}
            lines.append(f"#{e['seq']:>4}  {e['agent'] or '-':<18} {e['action']:<20} {json.dumps(d, default=str)[:160]}  [{e['hash']}]")
        return "\n".join(lines) or f"(no audit entries for {session})"
