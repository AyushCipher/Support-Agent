"""Durable session state in SQLite: every step is checkpointed, so a crash or an approval pause resumes exactly."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field

from .types import Message

SCHEMA = """CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY, customer_id TEXT, status TEXT, state TEXT, updated_at REAL)"""


@dataclass
class Session:
    id: str
    customer_id: str
    customer_email: str = ""
    seed: int = 0
    messages: list[Message] = field(default_factory=list)
    role: str = ""
    status: str = "active"          # active | awaiting_approval | done | escalated_budget
    step: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    tainted: bool = False
    risk_signals: list[str] = field(default_factory=list)
    write_calls: int = 0
    turns: int = 0
    events: list[dict] = field(default_factory=list)
    pending: dict | None = None
    summary: str = ""
    evidence: dict = field(default_factory=lambda: {"refund_amounts": [], "refund_targets": [], "escalated": False, "cancelled": False})
    models_used: list[str] = field(default_factory=list)
    prompt_hashes: dict = field(default_factory=dict)
    prompt_version: str = ""
    vault: dict = field(default_factory=dict)                  # PII token vault (encrypt at rest in production)
    pending_clarification: dict | None = None
    dry_run: bool = False                                      # shadow sessions: write tools are simulated
    shadow_of: str = ""
    verifier_calls: int = 0                                    # independent pre-money verifier budget accounting (ADR 0013)
    verifier_cost_usd: float = 0.0
    retention: dict = field(default_factory=dict)              # set by `supportagent retention run` (e.g. {"vault_purged_at": ts})

    def to_json(self, **override) -> str:
        d = {k: v for k, v in self.__dict__.items() if k != "messages"}
        d.update(override)
        d["messages"] = [m.to_dict() for m in self.messages]
        return json.dumps(d)

    @staticmethod
    def from_json(s: str) -> "Session":
        d = json.loads(s)
        msgs = [Message.from_dict(m) for m in d.pop("messages")]
        return Session(**d, messages=msgs)


_UNSET = object()


class CheckpointStore:
    """`protector` (default: security.crypto.default_protector(), i.e. SA_VAULT_KEY) seals Session.customer_email at rest with the
    customer's data key (subject c-<customer_id>, AAD bound to the session id). Before, the vault was sealed but the profile email sat
    in plaintext in every checkpoint (security gap G1). The in-memory Session keeps the plaintext (egress restore and the output
    guard need it); only the persisted JSON is sealed. After crypto-shredding (`privacy delete`) a sealed email loads as ''."""

    def __init__(self, conn: sqlite3.Connection, protector=_UNSET):
        self.conn, self._lock = conn, threading.Lock()
        if protector is _UNSET:
            from ..security.crypto import default_protector
            protector = default_protector()                    # None in dev without a key; raises in production without one
        self.prot = protector
        with self._lock:
            conn.execute(SCHEMA)

    @staticmethod
    def _aad(s_id: str) -> str:
        return f"session_email|{s_id}"

    def _dump(self, s: Session) -> str:
        from ..security.crypto import PREFIX, subject_for
        if self.prot is None or not s.customer_email or s.customer_email.startswith(PREFIX):
            return s.to_json()
        return s.to_json(customer_email=self.prot.seal(subject_for(s.customer_id), s.customer_email, self._aad(s.id)))

    def save(self, s: Session) -> None:
        state = self._dump(s)
        with self._lock:
            self.conn.execute("INSERT INTO sessions VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status, state=excluded.state, updated_at=excluded.updated_at",
                              (s.id, s.customer_id, s.status, state, time.time()))

    def load(self, sid: str) -> Session:
        with self._lock:
            r = self.conn.execute("SELECT state FROM sessions WHERE id=?", (sid,)).fetchone()
        if r is None:
            raise KeyError(sid)
        s = Session.from_json(r[0])
        from ..security.crypto import PREFIX, KeyShredded, subject_for
        if s.customer_email.startswith(PREFIX):
            try:
                s.customer_email = self.prot.open(subject_for(s.customer_id), s.customer_email, self._aad(s.id)) if self.prot else ""
            except (KeyShredded, ValueError):                  # shredded key / wrong key / tampered: never surface ciphertext as an email
                s.customer_email = ""
        return s

    def list(self, status: str | None = None, limit: int = 100) -> list[dict]:
        q, a = "SELECT id, customer_id, status, updated_at FROM sessions", ()
        if status:
            q, a = q + " WHERE status=?", (status,)
        with self._lock:
            rows = self.conn.execute(q + " ORDER BY updated_at DESC LIMIT ?", a + (limit,)).fetchall()
        return [{"id": r[0], "customer_id": r[1], "status": r[2], "updated_at": r[3]} for r in rows]
