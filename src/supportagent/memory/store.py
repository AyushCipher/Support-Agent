"""Long-term customer memory (SQLite).

Write policy (memory-poisoning defense): memories are written ONLY by the harness from structured, verified tool
outcomes (refund issued, ticket escalated, subscription cancelled) - never from free-text the customer typed and never
from untrusted tool output. Reads are rendered into the system prompt as system-recorded history.

Encryption at rest + erasure (ADR-0011): with SA_VAULT_KEY (or a `protector`) the value column is AES-256-GCM ciphertext under
the customer's data key; `erase()` deletes the rows and `crypto-shreds` the key, so copies of the old database are unreadable.
Production mode (SA_ENV=production) without a key fails closed.
"""
from __future__ import annotations

import sqlite3
import threading
import time

from ..security.crypto import KeyShredded, Protector, default_protector, subject_for

SCHEMA = "CREATE TABLE IF NOT EXISTS customer_memory(customer_id TEXT, key TEXT, value TEXT, source TEXT, ts REAL)"


class CustomerMemory:
    def __init__(self, conn: sqlite3.Connection, protector: Protector | None = None):
        self.conn, self._lock = conn, threading.Lock()
        self.prot = protector if protector is not None else default_protector()
        with self._lock:
            conn.execute(SCHEMA)

    def remember(self, customer_id: str, key: str, value: str, source: str) -> None:
        if not source.startswith("tool:"):
            raise PermissionError("memory writes must originate from verified tool outcomes")
        with self._lock:
            self.conn.execute("INSERT INTO customer_memory VALUES(?,?,?,?,?)", (customer_id, key, self._seal(customer_id, key, value[:200]), source, time.time()))

    def recall(self, customer_id: str, limit: int = 8) -> list[str]:
        with self._lock:
            rows = self.conn.execute("SELECT key, value FROM customer_memory WHERE customer_id=? ORDER BY ts DESC LIMIT ?", (customer_id, limit)).fetchall()
        out = []
        for k, v in rows:
            try:
                out.append(f"{k}: {self._open(customer_id, k, v)}")
            except KeyShredded:
                continue                                       # erased subject: nothing to recall
        return out

    def _seal(self, customer_id: str, key: str, value: str) -> str:
        return self.prot.seal(subject_for(customer_id), value, f"memory|{customer_id}|{key}") if self.prot else value

    def _open(self, customer_id: str, key: str, value: str) -> str:
        return self.prot.open(subject_for(customer_id), value, f"memory|{customer_id}|{key}") if self.prot else value

    def erase(self, customer_id: str) -> int:
        """Delete every memory row for the customer and shred their memory key. Returns rows deleted."""
        with self._lock:
            n = self.conn.execute("DELETE FROM customer_memory WHERE customer_id=?", (customer_id,)).rowcount
        if self.prot:
            self.prot.shred(subject_for(customer_id))
        return n
