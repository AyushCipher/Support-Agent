"""API request de-duplication (idempotent POST /api/chat).

A client that times out and retries must not run the turn twice (double refund attempts, double tickets, double spend). The client
sends `Idempotency-Key: <uuid>` (header) or `request_id` (body). The first request with a key runs; a retry with the same key and
the same payload gets the stored response; the same key with a DIFFERENT payload is rejected (422) - it is a client bug, and replaying
the old answer for new content would be wrong. Persisted in SQLite so it survives a process restart.
Write tools additionally carry harness-derived idempotency keys end-to-end (stable_hash(session, tool, args)), so even a duplicate
that slips past this layer cannot move money twice.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from typing import Callable

TABLE = "api_requests"
SCHEMA = f"""CREATE TABLE IF NOT EXISTS {TABLE}(
  key TEXT PRIMARY KEY, fingerprint TEXT, status TEXT, response TEXT, created_at REAL)"""


def fingerprint(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:32]


class RequestDedup:
    def __init__(self, conn: sqlite3.Connection, inflight_ttl_s: float = 120.0, clock: Callable[[], float] = time.time):
        self.conn, self.inflight_ttl_s, self.clock = conn, inflight_ttl_s, clock
        self._lock = threading.Lock()
        with self._lock:
            conn.execute(SCHEMA)
        self.stats = {"new": 0, "replayed": 0, "conflict": 0, "in_flight": 0}

    def begin(self, key: str, fp: str) -> tuple[str, dict | None]:
        """-> ('new', None) | ('replay', stored_response) | ('conflict', None) | ('in_flight', None)"""
        now = self.clock()
        with self._lock:
            row = self.conn.execute(f"SELECT fingerprint, status, response, created_at FROM {TABLE} WHERE key=?", (key,)).fetchone()
            if row is not None:
                rfp, status, resp, created = tuple(row)
                if rfp != fp:
                    self.stats["conflict"] += 1
                    return "conflict", None
                if status == "done":
                    self.stats["replayed"] += 1
                    return "replay", json.loads(resp)
                if now - created < self.inflight_ttl_s:
                    self.stats["in_flight"] += 1
                    return "in_flight", None
                # a stale in-flight row means the previous worker died mid-turn: let this retry take over (tool-level idempotency keys
                # make re-running safe for writes the dead worker already completed)
            self.conn.execute(f"INSERT INTO {TABLE} VALUES(?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET fingerprint=excluded.fingerprint, "
                              "status=excluded.status, response=NULL, created_at=excluded.created_at", (key, fp, "in_flight", None, now))
            self.stats["new"] += 1
            return "new", None

    def complete(self, key: str, response: dict) -> None:
        with self._lock:
            self.conn.execute(f"UPDATE {TABLE} SET status='done', response=? WHERE key=?", (json.dumps(response, default=str), key))

    def fail(self, key: str) -> None:
        """The turn raised: forget the key so the client's retry runs again instead of replaying nothing."""
        with self._lock:
            self.conn.execute(f"DELETE FROM {TABLE} WHERE key=? AND status='in_flight'", (key,))
