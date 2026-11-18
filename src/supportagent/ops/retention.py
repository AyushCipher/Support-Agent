"""Data retention scheduler: `supportagent retention run [--dry-run]` (ADR 0016).

Per data class, a policy says how long to keep data and what to do after (purge, or archive-then-purge):

  transcripts   message bodies of FINISHED sessions are replaced by a marker (roles, tool names, events, cost, evidence kept)
  vault         the per-session PII token vault is dropped, so tokens can no longer be turned back into real values
  checkpoints   the whole session row of a FINISHED session is deleted (optionally archived first, WITHOUT its vault)
  traces        spans older than the cutoff leave runs/traces.jsonl (optionally archived to .jsonl.gz)
  audit         ARCHIVE-ONLY: expired chain entries are exported with a manifest; live rows are never deleted here (see below)
  approvals     decided approvals (pending ones are never touched)
  request_dedup API idempotency-key rows

Safety rules: sessions that are not finished (active / awaiting_approval / awaiting_user) are never touched; sessions or customers
on the legal hold list (flag `retention.legal_hold`) are skipped; every non-dry-run action leaves a tombstone in the tamper-evident
audit chain (class, count, cutoff, id sample + digest, archive path) - purged data disappears, the fact of the purge does not.
`--dry-run` computes exactly what would happen and writes nothing (no archive files, no tombstones, no deletes).

AUDIT LIMITATION: the audit log is a hash chain from genesis; deleting its head breaks `verify()`. Truncating it safely needs a
chain re-anchoring primitive in security/audit.py (owned by the security work-stream). Until then audit retention is archive-only and
`action: purge` for audit is refused with an explanation. INTEGRATION POINT below: `AuditLog.prune_before` is used if it exists.

CRYPTO-SHREDDING / DELETE-ON-REQUEST is owned by security/privacy.py (`erase_customer`, `supportagent privacy delete`). This module
only schedules AGE-based retention and deliberately does NOT destroy keys: data keys are per customer (security/crypto.subject_for) and
shared by all of that customer's sessions and memory, so shredding a key because ONE old session expired would make the customer's
newer data unreadable. Age-based vault retention therefore deletes the (sealed) vault ciphertext only. It reuses the security module's
`redact_session_state` whitelist for archived checkpoints, so an archive can never carry more personal data than an erased session.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..harness.checkpoint import SCHEMA as CHECKPOINT_SCHEMA
from ..security.audit import AuditLog
from ..security.privacy import redact_session_state
from .dedup import TABLE as DEDUP_TABLE

TERMINAL = ("done", "killed", "escalated_budget", "degraded_handoff")
ACTIONS = ("purge", "archive")
PURGED = "[purged by retention policy]"
DAY = 86400.0

DEFAULT_POLICY: dict[str, dict] = {
    "transcripts": {"days": 90, "action": "purge"},
    "vault": {"days": 30, "action": "purge"},
    "checkpoints": {"days": 180, "action": "archive"},
    "traces": {"days": 30, "action": "archive"},
    "audit": {"days": 365, "action": "archive"},
    "approvals": {"days": 365, "action": "purge"},
    "request_dedup": {"days": 1, "action": "purge"},
}
RUNS_SCHEMA = "CREATE TABLE IF NOT EXISTS retention_runs(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, dry_run INTEGER, report TEXT)"


class PolicyError(ValueError):
    pass


def load_policy(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, dict]:
    """Defaults < JSON file (or $SA_RETENTION_POLICY path) < overrides. Unknown classes / bad values are rejected, never ignored."""
    pol = {k: dict(v) for k, v in DEFAULT_POLICY.items()}
    path = path or os.getenv("SA_RETENTION_POLICY")
    for src in ((json.loads(Path(path).read_text(encoding="utf-8")) if path else {}), overrides or {}):
        for cls, cfg in src.items():
            if cls not in pol:
                raise PolicyError(f"unknown data class {cls!r}; known: {sorted(pol)}")
            pol[cls].update(cfg)
    for cls, cfg in pol.items():
        if not isinstance(cfg["days"], (int, float)) or isinstance(cfg["days"], bool) or cfg["days"] < 0:
            raise PolicyError(f"{cls}.days must be a non-negative number")
        if cfg["action"] not in ACTIONS:
            raise PolicyError(f"{cls}.action must be one of {ACTIONS}")
    if pol["audit"]["action"] == "purge":
        raise PolicyError("audit.action=purge is refused: the hash chain cannot be truncated safely yet (archive-only; see ops/retention.py)")
    return pol


@dataclass
class ClassReport:
    cls: str
    cutoff_ts: float
    eligible: int = 0
    acted: int = 0
    skipped_hold: int = 0
    skipped_nonterminal: int = 0
    skipped_changed: int = 0
    archive: str | None = None
    ids: list | None = None
    note: str = ""
    exported_upto_seq: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "ids" and not (k == "exported_upto_seq" and not v)}


class RetentionJob:
    def __init__(self, conn: sqlite3.Connection, policy: dict | None = None, runs_dir: str | Path = "runs", archive_dir: str | Path | None = None,
                 clock: Callable[[], float] = time.time):
        self.conn, self.policy, self.runs_dir = conn, policy or load_policy(), Path(runs_dir)
        self.archive_dir = Path(archive_dir) if archive_dir else self.runs_dir / "archive"
        self.clock = clock
        conn.execute(CHECKPOINT_SCHEMA)
        AuditLog(conn)                                                                  # ensures the audit_chain table exists
        conn.execute(RUNS_SCHEMA)

    # ------------------------------------------------------------------ helpers
    def _cutoff(self, cls: str, now: float) -> float:
        return now - self.policy[cls]["days"] * DAY

    def _holds(self) -> set[str]:
        try:
            r = self.conn.execute("SELECT value FROM flags WHERE key='retention.legal_hold'").fetchone()
            return set(json.loads(r[0])) if r else set()
        except sqlite3.Error:
            return set()

    def _archive_writer(self, cls: str, now: float, dry: bool):
        """Returns (write(obj), close()->path|None). Writes nothing in dry-run."""
        if dry or self.policy[cls]["action"] != "archive":
            return (lambda o: None), (lambda: None)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        path = self.archive_dir / f"{cls}-{datetime.fromtimestamp(now, timezone.utc):%Y%m%dT%H%M%SZ}.jsonl.gz"
        f = gzip.open(path, "wt", encoding="utf-8")
        state = {"n": 0}

        def write(o):
            state["n"] += 1
            f.write(json.dumps(o, default=str) + "\n")

        def close():
            f.close()
            if state["n"] == 0:
                path.unlink(missing_ok=True)
                return None
            return str(path)
        return write, close

    # ------------------------------------------------------------------ run
    def run(self, dry_run: bool = False) -> dict:
        now = self.clock()
        reports = [self._sessions(now, dry_run), self._traces(now, dry_run), self._audit(now, dry_run),
                   self._table("approvals", "approvals", "status != 'pending' AND COALESCE(decided_at, created_at) < ?", now, dry_run),
                   self._table("request_dedup", DEDUP_TABLE, "created_at < ?", now, dry_run)]
        flat = [r for group in reports for r in (group if isinstance(group, list) else [group])]
        out = {"dry_run": dry_run, "ran_at": datetime.fromtimestamp(now, timezone.utc).isoformat(), "classes": {r.cls: r.to_dict() for r in flat}}
        if not dry_run:
            for r in flat:
                if r.acted:
                    self._tombstone(r, now)
            self.conn.execute("INSERT INTO retention_runs(ts, dry_run, report) VALUES(?,?,?)", (now, 0, json.dumps(out)))
        return out

    def _tombstone(self, r: ClassReport, now: float) -> None:
        ids = r.ids or []
        AuditLog(self.conn).append("retention_purge", agent="system:retention", detail={
            "class": r.cls, "count": r.acted, "cutoff": datetime.fromtimestamp(r.cutoff_ts, timezone.utc).isoformat(),
            "policy_days": self.policy[r.cls]["days"], "action": self.policy[r.cls]["action"], "archive": r.archive,
            "ids_sample": ids[:200], "ids_sha256": hashlib.sha256(json.dumps(sorted(map(str, ids))).encode()).hexdigest()})

    # ------------------------------------------------------------------ sessions: transcripts / vault / checkpoints
    def _sessions(self, now: float, dry: bool) -> list[ClassReport]:
        t, v, c = (ClassReport(k, self._cutoff(k, now), ids=[]) for k in ("transcripts", "vault", "checkpoints"))
        holds = self._holds()
        write, close = self._archive_writer("checkpoints", now, dry)
        newest_cutoff = max(t.cutoff_ts, v.cutoff_ts, c.cutoff_ts)       # a row is a candidate if ANY class has expired for it
        rows = self.conn.execute("SELECT id, customer_id, status, state, updated_at FROM sessions WHERE updated_at < ?", (newest_cutoff,)).fetchall()
        for sid, cust, status, state, upd in (tuple(r) for r in rows):
            expired = {"transcripts": upd < t.cutoff_ts, "vault": upd < v.cutoff_ts, "checkpoints": upd < c.cutoff_ts}
            if status not in TERMINAL:
                for rep in (t, v, c):
                    rep.skipped_nonterminal += expired[rep.cls]
                continue
            if sid in holds or cust in holds:
                for rep in (t, v, c):
                    rep.skipped_hold += expired[rep.cls]
                continue
            d = json.loads(state)
            if expired["checkpoints"]:
                c.eligible += 1
                if not dry:
                    write({**json.loads(redact_session_state(state)), "archived_by": "retention"})   # security's whitelist: no PII, no vault
                    cur = self.conn.execute("DELETE FROM sessions WHERE id=? AND updated_at=?", (sid, upd))
                    if cur.rowcount:
                        c.acted += 1
                        c.ids.append(sid)                                               # type: ignore[union-attr]
                    else:
                        c.skipped_changed += 1
                continue
            changed = False
            if expired["vault"] and d.get("vault") and "vault_purged_at" not in d.get("retention", {}):
                v.eligible += 1
                if not dry:
                    d["vault"], d["customer_email"] = {}, ""                              # the profile email is PII of the same class
                    d.setdefault("retention", {})["vault_purged_at"] = now
                    v.ids.append(sid)                                                   # type: ignore[union-attr]
                    changed = True
            if expired["transcripts"] and "transcript_purged_at" not in d.get("retention", {}):
                t.eligible += 1
                if not dry:
                    for m in d.get("messages", []):
                        m["content"], m["tool_calls"] = PURGED, [{**tc, "arguments": {}} for tc in m.get("tool_calls", [])]
                    d["summary"], d["pending"], d["pending_clarification"] = "", None, None
                    d.setdefault("retention", {})["transcript_purged_at"] = now
                    t.ids.append(sid)                                                   # type: ignore[union-attr]
                    changed = True
            if changed:
                cur = self.conn.execute("UPDATE sessions SET state=? WHERE id=? AND updated_at=?", (json.dumps(d), sid, upd))   # keep updated_at: age is unchanged
                if not cur.rowcount:                                                    # the session moved on since we read it: back off, retry next run
                    t.ids, v.ids = [i for i in t.ids if i != sid], [i for i in v.ids if i != sid]
                    t.skipped_changed += 1
        t.acted, v.acted = len(t.ids), len(v.ids)
        c.archive = close()
        c.note = "archive is redacted with security.privacy.redact_session_state" if self.policy["checkpoints"]["action"] == "archive" else ""
        return [t, v, c]

    # ------------------------------------------------------------------ traces
    def _traces(self, now: float, dry: bool) -> ClassReport:
        r = ClassReport("traces", self._cutoff("traces", now), ids=[])
        path = self.runs_dir / "traces.jsonl"
        if not path.exists():
            r.note = "no traces file"
            return r
        size = path.stat().st_size
        keep: list[bytes] = []
        write, close = self._archive_writer("traces", now, dry)
        with path.open("rb") as f:
            data = f.read(size)                                                         # snapshot: spans appended meanwhile are copied below
        for raw in data.splitlines(keepends=True):
            try:
                span = json.loads(raw)
                ts = float(span.get("end") or span.get("start") or 0)
            except ValueError:
                keep.append(raw)                                                        # never destroy lines we cannot parse
                continue
            if 0 < ts < r.cutoff_ts:
                r.eligible += 1
                write(span)
            else:
                keep.append(raw)
        if dry or not r.eligible:
            close()
            return r
        r.archive = close()
        tmp = path.with_suffix(".jsonl.retention-tmp")
        with path.open("rb") as f:
            f.seek(size)
            tail = f.read()                                                             # RESIDUAL RACE: a span written after this read and before os.replace is lost
        tmp.write_bytes(b"".join(keep) + tail)
        os.replace(tmp, path)
        r.acted = r.eligible
        r.ids = []
        r.note = "rewrites traces.jsonl; run when no writer is mid-append (see ADR 0016)"
        return r

    # ------------------------------------------------------------------ audit (archive-only)
    def _audit(self, now: float, dry: bool) -> ClassReport:
        r = ClassReport("audit", self._cutoff("audit", now), ids=[])
        if hasattr(AuditLog, "prune_before") and self.policy["audit"]["action"] == "purge":   # INTEGRATION POINT (security agent)
            r.note = "AuditLog.prune_before exists but is not wired yet"
        last = self.conn.execute("SELECT report FROM retention_runs WHERE dry_run=0 ORDER BY id DESC").fetchall()
        done_upto = 0
        for (rep,) in last:
            done_upto = max(done_upto, (json.loads(rep).get("classes", {}).get("audit", {}) or {}).get("exported_upto_seq", 0))
        rows = self.conn.execute("SELECT seq,ts,agent,session,customer,action,detail,prev_hash,hash FROM audit_chain WHERE ts < ? AND seq > ? ORDER BY seq",
                                 (r.cutoff_ts, done_upto)).fetchall()
        r.eligible = len(rows)
        if dry or not rows:
            return r
        write, close = self._archive_writer("audit", now, dry)
        for row in rows:
            write(dict(zip(("seq", "ts", "agent", "session", "customer", "action", "detail", "prev_hash", "hash"), tuple(row))))
        r.archive = close()
        r.acted = 0                                                                     # exported, NOT deleted: no tombstone needed, nothing disappeared
        r.note = f"exported {len(rows)} entries (seq {rows[0][0]}..{rows[-1][0]}) with prev_hash links; live chain untouched"
        r.exported_upto_seq = rows[-1][0]
        return r

    # ------------------------------------------------------------------ simple tables
    def _table(self, cls: str, table: str, where: str, now: float, dry: bool) -> ClassReport:
        r = ClassReport(cls, self._cutoff(cls, now), ids=[])
        try:
            r.eligible = self.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", (r.cutoff_ts,)).fetchone()[0]
        except sqlite3.Error:
            r.note = f"table {table} absent"
            return r
        if dry or not r.eligible:
            return r
        write, close = self._archive_writer(cls, now, dry)
        if self.policy[cls]["action"] == "archive":
            cols = [d[1] for d in self.conn.execute(f"PRAGMA table_info({table})").fetchall()]
            for row in self.conn.execute(f"SELECT * FROM {table} WHERE {where}", (r.cutoff_ts,)).fetchall():
                write(dict(zip(cols, tuple(row))))
        r.archive = close()
        r.acted = self.conn.execute(f"DELETE FROM {table} WHERE {where}", (r.cutoff_ts,)).rowcount
        return r


def report_text(rep: dict) -> str:
    lines = [f"retention run at {rep['ran_at']}  ({'DRY RUN - nothing changed' if rep['dry_run'] else 'APPLIED'})"]
    for cls, r in rep["classes"].items():
        extra = "".join(f" {k}={r[k]}" for k in ("skipped_hold", "skipped_nonterminal", "skipped_changed") if r.get(k))
        lines.append(f"  {cls:<14} eligible={r['eligible']:<5} acted={r['acted']:<5}{extra}" + (f" archive={r['archive']}" if r.get("archive") else "")
                     + (f"  # {r['note']}" if r.get("note") else ""))
    return "\n".join(lines)
