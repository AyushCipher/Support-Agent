"""Delete-on-request (GDPR art. 17 style) that preserves audit-chain integrity (ADR-0011).

`erase_customer()` does, in this order:
  1. crypto-shred: destroys the data keys for the customer (memory key, and the vault key recorded in each of their sessions).
     From this instant every ciphertext copy - old DB backups, WAL files, replicas - is permanently unreadable, even to
     someone holding the master key. (No-op for plaintext dev data; rows are still deleted below.)
  2. redact retained transcripts in the checkpoint store: message text and tool-call arguments are replaced and every
     non-whitelisted session field (email, vault, summary, evidence, events, pending state) is blanked; the session row
     stays so counts/status/cost reporting survive.
  3. delete long-term memory rows; redact approval records (args/summary/note) for those sessions, deny any still pending.
  4. append a `privacy_erasure` tombstone to the audit chain(s). The chain is NEVER rewritten: history stays verifiable
     and shows *that* an erasure happened, when, by whom and how much was removed - containing no personal data.

What it deliberately does not touch: shop business records (orders, refunds, tickets) that law requires you to retain, and the
pseudonymous customer id on audit rows (needed to prove the erasure and for fraud/finance audit). Both are listed in
docs/compliance.md so the DPO can decide the retention basis.
"""
from __future__ import annotations

import json
import sqlite3
import time

from ..memory.store import CustomerMemory
from .audit import AuditLog
from .crypto import PREFIX, Protector, subject_for

ERASED = "[erased]"


# Fields that carry no personal data (ids, counters, flags, model/prompt metadata). Everything else in a checkpoint is blanked by
# type - a whitelist, so a field added to Session later is erased by default rather than leaking by default.
KEEP = {"id", "customer_id", "seed", "role", "status", "step", "tokens", "cost_usd", "tainted", "write_calls", "turns", "models_used",
        "prompt_hashes", "prompt_version", "dry_run", "shadow_of", "verifier_calls", "verifier_cost_usd"}


def redact_session_state(state_json: str) -> str:
    d = json.loads(state_json)
    for k, v in list(d.items()):
        if k in KEEP or k == "messages":
            continue
        d[k] = "" if isinstance(v, str) else [] if isinstance(v, list) else {} if isinstance(v, dict) else None if v is None else v
    d["pending"], d["pending_clarification"] = None, None
    if str(d.get("status", "")).startswith("awaiting"):          # nobody will answer an erased customer's pending question/approval
        d["status"] = "done"
    d["evidence"] = {"refund_amounts": [], "refund_targets": [], "escalated": False, "cancelled": False}
    for m in d.get("messages", []):
        m["content"] = ERASED if m.get("content") else ""
        for tc in m.get("tool_calls", []):
            tc["arguments"] = {}
    return json.dumps(d)


def erase_customer(state: sqlite3.Connection, customer_id: str, audit: AuditLog | None = None, shop_audit: AuditLog | None = None,
                   protector: Protector | None = None, requested_by: str = "cli", reason: str = "data subject request",
                   extra_emails: tuple[str, ...] = (), traces_path=None) -> dict:
    if not customer_id:
        raise ValueError("customer_id required")
    try:
        state.execute("PRAGMA secure_delete=ON")
    except sqlite3.Error:                                              # pragma: no cover
        pass
    rows = state.execute("SELECT id, state FROM sessions WHERE customer_id=?", (customer_id,)).fetchall() \
        if _has_table(state, "sessions") else []
    sids = [r[0] for r in rows]

    # -- 1. crypto-shred
    subjects = {subject_for(customer_id)} | {subject_for(email=e) for e in extra_emails}
    for _, sj in rows:
        d = json.loads(sj)
        if d.get("vault", {}).get("subj"):
            subjects.add(d["vault"]["subj"])
        if d.get("customer_email") and not str(d["customer_email"]).startswith(PREFIX):   # legacy plaintext; sealed ones use c-<id>
            subjects.add(subject_for(email=d["customer_email"]))
    shredded = sum(1 for s in subjects if protector and protector.shred(s))

    # -- 1b. local trace file (gap G3): spans of the customer's sessions are deleted, not archived
    spans = 0
    if traces_path is not None:
        from ..obs.tracing import erase_spans
        spans = erase_spans(traces_path, sids, customer_id)

    # -- 2/3. redact + delete, atomically with the tombstone when the audit log shares this connection
    own_tx = not state.in_transaction
    if own_tx:
        state.execute("BEGIN IMMEDIATE")
    try:
        for sid, sj in rows:
            red = redact_session_state(sj)
            state.execute("UPDATE sessions SET state=?, status=?, updated_at=? WHERE id=?", (red, json.loads(red)["status"], time.time(), sid))
        approvals = 0
        if sids and _has_table(state, "approvals"):
            q = ",".join("?" * len(sids))
            approvals = state.execute(
                f"UPDATE approvals SET args='{{}}', summary=?, note='', status=CASE WHEN status='pending' THEN 'denied' ELSE status END "
                f"WHERE session_id IN ({q})", (ERASED, *sids)).rowcount
        mem = CustomerMemory(state, protector).erase(customer_id) if protector else _delete_memory_plain(state, customer_id)
        report = {"customer": customer_id, "sessions_redacted": len(sids), "memory_rows_deleted": mem,
                  "approvals_redacted": approvals, "keys_shredded": shredded, "trace_spans_deleted": spans}
        detail = {**report, "reason": reason, "retained": ["business records", "pseudonymous customer id on audit rows"]}
        if audit is not None:
            report["tombstone_hash"] = audit.append("privacy_erasure", agent=f"human:{requested_by}", customer=customer_id, detail=detail)
        if own_tx:
            state.execute("COMMIT")
    except Exception:
        if own_tx and state.in_transaction:
            state.execute("ROLLBACK")
        raise
    if shop_audit is not None:
        shop_audit.append("privacy_erasure", agent=f"human:{requested_by}", customer=customer_id, detail=detail)
    return report


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _delete_memory_plain(conn: sqlite3.Connection, customer_id: str) -> int:
    if not _has_table(conn, "customer_memory"):
        return 0
    return conn.execute("DELETE FROM customer_memory WHERE customer_id=?", (customer_id,)).rowcount
