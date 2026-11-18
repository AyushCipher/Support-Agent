"""Ticketing MCP server: create tickets, escalate to a human, check status."""
from __future__ import annotations

import os
import uuid

from ...config import NOW
from ...shop import db
from ..protocol import MCPServer, ToolError

ACTING = {"type": "string", "description": "Injected by the harness from the authenticated session."}
IDEM = {"type": "string", "description": "Injected by the harness; makes retries and crash-resumes safe (exactly one ticket per key)."}


def make_server(db_path: str | None = None, allow_anon: bool | None = None, tokens: dict | None = None, jwt_secret: str | None = None) -> MCPServer:
    path = db_path or os.environ["SHOP_DB"]
    anon = allow_anon if allow_anon is not None else os.getenv("SA_ALLOW_ANON", "0") == "1"
    srv = MCPServer("tickets", tokens=tokens, jwt_secret=jwt_secret)

    def who(args):
        a = args.get("acting_customer_id")
        if not a and not anon:
            raise ToolError("unauthenticated", "acting_customer_id is required")
        return a or "anon"

    def _create(args, escalated: int, subject: str, body: str, priority: str, code: str | None):
        cid = who(args)
        key = args.get("idempotency_key")
        tid = f"TCK-{uuid.uuid4().hex[:6].upper()}"
        conn = db.connect(path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("CREATE TABLE IF NOT EXISTS ticket_idempotency(key TEXT PRIMARY KEY, ticket_id TEXT)")
            prior = conn.execute("SELECT ticket_id FROM ticket_idempotency WHERE key=?", (key,)).fetchone() if key else None
            if prior:                                              # retry / crash-resume of the same logical call: replay, don't duplicate
                conn.execute("COMMIT")
                return {"ticket_id": prior[0], "status": "open", "escalated": bool(escalated), "sla": "1 business day", "replayed": True}
            conn.execute("INSERT INTO tickets VALUES(?,?,?,?,?,?,?,?,?)", (tid, cid, subject[:200], body[:2000], priority, "open", escalated, code, NOW.isoformat()))
            if key:
                conn.execute("INSERT INTO ticket_idempotency VALUES(?,?)", (key, tid))
            i = args.get("_identity") or {}
            db.audit(conn, cid, "ticket_created", tid, agent=i.get("sub", ""), session=i.get("sid", ""))
            conn.execute("COMMIT")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        return {"ticket_id": tid, "status": "open", "escalated": bool(escalated), "sla": "1 business day"}

    @srv.tool("create_ticket", "Create a support ticket for follow-up.",
              {"type": "object", "properties": {"subject": {"type": "string", "maxLength": 200}, "body": {"type": "string", "maxLength": 2000},
                                                "priority": {"type": "string", "enum": ["low", "normal", "high"]}, "idempotency_key": IDEM,
                                                "acting_customer_id": ACTING},
               "required": ["subject", "body", "acting_customer_id"]},
              scope="tickets:write", risk="write", hidden=("acting_customer_id", "idempotency_key"))
    def create_ticket(args):
        return _create(args, 0, args["subject"], args["body"], args.get("priority", "normal"), None)

    @srv.tool("escalate_to_human", "Hand the case to a human specialist (exceptions, abuse review, anything outside policy).",
              {"type": "object", "properties": {"summary": {"type": "string", "maxLength": 1500},
                                                "reason_code": {"type": "string", "enum": ["exception_request", "abuse_review", "requires_human", "customer_request", "budget_exceeded", "other"]},
                                                "idempotency_key": IDEM, "acting_customer_id": ACTING},
               "required": ["summary", "reason_code", "acting_customer_id"]},
              scope="tickets:write", risk="write", hidden=("acting_customer_id", "idempotency_key"))
    def escalate_to_human(args):
        return _create(args, 1, f"[ESCALATION] {args['reason_code']}", args["summary"], "high", args["reason_code"])

    @srv.tool("get_ticket_status", "Check a ticket's status.",
              {"type": "object", "properties": {"ticket_id": {"type": "string", "pattern": r"^TCK-[A-Z0-9]{6}$"}, "acting_customer_id": ACTING},
               "required": ["ticket_id", "acting_customer_id"]}, scope="tickets:read", hidden=("acting_customer_id",))
    def get_ticket_status(args):
        cid = who(args)
        conn = db.connect(path)
        try:
            r = conn.execute("SELECT id,status,priority,escalated,customer_id FROM tickets WHERE id=?", (args["ticket_id"],)).fetchone()
            if r is None or (not anon and r["customer_id"] != cid):
                raise ToolError("ticket_not_found", "No such ticket on this account.")
            return {k: r[k] for k in ("id", "status", "priority", "escalated")}
        finally:
            conn.close()

    srv.prompt("escalation_summary", "Template for escalation hand-off notes", [{"name": "case", "required": True}],
               lambda a: f"Summarize for a human: who, what was asked, what policy says, what was tried. Case: {a.get('case','')}")
    return srv


def main() -> None:
    make_server().run_stdio()


if __name__ == "__main__":
    main()
