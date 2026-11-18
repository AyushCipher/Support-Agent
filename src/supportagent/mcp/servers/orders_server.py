"""Orders/billing MCP server backed by the shop SQLite DB.

Enforces HARD INVARIANTS server-side (ownership, positive amount, remaining balance, idempotency).
Business-policy decisions are exposed via check_* tools and verified again by the harness ToolGuard.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import timedelta

from ...config import NOW
from ...shop import db
from ...shop.policy import evaluate_refund, evaluate_subscription
from ..protocol import MCPServer, ToolError

ORDER_ID = {"type": "string", "pattern": r"^ORD-\d{4}$"}
SUB_ID = {"type": "string", "pattern": r"^SUB-\d{4}$"}
ACTING = {"type": "string", "description": "Injected by the harness from the authenticated session."}
IDEM = {"type": "string", "description": "Injected by the harness; makes retries and resumes safe."}
REASONS = {"type": "string", "enum": ["changed_mind", "defective", "wrong_item", "other"]}
PUBLIC_ORDER_FIELDS = ("id", "status", "placed_at", "delivered_at", "est_delivery", "item", "category", "total_cents",
                       "shipping_cents", "refunded_cents", "notes", "ship_to")


def make_server(db_path: str | None = None, allow_anon: bool | None = None, tokens: dict | None = None, jwt_secret: str | None = None) -> MCPServer:
    path = db_path or os.environ["SHOP_DB"]
    anon = allow_anon if allow_anon is not None else os.getenv("SA_ALLOW_ANON", "0") == "1"
    srv = MCPServer("orders", tokens=tokens, jwt_secret=jwt_secret)

    def ident(args: dict) -> dict:
        i = args.get("_identity") or {}
        return {"agent": i.get("sub", ""), "session": i.get("sid", "")}

    def actor(args: dict) -> str | None:
        a = args.get("acting_customer_id")
        if not a and not anon:
            raise ToolError("unauthenticated", "acting_customer_id is required")
        return a

    def get_order_row(conn, oid: str, who: str | None) -> dict:
        row = conn.execute("SELECT o.*, c.email AS customer_email FROM orders o JOIN customers c ON c.id=o.customer_id WHERE o.id=?", (oid,)).fetchone()
        if row is None or (who and row["customer_id"] != who):
            raise ToolError("order_not_found", f"No order {oid} on this account.")   # same error for missing/not-owned
        return dict(row)

    def refunds_90d(conn, cid: str) -> int:
        cutoff = (NOW - timedelta(days=90)).isoformat()
        return conn.execute("SELECT COUNT(*) FROM refunds WHERE customer_id=? AND created_at>=?", (cid, cutoff)).fetchone()[0]

    @srv.tool("get_order", "Look up one order (status, dates, amounts, item, notes, ship-to).",
              {"type": "object", "properties": {"order_id": ORDER_ID, "acting_customer_id": ACTING}, "required": ["order_id", "acting_customer_id"]},
              scope="orders:read", hidden=("acting_customer_id",))
    def get_order(args):
        conn = db.connect(path)
        try:
            row = get_order_row(conn, args["order_id"], actor(args))
            out = {k: row[k] for k in PUBLIC_ORDER_FIELDS}
            out["customer_email"] = row["customer_email"]
            return out
        finally:
            conn.close()

    @srv.tool("list_orders", "List the customer's orders, newest first.",
              {"type": "object", "properties": {"acting_customer_id": ACTING}, "required": ["acting_customer_id"]},
              scope="orders:read", hidden=("acting_customer_id",))
    def list_orders(args):
        who = actor(args)
        conn = db.connect(path)
        try:
            rows = conn.execute("SELECT id,status,item,placed_at,total_cents FROM orders WHERE customer_id=? ORDER BY placed_at DESC", (who,)).fetchall()
            return {"orders": [dict(r) for r in rows]}
        finally:
            conn.close()

    @srv.tool("get_subscriptions", "List the customer's subscriptions.",
              {"type": "object", "properties": {"acting_customer_id": ACTING}, "required": ["acting_customer_id"]},
              scope="orders:read", hidden=("acting_customer_id",))
    def get_subscriptions(args):
        who = actor(args)
        conn = db.connect(path)
        try:
            rows = conn.execute("SELECT id,plan,status,interval,price_cents,renewed_at,period_end FROM subscriptions WHERE customer_id=?", (who,)).fetchall()
            return {"subscriptions": [dict(r) for r in rows]}
        finally:
            conn.close()

    @srv.tool("check_refund_eligibility", "Apply refund policy to an order. Returns eligible, max_amount_cents, reason_code, requires_human.",
              {"type": "object", "properties": {"order_id": ORDER_ID, "reason": REASONS, "acting_customer_id": ACTING},
               "required": ["order_id", "reason", "acting_customer_id"]},
              scope="orders:read", hidden=("acting_customer_id",))
    def check_refund_eligibility(args):
        conn = db.connect(path)
        try:
            row = get_order_row(conn, args["order_id"], actor(args))
            return evaluate_refund(row, refunds_90d(conn, row["customer_id"]), args.get("reason", "changed_mind"))
        finally:
            conn.close()

    @srv.tool("issue_refund", "Issue a refund on an order. HIGH RISK: moves money.",
              {"type": "object", "properties": {"order_id": ORDER_ID, "amount_cents": {"type": "integer", "minimum": 1, "maximum": 1000000},
                                                "reason": REASONS, "idempotency_key": IDEM, "acting_customer_id": ACTING},
               "required": ["order_id", "amount_cents", "reason", "acting_customer_id"]},
              scope="refunds:issue", risk="high", hidden=("acting_customer_id", "idempotency_key"))
    def issue_refund(args):
        who = actor(args)
        key = args.get("idempotency_key") or f"auto-{uuid.uuid4().hex}"
        conn = db.connect(path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            prior = conn.execute("SELECT * FROM refunds WHERE idempotency_key=?", (key,)).fetchone()
            if prior:
                conn.execute("COMMIT")
                return {"status": "refunded", "refund_id": prior["id"], "amount_cents": prior["amount_cents"], "order_id": prior["target_id"], "replayed": True}
            row = get_order_row(conn, args["order_id"], who)
            remaining = row["total_cents"] - row["refunded_cents"]
            amt = int(args["amount_cents"])
            if amt <= 0 or amt > remaining:
                conn.execute("ROLLBACK")
                raise ToolError("amount_exceeds_balance", f"Refundable balance is {remaining} cents.", max_amount_cents=remaining)
            rid = f"RFN-{uuid.uuid4().hex[:8]}"
            conn.execute("INSERT INTO refunds VALUES(?,?,?,?,?,?,?,?)", (rid, row["customer_id"], "order", row["id"], amt, args.get("reason", "other"), key, NOW.isoformat()))
            conn.execute("UPDATE orders SET refunded_cents=refunded_cents+? WHERE id=?", (amt, row["id"]))
            if row["total_cents"] - row["refunded_cents"] - amt <= 0:
                conn.execute("UPDATE orders SET status='refunded' WHERE id=?", (row["id"],))
            db.audit(conn, who or "anon", "issue_refund", json.dumps({"order": row["id"], "amount": amt, "id": rid}), **ident(args))
            conn.execute("COMMIT")
            return {"status": "refunded", "refund_id": rid, "amount_cents": amt, "order_id": row["id"], "remaining_cents": remaining - amt}
        except ToolError:
            raise
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()

    def sub_row(conn, sid, who):
        row = conn.execute("SELECT * FROM subscriptions WHERE id=?", (sid,)).fetchone()
        if row is None or (who and row["customer_id"] != who):
            raise ToolError("subscription_not_found", f"No subscription {sid} on this account.")
        return dict(row)

    @srv.tool("check_subscription_options", "Apply subscription policy: can_cancel, refund_eligible, refund_amount_cents.",
              {"type": "object", "properties": {"subscription_id": SUB_ID, "acting_customer_id": ACTING}, "required": ["subscription_id", "acting_customer_id"]},
              scope="orders:read", hidden=("acting_customer_id",))
    def check_subscription_options(args):
        conn = db.connect(path)
        try:
            return evaluate_subscription(sub_row(conn, args["subscription_id"], actor(args)))
        finally:
            conn.close()

    @srv.tool("cancel_subscription", "Cancel a subscription at the end of the billing period (idempotent).",
              {"type": "object", "properties": {"subscription_id": SUB_ID, "idempotency_key": IDEM, "acting_customer_id": ACTING},
               "required": ["subscription_id", "acting_customer_id"]},
              scope="subscriptions:write", risk="write", hidden=("acting_customer_id", "idempotency_key"))
    def cancel_subscription(args):
        who = actor(args)
        conn = db.connect(path)
        try:
            row = sub_row(conn, args["subscription_id"], who)
            if row["status"] == "cancel_at_period_end":              # naturally idempotent: a retry/replay changes nothing and is not re-audited
                return {"status": "cancel_at_period_end", "subscription_id": row["id"], "access_until": row["period_end"], "replayed": True}
            conn.execute("UPDATE subscriptions SET status='cancel_at_period_end' WHERE id=?", (row["id"],))
            db.audit(conn, who or "anon", "cancel_subscription", row["id"], **ident(args))
            return {"status": "cancel_at_period_end", "subscription_id": row["id"], "access_until": row["period_end"]}
        finally:
            conn.close()

    @srv.tool("refund_subscription", "Refund a subscription charge. HIGH RISK: moves money.",
              {"type": "object", "properties": {"subscription_id": SUB_ID, "amount_cents": {"type": "integer", "minimum": 1, "maximum": 1000000},
                                                "idempotency_key": IDEM, "acting_customer_id": ACTING},
               "required": ["subscription_id", "amount_cents", "acting_customer_id"]},
              scope="refunds:issue", risk="high", hidden=("acting_customer_id", "idempotency_key"))
    def refund_subscription(args):
        who = actor(args)
        key = args.get("idempotency_key") or f"auto-{uuid.uuid4().hex}"
        conn = db.connect(path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            prior = conn.execute("SELECT * FROM refunds WHERE idempotency_key=?", (key,)).fetchone()
            if prior:
                conn.execute("COMMIT")
                return {"status": "refunded", "refund_id": prior["id"], "amount_cents": prior["amount_cents"], "replayed": True}
            row = sub_row(conn, args["subscription_id"], who)
            remaining = row["price_cents"] - row["refunded_cents"]
            amt = int(args["amount_cents"])
            if amt <= 0 or amt > remaining:
                conn.execute("ROLLBACK")
                raise ToolError("amount_exceeds_balance", f"Refundable balance is {remaining} cents.", max_amount_cents=remaining)
            rid = f"RFN-{uuid.uuid4().hex[:8]}"
            conn.execute("INSERT INTO refunds VALUES(?,?,?,?,?,?,?,?)", (rid, row["customer_id"], "subscription", row["id"], amt, "subscription", key, NOW.isoformat()))
            conn.execute("UPDATE subscriptions SET refunded_cents=refunded_cents+? WHERE id=?", (amt, row["id"]))
            db.audit(conn, who or "anon", "refund_subscription", json.dumps({"sub": row["id"], "amount": amt}), **ident(args))
            conn.execute("COMMIT")
            return {"status": "refunded", "refund_id": rid, "amount_cents": amt, "subscription_id": row["id"]}
        except ToolError:
            raise
        finally:
            conn.close()

    srv.resource("orders://schema", "Orders schema", "Fields returned by get_order", lambda: ", ".join(PUBLIC_ORDER_FIELDS), "text/plain")
    srv.prompt("refund_triage", "Checklist for handling a refund request",
               [{"name": "order_id", "required": True}],
               lambda a: f"Handle refund for {a.get('order_id','<order>')}: 1) get_order 2) check_refund_eligibility 3) act within the returned max 4) escalate if requires_human.")
    return srv


def main() -> None:
    make_server().run_stdio()


if __name__ == "__main__":
    main()
