"""Deterministic business-policy engine. The single source of truth used by:
   * the orders MCP server (check_* tools the agent can call),
   * the harness ToolGuard (independent verification before any money moves),
   * the eval suite (ground truth).
"""
from __future__ import annotations

from datetime import datetime

from ..config import AUTO_REFUND_CAP_CENTS, NOW

ABUSE_REFUNDS_90D = 3


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def _out(eligible, max_cents, code, why, refs, requires_human=False) -> dict:
    return {"eligible": bool(eligible), "requires_human": requires_human, "max_amount_cents": int(max(0, max_cents)),
            "reason_code": code, "explanation": why, "policy_refs": refs, "auto_cap_cents": AUTO_REFUND_CAP_CENTS}


def evaluate_refund(order: dict, refunds_90d: int, reason: str = "changed_mind", now: datetime = NOW) -> dict:
    total, ship, refunded = order["total_cents"], order["shipping_cents"], order["refunded_cents"]
    remaining = total - refunded
    if remaining <= 0 or order["status"] in ("refunded", "cancelled"):
        return _out(False, 0, "already_refunded", "This order has already been fully refunded.", ["refund_policy#limits"])
    if order["category"] == "final_sale":
        return _out(False, 0, "final_sale", "Final-sale items are not eligible for refunds.", ["refund_policy#final-sale"])
    if refunds_90d >= ABUSE_REFUNDS_90D:
        return _out(False, 0, "abuse_review", "Account has several recent refunds; a human specialist must review.",
                    ["escalation_policy#refund-abuse-review"], requires_human=True)
    status = order["status"]
    if status == "processing":
        return _out(True, remaining, "cancel_before_ship", "Order has not shipped, so it can be cancelled for a full refund.",
                    ["refund_policy#cancellations"])
    if status == "shipped":
        est = _dt(order["est_delivery"])
        if est and (now - est).days > 14:
            return _out(True, remaining, "lost_in_transit", "Shipment is more than 14 days overdue and treated as lost; full refund.",
                        ["refund_policy#lost-packages"])
        return _out(False, 0, "not_yet_delivered", "The order is still in transit; refunds open after delivery or if it is 14 days overdue.",
                    ["refund_policy#lost-packages"])
    if status == "delivered":
        days = (now - _dt(order["delivered_at"])).days
        if reason in ("defective", "wrong_item"):
            if days <= 60:
                return _out(True, remaining, "defective_extended_window",
                            "Defective or wrong items can be refunded within 60 days of delivery, shipping included.",
                            ["refund_policy#defective-or-wrong-items"])
            return _out(False, 0, "outside_window", f"Delivered {days} days ago; the 60-day defect window has passed.",
                        ["refund_policy#defective-or-wrong-items"])
        window = 14 if order["category"] == "digital" else 30
        if days <= window:
            item_remaining = max(0, (total - ship) - refunded)
            if item_remaining <= 0:
                return _out(False, 0, "already_refunded", "The item price has already been refunded; shipping is not refundable.",
                            ["refund_policy#standard-returns"])
            return _out(True, item_remaining, "standard_return",
                        f"Within the {window}-day return window; item price refundable, original shipping is not.",
                        ["refund_policy#standard-returns" if window == 30 else "refund_policy#digital-goods"])
        return _out(False, 0, "outside_window", f"Delivered {days} days ago; the {window}-day return window has passed.",
                    ["refund_policy#standard-returns" if window == 30 else "refund_policy#digital-goods"])
    return _out(False, 0, "unknown_status", "Order status does not allow a refund.", ["refund_policy#limits"])


def evaluate_subscription(sub: dict, now: datetime = NOW) -> dict:
    """Cancellation state and refund eligibility are independent: a customer may cancel and then be refunded.
    (Regression: an earlier version treated cancel_at_period_end as 'nothing to do' and denied the refund - eval A03.)"""
    refs = ["subscription_policy#cancellation"]
    can_cancel = sub["status"] == "active"
    days = (now - _dt(sub["renewed_at"])).days
    remaining = sub["price_cents"] - sub["refunded_cents"]
    base = {"can_cancel": can_cancel, "auto_cap_cents": AUTO_REFUND_CAP_CENTS}
    if sub["status"] == "cancelled":
        return {**base, "refund_eligible": False, "refund_amount_cents": 0, "reason_code": "already_ended",
                "explanation": "This subscription has already ended.", "policy_refs": refs}
    if sub["interval"] == "annual" and days <= 14 and remaining > 0:
        return {**base, "refund_eligible": True, "refund_amount_cents": remaining, "reason_code": "annual_within_14_days",
                "explanation": "Annual plans renewed within the last 14 days are fully refundable.",
                "policy_refs": ["subscription_policy#annual-refunds"]}
    why = ("Monthly plans are not refundable; cancellation takes effect at the end of the billing period."
           if sub["interval"] == "monthly" else
           f"Annual plan renewed {days} days ago; refunds apply only within 14 days of renewal. Cancellation takes effect at period end.")
    return {**base, "refund_eligible": False, "refund_amount_cents": 0,
            "reason_code": "monthly_no_refund" if sub["interval"] == "monthly" else "annual_outside_window",
            "explanation": why, "policy_refs": ["subscription_policy#annual-refunds"]}
