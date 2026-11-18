import pytest

from supportagent.shop.db import connect
from supportagent.shop.policy import evaluate_refund, evaluate_subscription


def order(db, oid):
    c = connect(db)
    r = dict(c.execute("SELECT * FROM orders WHERE id=?", (oid,)).fetchone())
    c.close()
    return r


@pytest.mark.parametrize("oid,reason,eligible,max_amt,code", [
    ("ORD-1001", "changed_mind", True, 3750, "standard_return"),        # item only, shipping excluded
    ("ORD-1002", "changed_mind", False, 0, "outside_window"),
    ("ORD-1002", "defective", True, 8000, "defective_extended_window"),  # 45d <= 60d, shipping included
    ("ORD-2002", "defective", False, 0, "outside_window"),              # 65d > 60d
    ("ORD-1003", "changed_mind", False, 0, "final_sale"),
    ("ORD-1004", "changed_mind", True, 12000, "cancel_before_ship"),
    ("ORD-1006", "changed_mind", False, 0, "not_yet_delivered"),
    ("ORD-1007", "changed_mind", True, 6500, "lost_in_transit"),
    ("ORD-1008", "changed_mind", True, 3000, "standard_return"),
    ("ORD-1009", "changed_mind", False, 0, "outside_window"),           # digital 20d > 14d
    ("ORD-1010", "changed_mind", True, 4500, "standard_return"),        # partial refund remembered
])
def test_refund_rules(db, oid, reason, eligible, max_amt, code):
    r = evaluate_refund(order(db, oid), refunds_90d=1, reason=reason)
    assert (r["eligible"], r["max_amount_cents"], r["reason_code"]) == (eligible, max_amt, code)


def test_abuse_review_requires_human(db):
    r = evaluate_refund(order(db, "ORD-5001"), refunds_90d=3)
    assert not r["eligible"] and r["requires_human"]


def test_fully_refunded_order(db):
    assert evaluate_refund(order(db, "ORD-5101"), 0)["reason_code"] == "already_refunded"


def test_subscription_refund_independent_of_cancel_state(db):
    """Regression for eval finding F1: cancelling first must not make a refund-eligible plan ineligible."""
    c = connect(db)
    s = dict(c.execute("SELECT * FROM subscriptions WHERE id='SUB-3002'").fetchone())
    assert evaluate_subscription(s)["refund_eligible"]
    s["status"] = "cancel_at_period_end"
    r = evaluate_subscription(s)
    assert r["refund_eligible"] and not r["can_cancel"]


def test_subscription_rules(db):
    c = connect(db)
    get = lambda i: dict(c.execute("SELECT * FROM subscriptions WHERE id=?", (i,)).fetchone())   # noqa: E731
    assert evaluate_subscription(get("SUB-3001"))["reason_code"] == "monthly_no_refund"
    assert evaluate_subscription(get("SUB-7001"))["reason_code"] == "annual_outside_window"
