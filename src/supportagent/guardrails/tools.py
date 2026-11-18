"""Tool-level guardrails: allowlist, schema validation, authz injection, policy verification, caps, taint, rate limits.

The key design rule: the model PROPOSES, the harness DISPOSES. Nothing the model says (or was tricked into saying)
can bypass checks that run outside the model, against ground truth fetched by the harness itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import AUTO_REFUND_CAP_CENTS, MAX_WRITE_CALLS_PER_SESSION
from ..harness.types import ToolSpec
from .schema import validate

MONEY_INTENT = re.compile(r"refund|money back|reimburs|\breturn\b|cancel|charge ?back|\$\s?\d[\d,.]* back|get .{0,20} back", re.I)
INTENT_GROUNDED = {"orders__issue_refund", "orders__refund_subscription", "orders__cancel_subscription"}
MONEY_TOOLS = {"orders__issue_refund": "order_id", "orders__refund_subscription": "subscription_id"}


@dataclass
class Decision:
    action: str                      # allow | block | approve
    code: str = ""
    message: str = ""
    details: dict = field(default_factory=dict)

    def as_error(self) -> dict:
        return {"error": "blocked_by_policy", "code": self.code, "message": self.message, **self.details}


class ToolGuard:
    def __init__(self, registry, max_writes: int = MAX_WRITE_CALLS_PER_SESSION, cap_cents: int = AUTO_REFUND_CAP_CENTS):
        self.registry, self.max_writes, self.cap = registry, max_writes, cap_cents

    @staticmethod
    def validate_args(spec: ToolSpec, args: dict) -> list[str]:
        return validate(args, spec.public_schema())

    def authorize(self, session, spec: ToolSpec, args: dict) -> Decision:
        # 1. blast-radius limiter
        if spec.risk in ("write", "high") and session.write_calls >= self.max_writes:
            return Decision("block", "rate_limited", f"Too many write actions in one session (max {self.max_writes}). Escalate to a human.")
        # 2a. intent grounding: tool output (untrusted) can never create the user's intent. A money/cancel action requires the
        #     CUSTOMER'S OWN messages to ask for it - this holds even when injection detection misses a paraphrased attack.
        if spec.name in INTENT_GROUNDED and not self._user_wants_action(session):
            return Decision("block", "no_user_intent", "The customer has not asked for a refund or cancellation in this conversation. Do not take this action.")
        # 2b. money-moving tools: verify against policy ground truth, independently of the model
        if spec.name in MONEY_TOOLS:
            return self._authorize_money(session, spec, args)
        return Decision("allow")

    def _authorize_money(self, session, spec: ToolSpec, args: dict) -> Decision:
        amount = int(args.get("amount_cents", 0))
        acting = {"acting_customer_id": session.customer_id}
        if spec.name == "orders__issue_refund":
            data, err = self.registry.call("orders__check_refund_eligibility",
                                           {"order_id": args["order_id"], "reason": args.get("reason", "changed_mind"), **acting}, role="system")
            if err:
                return Decision("block", "policy_check_failed", f"Could not verify eligibility: {data.get('message', 'error')}")
            if data["requires_human"]:
                return Decision("block", "requires_human", "This case requires a human specialist. Use escalate_to_human; do not retry the refund.",
                                {"reason_code": data["reason_code"]})
            if not data["eligible"]:
                return Decision("block", "policy_ineligible", data["explanation"] + " Do not retry; explain this to the customer or escalate if they request an exception.",
                                {"reason_code": data["reason_code"]})
            max_amt = data["max_amount_cents"]
        else:
            data, err = self.registry.call("orders__check_subscription_options", {"subscription_id": args["subscription_id"], **acting}, role="system")
            if err:
                return Decision("block", "policy_check_failed", f"Could not verify eligibility: {data.get('message', 'error')}")
            if not data["refund_eligible"]:
                return Decision("block", "policy_ineligible", data["explanation"], {"reason_code": data["reason_code"]})
            max_amt = data["refund_amount_cents"]
        if amount > max_amt:
            return Decision("block", "max_amount_exceeded", f"Requested {amount} cents exceeds the policy maximum of {max_amt} cents. Retry with at most {max_amt}.",
                            {"max_amount_cents": max_amt})
        if session.tainted:
            return Decision("approve", "tainted_session", "Session contains suspected prompt injection; human approval required for all money movement.")
        if amount > self.cap:
            return Decision("approve", "over_auto_cap", f"Refund of {amount} cents exceeds the autonomous cap of {self.cap} cents.")
        return Decision("allow")

    @staticmethod
    def _user_wants_action(session) -> bool:
        return any(MONEY_INTENT.search(m.content) for m in session.messages
                   if (m.role == "user" and not m.content.startswith("[")) or (m.role == "tool" and m.tool_name == "ask_user" and not m.is_error))

    @staticmethod
    def authorize_handoff(session, to_role: str) -> Decision:
        if to_role == "billing" and not ToolGuard._user_wants_action(session):
            return Decision("block", "no_user_intent", "Handoff to billing blocked: the customer has not asked for a refund or cancellation.")
        if session.tainted and to_role == "billing":
            return Decision("block", "tainted_handoff", "Handoff to a privileged agent is blocked while the session is tainted by suspected injection.")
        return Decision("allow")
