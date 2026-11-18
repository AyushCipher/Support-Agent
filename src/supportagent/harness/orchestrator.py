"""Specialist agent definitions (least-privilege tool allowlists) and local harness tools."""
from __future__ import annotations

from dataclasses import dataclass

from .types import ToolSpec


@dataclass(frozen=True)
class AgentSpec:
    role: str
    tools: tuple[str, ...]
    description: str


SPECS: dict[str, AgentSpec] = {
    "orders": AgentSpec("orders", (
        "orders__get_order", "orders__list_orders", "orders__get_subscriptions", "tickets__get_ticket_status",
        "handoff", "run_calculation", "ask_user"), "Read-only order & subscription status"),
    "billing": AgentSpec("billing", (
        "orders__get_order", "orders__list_orders", "orders__get_subscriptions", "orders__check_refund_eligibility",
        "orders__issue_refund", "orders__check_subscription_options", "orders__cancel_subscription",
        "orders__refund_subscription", "tickets__create_ticket", "tickets__escalate_to_human",
        "kb__search_policy", "handoff", "run_calculation", "ask_user"), "Refunds, cancellations, subscriptions (money-moving)"),
    "policy": AgentSpec("policy", ("kb__search_policy", "handoff", "ask_user"), "Policy questions via RAG"),
}

INTENT_ROLE = {"refund": "billing", "subscription": "billing", "order_status": "orders",
               "policy_question": "policy", "other": "orders"}      # unknown -> least privilege

CLASSIFY = ToolSpec("classify_intent", "Classify the customer's request.", {
    "type": "object", "properties": {
        "intent": {"type": "string", "enum": ["refund", "subscription", "order_status", "policy_question", "other"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}}, "required": ["intent"]})

HANDOFF = ToolSpec("handoff", "Transfer the conversation to another specialist agent.", {
    "type": "object", "properties": {"to": {"type": "string", "enum": ["orders", "billing", "policy"]},
                                     "reason": {"type": "string", "maxLength": 200}}, "required": ["to"]})

CALC = ToolSpec("run_calculation", "Evaluate an arithmetic expression in a sandbox (e.g. proration). Numbers and + - * / only.", {
    "type": "object", "properties": {"expression": {"type": "string", "maxLength": 200}}, "required": ["expression"]})

ASK = ToolSpec("ask_user", "Pause and ask the customer a structured question; the run resumes with their validated answer. "
               "Use when required information (e.g. which order or subscription) is missing or ambiguous.", {
    "type": "object", "properties": {
        "question": {"type": "string", "maxLength": 300},
        "kind": {"type": "string", "enum": ["free_text", "choice", "confirm"]},
        "options": {"type": "array", "items": {"type": "string", "maxLength": 80}, "maxItems": 6, "minItems": 2},
        "field": {"type": "string", "pattern": r"^[a-z_]{1,40}$"}}, "required": ["question", "kind", "field"]})

LOCAL_TOOLS = {"handoff": HANDOFF, "run_calculation": CALC, "ask_user": ASK}
