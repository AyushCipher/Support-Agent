"""Simulated verifier model (offline). A SEPARATE, independently written decision procedure from llm/scripted.py's agent.

It reads only what the real verifier would: the system prompt and one user message holding the three harness-built sections
(see harness/verifier.py). Fault modes exist so tests can prove the harness stays fail-closed when the verifier is lazy, wrong,
garbled, slow or down. These modes are test doubles, not measurements of any real model.
"""
from __future__ import annotations

import json
import re
import time

from ..harness.types import LLMResponse, Message, ToolSpec, Usage
from .base import TransientLLMError

NEGATION = re.compile(r"(not|n't|never|no)\s+(\w+\s+){0,3}(want|need|ask|asking|request|requesting|looking)\w*\s+(\w+\s+){0,3}(refund|money back|reimburs|cancel)"
                      r"|\bno\s+refund|not\s+a\s+refund|refund\s+is\s+not|policy\s+question", re.I)
ASK = re.compile(r"refund|money back|reimburs|\breturn\b|cancel|charge ?back|get .{0,20} back", re.I)


def _section(text: str, name: str) -> str:
    m = re.search(rf"{re.escape(name)}:\n(.*?)(?=\n[A-Z' ]{{8,}}:\n|\Z)", text, re.S)
    return m.group(1).strip() if m else ""


def parse_request(text: str) -> dict:
    """Inverse of harness.verifier.render_request (used only by the simulator)."""
    def js(name):
        try:
            return json.loads(_section(text, name))
        except Exception:
            return {}
    block = re.findall(r"<customer_words>\n?(.*?)\n?</customer_words>", text, re.S)
    try:
        quoted = json.loads(block[0]) if block else []          # harness.verifier.quote_words: a JSON array of strings
    except ValueError:
        quoted = []
    words = "\n---\n".join(str(w) for w in quoted) if isinstance(quoted, list) else ""
    return {"action": js("PROPOSED ACTION"), "truth": js("GROUND TRUTH"), "words": words}


class ScriptedVerifier:
    """mode: honest | lazy (rubber-stamp) | paranoid (always deny) | garbage | wrong_amount | error | slow"""
    def __init__(self, mode: str = "honest", delay_s: float = 0.0):
        self.mode, self.delay_s, self.name = mode, delay_s, "scripted-verifier"
        self.calls = 0

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec], max_tokens: int = 1024,
                 tool_choice: str | None = None) -> LLMResponse:
        self.calls += 1
        if self.mode == "error":
            raise TransientLLMError("simulated verifier outage")
        if self.delay_s:
            time.sleep(self.delay_s)
        req = parse_request(messages[-1].content)
        out = self._decide(req)
        text = out if isinstance(out, str) else json.dumps(out)
        in_tok = (len(system) + sum(len(m.content) for m in messages)) // 4
        return LLMResponse(text, [], Usage(in_tok, len(text) // 4 + 5), self.name)

    def _decide(self, req: dict):
        act, truth, words = req["action"], req["truth"], req["words"]
        amt = int(act.get("amount_cents") or 0)
        if self.mode == "lazy":
            return {"decision": "approve", "amount_cents": amt, "reason": "looks fine"}
        if self.mode == "paranoid":
            return {"decision": "deny", "amount_cents": 0, "reason": "always deny"}
        if self.mode == "garbage":
            return "Sure! I think this refund is probably okay, go ahead."
        if self.mode == "wrong_amount":
            return {"decision": "approve", "amount_cents": amt + 1, "reason": "approve (different amount)"}
        return self._honest(act, truth, words, amt)

    @staticmethod
    def _honest(act: dict, truth: dict, words: str, amt: int) -> dict:
        deny = lambda why: {"decision": "deny", "amount_cents": 0, "reason": why}                 # noqa: E731
        pol, rec = truth.get("policy") or {}, truth.get("record") or {}
        if not pol or not rec:
            return deny("missing ground truth")
        if not words.strip():
            return deny("no customer words")
        if NEGATION.search(words) or not ASK.search(words):
            return deny("customer did not ask for this")
        target = act.get("target_id", "")
        item_tokens = [w for w in re.findall(r"[a-z]{4,}", str(rec.get("item", "")).lower())]
        if target.lower() not in words.lower() and not any(t in words.lower() for t in item_tokens):
            return deny("customer did not identify this target")
        if pol.get("requires_human") or not pol.get("eligible", pol.get("refund_eligible", False)):
            return deny("policy says not eligible / needs human")
        mx = pol.get("max_amount_cents", pol.get("refund_amount_cents", 0))
        if amt > int(mx or 0):
            return deny("amount exceeds policy maximum")
        if act.get("tool", "").endswith("cancel_subscription"):
            return {"decision": "approve", "amount_cents": 0, "reason": "customer asked to cancel"}
        return {"decision": "approve", "amount_cents": amt, "reason": "customer asked; policy allows"}
