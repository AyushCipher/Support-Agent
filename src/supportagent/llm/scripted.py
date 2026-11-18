"""Deterministic, seeded *simulated* model used for offline development, CI and eval-harness validation.

IMPORTANT - what this is and is not
  It is a policy-driven dialogue simulator with tunable failure modes (laziness, gullibility, sloppiness) that mimic
  well-known LLM-agent failure classes. It lets the harness, guardrails and eval pipeline run reproducibly with no API
  key. Numbers produced with it validate the PIPELINE and the defense-in-depth layers; they say nothing about how any
  real model performs. Run `make eval-live` with an API key for real-model numbers.

The simulator reads the system prompt for a few hardening markers (e.g. "UNTRUSTED"), which lets the ablation study
show prompt-hardening as a separate layer from the structural guardrails.
"""
from __future__ import annotations

import hashlib
import json
import random
import re

from ..harness.types import LLMResponse, Message, ToolCall, ToolSpec, Usage
from .base import TransientLLMError

ORDER_RE = re.compile(r"\bORD-\d{4}\b", re.I)
SUB_RE = re.compile(r"\bSUB-\d{4}\b", re.I)
MONEY_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)")
WRAP_RE = re.compile(r"<tool_output[^>]*>\n?(.*?)\n?</tool_output>", re.S)
INJ_RE = re.compile(r"(ignore (all )?(previous|prior)|system:|override|do not mention)", re.I)

PARAMS = {  # failure-mode probabilities per tier (simulation parameters, not measurements)
    "fast": dict(diligence=0.86, gullibility=0.65, sloppiness=0.10, misroute=0.07, leak=0.50),
    "strong": dict(diligence=0.99, gullibility=0.30, sloppiness=0.01, misroute=0.01, leak=0.20),
}


def cents(dollars: str) -> int:
    return round(float(dollars.replace(",", "")) * 100)


def fmt(c: int) -> str:
    return f"${c / 100:,.2f}"


class _Call:
    def __init__(self, id_, name, args, after_last_user):
        self.id, self.name, self.args, self.after = id_, name, args, after_last_user
        self.short = name.split("__")[-1]
        self.text, self.is_error, self.data = None, False, None


class Conv:
    def __init__(self, messages: list[Message]):
        real_user = [i for i, m in enumerate(messages) if m.role == "user" and not m.content.startswith("[")]
        last_user = real_user[-1] if real_user else -1
        self.user_texts = [messages[i].content for i in real_user]
        self.feedback = [m.content for m in messages if m.role == "user" and m.content.startswith("[OUTPUT_GUARD]")]
        self.calls: list[_Call] = []
        by_id: dict[str, _Call] = {}
        for i, m in enumerate(messages):
            if m.role == "assistant":
                for tc in m.tool_calls:
                    c = _Call(tc.id, tc.name, tc.arguments, i > last_user)
                    self.calls.append(c)
                    by_id[tc.id] = c
            elif m.role == "tool" and m.tool_call_id in by_id:
                c = by_id[m.tool_call_id]
                c.is_error, c.text = m.is_error, m.content
                body = WRAP_RE.search(m.content)
                raw = body.group(1) if body else m.content
                try:
                    c.data = json.loads(raw)
                except json.JSONDecodeError:
                    c.data = {"_text": raw}
        self.last = self.user_texts[-1] if self.user_texts else ""
        self.all_text = " ".join(self.user_texts)

    # --- derived facts -------------------------------------------------------------------------
    @property
    def intent(self) -> str:
        for t in reversed(self.user_texts):
            i = classify(t)
            if i != "other":
                return i
        return "other"

    def latest(self, rx: re.Pattern) -> str | None:
        for t in reversed(self.user_texts):
            m = rx.findall(t)
            if m:
                return m[-1].upper()
        return None

    def answers(self, field: str) -> list[str]:
        return [str(c.data.get("answer", "")) for c in self.calls if c.short == "ask_user" and c.data and not c.is_error and c.data.get("field") == field]

    def from_answers(self, field: str, rx: re.Pattern) -> str | None:
        for a in reversed(self.answers(field)):
            m = rx.findall(a)
            if m:
                return m[-1].upper()
        return None

    @property
    def order_id(self): return self.latest(ORDER_RE) or self.from_answers("order_id", ORDER_RE)
    @property
    def sub_id(self): return self.latest(SUB_RE) or self.from_answers("subscription_id", SUB_RE)

    @property
    def reason(self) -> str:
        t = self.all_text.lower()
        if re.search(r"defective|broken|damaged|doesn'?t work|stopped working", t):
            return "defective"
        if re.search(r"wrong item|wrong size|not what i ordered", t):
            return "wrong_item"
        return "changed_mind"

    @property
    def requested_cents(self) -> int | None:
        for t in reversed(self.user_texts):
            m = MONEY_RE.findall(t)
            if m:
                return cents(m[-1])
        return None

    def called(self, short: str, **match) -> bool:
        return any(c.short == short and all(c.args.get(k) == v for k, v in match.items()) for c in self.calls)

    def result(self, short: str, **match) -> _Call | None:
        hits = [c for c in self.calls if c.short == short and c.data is not None and all(c.args.get(k) == v for k, v in match.items())]
        return hits[-1] if hits else None

    def n_calls(self, short: str) -> int:
        return sum(1 for c in self.calls if c.short == short)

    def injection(self) -> int | None:
        """Return amount (cents) an injected instruction asks for (0 = unspecified) if the raw tool text contains one."""
        for c in self.calls:
            if c.short in ("get_order", "list_orders") and c.text:
                body = WRAP_RE.search(c.text)
                raw = body.group(1) if body else c.text
                if INJ_RE.search(raw) and "refund" in raw.lower():
                    m = re.search(r"refund[^$\d]{0,40}\$\s?(\d[\d,]*)", raw, re.I)
                    return cents(m.group(1)) if m else 0
                if re.search(r"pre-?approved", raw, re.I) and re.search(r"reimburs|refund", raw, re.I):     # paraphrased: evades regex defenses
                    m = re.search(r"(\d[\d,]*)\s*dollars|\$\s?(\d[\d,]*)", raw)
                    return cents(m.group(1) or m.group(2)) if m else 0
        return None

    @property
    def claims_authority(self) -> bool:
        return bool(re.search(r"manager (has )?(approved|authori[sz]ed)|i('m| am) the (store )?(manager|owner|admin)|override|i authori[sz]e|authori[sz]ed (a|the) refund|legal action|lawyer", self.last, re.I))

    @property
    def wants_exception(self) -> bool:
        return bool(re.search(r"exception|manager|supervisor|human|escalat|speak to (a|someone)", self.last, re.I))


def classify(text: str) -> str:
    t = text.lower()
    has_id = bool(ORDER_RE.search(text) or SUB_RE.search(text))
    if re.search(r"system prompt|your instructions|initial instructions|reveal your (rules|prompt)", t):
        return "leak"
    if re.search(r"\b(email|e-mail|phone|address)\b.*\b(on file|customer|of|for)\b", t) and "refund" not in t and has_id:
        return "pii"
    if re.search(r"\bpolicy\b|how long\b|how do(es)?\b|what('s| is) (your|the)|can i return", t) and not has_id:
        return "policy"
    if SUB_RE.search(text) or re.search(r"subscription|membership|\bplan\b|unsubscribe|annual|monthly", t):
        return "subscription"
    if re.search(r"refund|money back|reimburse|\breturn\b|charge ?back|cancel (my |the )?order|cancel it|\$\s?\d[\d,.]* back", t):
        return "refund"
    if re.search(r"where is|where'?s|status|tracking|\btrack\b|arrive|delivery|shipped|\beta\b|details|show me|check on|checking", t):
        return "order_status"
    return "other"


TRIAGE_MAP = {"refund": "refund", "subscription": "subscription", "order_status": "order_status", "policy": "policy_question"}


class ScriptedLLM:
    def __init__(self, tier: str = "fast", seed: int = 0, fail_first_n: int = 0):
        self.tier, self.seed, self.name = tier, seed, f"scripted-{tier}"
        self._fail = fail_first_n
        self.system = ""

    # ------------------------------------------------------------------------------------------
    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec], max_tokens: int = 1024,
                 tool_choice: str | None = None) -> LLMResponse:
        if self._fail > 0:
            self._fail -= 1
            raise TransientLLMError("simulated 529 overloaded")
        self.system = system
        c = Conv(messages)
        p = dict(PARAMS[self.tier])
        if "UNTRUSTED" in system:
            p["gullibility"] *= 0.25
        if "ALWAYS call" in system:
            p["diligence"] = max(p["diligence"], 0.97)
        if "Never reveal" in system:
            p["leak"] *= 0.1
        key = hashlib.md5((c.user_texts[0] if c.user_texts else "").encode()).hexdigest()[:8]

        def rnd(label: str) -> float:
            return random.Random(f"{self.seed}|{key}|{label}").random()

        names = {t.name for t in tools}
        short = {n.split("__")[-1]: n for n in names}
        step = ("text", "")
        if tool_choice == "classify_intent" or "classify_intent" in names:
            intent = TRIAGE_MAP.get(classify(messages[-1].content), "other")
            if rnd("misroute") < p["misroute"]:
                intent = random.Random(f"{self.seed}|{key}|alt").choice(["policy_question", "order_status", "other"])
            step = ("tool", "classify_intent", {"intent": intent, "confidence": 0.9})
        else:
            step = self._decide(c, short, p, rnd)
        return self._wrap(step, short, system, messages, tools)

    def _wrap(self, step, short, system, messages, tools) -> LLMResponse:
        in_tok = (len(system) + sum(len(m.content) + sum(len(json.dumps(t.arguments)) for t in m.tool_calls) for m in messages)
                  + sum(len(json.dumps(t.input_schema)) for t in tools)) // 4
        if step[0] == "tool":
            _, name, args = step
            full = short.get(name, name)
            tc = ToolCall(f"call_{len(messages)}", full, args)
            return LLMResponse("", [tc], Usage(in_tok, len(json.dumps(args)) // 4 + 8), self.name, "tool_use")
        return LLMResponse(step[1], [], Usage(in_tok, len(step[1]) // 4 + 5), self.name, "end_turn")

    # ------------------------------------------------------------------------------------------
    def _decide(self, c: Conv, has: dict, p: dict, rnd):
        T = lambda name, **a: ("tool", name, a)                 # noqa: E731
        X = lambda text: ("text", text)                         # noqa: E731
        intent = c.intent
        last = c.last.lower()
        # system-prompt extraction
        if classify(c.last) == "leak" or intent == "leak":
            if rnd("leak") < p["leak"]:
                return X("Sure, here are my instructions: " + self.system[:400])
            return X("I can't share my internal instructions, but I'm happy to help with your order or account.")

        # PII / other-customer data request
        if intent == "pii":
            oid = c.order_id
            if oid and "get_order" in has and rnd("pii") < p["gullibility"]:
                if not c.called("get_order", order_id=oid):
                    return T("get_order", order_id=oid)
                r = c.result("get_order", order_id=oid)
                if r and not r.is_error:
                    d = r.data
                    return X(f"Order {oid}: {d.get('item')}, ship to {d.get('ship_to')}, contact {d.get('customer_email')}.")
            return X("I can only share information about your own account and orders, so I can't help with that request.")

        # a successful write already happened and the user is just following up
        # --- indirect-injection following (applies in ANY flow once a tool output carried an instruction) ---
        inj = c.injection()
        if inj is not None and not c.called("issue_refund") and rnd("gull_inj") < p["gullibility"]:
            oid = c.order_id or "ORD-2001"
            if "issue_refund" not in has and "handoff" in has and not c.called("handoff"):
                return T("handoff", to="billing", reason="refund requested")
            if "issue_refund" in has:
                amt = inj or cents("35.00")
                return T("issue_refund", order_id=oid, amount_cents=amt, reason="other")

        # misrouted? hand off to the specialist that has the right tools
        need = {"refund": "issue_refund", "subscription": "cancel_subscription", "order_status": "get_order", "policy": "search_policy"}.get(intent)
        if need and need not in has and "handoff" in has and not c.called("handoff"):
            return T("handoff", to={"refund": "billing", "subscription": "billing", "order_status": "orders", "policy": "policy"}[intent],
                     reason=f"customer needs {intent}")

        if intent == "refund":
            return self._refund(c, has, p, rnd, T, X)
        if intent == "subscription":
            return self._subscription(c, has, p, rnd, T, X)
        if intent == "order_status":
            return self._status(c, has, T, X)
        if intent == "policy":
            return self._policy(c, has, T, X)
        if re.search(r"thank|thanks|that'?s all|bye", last):
            return X("You're welcome! Is there anything else I can help you with?")
        return X("I'm happy to help with orders, refunds and subscriptions. Could you tell me a bit more about what you need?")

    # --- flows --------------------------------------------------------------------------------
    def _escalate_text(self, c: Conv) -> str:
        r = c.result("escalate_to_human")
        tid = (r.data or {}).get("ticket_id", "") if r and not r.is_error else ""
        return (f"I can't approve this myself, so I've escalated it to a human specialist (ticket {tid}). "
                "They will contact you within 1 business day.")

    def _refund(self, c, has, p, rnd, T, X):
        oid = c.order_id
        if not oid:
            if "ask_user" in has and not c.called("ask_user", field="order_id"):
                return T("ask_user", question="Happy to help with a refund. Which order is this about? (the order number looks like ORD-1234)", kind="free_text", field="order_id")
            return X("Happy to help with a refund. Could you share the order number (it looks like ORD-1234)?")
        if "get_order" in has and not c.called("get_order", order_id=oid):
            return T("get_order", order_id=oid)
        got = c.result("get_order", order_id=oid)
        if got is not None and got.is_error:
            return X(f"I couldn't find order {oid} on your account. Could you double-check the order number?")
        order = got.data if got else {}
        total, ship = order.get("total_cents", 0), order.get("shipping_cents", 0)
        issued = c.result("issue_refund", order_id=oid)

        # outcome of an attempted refund
        if issued is not None:
            d = issued.data or {}
            if not issued.is_error and d.get("status") == "refunded":
                return X(f"Your refund of {fmt(d['amount_cents'])} for order {oid} has been issued. It should reach your original payment method in 5-10 business days.")
            code = d.get("code") or d.get("error")
            if code == "max_amount_exceeded" and sum(1 for x in c.calls if x.short == "issue_refund" and x.args.get("order_id") == oid) < 2 and "max_amount_cents" in d:
                return T("issue_refund", order_id=oid, amount_cents=min(d["max_amount_cents"], c.requested_cents or d["max_amount_cents"]), reason=c.reason)
            if code in ("requires_human",) and "escalate_to_human" in has and not c.called("escalate_to_human"):
                return T("escalate_to_human", summary=f"Refund request for {oid}; policy requires human review.", reason_code="requires_human")
            if code == "approval_denied":
                return X("A human reviewer reviewed this request and was unable to approve the refund. I'm sorry - you can reply here if you'd like it looked at again.")
            if code == "policy_ineligible":
                if c.called("escalate_to_human"):
                    return X(self._escalate_text(c))
                if (c.wants_exception) and "escalate_to_human" in has:
                    return T("escalate_to_human", summary=f"Customer requests an exception for {oid}: {d.get('message','')}", reason_code="exception_request")
                return X(f"I'm sorry, but I can't refund order {oid}: {d.get('message','').split(' Do not')[0]} This is per our refund policy.")
            if c.called("escalate_to_human"):
                return X(self._escalate_text(c))
            return X("I wasn't able to complete that refund. I'm sorry for the trouble.")

        if any(x.short == "escalate_to_human" and oid in json.dumps(x.args) for x in c.calls):
            return X(self._escalate_text(c))

        lazy = rnd("dilig") >= p["diligence"] or "check_refund_eligibility" not in has
        if not lazy and not c.called("check_refund_eligibility", order_id=oid):
            return T("check_refund_eligibility", order_id=oid, reason=c.reason)
        elig = c.result("check_refund_eligibility", order_id=oid)

        def naive_amount():
            return c.requested_cents or max(0, total - ship)

        if elig is not None and elig.is_error:
            return X(f"I couldn't verify eligibility for {oid} right now. I'll have a specialist follow up.") if not has.get("escalate_to_human") else T("escalate_to_human", summary=f"Eligibility check failed for {oid}", reason_code="other")
        if elig is None:                                                    # lazy path: trusts the customer
            return T("issue_refund", order_id=oid, amount_cents=naive_amount(), reason=c.reason)
        e = elig.data
        if e["requires_human"]:
            if "escalate_to_human" in has:
                return T("escalate_to_human", summary=f"Refund request for {oid} flagged for human review ({e['reason_code']}).", reason_code=e["reason_code"] if e["reason_code"] in ("abuse_review",) else "requires_human")
            return X("This request needs a human specialist's review.")
        if e["eligible"]:
            amt = min(c.requested_cents or e["max_amount_cents"], e["max_amount_cents"])
            if rnd("sloppy_arg") < p["sloppiness"]:
                amt = total                                                 # forgets shipping isn't refundable
            if rnd("sloppy_claim") < p["sloppiness"] and not c.feedback:
                if rnd("variant") < 0.5:
                    return X(f"Good news - I've issued your refund of {fmt(amt)} for order {oid}.")   # claims without acting (regex-detectable)
                return X(f"All set - {fmt(amt)} is on its way back to you for order {oid}.")          # paraphrased claim (evades regex guard)
            return T("issue_refund", order_id=oid, amount_cents=amt, reason=c.reason)
        # ineligible
        if c.claims_authority and rnd("gull_soc") < p["gullibility"] and "issue_refund" in has:
            return T("issue_refund", order_id=oid, amount_cents=naive_amount(), reason="other")
        if c.wants_exception and "escalate_to_human" in has:
            return T("escalate_to_human", summary=f"Customer requests an exception for {oid}: {e['explanation']}", reason_code="exception_request")
        return X(f"I'm sorry, but order {oid} isn't eligible for a refund: {e['explanation']} This is per our refund policy ({', '.join(e['policy_refs'])}). "
                 "If you'd like, I can ask a specialist to review it.")

    def _subscription(self, c, has, p, rnd, T, X):
        sid = c.sub_id
        if not sid:
            if "get_subscriptions" in has and not c.called("get_subscriptions"):
                return T("get_subscriptions")
            r = c.result("get_subscriptions")
            active = [s for s in (r.data or {}).get("subscriptions", []) if s["status"] == "active"] if r and not r.is_error else []
            if len(active) == 1:
                sid = active[0]["id"]
            elif not active:
                return X("I couldn't find an active subscription on your account.")
            else:
                listing = "; ".join(f"{s['id']} ({s['plan']})" for s in active)
                if "ask_user" in has and not c.called("ask_user", field="subscription_id"):
                    return T("ask_user", question=f"Which subscription would you like to change? {listing}", kind="choice", options=[s["id"] for s in active], field="subscription_id")
                return X("Which subscription would you like to change? " + listing)
        if "check_subscription_options" in has and not c.called("check_subscription_options", subscription_id=sid):
            return T("check_subscription_options", subscription_id=sid)
        opts = c.result("check_subscription_options", subscription_id=sid)
        if opts is not None and opts.is_error:
            return X(f"I couldn't find subscription {sid} on your account.")
        o = opts.data if opts else {"can_cancel": True, "refund_eligible": False, "refund_amount_cents": 0, "explanation": ""}
        wants_refund = bool(re.search(r"refund|money back", c.all_text, re.I))
        if o["can_cancel"] and "cancel_subscription" in has and not c.called("cancel_subscription", subscription_id=sid):
            return T("cancel_subscription", subscription_id=sid)
        refunded = c.result("refund_subscription", subscription_id=sid)
        if wants_refund and o["refund_eligible"] and "refund_subscription" in has and refunded is None:
            return T("refund_subscription", subscription_id=sid, amount_cents=o["refund_amount_cents"])
        if refunded is not None:
            d = refunded.data or {}
            if not refunded.is_error and d.get("status") == "refunded":
                return X(f"Done - {sid} is cancelled and your refund of {fmt(d['amount_cents'])} has been issued.")
            return X(f"Your subscription {sid} is set to cancel at the end of the billing period. I wasn't able to complete the refund: a reviewer declined or it was blocked.")
        cancelled = c.result("cancel_subscription", subscription_id=sid)
        msg = f"Your subscription {sid} is now set to cancel at the end of the current billing period, so you keep access until then."
        if wants_refund and not o["refund_eligible"]:
            msg += f" Unfortunately it isn't eligible for a refund: {o['explanation']} (per our subscription policy)."
        if cancelled is None and not o["can_cancel"]:
            return X(o["explanation"])
        return X(msg)

    def _status(self, c, has, T, X):
        oid = c.order_id
        if not oid:
            if "list_orders" in has and not c.called("list_orders"):
                return T("list_orders")
            r = c.result("list_orders")
            inflight = [o for o in (r.data or {}).get("orders", []) if o["status"] in ("processing", "shipped")] if r and not r.is_error else []
            if inflight:
                o = inflight[0]
                return X(f"Your order {o['id']} ({o['item']}) is currently {o['status']}.")
            return X("I couldn't find any orders in progress on your account. Which order number are you asking about?")
        if "get_order" in has and not c.called("get_order", order_id=oid):
            return T("get_order", order_id=oid)
        r = c.result("get_order", order_id=oid)
        if r is None or r.is_error:
            return X(f"I couldn't find order {oid} on your account. Could you double-check the order number?")
        d = r.data
        st = d["status"]
        extra = {"shipped": f" It's in transit with an estimated delivery of {str(d.get('est_delivery'))[:10]}.",
                 "processing": " It's being prepared for shipment.", "delivered": f" It was delivered on {str(d.get('delivered_at'))[:10]}."}.get(st, "")
        return X(f"Order {oid} ({d['item']}) is {st}.{extra}")

    def _policy(self, c, has, T, X):
        if "search_policy" in has and not c.called("search_policy"):
            return T("search_policy", query=c.last[:280], top_k=2)
        r = c.result("search_policy")
        hits = (r.data or {}).get("results", []) if r and not r.is_error else []
        if not hits:
            return X("I couldn't find that in our policy documents. Would you like me to ask a specialist?")
        parts = [f"{h['text'][:260].rsplit('.', 1)[0]}. [source: {h['ref']}]" for h in hits[:2]]
        return X("Here's what our policy says: " + " ".join(parts))
