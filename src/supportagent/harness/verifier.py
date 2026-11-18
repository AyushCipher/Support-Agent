"""Independent verifier before money moves (ADR 0013).

The agent proposes a money action; the harness already verified it against policy (ToolGuard). This adds a second, *independent
model* opinion for the risky slice of actions. Independence is structural, not a prompt promise:
  - a separate model role (llm/router.make_verifier_llm) and a separate prompt family ('verifier', prompts/registry.py);
  - the verifier sees ONLY harness-fetched ground truth (whitelisted order/subscription fields + policy verdict) and the
    customer's own words - never the agent's messages, tool output, notes or reasoning, so a prompt injection that fooled the
    agent has nothing to propagate through;
  - it can only ADD friction: approve+agree -> proceed on the existing path; anything else -> human approval. It can never
    override a ToolGuard block (the verifier runs after authorize()), so a lazy verifier degrades to today's behaviour.
Fail-closed: timeout, outage, malformed output, a different amount, exhausted budget, or missing ground truth all escalate.
"""
from __future__ import annotations

import concurrent.futures as cf
import contextvars
import json
import time
from dataclasses import dataclass

from ..harness.types import Message, ToolCall
from ..obs.cost import cost_usd
from ..prompts import registry as prompts

MONEY_TOOLS = ("orders__issue_refund", "orders__refund_subscription", "orders__cancel_subscription")
ORDER_FIELDS = ("id", "status", "item", "category", "placed_at", "delivered_at", "total_cents", "shipping_cents", "refunded_cents")
# refunded_cents is deliberately absent: get_subscriptions does not return it, so listing it showed the verifier a misleading null
# (review finding R8). The remaining refundable amount is in the policy verdict (check_subscription_options.refund_amount_cents).
SUB_FIELDS = ("id", "plan", "status", "interval", "price_cents", "period_end")
WORDS_PER_MSG, MAX_WORDS_MSGS = 500, 6


@dataclass
class VerifierVerdict:
    status: str                       # agree | disagree | unavailable
    decision: str = "none"            # approve | deny | none
    amount_cents: int = 0
    reason: str = ""
    trigger: str = ""
    model: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tokens: int = 0
    prompt_hash: str = ""

    @property
    def code(self) -> str:
        return "verifier_disagreement" if self.status == "disagree" else "verifier_unavailable"


def customer_words(session) -> list[str]:
    """The customer's OWN words only: real user turns plus validated ask_user answers. Excludes agent text, tool output, harness
    feedback and context summaries (anything the model or a tool author could have shaped)."""
    out: list[str] = []
    for m in session.messages:
        if m.role == "user" and not m.content.startswith("["):
            out.append(m.content)
        elif m.role == "tool" and m.tool_name == "ask_user" and not m.is_error:
            try:
                d = json.loads(m.content)
                if d.get("source") == "customer":
                    out.append(str(d.get("answer", "")))
            except (ValueError, AttributeError):
                pass
    return [w[:WORDS_PER_MSG] for w in out[-MAX_WORDS_MSGS:]]


def quote_words(words: list[str]) -> str:
    """Customer text is embedded as ONE JSON array of strings with '<' and '>' escaped as \\u003c / \\u003e. Every byte the customer
    controls therefore sits inside a JSON string literal on a single line: they can neither close the <customer_words> block (in any
    case or spelling, e.g. '</CUSTOMER_WORDS >') nor start a fake 'GROUND TRUTH:' header on a new line. Review finding R6: the
    previous exact-string .replace('</customer_words>', '') was bypassed by case/whitespace variants."""
    return json.dumps(list(words), ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")


def render_request(action: dict, truth: dict, words: list[str]) -> str:
    return ("PROPOSED ACTION:\n" + json.dumps(action, sort_keys=True) + "\n\n"
            "GROUND TRUTH:\n" + json.dumps(truth, sort_keys=True) + "\n\n"
            "CUSTOMER'S OWN WORDS:\n<customer_words>\n" + quote_words(words) + "\n</customer_words>")


def parse_verdict(text: str) -> tuple[str, int, str] | None:
    """Strict contract. None = malformed (treated as unavailable, i.e. fail closed)."""
    try:
        d = json.loads(text[text.index("{"): text.rindex("}") + 1])
        decision, amt = d["decision"], d.get("amount_cents", 0)
        if decision not in ("approve", "deny") or isinstance(amt, bool) or not isinstance(amt, int):
            return None
        return decision, amt, str(d.get("reason", ""))[:200]
    except (ValueError, KeyError, TypeError):
        return None


class Verifier:
    def __init__(self, cfg, registry, llm_factory, tracer, sleep=time.sleep):
        self.cfg, self.registry, self.llm_factory, self.tracer, self.sleep = cfg, registry, llm_factory, tracer, sleep

    # ------------------------------------------------------------------ when
    def trigger(self, s, spec, args: dict) -> str:
        """Why this action deserves a second opinion ('' = skip): tainted session, risk signals, or over the amount threshold."""
        if spec.name not in MONEY_TOOLS or not self.cfg.verifier:
            return ""
        if s.tainted:
            return "tainted_session"
        if s.risk_signals:
            return "risk_signals"
        if int(args.get("amount_cents", 0) or 0) >= self.cfg.verifier_threshold_cents:
            return "amount_threshold"
        return ""

    # ------------------------------------------------------------------ ground truth (harness-fetched, whitelisted, no PII)
    def ground_truth(self, s, spec, args: dict) -> dict | None:
        acting = {"acting_customer_id": s.customer_id}

        def call(tool, a):
            return self.registry.call(tool, {**a, **acting}, role="system", timeout=self.cfg.tool_timeout_s, session_id=s.id)

        if spec.name == "orders__issue_refund":
            rec, e1 = call("orders__get_order", {"order_id": args["order_id"]})
            pol, e2 = call("orders__check_refund_eligibility", {"order_id": args["order_id"], "reason": args.get("reason", "changed_mind")})
            if e1 or e2:
                return None
            return {"record": {k: rec.get(k) for k in ORDER_FIELDS}, "policy": pol}
        subs, e1 = call("orders__get_subscriptions", {})
        pol, e2 = call("orders__check_subscription_options", {"subscription_id": args["subscription_id"]})
        if e1 or e2:
            return None
        row = next((x for x in subs.get("subscriptions", []) if x.get("id") == args["subscription_id"]), None)
        return None if row is None else {"record": {k: row.get(k) for k in SUB_FIELDS}, "policy": pol}

    # ------------------------------------------------------------------ the call
    def verify(self, s, tc: ToolCall, spec, args: dict, trigger: str) -> VerifierVerdict:
        amount = int(args.get("amount_cents", 0) or 0)
        target = args.get("order_id") or args.get("subscription_id") or ""
        with self.tracer.span("verifier.call", **{"tool.name": spec.name, "verifier.trigger": trigger, "verifier.amount_cents": amount}) as sp:
            v = self._verify(s, spec, args, amount, target, trigger)
            sp["attrs"].update({"verifier.status": v.status, "verifier.decision": v.decision, "verifier.reason": v.reason[:120],
                                "verifier.cost_usd": round(v.cost_usd, 6), "verifier.latency_ms": round(v.latency_ms, 2),
                                "gen_ai.request.model": v.model, "verifier.prompt_hash": v.prompt_hash})
            if v.status == "unavailable":
                sp["status"] = "error"
            return v

    def _verify(self, s, spec, args, amount, target, trigger) -> VerifierVerdict:
        if s.verifier_calls >= self.cfg.verifier_max_calls or s.verifier_cost_usd >= self.cfg.verifier_max_cost_usd:
            return VerifierVerdict("unavailable", reason="verifier budget exhausted", trigger=trigger)    # fail closed
        s.verifier_calls += 1                                          # counted BEFORE the call: failures consume budget too
        truth = self.ground_truth(s, spec, args)
        if truth is None:
            return VerifierVerdict("unavailable", reason="could not fetch ground truth", trigger=trigger)
        system, phash = prompts.load_verifier("v1")
        req = render_request({"tool": spec.name, "target_id": target, "amount_cents": amount, "reason": args.get("reason", "")},
                             truth, customer_words(s))
        t0 = time.perf_counter()
        resp, err = self._complete(system, req)
        ms = (time.perf_counter() - t0) * 1000
        if resp is None:
            return VerifierVerdict("unavailable", reason=err, latency_ms=ms, prompt_hash=phash, trigger=trigger)
        c = cost_usd(resp.model, resp.usage)
        toks = resp.usage.input_tokens + resp.usage.output_tokens
        s.verifier_cost_usd += c
        s.cost_usd += c
        s.tokens += toks
        if resp.model not in s.models_used:
            s.models_used.append(resp.model)
        extra = dict(model=resp.model, latency_ms=ms, cost_usd=c, tokens=toks, prompt_hash=phash, trigger=trigger)
        parsed = parse_verdict(resp.text)
        if parsed is None:
            return VerifierVerdict("unavailable", reason="malformed verifier output", **extra)
        decision, amt, reason = parsed
        same_amount = spec.name.endswith("cancel_subscription") or amt == amount
        if decision == "approve" and not same_amount:
            reason = f"verifier would approve {amt} cents, agent proposed {amount}: {reason}"
        return VerifierVerdict("agree" if decision == "approve" and same_amount else "disagree", decision, amt, reason, **extra)

    def _complete(self, system: str, req: str):
        """One bounded attempt (+1 retry on transient error), under a hard wall-clock timeout. -> (response|None, error)"""
        from ..llm.base import TransientLLMError
        ctx = contextvars.copy_context()
        ex = cf.ThreadPoolExecutor(1, thread_name_prefix="verifier")

        def run():
            last = None
            for attempt in range(2):
                try:
                    return self.llm_factory().complete(system, [Message("user", req)], [], 300)
                except TransientLLMError as e:
                    last = e
                    self.sleep(self.cfg.llm_backoff_s * (2 ** attempt))
            raise last  # type: ignore[misc]

        try:
            fut = ex.submit(ctx.run, run)
            try:
                return fut.result(timeout=self.cfg.verifier_timeout_s), ""
            except cf.TimeoutError:
                return None, f"verifier timeout after {self.cfg.verifier_timeout_s}s"
            except Exception as e:                                      # any failure of the verifier is a fail-closed signal
                return None, f"verifier error: {type(e).__name__}"
        finally:
            ex.shutdown(wait=False)
