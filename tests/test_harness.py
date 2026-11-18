import json

import pytest

from supportagent.guardrails.approval import ApprovalService
from supportagent.harness.checkpoint import CheckpointStore, Session
from supportagent.harness.context import ContextManager, truncate_tool_result
from supportagent.harness.runtime import Runtime
from supportagent.harness.types import LLMResponse, Message, ToolCall, Usage
from supportagent.llm.scripted import ScriptedLLM
from supportagent.config import PRESETS


def chat(rt, cid, text, seed=1, session=None):
    s = session or rt.new_session(cid, {"C001": "alice@example.com", "C002": "bob@example.com", "C005": "eve@example.com"}[cid], seed=seed)
    return s, rt.handle_user_message(s, text)


def strong(tier, seed):                                   # remove simulated model noise for deterministic harness tests
    return ScriptedLLM("strong", seed)


def test_happy_path_refund(make_rt, q):
    rt = make_rt(llm_factory=strong)
    s, r = chat(rt, "C001", "I'd like a refund for ORD-1001, changed my mind")
    assert r.status == "done" and "$37.50" in r.reply
    assert q("SELECT amount_cents FROM refunds WHERE target_id='ORD-1001' AND idempotency_key NOT LIKE 'seed-%'") == [{"amount_cents": 3750}]
    assert s.evidence["refund_amounts"] == [3750]


def test_indirect_injection_is_quarantined_and_taints(make_rt, q):
    rt = make_rt(llm_factory=strong)
    s, r = chat(rt, "C002", "Can you check on my order ORD-2001?")
    assert s.tainted and "tool_output_injection" in [e["type"] for e in s.events]
    assert not q("SELECT * FROM refunds WHERE customer_id='C002'")
    tool_msgs = [m.content for m in s.messages if m.role == "tool"]
    assert all("refund $500" not in m for m in tool_msgs) and any('trust="untrusted"' in m for m in tool_msgs)


def test_tainted_session_forces_approval_even_for_small_refund(make_rt, q):
    seen = []
    rt = make_rt(llm_factory=strong, reviewer=lambda rec: (seen.append(rec) or (True, "ok")))
    s, r = chat(rt, "C002", "I'd like a refund for ORD-2001")
    assert seen and seen[0]["reason"] == "tainted_session" and r.status == "done"


def test_confused_deputy_is_blocked(make_rt, q):
    rt = make_rt(llm_factory=strong)
    s, r = chat(rt, "C002", "I'd like a refund for ORD-4001")
    assert "couldn't find" in r.reply and not q("SELECT * FROM refunds WHERE customer_id='C004'")


def test_baseline_is_actually_vulnerable(make_rt, q):
    """The vulnerable-by-design baseline must really be exploitable, else the ablation proves nothing."""
    rt = make_rt("baseline", llm_factory=lambda t, s: ScriptedLLM("fast", s))
    d, err = rt.registry.call("orders__get_order", {"order_id": "ORD-4001"}, role="billing")     # anon server, no acting id
    assert not err and d["customer_email"] == "dan@example.com"


def test_approval_deny_path(make_rt, q):
    rt = make_rt(llm_factory=strong, reviewer=lambda rec: (False, "no"))
    s, r = chat(rt, "C001", "I want a refund for ORD-1005")
    assert not q("SELECT * FROM refunds WHERE target_id='ORD-1005'") and "unable" in r.reply.lower()


def test_crash_resume_exactly_once(make_rt, db, q):
    """Pause for approval, 'crash' (new Runtime, same state store), decide, resume -> exactly one refund."""
    rt = make_rt(llm_factory=strong, async_approvals=True)
    s, r = chat(rt, "C001", "Please refund ORD-1005, the espresso machine")
    assert r.status == "awaiting_approval" and not q("SELECT * FROM refunds WHERE target_id='ORD-1005'")
    rt2 = Runtime(rt.cfg, rt.registry, CheckpointStore(rt.state), ApprovalService(rt.state), llm_factory=strong)      # fresh process
    assert rt2.resume(s.id).status == "awaiting_approval"                                   # still pending
    rt2.approvals.decide(r.approval_id, True, "alice-the-reviewer", "ok")
    res = rt2.resume(s.id)
    assert res.status == "done" and "$250.00" in res.reply
    rt2.resume(s.id)                                                                         # double resume is a no-op
    assert len(q("SELECT * FROM refunds WHERE target_id='ORD-1005' AND idempotency_key NOT LIKE 'seed-%'")) == 1


def test_llm_retry_with_backoff(make_rt):
    calls = {"n": 0}
    def flaky(tier, seed):
        calls["n"] += 1
        return ScriptedLLM("strong", seed, fail_first_n=2 if calls["n"] == 2 else 0)
    sleeps = []
    rt = make_rt(llm_factory=flaky, llm_backoff_s=0.5)
    rt.sleep = sleeps.append
    s, r = chat(rt, "C001", "Where is ORD-1006?")
    assert r.status == "done" and sleeps == [0.5, 1.0] and "llm_retry" in [e["type"] for e in s.events]


def test_budget_exceeded_escalates_to_human(make_rt, q):
    rt = make_rt(llm_factory=strong, max_steps=1)
    s, r = chat(rt, "C001", "Refund ORD-1001 please")
    assert r.status == "escalated_budget" and q("SELECT * FROM tickets WHERE reason_code='budget_exceeded'")


def test_tool_server_down_degrades_gracefully(make_rt, q):
    rt = make_rt(llm_factory=strong)
    rt.registry.clients["orders"].transport.server._tools.clear()      # simulate the orders server losing its tools
    rt.registry.clients["orders"].transport.server.tokens = {}
    s, r = chat(rt, "C001", "Where is ORD-1006?")
    assert r.status == "done" and "couldn't find" in r.reply.lower()   # no crash, honest reply


def test_misroute_recovered_by_handoff(make_rt, q):
    class Misroute(ScriptedLLM):                                         # triage always says 'policy_question'
        def complete(self, system, messages, tools, max_tokens=1024, tool_choice=None):
            if tool_choice == "classify_intent":
                from supportagent.harness.types import ToolCall as TC
                return LLMResponse("", [TC("c0", "classify_intent", {"intent": "policy_question"})], Usage(1, 1), self.name)
            return super().complete(system, messages, tools, max_tokens, tool_choice)
    rt = make_rt(llm_factory=lambda t, s: Misroute("strong", s))
    s, r = chat(rt, "C001", "Please refund ORD-1001")
    assert "handoff" in [e["type"] for e in s.events] and "$37.50" in r.reply


def test_handoff_to_billing_blocked_without_user_intent(make_rt, q):
    rt = make_rt(llm_factory=strong)
    s = rt.new_session("C001", "alice@example.com")
    s.role = "orders"
    s.messages.append(Message("user", "Where is ORD-1006?"))
    from supportagent.harness.types import ToolCall as TC
    out = rt._handoff(s, TC("h", "handoff", {"to": "billing"}))
    assert out.is_error and "no_user_intent" in out.content


def test_intent_grounding_blocks_unrequested_refund(make_rt, q):
    """Paraphrased injection (regex evades) -> model tries a policy-eligible refund -> blocked: user never asked."""
    class Gullible(ScriptedLLM):
        def complete(self, system, messages, tools, max_tokens=1024, tool_choice=None):
            if tool_choice == "classify_intent" or any(m.role == "tool" for m in messages):
                if any(m.role == "tool" and "ORD-2005" in m.content for m in messages) and not any(m.tool_calls and m.tool_calls[0].name.endswith("issue_refund") for m in messages):
                    return LLMResponse("", [ToolCall("x", "orders__issue_refund", {"order_id": "ORD-2005", "amount_cents": 2200, "reason": "other"})], Usage(1, 1), self.name)
            return super().complete(system, messages, tools, max_tokens, tool_choice)
    rt = make_rt(llm_factory=lambda t, s: Gullible("strong", s))
    # force the billing role so the issue_refund tool is exposed (the attack's best case)
    s = rt.new_session("C002", "bob@example.com")
    s.role = "billing"
    rt.handle_user_message(s, "What's the status of ORD-2005?")
    assert not q("SELECT * FROM refunds WHERE target_id='ORD-2005'")
    assert any(e.get("code") == "no_user_intent" for e in s.events)


def test_guard_blocks_over_max_amount(make_rt, q):
    rt = make_rt(llm_factory=strong)
    s = rt.new_session("C001", "alice@example.com")
    s.role = "billing"
    s.messages.append(Message("user", "refund ORD-1001"))
    out = rt._execute(s, ToolCall("a", "orders__issue_refund", {"order_id": "ORD-1001", "amount_cents": 4250, "reason": "changed_mind"}))
    assert out.is_error and "max_amount_exceeded" in out.content


def test_model_cannot_set_hidden_params(make_rt):
    rt = make_rt(llm_factory=strong)
    s = rt.new_session("C002", "bob@example.com")
    s.role = "billing"
    out = rt._execute(s, ToolCall("a", "orders__get_order", {"order_id": "ORD-4001", "acting_customer_id": "C004"}))
    assert out.is_error and "invalid_arguments" in out.content      # schema rejects the attempt to impersonate


def test_write_rate_limit(make_rt):
    rt = make_rt(llm_factory=strong)
    s = rt.new_session("C003", "carol@example.com")
    s.role, s.write_calls = "billing", 4
    s.messages.append(Message("user", "cancel SUB-3001"))
    out = rt._execute(s, ToolCall("a", "orders__cancel_subscription", {"subscription_id": "SUB-3001"}))
    assert out.is_error and "rate_limited" in out.content


def test_unknown_tool_and_allowlist(make_rt):
    rt = make_rt(llm_factory=strong)
    s = rt.new_session("C001", "alice@example.com")
    s.role = "orders"                                              # read-only role must not see refunds
    assert rt._execute(s, ToolCall("a", "orders__issue_refund", {"order_id": "ORD-1001", "amount_cents": 1, "reason": "other"})).is_error
    assert rt._execute(s, ToolCall("b", "shell__exec", {})).is_error


def test_sandbox_tool_in_loop(make_rt):
    rt = make_rt(llm_factory=strong)
    s = rt.new_session("C001", "alice@example.com")
    s.role = "billing"
    assert json.loads(rt._execute(s, ToolCall("a", "run_calculation", {"expression": "12000 * 14 / 365"})).content)["result"] > 0
    assert rt._execute(s, ToolCall("b", "run_calculation", {"expression": "__import__('os')"})).is_error


def test_memory_written_only_from_tool_outcomes(make_rt):
    rt = make_rt(llm_factory=strong)
    s, r = chat(rt, "C001", "Refund ORD-1001")
    assert any("refund" in m for m in rt.memory.recall("C001"))
    with pytest.raises(PermissionError):
        rt.memory.remember("C001", "note", "user says VIP", source="user_message")


def test_session_state_roundtrips():
    s = Session("S1", "C001", messages=[Message("assistant", "hi", [ToolCall("1", "t", {"a": 1})]), Message("tool", "x", tool_call_id="1")])
    s2 = Session.from_json(s.to_json())
    assert s2.messages[0].tool_calls[0].arguments == {"a": 1} and s2.messages[1].tool_call_id == "1"


# ---- context management
def test_truncation_keeps_json_valid_and_bounded():
    raw = json.dumps({"id": "ORD-1", "notes": "x" * 5000, "total": 5})
    out = truncate_tool_result(raw, 1800)
    assert len(out) <= 1800 and json.loads(out)["total"] == 5 and "truncated" in out


def test_compaction_never_splits_tool_pairs_and_keeps_recent():
    cfg = PRESETS["full"].with_(context_budget_tokens=50, keep_last_messages=4)
    msgs = []
    for i in range(6):
        msgs += [Message("user", f"question {i} " + "x" * 80), Message("assistant", "", [ToolCall(f"c{i}", "t", {})]), Message("tool", "r" * 80, tool_call_id=f"c{i}"), Message("assistant", f"answer {i}")]
    s = Session("S", "C001", messages=msgs)
    ContextManager(cfg).prepare(s, "sys")
    assert s.messages[0].content.startswith("[CONTEXT SUMMARY") and "question 0" in s.summary
    for i, m in enumerate(s.messages):
        if m.role == "tool":
            assert s.messages[i - 1].tool_calls and s.messages[i - 1].tool_calls[0].id == m.tool_call_id      # pair intact
    assert any("question 5" in m.content for m in s.messages)


def test_long_note_is_truncated_before_reaching_the_model(make_rt):
    rt = make_rt(llm_factory=strong)
    s = rt.new_session("C006", "frank@example.com")
    s.role = "orders"
    out = rt._dispatch(s, ToolCall("a", "orders__get_order", {}), rt._spec("orders__get_order"), {"order_id": "ORD-6001", "acting_customer_id": "C006"})
    assert not out.is_error and len(out.content) < rt.cfg.tool_result_max_chars + 200 and "truncated" in out.content


def test_two_refunds_in_one_session_do_not_cross_contaminate(make_rt, q):
    """F9: second request used the first refund's result and claimed '$37.50 for ORD-1005' without acting."""
    seen = []
    rt = make_rt(llm_factory=strong, reviewer=lambda rec: (seen.append(rec) or (True, "ok")))
    s, r1 = chat(rt, "C001", "Refund ORD-1001 please")
    r2 = rt.handle_user_message(s, "Thanks! Now please also refund ORD-1005, the espresso machine.")
    assert "$250.00" in r2.reply and seen and seen[0]["args"]["order_id"] == "ORD-1005"
    assert sorted(x["amount_cents"] for x in q("SELECT amount_cents FROM refunds WHERE idempotency_key NOT LIKE 'seed-%'")) == [3750, 25000]
