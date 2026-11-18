import json

from supportagent.guardrails.pii import PIIVault, new_vault
from supportagent.harness.types import LLMResponse, ToolCall, Usage
from supportagent.llm.scripted import ScriptedLLM


def test_tokenize_text_roundtrip_and_ownership():
    v = PIIVault(new_vault(), "alice@example.com")
    t, kinds = v.tokenize_text("mail alice@example.com or 415-555-0142; also dan@example.com", own=False)
    assert "alice@example.com" not in t and "415-555-0142" not in t and "<EMAIL_1>" in t and set(kinds) == {"email", "phone"}
    out, unresolved = v.detokenize_own(t)
    assert "alice@example.com" in out                                                   # profile email is always 'own'
    assert "dan@example.com" not in out and "<EMAIL_2>" in out and "<PHONE_1>" in out    # foreign + unproven values stay masked
    assert unresolved == ["<PHONE_1>", "<EMAIL_2>"] or set(unresolved) == {"<PHONE_1>", "<EMAIL_2>"}


def test_tokens_are_stable_and_user_typed_values_are_own():
    v = PIIVault(new_vault(), "alice@example.com")
    a = v.tokenize_text("x@y.com", own=True)[0]
    assert v.tokenize_text("x@y.com", own=True)[0] == a                                 # same value -> same token
    assert v.detokenize_own(a)[0] == "x@y.com"


def test_json_tokenization_is_field_aware_and_record_scoped():
    v = PIIVault(new_vault(), "alice@example.com")
    own = v.tokenize_json({"id": "ORD-1", "customer_email": "alice@example.com", "ship_to": "22 Harbor Rd", "notes": "call 415-555-0142", "total_cents": 5})
    other = v.tokenize_json({"id": "ORD-2", "customer_email": "dan@example.com", "ship_to": "14 Elm St"})
    assert own["total_cents"] == 5 and own["id"] == "ORD-1" and "22 Harbor" not in json.dumps(own)
    assert v.detokenize_own(json.dumps(own))[0].count("22 Harbor Rd") == 1             # own record: restorable
    restored_other, unresolved = v.detokenize_own(json.dumps(other))
    assert "14 Elm" not in restored_other and "dan@example.com" not in restored_other and unresolved


def test_foreign_pii_never_restored_to_customer(make_rt):
    """Even if server-side authz were bypassed, another customer's PII comes back to the customer only as inert tokens."""
    rt = make_rt("+tool_guards")
    s = rt.new_session("C002", "bob@example.com")
    v = rt._vault(s)
    tok = v.token_for("EMAIL", "dan@example.com", own=False)
    assert rt._egress(s, f"Their email is {tok}") == f"Their email is {tok}"
    assert any(e["type"] == "pii_token_withheld" for e in s.events)


class Capture(ScriptedLLM):
    sent: list = []

    def complete(self, system, messages, tools, max_tokens=1024, tool_choice=None):
        Capture.sent.append(system + json.dumps([m.to_dict() for m in messages]))
        return super().complete(system, messages, tools, max_tokens, tool_choice)


def run(make_rt, config, text, cid="C001", email="alice@example.com"):
    Capture.sent = []
    rt = make_rt(config, llm_factory=lambda t, s: Capture("strong", s))
    s = rt.new_session(cid, email)
    r = rt.handle_user_message(s, text)
    return rt, s, r, "\n".join(Capture.sent)


def test_llm_never_sees_raw_pii_from_user_or_tools(make_rt):
    rt, s, r, sent = run(make_rt, "full", "I'm alice@example.com, 415-555-0142. Refund ORD-1001 please, card 4242 4242 4242 4242")
    for raw in ("alice@example.com", "415-555-0142", "22 Harbor Rd", "4242 4242"):
        assert raw not in sent, raw
    assert "<EMAIL_1>" in sent and "$37.50" in r.reply
    assert "4242" not in json.dumps([m.to_dict() for m in s.messages])                  # nor persisted in checkpoints


def test_baseline_really_leaks_so_the_metric_means_something(make_rt):
    rt, s, r, sent = run(make_rt, "baseline", "Refund ORD-1001 please")
    assert "alice@example.com" in sent and "22 Harbor Rd" in sent


def test_own_pii_is_restored_in_the_customers_reply(make_rt):
    class Echo:
        name = "echo"

        def complete(self, system, messages, tools, max_tokens=1024, tool_choice=None):
            if tool_choice == "classify_intent":
                return LLMResponse("", [ToolCall("c", "classify_intent", {"intent": "order_status"})], Usage(1, 1), "echo")
            res = [m for m in messages if m.role == "tool"]
            if not res:
                return LLMResponse("", [ToolCall("t", "orders__get_order", {"order_id": "ORD-1001"})], Usage(1, 1), "echo")
            d = json.loads(res[-1].content.split("\n")[1])
            return LLMResponse(f"It ships to {d['ship_to']}; we'll write to {d['customer_email']}.", [], Usage(1, 1), "echo")
    rt = make_rt("full", llm_factory=lambda t, s: Echo())
    s = rt.new_session("C001", "alice@example.com")
    r = rt.handle_user_message(s, "where is ORD-1001")
    assert "22 Harbor Rd" in r.reply and "alice@example.com" in r.reply                # restored for the owner on egress
    assert "<ADDR_1>" in s.messages[-1].content                                         # but stored/checkpointed tokenized


def test_model_tool_args_are_detokenized_for_internal_systems(make_rt, q):
    rt = make_rt("full", llm_factory=lambda t, s: ScriptedLLM("strong", s))
    s = rt.new_session("C001", "alice@example.com")
    s.role = "billing"
    tok = rt._vault(s).token_for("EMAIL", "alice@example.com", own=True)
    out = rt._execute(s, ToolCall("a", "tickets__escalate_to_human", {"summary": f"call back at {tok}", "reason_code": "other"}))
    assert not out.is_error
    assert "alice@example.com" in q("SELECT body FROM tickets")[0]["body"]
