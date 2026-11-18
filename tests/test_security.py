import json
import sqlite3

import pytest

from supportagent.mcp.client import build_registry
from supportagent.security.audit import GENESIS, AuditLog
from supportagent.security.identity import IdentityService, TokenError, _b64


# ---- agent identities (JWT)
def test_jwt_roundtrip_and_claims():
    svc = IdentityService("k")
    c = svc.verify(svc.issue("billing", "S1", ["orders:read", "refunds:issue"], "orders"), "orders")
    assert c["sub"] == "agent:billing" and c["sid"] == "S1" and "refunds:issue" in c["scope"].split() and c["aud"] == "orders"


@pytest.mark.parametrize("mutate,err", [
    (lambda t, s: t[:-3] + ("AAA" if not t.endswith("AAA") else "BBB"), "bad signature"),
    (lambda t, s: t.split(".")[0] + "." + _b64(json.dumps({**json.loads(__import__("base64").urlsafe_b64decode(t.split(".")[1] + "==")), "scope": "refunds:issue"}).encode()) + "." + t.split(".")[2], "bad signature"),
    (lambda t, s: "garbage", "malformed"),
    (lambda t, s: _b64(b'{"alg":"none","typ":"JWT"}') + "." + t.split(".")[1] + ".", "unsupported alg"),
    (lambda t, s: _b64(b'{"alg":"HS512","typ":"JWT"}') + "." + t.split(".")[1] + "." + t.split(".")[2], "unsupported alg"),
])
def test_jwt_rejects_forgery(mutate, err):
    svc = IdentityService("k")
    tok = svc.issue("orders", "S1", ["orders:read"], "orders")
    with pytest.raises(TokenError, match=err):
        svc.verify(mutate(tok, svc), "orders")


def test_jwt_expiry_audience_and_wrong_key():
    svc = IdentityService("k")
    t = svc.issue("orders", "S1", ["orders:read"], "orders", ttl=60, now=1000)
    assert svc.verify(t, "orders", now=1059)
    with pytest.raises(TokenError, match="expired"):
        svc.verify(t, "orders", now=1060)
    with pytest.raises(TokenError, match="wrong audience"):
        svc.verify(svc.issue("orders", "S1", [], "orders"), "kb")                      # token for orders is useless on kb
    with pytest.raises(TokenError, match="bad signature"):
        IdentityService("other").verify(svc.issue("orders", "S1", [], "orders"), "orders")


def test_server_rejects_static_tokens_replay_and_spoofed_identity(db):
    reg = build_registry(db)
    srv = reg.clients["orders"].transport.server
    call = lambda tok, args: srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "get_order", "arguments": args, "_meta": {"token": tok}}})["result"]   # noqa: E731
    a = {"order_id": "ORD-1001", "acting_customer_id": "C001"}
    assert "invalid identity" in call("tok-orders", a)["content"][0]["text"]            # legacy static token is dead in JWT mode
    good = reg.identity.issue("orders", "S9", ["orders:read"], "orders")
    assert not call(good, a)["isError"]
    assert "replay" in call(good, a)["content"][0]["text"]                              # same jti twice
    weak = reg.identity.issue("orders", "S9", [], "orders")
    assert "missing scope" in call(weak, a)["content"][0]["text"]


def test_actions_are_attributed_to_agent_identity_and_session(db):
    reg = build_registry(db)
    args = {"order_id": "ORD-1001", "amount_cents": 3750, "reason": "other", "acting_customer_id": "C001", "idempotency_key": "k",
            "_identity": {"sub": "agent:evil", "sid": "forged"}}                       # a client-supplied identity must be ignored
    d, err = reg.call("orders__issue_refund", args, role="billing", session_id="S-attr")
    assert not err
    log = AuditLog(sqlite3.connect(db))
    e = [x for x in log.entries() if x["action"] == "issue_refund"][-1]
    assert e["agent"] == "agent:billing" and e["session"] == "S-attr" and e["customer"] == "C001"
    assert log.verify()[0]


def test_attribution_survives_real_stdio_processes(db):
    reg = build_registry(db, mode="stdio")
    try:
        reg.call("orders__issue_refund", {"order_id": "ORD-1001", "amount_cents": 100, "reason": "other", "acting_customer_id": "C001", "idempotency_key": "a"}, "billing", session_id="S-x")
        reg.call("tickets__escalate_to_human", {"summary": "s", "reason_code": "other", "acting_customer_id": "C001"}, "billing", session_id="S-x")
    finally:
        reg.close()
    log = AuditLog(sqlite3.connect(db))
    got = {(e["action"], e["agent"], e["session"]) for e in log.entries(session="S-x")}
    assert ("issue_refund", "agent:billing", "S-x") in got and ("ticket_created", "agent:billing", "S-x") in got
    assert log.verify()[0]                              # two server processes appended to one chain without forking it


# ---- audit chain
def make_log():
    conn = sqlite3.connect(":memory:")
    log = AuditLog(conn)
    for i in range(6):
        log.append(f"act{i}", agent="agent:billing", session="S1", customer="C001", detail={"i": i})
    return conn, log


def test_chain_verifies_and_detects_edit_delete():
    conn, log = make_log()
    assert log.verify() == (True, None, 6) and log.entries()[0]["hash"]
    conn.execute("UPDATE audit_chain SET detail='{\"i\": 99}' WHERE seq=3")
    assert log.verify()[:2] == (False, 3)                                               # edit localized to the first bad row
    conn2, log2 = make_log()
    conn2.execute("DELETE FROM audit_chain WHERE seq=3")
    assert log2.verify()[0] is False                                                    # deletion breaks the link
    conn3, log3 = make_log()
    conn3.execute("UPDATE audit_chain SET agent='agent:someone-else' WHERE seq=2")
    assert log3.verify()[:2] == (False, 2)                                              # attribution cannot be rewritten


def test_tail_truncation_needs_external_anchor():
    conn, log = make_log()
    anchored = log.head()
    conn.execute("DELETE FROM audit_chain WHERE seq=6")
    assert log.verify()[0] is True and log.head() != anchored                           # documented limit: anchor head hashes externally
    assert AuditLog(sqlite3.connect(":memory:")).head() == GENESIS


def test_replay_timeline_readable():
    _, log = make_log()
    out = log.replay("S1")
    assert "agent:billing" in out and out.count("\n") == 5 and "act3" in out


def test_runtime_audit_covers_the_whole_session(make_rt, db):
    from supportagent.llm.scripted import ScriptedLLM
    rt = make_rt(llm_factory=lambda t, s: ScriptedLLM("strong", s))
    s = rt.new_session("C001", "alice@example.com")
    rt.handle_user_message(s, "Refund ORD-1001 please")
    actions = [e["action"] for e in rt.audit.entries(session=s.id)]
    assert actions[0] == "session_started" and "tool_call" in actions and actions[-1] == "reply_sent"
    assert rt.audit.verify()[0] and AuditLog(sqlite3.connect(db)).verify()[0]
    assert "agent:billing" in rt.audit.replay(s.id)
