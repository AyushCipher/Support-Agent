import importlib
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from evals.promote import decide
from supportagent.harness.types import LLMResponse, ToolCall, Usage
from supportagent.llm.scripted import ScriptedLLM
from supportagent.ops.flags import FlagStore


def strong(t, s):
    return ScriptedLLM("strong", s)


class Boom:
    name = "boom"

    def complete(self, *a, **k):
        raise AssertionError("LLM must not be called while the kill switch is on")


# ---- flags + kill switches
def test_flags_persist_and_are_audit_chained(make_rt):
    rt = make_rt()
    rt.flags.set("prompt.canary_percent", 25, "alice")
    assert rt.flags.get("prompt.canary_percent") == 25 and rt.flags.all()["prompt.canary_percent"] == 25
    e = [x for x in rt.audit.entries() if x["action"] == "flag_set"][-1]
    assert e["agent"] == "human:alice" and e["detail"] == {"key": "prompt.canary_percent", "value": 25}
    assert rt.audit.verify()[0]


def test_global_kill_switch_stops_everything_before_the_llm(make_rt):
    rt = make_rt(llm_factory=lambda t, s: Boom())
    rt.flags.set_kill("global", True, "oncall")
    s = rt.new_session("C001", "alice@example.com")
    r = rt.handle_user_message(s, "Refund ORD-1001")
    assert r.status == "killed" and "temporarily unavailable" in r.reply
    assert any(e["action"] == "kill_switch_blocked_turn" for e in rt.audit.entries(session=s.id))


def test_kill_switch_takes_effect_mid_session(make_rt, q):
    rt = make_rt(llm_factory=strong)
    s = rt.new_session("C001", "alice@example.com")
    assert rt.handle_user_message(s, "Where is ORD-1006?").status == "done"
    rt.flags.set_kill("global", True)
    assert rt.handle_user_message(s, "Refund ORD-1001").status == "killed" and not q("SELECT * FROM refunds WHERE idempotency_key NOT LIKE 'seed-%'")
    rt.flags.set_kill("global", False)
    assert rt.handle_user_message(s, "Refund ORD-1001").status == "done"


def test_money_kill_switch_keeps_read_only_help_alive(make_rt, q):
    rt = make_rt(llm_factory=strong)
    rt.flags.set_kill("money", True)
    s = rt.new_session("C001", "alice@example.com")
    r = rt.handle_user_message(s, "Refund ORD-1001 please")
    assert not q("SELECT * FROM refunds WHERE idempotency_key NOT LIKE 'seed-%'") and any(e.get("code") == "kill_switch" for e in s.events)
    assert r.status == "done"                                                           # graceful, honest reply; no crash
    s2 = rt.new_session("C001", "alice@example.com")
    assert "shipped" in rt.handle_user_message(s2, "Where is ORD-1006?").reply          # status checks still work


def test_money_kill_also_stops_an_already_approved_action(make_rt, q):
    rt = make_rt(llm_factory=strong, async_approvals=True)
    s = rt.new_session("C001", "alice@example.com")
    r = rt.handle_user_message(s, "Refund ORD-1005 please")
    assert r.status == "awaiting_approval"
    rt.flags.set_kill("money", True, "oncall")                                          # ops hits the switch while approval is pending
    rt.approvals.decide(r.approval_id, True, "reviewer")
    rt.resume(s.id)
    assert not q("SELECT * FROM refunds WHERE target_id='ORD-1005'")


# ---- canary prompt rollout
def test_canary_assignment_is_sticky_and_proportional(make_rt):
    f = FlagStore(make_rt().state)
    f.set("prompt.canary_version", "v3")
    f.set("prompt.canary_percent", 20)
    arms = [f.assign_prompt(f"S-{i}", "v2") for i in range(2000)]
    share = arms.count("v3") / len(arms)
    assert 0.16 < share < 0.24 and set(arms) == {"v2", "v3"}
    assert f.assign_prompt("S-7", "v2") == f.assign_prompt("S-7", "v2")                 # sticky per session
    f.set("prompt.canary_percent", 0)
    assert {f.assign_prompt(f"S-{i}", "v2") for i in range(200)} == {"v2"}
    f.set("prompt.stable", "v1")
    assert f.assign_prompt("S-1", "v2") == "v1"                                          # promotion/rollback = one flag


def test_sessions_record_their_prompt_arm_and_bad_flags_fail_safe(make_rt):
    rt = make_rt()
    rt.flags.set("prompt.canary_version", "v3")
    rt.flags.set("prompt.canary_percent", 100)
    s = rt.new_session("C001", "alice@example.com")
    from supportagent.prompts import registry as prompts
    assert s.prompt_version == "v3" and s.prompt_hashes == prompts.manifest("v3") and s.prompt_hashes != prompts.manifest("v2")
    rt.flags.set("prompt.canary_version", "v999")                                        # typo in prod must not take the agent down
    assert rt.new_session("C001", "alice@example.com").prompt_version == rt.cfg.prompt_set


# ---- shadow mode
class Divergent:
    name = "div"

    def complete(self, system, messages, tools, max_tokens=1024, tool_choice=None):
        if tool_choice == "classify_intent":
            return LLMResponse("", [ToolCall("c", "classify_intent", {"intent": "order_status"})], Usage(1, 1), "div")
        if "UNTRUSTED" in system or any(m.role == "tool" for m in messages):
            return LLMResponse("Okay.", [], Usage(1, 1), "div")
        return LLMResponse("", [ToolCall("t", "orders__get_order", {"order_id": "ORD-1001"})], Usage(1, 1), "div")


def test_shadow_mode_records_divergence_without_side_effects(make_rt):
    rt = make_rt(llm_factory=lambda t, s: Divergent())
    rt.flags.set("prompt.shadow_version", "v1")                                          # live is v2 (has UNTRUSTED), shadow is v1
    s = rt.new_session("C001", "alice@example.com")
    r = rt.handle_user_message(s, "where is ORD-1001")
    assert r.status == "done" and r.reply == "Okay."                                     # customer saw only the live answer
    rep = rt.flags.shadow_report()
    assert rep["n"] == 1 and rep["tool_call_agreement"] == 0.0 and rep["versions"] == [("v2", "v1")]
    assert f"{s.id}~shadow" in [x["id"] for x in rt.store.list()]


def test_shadow_runs_in_dry_run_and_never_moves_money(make_rt, q):
    approvals_seen = []
    rt = make_rt(llm_factory=strong, reviewer=lambda rec: (approvals_seen.append(rec) or (True, "ok")))
    rt.flags.set("prompt.shadow_version", "v1")
    s = rt.new_session("C001", "alice@example.com")
    rt.handle_user_message(s, "Refund ORD-1005 please")                                  # $250 -> needs approval on the live run
    assert len(q("SELECT * FROM refunds WHERE target_id='ORD-1005'")) == 1               # exactly one real refund
    assert len(approvals_seen) == 1                                                      # shadow never bothers a human
    assert len(rt.memory.recall("C001")) == 1                                            # nor writes memory
    rep = rt.flags.shadow_report()
    assert rep["n"] == 1 and rep["outcome_agreement"] == 1.0
    assert any(e["agent"].startswith("shadow:") for e in rt.audit.entries(session=f"{s.id}~shadow"))   # clearly labeled in the audit trail


# ---- eval-gated promotion
def test_promotion_gate():
    base = {"task_success": 0.95, "attack_success_rate": 0.0, "safety_violation_rate": 0.0, "false_claim_rate": 0.02, "pii_to_llm_rate": 0.0}
    assert decide(base, {**base, "task_success": 0.96}) == []
    assert decide(base, {**base, "task_success": 0.92})                                  # quality regression
    assert decide(base, {**base, "attack_success_rate": 0.01})                           # any safety regression blocks promotion
    assert decide(base, {**base, "pii_to_llm_rate": 0.05})


# ---- admin API + CLI
@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("SA_RUNS_DIR", str(tmp_path))
    import supportagent.llm.scripted as sc
    monkeypatch.setitem(sc.PARAMS, "fast", {"diligence": 1.0, "gullibility": 0.0, "sloppiness": 0.0, "misroute": 0.0, "leak": 0.0})
    import supportagent.config as c
    importlib.reload(c)
    from supportagent.api import server as srvmod
    importlib.reload(srvmod)
    app = srvmod.App("full")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srvmod.make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", monkeypatch
    httpd.shutdown()


def req(base, path, body=None, token=None):
    r = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None, method="POST" if body is not None else "GET",
                               headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_admin_api_auth_kill_switch_and_audit_verify(server):
    base, mp = server
    assert req(base, "/api/admin/flags")[0] == 403                                       # disabled until SA_ADMIN_TOKEN is set
    mp.setenv("SA_ADMIN_TOKEN", "s3cret")
    assert req(base, "/api/admin/flags", token="wrong")[0] == 401
    assert req(base, "/api/admin/flags", token="s3cret")[0] == 200
    code, body = req(base, "/api/chat", {"customer_id": "C001", "message": "Where is ORD-1006?"})
    assert code == 200 and body["status"] == "done"
    assert req(base, "/api/admin/kill", {"scope": "global", "on": True}, token="s3cret")[1]["global"] is True
    assert req(base, "/api/chat", {"customer_id": "C001", "message": "hi"})[1]["status"] == "killed"
    assert b"supportagent_kill_switch_global 1" in urllib.request.urlopen(base + "/metrics").read()
    v = req(base, "/api/admin/audit/verify", token="s3cret")[1]
    assert v["agent_runtime_chain"]["ok"] and v["business_chain"]["ok"]


def test_clarification_over_the_api(server):
    base, _ = server
    _, r = req(base, "/api/chat", {"customer_id": "C003", "message": "cancel my subscription"})
    assert r["status"] == "awaiting_user" and r["clarification"]["options"] == ["SUB-3001", "SUB-3002"]
    _, r2 = req(base, "/api/chat", {"customer_id": "C003", "session_id": r["session_id"], "session_token": r["session_token"], "message": "SUB-3001"})
    assert r2["status"] == "done" and "SUB-3001" in r2["reply"]


def test_cli_ops_commands(tmp_path):
    env = {**os.environ, "SA_RUNS_DIR": str(tmp_path), "PYTHONPATH": os.pathsep.join(["src", "."])}
    run = lambda *a: subprocess.run([sys.executable, "-m", "supportagent.cli", *a], capture_output=True, text=True, env=env, timeout=30).stdout   # noqa: E731
    assert "kill_switch.money = on" in run("kill", "on", "--scope", "money")
    assert "kill_switch.money = True" in run("flags", "list")
    assert "prompt.stable = v3" in run("flags", "set", "prompt.stable", "v3")
    assert "agent runtime chain: OK" in run("audit", "verify")
