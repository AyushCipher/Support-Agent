import json
import subprocess
import sys

import pytest

from supportagent.config import POLICY_DIR
from supportagent.mcp.client import MCPClient, MCPTimeout, MCPUnavailable, StdioTransport, build_registry
from supportagent.mcp.protocol import MCPServer


def test_protocol_surface(db):
    reg = build_registry(db)
    names = set(reg.specs())
    assert {"orders__get_order", "orders__issue_refund", "kb__search_policy", "tickets__escalate_to_human"} <= names
    c = reg.clients["orders"]
    assert any(r["uri"] == "orders://schema" for r in c.list_resources())
    assert "refund_triage" in [p["name"] for p in c.list_prompts()]
    assert "ORD-1" in c.get_prompt("refund_triage", {"order_id": "ORD-1"})
    kb = reg.clients["kb"]
    assert "30 days" in kb.read_resource("policy://refund_policy")


def test_tool_annotations_drive_risk_and_hidden_params(db):
    spec = build_registry(db).specs()["orders__issue_refund"]
    assert spec.risk == "high" and set(spec.hidden_params) == {"acting_customer_id", "idempotency_key"}
    assert "acting_customer_id" not in spec.public_schema()["properties"]       # the model can never set it


def test_scope_enforcement_per_role(db):
    reg = build_registry(db)
    args = {"order_id": "ORD-1001", "amount_cents": 100, "reason": "other", "acting_customer_id": "C001"}
    data, err = reg.call("orders__issue_refund", args, role="orders")           # read-only role
    assert err and data["error"] == "forbidden"
    data, err = reg.call("orders__get_order", {"order_id": "ORD-1001", "acting_customer_id": "C001"}, role="orders")
    assert not err
    data, err = reg.call("orders__get_order", {"order_id": "ORD-1001", "acting_customer_id": "C001"}, role="triage")
    assert err and data["error"] == "forbidden"                                 # triage has no scopes at all


def test_ownership_enforced_server_side(db):
    reg = build_registry(db)
    data, err = reg.call("orders__get_order", {"order_id": "ORD-4001", "acting_customer_id": "C002"}, role="billing")
    assert err and data["error"] == "order_not_found"
    data, err = reg.call("orders__get_order", {"order_id": "ORD-4001"}, role="billing")
    assert err and data["error"] == "unauthenticated"


def test_refund_hard_invariants_and_idempotency(db, q):
    reg = build_registry(db)
    base = {"order_id": "ORD-1001", "reason": "other", "acting_customer_id": "C001", "idempotency_key": "k1"}
    d, e = reg.call("orders__issue_refund", {**base, "amount_cents": 999999}, role="billing")
    assert e and d["error"] == "amount_exceeds_balance"
    d1, e1 = reg.call("orders__issue_refund", {**base, "amount_cents": 3750}, role="billing")
    d2, e2 = reg.call("orders__issue_refund", {**base, "amount_cents": 3750}, role="billing")          # same key = replay, not double-spend
    assert not e1 and not e2 and d2.get("replayed") and d1["refund_id"] == d2["refund_id"]
    assert len(q("SELECT * FROM refunds WHERE target_id='ORD-1001' AND idempotency_key NOT LIKE 'seed-%'")) == 1


def test_unknown_method_and_tool_are_protocol_errors():
    s = MCPServer("t", tokens={})
    assert s.handle({"jsonrpc": "2.0", "id": 1, "method": "nope"})["error"]["code"] == -32601
    assert s.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "x"}})["error"]["code"] == -32602
    assert s.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_stdio_transport_real_subprocess_parity(db):
    """The same tools work over a real stdio subprocess (newline-delimited JSON-RPC) and in-process."""
    stdio = build_registry(db, mode="stdio")
    try:
        assert not stdio.degraded
        d, e = stdio.call("orders__get_order", {"order_id": "ORD-1001", "acting_customer_id": "C001"}, role="orders")
        assert not e and d["item"] == "Ceramic Pour-Over Set"
        d, e = stdio.call("orders__issue_refund", {"order_id": "ORD-1001", "amount_cents": 1, "reason": "other", "acting_customer_id": "C001"}, role="orders")
        assert e and d["error"] == "forbidden"
        assert stdio.call("kb__search_policy", {"query": "return window"}, role="orders")[0]["results"]
    finally:
        stdio.close()


def test_graceful_degradation_when_server_cannot_start(db):
    bad = MCPClient("docs", lambda: StdioTransport([sys.executable, "-c", "import sys; sys.exit(3)"]), timeout=1)
    reg = build_registry(db, extra=[bad])
    assert "docs" in reg.degraded and "orders__get_order" in reg.specs()          # the rest keeps working
    d, e = reg.call("docs__read", {}, role="orders")
    assert e and d["error"] == "tool_unavailable"


def test_crash_mid_session_reconnects_once(db):
    reg = build_registry(db, mode="stdio")
    try:
        c = reg.clients["orders"]
        c.transport.proc.kill()
        c.transport.proc.wait()
        d, e = reg.call("orders__get_order", {"order_id": "ORD-1001", "acting_customer_id": "C001"}, role="orders")
        assert not e and d["id"] == "ORD-1001"                                    # restarted transparently
    finally:
        reg.close()


def test_circuit_breaker_opens(db):
    class Dead:
        def request(self, *a, **k):
            from supportagent.mcp.client import MCPTimeout
            raise MCPTimeout("slow")
        def notify(self, *a): ...
        def close(self): ...
    c = MCPClient("x", lambda: Dead())
    c.transport = Dead()
    for _ in range(3):
        with pytest.raises(MCPTimeout):
            c.call_tool("t", {}, None)
    with pytest.raises(MCPUnavailable):
        c.call_tool("t", {}, None)


def test_server_is_runnable_as_module(db):
    env = {"SHOP_DB": db, "MCP_TOKENS": json.dumps({"t": ["orders:read"]}), "PYTHONPATH": "src"}
    p = subprocess.run([sys.executable, "-m", "supportagent.mcp.servers.orders_server"], input=json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n", capture_output=True, text=True, env={**__import__("os").environ, **env}, timeout=10)
    assert "get_order" in p.stdout


@pytest.mark.skipif(__import__("shutil").which("npx") is None, reason="needs node/npx (CI installs it)")
def test_third_party_server_least_privilege(db, monkeypatch, tmp_path):
    """Official filesystem MCP server: loaded, defaulted to risky tier, only explicitly granted tools reach the agent."""
    from supportagent.harness import orchestrator
    saved = dict(orchestrator.SPECS)
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"servers": [{"name": "docs", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", str(POLICY_DIR)],
                                            "timeout_s": 60, "grant": {"policy": ["read_text_file"]}}]}))
    monkeypatch.setenv("SA_EXTRA_MCP", str(cfg))
    reg = None
    try:
        reg = build_registry(db)
        if "docs" in reg.degraded:
            pytest.skip(f"npx server unavailable offline: {reg.degraded['docs']}")
        assert reg.specs()["docs__write_file"].risk == "write"                       # unannotated -> fail-safe tier
        assert "docs__read_text_file" in orchestrator.SPECS["policy"].tools           # granted
        assert "docs__write_file" not in orchestrator.SPECS["policy"].tools           # exists on the server, unreachable by any agent
        assert all("docs__" not in t for t in orchestrator.SPECS["billing"].tools)
        d, e = reg.call("docs__read_text_file", {"path": str(POLICY_DIR / "refund_policy.md")}, role="policy")
        assert not e and "30 days" in json.dumps(d)
    finally:
        orchestrator.SPECS.clear()
        orchestrator.SPECS.update(saved)
        if reg:
            reg.close()
