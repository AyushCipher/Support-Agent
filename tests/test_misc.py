import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from supportagent.harness.types import Message, ToolCall, ToolSpec
from supportagent.llm.anthropic_client import AnthropicLLM, to_api_messages, to_api_tools
from supportagent.obs.tracing import JsonlSink, MemorySink, Tracer
from supportagent.prompts import registry as prompts


def test_message_conversion_alternates_and_merges_tool_results():
    msgs = [Message("user", "hi"), Message("assistant", "", [ToolCall("t1", "a", {"x": 1}), ToolCall("t2", "b", {})]),
            Message("tool", "r1", tool_call_id="t1"), Message("tool", "r2", tool_call_id="t2", is_error=True), Message("user", "[OUTPUT_GUARD] fix")]
    api = to_api_messages(msgs)
    assert [m["role"] for m in api] == ["user", "assistant", "user"]
    assert [b["type"] for b in api[2]["content"]] == ["tool_result", "tool_result", "text"] and api[2]["content"][1]["is_error"]


def test_tools_expose_public_schema_only():
    t = ToolSpec("s__t", "d", {"type": "object", "properties": {"a": {"type": "string"}, "acting": {"type": "string"}}, "required": ["a", "acting"]}, hidden_params=("acting",))
    assert list(to_api_tools([t])[0]["input_schema"]["properties"]) == ["a"]


def test_anthropic_adapter_parses_response_and_maps_errors():
    class B:  # minimal content blocks
        def __init__(self, **k): self.__dict__.update(k)
    class Resp:
        content = [B(type="text", text="hello"), B(type="tool_use", id="u1", name="n", input={"k": 1})]
        usage = B(input_tokens=10, output_tokens=5)
        stop_reason = "tool_use"
    class Client:
        class messages:
            @staticmethod
            def create(**kw):
                assert kw["system"][0]["cache_control"] and kw["tool_choice"] == {"type": "tool", "name": "n"}
                return Resp()
    r = AnthropicLLM("claude-haiku-5-5", client=Client).complete("sys", [Message("user", "x")], [ToolSpec("n", "d", {"type": "object", "properties": {}})], tool_choice="n")
    assert r.text == "hello" and r.tool_calls[0].arguments == {"k": 1} and r.usage.input_tokens == 10

    class Rate(Exception): status_code = 429
    class Bad:
        class messages:
            @staticmethod
            def create(**kw): raise Rate()
    from supportagent.llm.base import TransientLLMError
    import pytest
    with pytest.raises(TransientLLMError):
        AnthropicLLM("m", client=Bad).complete("s", [Message("user", "x")], [])


def test_prompt_registry_hashes_and_hardening_markers():
    assert prompts.manifest("v1") != prompts.manifest("v2")
    for role in ("orders", "billing", "policy"):
        v2 = prompts.load(role, "v2")[0]
        assert "UNTRUSTED" in v2 and "Never reveal" in v2
        assert "UNTRUSTED" not in prompts.load(role, "v1")[0]
    assert "ALWAYS call check_refund_eligibility" in prompts.load("billing", "v2")[0]


def test_tracing_spans_nest_and_export(tmp_path):
    mem, path = MemorySink(), tmp_path / "t.jsonl"
    t = Tracer([mem, JsonlSink(path)])
    with t.trace("sess-1", "turn"):
        with t.span("llm.call", **{"gen_ai.usage.input_tokens": 5}):
            pass
    parent = next(s for s in mem.spans if s["name"] == "turn")
    child = next(s for s in mem.spans if s["name"] == "llm.call")
    assert child["parent_id"] == parent["span_id"] and child["trace_id"] == "sess-1"
    assert json.loads(path.read_text().splitlines()[0])["trace_id"] == "sess-1"


def test_runtime_emits_spans_with_genai_attrs(make_rt):
    rt = make_rt()
    s = rt.new_session("C001", "alice@example.com")
    rt.handle_user_message(s, "Where is ORD-1006?")
    names = {sp["name"] for sp in rt.sink.spans}
    assert {"agent.turn", "llm.call", "tool.call", "llm.triage"} <= names
    llm = next(sp for sp in rt.sink.spans if sp["name"] == "llm.call")
    assert llm["attrs"]["gen_ai.usage.input_tokens"] > 0 and "cost_usd" in llm["attrs"]


def test_api_end_to_end(tmp_path, monkeypatch):
    import importlib
    monkeypatch.setenv("SA_RUNS_DIR", str(tmp_path))
    import supportagent.llm.scripted as sc
    monkeypatch.setitem(sc.PARAMS, "fast", {"diligence": 1.0, "gullibility": 0.0, "sloppiness": 0.0, "misroute": 0.0, "leak": 0.0})   # deterministic
    import supportagent.config as c
    importlib.reload(c)
    from supportagent.api import server
    importlib.reload(server)
    app = server.App("full")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    monkeypatch.setenv("SA_ADMIN_TOKEN", "desk-token")                                # reviewer desk is behind real auth now
    hdr = {"Authorization": "Bearer desk-token", "Content-Type": "application/json"}
    post = lambda p, b: json.loads(urllib.request.urlopen(urllib.request.Request(base + p, json.dumps(b).encode(), method="POST", headers=hdr)).read())   # noqa: E731
    get = lambda p: json.loads(urllib.request.urlopen(urllib.request.Request(base + p, headers=hdr)).read())                                              # noqa: E731
    try:
        r = post("/api/chat", {"customer_id": "C001", "message": "Refund ORD-1005 please, it's the espresso machine"})
        assert r["status"] == "awaiting_approval"
        again = post("/api/chat", {"session_id": r["session_id"], "session_token": r["session_token"], "customer_id": "C001", "message": "hello?"})
        assert again["status"] == "awaiting_approval"                              # cannot talk past a pending approval
        pend = get("/api/approvals")["pending"]
        assert len(pend) == 1 and pend[0]["tool"] == "orders__issue_refund"
        d = post(f"/api/approvals/{pend[0]['id']}/decide", {"approve": True, "reviewer": "t"})
        assert d["status"] == "done" and "$250.00" in d["reply"]
        assert get("/api/sessions/" + r["session_id"])["events"]
        assert get("/healthz")["ok"] and "supportagent_chat_requests" in urllib.request.urlopen(base + "/metrics").read().decode()
        assert post("/api/chat", {"customer_id": "NOPE", "message": "x"}).get("error")
        for page in ("/", "/approvals"):
            assert b"<html" in urllib.request.urlopen(base + page).read()
    finally:
        srv.shutdown()
