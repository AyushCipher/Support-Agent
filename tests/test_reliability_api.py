"""API-level reliability: request de-duplication (Idempotency-Key), /metrics for SLOs, /api/admin/slo."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from supportagent.ops.dedup import RequestDedup, fingerprint


@pytest.fixture
def api(tmp_path, monkeypatch):
    import supportagent.llm.scripted as sc
    monkeypatch.setitem(sc.PARAMS, "fast", {"diligence": 1.0, "gullibility": 0.0, "sloppiness": 0.0, "misroute": 0.0, "leak": 0.0})
    from supportagent.api import server
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    monkeypatch.setenv("SA_ADMIN_TOKEN", "tok")
    app = server.App("full")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(path, body=None, headers=None, admin=False):
        h = {"Content-Type": "application/json", **(headers or {})}
        if admin:
            h["Authorization"] = "Bearer tok"
        req = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None, headers=h, method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(req) as r:
                raw = r.read().decode()
                return r.status, (json.loads(raw) if raw.startswith(("{", "[")) else raw)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())
    yield app, call, tmp_path
    srv.shutdown()


def refund_rows(app):
    return app.state.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], __import__("sqlite3").connect(app.db_path).execute(
        "SELECT COUNT(*) FROM refunds WHERE target_id='ORD-1001' AND idempotency_key NOT LIKE 'seed-%'").fetchone()[0]


def test_retry_with_same_idempotency_key_replays_and_does_not_rerun_the_turn(api):
    app, call, _ = api
    body = {"customer_id": "C001", "message": "Refund ORD-1001 please, changed my mind"}
    s1, r1 = call("/api/chat", body, {"Idempotency-Key": "req-1"})
    s2, r2 = call("/api/chat", body, {"Idempotency-Key": "req-1"})                       # the client timed out and retried
    assert s1 == s2 == 200 and r1["status"] == "done" and r2["deduplicated"] is True
    assert r2["session_id"] == r1["session_id"] and r2["reply"] == r1["reply"]
    sessions, refunds = refund_rows(app)
    assert sessions == 1 and refunds == 1                                                 # ONE session, ONE refund
    assert app.counters["chat_requests"] == 1 and app.dedup.stats["replayed"] == 1


def test_same_key_different_payload_is_rejected_and_body_request_id_works(api):
    app, call, _ = api
    call("/api/chat", {"customer_id": "C001", "message": "Where is ORD-1006?"}, {"Idempotency-Key": "k"})
    status, out = call("/api/chat", {"customer_id": "C001", "message": "Refund ORD-1001"}, {"Idempotency-Key": "k"})
    assert status == 422 and "different request" in out["error"]
    a = call("/api/chat", {"customer_id": "C001", "message": "Where is ORD-1006?", "request_id": "body-key"})[1]
    b = call("/api/chat", {"customer_id": "C001", "message": "Where is ORD-1006?", "request_id": "body-key"})[1]
    assert b.get("deduplicated") and a["session_id"] == b["session_id"]
    c = call("/api/chat", {"customer_id": "C001", "message": "Where is ORD-1006?"})[1]    # no key: no dedup, a new session
    assert c["session_id"] != a["session_id"] and not c.get("deduplicated")


def test_failed_request_does_not_poison_the_key(api):
    app, call, _ = api
    assert call("/api/chat", {"customer_id": "NOPE", "message": "x"}, {"Idempotency-Key": "bad"})[1].get("error")
    ok = call("/api/chat", {"customer_id": "NOPE", "message": "x"}, {"Idempotency-Key": "bad"})[1]
    assert ok.get("error") and not ok.get("deduplicated")                                 # error results are not stored: the retry runs again


def test_dedup_state_machine_and_persistence():
    import sqlite3
    conn = sqlite3.connect(":memory:", isolation_level=None)
    now = [0.0]
    d = RequestDedup(conn, inflight_ttl_s=60, clock=lambda: now[0])
    fp = fingerprint("C001", None, "hi")
    assert d.begin("k", fp) == ("new", None)
    assert d.begin("k", fp) == ("in_flight", None)                                        # duplicate while the first is still running
    assert d.begin("k", fingerprint("C001", None, "other"))[0] == "conflict"
    now[0] = 61
    assert d.begin("k", fp)[0] == "new"                                                   # the worker died: a retry may take over
    d.complete("k", {"reply": "x"})
    assert RequestDedup(conn, clock=lambda: now[0]).begin("k", fp) == ("replay", {"reply": "x"})   # survives a restart
    d.begin("j", fp)
    d.fail("j")
    assert d.begin("j", fp)[0] == "new"


def test_metrics_endpoint_exposes_slo_series_and_admin_slo_route(api):
    app, call, _ = api
    call("/api/chat", {"customer_id": "C001", "message": "Where is ORD-1006?"})
    status, text = call("/metrics")
    assert status == 200
    for name in ("supportagent_turns_total", "supportagent_turn_duration_seconds_bucket", "supportagent_tool_calls_total",
                 "supportagent_approvals_oldest_pending_age_seconds", "supportagent_kill_switch_global", "supportagent_tool_cache_hits_total"):
        assert name in text, name
    assert call("/api/admin/slo")[0] in (401, 403)                                        # admin-only
    status, rep = call("/api/admin/slo", admin=True)
    assert status == 200 and {r["name"] for r in rep} >= {"task_success", "turn_latency", "kill_switch_global"}
    assert next(r for r in rep if r["name"] == "task_success")["windows"]["3600s"]["events"] >= 1      # evaluated over the real trace file
