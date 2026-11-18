"""HTTP API + two small UIs (customer chat, reviewer approvals desk). Stdlib only.

POST /api/chat                       {session_id?, customer_id, message}  -> reply | awaiting_approval, + session_token
                                     continuing a session needs its token (X-Session-Token header or `session_token`); see customerauth
GET  /api/sessions/<id>              transcript + guardrail events (trace panel): X-Session-Token of that session, or admin Bearer
GET  /api/approvals                  pending + recent                      (Bearer SA_ADMIN_TOKEN or OIDC JWT)
POST /api/approvals/<id>/decide      {approve, note} -> resumes the paused run (Bearer; application/json; reviewer = authenticated principal)
GET  /.well-known/jwks.json          PUBLIC agent-identity keys (EdDSA JWKS) so verifiers can check tokens but never mint
GET  /healthz, /metrics              liveness, Prometheus counters/histograms (turns, latency, tool outcomes, cost, verifier)
GET  /api/admin/slo                  SLO status (burn rates, thresholds) evaluated over runs/traces.jsonl + state
POST /api/chat                       idempotent with an `Idempotency-Key` header or `request_id` (retry => stored reply; 422 if the payload changed)
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..config import PRESETS, RUNS_DIR
from ..guardrails.approval import ApprovalService
from ..harness.checkpoint import CheckpointStore
from ..harness.runtime import Runtime
from ..mcp.client import build_registry
from ..memory.store import CustomerMemory
from ..obs.metrics import MetricsRegistry, MetricsSink
from ..obs.tracing import JsonlSink, Tracer
from ..ops import slo as slo_mod
from ..ops.dedup import RequestDedup, fingerprint
from ..ops.flags import FlagStore
from ..security import adminauth, customerauth, prodcheck
from ..security.audit import AuditLog
from ..shop.db import CUSTOMERS, build_seed_db

UI = Path(__file__).parent / "ui"


class App:
    def __init__(self, config: str = "full", backend: str | None = None):
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        be = backend or os.getenv("SA_BACKEND", "scripted")
        self.cfg = PRESETS[config].with_(backend=be, llm_backoff_s=1.0 if be in ("anthropic", "gemini") else 0.0,
                                         handoff_on_llm_error=True)               # a provider 4xx must not 500 a customer
        self.config_warnings = prodcheck.enforce(self.cfg)               # raises in SA_ENV=production; warnings otherwise
        db = RUNS_DIR / "shop.db"
        if not db.exists():
            build_seed_db(db)
        self.state = sqlite3.connect(RUNS_DIR / "state.db", check_same_thread=False, isolation_level=None)
        self.state.row_factory = sqlite3.Row
        self.approvals = ApprovalService(self.state)                     # async: runs pause until a human decides
        self.metrics_reg = MetricsRegistry()
        self.metrics_reg.inc("supportagent_tool_cache_hits_total", by=0, help_="Read-tool calls served from the TTL cache")   # export 0, not absence
        sinks = [JsonlSink(RUNS_DIR / "traces.jsonl"), MetricsSink(self.metrics_reg)]
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            from ..obs.tracing import OTelSink
            sinks.append(OTelSink(endpoint=os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] + "/v1/traces"))
        registry = build_registry(str(db), self.cfg.allow_anon_server, mode=os.getenv("SA_MCP_MODE", "inprocess"))
        self.store = CheckpointStore(self.state)
        self.flags, self.audit = FlagStore(self.state), AuditLog(self.state)
        self.db_path = str(db)
        self.registry = registry
        self.rt = Runtime(self.cfg, registry, self.store, self.approvals, Tracer(sinks), CustomerMemory(self.state), flags=self.flags, audit=self.audit)
        self.emails = {c[0]: c[2] for c in CUSTOMERS}
        self.lock = threading.Lock()
        self.counters = {"chat_requests": 0, "approvals_decided": 0}
        self.dedup = RequestDedup(self.state)

    def chat(self, body: dict, idem_key: str | None = None, headers=None) -> dict:
        """Idempotent when the client sends an Idempotency-Key header / `request_id`: a retry gets the stored response (the turn is not
        re-run); the same key with a different payload is 422; a still-running duplicate is 409. The result may carry `_status` (HTTP code)."""
        key = idem_key or body.get("request_id")
        if not key:
            return self._chat(body, headers)
        key = str(key)[:128]
        tok = (headers or {}).get("X-Session-Token") or body.get("session_token")
        # the session token is part of the fingerprint: replaying a stored reply (which carries the token) needs the token too
        verdict, stored = self.dedup.begin(key, fingerprint(body.get("customer_id"), body.get("session_id"), body.get("message"), tok))
        if verdict == "replay":
            return {**stored, "deduplicated": True}
        if verdict == "conflict":
            return {"error": "idempotency key reused with a different request", "_status": 422}
        if verdict == "in_flight":
            return {"error": "a request with this idempotency key is still being processed", "_status": 409}
        try:
            out = self._chat(body, headers)
        except Exception:
            self.dedup.fail(key)
            raise
        if "error" in out:
            self.dedup.fail(key)                                         # nothing ran: let the client's retry run for real
        else:
            self.dedup.complete(key, out)
        return out

    def _chat(self, body: dict, headers=None) -> dict:
        headers = headers or {}
        with self.lock:
            self.counters["chat_requests"] += 1
            cid = body.get("customer_id", "")
            sid = body.get("session_id")
            if sid:
                s = self._authorized_session(str(sid), headers.get("X-Session-Token") or body.get("session_token"))
                if s is None or (cid and cid != s.customer_id):     # same answer for unknown / foreign / bad token: no session oracle
                    return {"error": "session not found or token invalid", "_status": 404}
            else:
                if cid not in self.emails:
                    return {"error": "unknown customer"}
                try:
                    customerauth.authenticate_customer(headers, cid)
                except customerauth.CustomerAuthError as e:
                    return {"error": e.message, "_status": e.code}
                s = self.rt.new_session(cid, self.emails[cid])
            token = customerauth.issue(s.id, s.customer_id)
            if s.status == "awaiting_approval":
                return {"session_id": s.id, "session_token": token, "status": "awaiting_approval",
                        "reply": "Your request is waiting for human approval."}
            r = self.rt.handle_user_message(s, str(body.get("message", ""))[:2000])
            return {"session_id": s.id, "session_token": token, "status": r.status, "reply": r.reply, "approval_id": r.approval_id,
                    "clarification": r.clarification}

    def _authorized_session(self, sid: str, token: str | None):
        """The session iff `token` is its session-bound token, else None (no distinction between missing, foreign and bad)."""
        try:
            s = self.store.load(sid)
        except KeyError:
            return None
        return s if customerauth.check(token, s.id, s.customer_id) else None

    def decide(self, aid: str, body: dict) -> dict:
        with self.lock:
            self.counters["approvals_decided"] += 1
            ap = self.approvals.decide(aid, bool(body.get("approve")), str(body.get("reviewer", "reviewer"))[:40], str(body.get("note", ""))[:300])
            r = self.rt.resume(ap["session_id"])
            return {"approval": ap, "session_id": ap["session_id"], "status": r.status, "reply": r.reply}

    def session_view(self, sid: str, token: str | None = None, admin: bool = False) -> dict:
        """KeyError unless the caller holds this session's token (or is an authenticated admin/reviewer)."""
        s = self.store.load(sid) if admin else self._authorized_session(sid, token)
        if s is None:
            raise KeyError(sid)
        return {"id": s.id, "status": s.status, "role": s.role, "tainted": s.tainted, "tokens": s.tokens, "cost_usd": round(s.cost_usd, 5),
                "models": s.models_used, "events": s.events,
                "messages": [{"role": m.role, "content": m.content[:600], "tools": [t.name for t in m.tool_calls]} for m in s.messages]}

    def verify_audit(self) -> dict:
        shop = AuditLog(sqlite3.connect(self.db_path))
        a, b = self.audit.verify(), shop.verify()
        return {"agent_runtime_chain": {"ok": a[0], "first_bad_seq": a[1], "entries": a[2], "head": self.audit.head()},
                "business_chain": {"ok": b[0], "first_bad_seq": b[1], "entries": b[2], "head": shop.head()}}

    def metrics(self) -> str:
        lines = [f"supportagent_{k} {v}" for k, v in self.counters.items()]
        lines.append(f"supportagent_approvals_pending {len(self.approvals.pending())}")
        lines.append(f"supportagent_kill_switch_global {int(self.flags.killed('global'))}")
        lines.append(f"supportagent_kill_switch_money {int(self.flags.killed('money'))}")
        lines.append(f"supportagent_approvals_oldest_pending_age_seconds {slo_mod.load_gauges(self.state).get('approval_oldest_pending_s', 0.0):.1f}")
        lines += [f"supportagent_request_dedup_{k}_total {v}" for k, v in self.dedup.stats.items()]
        # one labelled family: the old per-stat names collided with MetricsSink's supportagent_tool_cache_hits_total and produced
        # a duplicate series, which makes Prometheus reject the WHOLE scrape (review finding R5)
        lines += [f'supportagent_tool_cache_ops_total{{op="{k}"}} {v}' for k, v in self.rt.cache.stats.items()]
        return "\n".join(lines) + "\n" + self.metrics_reg.render()

    def slo_report(self) -> list[dict]:
        samples, gauges = slo_mod.load_samples(RUNS_DIR / "traces.jsonl"), slo_mod.load_gauges(self.state)
        return [st.to_dict() for st in slo_mod.SLOEvaluator().evaluate(samples, gauges)]


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes | str, ctype: str = "application/json") -> None:
            b = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(b)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, default=str))

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            try:
                return json.loads(self.rfile.read(min(n, 65536)) or b"{}")
            except json.JSONDecodeError:
                return {}

        def _admin(self, csrf: bool = False, require_json: bool = False):
            """Admin + reviewer endpoints: SA_ADMIN_TOKEN (constant-time) or an OIDC bearer JWT; disabled (403) when neither is
            configured. Returns the Principal, or None after sending the error. See security/adminauth.py."""
            try:
                who = adminauth.authenticate(self.headers)
                if csrf:
                    adminauth.check_csrf(self.headers, require_json)
                return who
            except adminauth.AuthError as e:
                self._json(e.code, {"error": e.message})
                return None

        def do_GET(self):  # noqa: N802
            p = self.path.split("?")[0]
            if p == "/.well-known/jwks.json":
                ident = getattr(app.registry, "identity", None)
                if not hasattr(ident, "jwks"):
                    return self._json(404, {"error": "no asymmetric identity configured (HS256 dev mode has no public keys)"})
                return self._send(200, json.dumps(ident.jwks()), "application/jwk-set+json")
            if p.startswith("/api/admin/"):
                if not self._admin():
                    return
                if p == "/api/admin/flags":
                    return self._json(200, app.flags.all())
                if p == "/api/admin/audit/verify":
                    return self._json(200, app.verify_audit())
                if p == "/api/admin/shadow":
                    return self._json(200, app.flags.shadow_report())
                if p == "/api/admin/slo":
                    return self._json(200, app.slo_report())
                if p.startswith("/api/admin/audit/replay/"):
                    return self._json(200, {"timeline": app.audit.replay(p.rsplit("/", 1)[1])})
            if p == "/":
                return self._send(200, (UI / "chat.html").read_bytes(), "text/html; charset=utf-8")
            if p == "/approvals":
                return self._send(200, (UI / "approvals.html").read_bytes(), "text/html; charset=utf-8")
            if p == "/healthz":
                return self._json(200, {"ok": True, "config": app.cfg.name, "backend": app.cfg.backend})
            if p == "/metrics":
                return self._send(200, app.metrics(), "text/plain")
            if p == "/api/customers":
                return self._json(200, [{"id": c[0], "name": c[1]} for c in CUSTOMERS])
            if p == "/api/approvals":
                if not self._admin():
                    return
                return self._json(200, {"pending": app.approvals.pending(), "recent": app.approvals.all(15)})
            if p.startswith("/api/sessions/"):
                tok = self.headers.get("X-Session-Token")
                admin = False
                if not tok:
                    if not adminauth.bearer(self.headers):
                        return self._json(401, {"error": "X-Session-Token required"})
                    if not self._admin():                                     # reviewers may open any transcript
                        return
                    admin = True
                try:
                    return self._json(200, app.session_view(p.rsplit("/", 1)[1], tok, admin))
                except KeyError:
                    return self._json(404, {"error": "not found"})
            self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            p = self.path.split("?")[0]
            try:
                if p.startswith("/api/admin/"):
                    if not self._admin(csrf=True):
                        return
                    b = self._body()
                    if p == "/api/admin/flags":
                        app.flags.set(str(b["key"]), b["value"], "admin-api")
                        return self._json(200, app.flags.all())
                    if p == "/api/admin/kill":
                        app.flags.set_kill(str(b.get("scope", "global")), bool(b.get("on", True)), "admin-api")
                        return self._json(200, {"global": app.flags.killed("global"), "money": app.flags.killed("money")})
                if p == "/api/chat":
                    res = app.chat(self._body(), self.headers.get("Idempotency-Key"), self.headers)
                    return self._json(res.pop("_status", 200), res)
                if p.startswith("/api/approvals/") and p.endswith("/decide"):
                    who = self._admin(csrf=True, require_json=True)
                    if who is None:
                        return
                    aid = p.split("/")[3]
                    try:
                        if app.approvals.get(aid)["status"] != "pending":
                            return self._json(409, {"error": "already decided"})
                    except (TypeError, KeyError):
                        return self._json(404, {"error": "unknown approval"})
                    return self._json(200, app.decide(aid, {**self._body(), "reviewer": who.name}))
            except Exception as e:                      # never leak stack traces to clients
                return self._json(500, {"error": "internal error", "type": type(e).__name__})
            self._json(404, {"error": "not found"})

        def log_message(self, *a):  # quiet
            pass
    return H


def serve(host: str = "0.0.0.0", port: int = 8080, config: str = "full", backend: str | None = None) -> None:
    try:
        app = App(config, backend)                                        # raises InsecureConfig in production (prodcheck)
    except prodcheck.InsecureConfig as e:
        print(f"supportagent: {e}", file=sys.stderr)                      # names of env vars only, never their values
        raise SystemExit(2) from None
    for w in app.config_warnings:
        print(f"supportagent: WARNING (would refuse in SA_ENV=production): {w}")
    print(f"supportagent listening on http://{host}:{port}  (config={app.cfg.name}, backend={app.cfg.backend})")
    ThreadingHTTPServer((host, port), make_handler(app)).serve_forever()


if __name__ == "__main__":
    serve(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8080")), config=os.getenv("SA_CONFIG", "full"))
