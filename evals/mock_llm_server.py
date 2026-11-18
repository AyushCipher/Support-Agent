"""Local HTTP server that speaks the Anthropic Messages API (/v1/messages) and the Gemini generateContent API, answering with the
deterministic ScriptedLLM. It lets the REAL live code path (SDK / urllib client -> HTTP -> JSON conversion -> our adapters ->
cache / backoff / cost cap / resume) run end to end offline.  Results through it are SIMULATED-model results, not live results.

Fault injection (to exercise backoff): `faults` is a list of {"every": N, "status": 429|529|500|503, "retry_after": "0"}; the Nth, 2Nth, ...
request gets that error instead of an answer. The seed of the simulated model is taken from the `x-sa-seed` header.
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from supportagent.harness.types import Message, ToolCall, ToolSpec
from supportagent.llm.scripted import ScriptedLLM


def tier_for(model: str) -> str:
    return "fast" if re.search(r"haiku|lite|fast", model) else "strong"


# ---------------------------------------------------------------- request -> harness types
def anthropic_to_messages(msgs: list[dict]) -> list[Message]:
    out: list[Message] = []
    names: dict[str, str] = {}
    for m in msgs:
        blocks = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        if m["role"] == "user":
            for b in blocks:
                if b["type"] == "text":
                    out.append(Message("user", b["text"]))
                elif b["type"] == "tool_result":
                    c = b["content"] if isinstance(b["content"], str) else "".join(x.get("text", "") for x in b["content"])
                    out.append(Message("tool", c, tool_call_id=b["tool_use_id"], tool_name=names.get(b["tool_use_id"]), is_error=bool(b.get("is_error"))))
        else:
            text = "".join(b["text"] for b in blocks if b["type"] == "text")
            calls = [ToolCall(b["id"], b["name"], b.get("input", {})) for b in blocks if b["type"] == "tool_use"]
            for c in calls:
                names[c.id] = c.name
            out.append(Message("assistant", "" if text == "(no content)" else text, calls))
    return out


def gemini_to_messages(contents: list[dict]) -> list[Message]:
    out: list[Message] = []
    pending_ids: list[tuple[str, str]] = []          # (id, name) of calls awaiting responses, in order
    n = 0
    for c in contents:
        parts = c.get("parts", [])
        if c["role"] == "user":
            for p in parts:
                if "text" in p:
                    out.append(Message("user", p["text"]))
                elif "functionResponse" in p:
                    cid, name = pending_ids.pop(0) if pending_ids else (f"call_x{n}", p["functionResponse"]["name"])
                    resp = p["functionResponse"]["response"]
                    is_err = isinstance(resp, dict) and set(resp) == {"error"}
                    body = resp["error"] if is_err else resp
                    out.append(Message("tool", json.dumps(body) if not (isinstance(body, dict) and set(body) == {"result"} and isinstance(body["result"], str)) else body["result"],
                                       tool_call_id=cid, tool_name=name, is_error=is_err))
        else:
            text = "".join(p.get("text", "") for p in parts if "text" in p)
            calls = []
            for p in parts:
                if "functionCall" in p:
                    n += 1
                    tc = ToolCall(f"call_g{n}", p["functionCall"]["name"], p["functionCall"].get("args", {}))
                    calls.append(tc)
                    pending_ids.append((tc.id, tc.name))
            out.append(Message("assistant", "" if text == "(no content)" else text, calls))
    return out


def _specs(tools: list[dict], gemini: bool) -> list[ToolSpec]:
    if gemini:
        return [ToolSpec(d["name"], d.get("description", ""), d.get("parameters", {"type": "object", "properties": {}}))
                for t in tools for d in t.get("functionDeclarations", [])]
    return [ToolSpec(t["name"], t.get("description", ""), t.get("input_schema", {})) for t in tools]


class MockLLMServer:
    def __init__(self, faults: list[dict] | None = None):
        self.faults = faults or []
        self.requests = 0
        self.errors_sent = 0
        self.lock = threading.Lock()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "MockLLMServer":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def _fault(self) -> dict | None:
        with self.lock:
            self.requests += 1
            for f in self.faults:
                if self.requests % f["every"] == 0:
                    self.errors_sent += 1
                    return f
        return None

    def _handler(self):
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):          # quiet
                pass

            def _send(self, code: int, obj: dict, headers: dict | None = None):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
                f = srv._fault()
                gem = "/models/" in self.path
                if f:
                    hdr = {"retry-after": str(f.get("retry_after", "0"))}
                    if gem:
                        return self._send(f["status"], {"error": {"code": f["status"], "message": f.get("message", "mock fault"), "status": "RESOURCE_EXHAUSTED"}}, hdr)
                    kind = {429: "rate_limit_error", 529: "overloaded_error"}.get(f["status"], "api_error")
                    return self._send(f["status"], {"type": "error", "error": {"type": kind, "message": f.get("message", "mock fault")}}, hdr)
                seed = int(self.headers.get("x-sa-seed", "0"))
                if gem:
                    model = self.path.split("/models/")[1].split(":")[0]
                    msgs = gemini_to_messages(body["contents"])
                    system = body["systemInstruction"]["parts"][0]["text"]
                    tools = _specs(body.get("tools", []), True)
                    tc = (body.get("toolConfig", {}).get("functionCallingConfig", {}).get("allowedFunctionNames") or [None])[0]
                else:
                    model = body["model"]
                    msgs = anthropic_to_messages(body["messages"])
                    system = "".join(b["text"] for b in body["system"]) if isinstance(body.get("system"), list) else body.get("system", "")
                    tools = _specs(body.get("tools", []), False)
                    tc = body.get("tool_choice", {}).get("name") if body.get("tool_choice") else None
                r = ScriptedLLM(tier_for(model), seed).complete(system, msgs, tools, body.get("max_tokens") or body.get("generationConfig", {}).get("maxOutputTokens", 1024), tc)
                if gem:
                    parts = ([{"text": r.text}] if r.text else []) + [{"functionCall": {"name": c.name, "args": c.arguments}, "thoughtSignature": "mocksig"} for c in r.tool_calls]
                    return self._send(200, {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
                                            "usageMetadata": {"promptTokenCount": r.usage.input_tokens, "candidatesTokenCount": r.usage.output_tokens}})
                content = ([{"type": "text", "text": r.text}] if r.text else []) + [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in r.tool_calls]
                return self._send(200, {"id": "msg_mock", "type": "message", "role": "assistant", "model": model, "content": content or [{"type": "text", "text": ""}],
                                        "stop_reason": "tool_use" if r.tool_calls else "end_turn", "stop_sequence": None,
                                        "usage": {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}})
        return H
