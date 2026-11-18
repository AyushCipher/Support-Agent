"""Google Gemini adapter (function calling) behind the same `LLM` interface as anthropic_client.AnthropicLLM.

Stdlib only (urllib): no SDK dependency, so the core keeps ADR-0001's "no runtime deps" property.

  SA_BACKEND=gemini  SA_MODEL_FAST=gemini-3.5-flash-lite  SA_MODEL_STRONG=gemini-3.5-flash  GEMINI_API_KEY=...
  SA_GEMINI_RPM=15        per-minute pacing shared by all clients in the process (0 = off)
  SA_GEMINI_BASE_URL=...  override the endpoint (tests point it at a local mock server)

Behaviour
  * messages / tools are converted to Gemini `contents` / `functionDeclarations` (schemas are reduced to the supported subset);
  * `thoughtSignature` parts (required by Gemini 3 to continue a function-calling turn) are carried inside ToolCall.id and
    re-attached when the history is sent back, so signatures survive checkpointing and per-step client re-creation;
  * 429 / 500 / 503 / timeouts are retried with exponential backoff + jitter (Retry-After / `retryDelay` honoured);
    after `max_retries` they raise TransientLLMError so the harness's own retry/degradation logic still applies;
  * a DAILY quota exhaustion (429 with a per-day quota id) raises QuotaExhausted immediately: retrying cannot help;
  * usage -> Usage(input, output); thinking tokens are billed as output.
The API key is read from the environment or the git-ignored .env and is never logged.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from ..harness.types import LLMResponse, Message, ToolCall, ToolSpec, Usage
from .base import TransientLLMError

DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta"
SIG_SEP = "~sig~"


class QuotaExhausted(Exception):
    """Daily (or otherwise non-retryable) quota exhausted: stop the run and resume later."""


# ---------------------------------------------------------------- key + pacing
def load_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    for d in (Path.cwd(), *Path(__file__).resolve().parents[:5]):
        f = d / ".env"
        if f.is_file():
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                m = re.match(r"\s*GEMINI_API_KEY\s*=\s*(.*)\s*$", line)
                if m and m.group(1).strip().strip("\"'"):
                    return m.group(1).strip().strip("\"'")
    return None


class Pacer:
    """Process-wide minimum spacing between requests (thread-safe)."""
    def __init__(self):
        self._lock, self._next = threading.Lock(), 0.0

    def wait(self, rpm: float, now=time.monotonic, sleep=time.sleep) -> None:
        if rpm <= 0:
            return
        gap = 60.0 / rpm
        with self._lock:
            t = now()
            delay = max(0.0, self._next - t)
            self._next = max(t, self._next) + gap
        if delay:
            sleep(delay)


PACER = Pacer()


# ---------------------------------------------------------------- conversion
_SCHEMA_KEEP = {"type", "description", "enum", "properties", "required", "items", "minimum", "maximum", "minItems", "maxItems",
                "format", "nullable", "minLength", "maxLength", "pattern"}


def to_gemini_schema(node):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "properties":
                out[k] = {pk: to_gemini_schema(pv) for pk, pv in v.items()}
            elif k in _SCHEMA_KEEP:
                out[k] = to_gemini_schema(v) if k == "items" else v
        return out
    return node


def to_gemini_tools(tools: list[ToolSpec]) -> list[dict]:
    decls = []
    for t in tools:
        d = {"name": t.name, "description": t.description}
        schema = to_gemini_schema(t.public_schema())
        if schema.get("properties"):                      # Gemini rejects an object schema with no properties
            d["parameters"] = schema
        decls.append(d)
    return [{"functionDeclarations": decls}] if decls else []


def to_gemini_contents(messages: list[Message]) -> list[dict]:
    out: list[dict] = []

    def push(role: str, parts: list[dict]) -> None:
        if out and out[-1]["role"] == role:
            out[-1]["parts"].extend(parts)
        else:
            out.append({"role": role, "parts": parts})

    for m in messages:
        if m.role == "user":
            push("user", [{"text": m.content}])
        elif m.role == "assistant":
            parts: list[dict] = [{"text": m.content}] if m.content else []
            for tc in m.tool_calls:
                p: dict = {"functionCall": {"name": tc.name, "args": tc.arguments}}
                if SIG_SEP in tc.id:
                    p["thoughtSignature"] = tc.id.split(SIG_SEP, 1)[1]
                parts.append(p)
            push("model", parts or [{"text": "(no content)"}])
        elif m.role == "tool":
            body = m.content
            try:
                parsed = json.loads(body)
                resp = parsed if isinstance(parsed, dict) else {"result": parsed}
            except (json.JSONDecodeError, TypeError):
                resp = {"result": body}
            if m.is_error:
                resp = {"error": resp}
            push("user", [{"functionResponse": {"name": m.tool_name or "tool", "response": resp}}])
    return out


def _retry_delay(headers, body: str) -> float | None:
    ra = headers.get("Retry-After") if headers else None
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    m = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', body)
    return float(m.group(1)) if m else None


class GeminiLLM:
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, timeout: float = 90.0, max_retries: int = 6,
                 rpm: float | None = None, backoff_base: float = 2.0, backoff_cap: float = 60.0, sleep=time.sleep, extra_headers: dict | None = None,
                 thinking: str | None = None):
        self.name = model
        self.key = api_key or load_api_key()
        self.base = (base_url or os.environ.get("SA_GEMINI_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self.timeout, self.max_retries, self.sleep = timeout, max_retries, sleep
        self.backoff_base, self.backoff_cap = backoff_base, backoff_cap
        self.rpm = float(os.environ.get("SA_GEMINI_RPM", "0")) if rpm is None else rpm
        self.extra_headers = extra_headers or {}
        self.thinking = thinking if thinking is not None else os.environ.get("SA_GEMINI_THINKING")   # e.g. "minimal" | "low" | "budget:0"
        self.requests = 0                                                                              # HTTP requests attempted by this instance
        if not self.key:
            raise RuntimeError("GEMINI_API_KEY is not set (environment or .env)")

    # ------------------------------------------------------------
    def _payload(self, system, messages, tools, max_tokens, tool_choice) -> dict:
        body: dict = {"systemInstruction": {"parts": [{"text": system}]}, "contents": to_gemini_contents(messages),
                      "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.0}}
        decl = to_gemini_tools(tools)
        if decl:
            body["tools"] = decl
            if tool_choice:
                body["toolConfig"] = {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [tool_choice]}}
        if self.thinking:
            if self.thinking.startswith("budget:"):
                body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": int(self.thinking.split(":")[1])}
            else:
                body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self.thinking}
        return body

    def _post(self, body: dict) -> dict:
        url = f"{self.base}/models/{self.name}:generateContent"
        data = json.dumps(body).encode()
        for attempt in range(self.max_retries + 1):
            PACER.wait(self.rpm)
            self.requests += 1
            req = urllib.request.Request(url, data=data, method="POST", headers={
                "content-type": "application/json", "x-goog-api-key": self.key, **self.extra_headers})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="ignore")
                if e.code == 429 and re.search(r"PerDay|per day|daily", text, re.I) and not re.search(r"PerMinute", text):
                    raise QuotaExhausted(f"daily quota exhausted for {self.name}") from None
                if e.code in (408, 429, 500, 502, 503, 504):
                    err: Exception = TransientLLMError(f"HTTP {e.code} from Gemini")
                    wait = _retry_delay(e.headers, text)
                elif e.code == 400 and "thought" in text.lower() and "signature" in text.lower():
                    raise RuntimeError("Gemini rejected the request: missing thought signature (" + text[:200] + ")") from None
                else:
                    raise RuntimeError(f"Gemini HTTP {e.code}: {text[:300]}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                err, wait = TransientLLMError(f"{type(e).__name__}: {e}"), None
            if attempt == self.max_retries:
                raise err
            if wait is not None:
                delay = min(self.backoff_cap, wait)
            else:
                delay = min(self.backoff_cap, self.backoff_base * (2 ** attempt)) * (0.75 + 0.5 * random.random())
            self.sleep(delay)
        raise TransientLLMError("unreachable")  # pragma: no cover

    # ------------------------------------------------------------
    def complete(self, system, messages, tools, max_tokens=1024, tool_choice=None) -> LLMResponse:
        raw = self._post(self._payload(system, messages, tools, max_tokens, tool_choice))
        cand = (raw.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if "text" in p and not p.get("thought"))
        calls = []
        for p in parts:
            fc = p.get("functionCall")
            if fc:
                cid = f"call_{uuid.uuid4().hex[:10]}" + (SIG_SEP + p["thoughtSignature"] if p.get("thoughtSignature") else "")
                calls.append(ToolCall(cid, fc["name"], dict(fc.get("args") or {})))
        u = raw.get("usageMetadata") or {}
        usage = Usage(int(u.get("promptTokenCount", 0)), int(u.get("candidatesTokenCount", 0)) + int(u.get("thoughtsTokenCount", 0)))
        reason = "tool_use" if calls else {"STOP": "end_turn", "MAX_TOKENS": "max_tokens"}.get(cand.get("finishReason", "STOP"), cand.get("finishReason", "end_turn").lower())
        if not parts and (raw.get("promptFeedback") or {}).get("blockReason"):
            reason = "blocked:" + raw["promptFeedback"]["blockReason"].lower()
        return LLMResponse(text, calls, usage, self.name, reason)
