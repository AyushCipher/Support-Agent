"""Model layer for the tau-bench runs: one cached, budgeted Gemini wrapper used by ALL three roles
(supportagent's harness, tau-bench's plain ToolCallingAgent, tau-bench's LLM user simulator).

Why a `litellm` stand-in: tau-bench calls `litellm.completion(...)` from its user simulator and ToolCallingAgent. Installing litellm
(and its transitive deps) is not needed to run those two files; `install_litellm_shim()` registers a tiny module exposing
`completion()` that converts OpenAI-format messages/tools to supportagent's `Message`/`ToolSpec` and calls supportagent's stdlib
Gemini client (llm/gemini_client.py), which already handles Gemini-3 thought signatures, backoff and daily-quota detection.
tau-bench's code runs unmodified on top of it.

Robustness: disk cache keyed by (role, model, run-key, prompt, messages, tools) so an interrupted task replays for free; a global
network-request cap; per-task agent model-call cap (tau-bench's max_num_steps=30, applied to both agents); request/token logs.
The API key is read by GeminiLLM from the environment / git-ignored .env and is never logged here.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Callable

from supportagent.harness.types import LLMResponse, Message, ToolCall, ToolSpec, Usage

MAX_OUTPUT_TOKENS = 4096        # same for every role; Gemini thinking tokens count against maxOutputTokens
AGENT_CALL_CAP = 30             # tau-bench ToolCallingAgent default max_num_steps


class RunStop(Exception):
    """Global request cap reached or daily quota exhausted: stop the whole run cleanly (partial results are kept)."""


class StepCap(Exception):
    """This task used its agent model-call budget without the episode ending (scored 0, like tau-bench's step limit)."""


class Budget:
    def __init__(self, max_requests: int | None):
        self.max_requests = max_requests
        self.requests = self.cache_hits = 0
        self.tokens = {"in": 0, "out": 0}
        self.lock = threading.Lock()

    def check(self) -> None:
        with self.lock:
            if self.max_requests is not None and self.requests >= self.max_requests:
                raise RunStop(f"--max-requests {self.max_requests} reached")

    def add(self, n: int, usage: Usage | None = None) -> None:
        with self.lock:
            self.requests += n
            if usage:
                self.tokens["in"] += usage.input_tokens
                self.tokens["out"] += usage.output_tokens


class TaskContext(threading.local):
    """Per-thread (= per running task) state: cache namespace and the agent call counter."""
    run_key: str = ""
    agent_calls: int = 0
    call_cap: int = AGENT_CALL_CAP
    calls_by_model: dict | None = None


CTX = TaskContext()


def _cache_key(role: str, model: str, thinking: str | None, system: str, messages: list[Message], tools: list[ToolSpec], max_tokens: int,
               tool_choice: str | None) -> str:
    blob = json.dumps({"r": role, "m": model, "th": thinking, "k": CTX.run_key, "sys": system, "msgs": [m.to_dict() for m in messages],
                       "tools": [(t.name, t.public_schema()) for t in tools], "mt": max_tokens, "tc": tool_choice}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class CachedGemini:
    """Implements supportagent's `LLM` protocol. role='agent' calls count toward the per-task cap; role='user' (simulator) do not."""

    def __init__(self, model: str, role: str, budget: Budget, cache_dir: Path, rpm: float = 0.0, log: Callable[[dict], None] | None = None,
                 inner_factory: Callable[[str], Any] | None = None, thinking: str | None = None):
        self.name, self.role, self.budget, self.cache_dir, self.rpm, self.log = model, role, budget, cache_dir, rpm, log
        self.thinking = thinking                                  # Gemini thinking level for this role (None = model default)
        self.inner_factory = inner_factory or self._gemini

    def _gemini(self, model: str):
        from supportagent.llm.gemini_client import GeminiLLM
        return GeminiLLM(model, rpm=self.rpm, max_retries=6, backoff_base=2.0, backoff_cap=60.0, thinking=self.thinking or "")

    def complete(self, system, messages, tools, max_tokens=1024, tool_choice=None) -> LLMResponse:
        max_tokens = max(max_tokens, MAX_OUTPUT_TOKENS)
        if self.role == "agent":
            CTX.agent_calls += 1
            if CTX.agent_calls > CTX.call_cap:
                raise StepCap(f"agent used {CTX.call_cap} model calls")
            if CTX.calls_by_model is not None:
                CTX.calls_by_model[self.name] = CTX.calls_by_model.get(self.name, 0) + 1
        key = _cache_key(self.role, self.name, self.thinking, system, messages, tools, max_tokens, tool_choice)
        f = self.cache_dir / key[:2] / f"{key}.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            with self.budget.lock:
                self.budget.cache_hits += 1
            return LLMResponse(d["text"], [ToolCall(c["id"], c["name"], c["arguments"]) for c in d["calls"]], Usage(*d["usage"]), d["model"], d["stop"])
        self.budget.check()
        inner = self.inner_factory(self.name)
        t0 = time.time()
        try:
            resp = inner.complete(system, messages, tools, max_tokens, tool_choice)
        except Exception as e:
            self.budget.add(max(1, getattr(inner, "requests", 1)))
            if type(e).__name__ == "QuotaExhausted":
                raise RunStop(f"daily quota exhausted for {self.name}") from None
            raise
        n = max(1, getattr(inner, "requests", 1))
        self.budget.add(n, resp.usage)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"text": resp.text, "calls": [c.to_dict() for c in resp.tool_calls],
                                   "usage": [resp.usage.input_tokens, resp.usage.output_tokens], "model": resp.model, "stop": resp.stop_reason}),
                       encoding="utf-8")
        try:
            tmp.replace(f)
        except PermissionError:                                   # OneDrive/Windows: another writer won the race; content is equivalent
            tmp.unlink(missing_ok=True)
        if self.log:
            self.log({"ts": round(time.time(), 2), "run_key": CTX.run_key, "role": self.role, "model": self.name, "http_requests": n,
                      "in_tok": resp.usage.input_tokens, "out_tok": resp.usage.output_tokens, "latency_s": round(time.time() - t0, 2),
                      "stop": resp.stop_reason})
        return resp


# ------------------------------------------------------------------------------------------------ litellm stand-in
def openai_to_messages(messages: list[dict]) -> tuple[str, list[Message]]:
    system = "\n\n".join(m.get("content") or "" for m in messages if m.get("role") == "system")
    out: list[Message] = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            out.append(Message("user", m.get("content") or "(empty message)"))
        elif role == "assistant":
            calls = [ToolCall(tc["id"], tc["function"]["name"], json.loads(tc["function"].get("arguments") or "{}"))
                     for tc in (m.get("tool_calls") or [])]
            out.append(Message("assistant", m.get("content") or "", calls))
        elif role == "tool":
            out.append(Message("tool", m.get("content") or "", tool_call_id=m.get("tool_call_id"), tool_name=m.get("name")))
    return system, out


def openai_tools_to_specs(tools: list[dict] | None) -> list[ToolSpec]:
    return [ToolSpec(t["function"]["name"], t["function"].get("description", ""), t["function"].get("parameters") or {"type": "object", "properties": {}})
            for t in (tools or [])]


class _Msg:
    def __init__(self, resp: LLMResponse):
        self.tool_calls = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                           for c in resp.tool_calls] or None
        # an empty model turn becomes "" (not None): tau-bench's reward code calls .lower() on respond content and crashes on None
        self.content = resp.text or (None if self.tool_calls else "")
        self.role = "assistant"

    def model_dump(self) -> dict:
        return {"content": self.content, "role": self.role, "tool_calls": self.tool_calls, "function_call": None}


class _Choice:
    def __init__(self, resp: LLMResponse):
        self.message = _Msg(resp)
        self.finish_reason = resp.stop_reason


class _Response:
    def __init__(self, resp: LLMResponse, cost: float):
        self.choices = [_Choice(resp)]
        self._hidden_params = {"response_cost": cost}
        self.usage = resp.usage


class LiteLLMShim:
    """`completion()` compatible with how tau-bench calls litellm. Calls WITH tools come from the agent under test, calls WITHOUT tools
    from the user simulator (tau-bench's user simulator never passes tools)."""

    def __init__(self, make_llm: Callable[[str, str], CachedGemini]):
        self.make_llm = make_llm

    def completion(self, model: str, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.0,
                   custom_llm_provider: str | None = None, **_: Any) -> _Response:
        role = "agent" if tools else "user"
        system, msgs = openai_to_messages(messages)
        resp = self.make_llm(model, role).complete(system, msgs, openai_tools_to_specs(tools), MAX_OUTPUT_TOKENS)
        return _Response(resp, 0.0)


def install_litellm_shim(shim: LiteLLMShim) -> None:
    mod = types.ModuleType("litellm")
    mod.completion = shim.completion                       # type: ignore[attr-defined]
    mod.__dict__["__shim__"] = True
    sys.modules["litellm"] = mod
    for name in ("tau_bench.envs.user", "tau_bench.agents.tool_calling_agent"):   # already imported -> rebind their module-level name
        if name in sys.modules:
            sys.modules[name].completion = shim.completion  # type: ignore[attr-defined]
