"""Anthropic Messages API adapter (tool use). Install: pip install 'supportagent[anthropic]'.

Not exercised by the offline test-suite against the live API (no key in CI) - the message/tool conversion is unit-tested
in tests/test_anthropic_adapter.py with a fake client, and `make eval-live` runs the full suite against real models.
"""
from __future__ import annotations

import os

from ..harness.types import LLMResponse, Message, ToolCall, ToolSpec, Usage
from .base import TransientLLMError


def to_api_messages(messages: list[Message]) -> list[dict]:
    out: list[dict] = []

    def push(role: str, blocks: list[dict]) -> None:
        if out and out[-1]["role"] == role:               # the API requires alternating roles: merge neighbours
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for m in messages:
        if m.role == "user":
            push("user", [{"type": "text", "text": m.content}])
        elif m.role == "assistant":
            blocks = [{"type": "text", "text": m.content}] if m.content else []
            blocks += [{"type": "tool_use", "id": t.id, "name": t.name, "input": t.arguments} for t in m.tool_calls]
            push("assistant", blocks or [{"type": "text", "text": "(no content)"}])
        elif m.role == "tool":
            push("user", [{"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content, "is_error": m.is_error}])
    return out


def to_api_tools(tools: list[ToolSpec]) -> list[dict]:
    return [{"name": t.name, "description": t.description, "input_schema": t.public_schema()} for t in tools]


class AnthropicLLM:
    def __init__(self, model: str, client=None, timeout: float = 60.0):
        self.name = model
        if client is None:
            import anthropic  # lazy: optional dependency
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=timeout)
        self.client = client

    def complete(self, system, messages, tools, max_tokens=1024, tool_choice=None) -> LLMResponse:
        kwargs = dict(model=self.name, max_tokens=max_tokens,
                      system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                      messages=to_api_messages(messages))
        if tools:
            kwargs["tools"] = to_api_tools(tools)
            if tool_choice:
                kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}
        try:
            r = self.client.messages.create(**kwargs)
        except Exception as e:                                  # map provider errors to retryable/non-retryable
            status = getattr(e, "status_code", None)
            if status in (408, 409, 429) or (status or 0) >= 500 or type(e).__name__ in ("APIConnectionError", "APITimeoutError"):
                raise TransientLLMError(str(e)) from e
            raise
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"]
        return LLMResponse(text, calls, Usage(r.usage.input_tokens, r.usage.output_tokens), self.name, r.stop_reason)
