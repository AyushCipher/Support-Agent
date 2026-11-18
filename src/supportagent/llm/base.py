from __future__ import annotations

from typing import Protocol

from ..harness.types import LLMResponse, Message, ToolSpec


class TransientLLMError(Exception):
    """Retryable provider error (rate limit, 5xx, timeout)."""


class LLM(Protocol):
    name: str

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec], max_tokens: int = 1024,
                 tool_choice: str | None = None) -> LLMResponse: ...
