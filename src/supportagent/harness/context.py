"""Context management: smart tool-result truncation, token estimation, and compaction with pinned facts.

Compaction invariants: never split an assistant tool_call from its tool results (cut only at a user-message boundary),
always keep the last N messages verbatim, always keep pinned facts (actions already taken) outside the summary so a
lossy summary can never make the agent forget that a refund already happened.
"""
from __future__ import annotations

import json
from typing import Any

from ..config import AgentConfig
from .types import Message

LONG_FIELD = 400


def est_tokens(text: str) -> int:
    return len(text) // 4 + 1


def _shrink(node: Any) -> Any:
    if isinstance(node, str) and len(node) > LONG_FIELD:
        return node[:LONG_FIELD - 80] + f" …[{len(node) - LONG_FIELD + 40} chars truncated]… " + node[-40:]
    if isinstance(node, list):
        return [_shrink(x) for x in node]
    if isinstance(node, dict):
        return {k: _shrink(v) for k, v in node.items()}
    return node


def truncate_tool_result(raw: str, max_chars: int) -> str:
    """Shrink long string fields first (keeps JSON valid), then hard-cap with head+tail."""
    if len(raw) <= max_chars:
        return raw
    try:
        raw = json.dumps(_shrink(json.loads(raw)))
    except json.JSONDecodeError:
        pass
    if len(raw) <= max_chars:
        return raw
    head, tail = int(max_chars * 0.7), int(max_chars * 0.2)
    return raw[:head] + f"\n…[{len(raw) - head - tail} chars truncated]…\n" + raw[-tail:]


def extractive_summary(msgs: list[Message]) -> str:
    lines = []
    for m in msgs:
        if m.role == "user" and not m.content.startswith("["):
            lines.append(f"- Customer said: {m.content[:160]}")
        elif m.role == "assistant":
            for tc in m.tool_calls:
                lines.append(f"- Agent called {tc.name}({json.dumps(tc.arguments)[:100]})")
            if m.content and not m.tool_calls:
                lines.append(f"- Agent replied: {m.content[:160]}")
    return "\n".join(lines[-30:])


class ContextManager:
    def __init__(self, cfg: AgentConfig, summarizer=None):
        self.cfg, self.summarizer = cfg, summarizer or extractive_summary

    def tokens(self, system: str, msgs: list[Message]) -> int:
        return est_tokens(system) + sum(est_tokens(m.content) + sum(est_tokens(json.dumps(t.arguments)) for t in m.tool_calls) for m in msgs)

    def prepare(self, session, system: str) -> list[Message]:
        if self.tokens(system, session.messages) > self.cfg.context_budget_tokens:
            self._compact(session)
        return session.messages

    def _compact(self, session) -> None:
        msgs = session.messages
        keep_from = max(0, len(msgs) - self.cfg.keep_last_messages)
        while keep_from < len(msgs) and not (msgs[keep_from].role == "user" and not msgs[keep_from].content.startswith("[")):
            keep_from += 1                                   # snap forward to a real user-turn boundary
        if keep_from <= 0 or keep_from >= len(msgs):
            return
        old = msgs[:keep_from]
        session.summary = (session.summary + "\n" if session.summary else "") + self.summarizer(old)
        note = Message("user", "[CONTEXT SUMMARY of earlier conversation]\n" + session.summary)
        session.messages = [note] + msgs[keep_from:]
        session.events.append({"type": "context_compacted", "dropped": len(old), "kept": len(session.messages)})
