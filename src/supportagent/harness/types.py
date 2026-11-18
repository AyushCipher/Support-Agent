from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @staticmethod
    def from_dict(d: dict) -> "ToolCall":
        return ToolCall(d["id"], d["name"], d.get("arguments", {}))


@dataclass
class Message:
    role: str                       # user | assistant | tool
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    is_error: bool = False

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content,
                "tool_calls": [t.to_dict() for t in self.tool_calls],
                "tool_call_id": self.tool_call_id, "tool_name": self.tool_name, "is_error": self.is_error}

    @staticmethod
    def from_dict(d: dict) -> "Message":
        return Message(d["role"], d.get("content", ""), [ToolCall.from_dict(t) for t in d.get("tool_calls", [])],
                       d.get("tool_call_id"), d.get("tool_name"), d.get("is_error", False))


@dataclass
class ToolSpec:
    name: str                       # namespaced: server__tool
    description: str
    input_schema: dict
    risk: str = "read"              # read | write | high
    server: str = "local"
    hidden_params: tuple[str, ...] = ()

    def public_schema(self) -> dict:
        """Schema shown to the model: injected/hidden params are removed (the model cannot set them)."""
        s = json.loads(json.dumps(self.input_schema))
        for h in self.hidden_params:
            s.get("properties", {}).pop(h, None)
            if h in s.get("required", []):
                s["required"].remove(h)
        s["additionalProperties"] = False
        return s


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    model: str
    stop_reason: str = "end_turn"


def stable_hash(*parts: Any, n: int = 12) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:n]
