"""Indirect prompt-injection defense for tool results.

Three independent layers (any one of them is enough to stop the demo attack):
  1. sanitize: instruction-like sentences inside data fields are removed before the model sees them
  2. quarantine: results are wrapped in a <tool_output trust="untrusted"> envelope
  3. taint: a finding taints the session; tainted sessions need human approval for every high-risk action
"""
from __future__ import annotations

import json
import re
from typing import Any

from .patterns import injection_score

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
REMOVED = "[removed: suspected injected instruction]"
THRESHOLD = 0.5


def sanitize_text(text: str) -> tuple[str, list[dict]]:
    findings: list[dict] = []
    parts = SENTENCE_SPLIT.split(text)
    out = []
    for p in parts:
        score, labels = injection_score(p)
        if score >= THRESHOLD:
            findings.append({"score": score, "labels": labels, "snippet": p[:80]})
            out.append(REMOVED)
        else:
            out.append(p)
    return " ".join(out), findings


def _walk(node: Any, findings: list[dict]) -> Any:
    if isinstance(node, str):
        s, f = sanitize_text(node)
        findings.extend(f)
        return s
    if isinstance(node, list):
        return [_walk(x, findings) for x in node]
    if isinstance(node, dict):
        return {k: _walk(v, findings) for k, v in node.items()}
    return node


def sanitize_tool_output(raw: str) -> tuple[str, list[dict]]:
    findings: list[dict] = []
    try:
        return json.dumps(_walk(json.loads(raw), findings)), findings
    except json.JSONDecodeError:
        return sanitize_text(raw)


def wrap_untrusted(tool_name: str, body: str) -> str:
    body = body.replace("</tool_output>", "&lt;/tool_output&gt;")
    return f'<tool_output tool="{tool_name}" trust="untrusted">\n{body}\n</tool_output>'
