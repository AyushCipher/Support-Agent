"""Plug in third-party MCP servers from a JSON file (env SA_EXTRA_MCP=path).

Security defaults (fail safe):
  * tools from third-party servers are treated as risk="write" unless the server annotates otherwise,
  * they are NOT available to any agent until a role explicitly allowlists them in the config ("grant"),
  * results still pass through truncation + untrusted-output quarantine + injection sanitization like any tool output.
Example: mcp_servers.example.json (official filesystem server exposing the policy docs read-only to the policy agent).
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from ..harness import orchestrator
from .client import MCPClient, StdioTransport


def load(path: str | Path) -> list[MCPClient]:
    cfg = json.loads(Path(path).read_text())
    clients = []
    for s in cfg["servers"]:
        cmd = [s["command"], *s.get("args", [])]
        env = s.get("env", {})
        clients.append(MCPClient(s["name"], (lambda cmd=cmd, env=env: StdioTransport(cmd, env=env)), timeout=float(s.get("timeout_s", 20))))
        for role, tools in s.get("grant", {}).items():           # explicit least-privilege grants
            spec = orchestrator.SPECS[role]
            orchestrator.SPECS[role] = replace(spec, tools=tuple(dict.fromkeys(spec.tools + tuple(f"{s['name']}__{t}" for t in tools))))
    return clients
