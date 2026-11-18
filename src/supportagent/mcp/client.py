"""MCP client: stdio + in-process transports, timeouts, one-shot reconnect, circuit breaker, registry."""
from __future__ import annotations

import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..harness.types import ToolSpec
from .protocol import MCPServer


class MCPError(Exception): ...
class MCPTimeout(MCPError): ...
class MCPDown(MCPError): ...
class MCPUnavailable(MCPError): ...


class InProcessTransport:
    """Same JSON-RPC dispatch as stdio, without a subprocess (fast evals/tests). JSON round-trip enforced."""
    def __init__(self, server: MCPServer):
        self.server = server

    def request(self, msg: dict, timeout: float) -> dict | None:
        resp = self.server.handle(json.loads(json.dumps(msg)))
        return json.loads(json.dumps(resp)) if resp is not None else None

    def notify(self, msg: dict) -> None: ...
    def close(self) -> None: ...


class StdioTransport:
    def __init__(self, command: list[str], env: dict[str, str] | None = None, cwd: str | None = None):
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1, env={**os.environ, **(env or {})}, cwd=cwd)
        self._pending: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        self._dead = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        for line in self.proc.stdout:  # type: ignore[union-attr]
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = self._pending.get(obj.get("id"))
            if q:
                q.put(obj)
        self._dead = True
        for q in list(self._pending.values()):
            q.put(None)

    def request(self, msg: dict, timeout: float) -> dict | None:
        if self._dead or self.proc.poll() is not None:
            raise MCPDown("server process exited")
        q: queue.Queue = queue.Queue()
        self._pending[msg["id"]] = q
        try:
            with self._lock:
                self.proc.stdin.write(json.dumps(msg) + "\n")  # type: ignore[union-attr]
                self.proc.stdin.flush()  # type: ignore[union-attr]
            try:
                resp = q.get(timeout=timeout)
            except queue.Empty:
                raise MCPTimeout(f"no response in {timeout}s") from None
            if resp is None:
                raise MCPDown("server closed the pipe")
            return resp
        except (BrokenPipeError, OSError) as e:
            raise MCPDown(str(e)) from e
        finally:
            self._pending.pop(msg["id"], None)

    def notify(self, msg: dict) -> None:
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")  # type: ignore[union-attr]
            self.proc.stdin.flush()  # type: ignore[union-attr]
        except OSError:
            pass

    def close(self) -> None:
        try:
            self.proc.kill()
        except Exception:
            pass


@dataclass
class _Breaker:
    threshold: int = 3
    cooldown_s: float = 20.0
    failures: int = 0
    opened_at: float = 0.0

    def allow(self) -> bool:
        if self.failures < self.threshold:
            return True
        if time.time() - self.opened_at > self.cooldown_s:      # half-open
            self.failures = self.threshold - 1
            return True
        return False

    def ok(self) -> None:
        self.failures = 0

    def fail(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.time()


class MCPClient:
    def __init__(self, name: str, factory: Callable[[], Any], timeout: float = 10.0):
        self.name, self._factory, self.timeout = name, factory, timeout
        self._ids = itertools.count(1)
        self.transport: Any = None
        self.breaker = _Breaker()
        self.tools: list[dict] = []

    def connect(self) -> None:
        self.transport = self._factory()
        self._rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                 "clientInfo": {"name": "supportagent", "version": "1.0.0"}}, reconnect=False)
        self.transport.notify({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.tools = self._rpc("tools/list", {}, reconnect=False)["tools"]

    def _rpc(self, method: str, params: dict, reconnect: bool = True, timeout: float | None = None) -> dict:
        if not self.breaker.allow():
            raise MCPUnavailable(f"{self.name}: circuit open")
        msg = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params}
        for attempt in (0, 1):
            try:
                resp = self.transport.request(msg, timeout or self.timeout)
                self.breaker.ok()
                if resp and "error" in resp:
                    raise MCPError(resp["error"].get("message", "rpc error"))
                return resp["result"]  # type: ignore[index]
            except (MCPDown, MCPTimeout) as e:
                self.breaker.fail()
                if attempt == 0 and reconnect and isinstance(e, MCPDown):
                    try:
                        self.connect()           # one restart attempt
                        msg["id"] = next(self._ids)
                        continue
                    except Exception:
                        pass
                raise
        raise MCPDown("unreachable")  # pragma: no cover

    def call_tool(self, name: str, args: dict, token: str | None, timeout: float | None = None) -> tuple[Any, bool]:
        res = self._rpc("tools/call", {"name": name, "arguments": args, "_meta": {"token": token}}, timeout=timeout)
        text = res["content"][0]["text"]
        try:
            return json.loads(text), bool(res.get("isError"))
        except json.JSONDecodeError:
            return text, bool(res.get("isError"))

    def list_resources(self) -> list[dict]:
        return self._rpc("resources/list", {})["resources"]

    def read_resource(self, uri: str) -> str:
        return self._rpc("resources/read", {"uri": uri})["contents"][0]["text"]

    def list_prompts(self) -> list[dict]:
        return self._rpc("prompts/list", {})["prompts"]

    def get_prompt(self, name: str, arguments: dict | None = None) -> str:
        return self._rpc("prompts/get", {"name": name, "arguments": arguments or {}})["messages"][0]["content"]["text"]

    def close(self) -> None:
        if self.transport:
            self.transport.close()


class MCPRegistry:
    """Aggregates servers, namespaces tools as `server__tool`, degrades gracefully when a server is down."""
    FIRST_PARTY = ("orders", "kb", "tickets")

    def __init__(self, clients: list[MCPClient], role_tokens: dict[str, dict[str, str]] | None = None, identity=None):
        self.clients = {c.name: c for c in clients}
        self.role_tokens = role_tokens or {}
        self.identity = identity
        self.degraded: dict[str, str] = {}
        self._specs: dict[str, ToolSpec] = {}

    def load(self) -> "MCPRegistry":
        for name, c in self.clients.items():
            try:
                c.connect()
            except Exception as e:                      # server down at startup: continue without it
                self.degraded[name] = f"{type(e).__name__}: {e}"
                continue
            for t in c.tools:
                ann = t.get("annotations", {})
                full = f"{name}__{t['name']}"
                self._specs[full] = ToolSpec(full, t["description"], t["inputSchema"], ann.get("risk", "read"), name,
                                             tuple(ann.get("hiddenParams", [])))
        return self

    def specs(self) -> dict[str, ToolSpec]:
        return self._specs

    def call(self, full_name: str, args: dict, role: str = "system", timeout: float | None = None, session_id: str | None = None) -> tuple[Any, bool]:
        server, tool = full_name.split("__", 1)
        client = self.clients.get(server)
        if client is None or server in self.degraded and not client.tools:
            return {"error": "tool_unavailable", "message": f"{server} server is unavailable"}, True
        if self.identity and server in self.FIRST_PARTY:        # per-call, audience-bound, 60s agent identity
            token = self.identity.issue(role, session_id or "-", SCOPES_BY_ROLE.get(role, []), audience=server)
        else:
            token = self.role_tokens.get(role, {}).get(server)
        try:
            return client.call_tool(tool, args, token, timeout)
        except (MCPError, OSError) as e:
            return {"error": "tool_unavailable", "message": f"{server} failed: {type(e).__name__}"}, True

    def close(self) -> None:
        for c in self.clients.values():
            c.close()


# ---- wiring ------------------------------------------------------------------------------------
SCOPES_BY_ROLE = {
    "triage": [],
    "orders": ["orders:read", "tickets:read", "kb:read"],
    "billing": ["orders:read", "subscriptions:write", "refunds:issue", "tickets:read", "tickets:write", "kb:read"],
    "policy": ["kb:read"],
    "system": ["orders:read", "subscriptions:write", "refunds:issue", "tickets:read", "tickets:write", "kb:read"],
}


def build_registry(db_path: str, allow_anon: bool = False, mode: str = "inprocess", extra: list[MCPClient] | None = None,
                   jwt_secret: str | None = None) -> MCPRegistry:
    from ..security.asymmetric import make_identity
    from .servers import kb_server, orders_server, tickets_server
    ident = make_identity(jwt_secret)        # EdDSA+JWKS by default (servers get PUBLIC keys only); HS256 only as an explicit dev fallback
    secret = ident.server_credential

    tokens = {f"tok-{role}": scopes for role, scopes in SCOPES_BY_ROLE.items()}
    role_tokens = {role: {s: f"tok-{role}" for s in ("orders", "kb", "tickets")} for role in SCOPES_BY_ROLE}
    if mode == "inprocess":
        servers = {"orders": orders_server.make_server(db_path, allow_anon, tokens, secret), "kb": kb_server.make_server(tokens, secret),
                   "tickets": tickets_server.make_server(db_path, allow_anon, tokens, secret)}
        clients = [MCPClient(n, (lambda s=s: InProcessTransport(s))) for n, s in servers.items()]
    else:
        env = {"SHOP_DB": db_path, "MCP_TOKENS": json.dumps(tokens), "SA_ALLOW_ANON": "1" if allow_anon else "0", **ident.server_env()}
        # the child must import THIS supportagent package even when it is not pip-installed (found by the review: the stdio
        # parity test failed unless the caller exported PYTHONPATH, and the servers died with "server closed the pipe")
        pkg_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join([pkg_root] + [x for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x])
        clients = [MCPClient(n, (lambda n=n: StdioTransport([sys.executable, "-m", f"supportagent.mcp.servers.{n}_server"], env=env)))
                   for n in ("orders", "kb", "tickets")]
    if os.getenv("SA_EXTRA_MCP"):
        from . import third_party
        extra = (extra or []) + third_party.load(os.environ["SA_EXTRA_MCP"])
    reg = MCPRegistry(clients + (extra or []), role_tokens, ident.issuer).load()
    for sp in reg.specs().values():                      # unannotated (third-party) tools default to the risky tier
        if sp.server not in ("orders", "kb", "tickets") and sp.risk == "read" and not any(
                t.get("annotations", {}).get("risk") for t in reg.clients[sp.server].tools if f"{sp.server}__{t['name']}" == sp.name):
            sp.risk = "write"
    return reg
