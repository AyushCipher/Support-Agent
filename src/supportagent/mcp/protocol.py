"""Minimal, spec-shaped MCP server (JSON-RPC 2.0, newline-delimited stdio transport).

Implements: initialize, ping, tools/list, tools/call, resources/list, resources/read, prompts/list, prompts/get.
Written from scratch (stdlib only) so the project runs anywhere and the protocol is visible; swap in the
official SDK's FastMCP if you prefer - the tool functions are plain callables.

Auth model: every request carries `params._meta.token`. The server maps token -> scopes (MCP_TOKENS env, JSON)
and enforces the scope each tool declares. For HTTP transports you would replace this with OAuth 2.1 bearer
tokens (see docs/adr/0003).
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Callable

from ..security.identity import IdentityService, TokenError

PROTOCOL_VERSION = "2025-06-18"


class ToolError(Exception):
    def __init__(self, code: str, message: str, **extra: Any):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra

    def payload(self) -> dict:
        return {"error": self.code, "message": self.message, **self.extra}


class MCPServer:
    def __init__(self, name: str, version: str = "1.0.0", tokens: dict[str, list[str]] | None = None,
                 require_auth: bool | None = None, jwt_secret: str | None = None):
        self.name, self.version = name, version
        self._tools: dict[str, dict] = {}
        self._resources: dict[str, dict] = {}
        self._prompts: dict[str, dict] = {}
        self.tokens = tokens if tokens is not None else json.loads(os.getenv("MCP_TOKENS", "{}"))
        self.require_auth = require_auth if require_auth is not None else os.getenv("MCP_REQUIRE_AUTH", "1") == "1"
        self.identity = self._resolve_identity(jwt_secret)             # JWT mode: signed, audience-bound, replay-protected
        self._replay = None

    @staticmethod
    def _resolve_identity(credential: Any):
        """What this server verifies tokens with. `credential` may be:
          * a verifier object (JwksVerifier: PUBLIC keys only - this server can verify but can never mint)  [production]
          * a shared-secret string (HS256 dev fallback - any verifier can also mint)
          * None: look at MCP_JWKS / MCP_JWKS_FILE / MCP_JWKS_URL (public keys), then MCP_JWT_SECRET (dev).
        """
        if credential is not None and not isinstance(credential, (str, bytes)):
            return credential
        if credential:
            return IdentityService(credential)
        from ..security.crypto import production_mode
        if os.getenv("MCP_JWKS") or os.getenv("MCP_JWKS_FILE") or os.getenv("MCP_JWKS_URL"):
            from ..security.asymmetric import verifier_from_env
            return verifier_from_env()
        if os.getenv("MCP_JWT_SECRET"):
            if production_mode():
                raise RuntimeError("MCP_JWT_SECRET (HS256 shared secret) is refused when SA_ENV=production; configure MCP_JWKS_URL/FILE")
            return IdentityService(os.environ["MCP_JWT_SECRET"])
        return None

    # ---- registration -------------------------------------------------------------------------
    def tool(self, name: str, description: str, schema: dict, scope: str, risk: str = "read", hidden: tuple[str, ...] = ()):
        def deco(fn: Callable[[dict], Any]):
            self._tools[name] = {"fn": fn, "description": description, "schema": schema, "scope": scope,
                                 "risk": risk, "hidden": list(hidden)}
            return fn
        return deco

    def resource(self, uri: str, name: str, description: str, reader: Callable[[], str], mime: str = "text/markdown"):
        self._resources[uri] = {"name": name, "description": description, "reader": reader, "mime": mime}

    def prompt(self, name: str, description: str, arguments: list[dict], render: Callable[[dict], str]):
        self._prompts[name] = {"description": description, "arguments": arguments, "render": render}

    # ---- dispatch -----------------------------------------------------------------------------
    def handle(self, req: dict) -> dict | None:
        rid, method, params = req.get("id"), req.get("method"), req.get("params") or {}
        if rid is None:                     # notification
            return None
        try:
            result = self._dispatch(method, params)
            return {"jsonrpc": "2.0", "id": rid, "result": result}
        except KeyError as e:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"not found: {e}"}}
        except ValueError as e:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": str(e)}}
        except Exception as e:  # pragma: no cover
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": f"internal error: {e}"}}

    def _auth(self, params: dict, required_scope: str) -> tuple[bool, dict | None, str]:
        """-> (allowed, identity or None, error). JWT mode verifies signature/audience/expiry/replay; static mode maps token->scopes."""
        token = (params.get("_meta") or {}).get("token")
        ident = None
        if self.identity:
            try:
                claims = self.identity.verify(token, audience=self.name)
            except TokenError as e:
                return False, None, f"invalid identity: {e}"
            if self._replay is None:
                from ..security.asymmetric import ReplayGuard
                self._replay = ReplayGuard()
            if not claims.get("jti") or not self._replay.check_and_add(claims["jti"], claims["exp"]):
                return False, None, "invalid identity: token replay"
            scopes = set(claims.get("scope", "").split())
            ident = {"sub": claims["sub"], "sid": claims["sid"], "jti": claims["jti"]}
        else:
            scopes = set(self.tokens.get(token, []))
        if self.require_auth and required_scope not in scopes:
            return False, ident, f"missing scope {required_scope}"
        return True, ident, ""

    def _dispatch(self, method: str, params: dict) -> dict:
        if method == "initialize":
            return {"protocolVersion": PROTOCOL_VERSION, "serverInfo": {"name": self.name, "version": self.version},
                    "capabilities": {"tools": {}, "resources": {}, "prompts": {}}}
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [{"name": n, "description": t["description"], "inputSchema": t["schema"],
                               "annotations": {"risk": t["risk"], "requiredScope": t["scope"], "hiddenParams": t["hidden"]}}
                              for n, t in self._tools.items()]}
        if method == "tools/call":
            name, args = params.get("name"), params.get("arguments") or {}
            if name not in self._tools:
                raise ValueError(f"unknown tool {name}")
            t = self._tools[name]
            ok, ident, why = self._auth(params, t["scope"])
            if not ok:
                return self._tool_result({"error": "forbidden", "message": why}, True)
            args = {k: v for k, v in args.items() if k != "_identity"}       # clients can never supply their own identity
            if ident:
                args["_identity"] = ident                                    # server-verified: used for audit attribution
            try:
                return self._tool_result(t["fn"](args), False)
            except ToolError as e:
                return self._tool_result(e.payload(), True)
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                return self._tool_result({"error": "internal_error", "message": str(e)[:200]}, True)
        if method == "resources/list":
            return {"resources": [{"uri": u, "name": r["name"], "description": r["description"], "mimeType": r["mime"]}
                                  for u, r in self._resources.items()]}
        if method == "resources/read":
            uri = params.get("uri")
            r = self._resources[uri]
            return {"contents": [{"uri": uri, "mimeType": r["mime"], "text": r["reader"]()}]}
        if method == "prompts/list":
            return {"prompts": [{"name": n, "description": p["description"], "arguments": p["arguments"]}
                                for n, p in self._prompts.items()]}
        if method == "prompts/get":
            p = self._prompts[params.get("name")]
            return {"description": p["description"], "messages": [
                {"role": "user", "content": {"type": "text", "text": p["render"](params.get("arguments") or {})}}]}
        raise KeyError(method)

    @staticmethod
    def _tool_result(payload: Any, is_error: bool) -> dict:
        return {"content": [{"type": "text", "text": json.dumps(payload, default=str)}], "isError": is_error}

    # ---- stdio transport ----------------------------------------------------------------------
    def run_stdio(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = self.handle(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
