"""Customer-facing authentication for /api/chat and /api/sessions/<id> (closes threat-model gap G2).

Before: /api/chat trusted `customer_id` from the request body AND accepted any `session_id`, so anyone who learned a session id
could continue another customer's conversation (acting with that customer's authorization) and read it via /api/sessions/<id>.

Two layers:
  1. Session-bound token. Creating a session returns `session_token` = HMAC-SHA256(SA_SESSION_SECRET, "v1|<session id>|<customer id>").
     Every later /api/chat on that session and every GET /api/sessions/<id> must present it (`X-Session-Token` header, or
     `session_token` in the chat body). Compared in constant time. Stateless, survives restarts when SA_SESSION_SECRET is set
     (dev without it: a random per-process secret, so tokens die with the process). The session's own customer id is bound into
     the MAC, so a token cannot be replayed against a session of another customer.
  2. Customer identity at session creation (optional, recommended for production): an OIDC/JWT bearer from the host site's login,
     verified against SA_CUSTOMER_OIDC_JWKS + SA_CUSTOMER_OIDC_ISSUER + SA_CUSTOMER_OIDC_AUDIENCE. The token's `customer_id`
     claim (or `sub`) must equal the requested customer_id. Without it the demo's customer picker is trusted, which production
     startup refuses unless SA_PUBLIC_DEMO=1 is set explicitly (see security/prodcheck.py).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets

from .adminauth import bearer
from .identity import TokenError

_DEV_SECRET = secrets.token_bytes(32)
PREFIX = "st1."


class CustomerAuthError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _secret() -> bytes:
    s = os.getenv("SA_SESSION_SECRET", "")
    return s.encode() if s else _DEV_SECRET


def issue(session_id: str, customer_id: str) -> str:
    mac = hmac.new(_secret(), f"v1|{session_id}|{customer_id}".encode(), hashlib.sha256).digest()
    return PREFIX + base64.urlsafe_b64encode(mac).rstrip(b"=").decode()


def check(token: str | None, session_id: str, customer_id: str) -> bool:
    if not token or not isinstance(token, str):
        return False
    return hmac.compare_digest(token.encode(), issue(session_id, customer_id).encode())


_VERIFIERS: dict[tuple, object] = {}


def oidc_configured() -> bool:
    return all(os.getenv(k) for k in ("SA_CUSTOMER_OIDC_JWKS", "SA_CUSTOMER_OIDC_ISSUER", "SA_CUSTOMER_OIDC_AUDIENCE"))


def authenticate_customer(headers, customer_id: str) -> None:
    """Session creation: when customer OIDC is configured the bearer token must prove `customer_id`. No-op otherwise (demo)."""
    if not oidc_configured():
        return
    tok = bearer(headers)
    if not tok:
        raise CustomerAuthError(401, "customer authentication required")
    key = (os.environ["SA_CUSTOMER_OIDC_JWKS"], os.environ["SA_CUSTOMER_OIDC_ISSUER"])
    if key not in _VERIFIERS:
        from .asymmetric import JwksVerifier
        _VERIFIERS[key] = JwksVerifier(key[0], issuer=key[1], allowed_algs=("RS256", "EdDSA"))
    try:
        claims = _VERIFIERS[key].verify(tok, os.environ["SA_CUSTOMER_OIDC_AUDIENCE"], max_ttl=None)
    except TokenError:
        raise CustomerAuthError(401, "customer authentication required") from None
    if str(claims.get("customer_id") or claims.get("sub") or "") != customer_id:
        raise CustomerAuthError(403, "token does not belong to this customer")
