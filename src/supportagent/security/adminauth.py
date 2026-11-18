"""Authentication for the admin API and the reviewer desk (approvals), ADR-0010.

Two credential types, both sent as `Authorization: Bearer <credential>` (never a cookie, so browsers never attach them
ambiently - the root of CSRF safety):
  1. SA_ADMIN_TOKEN        shared static token, compared in constant time (hmac.compare_digest). Principal: `admin`.
  2. OIDC-style JWT         verified with the same JWKS machinery as agent identities (RS256/EdDSA, kid, issuer, audience,
                            expiry; optional required role). Enabled by SA_OIDC_JWKS (URL or file) + SA_OIDC_ISSUER +
                            SA_OIDC_AUDIENCE. Principal: `oidc:<sub>` - the real human reviewer identity.
Everything is read from the environment per request, and endpoints are DISABLED (403) when neither method is configured.

CSRF defense-in-depth for state-changing requests (`check_csrf`): bearer-header auth (above); an `Origin` header, if present,
must match `Host` (a cross-site page cannot forge Host); the approval POST must be `application/json` (a cross-site HTML form
cannot send that without a CORS preflight, which this server never approves - it sends no CORS headers).
"""
from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from urllib.parse import urlparse

from .identity import TokenError


@dataclass
class Principal:
    name: str            # attribution string used as the reviewer / operator identity
    via: str             # "token" | "oidc"


class AuthError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code, self.message = code, message


_OIDC_CACHE: dict[tuple, object] = {}


def _oidc_verifier():
    src = os.getenv("SA_OIDC_JWKS", "")
    if not src or not os.getenv("SA_OIDC_ISSUER") or not os.getenv("SA_OIDC_AUDIENCE"):
        return None
    algs = tuple(a.strip() for a in os.getenv("SA_OIDC_ALGS", "RS256,EdDSA").split(",") if a.strip())
    key = (src, os.environ["SA_OIDC_ISSUER"], algs)
    if key not in _OIDC_CACHE:
        from .asymmetric import JwksVerifier
        _OIDC_CACHE[key] = JwksVerifier(src, issuer=os.environ["SA_OIDC_ISSUER"], allowed_algs=algs)
    return _OIDC_CACHE[key]


def bearer(headers) -> str:
    h = headers.get("Authorization") or ""
    return h[7:].strip() if h[:7].lower() == "bearer " else ""


def authenticate(headers) -> Principal:
    """Returns the authenticated Principal or raises AuthError(403 disabled | 401 unauthorized)."""
    static = os.getenv("SA_ADMIN_TOKEN", "")
    oidc = _oidc_verifier() if os.getenv("SA_OIDC_JWKS") else None
    if not static and oidc is None:
        raise AuthError(403, "admin endpoints disabled: set SA_ADMIN_TOKEN (or SA_OIDC_JWKS/ISSUER/AUDIENCE)")
    given = bearer(headers)
    if not given:
        raise AuthError(401, "unauthorized")
    if static and hmac.compare_digest(given.encode(), static.encode()):
        return Principal("admin", "token")
    if oidc is not None and given.count(".") == 2:
        try:
            claims = oidc.verify(given, os.environ["SA_OIDC_AUDIENCE"], max_ttl=None)
        except TokenError:
            raise AuthError(401, "unauthorized") from None
        role = os.getenv("SA_OIDC_ROLE")
        if role:
            have = set(claims.get("roles") or []) | set(claims.get("groups") or []) | set(str(claims.get("scope", "")).split())
            if role not in have:
                raise AuthError(403, "forbidden: missing required role")
        return Principal(f"oidc:{str(claims.get('sub', 'unknown'))[:60]}", "oidc")
    raise AuthError(401, "unauthorized")


def check_csrf(headers, require_json: bool = False) -> None:
    origin = headers.get("Origin")
    if origin and urlparse(origin).netloc != (headers.get("Host") or ""):
        raise AuthError(403, "cross-origin request refused")
    if require_json and not (headers.get("Content-Type") or "").lower().startswith("application/json"):
        raise AuthError(415, "Content-Type must be application/json")
