"""Cryptographic agent identities: short-lived signed JWTs, one per (agent role, session, target server).

Claims: iss, sub=agent:<role>, sid=<session id>, aud=<server>, scope=<space-separated>, iat, exp, jti.
 * audience-bound: a token minted for the `kb` server is useless against `orders` (confused-deputy across servers)
 * short TTL (60 s) + unique jti: servers reject replays within the TTL
 * every first-party server writes `sub` and `sid` into the audit chain, so each action is attributable.

Two signing modes (ADR-0010, supersedes the trade-off in ADR-0007):
 * EdDSA/Ed25519 + JWKS (default; implemented in `asymmetric.py`, needs the optional `[crypto]` extra). The *issuer* holds private keys
   in a `KeyRing` with `kid`s and rotation with an overlap window; MCP servers hold a `JwksVerifier` that only ever sees
   PUBLIC keys, so a compromised server can verify but can never mint. Only EdDSA/RS256 are accepted by a JWKS verifier:
   HS256 tokens (algorithm confusion with the public key as an HMAC secret) and `alg:none` are rejected.
 * HS256 shared secret (`IdentityService`) - dev fallback only. Selected explicitly (SA_IDENTITY_ALG=hs256, SA_JWT_SECRET,
   or by passing a secret). Any verifier can also mint; never use in production.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid


class TokenError(Exception): ...


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class IdentityService:
    def __init__(self, secret: str | bytes | None = None, issuer: str = "supportagent"):
        secret = secret or os.getenv("SA_JWT_SECRET") or secrets.token_hex(32)
        self._key = secret.encode() if isinstance(secret, str) else secret
        self.issuer = issuer

    def _sign(self, signing_input: bytes) -> str:
        return _b64(hmac.new(self._key, signing_input, hashlib.sha256).digest())

    def issue(self, agent: str, session_id: str, scopes: list[str], audience: str, ttl: int = 60, now: float | None = None) -> str:
        now = int(now if now is not None else time.time())
        header = {"alg": "HS256", "typ": "JWT"}
        claims = {"iss": self.issuer, "sub": f"agent:{agent}", "sid": session_id or "-", "aud": audience, "scope": " ".join(scopes),
                  "iat": now, "exp": now + ttl, "jti": uuid.uuid4().hex}
        si = f"{_b64(json.dumps(header, separators=(',', ':')).encode())}.{_b64(json.dumps(claims, separators=(',', ':')).encode())}"
        return f"{si}.{self._sign(si.encode())}"

    def verify(self, token: str | None, audience: str, now: float | None = None) -> dict:
        if not token or token.count(".") != 2:
            raise TokenError("malformed token")
        h64, c64, sig = token.split(".")
        try:
            header = json.loads(_unb64(h64))
            claims = json.loads(_unb64(c64))
        except Exception:
            raise TokenError("malformed token") from None
        if header.get("alg") != "HS256":                     # rejects alg:none and algorithm-confusion attempts
            raise TokenError("unsupported alg")
        if not hmac.compare_digest(sig, self._sign(f"{h64}.{c64}".encode())):
            raise TokenError("bad signature")
        if claims.get("iss") != self.issuer:
            raise TokenError("bad issuer")
        if claims.get("aud") != audience:
            raise TokenError("wrong audience")
        if int(now if now is not None else time.time()) >= int(claims.get("exp", 0)):
            raise TokenError("expired")
        return claims
