"""Asymmetric agent identities: EdDSA (Ed25519) JWTs, JWKS, key ids, rotation with an overlap window (ADR-0010).

Needs the optional `cryptography` package (`pip install "supportagent[crypto]"`); every entry point raises one clear error
when it is missing, and the HS256 `IdentityService` in identity.py remains the explicit dev fallback.

  harness (issuer)  : KeyRing holds PRIVATE keys -> IdentityIssuer.issue() mints 60 s, audience-bound, jti-carrying tokens
  MCP servers       : JwksVerifier holds PUBLIC keys only (no `issue`, no private material) -> can verify, can never mint
  GET /.well-known/jwks.json serves `issuer.jwks()`; verifiers re-fetch on an unknown `kid`.

Key lifecycle: active (signs+verifies) -> retiring (verifies until `verify_until`, = rotation time + overlap) -> retired
(dropped from the JWKS; tokens signed by it are rejected as `unknown kid` / `key retired`).
A JwksVerifier only accepts the algorithms it was configured with (EdDSA by default), so `alg:none` and HS256
"public key used as HMAC secret" confusion attacks are rejected before any signature is checked.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
import threading
import time
import uuid
from typing import Any, Callable

from .crypto import CryptoUnavailable, crypto_available, production_mode, require_crypto
from .identity import IdentityService, TokenError, _b64, _unb64

ALG_EDDSA, ALG_RS256 = "EdDSA", "RS256"
DEFAULT_OVERLAP_S = 120          # >= 2x token TTL: tokens minted just before a rotation still verify while in flight
MAX_TTL_S = 300                  # a verifier refuses tokens whose exp-iat exceeds this, even if correctly signed


def _kid_for(raw_public: bytes) -> str:
    return "k-" + hashlib.sha256(raw_public).hexdigest()[:12]


class KeyRing:
    """Issuer-side key material with rotation. Holds PRIVATE keys: lives only in the issuing process (the agent harness).

    `path` optionally persists the ring (0600 JSON). In production keep the private key in a KMS/HSM instead.
    """

    def __init__(self, overlap_s: int = DEFAULT_OVERLAP_S, path: str | os.PathLike | None = None):
        require_crypto("asymmetric agent identities (EdDSA)")
        self.overlap_s, self.path = overlap_s, path
        self._keys: list[dict] = []
        self._lock = threading.Lock()
        if path and os.path.exists(path):
            self._load()
        if not self._keys:
            self._add(time.time())
            self._save()

    def _add(self, now: float) -> dict:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        priv = Ed25519PrivateKey.generate()
        raw = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        k = {"kid": _kid_for(raw), "priv": priv, "x": _b64(raw), "created": now, "verify_until": None}
        self._keys.append(k)
        return k

    def _load(self) -> None:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        with open(self.path) as f:
            data = json.load(f)
        for d in data["keys"]:
            priv = Ed25519PrivateKey.from_private_bytes(_unb64(d["d"]))
            raw = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            self._keys.append({"kid": d["kid"], "priv": priv, "x": _b64(raw), "created": d["created"], "verify_until": d.get("verify_until")})

    def _save(self) -> None:
        if not self.path:
            return
        from cryptography.hazmat.primitives import serialization
        keys = [{"kid": k["kid"], "created": k["created"], "verify_until": k["verify_until"],
                 "d": _b64(k["priv"].private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()))}
                for k in self._keys]
        tmp = f"{self.path}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"keys": keys}, f)
        os.replace(tmp, self.path)

    @property
    def active(self) -> dict:
        with self._lock:
            return next(k for k in reversed(self._keys) if k["verify_until"] is None)

    def rotate(self, now: float | None = None) -> str:
        """New active key; the previous one keeps verifying for `overlap_s`, then is retired. Returns the new kid."""
        now = time.time() if now is None else now
        with self._lock:
            for k in self._keys:
                if k["verify_until"] is None:
                    k["verify_until"] = now + self.overlap_s
            new = self._add(now)
            self._save()
            return new["kid"]

    def prune(self, now: float | None = None) -> None:
        """Forget retired private keys entirely."""
        now = time.time() if now is None else now
        with self._lock:
            self._keys = [k for k in self._keys if k["verify_until"] is None or k["verify_until"] > now]
            self._save()

    def jwks(self, now: float | None = None) -> dict:
        """The PUBLIC key set (what /.well-known/jwks.json serves). Retired keys omitted; retiring keys carry their deadline."""
        now = time.time() if now is None else now
        out = []
        with self._lock:
            for k in self._keys:
                if k["verify_until"] is not None and k["verify_until"] <= now:
                    continue
                jwk = {"kty": "OKP", "crv": "Ed25519", "use": "sig", "alg": ALG_EDDSA, "kid": k["kid"], "x": k["x"]}
                if k["verify_until"] is not None:
                    jwk["x-verify-until"] = k["verify_until"]
                out.append(jwk)
        return {"keys": out}


class IdentityIssuer:
    """Mints EdDSA agent tokens. Same `issue()` signature as the HS256 `IdentityService`. Lives in the harness only."""
    mode, can_mint = "eddsa", True

    def __init__(self, keyring: KeyRing | None = None, issuer: str = "supportagent"):
        self.keyring, self.issuer = keyring or KeyRing(), issuer
        self._verifier = JwksVerifier(self.keyring.jwks, issuer=issuer, cache_ttl=0)

    def issue(self, agent: str, session_id: str, scopes: list[str], audience: str, ttl: int = 60, now: float | None = None) -> str:
        now = int(now if now is not None else time.time())
        key = self.keyring.active
        header = {"alg": ALG_EDDSA, "typ": "JWT", "kid": key["kid"]}
        claims = {"iss": self.issuer, "sub": f"agent:{agent}", "sid": session_id or "-", "aud": audience, "scope": " ".join(scopes),
                  "iat": now, "exp": now + ttl, "jti": uuid.uuid4().hex}
        si = f"{_b64(json.dumps(header, separators=(',', ':')).encode())}.{_b64(json.dumps(claims, separators=(',', ':')).encode())}"
        return f"{si}.{_b64(key['priv'].sign(si.encode()))}"

    def verify(self, token: str | None, audience: str, now: float | None = None) -> dict:
        return self._verifier.verify(token, audience, now=now, max_ttl=MAX_TTL_S)

    def jwks(self, now: float | None = None) -> dict:
        return self.keyring.jwks(now)

    def rotate(self, now: float | None = None) -> str:
        return self.keyring.rotate(now)

    def verifier(self, remote: bool = False) -> "JwksVerifier":
        """A verify-only view for MCP servers. In-process it follows rotations live; `remote=True` is a static JWKS snapshot."""
        if remote:
            return JwksVerifier(self.keyring.jwks(), issuer=self.issuer)
        return JwksVerifier(self.keyring.jwks, issuer=self.issuer, cache_ttl=0)


class JwksVerifier:
    """Verify-only: holds public keys (from a JWKS dict, a callable, a file path or an http(s) URL). No `issue()` by design.

    Also verifies external OIDC bearer tokens (RS256/EdDSA) for the admin API: pass `allowed_algs`, `issuer`, `max_ttl=None`.
    """
    can_mint = False

    def __init__(self, source: dict | Callable[[], dict] | str, issuer: str | None = "supportagent",
                 allowed_algs: tuple[str, ...] = (ALG_EDDSA,), cache_ttl: float = 30.0):
        require_crypto("JWKS token verification")
        self.source, self.issuer, self.allowed_algs, self.cache_ttl = source, issuer, allowed_algs, cache_ttl
        self._keys: dict[str, dict] = {}
        self._loaded_at = -1e9
        self._lock = threading.Lock()

    def _fetch(self) -> dict:
        s = self.source
        if isinstance(s, dict):
            return s
        if callable(s):
            return s()
        if s.startswith(("http://", "https://")):
            import urllib.request
            with urllib.request.urlopen(s, timeout=3) as r:                     # operator-configured URL, not user input
                return json.loads(r.read(1_000_000))
        with open(s) as f:
            return json.load(f)

    def _refresh(self, force: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            if not force and self._keys and now - self._loaded_at < self.cache_ttl:
                return
            try:
                doc = self._fetch()
            except Exception:
                if self._keys:                                                  # keep last known good keys on a fetch failure
                    return
                raise TokenError("jwks unavailable") from None
            self._keys = {k["kid"]: k for k in doc.get("keys", []) if k.get("kid")}
            self._loaded_at = now

    @staticmethod
    def _public_key(jwk: dict, alg: str):
        try:
            if alg == ALG_EDDSA and jwk.get("kty") == "OKP" and jwk.get("crv") == "Ed25519":
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
                return Ed25519PublicKey.from_public_bytes(_unb64(jwk["x"]))
            if alg == ALG_RS256 and jwk.get("kty") == "RSA":
                from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
                return RSAPublicNumbers(int.from_bytes(_unb64(jwk["e"]), "big"), int.from_bytes(_unb64(jwk["n"]), "big")).public_key()
        except (KeyError, ValueError):
            pass
        raise TokenError("key does not match alg")

    def verify(self, token: str | None, audience: str, now: float | None = None, max_ttl: int | None = MAX_TTL_S) -> dict:
        from cryptography.exceptions import InvalidSignature
        if not token or token.count(".") != 2:
            raise TokenError("malformed token")
        h64, c64, sig = token.split(".")
        try:
            header, claims, sig_b = json.loads(_unb64(h64)), json.loads(_unb64(c64)), _unb64(sig)
            if not isinstance(header, dict) or not isinstance(claims, dict):
                raise ValueError
        except Exception:
            raise TokenError("malformed token") from None
        alg = header.get("alg")
        if alg not in self.allowed_algs:                    # rejects alg:none, HS256 (key confusion) and anything unexpected
            raise TokenError("unsupported alg")
        kid = header.get("kid")
        if not kid:
            raise TokenError("missing kid")
        self._refresh()
        jwk = self._keys.get(kid)
        if jwk is None:
            self._refresh(force=True)                       # unknown kid: maybe a fresh rotation, re-fetch once
            jwk = self._keys.get(kid)
        if jwk is None:
            raise TokenError("unknown kid")                 # includes retired keys (dropped from the JWKS)
        t = int(now if now is not None else time.time())
        if jwk.get("x-verify-until") is not None and t >= jwk["x-verify-until"]:
            raise TokenError("key retired")                 # enforced even if a stale JWKS copy still lists it
        pub = self._public_key(jwk, alg)
        try:
            if alg == ALG_EDDSA:
                pub.verify(sig_b, f"{h64}.{c64}".encode())
            else:
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import padding
                pub.verify(sig_b, f"{h64}.{c64}".encode(), padding.PKCS1v15(), hashes.SHA256())
        except InvalidSignature:
            raise TokenError("bad signature") from None
        if self.issuer is not None and claims.get("iss") != self.issuer:
            raise TokenError("bad issuer")
        aud = claims.get("aud")
        if not (aud == audience or (isinstance(aud, list) and audience in aud)):
            raise TokenError("wrong audience")
        try:
            exp, iat = int(claims.get("exp", 0)), int(claims.get("iat", 0))
        except (TypeError, ValueError):
            raise TokenError("malformed token") from None
        if t >= exp:
            raise TokenError("expired")
        if max_ttl is not None and exp - iat > max_ttl:
            raise TokenError("ttl too long")
        return claims


class ReplayGuard:
    """Remember jti until its exp; reject a second use. Bounded by the token TTL, so memory is O(calls per minute)."""

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def check_and_add(self, jti: str, exp: float, now: float | None = None) -> bool:
        """True if fresh, False if replay."""
        now = time.time() if now is None else now
        with self._lock:
            self._seen = {j: e for j, e in self._seen.items() if e > now}
            if jti in self._seen:
                return False
            self._seen[jti] = exp
            return True


# ---- wiring ------------------------------------------------------------------------------------------
class IdentityBundle:
    """What build_registry needs: the issuer (harness side) and the credential servers get (public keys; dev: the secret)."""

    def __init__(self, issuer: Any, server_credential: Any, mode: str):
        self.issuer, self.server_credential, self.mode = issuer, server_credential, mode

    def server_env(self) -> dict[str, str]:
        """Env for stdio server subprocesses. EdDSA mode ships a PUBLIC JWKS snapshot only - no private key leaves the harness."""
        if self.mode == "eddsa":
            return {"MCP_JWKS": json.dumps(self.issuer.jwks())}
        return {"MCP_JWT_SECRET": self.server_credential}


def make_identity(jwt_secret: str | None = None, alg: str | None = None, keyring_path: str | None = None) -> IdentityBundle:
    """Choose the identity mode.

    SA_IDENTITY_ALG: `eddsa` (needs [crypto]; hard error if missing), `hs256` (explicit dev opt-in), `auto` (default).
    `auto`: an explicit secret (argument or SA_JWT_SECRET) means HS256; otherwise EdDSA when `cryptography` is installed;
    otherwise a loud warning and a random HS256 secret in dev, or a hard error when SA_ENV=production.
    HS256 is refused outright in production mode.
    """
    alg = (alg or os.getenv("SA_IDENTITY_ALG", "auto")).lower()
    secret = jwt_secret or os.getenv("SA_JWT_SECRET")
    if alg not in ("auto", "eddsa", "hs256"):
        raise ValueError(f"SA_IDENTITY_ALG must be auto|eddsa|hs256, got {alg!r}")
    if alg == "hs256" or (alg == "auto" and secret):
        if production_mode():
            raise CryptoUnavailable("HS256 shared-secret identities are a dev fallback and are refused when SA_ENV=production")
        secret = secret or secrets.token_hex(16)
        return IdentityBundle(IdentityService(secret), secret, "hs256")
    if not crypto_available():
        if alg == "eddsa" or production_mode():
            require_crypto("EdDSA agent identities")
        print("supportagent: WARNING 'cryptography' not installed - using the HS256 dev fallback for agent identities "
              "(any verifier can mint). Install supportagent[crypto] or set SA_IDENTITY_ALG=hs256 to silence.", file=sys.stderr)
        secret = secrets.token_hex(16)
        return IdentityBundle(IdentityService(secret), secret, "hs256")
    issuer = IdentityIssuer(KeyRing(path=keyring_path or os.getenv("SA_KEYRING_PATH")))
    return IdentityBundle(issuer, issuer.verifier(), "eddsa")


def verifier_from_env() -> JwksVerifier | None:
    """Server side (stdio subprocess / standalone): MCP_JWKS (inline JSON), MCP_JWKS_FILE or MCP_JWKS_URL -> public keys only."""
    if os.getenv("MCP_JWKS"):
        return JwksVerifier(json.loads(os.environ["MCP_JWKS"]))
    src = os.getenv("MCP_JWKS_FILE") or os.getenv("MCP_JWKS_URL")
    return JwksVerifier(src) if src else None
