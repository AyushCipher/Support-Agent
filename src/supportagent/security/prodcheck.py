"""Startup gate: refuse to serve with an insecure configuration when SA_ENV=production.

`problems(cfg)` returns human-readable reasons (env var NAMES only - never values). `enforce(cfg)` raises InsecureConfig listing all
of them, so an operator fixes everything in one deploy instead of one error per restart. Outside production it only returns the list
(the CLI prints them as warnings). Each rule maps to a finding or gap in docs/review_findings.md.
"""
from __future__ import annotations

import os

from .crypto import crypto_available, production_mode

KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}


class InsecureConfig(RuntimeError):
    pass


def problems(cfg) -> list[str]:
    p: list[str] = []
    if cfg.backend not in KEY_VARS:
        p.append(f"SA_BACKEND={cfg.backend!r}: production needs a live model backend (anthropic|gemini), not the offline simulator")
    else:
        if not os.getenv(KEY_VARS[cfg.backend]):
            p.append(f"{KEY_VARS[cfg.backend]} is not set (required by SA_BACKEND={cfg.backend})")
        from ..config import MODEL_TIERS
        family = {"anthropic": "claude", "gemini": "gemini"}[cfg.backend]
        for tier, var in (("fast", "SA_MODEL_FAST"), ("strong", "SA_MODEL_STRONG"), ("verifier", "SA_MODEL_VERIFIER")):
            if not str(MODEL_TIERS.get(tier, "")).startswith(family):
                p.append(f"{var}={MODEL_TIERS.get(tier)!r} is not a {cfg.backend} model (set it explicitly for SA_BACKEND={cfg.backend})")
    if cfg.name != "full" or not (cfg.guard_input and cfg.guard_tool_output and cfg.guard_tools and cfg.guard_output and cfg.guard_pii
                                  and cfg.require_approval):
        p.append(f"SA_CONFIG={cfg.name!r}: production must run the 'full' preset (every guard on, human approval required)")
    if cfg.allow_anon_server or os.getenv("SA_ALLOW_ANON") == "1":
        p.append("anonymous MCP access is enabled (SA_ALLOW_ANON / a baseline preset)")
    if not os.getenv("SA_VAULT_KEY"):
        p.append("SA_VAULT_KEY is not set: PII vault, memory and session emails would be stored unencrypted")
    if not crypto_available():
        p.append("the 'cryptography' package is missing (install supportagent[crypto]): needed for AES-GCM at rest and EdDSA identities")
    if len(os.getenv("SA_SESSION_SECRET", "")) < 32:
        p.append("SA_SESSION_SECRET must be set (>= 32 chars): it signs the per-session customer tokens")
    admin = os.getenv("SA_ADMIN_TOKEN", "")
    oidc = all(os.getenv(k) for k in ("SA_OIDC_JWKS", "SA_OIDC_ISSUER", "SA_OIDC_AUDIENCE"))
    if not oidc and len(admin) < 32:
        p.append("reviewer/admin auth: set SA_OIDC_JWKS/SA_OIDC_ISSUER/SA_OIDC_AUDIENCE, or an SA_ADMIN_TOKEN of >= 32 chars")
    if (os.getenv("SA_IDENTITY_ALG", "").lower() == "hs256") or os.getenv("SA_JWT_SECRET"):
        p.append("HS256 agent identities (SA_IDENTITY_ALG=hs256 / SA_JWT_SECRET) are a dev fallback; use EdDSA (unset both)")
    from .customerauth import oidc_configured
    if not oidc_configured() and os.getenv("SA_PUBLIC_DEMO") != "1":
        p.append("customer identity: configure SA_CUSTOMER_OIDC_JWKS/ISSUER/AUDIENCE, or set SA_PUBLIC_DEMO=1 to knowingly run a "
                 "public demo where the customer picker is trusted (seeded fake customers only)")
    return p


def enforce(cfg) -> list[str]:
    """Raise InsecureConfig in production if anything is wrong; otherwise return the (warning-only) list."""
    found = problems(cfg)
    if found and production_mode():
        raise InsecureConfig("refusing to start with SA_ENV=production:\n  - " + "\n  - ".join(found))
    return found
