"""File-based prompt registry: prompts/<role>/<version>.md, content-hashed so every trace/eval records exactly what ran."""
from __future__ import annotations

import hashlib
from pathlib import Path

BASE = Path(__file__).parent


def load(role: str, version: str) -> tuple[str, str]:
    text = (BASE / role / f"{version}.md").read_text().strip()
    return text, hashlib.sha256(text.encode()).hexdigest()[:8]


def versions(role: str) -> list[str]:
    return sorted(p.stem for p in (BASE / role).glob("*.md"))


def manifest(version: str) -> dict[str, str]:
    return {role: load(role, version)[1] for role in ("triage", "orders", "billing", "policy")}


# Prompt family for the independent pre-money verifier (ADR 0013). It is a separate family, NOT a per-session specialist role, so it
# is deliberately absent from manifest() (which keys per-session prompt hashes). It is versioned and content-hashed like the others.
VERIFIER_FAMILY = "verifier"


def load_verifier(version: str = "v1") -> tuple[str, str]:
    return load(VERIFIER_FAMILY, version)
