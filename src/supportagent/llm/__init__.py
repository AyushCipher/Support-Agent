"""LLM backends. `make_llm_any` is kept for callers that imported it; router.make_llm now dispatches scripted|anthropic|gemini itself."""
from __future__ import annotations


def make_llm_any(cfg, tier: str, seed: int = 0):
    from .router import make_llm
    return make_llm(cfg, tier, seed)
