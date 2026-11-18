from __future__ import annotations

from ..config import MODEL_TIERS, AgentConfig
from .scripted import ScriptedLLM


LIVE_BACKENDS = ("anthropic", "gemini")


def _live(backend: str, model: str):
    if backend == "anthropic":
        from .anthropic_client import AnthropicLLM
        return AnthropicLLM(model)
    from .gemini_client import GeminiLLM
    return GeminiLLM(model)


def make_llm(cfg: AgentConfig, tier: str, seed: int = 0):
    """scripted | anthropic | gemini. An unknown backend is an error: silently falling back to the scripted model would let a
    misconfigured production deploy answer customers with the offline simulator (review finding R1)."""
    if cfg.backend in LIVE_BACKENDS:
        return _live(cfg.backend, MODEL_TIERS[tier])
    if cfg.backend != "scripted":
        raise ValueError(f"unknown backend {cfg.backend!r}; expected scripted|anthropic|gemini")
    return ScriptedLLM(tier, seed)


def make_verifier_llm(cfg: AgentConfig, mode: str = "honest"):
    """The independent verifier is its OWN model role (never the agent's llm instance, never the agent's conversation).
    Live: MODEL_TIERS['verifier'] (SA_MODEL_VERIFIER, default = strong). Offline: a separate rule-based simulator."""
    if cfg.backend in LIVE_BACKENDS:                       # never the simulator behind a live agent
        return _live(cfg.backend, MODEL_TIERS["verifier"])
    if cfg.backend != "scripted":
        raise ValueError(f"unknown backend {cfg.backend!r}; expected scripted|anthropic|gemini")
    from .verifier_sim import ScriptedVerifier
    return ScriptedVerifier(mode)


class ModelRouter:
    """Cheap model by default; escalate the *specialist* to the strong model on risk signals.
    Signals: injection flagged (input or tool output), guard block, output-guard repair, repeated failures.

    Degradation: `fallback_chain` lists the tiers to try, in order, when a model call keeps failing (provider outage / overload
    on one model). If every tier fails the harness hands the case to a human (see Runtime._degraded_handoff)."""
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg

    def tier_for(self, session) -> str:
        if not self.cfg.use_router:
            return "fast"
        return "strong" if session.risk_signals else "fast"

    def fallback_chain(self, tier: str) -> list[str]:
        if not getattr(self.cfg, "llm_fallback", True):
            return [tier]
        return [tier] + [t for t in ("strong", "fast") if t != tier]
