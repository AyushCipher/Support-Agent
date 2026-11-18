"""Central configuration. Everything tunable lives here or in AgentConfig."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = Path(os.getenv("SA_POLICY_DIR", ROOT / "policies"))
RUNS_DIR = Path(os.getenv("SA_RUNS_DIR", ROOT / "runs"))

# Frozen clock so the simulated shop and the eval suite are fully deterministic.
NOW = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)

AUTO_REFUND_CAP_CENTS = 10_000      # above this a human must approve
MAX_WRITE_CALLS_PER_SESSION = 4     # blast-radius limiter

MODEL_TIERS = {
    "fast": os.getenv("SA_MODEL_FAST", "claude-haiku-5-5"),
    "strong": os.getenv("SA_MODEL_STRONG", "claude-sonnet-5-5"),
}
# The verifier is a SEPARATE model role (independent second opinion before money moves). Defaults to the strong model.
MODEL_TIERS["verifier"] = os.getenv("SA_MODEL_VERIFIER", MODEL_TIERS["strong"])

# USD per 1M tokens (input, output). PLACEHOLDERS: set from the current pricing page before
# trusting any cost number produced against a live model.
PRICING = {
    "claude-haiku-5-5": (1.0, 5.0),
    "claude-sonnet-5-5": (3.0, 15.0),
    "scripted-fast": (1.0, 5.0),      # notional, mirrors the fast tier
    "scripted-strong": (3.0, 15.0),   # notional, mirrors the strong tier
    "scripted-verifier": (3.0, 15.0), # notional, mirrors the strong tier
    # Gemini (SA_BACKEND=gemini). PLACEHOLDERS like the rows above: check the current pricing page; free-tier keys cost $0 but are rate-limited.
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-3.5-flash": (0.30, 2.50),
}


@dataclass(frozen=True)
class AgentConfig:
    name: str = "full"
    prompt_set: str = "v2"
    guard_input: bool = True          # injection scoring + PII redaction on user input
    guard_tool_output: bool = True    # sanitize + quarantine tool output (indirect injection)
    guard_tools: bool = True          # schema, authz injection, policy verification, caps, rate limits
    guard_output: bool = True         # claim verification, leak + PII checks, repair loop
    guard_pii: bool = True            # reversible PII tokenization: LLM provider never sees raw PII
    require_approval: bool = True     # human-in-the-loop for high-risk actions
    use_router: bool = True           # escalate fast -> strong model on risk signals
    allow_anon_server: bool = False   # vulnerable-by-design legacy server (baseline only)
    backend: str = "scripted"         # scripted | anthropic | gemini  (gemini: see llm/gemini_client.py)
    claim_verifier: str = "regex"     # regex | llm (llm requires a live backend)
    max_steps: int = 14
    max_total_tokens: int = 60_000
    max_cost_usd: float = 0.50
    timeout_s: float = 90.0
    llm_retries: int = 2
    llm_backoff_s: float = 0.0        # set >0 for live backends
    tool_timeout_s: float = 10.0
    tool_result_max_chars: int = 1800
    context_budget_tokens: int = 6000
    keep_last_messages: int = 8
    # --- reliability / performance (ADR 0013, 0014). Defaults leave the eval ladder's numbers unchanged (see tests/test_harness_v2.py).
    verifier: bool = False            # independent second opinion before money moves (OPTIONAL; costs one extra model call)
    verifier_threshold_cents: int = 5_000   # verify money actions at/above this amount (or on a tainted / risk-signalled session)
    verifier_max_calls: int = 3       # per-session budget; exhausted budget fails CLOSED (escalates to a human)
    verifier_max_cost_usd: float = 0.05     # per-session verifier spend cap; exceeded -> fail closed
    verifier_timeout_s: float = 15.0
    parallel_tools: bool = True       # run consecutive independent READ tool calls concurrently (writes are never parallel)
    max_parallel_tools: int = 4
    tool_cache_ttl_s: float = 30.0    # TTL cache for idempotent read tools; 0 disables
    llm_fallback: bool = True         # on exhausted retries try the other model tier, then hand off to a human
    handoff_on_llm_error: bool = False  # NON-transient provider errors (4xx, quota) -> safe human handoff instead of raising.
                                        # On in the API server; off for evals, which must see QuotaExhausted to stop a run.

    def with_(self, **kw) -> "AgentConfig":
        return replace(self, **kw)


PRESETS = {
    "baseline": AgentConfig(name="baseline", prompt_set="v1", guard_input=False, guard_tool_output=False,
                            guard_tools=False, guard_output=False, guard_pii=False, require_approval=False,
                            use_router=False, allow_anon_server=True),
    "+prompt_v2": AgentConfig(name="+prompt_v2", prompt_set="v2", guard_input=False, guard_tool_output=False,
                              guard_tools=False, guard_output=False, guard_pii=False, require_approval=False,
                              use_router=False, allow_anon_server=True),
    "+io_guards": AgentConfig(name="+io_guards", guard_tools=False, require_approval=False,
                              use_router=False, allow_anon_server=True),
    "+tool_guards": AgentConfig(name="+tool_guards", require_approval=False, use_router=False),
    "full": AgentConfig(name="full"),
}
