# ADR-0002: Guardrails live in the harness, are layered, and the structural ones carry the weight

**Status:** accepted

**Context.** Prompts and classifiers are probabilistic; money movement must not be.

**Decision.** Put every enforcement point outside the model, at a boundary the model cannot cross:
1. *Input* (taint, not refuse), 2. *Tool-call* (allowlist, schema, authz context, policy re-verification, caps, rate limit, intent grounding, approval), 3. *Tool-output* (sanitize, quarantine, taint), 4. *Output* (evidence-based claim check, leak check, bounded repair).
Detection layers (1, 3, 4-regex) reduce load; **structural** layers (2) bound the damage when detection fails.

**Evidence (ablation, `docs/results.md`).** Prompt + detection layers: attack success 43% → 26%. Adding structural tool guards: → 0%. The paraphrased-injection finding (F4) is the proof that detection alone is insufficient and intent grounding is not optional.

**Consequences.** Tainting instead of refusing keeps false positives cheap (a benign user is only asked for approval on money actions). Guard errors are written as actionable instructions to the model — and reviewed like prompts (F2).

**Rejected.** An LLM "guard model" as the only line of defense (non-deterministic, attackable by the same injection); refusing flagged users (bad UX, trivially probed).
