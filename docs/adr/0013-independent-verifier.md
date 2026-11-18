# ADR-0013: Independent verifier before money moves (optional)

**Status:** accepted (off by default: `AgentConfig.verifier=False`). Written retroactively during the review; the code predates it.

**Context.** ToolGuard already verifies every money action against policy ground truth fetched by the harness. What it cannot judge is *intent*: whether the customer actually asked for this refund of this order (the regex intent check is satisfied by "I do NOT want a refund"). A second opinion from the same model and conversation would share the agent's blind spots and any injection that fooled it.

**Decision.** `harness/verifier.py` asks a separate model role (`MODEL_TIERS["verifier"]`, prompt family `prompts/verifier/`) for approve/deny on risky money actions (tainted session, risk signals, or amount >= `verifier_threshold_cents`). It sees only: the proposed action; whitelisted fields of the order/subscription and the policy verdict, fetched by the harness with direct registry calls (never the read cache); and the customer's own words, embedded as one JSON array with `<` and `>` escaped (review finding R6). It never sees agent text, tool output, notes or reasoning. It can only add friction: agree -> existing path; disagree, malformed output, timeout, outage, exhausted per-session budget (calls and USD) or missing ground truth -> human approval (or a block when approvals are disabled). It runs after `ToolGuard.authorize`, so it can never unblock anything. With a live backend it is the live model of that backend (review finding R1: Gemini used to get the simulator).

**Consequences.**
- (+) A lazy or broken verifier degrades to the previous behaviour; a manipulated one cannot move money that policy forbids.
- (+) Every verdict is traced and audit-chained with the prompt hash.
- (-) One more model call (latency, cost) on the risky slice.
- (-) The customer's own words are still model input: a customer can try to talk the verifier into "approve", which at most restores the pre-verifier path.
- (-) The offline `ScriptedVerifier` is a rule-based test double, not evidence about any real model.
