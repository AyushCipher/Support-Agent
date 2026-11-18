# ADR-0009: Kill switches, canary rollout and shadow mode

**Status:** accepted

**Decision.** Operational state lives in a flag store read on every use, so changes apply mid-session. Kill switches have two scopes: *global* (no model calls) and *money* (checked again at dispatch, so an action approved before the switch is still refused). Prompt versions are assigned per session by a deterministic hash (sticky canary) and recorded with their content hash. **Shadow mode** replays the turn against a candidate prompt with writes simulated, no approvals and no memory, and records tool-call and outcome agreement. Promotion is gated by the same eval suite as CI.

**A bug that shaped the design (F10).** The first shadow implementation ran after the live turn and saw the order the live run had just refunded, so it "disagreed" for the wrong reason. Shadow must observe the same pre-turn world as live: it now runs first, in dry-run. In production run it asynchronously against a snapshot or read replica.

**Consequences.** Rollback and incident response are one command each and are audit-chained. Shadow in-process roughly doubles per-turn latency; acceptable for evaluation traffic, not for the hot path at scale.
