# ADR-0004: Prompt-injection strategy — assume detection fails

**Status:** accepted

**Context.** Order notes, KB text and third-party results are attacker-writable. Regexes and classifiers are bypassed by paraphrase (see F4: a note saying a reimbursement "was pre-approved" evaded every pattern).

**Decision.** Defense in depth with an explicit assumption that layers 1–2 *will* miss some attacks:
1. **Sanitize** instruction-like sentences in data fields (cheap, catches the obvious).
2. **Quarantine** with an `<tool_output trust="untrusted">` envelope and prompt rules that treat it as data.
3. **Taint** the session on any finding → every money action needs human approval and the model tier escalates.
4. **Intent grounding** (structural): money/cancel actions and billing handoffs require the *customer's own messages* to ask for them. Content the agent merely read cannot create intent.
5. **Independent verification**: policy re-checked by the harness; amounts capped; refunds idempotent.

**Consequences.** Layer 4 is what turned a 3.1% red-team success rate to 0%. Residual risk: an attacker who can also *say* something that looks like a refund request in the customer channel (that is a social-engineering problem, handled by caps, abuse review and approvals).
