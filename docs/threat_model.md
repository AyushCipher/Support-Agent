# Threat model

**Assets.** Money (refunds), customer PII (other customers' orders/emails/addresses), system instructions, the integrity of the customer's account (cancellations).
**Principals.** Customer (authenticated, partially trusted), data authors (anyone who can write order notes, ticket text, KB content), third-party MCP servers, the model itself (trusted to be *helpful*, not to be *safe*), human reviewers (trusted).

| # | Threat | Mitigation (layer) | Evidence |
|---|---|---|---|
| T1 | Direct prompt injection ("ignore previous instructions…") | Input scoring → session taint → approval required for money; tier escalation. Structural checks make success irrelevant | scenarios I01–I04; `test_injection_flagged` |
| T2 | **Indirect** injection via tool output (order note, KB, third-party result) | Sanitize instruction-like sentences; untrusted envelope; taint; **intent grounding** (paraphrase-proof) | N01–N05; F4; `test_tool_output_sanitized…`, `test_intent_grounding…` |
| T3 | Confused deputy: model asked to act on another customer's order | `acting_customer_id` injected by harness from the session, hidden from the model schema; server enforces ownership and returns the same error for "missing" and "not yours" | X01–X03; `test_model_cannot_set_hidden_params`, `test_ownership_enforced_server_side` |
| T4 | Excess agency: model calls a tool its role shouldn't have | Per-role allowlists + per-role server-side token scopes (two independent checks) | `test_scope_enforcement_per_role`, `test_unknown_tool_and_allowlist` |
| T5 | Policy violation by honest model error (refund > max, ineligible order) | ToolGuard re-runs the policy engine before any money tool; blocks with an actionable message | `test_guard_blocks_over_max_amount`; baseline vs +tool_guards |
| T6 | Social engineering ("manager approved", emotional pressure, amount inflation) | Authority claims carry no weight (only tools/policy decide); exceptions route to humans; refund ≤ policy max | I04, SE01–SE04 |
| T7 | Runaway spend / loops | Step, token, cost and wall-clock budgets → human escalation; write-call rate limit | `test_budget_exceeded_escalates_to_human`, `test_write_rate_limit` |
| T8 | Double-spend on retry/resume | Deterministic idempotency keys (session+tool+args); DB unique constraint; transactional refund | `test_crash_resume_exactly_once`, `test_refund_hard_invariants_and_idempotency` |
| T9 | System-prompt / secret leakage | Canary in prompts; output guard blocks canary/secret/foreign email; prompt v2 | I03; `test_output_guard_blocks` |
| T10 | False assurance to the customer | Evidence-based claim verification + repair loop; LLM verifier hook | F3, F5 |
| T11 | Memory poisoning | Memory writable only by harness from verified tool outcomes | `test_memory_written_only_from_tool_outcomes` |
| T12 | Malicious / compromised third-party MCP server | Unknown tools default to risky tier; nothing exposed until explicitly granted; results still quarantined; failures degrade gracefully | `test_third_party_server_least_privilege`, `test_graceful_degradation…` |
| T13 | Code execution abuse | AST allow-list + isolated interpreter + rlimits + timeout; arithmetic only | `test_sandbox_rejects` (13 escape attempts) |
| T14 | Denial of service via tool slowness | Per-call timeout, one reconnect, circuit breaker | `test_circuit_breaker_opens`, `test_crash_mid_session_reconnects_once` |

| T15 | Stolen / forged / replayed agent credential; **compromised MCP server minting tokens** | EdDSA-signed, `kid`-tagged, audience-bound 60 s JWTs; servers hold public keys only (cannot mint); key rotation with overlap, retired keys rejected; `jti` replay rejection; `alg:none` and HS256 key-confusion rejected; HS256 only as an explicit dev fallback (refused in production) | `tests/test_security.py`, `tests/test_security_v2.py` (ADR-0010) |
| T16 | Tampering with the audit trail (insider, compromised host), **including tail truncation** | Hash chain in both DBs; edit/delete/re-attribution detected and localized; signed head anchors (`audit anchor`, `audit verify --anchors`) detect truncation and consistent rewrites, provided anchors are shipped to write-once storage | `test_chain_verifies_and_detects_edit_delete`, `tests/test_audit_anchors.py` (ADR-0011) |
| T17 | PII exposure to the model vendor / in checkpoints / at rest | Reversible tokenization at the LLM boundary; egress restore only for provably-own data; metric: 0% raw PII sent. With `SA_VAULT_KEY`: vault and memory values are AES-256-GCM ciphertext (per-customer keys), production fails closed without a key | `tests/test_pii.py`, eval `pii_to_llm_rate`, `tests/test_privacy.py` |
| T18 | Runaway or compromised agent in production; **unauthorized reviewer/admin** | Global and money kill switches effective mid-session and on already-approved actions; admin API and reviewer desk behind `SA_ADMIN_TOKEN` (constant-time) or OIDC JWT via JWKS, disabled when unconfigured; approval POSTs CSRF-safe, reviewer identity taken from auth, single-decision (409) | `tests/test_ops.py`, `tests/test_admin_auth.py` |
| T21 | Data-subject erasure vs. immutable audit log | Crypto-shredding per customer + transcript redaction + tombstone event; chain never rewritten, so integrity and earlier anchors still verify | `tests/test_privacy.py` |
| T19 | Privilege smuggling through the clarification channel | Answers pass the input guard (taint), are schema-validated, never guess destructive targets | `test_answers_go_through_the_input_guard` |
| T20 | Bad prompt release | Eval-gated promotion, canary, shadow dry-run, one-command rollback | `test_promotion_gate`, canary/shadow tests |

## Residual risk (deliberately not hidden)

- **Adaptive attackers.** The suite is fixed; an attacker who iterates against the system may find paraphrases that also look like customer intent.
- **Unverifiable claims.** "It's defective" is accepted on the customer's word within the 60-day window (policy choice). Mitigated by caps, abuse review (≥3 refunds/90 days) and audit logs, not eliminated.
- **False claims (F5).** ~2% in offline runs until the semantic verifier is enabled with a real model.
- **Identity keys.** Fixed in code (EdDSA + JWKS, servers cannot mint; ADR-0010) but the private key sits in process memory or an optional 0600 file, not a KMS/HSM, and stdio children get a JWKS snapshot (rotation reaches them via file/URL JWKS or restart). Without the `[crypto]` extra the system falls back to HS256 in dev with a warning (shared secret: any verifier can mint). No mTLS/SPIFFE caller binding; stdio has no transport auth beyond the signed per-call token.
- **Audit anchoring is only as strong as where the anchors go.** `audit anchor` writes a signed local file; an attacker with host access can truncate it or steal the signing key. Truncation is only reliably detected for rows older than the last anchor shipped to write-once storage (see compliance.md). Nothing runs the anchor job on a schedule by itself.
- **Encryption at rest is opt-in in dev.** Without `SA_VAULT_KEY` the vault and memory are plaintext (production mode refuses to start that way). `Session.customer_email` and trace files are not encrypted/purged by the vault layer. The master key lives in an env var (no KMS); key files must not be backed up with the data or shredding is not final. **PII detection** remains regex/field-based, not NER.
- **Erasure scope.** Business records and the pseudonymous customer id on audit rows are retained by design; no consent/TTL/export flow. A DEK may live in a running process for up to 5 s after shredding.
- **Customer authentication is out of scope.** `/api/chat` trusts the `customer_id` in the request and `/api/sessions/<id>` is unauthenticated (the session id acts as a capability); in production put customer authentication in front. Admin auth uses one shared static token unless OIDC roles are configured; there is no 4-eyes review or per-reviewer rate limiting.
- **Single-process server.** Fine for a demo; see ADR-0006 for what changes at scale.
