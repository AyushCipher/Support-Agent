# ADR-0003: MCP servers, two transports, scoped tokens

**Status:** accepted

**Decision.**
- Three first-party servers (orders, kb, tickets) implement tools, resources and prompts per the MCP spec over newline-delimited JSON-RPC.
- One dispatcher, two transports: **in-process** (evals/tests) and **stdio subprocess** (integration/production); a parity test keeps them identical.
- Authorization: each tool declares a required scope; each agent role holds a token mapping to scopes; the server enforces it (`forbidden`). The harness *also* allowlists tools per role — two independent checks.
- Hard invariants (ownership, positive amount, remaining balance, idempotency) are enforced in the server, even though the harness checks business policy too.
- Third-party servers load from config; unannotated tools default to the `write` risk tier and are reachable by no agent until explicitly granted.
- Client resilience: per-call timeout, one transparent reconnect, circuit breaker; startup failure marks the server `degraded` and the agent keeps working without its tools.

**Consequences.** `_meta.token` is a stand-in for real auth on stdio (no transport-level auth exists). For HTTP/SSE transports use OAuth 2.1 bearer tokens with audience-bound scopes; the scope-per-tool model carries over unchanged.

**Hidden parameters.** `acting_customer_id` and `idempotency_key` are in the server schema but stripped from the model-visible schema and overwritten by the harness — the model cannot impersonate another customer or defeat idempotency even if it tries (`test_model_cannot_set_hidden_params`).
