# Architecture

## Request lifecycle

1. **Input guard** (`guardrails/input.py`): score the message for injection patterns (noisy-or over weighted regexes + optional LLM classifier hook); redact card/SSN (Luhn-validated). A flag does *not* refuse the customer — it **taints** the session and raises the model tier. False positives stay cheap; attacks lose privileges.
2. **Triage** (`runtime._route`): cheap model calls `classify_intent` (forced tool choice). Intent maps to a specialist. Unknown intent maps to the *least-privileged* specialist.
3. **Specialist loop** (`runtime._loop`): budget check → context prepare → LLM call (retry + exponential backoff on `TransientLLMError`) → tool calls or final text.
4. **Tool execution** (`runtime._execute`): allowlist → JSON-schema validation (against the *public* schema; hidden params are not settable by the model) → inject `acting_customer_id` and a deterministic idempotency key → `ToolGuard.authorize` → optional human approval → MCP dispatch (timeout, reconnect, circuit breaker) → truncate → sanitize → quarantine envelope → taint → record evidence/memory → checkpoint.
5. **Final text**: output guard verifies claims against *evidence* (what tools actually confirmed). Failure → feedback message → bounded repair loop (2) → safe fallback + escalation ticket.

## Trust boundaries

| Source | Trust | Why |
|---|---|---|
| Customer message | Untrusted *as instructions*, trusted as the customer's own intent | Only the customer's words can create a refund/cancel intent (intent grounding) |
| Tool output (order notes, KB text, third-party results) | **Untrusted data** | Anyone can write into a notes field; wrapped, sanitized, taints the session |
| Authenticated session (customer id, email) | Trusted | Comes from the auth layer; injected into tool calls by the harness, never by the model |
| Policy engine / DB | Ground truth | Used by the agent's check tool **and** independently by the guard and the evals |
| Memory | Trusted *only if harness-written* | Writes require `source="tool:..."`; user text can never become a memory |

## State

- `shop.db` — simulated business data (customers, orders, subscriptions, refunds, tickets, audit log).
- `state.db` — `sessions` (full serialized session, one row per session, updated after every step), `approvals`, `customer_memory`.
- Resume contract: a session in `awaiting_approval` stores the pending call, its resolved arguments and the remaining batched calls; `resume()` executes the approved call *with the stored idempotency key*, so a double resume cannot double-refund.

## Why in-process **and** stdio MCP

Both transports share one JSON-RPC dispatcher (`MCPServer.handle`). Evals use in-process (≈5 ms/trial, 2,000 trials in seconds); production and integration tests use stdio subprocesses. `tests/test_mcp.py::test_stdio_transport_real_subprocess_parity` keeps them honest.

## Model routing

`ModelRouter.tier_for(session)` returns `strong` when the session has any risk signal (`input_injection`, `tool_output_injection`, `guard_block`, `output_guard`), otherwise `fast`. Risk signals accumulate per session. Triage is always `fast`.
