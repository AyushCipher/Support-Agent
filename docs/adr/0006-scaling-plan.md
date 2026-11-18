# ADR-0006: What changes at scale

**Status:** proposed (not built — this repo is single-process by design)

| Concern | Today | At scale |
|---|---|---|
| Durable execution | SQLite checkpoint after every step; resume by session id | **Temporal** workflows (activities = LLM/tool calls, signals = approval decisions, timers = SLA); keep the idempotency-key contract |
| State | SQLite | Postgres; sessions append-only (event-sourced) so replays and audits are free |
| Concurrency | One lock in the HTTP server | Stateless workers + queue; per-session ordering via workflow id |
| Approvals | Poll UI | Webhook/Slack/Zendesk integration, SLA timers, escalation of unanswered approvals, reviewer RBAC and 4-eyes above a second threshold |
| LLM access | Direct SDK | Gateway with per-tenant rate limits, provider failover, prompt caching (system prompt is already cache-marked), budget alerts |
| Tracing | JSONL + optional OTLP | OTel Collector → Tempo/Honeycomb; metrics from spans (cost/latency/guard rates); alert on injection-flag and block rates |
| Evals | 2k trials in seconds (simulated) | Nightly live eval on a sampled/stratified set, shadow-mode new prompts against production traffic, per-category regression budgets |
| MCP | stdio subprocesses | Remote MCP over HTTP with OAuth 2.1, per-tenant audience-bound tokens, server-side rate limits |
| Retrieval | BM25 | Hybrid BM25 + embeddings + re-rank; eval set grown from real queries |
| Safety | Static scripted attacks | Continuous adaptive red-teaming (attacker model vs. system), canary tokens in the wild, human review of a sample of *approved* actions |
