# Operations runbook

All commands read/write `runs/state.db` (set `SA_RUNS_DIR`). The same operations exist on the admin API when `SA_ADMIN_TOKEN` (or `SA_OIDC_*`) is set (`/api/admin/flags|kill|audit/verify|shadow|slo`). Deployment itself is in [deploy.md](deploy.md).

## Something is wrong right now
1. **Stop money first:** `supportagent kill on --scope money`. Read-only help continues; pending approvals that get approved are still refused at execution time; the agent escalates to humans.
2. **Stop everything:** `supportagent kill on`. Every turn returns a "human will follow up" message without calling the model.
3. Restore: `supportagent kill off [--scope money]`.
4. Investigate: `supportagent audit replay <session-id>` (agent + business-side timeline with attribution), `supportagent audit verify` (integrity), traces in `runs/traces.jsonl` or Jaeger. Reviewers can open any transcript with `GET /api/sessions/<id>` and their admin bearer token.

## Startup refused (`refusing to start with SA_ENV=production`)
The process exits with code 2 and lists every problem by environment-variable NAME (values are never printed). Fix them all, redeploy. The rules live in `src/supportagent/security/prodcheck.py`: live backend + its API key + matching model names, `full` preset, no anonymous MCP, `SA_VAULT_KEY`, `cryptography` installed, `SA_SESSION_SECRET` >= 32 chars, reviewer auth (OIDC or `SA_ADMIN_TOKEN` >= 32 chars), no HS256 agent identities, and customer identity (`SA_CUSTOMER_OIDC_*`) or the explicit `SA_PUBLIC_DEMO=1` opt-in. Outside production the same list is printed as warnings.

## Customer sessions
- A new chat returns `session_token`; every later `/api/chat` on that session and every `GET /api/sessions/<id>` needs it (`X-Session-Token`). Missing, foreign and wrong tokens all get the same 404 (no session-existence oracle).
- **Rotating `SA_SESSION_SECRET` invalidates every open session token.** Customers mid-conversation must start a new chat; paused runs still resume when a reviewer decides (that path does not use customer tokens). Rotate during low traffic.
- Customers who lost their token cannot be re-attached by support; open a new session (the old one stays in the audit trail).

## SLOs and alerts
`supportagent slo eval [--webhook URL] [--alerts-file F]` evaluates the SLOs in `ops/slo.py` over `runs/traces.jsonl` + state and notifies sinks (de-duplicated through `runs/alert_state.json`; exit code 1 while anything fires). Prometheus users load `src/supportagent/ops/prometheus_rules.yml` (regenerate with `supportagent slo rules`; run `promtool check rules` first; the file is not validated by promtool in CI) and import `grafana_dashboard.json`. Objectives are illustrative defaults, not derived from real traffic; `task_success` is a proxy.

Delivery semantics: an alert is recorded as notified only when at least one sink accepted it, so a pager outage is retried on the next evaluation instead of being silenced for `renotify_s`; a resolution is likewise retried until delivered. Run the cron from ONE place (two evaluators sharing a state file can double-notify).

### alert-task-success
Turns ending in `escalated_budget`, `degraded_handoff` or the safe-fallback reply are burning budget. Check `llm_unavailable` / `llm_fallback` events (provider outage: the agent is already handing off safely), then `budget_exceeded` (loops: a prompt regression; roll back the prompt via flags), then `output_guard_blocked` (model making unverifiable claims).
### alert-turn-latency
p95 over the target. Look at `llm.call` span durations (provider slowness, retries with backoff, Gemini RPM pacing via `SA_GEMINI_RPM`) and `tools.batch` / `tool.call` durations (MCP server slow; breaker opening). Verifier calls add one model round trip on money actions.
### alert-tool-error-rate
Infrastructure tool failures (policy blocks are excluded). Check `tool_unavailable` / `tool_timeout` events and the MCP server logs; an open circuit breaker fails fast by design. If the orders server is down, consider `kill on --scope money`.
### alert-guard-block-rate
Unusual share of policy blocks: either an attack wave (look for `input_injection_flagged`, `tool_output_injection`, cross-customer `order_not_found` patterns by customer) or a prompt regression making the model propose disallowed actions (compare by `prompt.version`).
### alert-cost-per-resolution
Notional cost per resolved turn over threshold. Check router escalations to the strong tier (risk signals), repair loops, verifier spend (`verifier.cost_usd`), and whether pricing in `config.PRICING` is still a placeholder.
### alert-approval-backlog-age
The oldest pending human approval is older than the SLA. Staff the reviewer desk (`/approvals`). Customers are waiting with money in limbo; nothing executes without a decision.
### alert-kill-switch-global
The global kill switch is on. Expected during an incident; if not, find who set it: `supportagent audit replay` / `flags` history (every flag change is audit-chained with the operator).
### alert-kill-switch-money
Money actions are disabled. Same as above; remember approved refunds are refused at execution time while it is on.

## Data retention and erasure
- `supportagent retention show` prints the policy; `supportagent retention run --dry-run` reports what would change and writes nothing; `supportagent retention run` applies it and leaves a `retention_purge` tombstone per class in the audit chain. Only finished sessions are touched; legal holds are the `retention.legal_hold` flag (list of session or customer ids). Audit retention is archive-only.
- The traces file is rewritten in place; run retention when the API is idle or accept that a span written in the last instant can be lost (documented residual race).
- **Delete-on-request:** `supportagent privacy delete --customer <id>` crypto-shreds the customer's keys, redacts sessions/approvals/memory, and now also deletes that customer's spans from `runs/traces.jsonl` (no archive copy). Spans already exported to an OTLP backend (Jaeger/Tempo) must be purged there by trace id (= session id); the command prints the count it removed locally.
- Keep `SA_KEY_DIR` (default `runs/keys`) out of data backups, otherwise shredding is not final.

## Ship a prompt change safely
1. Write `prompts/<role>/vN.md` for every role (content hash is recorded per session).
2. **Gate on evals:** `python -m evals.promote --candidate vN` (blocks on any regression in success, attack success, safety, PII egress, false claims).
3. **Shadow:** `supportagent flags set prompt.shadow_version vN` → live traffic is replayed in dry-run; check `supportagent shadow-report` (tool-call and outcome agreement, shadow cost). Investigate every disagreement.
4. **Canary:** `flags set prompt.canary_version vN` and `flags set prompt.canary_percent 5`, then 25 → 50 → 100 while watching blocks/approvals/escalations. Assignment is sticky per session.
5. **Promote:** `flags set prompt.stable vN` and `flags set prompt.canary_percent 0`.
6. **Rollback:** `flags set prompt.stable <previous>` (one command, audit-chained).

## Routine
- Daily: `supportagent audit verify`; publish the printed `head` hashes to write-once storage (this is what makes tail truncation detectable).
- Daily (cron): `supportagent slo eval --webhook $URL` and `supportagent retention run`.
- Weekly: `make online` (sampled reference-free evals), review guard-block / injection-flag / approval-queue rates; add flagged traces to `evals/build_dataset.py`.
- On every deploy: CI runs `pytest` and the eval gate (`evals/thresholds.json`).
