# Deploying supportagent

Nothing in this repo deploys automatically. These are the exact steps for a single-instance deploy of the API (chat UI at `/`, reviewer desk at `/approvals`, `/metrics`, `/healthz`) with the Gemini backend. State is SQLite on a persistent volume, so run **exactly one instance** (scaling path: ADR-0006).

## What the image guarantees
- Non-root user (uid 10001); app code is root-owned and read-only to it; only `src/` and `policies/` ship (`.dockerignore` keeps `.env`, `runs/`, `evals/`, `tests/`, `docs/` out of the build context).
- No secrets in the image: every key is injected at runtime from the platform's secret store.
- `HEALTHCHECK` on `/healthz`.
- With `SA_ENV=production`, startup **refuses** an insecure configuration and exits 2, listing every problem by variable name (`src/supportagent/security/prodcheck.py`, ADR-0017).

## Required configuration (production)
| Variable | Kind | Value |
|---|---|---|
| `SA_ENV` | env | `production` |
| `SA_BACKEND` | env | `gemini` (or `anthropic` with `ANTHROPIC_API_KEY` and `claude-*` model names) |
| `SA_CONFIG` | env | `full` |
| `SA_MODEL_FAST`, `SA_MODEL_STRONG`, `SA_MODEL_VERIFIER` | env | Gemini model ids, e.g. `gemini-3.5-flash-lite`, `gemini-3.5-flash`, `gemini-3.5-flash` (the defaults are Claude names and are refused with `SA_BACKEND=gemini`) |
| `SA_RUNS_DIR` | env | `/app/runs` (the mounted volume) |
| `GEMINI_API_KEY` | **secret** | your key |
| `SA_VAULT_KEY` | **secret** | long random passphrase; must never change (sealed PII becomes unreadable) and must be backed up separately from the volume |
| `SA_SESSION_SECRET` | **secret** | >= 32 random chars; rotating it logs every customer out of open sessions |
| `SA_ADMIN_TOKEN` | **secret** | >= 32 random chars for the reviewer desk / admin API, or configure `SA_OIDC_JWKS`, `SA_OIDC_ISSUER`, `SA_OIDC_AUDIENCE` instead |
| `SA_CUSTOMER_OIDC_JWKS`, `_ISSUER`, `_AUDIENCE` | env | for real customers: your site's login issues the JWT (`customer_id` or `sub` claim) |
| `SA_PUBLIC_DEMO` | env | `1` ONLY for a public demo on the seeded fake customers (trusts the customer picker). Remove for real customers |
| `SA_GEMINI_RPM` | env (optional) | request pacing for rate-limited keys; `0` = off |

Generate each secret with `python -c "import secrets; print(secrets.token_urlsafe(48))"` and paste it into the platform dashboard, or write `NAME=value` lines to a file outside the repo and feed it to `fly secrets import < file` (keeps values out of shell history). Delete the file afterwards.

## 1. Verify locally first
```bash
make test                                          # full suite, offline
docker build -t supportagent:local .               # EXTRAS defaults to "crypto"
# refusal check: must exit 2 and list problems
docker run --rm -e SA_ENV=production supportagent:local; echo "exit=$?"
# full local run (use throwaway values; put real ones in an env file you do not commit)
docker run --rm -p 8080:8080 --env-file ./prod.env -v sa_runs:/app/runs supportagent:local
curl -s localhost:8080/healthz                     # {"ok": true, "config": "full", "backend": "gemini"}
```
`prod.env` must contain every row of the table above. It is covered by `.gitignore`'s `.env` rule only if you name it `.env`; otherwise keep it outside the repo.

## 2a. Render (render.yaml blueprint)
1. Push the repo to GitHub. In Render: **New > Blueprint**, select the repo; Render reads `render.yaml`.
2. On the first apply Render prompts for `sync: false` values: paste `GEMINI_API_KEY` (and `SA_GEMINI_RPM` or `0`). `SA_VAULT_KEY`, `SA_SESSION_SECRET`, `SA_ADMIN_TOKEN` are generated once by Render (`generateValue`).
3. Copy the generated `SA_VAULT_KEY` from the dashboard into your password manager (it is the only way to read the volume's sealed data after a disaster).
4. Deploy (autoDeploy is off: trigger it manually). The plan must support a persistent disk (`plan: starter` in the blueprint).
5. Check: `curl https://<service>.onrender.com/healthz`. If the deploy fails, the log shows the `refusing to start` list.
6. For real customers: remove `SA_PUBLIC_DEMO`, add the three `SA_CUSTOMER_OIDC_*` variables, redeploy.

## 2b. Fly.io (fly.toml)
```bash
fly auth login
# edit fly.toml: set app = "<unique-name>" and primary_region = "<region>" (both are CHANGE-ME placeholders)
fly apps create <unique-name>
fly volumes create supportagent_runs --size 1 --region <region>
fly secrets set GEMINI_API_KEY=... SA_VAULT_KEY=... SA_SESSION_SECRET=... SA_ADMIN_TOKEN=...   # prefer: fly secrets import < file
fly deploy
fly status && curl -s https://<unique-name>.fly.dev/healthz
fly scale count 1                                   # SQLite: never more than one machine
```
`auto_stop_machines = "off"` keeps the machine up so paused runs and the reviewer desk stay available.

## 3. After the first deploy
- Open `/approvals`, paste the admin token, confirm the queue loads.
- Run one chat from `/` and confirm a `session_token` is returned and the trace panel loads (it sends `X-Session-Token`).
- Schedule (platform cron or an external job): `supportagent slo eval --webhook <url>` and `supportagent retention run` (see [ops_runbook.md](ops_runbook.md)).
- Keep `SA_KEY_DIR` (default `/app/runs/keys`) out of volume snapshots you keep long-term, or crypto-shredding is not final. On a single-volume platform this is a documented limitation; use a KMS-backed key store when available.

## Known limitations of this deploy shape
- One instance, SQLite on a volume: no HA; a volume loss without backups loses sessions and audit history (business records live in the same volume's `shop.db` for this demo).
- Base image is `python:3.12-slim` by tag; pin it by digest for reproducible builds (instructions in the Dockerfile).
- OTLP export is optional (`OTEL_EXPORTER_OTLP_ENDPOINT`); without it traces are only in `runs/traces.jsonl`.
