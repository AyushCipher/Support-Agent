# What only you can do (≈ half a day) — to turn this from "strong repo" into "proven on a real model"

Everything below is honest about what I could not do from inside the build environment (no API key, no human labels, no deploy target).

- [x] **Live eval, Gemini free tier (partial).** Baseline vs full on 60 scenarios (30 main + 30 held-out), `gemini-3.5-flash-lite` / `gemini-3.5-flash`; results lead `docs/results.md`. Run dir `evals/results/live/gemini_main_heldout`.
- [ ] **Finish the 3-trial live run.** Resume with the same command (it skips finished trials): `python -m evals.live --backend gemini --set both --limit-scenarios 60 --config baseline full --trials 3 --workers 2 --rpm 10 --max-requests 450 --run-dir evals/results/live/gemini_main_heldout`, then `make results`. Free tier = 500 req/day for flash-lite and only **20/day** for `gemini-3.5-flash` (the full config's strong tier), reset 08:00 UTC; ~4.5 requests per trial. A paid key finishes it in one sitting.
- [ ] **Finish the live ladder** (`+prompt_v2`, `+io_guards`, rest of `+tool_guards`): `python -m evals.live --backend gemini --set both --limit-scenarios 60 --config +prompt_v2 +io_guards +tool_guards --trials 1 --rpm 10 --run-dir evals/results/live/gemini_ladder_main_heldout`.
- [ ] **Scale the live red team** from the pilot to 20-40 attempts/goal: `python -m evals.redteam.adaptive --live --backend gemini --defender live --layers full --budget 25 --goal-workers 3 --rpm 10 --out redteam_gemini_live` (resumes). The attacker `gemini-3.6-flash` has a 20 req/day free quota, so this needs a paid key or ~2 weeks. Write any success up in `docs/redteam_findings.md` (generated block + analysis).
- [ ] **Fix the open live failures** in `docs/failure_analysis.md`: L1 (simulated user cannot answer the refund-reason question; agent over-asks), L2 (model-chosen refund reason widens the policy window: real safety bug in full), L4 (intent grounding misses "cash back"), MR01 escalation instead of approval. Then re-run baseline and full together (don't mix pre- and post-fix trials).
- [ ] **Anthropic live run** (none recorded): `export ANTHROPIC_API_KEY=...; make eval-live`.
- [ ] **Enable and measure the LLM claim verifier** (`claim_verifier="llm"`) — F5 is only closable with a real model.
- [ ] **Label ≥ 50 transcripts** (`make label`), then `make calibrate` and report Cohen's κ for the rule judge and (with the key set) the LLM judge in `docs/results.md`. Do not claim a calibrated judge until you have this number.
- [ ] **Run an LLM-simulated-user pass** (`evals.simulator.LLMUser`) on the personas and compare to the scripted users.
- [ ] **Record the demo** following `docs/demo_script.md`; embed the video/GIF at the top of the README.
- [ ] **Deploy** (Fly.io / Render / Cloud Run: the Dockerfile is ready, set `SA_BACKEND=anthropic` + secrets) and put the live link in the README. Put the reviewer desk behind auth before exposing it.
- [ ] **Publish** `docs/blog_post.md` with your live numbers; replace the "(simulated)" qualifiers only where the numbers are live.
- [ ] Optional: point `OTEL_EXPORTER_OTLP_ENDPOINT` at Jaeger (`docker compose up`) and screenshot a trace with a guard event for the README.
- [ ] Optional stretch: adaptive red-teaming (attacker model vs. system), hybrid retrieval, structured final answers (see ADR-0006 / F5).

## Added with the enterprise-security pass
- [ ] **Swap HS256 for asymmetric agent identities** (RS256/EdDSA + JWKS) and rotate keys; set `SA_JWT_SECRET` explicitly in deployment (a random one is generated per process otherwise).
- [ ] **Anchor audit heads**: schedule `supportagent audit verify` and publish each `head` to write-once storage.
- [ ] **Set `SA_ADMIN_TOKEN`** and put the admin API / reviewer desk behind real auth (SSO) before exposing them.
- [ ] **Encrypt the PII vault / checkpoints at rest**, and decide the delete-on-request story (`docs/compliance.md`).
- [~] **Live check of PII tokenization**: done for structured PII on the Gemini run (raw PII reached the LLM in 0 full trials vs about two thirds of baseline trials; see `docs/results.md`). Still to do: review a few traces for names/addresses typed in prose (the regex layer will miss some).
- [ ] Try shadow mode and a 5% canary on real traffic (`docs/ops_runbook.md`).
