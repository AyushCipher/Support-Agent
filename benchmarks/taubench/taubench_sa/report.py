"""results/<run>/results.jsonl -> results/<run>/summary.json -> docs/external_benchmark.md (the doc is generated from the JSON).

  python -m taubench_sa.report --run retail_test
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .run import RESULTS, read_jsonl

DOC = Path(__file__).resolve().parents[3] / "docs" / "external_benchmark.md"
TAU_COMMIT = "59a200c6d575d595120f1cb70fea53cef0632f6b"
N_TEST_TASKS = 115


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pass_hat_k(per_task: dict[int, list[float]], k: int) -> tuple[float, int]:
    """tau-bench's pass^k: mean over tasks with >= k trials of C(c, k) / C(n, k)."""
    vals = [math.comb(sum(1 for r in rs if r >= 1.0), k) / math.comb(len(rs), k) for rs in per_task.values() if len(rs) >= k]
    return (sum(vals) / len(vals) if vals else float("nan"), len(vals))


def summarize(run: str) -> dict:
    d = RESULTS / run
    res = read_jsonl(d / "results.jsonl")
    done_keys = {(r["agent"], r["task_id"], r["trial"]) for r in res}
    errs_all = read_jsonl(d / "errors.jsonl")
    errs = [e for e in errs_all if (e["agent"], e["task_id"], e["trial"]) not in done_keys]     # still unresolved
    retried_ok = len(errs_all) - len(errs)
    metas = read_jsonl(d / "run_meta.jsonl")
    # last record wins if an episode was ever written twice
    uniq = {(r["agent"], r["task_id"], r["trial"]): r for r in res}
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for r in uniq.values():
        by_agent[r["agent"]].append(r)
    agents = {}
    for ag, rs in sorted(by_agent.items()):
        per_task: dict[int, list[float]] = defaultdict(list)
        for r in sorted(rs, key=lambda x: x["trial"]):
            per_task[r["task_id"]].append(r["reward"])
        t1 = [r for r in rs if r["trial"] == 1]
        k1 = sum(r["reward"] >= 1.0 for r in t1)
        pooled_k = sum(r["reward"] >= 1.0 for r in rs)
        max_trials = max(len(v) for v in per_task.values())
        a = {
            "episodes": len(rs), "tasks": len(per_task), "trial1_episodes": len(t1),
            "pass1_trial1": {"k": k1, "n": len(t1), "rate": k1 / len(t1) if t1 else None, "wilson95": wilson(k1, len(t1))},
            "pooled_all_trials": {"k": pooled_k, "n": len(rs), "rate": pooled_k / len(rs), "wilson95": wilson(pooled_k, len(rs))},
            "pass_hat_k": {str(k): dict(zip(("value", "tasks"), pass_hat_k(per_task, k))) for k in range(1, max_trials + 1)},
            "terminations": dict(Counter(r["termination"] for r in rs)),
            "failure_categories": dict(Counter(r["category"] for r in rs if r["category"] != "pass")),
            "mean_agent_calls": round(sum(r["n_agent_calls"] for r in rs) / len(rs), 2),
            "calls_by_model": dict(sum((Counter(r.get("calls_by_model") or {}) for r in rs), Counter())),
            "errors_excluded": sum(1 for e in errs if e["agent"] == ag),
        }
        if ag.startswith("supportagent"):
            gb = sum((Counter(r.get("guard_blocks") or {}) for r in rs), Counter())
            a["guardrails"] = {
                "guard_blocks_by_code": dict(gb), "episodes_with_guard_block": sum(1 for r in rs if r.get("guard_blocks")),
                "approvals_requested": sum(r.get("approvals", 0) for r in rs), "output_guard_blocks": sum(r.get("output_guard_blocks", 0) for r in rs),
                "schema_rejects": sum(r.get("schema_rejects", 0) for r in rs), "clarification_failures": sum(r.get("clarification_failures", 0) for r in rs),
                "tainted_episodes": sum(1 for r in rs if r.get("tainted")),
                "pass_rate_with_guard_block": _rate([r for r in rs if r.get("guard_blocks")]),
                "pass_rate_without_guard_block": _rate([r for r in rs if not r.get("guard_blocks")]),
            }
        agents[ag] = a
    pairs = []
    sa = [a for a in agents if a.startswith("supportagent")]
    for s in sa:
        for b in [a for a in agents if not a.startswith("supportagent")]:
            A = {(r["task_id"], r["trial"]): r["reward"] >= 1.0 for r in by_agent[s]}
            B = {(r["task_id"], r["trial"]): r["reward"] >= 1.0 for r in by_agent[b]}
            shared = sorted(set(A) & set(B))
            both = sum(A[x] and B[x] for x in shared)
            only_s = sum(A[x] and not B[x] for x in shared)
            only_b = sum(B[x] and not A[x] for x in shared)
            neither = len(shared) - both - only_s - only_b
            pairs.append({"supportagent": s, "baseline": b, "shared_episodes": len(shared), "both_pass": both, "only_supportagent": only_s,
                          "only_baseline": only_b, "neither": neither, "mcnemar_exact_p": mcnemar_exact(only_s, only_b),
                          "supportagent_rate": (both + only_s) / len(shared) if shared else None,
                          "baseline_rate": (both + only_b) / len(shared) if shared else None,
                          "tasks_only_supportagent": sorted(x[0] for x in shared if A[x] and not B[x]),
                          "tasks_only_baseline": sorted(x[0] for x in shared if B[x] and not A[x])})
    return {"benchmark": "tau-bench retail, test split", "tau_bench_commit": TAU_COMMIT, "n_test_tasks": N_TEST_TASKS, "run": run,
            "user_model": sorted({r.get("user_model") for r in res}), "agents": agents, "paired": pairs,
            "errors_later_rerun_ok": retried_ok, "errors": [{k: e[k] for k in ("agent", "task_id", "trial", "error")} for e in errs],
            "run_meta": metas, "network_requests_total": sum(m.get("network_requests_this_run", 0) for m in metas),
            "stops": [m["stopped"] for m in metas if m.get("stopped")]}


def _rate(rs: list[dict]) -> dict:
    k = sum(r["reward"] >= 1.0 for r in rs)
    return {"k": k, "n": len(rs), "rate": k / len(rs) if rs else None}


def pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def ci(w) -> str:
    return "n/a" if any(math.isnan(v) for v in w) else f"{100 * w[0]:.1f}-{100 * w[1]:.1f}%"


def render(S: dict) -> str:
    A = S["agents"]
    L: list[str] = []
    w = L.append
    w("# External benchmark: tau-bench retail")
    w("")
    w("> Generated by `benchmarks/taubench/taubench_sa/report.py` from `benchmarks/taubench/results/" + S["run"] + "/summary.json`. Do not edit by hand.")
    w("")
    w("Everything else in `docs/` that measures task success uses scenarios written by this project. This page reports supportagent on a "
      "**public benchmark it was not designed for**: [tau-bench](https://github.com/sierra-research/tau-bench) (Sierra, MIT license), retail domain, "
      f"test split ({S['n_test_tasks']} tasks), pinned at commit `{S['tau_bench_commit'][:7]}`. tau-bench's own LLM user simulator produces the customer "
      "turns and tau-bench's own reward (final database state + required information in replies) decides pass/fail. Nothing in the adapter changes scoring.")
    w("")
    w("## What ran")
    w("")
    w("| Agent | Episodes completed | Distinct tasks | Episodes per task | Infra errors (excluded) | Mean agent model calls / episode |")
    w("|---|---|---|---|---|---|")
    for ag, a in A.items():
        w(f"| `{ag}` | {a['episodes']} | {a['tasks']} of {S['n_test_tasks']} | {a['episodes'] / max(1, a['tasks']):.2f} | {a['errors_excluded']} | {a['mean_agent_calls']} |")
    w("")
    w(f"User simulator: `{', '.join(x for x in S['user_model'] if x)}` (tau-bench `LLMUserSimulationEnv`, unmodified prompt). "
      f"Network requests to the model provider across all runs: {S['network_requests_total']}. "
      f"Episodes that crashed and were later re-run to completion: {S['errors_later_rerun_ok']}; still unresolved (excluded): {len(S['errors'])}"
      + ("" if not S["errors"] else " (" + "; ".join(f"{e['agent'].split('[')[0]} task {e['task_id']}: {e['error'][:80]}" for e in S["errors"]) + ")") + ".")
    if S["stops"]:
        w("")
        w("Runs stopped early: " + "; ".join(s[0] for s in S["stops"] if s) + ". Counts above are exactly what completed; nothing is extrapolated.")
    w("")
    w("## Results")
    w("")
    w("pass^1 = share of tasks solved on trial 1, with a Wilson 95% interval. pass^k (tau-bench's definition) is the probability that all k i.i.d. "
      "trials of a task succeed, averaged over the tasks that have at least k trials.")
    w("")
    w("| Agent | pass^1 (trial 1) | 95% CI | All trials pooled | " + " | ".join(f"pass^{k}" for k in range(2, 5)) + " |")
    w("|---|---|---|---|" + "---|" * 3)
    for ag, a in A.items():
        p1 = a["pass1_trial1"]
        pk = [a["pass_hat_k"].get(str(k)) for k in range(2, 5)]
        cells = [f"{pct(x['value'])} (n={x['tasks']})" if x else "-" for x in pk]
        po = a["pooled_all_trials"]
        w(f"| `{ag}` | {p1['k']}/{p1['n']} = **{pct(p1['rate'])}** | {ci(p1['wilson95'])} | {po['k']}/{po['n']} = {pct(po['rate'])} | " + " | ".join(cells) + " |")
    w("")
    for p in S["paired"]:
        w(f"**Paired comparison** on the {p['shared_episodes']} (task, trial) episodes both agents completed: supportagent "
          f"{pct(p['supportagent_rate'])} vs baseline {pct(p['baseline_rate'])}. Both pass {p['both_pass']}, only supportagent {p['only_supportagent']}, "
          f"only baseline {p['only_baseline']}, neither {p['neither']}. Exact McNemar p = {p['mcnemar_exact_p']:.3g}"
          + (" (no significant difference at 0.05)." if p["mcnemar_exact_p"] >= 0.05 else "."))
        w("")
        w(f"Tasks only supportagent solved: {p['tasks_only_supportagent'] or 'none'}. Tasks only the baseline solved: {p['tasks_only_baseline'] or 'none'}.")
        w("")
    w("### Failure categories (rule-based, from the episode data)")
    w("")
    cats = sorted({c for a in A.values() for c in a["failure_categories"]})
    w("| Category | " + " | ".join(f"`{ag}`" for ag in A) + " |")
    w("|---|" + "---|" * len(A))
    for c in cats:
        w(f"| {c} | " + " | ".join(str(a["failure_categories"].get(c, 0)) for a in A.values()) + " |")
    w("")
    w("Definitions: `missing_write` no consequential action executed although the task needs one; `wrong_write` executed actions differ from the "
      "ground-truth actions (wrong item/payment/order, extra or missing action); `unexpected_write` acted on a task that needs no write; "
      "`guard_blocked_required_write` a needed write was blocked by supportagent's guard and never executed; `transferred_to_human` ended by "
      "transfer when the task did not call for it; `step_cap` 30 agent model calls without the conversation ending; `missing_required_info` "
      "database correct but a required answer (e.g. a total) was not stated; `other` database-hash mismatch not explained by the action comparison.")
    w("")
    w("### Terminations")
    w("")
    terms = sorted({t for a in A.values() for t in a["terminations"]})
    w("| Termination | " + " | ".join(f"`{ag}`" for ag in A) + " |")
    w("|---|" + "---|" * len(A))
    for t in terms:
        w(f"| {t} | " + " | ".join(str(a["terminations"].get(t, 0)) for a in A.values()) + " |")
    w("")
    for ag, a in A.items():
        g = a.get("guardrails")
        if not g:
            continue
        w(f"### What supportagent's guardrails did (`{ag}`)")
        w("")
        w(f"- Guard blocks by code: {json.dumps(g['guard_blocks_by_code']) if g['guard_blocks_by_code'] else 'none'} "
          f"({g['episodes_with_guard_block']} of {a['episodes']} episodes had at least one block).")
        gw, gn = g["pass_rate_with_guard_block"], g["pass_rate_without_guard_block"]
        w(f"- Pass rate in episodes with a guard block: {gw['k']}/{gw['n']} ({pct(gw['rate'])}); without: {gn['k']}/{gn['n']} ({pct(gn['rate'])}). "
          "Correlational only: blocks happen in harder conversations.")
        w(f"- Human-approval requests (money above the $100 autonomous cap or tainted session; approved by the eval auto-approver): {g['approvals_requested']}.")
        w(f"- Output-guard repair prompts: {g['output_guard_blocks']}; schema rejections: {g['schema_rejects']}; `ask_user` calls that returned an error (invalid arguments or no valid answer): "
          f"{g['clarification_failures']}; sessions tainted by injection heuristics: {g['tainted_episodes']}.")
        w(f"- Model calls by model (router: fast by default, strong after a risk signal): {json.dumps(a['calls_by_model'])}.")
        w("")
    w(STATIC)
    return "\n".join(L) + "\n"


STATIC = """## Models, and why every role ended up on gemini-3.5-flash-lite

Planned: supportagent fast = gemini-3.5-flash-lite, strong = gemini-3.5-flash; user simulator = gemini-3.6-flash; baseline on the same model.
The key is on the Gemini **free tier**. Observed from the provider's 429 responses on 2026-11-11: gemini-3.6-flash allows 20 generate requests
per day (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, quotaValue 20), and requests to gemini-3.5-flash were rejected against that same
gemini-3.6-flash quota dimension; it was exhausted after 18 requests of the first smoke attempt (no episode finished, nothing recorded).
gemini-3.5-flash-lite allows 500 requests per day and is shared with the project's other live eval running at the same time.
So the user simulator, both supportagent tiers and the baseline all use **gemini-3.5-flash-lite** (supportagent's router therefore has no
stronger model to escalate to). Upside: the paired comparison is exactly like-for-like (same model for both agents and for the user).
Downside: a weak model plays the customer, which adds noise (e.g. ending the conversation early), and the sample is small.

## Status of this run and what the numbers can and cannot say

- Quota, not design, set the sample size: both daily free-tier buckets were exhausted on 2026-11-11 (gemini-3.5-flash-lite at 08:32 UTC, after
  this project's other live eval had used part of the same 500). Buckets reset at 08:00 UTC (midnight US Pacific). Episodes in flight at the stop
  were not recorded; their model responses are cached, so `taubench_sa.run` with the same arguments resumes them for free and continues.
- With 5 shared tasks the intervals span roughly 15-80% and McNemar cannot distinguish the agents. **No claim that either agent is better is
  supported.** 3 trials per task (pass^k) were not reached.
- Reading individual episodes: in task 1 supportagent's confirmation guard blocked an exchange the model attempted without a customer "yes"
  (tau-bench's policy requires one); the model then asked, but the simulated customer ended the chat instead of confirming, so the needed write
  never happened. The confirmation question it then asked described the ground-truth exchange (thermostat only, difference to the
  original card), so here enforcing tau-bench's own rule most likely cost a pass, together with a weak user simulator. In tasks 3 and 4 the same block was followed by a confirmation and the write executed.
- One baseline episode (task 2) crashed inside tau-bench's reward code: the model returned an empty turn, the stand-in passed `content=None`,
  and tau-bench calls `.lower()` on it. The stand-in now returns `""` for an empty turn; the episode stays excluded until it is re-run.

## How supportagent is plugged in

`benchmarks/taubench/taubench_sa/adapter.py` makes supportagent's `Runtime` the agent. tau-bench's 16 retail tools are exposed as a tool
registry (the interface the MCP registry offers), so every call goes through the same tool-guard path as in production:
allowlist -> schema validation -> hidden, session-injected `acting_customer_id` -> hidden idempotency key -> `RetailToolGuard` -> dispatch
(`env.step`) -> truncation -> tool-output sanitizer -> PII tokenization -> untrusted-data envelope -> evidence -> audit chain. Output guard,
input guard, PII vault, approvals and the fast->strong model router are on (`full` preset). The baseline is tau-bench's own
`ToolCallingAgent`, unmodified, given only tau-bench's policy as its system prompt.

Both agents and the user simulator call Gemini through one stdlib client (`supportagent/llm/gemini_client.py`); tau-bench's `litellm` import is
satisfied by a small stand-in (`taubench_sa/llm.py`) that converts OpenAI-format messages, so tau-bench's agent and user code run unmodified.
Both agents get the same limit of 30 agent model calls per episode (tau-bench's `max_num_steps`) and the same 4096 max output tokens
(Gemini thinking tokens count against it).

## Where supportagent's policies differ from tau-bench's retail policy (tau-bench's policy governs scoring)

| Topic | supportagent (own domain) | tau-bench retail policy | Resolution in the adapter |
|---|---|---|---|
| Identity | customer is pre-authenticated by login; `acting_customer_id` comes from the session | agent must authenticate via email or name+zip, even if the user gives a user id | session starts unauthenticated; the harness binds identity from a successful `find_user_id_*` result (never from model text); account/order tools are blocked until then; ownership of every `user_id` / `order_id` checked against env data |
| Consent for actions | intent grounding: the customer must have asked for a refund/cancel | explicit "yes" after the agent lists the action details | intent grounding replaced by confirmation grounding: a consequential write is blocked unless the customer's latest message (or `ask_user` answer) is affirmative |
| Eligibility | refund windows and checks in `policies/*.md` via `check_refund_eligibility` | status rules only (cancel/modify need `pending`, return/exchange need `delivered`) | supportagent's refund policy is not used; the guard checks tau-bench's status precondition against env data |
| Money caps | refunds above $100 need human approval | no approval concept | approval path kept (money moved > $100 or tainted session); the eval auto-approver approves, so outcomes are unaffected but approvals are counted |
| Write blast radius | max 4 successful writes per session | no limit | kept: the test split needs at most 4 writes per task |
| Tool calls per turn | executes every call in a model response | "at most one tool call at a time" | not enforced (tau-bench does not score it); parallel reads disabled; the baseline executes only the first call, as tau-bench does |
| Handoff | escalates to a human on budget/guard failure/outage | transfer only if out of scope; transfer ends the episode | supportagent's escalation is mapped to `transfer_to_human_agents` and scored by tau-bench like any transfer |
| Prompts / routing | triage + orders/billing/policy specialists with their own prompts | one agent, policy document as system prompt | single least-privilege role `tau_retail`; system prompt = tau-bench policy + 4-line harness preamble (untrusted tool output, PII placeholders, guard errors, `ask_user`) |
| PII | emails/phones tokenized before the model; restored only for the owner | none | kept; tool arguments are de-tokenized before reaching tau-bench |

Changed defaults (all others are the `full` preset): tool result cap 1800 -> 6000 chars (29 of 50 product records exceed 1800 and would be cut),
context budget 6000 -> 32000 estimated tokens and keep-last 8 -> 16 messages, parallel tool calls and read cache off (tau-bench state changes under
the agent), harness step/token/cost/time budgets raised so the shared 30-call cap governs.

## Reference numbers (context only, not comparable)

tau-bench's README (commit `59a200c`) reports retail pass^1 of 0.692 for tool-calling claude-3-5-sonnet-20241022, 0.626 for
claude-3-5-sonnet-20240620 and 0.604 for gpt-4o, with gpt-4o as the user simulator. Those runs used different agent models, a different user
simulator and possibly different task subsets, so no cross-model or cross-run comparison is claimed. The only like-for-like comparison on this page is the
paired one above (same model, same user simulator, same tasks).

## Limitations

- tau-bench's README warns that this repository's retail tasks are outdated; fixed versions live in tau2/tau3-bench (Python >= 3.12 and a much
  larger dependency set, so not used here). Label noise affects both agents equally but caps the attainable score.
- The user simulator is an LLM; episodes are stochastic even at temperature 0. Wilson intervals treat tasks as independent draws.
- supportagent's prompts, regexes and guard thresholds were built for its own domain; this measures transfer, not a tuned system.
- Cost is not reported: provider prices for these models are placeholders in `config.PRICING`.

## Reproduce

```
bash benchmarks/taubench/fetch_vendor.sh                 # pinned clone into benchmarks/taubench/vendor/ (git-ignored)
python -m venv benchmarks/taubench/.venv && benchmarks/taubench/.venv/Scripts/pip install pydantic==2.9.2 pytest==8.3.3
cd benchmarks/taubench
PYTHONPATH="../../src;." .venv/Scripts/python -m pytest -q tests                        # offline, scripted model
PYTHONPATH="../../src;." .venv/Scripts/python -m taubench_sa.run --agents supportagent baseline --tasks 0-114 --trials 1 --workers 4 --max-requests 6000
PYTHONPATH="../../src;." .venv/Scripts/python -m taubench_sa.report --run retail_test
```
(`;` separates PYTHONPATH entries on Windows, use `:` elsewhere.) `GEMINI_API_KEY` is read from the environment or the git-ignored `.env`.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="retail_test")
    ap.add_argument("--no-doc", action="store_true")
    a = ap.parse_args(argv)
    S = summarize(a.run)
    (RESULTS / a.run / "summary.json").write_text(json.dumps(S, indent=1), encoding="utf-8")
    if not a.no_doc:
        DOC.write_text(render(S), encoding="utf-8")
    print(json.dumps({ag: {"pass1": x["pass1_trial1"], "pooled": x["pooled_all_trials"]} for ag, x in S["agents"].items()}, indent=1))
    for p in S["paired"]:
        print(json.dumps({k: p[k] for k in ("shared_episodes", "both_pass", "only_supportagent", "only_baseline", "neither", "mcnemar_exact_p")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
