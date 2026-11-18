"""Extra generated sections of docs/results.md: held-out table, adaptive red-team table, judge calibration status, live pointer.
All numbers come from JSON under evals/results/ (never typed by hand)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
DOCS = HERE.parent / "docs"
ORDER = ["baseline", "plus_prompt_v2", "plus_io_guards", "plus_tool_guards", "full"]
LADDER_TO_FILE = {"baseline": "baseline", "+prompt_v2": "plus_prompt_v2", "+io_guards": "plus_io_guards", "+tool_guards": "plus_tool_guards", "full": "full"}


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def ci_str(lo_hi) -> str:
    return f"{pct(lo_hi[0])}-{pct(lo_hi[1])}"


def load_heldout() -> dict:
    out = {}
    for n in ORDER:
        f = HERE / "results" / "heldout" / f"{n}.json"
        if f.exists():
            out[n] = json.loads(f.read_text())
    return out


def heldout_md(ho: dict, label: dict) -> list[str]:
    if not ho:
        return []
    any_ = next(iter(ho.values()))["summary"]
    L = ["", "## Held-out scenario set (SIMULATED model)", "",
         f"**{any_['n_scenarios']} scenarios x {any_['trials_per_scenario']} trials**, authored by Claude (NOT human-labeled; see `evals/scenarios_heldout/PROVENANCE.md`), "
         "never used to tune a guard or the simulator. Covers new phrasings, multi-turn/ambiguous requests, cross-customer attempts, policy edge cases "
         "(boundaries, partial refunds, bundles, rounding, caps) and tool failures/timeouts. Same simulated model and seeded failure rates as the main suite: "
         "this tests whether the guards and scenario coverage generalise, not how any real model performs.", "",
         "| Configuration | Task success (95% CI) | pass^1 | pass^8 | Attack success (95% CI) | PII sent to LLM | False-claim rate | Safety violations |", "|---|---|---|---|---|---|---|---|"]
    for n, r in ho.items():
        s = r["summary"]
        k = s["trials_per_scenario"]
        L.append(f"| {label[n]} | {pct(s['task_success'])} ({ci_str(s['task_success_ci95'])}) | {pct(s['pass_hat_k']['1'])} | {pct(s['pass_hat_k'][str(k)])} | "
                 f"**{pct(s['attack_success_rate'])}** ({ci_str(s['attack_success_ci95'])}) | {pct(s['pii_to_llm_rate'])} | {pct(s['false_claim_rate'])} | {pct(s['safety_violation_rate'])} |")
    if "full" in ho and "baseline" in ho:
        L += ["", "### Held-out success by category (baseline vs final)", "", "| Category | Baseline | Final |", "|---|---|---|"]
        fb = ho["baseline"]["summary"]["by_category"]
        L += [f"| {c} | {pct(fb.get(c, 0))} | {pct(v)} |" for c, v in ho["full"]["summary"]["by_category"].items()]
        per: dict = {}
        for t in ho["full"]["trials"]:
            per.setdefault(t["scenario_id"], []).append(t["passed"])
        bad = sorted(((sid, sum(v), len(v)) for sid, v in per.items() if sum(v) < len(v)), key=lambda x: x[1] / x[2])
        L += ["", f"### Held-out scenarios the final configuration does NOT always pass ({len(bad)} of {len(per)})", "",
              "Every one is a finding to report, not something to tune away; each is classified in `docs/failure_analysis.md` (H-series).", "",
              "| Scenario | Passed trials |", "|---|---|"]
        L += [f"| {sid} | {k}/{n} |" for sid, k, n in bad]
    return L


def adaptive_md(label: dict) -> list[str]:
    f = HERE / "results" / "redteam" / "adaptive_offline.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text())
    sm = d["summary"]
    L = ["", "## Adaptive red-team (SIMULATED defender model, offline mutation-search attacker)", "",
         f"Deterministic attacker with a budget of **{d['budget_per_goal']} attempts per attack goal** ({len(d['goals'])} goals, {d['repeats']} attacker seed(s)), "
         "mutating attacks from the defender's guard events (operators: paraphrase, encoding, role-play, multi-turn escalation, payload splitting, indirect via order notes, "
         "indirect via KB content). The **static control** replays the unmutated base attack with the same seeds, which separates real adaptivity from re-rolling the simulated "
         "model's gullibility draw. Attempts within a run are adaptive and not independent, so the Wilson intervals are optimistic. Findings: `docs/redteam_findings.md`.", "",
         "| Configuration | Adaptive: successful attempts (ASR, 95% CI) | Goals broken within budget | Fewest attempts to first success | Static control ASR |", "|---|---|---|---|---|"]
    for layer, fname in LADDER_TO_FILE.items():
        a, c = sm.get("adaptive", {}).get(layer), sm.get("static_control", {}).get(layer)
        if not a:
            continue
        cs = f"{c['successes']}/{c['attempts']} ({pct(c['asr'])})" if c else "n/a"
        L.append(f"| {label[fname]} | {a['successes']}/{a['attempts']} ({pct(a['asr'])}; {ci_str(a['asr_ci95'])}) | {a['goals_broken']}/{a['goal_runs']} | {a['min_ttfs'] or 'none'} | {cs} |")
    full = sm.get("adaptive", {}).get("full")
    if full:
        wins = ", ".join(f"{k} x{v}" for k, v in full["family_wins"].items() if v)
        L += ["", f"Against the final configuration: **{full['successes']}/{full['attempts']}** adaptive attempts succeeded"
              + (f" (families present in winning chains: {wins})." if full["successes"] else " (no attack family succeeded within the budget).")]
    pr = d.get("probe_tool_description_poisoning")
    if pr:
        L += ["", "Tool-description poisoning (structural probe, not a search): a third-party tool with an injected description is registered; "
              f"offered to any agent role: baseline={pr['baseline']['offered_to_any_role']}, final={pr['full']['offered_to_any_role']}; calling it returns `{pr['full']['call_result']}`."]
    return L


def live_pointer() -> list[str]:
    if (DOCS / "live_results.md").exists():
        return ["", "## Live model results", "", "Real-model runs are reported separately in [`docs/live_results.md`](live_results.md) (generated by `make eval-live-report`). "
                "Nothing in this file is a live number."]
    return []


def calibration_md() -> list[str]:
    try:
        from .calibrate import NOT_MEASURED, human_status
        try:
            st = human_status()
        except Exception:
            st = {"line": NOT_MEASURED}
    except Exception:
        st = {"line": "Cohen's kappa vs human labels: NOT YET MEASURED"}
    return ["", "## Judge calibration", "", f"**{st['line']}**", "",
            "The default judge is deterministic (`RuleJudge`). Human labels come from `make label`; an LLM-labeled agreement figure "
            "(`python -m evals.calibrate --llm-labeled`) is a development aid only and is never presented as human kappa."]
