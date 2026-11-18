"""Statistics over the LIVE (real-model) runs: pass^k, Wilson intervals, paired per-scenario tests, bootstrap CIs, live red team.

Writes evals/results/live/live_stats.json, which docs/results.md, docs/live_results.md and the README results block are generated from.
Nothing here is typed by hand and nothing is extrapolated: every rate is over the scenario-trials that actually completed.

Paired design. baseline and full run the SAME (set, scenario, seed) back to back, so the comparison is paired by scenario:
  * McNemar exact (two-sided binomial on the discordant pairs) on one outcome per scenario. With several trials per scenario the
    trials of one scenario are correlated, so the primary test uses the per-scenario MAJORITY outcome (pass in > half of its trials);
    the seed-0 pairs are reported too (one independent pair per scenario). A trial-level McNemar would overstate n and is not used.
  * Paired cluster bootstrap over scenarios (10,000 resamples, fixed seed) for the difference in mean per-scenario pass rate.
A difference is called significant only if McNemar p < 0.05 AND the bootstrap 95% CI excludes 0.

  python -m evals.live_stats
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path

from .live import LIVE_DIR, read_jsonl
from .metrics import pass_hat_k, wilson

MAIN_RUN = "gemini_main_heldout"
LADDER_RUN = "gemini_ladder_main_heldout"
REDTEAM_RUN = "redteam_gemini_live"
LADDER = ["baseline", "+prompt_v2", "+io_guards", "+tool_guards", "full"]
OUT = LIVE_DIR / "live_stats.json"


# ------------------------------------------------------------------------------------------------ tests
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: b = pairs where only A succeeded, c = only B. p = P(Binomial(b+c, .5) as or more extreme)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def bootstrap_diff(a: dict[str, list[bool]], b: dict[str, list[bool]], iters: int = 10_000, seed: int = 0) -> dict:
    """Paired cluster bootstrap over scenarios present in both: mean_s(rate_b(s)) - mean_s(rate_a(s))."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"diff": None, "ci95": [None, None], "n_scenarios": 0}
    ra = {s: sum(a[s]) / len(a[s]) for s in ids}
    rb = {s: sum(b[s]) / len(b[s]) for s in ids}
    point = sum(rb[s] - ra[s] for s in ids) / len(ids)
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        smp = [rng.choice(ids) for _ in ids]
        diffs.append(sum(rb[s] - ra[s] for s in smp) / len(smp))
    diffs.sort()
    return {"diff": round(point, 4), "ci95": [round(diffs[int(0.025 * iters)], 4), round(diffs[int(0.975 * iters) - 1], 4)], "n_scenarios": len(ids)}


def paired(per_a: dict[str, list[bool]], per_b: dict[str, list[bool]], seed0_a: dict[str, bool], seed0_b: dict[str, bool]) -> dict:
    """A = reference (baseline), B = treatment (full). Outcome True = the event (pass, or attack success, or PII sent)."""
    ids = sorted(set(per_a) & set(per_b))
    maj_a = {s: sum(per_a[s]) * 2 > len(per_a[s]) for s in ids}
    maj_b = {s: sum(per_b[s]) * 2 > len(per_b[s]) for s in ids}
    b = sum(1 for s in ids if maj_a[s] and not maj_b[s])
    c = sum(1 for s in ids if maj_b[s] and not maj_a[s])
    ids0 = sorted(set(seed0_a) & set(seed0_b))
    b0 = sum(1 for s in ids0 if seed0_a[s] and not seed0_b[s])
    c0 = sum(1 for s in ids0 if seed0_b[s] and not seed0_a[s])
    boot = bootstrap_diff(per_a, per_b)
    p = mcnemar_exact(b, c)
    sig = p < 0.05 and boot["ci95"][0] is not None and (boot["ci95"][0] > 0 or boot["ci95"][1] < 0)
    return {"n_scenarios": len(ids), "majority": {"only_A": b, "only_B": c, "both": sum(1 for s in ids if maj_a[s] and maj_b[s]),
                                                 "neither": sum(1 for s in ids if not maj_a[s] and not maj_b[s]), "mcnemar_p": round(p, 5)},
            "seed0": {"n": len(ids0), "only_A": b0, "only_B": c0, "mcnemar_p": round(mcnemar_exact(b0, c0), 5)},
            "bootstrap_B_minus_A": boot, "significant_at_0.05": bool(sig)}


# ------------------------------------------------------------------------------------------------ data
def _dedupe(trials: list[dict]) -> list[dict]:
    seen = {}
    for t in trials:
        seen[(t["config_name"], t["set"], t["scenario_id"], t["seed"])] = t     # last write wins (a resumed trial)
    return list(seen.values())


RESCORE_LOG: dict[str, int] = {}


def rescore(trials: list[dict]) -> list[dict]:
    """Re-apply the CURRENT deterministic safety/honesty scorer to the stored end state, events and replies. The scorer had two false
    positives found in this live run (approval-bypass flagged for a sub-cap refund; a confirmation QUESTION counted as a false success
    claim, see docs/failure_analysis.md L3/L4). Every trial of every config is rescored the same way; the count of changed outcomes is
    reported. No model output is changed."""
    from .runner import load_scenarios
    from .scoring import score_safety
    scn = {("main", s["id"]): s for s in load_scenarios(None)} | {("heldout", s["id"]): s for s in load_scenarios(None, heldout=True)}
    for t in trials:
        s = scn.get((t["set"], t["scenario_id"]))
        if not s:
            continue
        end = dict(t["end_state"], refunds=[tuple(x) for x in t["end_state"]["refunds"]])
        harm_ok, honest_ok, _ = score_safety(s, end, t["events"], t["replies"])
        passed = t["end_state_ok"] and harm_ok and honest_ok and t["reply_ok"] and t["approval_ok"]
        for k, v in (("safety_ok", harm_ok), ("honesty_ok", honest_ok), ("passed", passed), ("attack_success", bool(t["attack"]) and not harm_ok)):
            if t[k] != v:
                RESCORE_LOG[f"{t['config_name']}:{k}"] = RESCORE_LOG.get(f"{t['config_name']}:{k}", 0) + 1
                t[k] = v
    return trials


def load_run(name: str) -> tuple[list[dict], dict, list[dict]]:
    d = LIVE_DIR / name
    meta = json.loads((d / "run_meta.json").read_text()) if (d / "run_meta.json").exists() else {}
    return rescore(_dedupe(read_jsonl(d / "trials.jsonl"))), meta, read_jsonl(d / "errors.jsonl")


def per_scenario(ts: list[dict], field: str) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = defaultdict(list)
    for t in sorted(ts, key=lambda t: t["seed"]):
        out[f"{t['set']}/{t['scenario_id']}"].append(bool(t[field]))
    return dict(out)


def seed0(ts: list[dict], field: str) -> dict[str, bool]:
    return {f"{t['set']}/{t['scenario_id']}": bool(t[field]) for t in ts if t["seed"] == 0}


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [round(max(0.0, lo), 4), round(min(1.0, hi), 4)]}


def group_stats(ts: list[dict]) -> dict:
    per = per_scenario(ts, "passed")
    kmax = max((len(v) for v in per.values()), default=0)
    attacks = [t for t in ts if t["attack"]]
    complete = {s: v for s, v in per.items() if len(v) >= 3}
    return {
        "scenario_trials": len(ts), "scenarios": len(per), "trials_per_scenario_max": kmax,
        "trials_per_scenario_hist": {str(n): sum(1 for v in per.values() if len(v) == n) for n in sorted({len(v) for v in per.values()})},
        "pass": rate(sum(t["passed"] for t in ts), len(ts)),
        "pass_seed0": rate(sum(t["passed"] for t in ts if t["seed"] == 0), sum(1 for t in ts if t["seed"] == 0)),
        "pass_hat_k": {str(k): round(pass_hat_k(per, k), 4) for k in range(1, min(kmax, 3) + 1)},
        "pass_hat_k_scenarios": {str(k): sum(1 for v in per.values() if len(v) >= k) for k in range(1, min(kmax, 3) + 1)},
        "pass_hat_k_on_3trial_scenarios": {str(k): round(pass_hat_k(complete, k), 4) for k in range(1, 4)} if complete else None,
        "scenarios_with_3_trials": len(complete),
        "flaky_scenarios": sorted(s for s, v in complete.items() if 0 < sum(v) < len(v)),
        "attack_success": rate(sum(t["attack_success"] for t in attacks), len(attacks)),
        "attack_scenarios": len({(t["set"], t["scenario_id"]) for t in attacks}),
        "pii_to_llm": rate(sum(t["pii_to_llm"] for t in ts), len(ts)),
        "false_claims": rate(sum(1 for t in ts if not t["honesty_ok"]), len(ts)),
        "safety_violations": rate(sum(1 for t in ts if not t["safety_ok"]), len(ts)),
        "degraded_handoffs": sum(1 for t in ts if t["status"] == "degraded_handoff"),
        "est_cost_usd": round(sum(t["cost_usd"] for t in ts), 4),
        "avg_tokens": round(sum(t["tokens"] for t in ts) / len(ts), 1) if ts else None,
        "models_seen": sorted({m for t in ts for m in t["models"]}),
        "trials_using_strong_model": sum(1 for t in ts if any(m.endswith("flash") or "sonnet" in m for m in t["models"])),
        "trials_with_simulated_model": sum(1 for t in ts if any(str(m).startswith("scripted") for m in t["models"])),
        "verifier_events": sum(1 for t in ts for e in t["event_types"] if e.startswith("verifier")),
    }


def run_window(name: str) -> dict:
    calls = read_jsonl(LIVE_DIR / name / "llm_calls.jsonl")
    ts = [c["ts"] for c in calls if "ts" in c]
    import time as _t
    fmt = lambda x: _t.strftime("%Y-%m-%d %H:%M UTC", _t.gmtime(x))     # noqa: E731
    return {"first_call": fmt(min(ts)) if ts else None, "last_call": fmt(max(ts)) if ts else None,
            "network_calls": sum(1 for c in calls if c.get("cache") == "miss" and "in_tok" in c),
            "retries_logged": sum(1 for c in calls if "retry_in" in c), "est_usd": round(sum(c.get("usd", 0) for c in calls), 4)}


def redteam_stats() -> dict | None:
    d = LIVE_DIR / REDTEAM_RUN
    rows = read_jsonl(d / "attempts.jsonl")
    if not rows:
        return None
    uniq = {(r["goal"], r["layer"], r["i"]): r for r in rows}
    rows = sorted(uniq.values(), key=lambda r: (r["layer"], r["goal"], r["i"]))
    meta = json.loads((d / "live_redteam.json").read_text()) if (d / "live_redteam.json").exists() else {}
    out: dict = {"attacker_model": meta.get("attacker_model"), "defender": meta.get("defender"), "budget_per_goal": meta.get("budget_per_goal"),
                 "window": run_window(REDTEAM_RUN), "layers": {}}
    for layer in sorted({r["layer"] for r in rows}):
        rs = [r for r in rows if r["layer"] == layer]
        goals = defaultdict(list)
        for r in rs:
            goals[r["goal"]].append(r)
        fam = defaultdict(lambda: [0, 0])
        for r in rs:
            fam[r.get("family") or "free"][0] += r["ok"]
            fam[r.get("family") or "free"][1] += 1
        per_goal = {}
        for g, hs in sorted(goals.items()):
            first = next((h for h in hs if h["ok"]), None)
            per_goal[g] = {"attempts": len(hs), "successes": sum(h["ok"] for h in hs), "asr": rate(sum(h["ok"] for h in hs), len(hs)),
                           "ttfs": first["i"] if first else None, "attacker_parse_failures": sum(1 for h in hs if not h.get("attacker_parsed", True)),
                           "degraded": sum(1 for h in hs if h.get("status") == "degraded_handoff")}
        broken = [g for g, v in per_goal.items() if v["successes"]]
        out["layers"][layer] = {"attempts": len(rs), "asr": rate(sum(r["ok"] for r in rs), len(rs)), "goals": len(per_goal), "goals_broken": rate(len(broken), len(per_goal)),
                                "broken_goals": broken, "per_goal": per_goal, "by_family": {f: rate(*v) for f, v in sorted(fam.items())},
                                "attacker_parse_failures": sum(1 for r in rs if not r.get("attacker_parsed", True)),
                                "successes": [{k: r[k] for k in ("goal", "i", "family", "trace_id", "opening", "followups", "note", "kb", "events", "tools", "why", "reply", "end_state")}
                                              for r in rs if r["ok"]]}
    return out


def build() -> dict:
    trials, meta, errors = load_run(MAIN_RUN)
    res: dict = {"main_run": MAIN_RUN, "meta": meta, "window": run_window(MAIN_RUN), "errored_trials": len(errors), "groups": {}, "paired": {}}
    by = defaultdict(list)
    for t in trials:
        by[t["config_name"]].append(t)
        by[f"{t['config_name']}@{t['set']}"].append(t)
    for k, ts in sorted(by.items()):
        res["groups"][k] = group_stats(ts)
    for scope in ("", "@main", "@heldout"):
        a, b = by.get("baseline" + scope, []), by.get("full" + scope, [])
        if not a or not b:
            continue
        att_a, att_b = [t for t in a if t["attack"]], [t for t in b if t["attack"]]
        res["paired"]["all" if not scope else scope[1:]] = {
            "task_success": paired(per_scenario(a, "passed"), per_scenario(b, "passed"), seed0(a, "passed"), seed0(b, "passed")),
            "attack_success": paired(per_scenario(att_a, "attack_success"), per_scenario(att_b, "attack_success"), seed0(att_a, "attack_success"), seed0(att_b, "attack_success")),
            "pii_to_llm": paired(per_scenario(a, "pii_to_llm"), per_scenario(b, "pii_to_llm"), seed0(a, "pii_to_llm"), seed0(b, "pii_to_llm")),
        }
    # ladder: intermediate configs from their own run dir + baseline/full seed 0 from the main run (same scenarios, same seed)
    lt, lmeta, lerr = load_run(LADDER_RUN)
    if lt:
        lad = {"run": LADDER_RUN, "meta": lmeta, "window": run_window(LADDER_RUN), "errored_trials": len(lerr), "rungs": {}, "vs_baseline_seed0": {}}
        base0 = [t for t in by.get("baseline", []) if t["seed"] == 0]
        for cfg in LADDER:
            ts = [t for t in lt if t["config_name"] == cfg] or [t for t in by.get(cfg, []) if t["seed"] == 0]
            if ts:
                lad["rungs"][cfg] = group_stats(ts) | {"source": LADDER_RUN if cfg not in ("baseline", "full") else MAIN_RUN + " (seed 0)"}
                if cfg != "baseline" and base0:
                    lad["vs_baseline_seed0"][cfg] = {
                        "task_success": paired(per_scenario(base0, "passed"), per_scenario(ts, "passed"), seed0(base0, "passed"), seed0(ts, "passed")),
                        "attack_success": paired(per_scenario([t for t in base0 if t["attack"]], "attack_success"), per_scenario([t for t in ts if t["attack"]], "attack_success"),
                                                 seed0([t for t in base0 if t["attack"]], "attack_success"), seed0([t for t in ts if t["attack"]], "attack_success"))}
        res["ladder"] = lad
    res["redteam"] = redteam_stats()
    res["rescored_changes"] = dict(sorted(RESCORE_LOG.items()))
    RESCORE_LOG.clear()
    return res


def main() -> None:
    res = build()
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(f"wrote {OUT}")
    for k, g in res["groups"].items():
        if "@" not in k:
            print(k, g["pass"], g["pass_hat_k"], g["attack_success"])
    for k, p in res["paired"].items():
        print(k, "task", p["task_success"]["majority"], p["task_success"]["bootstrap_B_minus_A"], "attack", p["attack_success"]["majority"])


if __name__ == "__main__":
    main()
