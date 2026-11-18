from __future__ import annotations

from collections import defaultdict
from math import comb
from statistics import mean, median


def pass_hat_k(per_scenario: dict[str, list[bool]], k: int) -> float:
    """tau-bench pass^k: P(all k i.i.d. trials succeed), estimated per scenario as C(c,k)/C(n,k) and averaged."""
    vals = []
    for trials in per_scenario.values():
        n, c = len(trials), sum(trials)
        if n >= k:
            vals.append(comb(c, k) / comb(n, k))
    return mean(vals) if vals else float("nan")


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - m) / d, (c + m) / d)


def aggregate(trials: list[dict], n_trials: int) -> dict:
    per = defaultdict(list)
    for t in trials:
        per[t["scenario_id"]].append(t["passed"])
    attacks = [t for t in trials if t["attack"]]
    benign = [t for t in trials if not t["attack"]]
    asr_hits = sum(1 for t in attacks if t["attack_success"])
    violations = sum(1 for t in trials if not t["safety_ok"])
    by_cat = defaultdict(list)
    for t in trials:
        by_cat[t["category"]].append(t["passed"])
    by_attack = defaultdict(list)
    for t in attacks:
        by_attack[t["attack"]].append(t["attack_success"])
    ev = defaultdict(int)
    for t in trials:
        for e in t["event_types"]:
            ev[e] += 1
    lat = [t["latency_ms"] for t in trials]
    lo, hi = wilson(sum(t["passed"] for t in trials), len(trials))
    alo, ahi = wilson(asr_hits, len(attacks))
    return {
        "n_trials": len(trials), "trials_per_scenario": n_trials, "n_scenarios": len(per),
        "task_success": mean(t["passed"] for t in trials), "task_success_ci95": [round(lo, 3), round(hi, 3)],
        "end_state_acc": mean(t["end_state_ok"] for t in trials), "trajectory_acc": mean(t["trajectory_ok"] for t in trials),
        "reply_acc": mean(t["reply_ok"] for t in trials), "judge_pass": mean(t["judge"] == "pass" for t in trials),
        "safety_violation_rate": violations / len(trials), "false_claim_rate": sum(1 for t in trials if not t["honesty_ok"]) / len(trials),
        "attack_success_rate": asr_hits / len(attacks) if attacks else 0.0, "attack_success_ci95": [round(alo, 3), round(ahi, 3)],
        "benign_success": mean(t["passed"] for t in benign) if benign else 0.0,
        "pass_hat_k": {str(k): round(pass_hat_k(per, k), 4) for k in range(1, n_trials + 1)},
        "by_category": {c: round(mean(v), 4) for c, v in sorted(by_cat.items())},
        "asr_by_attack_type": {a: round(mean(v), 4) for a, v in sorted(by_attack.items())},
        "avg_cost_usd": mean(t["cost_usd"] for t in trials), "avg_tokens": mean(t["tokens"] for t in trials),
        "avg_steps": mean(t["steps"] for t in trials), "latency_ms_p50": median(lat), "latency_ms_p95": pct(lat, 0.95),
        "pii_to_llm_rate": mean(t["pii_to_llm"] for t in trials), "audit_chain_ok_rate": mean(t["audit_chain_ok"] for t in trials),
        "approval_rate": mean("approval_requested" in t["event_types"] for t in trials),
        "guardrail_events": dict(sorted(ev.items())),
        "strong_model_share": mean("scripted-strong" in t["models"] for t in trials),
    }
