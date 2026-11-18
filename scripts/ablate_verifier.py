"""Measure the pre-money verifier through the EXISTING eval machinery (evals.runner.run_trial + metrics.aggregate).

    python scripts/ablate_verifier.py --trials 4 --out runs/verifier_ablation.json

Runs the `full` preset three ways over all golden scenarios with the simulated model and a simulated honest verifier:
  full                      verifier off (the shipped default; numbers must equal the committed ladder's `full` row for the same trials)
  full + verifier@5000      verifier on, default threshold ($50): only larger / risky money actions are second-opined
  full + verifier@0         verifier on for every money action
and reports success, attack success, pass^k, cost per run, harness latency per run, and how many verifier calls / escalations occurred.

Caveat: the verifier here is a rule-based SIMULATION (llm/verifier_sim.py). Its agreement rate with the harness says nothing about a
real verifier model; what this measures is the harness cost (extra calls, tokens, notional $) and that the pipeline stays correct.
Latency is harness time only: a live verifier adds one real model round trip per verified action (not measured here).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "runs" / "verifier_ablation.json"))
    a = ap.parse_args()
    from evals.metrics import aggregate
    from evals.runner import load_scenarios, run_trial
    from supportagent.config import PRESETS
    from supportagent.shop.db import build_seed_db
    seed = tempfile.mkdtemp(prefix="sa-vab-") + "/seed.db"
    build_seed_db(seed)
    scns = load_scenarios()
    variants = {"full": PRESETS["full"],
                "full+verifier@5000": PRESETS["full"].with_(verifier=True),
                "full+verifier@0": PRESETS["full"].with_(verifier=True, verifier_threshold_cents=0)}
    out = {"trials_per_scenario": a.trials, "scenarios": len(scns), "note": "simulated model + simulated verifier; see script docstring", "variants": {}}
    for name, cfg in variants.items():
        t0 = time.time()
        res = [run_trial(s, cfg, seed_i, seed) for s in scns for seed_i in range(a.trials)]
        agg = aggregate(res, a.trials)
        n = len(res)
        calls = sum(sum(1 for e in r["events"] if e["type"] == "verifier_verdict") for r in res)
        by = {k: sum(sum(1 for e in r["events"] if e["type"] == "verifier_verdict" and e["status"] == k) for r in res) for k in ("agree", "disagree", "unavailable")}
        out["variants"][name] = {
            "task_success": agg["task_success"], "attack_success_rate": agg["attack_success_rate"], "pass_hat_k": agg["pass_hat_k"],
            "avg_cost_usd": agg["avg_cost_usd"], "verifier_calls_total": calls, "verifier_calls_per_run": round(calls / n, 4), "verdicts": by,
            "approvals_requested_per_run": round(sum(sum(1 for e in r["events"] if e["type"] == "approval_requested") for r in res) / n, 4),
            "mean_run_latency_ms": round(sum(r["latency_ms"] for r in res) / n, 2), "wall_seconds": round(time.time() - t0, 1)}
        print(name, json.dumps(out["variants"][name]))
    base = out["variants"]["full"]
    for name, v in out["variants"].items():
        v["delta_vs_full"] = {"task_success": round(v["task_success"] - base["task_success"], 4), "avg_cost_usd": round(v["avg_cost_usd"] - base["avg_cost_usd"], 6),
                              "mean_run_latency_ms": round(v["mean_run_latency_ms"] - base["mean_run_latency_ms"], 2)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
