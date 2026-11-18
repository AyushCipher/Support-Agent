"""CI regression gate. Fails the build if quality, safety, reliability or cost regress.

  python -m evals.ci_gate [--trials 4] [--update-baseline]
  python -m evals.ci_gate --heldout [--trials 4] [--update-baseline]
Compares against evals/thresholds.json (absolute floors) and evals/baseline_full.json (relative regression budget).
The held-out gate uses evals/thresholds_heldout.json + evals/baseline_heldout.json the same way.
Baseline files are rewritten ONLY by `--update-baseline` (never by hand); the previous file is archived to evals/results/archive/ first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from supportagent.config import PRESETS

from .runner import run_config

HERE = Path(__file__).parent


def check(summary: dict, thr: dict, baseline: dict | None) -> list[str]:
    fails = []
    if summary["task_success"] < thr["min_task_success"]:
        fails.append(f"task_success {summary['task_success']:.3f} < {thr['min_task_success']}")
    if summary["attack_success_rate"] > thr["max_attack_success_rate"]:
        fails.append(f"attack_success_rate {summary['attack_success_rate']:.3f} > {thr['max_attack_success_rate']}")
    if summary["safety_violation_rate"] > thr["max_safety_violation_rate"]:
        fails.append(f"safety_violation_rate {summary['safety_violation_rate']:.3f} > {thr['max_safety_violation_rate']}")
    if summary["false_claim_rate"] > thr["max_false_claim_rate"]:
        fails.append(f"false_claim_rate {summary['false_claim_rate']:.3f} > {thr['max_false_claim_rate']}")
    if summary["avg_cost_usd"] > thr["max_avg_cost_usd"]:
        fails.append(f"avg_cost_usd {summary['avg_cost_usd']:.4f} > {thr['max_avg_cost_usd']}")
    if summary["pii_to_llm_rate"] > thr["max_pii_to_llm_rate"]:
        fails.append(f"pii_to_llm_rate {summary['pii_to_llm_rate']:.3f} > {thr['max_pii_to_llm_rate']}")
    if summary["audit_chain_ok_rate"] < 1.0:
        fails.append(f"audit chain failed verification in {1 - summary['audit_chain_ok_rate']:.1%} of trials")
    for k, floor in thr["min_pass_hat_k"].items():
        v = summary["pass_hat_k"].get(k)
        if v is not None and v < floor:
            fails.append(f"pass^{k} {v:.3f} < {floor}")
    for cat, floor in thr["required_categories_min"].items():
        if summary["by_category"].get(cat, 0) < floor:
            fails.append(f"category {cat} {summary['by_category'].get(cat, 0):.3f} < {floor}")
    if baseline:
        for m in ("task_success", "trajectory_acc", "reply_acc"):
            if summary[m] < baseline[m] - thr["max_regression_vs_baseline_file"]:
                fails.append(f"REGRESSION {m}: {summary[m]:.3f} vs committed baseline {baseline[m]:.3f}")
    return fails


def check_heldout(summary: dict, thr: dict, baseline: dict | None) -> list[str]:
    """Held-out gate: absolute floors + regression budget vs the committed held-out baseline."""
    fails = []
    if summary["task_success"] < thr["min_task_success"]:
        fails.append(f"heldout task_success {summary['task_success']:.3f} < {thr['min_task_success']}")
    for key, lim, op in (("attack_success_rate", thr["max_attack_success_rate"], ">"), ("safety_violation_rate", thr["max_safety_violation_rate"], ">"),
                         ("false_claim_rate", thr["max_false_claim_rate"], ">"), ("pii_to_llm_rate", thr["max_pii_to_llm_rate"], ">")):
        if summary[key] > lim:
            fails.append(f"heldout {key} {summary[key]:.4f} {op} {lim}")
    if summary["audit_chain_ok_rate"] < 1.0:
        fails.append("heldout audit chain failed verification")
    if baseline:
        if summary["task_success"] < baseline["task_success"] - thr["max_regression_vs_baseline_file"]:
            fails.append(f"HELDOUT REGRESSION task_success: {summary['task_success']:.3f} vs committed baseline {baseline['task_success']:.3f}")
        if summary["safety_violation_rate"] > baseline["safety_violation_rate"] + thr["max_safety_regression_vs_baseline_file"]:
            fails.append(f"HELDOUT REGRESSION safety_violation_rate: {summary['safety_violation_rate']:.4f} vs committed baseline {baseline['safety_violation_rate']:.4f}")
        if summary["attack_success_rate"] > baseline["attack_success_rate"]:
            fails.append(f"HELDOUT REGRESSION attack_success_rate: {summary['attack_success_rate']:.4f} vs committed baseline {baseline['attack_success_rate']:.4f}")
    return fails


def _write_baseline(bfile: Path, s: dict, keys: tuple[str, ...]) -> None:
    import shutil
    import time
    if bfile.exists():                                                  # keep the previous baseline
        arch = HERE / "results" / "archive"
        arch.mkdir(parents=True, exist_ok=True)
        shutil.copy(bfile, arch / f"{bfile.stem}_before_{time.strftime('%Y%m%d_%H%M%S')}.json")
    bfile.write_text(json.dumps({k: s[k] for k in keys}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--heldout", action="store_true", help="gate on evals/scenarios_heldout/ instead of the main suite")
    a = ap.parse_args()
    if a.heldout:
        thr = json.loads((HERE / "thresholds_heldout.json").read_text())
        out = run_config(PRESETS[thr["config"]], a.trials, workers=1, out_name="ci_last_run_heldout", heldout=True)
        s = out["summary"]
        bfile = HERE / "baseline_heldout.json"
        if a.update_baseline:
            _write_baseline(bfile, s, ("task_success", "safety_violation_rate", "attack_success_rate", "false_claim_rate", "pii_to_llm_rate"))
            print("held-out baseline updated")
            return
        fails = check_heldout(s, thr, json.loads(bfile.read_text()) if bfile.exists() else None)
        if fails:
            print("\nHELD-OUT EVAL GATE FAILED:\n  - " + "\n  - ".join(fails))
            sys.exit(1)
        print("\nHELD-OUT EVAL GATE PASSED")
        return
    thr = json.loads((HERE / "thresholds.json").read_text())
    out = run_config(PRESETS[thr["config"]], a.trials, workers=1, out_name="ci_last_run")
    s = out["summary"]
    bfile = HERE / "baseline_full.json"
    if a.update_baseline:
        _write_baseline(bfile, s, ("task_success", "trajectory_acc", "reply_acc", "attack_success_rate"))
        print("baseline updated")
        return
    fails = check(s, thr, json.loads(bfile.read_text()) if bfile.exists() else None)
    if fails:
        print("\nEVAL GATE FAILED:\n  - " + "\n  - ".join(fails))
        sys.exit(1)
    print("\nEVAL GATE PASSED")


if __name__ == "__main__":
    main()
