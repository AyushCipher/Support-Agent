"""Judge calibration: agreement between a judge and HUMAN labels (Cohen's kappa, accuracy, confusion matrix, bootstrap CI).

Workflow:  make label     -> sample transcripts, label pass/fail in the terminal -> evals/labels/human_labels.jsonl
           make calibrate -> judge-vs-human agreement for RuleJudge (and LLMJudge when a key is set)
Do not trust any judge whose kappa you have not measured on your own labels.

Until human labels exist the ONLY thing this module reports is:   Cohen's kappa vs human labels: NOT YET MEASURED

Clearly separated development aid (never a substitute, never reported as human kappa):
  `--llm-labeled` has an LLM label transcripts into evals/labels/llm_labels.jsonl and reports "LLM-labeled agreement" for the rule judge.
  That number measures agreement between two automatic raters; it says nothing about agreement with a human.

Multi-rater: human_labels.jsonl rows may carry a "rater" field; with >= 2 raters on the same trials, pairwise Cohen's kappa and Fleiss'
kappa are reported (inter-rater reliability: how much the humans agree with EACH OTHER, the ceiling for any judge).
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
LABELS = HERE / "labels" / "human_labels.jsonl"
LLM_LABELS = HERE / "labels" / "llm_labels.jsonl"
NOT_MEASURED = "Cohen's kappa vs human labels: NOT YET MEASURED"


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def bootstrap_kappa(a: list[str], b: list[str], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for Cohen's kappa (resamples item pairs). Degenerate resamples (kappa undefined) are skipped.
    With few labels the interval is wide: that is the honest answer."""
    n = len(a)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    ks = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        k = cohen_kappa([a[i] for i in idx], [b[i] for i in idx])
        if k == k:                                                 # not NaN
            ks.append(k)
    if not ks:
        return (float("nan"), float("nan"))
    ks.sort()
    return (ks[int(alpha / 2 * len(ks))], ks[min(len(ks) - 1, int((1 - alpha / 2) * len(ks)))])


def fleiss_kappa(ratings: list[list[str]]) -> float:
    """ratings[i] = the labels that item i received (same number of raters per item, >= 2)."""
    n_items = len(ratings)
    if n_items == 0:
        return float("nan")
    m = len(ratings[0])
    assert all(len(r) == m for r in ratings) and m >= 2, "Fleiss' kappa needs the same number (>=2) of raters on every item"
    cats = sorted({c for r in ratings for c in r})
    p_j = {c: sum(r.count(c) for r in ratings) / (n_items * m) for c in cats}
    p_i = [(sum(r.count(c) ** 2 for c in cats) - m) / (m * (m - 1)) for r in ratings]
    pbar, pe = sum(p_i) / n_items, sum(v * v for v in p_j.values())
    return 1.0 if pe == 1 else (pbar - pe) / (1 - pe)


def agreement(human: list[str], judge: list[str]) -> dict:
    cm = {}
    for h, j in zip(human, judge):
        cm[f"human={h},judge={j}"] = cm.get(f"human={h},judge={j}", 0) + 1
    return {"n": len(human), "accuracy": sum(h == j for h, j in zip(human, judge)) / max(1, len(human)),
            "kappa": cohen_kappa(human, judge), "confusion": cm}


def read_labels(path: Path = LABELS) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def inter_rater(rows: list[dict]) -> dict | None:
    """Pairwise Cohen + Fleiss over trials labelled by every rater. None unless >= 2 raters share >= 2 trials."""
    by_rater: dict[str, dict[str, str]] = defaultdict(dict)
    for r in rows:
        by_rater[r.get("rater", "rater1")][r["trial_id"]] = r["label"]
    if len(by_rater) < 2:
        return None
    shared = sorted(set.intersection(*(set(v) for v in by_rater.values())))
    if len(shared) < 2:
        return None
    raters = sorted(by_rater)
    pairs = {f"{a} vs {b}": round(cohen_kappa([by_rater[a][t] for t in shared], [by_rater[b][t] for t in shared]), 3)
             for i, a in enumerate(raters) for b in raters[i + 1:]}
    return {"n_shared_trials": len(shared), "raters": raters, "pairwise_cohen_kappa": pairs,
            "fleiss_kappa": round(fleiss_kappa([[by_rater[r][t] for r in raters] for t in shared]), 3)}


def human_status() -> dict:
    """Everything report.py needs: measured or not, and if measured the numbers (rule judge vs the first/only human rater)."""
    rows = read_labels()
    if not rows:
        return {"measured": False, "line": NOT_MEASURED}
    from .judge import RuleJudge
    from .runner import load_scenarios
    full = HERE / "results" / "full.json"
    if not full.exists():
        return {"measured": False, "line": NOT_MEASURED + " (labels exist but evals/results/full.json is missing)"}
    trials = {f"{t['scenario_id']}-{t['seed']}": t for t in json.loads(full.read_text())["trials"]}
    scns = {s["id"]: s for s in load_scenarios()}
    h, p = [], []
    for r in rows:
        t = trials.get(r["trial_id"])
        if t:
            h.append(r["label"]); p.append(RuleJudge().judge(scns[t["scenario_id"]], t)["verdict"])
    if not h:
        return {"measured": False, "line": NOT_MEASURED + " (no label matches a trial in full.json)"}
    lo, hi = bootstrap_kappa(h, p)
    k = cohen_kappa(h, p)
    warn = "" if len(h) >= 50 else f"  [n={len(h)} < 50: do not rely on this]"
    return {"measured": True, "n": len(h), "kappa": k, "ci95": [lo, hi], "inter_rater": inter_rater(rows),
            "line": f"Cohen's kappa vs human labels (rule judge): {k:.2f}, 95% bootstrap CI [{lo:.2f}, {hi:.2f}], n={len(h)}{warn}"}


def llm_label_trials(llm, trials: list[dict], scns: dict, limit: int = 50, out: Path = LLM_LABELS) -> list[dict]:
    """DEVELOPMENT AID: an LLM labels pass/fail with the judge rubric. Written to llm_labels.jsonl, never to human_labels.jsonl."""
    from .judge import LLMJudge
    j = LLMJudge(llm)
    rows = []
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for t in trials[:limit]:
            v = j.judge(scns[t["scenario_id"]], t)["verdict"]
            rows.append({"trial_id": f"{t['scenario_id']}-{t['seed']}", "label": v, "rater": "llm", "NOT_HUMAN": True})
            f.write(json.dumps(rows[-1]) + "\n")
    return rows


def llm_labeled_agreement(rule_verdicts: list[str], llm_labels: list[str]) -> dict:
    a = agreement(llm_labels, rule_verdicts)
    a["WARNING"] = "agreement between two automatic raters (rule judge vs LLM labels): NOT human kappa, a development aid only"
    return a


def main() -> None:
    import argparse
    import os

    from supportagent.config import PRESETS
    from supportagent.llm.router import make_llm

    from .judge import LLMJudge, RuleJudge
    from .runner import load_scenarios
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-labeled", action="store_true", help="development aid: LLM-labeled agreement (needs ANTHROPIC_API_KEY); NOT human kappa")
    a = ap.parse_args()
    rows = read_labels()
    if not rows:
        print(NOT_MEASURED + "\nNo human labels yet. Run `make label` first (aim for >= 50 labeled transcripts).")
    else:
        full = json.loads((HERE / "results" / "full.json").read_text())
        trials = {f"{t['scenario_id']}-{t['seed']}": t for t in full["trials"]}
        scns = {s["id"]: s for s in load_scenarios()}
        judges = [RuleJudge()]
        if os.getenv("ANTHROPIC_API_KEY"):
            judges.append(LLMJudge(make_llm(PRESETS["full"].with_(backend="anthropic"), "strong")))
        for j in judges:
            h, p = [], []
            for r in rows:
                t = trials[r["trial_id"]]
                h.append(r["label"]); p.append(j.judge(scns[t["scenario_id"]], t)["verdict"])
            res = agreement(h, p)
            res["kappa_ci95_bootstrap"] = bootstrap_kappa(h, p)
            print(j.name, json.dumps(res, indent=1))
        ir = inter_rater(rows)
        if ir:
            print("inter-rater:", json.dumps(ir, indent=1))
    if a.llm_labeled:
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("--llm-labeled needs ANTHROPIC_API_KEY (not set): skipped.")
            return
        full = json.loads((HERE / "results" / "full.json").read_text())
        scns = {s["id"]: s for s in load_scenarios()}
        ts = full["trials"][:50]
        lab = llm_label_trials(make_llm(PRESETS["full"].with_(backend="anthropic"), "strong"), ts, scns)
        print("LLM-labeled agreement (development aid only, NOT human kappa):",
              json.dumps(llm_labeled_agreement([RuleJudge().judge(scns[t["scenario_id"]], t)["verdict"] for t in ts], [r["label"] for r in lab]), indent=1))


if __name__ == "__main__":
    main()
