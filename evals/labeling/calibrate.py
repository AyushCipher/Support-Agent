"""Judge-vs-human agreement on the labeling pack (Cohen's kappa + bootstrap CI), once a human has filled labels.jsonl.

    python -m evals.labeling.calibrate            # human text report
    python -m evals.labeling.calibrate --json     # machine-readable (also written to evals/labeling/calibration.json)

Compares the human `label` (reply quality, RUBRIC.md R1-R5) with:
  rule_judge   evals.judge.RuleJudge re-run now on the stored trial (deterministic)
  stored_judge the `judge` verdict recorded in the trial (rule judge for simulated runs; whatever judge the live run used)
and the optional human `outcome_ok` (O1) with the harness's automatic end-to-end `passed`. Reported overall and per source
(live Gemini vs simulated) because a judge can agree with humans on one distribution and not the other.
With no labels it prints NOT YET MEASURED and exits 0: it never invents a number. Labels from several raters -> inter-rater kappa.
Reuses evals/calibrate.py (cohen_kappa, bootstrap_kappa, inter_rater) so both calibration paths compute kappa identically.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ..calibrate import NOT_MEASURED, bootstrap_kappa, cohen_kappa, inter_rater
from .label import LABELS, read_jsonl

HERE = Path(__file__).parent
KEY, OUT = HERE / "key.jsonl", HERE / "calibration.json"
MIN_N = 50


def _rule_verdicts(key: dict[str, dict]) -> dict[str, str]:
    from ..judge import RuleJudge
    from .build_pack import _scenarios
    scns, j = _scenarios(), RuleJudge()
    out = {}
    for iid, k in key.items():
        s = scns.get(k["scenario_id"])
        if s is not None and k.get("replies"):
            out[iid] = j.judge(s, {"replies": k["replies"], "end_state": _end_state(iid)})["verdict"]
    return out


_PACK_CACHE: dict[str, dict] = {}


def _end_state(iid: str) -> dict:
    if not _PACK_CACHE:
        for it in read_jsonl(HERE / "pack.jsonl"):
            _PACK_CACHE[it["item_id"]] = it
    return _PACK_CACHE.get(iid, {}).get("end_state") or {"refunds": []}


def _agree(pairs: list[tuple[str, str]]) -> dict:
    if not pairs:
        return {"n": 0, "kappa": None}
    h, j = [p[0] for p in pairs], [p[1] for p in pairs]
    k = cohen_kappa(h, j)
    lo, hi = bootstrap_kappa(h, j)
    cm: dict[str, int] = defaultdict(int)
    for a, b in pairs:
        cm[f"human={a},auto={b}"] += 1
    r = {"n": len(pairs), "accuracy": round(sum(a == b for a, b in pairs) / len(pairs), 3),
         "kappa": None if k != k else round(k, 3), "ci95": [None if lo != lo else round(lo, 3), None if hi != hi else round(hi, 3)],
         "confusion": dict(cm)}
    if len(pairs) < MIN_N:
        r["warning"] = f"n={len(pairs)} < {MIN_N}: do not rely on this kappa"
    if len(set(h)) < 2 or len(set(j)) < 2:
        r["warning"] = (r.get("warning", "") + "; " if r.get("warning") else "") + "one rater used a single label: kappa is degenerate"
    return r


def compute(rater: str | None = None) -> dict:
    rows = [r for r in read_jsonl(LABELS) if r.get("label") in ("pass", "fail")]
    if not rows:
        return {"measured": False, "line": NOT_MEASURED + " (evals/labeling/labels.jsonl is empty)"}
    key = {k["item_id"]: k for k in read_jsonl(KEY)}
    raters = sorted({r.get("rater", "rater1") for r in rows})
    primary = rater or raters[0]
    mine = {r["item_id"]: r for r in rows if r.get("rater", "rater1") == primary and r["item_id"] in key}
    rule = _rule_verdicts({i: key[i] for i in mine})
    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for iid, r in mine.items():
        k = key[iid]
        for g in ("all", k["source"]):
            if iid in rule:
                groups[g]["rule_judge"].append((r["label"], rule[iid]))
            if k.get("auto_judge") in ("pass", "fail"):
                groups[g]["stored_judge"].append((r["label"], k["auto_judge"]))
            if r.get("outcome_ok") in ("yes", "no"):
                groups[g]["outcome_vs_harness"].append(("pass" if r["outcome_ok"] == "yes" else "fail", "pass" if k["auto_passed"] else "fail"))
    res = {"measured": True, "rater": primary, "raters": raters, "n_labeled": len(mine), "pack_size": len(key),
           "agreement": {g: {name: _agree(p) for name, p in d.items()} for g, d in groups.items()},
           "inter_rater": inter_rater([{"trial_id": r["item_id"], "label": r["label"], "rater": r.get("rater", "rater1")} for r in rows])}
    a = res["agreement"].get("all", {}).get("rule_judge", {})
    res["line"] = (f"Cohen's kappa, human ({primary}) vs rule judge on the labeling pack: {a.get('kappa')}, 95% bootstrap CI {a.get('ci95')}, "
                   f"n={a.get('n', 0)}" + (f"  [{a['warning']}]" if a.get("warning") else ""))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evals.labeling.calibrate")
    ap.add_argument("--rater", default=None, help="whose labels to score against the judges (default: first rater alphabetically)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = compute(a.rater)
    if res["measured"]:
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1) if a.json else res["line"])
    if res["measured"] and not a.json:
        for g, d in res["agreement"].items():
            for name, r in d.items():
                print(f"  [{g:<8}] {name:<18} n={r['n']:<3} kappa={r.get('kappa')} ci95={r.get('ci95')} acc={r.get('accuracy')}"
                      + (f"  ({r['warning']})" if r.get("warning") else ""))
        if res["inter_rater"]:
            print("  inter-rater:", json.dumps(res["inter_rater"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
