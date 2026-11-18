"""Builds docs/live_results.md + evals/results/live/summary.json from the raw per-trial JSONL of live runs (evals/live.py).

Strictly separate from docs/results.md (simulated). Dry-run directories (mock server + ScriptedLLM) are reported as SIMULATED and
never as live. Nothing is extrapolated: every rate is over the scenario-trials that actually COMPLETED, with the count shown.

  python -m evals.live_report            # make eval-live-report
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .live import LIVE_DIR, read_jsonl
from .metrics import aggregate, wilson

HERE = Path(__file__).parent
DOCS = HERE.parent / "docs"
BANNER = ("> **Live-model results.** Generated from raw per-trial JSONL under `evals/results/live/`. These are measurements of the named real models; "
          "they are kept separate from the simulated-model tables in `docs/results.md` and are never mixed with them. Sample sizes are small "
          "(free-tier quota), so every interval is wide: read the counts, not just the percentages.")


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def ci(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{pct(k / n) if n else 'n/a'} ({k}/{n}; 95% CI {pct(lo)}-{pct(hi)})"


def sim_reference(sset: str, config: str, ids: set[str]) -> tuple[int, int] | None:
    f = HERE / "results" / ("heldout" if sset == "heldout" else "") / f"{'plus_' + config[1:] if config.startswith('+') else config}.json"
    if not f.exists():
        return None
    rows = [t for t in json.loads(f.read_text())["trials"] if t["scenario_id"] in ids]
    return (sum(t["passed"] for t in rows), len(rows)) if rows else None


def collect() -> list[dict]:
    runs = []
    for d in sorted(p for p in LIVE_DIR.glob("*") if p.is_dir() and p.name != "cache" and not p.name.startswith("_")):
        meta = d / "run_meta.json"
        trials = read_jsonl(d / "trials.jsonl")
        if not trials and not meta.exists():
            continue
        runs.append({"dir": d.name, "meta": json.loads(meta.read_text()) if meta.exists() else {}, "trials": trials, "errors": read_jsonl(d / "errors.jsonl")})
    return runs


def summarize(run: dict) -> dict:
    out = {"dir": run["dir"], "meta": run["meta"], "errored_trials": len(run["errors"]), "groups": {}}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in run["trials"]:
        groups[(t["set"], t["config_name"])].append(t)
    for (st, cfg), ts in sorted(groups.items()):
        k = max(1, max(sum(1 for x in ts if x["scenario_id"] == s) for s in {x["scenario_id"] for x in ts}))
        agg = aggregate(ts, k)
        n = len(ts)
        attacks = [t for t in ts if t["attack"]]
        fails = [{"scenario_id": t["scenario_id"], "seed": t["seed"], "why": t["why"][:3], "final_reply": (t["replies"][-1] if t["replies"] else "")[:160]} for t in ts if not t["passed"]]
        ids = {t["scenario_id"] for t in ts}
        out["groups"][f"{st}/{cfg}"] = {
            "set": st, "config": cfg, "completed_scenario_trials": n, "scenarios_covered": len(ids), "trials_per_scenario_max": k,
            "pass": [sum(t["passed"] for t in ts), n], "attacks": [sum(t["attack_success"] for t in attacks), len(attacks)],
            "pii_to_llm": [sum(t["pii_to_llm"] for t in ts), n], "false_claims": [sum(1 for t in ts if not t["honesty_ok"]), n],
            "safety_violations": [sum(1 for t in ts if not t["safety_ok"]), n], "summary": agg,
            "sim_reference_same_scenarios": sim_reference(st, cfg, ids), "failures": fails,
            "est_cost_usd": round(sum(t["cost_usd"] for t in ts), 4), "model": ts[0].get("model")}
    return out


def build_md(summaries: list[dict]) -> str:
    L = ["# Live model results", "", BANNER, ""]
    try:
        from .live_sections import headline_md, ladder_md, load_stats, redteam_md
        st = load_stats()
    except Exception:
        st = None
    if st:
        L += headline_md(st) + [""] + ladder_md(st) + [""] + redteam_md(st) + ["", "## Raw per-run tables and every failed trial", ""]
    live =[s for s in summaries if not s["meta"].get("dry_run", False)]
    dry = [s for s in summaries if s["meta"].get("dry_run", False)]
    if not live:
        L += ["**No live runs recorded yet.** Run `make eval-live` (Anthropic) or `python -m evals.live --backend gemini ...`, then `make eval-live-report`.", ""]
    for s in live:
        m = s["meta"]
        L += [f"## Live: {m.get('backend', '?').capitalize()} ({m.get('models', {}).get('fast', '?')} fast / {m.get('models', {}).get('strong', '?')} strong)", "",
              f"Run directory `evals/results/live/{s['dir']}`. Planned {m.get('planned_scenario_trials', '?')} scenario-trials; "
              f"{'stopped early: ' + m['stopped_early'][0] if m.get('stopped_early') else 'ran to completion or was resumed'}; trials that crashed (excluded from rates): {s['errored_trials']}. "
              f"Estimated spend {('$' + format(sum(g['est_cost_usd'] for g in s['groups'].values()), '.4f')) if s['groups'] else 'n/a'} (placeholder prices; free tier = $0 billed).", "",
              "| Set / config | Completed scenario-trials | Scenarios | Task success | Attack success | PII sent to LLM | False-claim rate | Safety violations | Simulated model, same scenarios |",
              "|---|---|---|---|---|---|---|---|---|"]
        for g in s["groups"].values():
            ref = g["sim_reference_same_scenarios"]
            L.append(f"| {g['set']} / {g['config']} | {g['completed_scenario_trials']} | {g['scenarios_covered']} | {ci(*g['pass'])} | {ci(*g['attacks']) if g['attacks'][1] else 'n/a (no attack scenarios completed)'} | "
                     f"{ci(*g['pii_to_llm'])} | {ci(*g['false_claims'])} | {ci(*g['safety_violations'])} | {ci(*ref) if ref else 'n/a'} (SIMULATED) |")
        for g in s["groups"].values():
            if g["failures"]:
                L += ["", f"**Failures, {g['set']} / {g['config']}** ({len(g['failures'])}):", ""]
                L += [f"- `{f['scenario_id']}` seed {f['seed']}: {'; '.join(f['why'])} -- final reply: \"{f['final_reply']}\"" for f in g["failures"][:25]]
        L.append("")
    if dry:
        L += ["## Pipeline validation runs (SIMULATED, not live)", "",
              "These went through the real client code path (SDK/urllib -> local mock HTTP server -> ScriptedLLM). They validate the live pipeline, not any model.", ""]
        for s in dry:
            m = s["meta"]
            L.append(f"- `{s['dir']}`: {m.get('completed_this_run', 0) + m.get('already_done_before_this_run', 0)} scenario-trials, {m.get('retries_this_run')} retries on injected faults, "
                     f"{m.get('cache_hits_this_run')} cache hits.")
    return "\n".join(L) + "\n"


def main() -> None:
    from . import live_stats
    if (LIVE_DIR / live_stats.MAIN_RUN / "trials.jsonl").exists():
        live_stats.OUT.write_text(json.dumps(live_stats.build(), indent=1, default=str))
    runs = collect()
    summaries = [summarize(r) for r in runs]
    (LIVE_DIR).mkdir(parents=True, exist_ok=True)
    (LIVE_DIR / "summary.json").write_text(json.dumps(summaries, indent=1, default=str))
    DOCS.mkdir(exist_ok=True)
    (DOCS / "live_results.md").write_text(build_md(summaries), encoding="utf-8")
    print(f"wrote {DOCS / 'live_results.md'} from {len(runs)} run dir(s)")


if __name__ == "__main__":
    main()
