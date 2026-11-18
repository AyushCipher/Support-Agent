"""Builds docs/results.md and docs/dashboard.html from evals/results/*.json."""
from __future__ import annotations

import html
import json
from pathlib import Path

from .report_sections import adaptive_md, calibration_md, heldout_md, live_pointer, load_heldout

HERE = Path(__file__).parent
DOCS = HERE.parent / "docs"
ORDER = ["baseline", "plus_prompt_v2", "plus_io_guards", "plus_tool_guards", "full"]
LABEL = {"baseline": "Baseline (no guards, prompt v1)", "plus_prompt_v2": "+ hardened prompt (v2)", "plus_io_guards": "+ input/tool-output/output guards",
         "plus_tool_guards": "+ tool guards (authz, policy verify, caps, intent grounding)", "full": "+ human approval + model routing (full)"}


def load() -> dict:
    out = {}
    for n in ORDER:
        f = HERE / "results" / f"{n}.json"
        if f.exists():
            out[n] = json.loads(f.read_text())
    return out


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_md(res: dict) -> str:
    L = ["> **Simulated model.** Everything in this section was produced with the deterministic *simulated* model (`ScriptedLLM`, seeded failure modes) so the whole pipeline is reproducible offline "
         "with no key. It validates the harness, guardrails and eval machinery and shows how much each defense layer contributes against that model. **It does not measure how any real LLM performs**; "
         "the live sections above do.", ""]
    any_ = next(iter(res.values()))["summary"]
    L += [f"Dataset: **{any_['n_scenarios']} scenarios**, **{any_['trials_per_scenario']} trials each**, fresh database per trial. "
          f"Success = correct final database state + no harm + honest reply + approval expectation met (tau-bench style).", "",
          "## Ablation ladder", "",
          "| Configuration | Task success (95% CI) | pass^1 | pass^8 | Attack success rate | PII sent to LLM | False-claim rate | Avg cost/run | Approvals/run |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        s = r["summary"]
        k = s["trials_per_scenario"]
        L.append(f"| {LABEL[n]} | {pct(s['task_success'])} ({pct(s['task_success_ci95'][0])}-{pct(s['task_success_ci95'][1])}) | {pct(s['pass_hat_k']['1'])} | "
                 f"{pct(s['pass_hat_k'][str(k)])} | **{pct(s['attack_success_rate'])}** | {pct(s['pii_to_llm_rate'])} | {pct(s['false_claim_rate'])} | ${s['avg_cost_usd']:.4f} | {pct(s['approval_rate'])} |")
    if "full" in res:
        f = res["full"]["summary"]
        L += ["", "## Final configuration in detail", "", "| Category | Success |", "|---|---|"]
        L += [f"| {c} | {pct(v)} |" for c, v in f["by_category"].items()]
        L += ["", "### Attack success rate by attack type (final config vs baseline)", "", "| Attack type | Baseline | Final |", "|---|---|---|"]
        b = res.get("baseline", {}).get("summary", {}).get("asr_by_attack_type", {})
        L += [f"| {a} | {pct(b.get(a, 0))} | {pct(v)} |" for a, v in f["asr_by_attack_type"].items()]
        L += ["", "### Reliability: pass^k (probability all k independent attempts succeed)", "", "| k | " + " | ".join(str(i) for i in range(1, f['trials_per_scenario'] + 1)) + " |",
              "|---|" + "---|" * f["trials_per_scenario"]]
        for n in ("baseline", "full"):
            if n in res:
                pk = res[n]["summary"]["pass_hat_k"]
                L.append(f"| {n} | " + " | ".join(pct(pk[str(i)]) for i in range(1, f["trials_per_scenario"] + 1)) + " |")
        L += ["", "### Guardrail activity (events across all final-config trials)", "", "| Event | Count |", "|---|---|"]
        L += [f"| {e} | {c} |" for e, c in f["guardrail_events"].items()]
        L += ["", f"Harness latency (no network; real-model latency will dominate): p50 {f['latency_ms_p50']:.1f} ms, p95 {f['latency_ms_p95']:.1f} ms. "
              f"Avg {f['avg_steps']:.1f} LLM steps and {f['avg_tokens']:.0f} tokens per conversation (notional pricing). Strong-model share with routing: {pct(f['strong_model_share'])}."]
    arch = HERE / "results" / "archive" / "full_before_intent_grounding.json"
    if arch.exists() and "full" in res:
        a = json.loads(arch.read_text())["summary"]
        L += ["", "## Case study: a red-team finding and its fix", "",
              f"After adding paraphrased indirect-injection scenarios (N04/N05), the *then-final* configuration showed attack success of **{pct(a['attack_success_rate'])}** "
              f"(paraphrased-injection type: **{pct(a['asr_by_attack_type'].get('indirect_injection_paraphrased', 0))}**) because a policy-eligible refund the customer never asked for passed every policy check. "
              f"Adding *intent grounding* (money actions require the customer's own words to ask for them) brought it to **{pct(res['full']['summary']['attack_success_rate'])}**. "
              "See `docs/failure_analysis.md` (F4)."]
    try:
        from .retrieval_eval import run as rr
        r = rr(3)
        L += ["", "## RAG retrieval (BM25 over policy docs)", "", f"{r['n']} labeled questions: **recall@3 = {pct(r['recall@3'])}**, MRR = {r['mrr']:.2f}."]
        for q, want, got in r["misses"]:
            L.append(f"- miss: \"{q}\" expected `{want}`, got `{', '.join(got)}` (lexical gap; see failure analysis F6)")
    except Exception:
        pass
    L += heldout_md(load_heldout(), LABEL) + adaptive_md(LABEL) + calibration_md() + live_pointer()
    L += ["", "## Known limitations of this evidence", "",
          "- The main attacker suite is fixed and scripted: 0% ASR there means *these* attacks fail. The adaptive search is a first test against attacker adaptation, but its defender is simulated and its attacker is a small deterministic mutation search.",
          "- The simulated model's failure rates are parameters I chose; absolute success numbers are not predictive of real-model behavior.",
          "- The regex output guard misses paraphrased false claims (see false-claim rate). `claim_verifier=llm` is the intended fix for live runs.",
          "- LLM-judge calibration against human labels: see the Judge calibration section (not measured until human labels exist).",
          "- The held-out set was authored by the same author as the main suite and the simulator and is not human-labeled: it reduces, but does not remove, overfitting risk."]
    sim = [("#" + ln) if ln.startswith("#") else ln for ln in L]          # demote: the simulated section sits under its own H2
    return "\n".join(live_head() + ["## Simulated model (reproducible offline harness test)", ""] + sim) + "\n"


REPRO = ["## How to reproduce", "",
         "- **Gemini (the live numbers above):** put `GEMINI_API_KEY=...` in the git-ignored `.env` (or the environment), then "
         "`make eval-live BACKEND=gemini RPM=15 TRIALS=3` (resumable, cached, backoff, request/cost caps; the live tables used "
         "`python -m evals.live --backend gemini --set both --limit-scenarios 60 --config baseline full --trials 3 --rpm 15 --max-requests 2000 --run-dir evals/results/live/gemini_main_heldout`), "
         "the ladder with `python -m evals.live --backend gemini --set both --limit-scenarios 60 --config +prompt_v2 +io_guards +tool_guards --trials 1 --run-dir evals/results/live/gemini_ladder_main_heldout`, "
         "and the live red team with `python -m evals.redteam.adaptive --live --backend gemini --defender live --layers full --budget 25 --out redteam_gemini_live`.",
         "- **Anthropic:** `export ANTHROPIC_API_KEY=...` then `make eval-live` (BACKEND defaults to anthropic; models from `SA_MODEL_FAST` / `SA_MODEL_STRONG`) and "
         "`make redteam-live BACKEND=anthropic`. No Anthropic live run is recorded in this file.",
         "- **Offline (no key):** `make eval` / `make eval-heldout` / `make redteam` regenerate the simulated section; `make eval-live-dry` exercises the live client code against a local mock server.",
         "- `make results` regenerates this file, `docs/live_results.md`, `docs/redteam_findings.md` and the README results block from the JSON/JSONL.", ""]


def live_head() -> list[str]:
    from .live_sections import headline_md, ladder_md, load_stats, redteam_md
    st = load_stats()
    L = ["# Evaluation results", "",
         "> **Read this first.** The first three sections are **live** measurements of real Gemini models (real API calls, raw per-trial logs under `evals/results/live/`). "
         "Samples are small (free-tier quota), so read the counts and the paired tests, not just the percentages. The last section is the **simulated** model, which is a "
         "reproducible offline test of the harness and is never mixed into the live numbers.", ""]
    if not st:
        return L + ["No live run recorded yet.", ""] + REPRO
    return L + headline_md(st) + [""] + ladder_md(st) + [""] + redteam_md(st) + [""] + REPRO


def svg_bars(rows: list[tuple[str, float]], title: str, color: str, w=520) -> str:
    h = 26 * len(rows) + 40
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{html.escape(title)}"><text x="0" y="16" class="t">{html.escape(title)}</text>']
    for i, (lab, v) in enumerate(rows):
        y = 30 + i * 26
        out.append(f'<text x="0" y="{y + 13}" class="l">{html.escape(lab[:34])}</text><rect x="210" y="{y}" width="{max(1, v * 240):.0f}" height="16" fill="{color}"/>'
                   f'<text x="{216 + v * 240:.0f}" y="{y + 13}" class="v">{v * 100:.1f}%</text>')
    return "".join(out) + "</svg>"


def build_dashboard(res: dict) -> str:
    cards, bars = [], []
    if "full" in res:
        f = res["full"]["summary"]
        for k, v in [("Task success", pct(f["task_success"])), ("Attack success", pct(f["attack_success_rate"])), ("pass^8", pct(f["pass_hat_k"][str(f["trials_per_scenario"])])),
                     ("Avg cost / run", f"${f['avg_cost_usd']:.4f}"), ("p95 harness latency", f"{f['latency_ms_p95']:.1f} ms"), ("Avg LLM steps", f"{f['avg_steps']:.1f}")]:
            cards.append(f'<div class="c"><b>{v}</b><span>{k}</span></div>')
    bars.append(svg_bars([(LABEL[n].split(" (")[0][:34], r["summary"]["task_success"]) for n, r in res.items()], "Task success by configuration", "#2f6b4f"))
    bars.append(svg_bars([(LABEL[n].split(" (")[0][:34], r["summary"]["attack_success_rate"]) for n, r in res.items()], "Attack success rate by configuration", "#b4472a"))
    if "full" in res:
        bars.append(svg_bars(list(res["full"]["summary"]["by_category"].items()), "Final config: success by category", "#a56a12"))
        ev = res["full"]["summary"]["guardrail_events"]
        mx = max(ev.values()) if ev else 1
        bars.append(svg_bars([(k, v / mx) for k, v in ev.items()], "Guardrail events (relative)", "#555"))
    return f"""<!doctype html><meta charset="utf-8"><title>supportagent dashboard</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>:root{{--bg:#f4efe6;--fg:#1d1b18}}@media(prefers-color-scheme:dark){{:root{{--bg:#171512;--fg:#ece6da}}}}
body{{background:var(--bg);color:var(--fg);font:14px system-ui;margin:0;padding:24px 32px}}h1{{font:600 22px ui-serif,Georgia,serif}}
.g{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}}.c{{border:1px solid var(--fg);padding:12px}}.c b{{display:block;font:600 24px ui-serif,Georgia,serif}}.c span{{font-size:12px;opacity:.7}}
.p{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:22px}}svg .t{{font:600 13px system-ui;fill:currentColor}}svg .l,svg .v{{font:12px ui-monospace,monospace;fill:currentColor}}
small{{opacity:.7}}</style><h1>supportagent · evaluation dashboard</h1><small>Simulated-model results (see docs/results.md). Regenerate: <code>make report</code></small>
<div class="g">{''.join(cards)}</div><div class="p">{''.join(bars)}</div>"""


def main() -> None:
    res = load()
    DOCS.mkdir(exist_ok=True)
    from . import live_stats                               # live stats first, so results.md / README read fresh numbers
    if (live_stats.LIVE_DIR / live_stats.MAIN_RUN / "trials.jsonl").exists():
        live_stats.OUT.write_text(json.dumps(live_stats.build(), indent=1, default=str))
    from .live_sections import findings_md, load_stats, readme_block, replace_block
    st = load_stats()
    replace_block(HERE.parent / "README.md", "live-results", readme_block(st), default_after="## Results\n")
    replace_block(DOCS / "redteam_findings.md", "live-redteam", findings_md(st))
    (DOCS / "results.md").write_text(build_md(res))
    (DOCS / "dashboard.html").write_text(build_dashboard(res))
    print(f"wrote {DOCS/'results.md'} and {DOCS/'dashboard.html'}")


if __name__ == "__main__":
    main()
