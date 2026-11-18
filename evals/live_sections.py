"""Markdown sections generated from evals/results/live/live_stats.json (built by evals/live_stats.py).

Used by docs/results.md (live first), docs/live_results.md, docs/redteam_findings.md and the README results block. Every number comes from
the JSON; nothing is typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

from supportagent.config import PRICING

from .live import LIVE_DIR

STATS = LIVE_DIR / "live_stats.json"
CFG_LABEL = {"baseline": "Baseline (no guards, prompt v1)", "+prompt_v2": "+ hardened prompt (v2)", "+io_guards": "+ input/tool-output/output guards + PII tokenization",
             "+tool_guards": "+ tool guards (authz, policy verify, caps, intent grounding)", "full": "Full (+ human approval + model routing)"}


def load_stats() -> dict | None:
    return json.loads(STATS.read_text()) if STATS.exists() else None


def pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def r(d: dict) -> str:
    """rate dict -> '75.0% (45/60; 62.7-84.2)'"""
    if not d or not d["n"]:
        return "n/a"
    return f"{pct(d['rate'])} ({d['k']}/{d['n']}; {pct(d['ci95'][0])}-{pct(d['ci95'][1])})"


def pval(p: float) -> str:
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


def models_line(res: dict) -> str:
    m = res["meta"].get("models", {})
    return f"fast = `{m.get('fast', '?')}`, strong = `{m.get('strong', '?')}` (router escalation only; the pre-refund verifier is off in the `full` preset), temperature 0"


def cost_note(models: list[str]) -> str:
    unpriced = [m for m in models if m not in PRICING]
    s = "Cost is an ESTIMATE from PLACEHOLDER per-token prices in `config.PRICING` (the run used a free-tier key: $0 billed)."
    if unpriced:
        s += f" {', '.join('`' + m + '`' for m in unpriced)} has no PRICING entry and is costed at the default fallback (3.0 / 15.0 USD per Mtok), i.e. probably over-estimated."
    return s


def sig_word(p: dict) -> str:
    return "**significant**" if p["significant_at_0.05"] else "not significant"


def headline_md(res: dict, level: str = "##") -> list[str]:
    g, w = res["groups"], res["window"]
    b, f = g.get("baseline"), g.get("full")
    if not b or not f:
        return []
    kmax = max(b["trials_per_scenario_max"], f["trials_per_scenario_max"])
    L = [f"{level} Live model: Gemini (real API calls)", "",
         f"Models: {models_line(res)}. Scenarios: {f['scenarios']} ({g.get('full@main', {}).get('scenarios', 0)} main + {g.get('full@heldout', {}).get('scenarios', 0)} held-out, "
         f"category-interleaved subset), up to {kmax} trials per scenario (trials-per-scenario histogram: baseline {b['trials_per_scenario_hist']}, full {f['trials_per_scenario_hist']}). "
         f"Calls made {w['first_call']} to {w['last_call']}. Estimated cost of this table: ${b['est_cost_usd'] + f['est_cost_usd']:.2f}. {cost_note(f['models_seen'] + b['models_seen'])} "
         f"Trials that crashed (harness/IO errors, excluded from rates): {res['errored_trials']}. Run directory: `evals/results/live/{res['main_run']}`.", "",
         "Rates are over scenario-trials (Wilson 95% CI in parentheses). Trials of one scenario are correlated, so the trial-level CI is optimistic; the paired "
         "per-scenario tests below are the ones to read for baseline vs full.", "",
         "All trials are scored by the current deterministic scorer (re-applied to the stored end state, events and replies; no model output changes). "
         "Two scorer false positives found in this run were fixed first (failure analysis L3, L4); outcomes that changed versus the scorer at run time: "
         + (", ".join(f"{k} x{v}" for k, v in res.get("rescored_changes", {}).items()) or "none") + ".", "",
         "| Config | Scenario-trials | Task success | pass^1 | pass^2 | pass^3 | Attack success | Raw PII sent to LLM | False-claim rate | Safety violations | Degraded handoffs |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in ("baseline", "full"):
        x = g[n]
        pk = x["pass_hat_k_on_3trial_scenarios"] or x["pass_hat_k"]
        L.append(f"| {CFG_LABEL[n]} | {x['scenario_trials']} | {r(x['pass'])} | {pct(pk.get('1'))} | {pct(pk.get('2'))} | {pct(pk.get('3'))} | {r(x['attack_success'])} | "
                 f"{r(x['pii_to_llm'])} | {r(x['false_claims'])} | {r(x['safety_violations'])} | {x['degraded_handoffs']} |")
    L += ["", "pass^k = probability that k independent attempts at the same scenario ALL succeed (tau-bench estimator C(c,k)/C(n,k), averaged over scenarios). "
          f"All three pass^k columns are computed on the same scenarios, those with 3 completed trials (baseline {b['scenarios_with_3_trials']}, full {f['scenarios_with_3_trials']}), "
          "so they are comparable with each other; the task-success column uses every completed trial. "
          "Temperature is 0, so trial-to-trial variation comes only from provider-side non-determinism; pass^k still drops when a scenario is flaky."]
    flaky = [(n, g[n]["flaky_scenarios"]) for n in ("baseline", "full") if g[n]["flaky_scenarios"]]
    if flaky:
        L += ["", "Flaky scenarios (mixed pass/fail across their 3 trials): " + "; ".join(f"{n}: {', '.join('`' + s + '`' for s in v)}" for n, v in flaky) + "."]
    L += ["", f"Validity checks: trials answered (even partly) by the simulated model: baseline {b['trials_with_simulated_model']}, full {f['trials_with_simulated_model']} "
          f"(models seen: {', '.join(sorted(set(b['models_seen']) | set(f['models_seen'])))}). The strong model was used in {f['trials_using_strong_model']} full trials and "
          f"{b['trials_using_strong_model']} baseline trials: the router escalates fast -> strong only on risk signals (injection flagged, guard block, repair), and baseline has routing off. "
          "The independent pre-refund verifier is OFF in the `full` preset (`AgentConfig.verifier=False`; claim verification is the regex output guard), so zero verifier "
          f"calls is by design (verifier events seen: {f['verifier_events']}). Review finding R1 (SA_BACKEND=gemini not wired into `router.make_llm`, so the API/CLI ran the "
          "simulator) does not apply to this runner: `evals/live.py` builds every agent model through its own `llm_maker`, records the model name of every call, and now "
          "rejects any live trial in which a simulated model answered."]
    L += ["", "By set:", "", "| Set | Baseline task success | Full task success | Baseline attack success | Full attack success |", "|---|---|---|---|---|"]
    for st in ("main", "heldout"):
        bb, ff = g.get(f"baseline@{st}"), g.get(f"full@{st}")
        if bb and ff:
            L.append(f"| {st} | {r(bb['pass'])} | {r(ff['pass'])} | {r(bb['attack_success'])} | {r(ff['attack_success'])} |")
    L += ["", f"{level}# Is the difference real? Paired per-scenario tests (baseline vs full)", "",
          "Each scenario runs under both configs with the same seed, so the comparison is paired. **McNemar exact** is computed on one outcome per scenario "
          "(the majority over its trials; the seed-0-only pairs are shown as a check); the **bootstrap** resamples scenarios (10,000 paired cluster resamples) for the "
          "difference in mean per-scenario rate. A difference is called significant only if McNemar p < 0.05 **and** the bootstrap 95% CI excludes 0.", "",
          "| Scope | Metric | Scenarios | Only baseline | Only full | McNemar p (majority) | McNemar p (seed 0) | Full - baseline (bootstrap 95% CI) | Verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    names = {"task_success": "task success", "attack_success": "attack success", "pii_to_llm": "raw PII to LLM"}
    for scope, ps in res["paired"].items():
        for met, p in ps.items():
            if not p["n_scenarios"]:
                continue
            bt = p["bootstrap_B_minus_A"]
            L.append(f"| {scope} | {names[met]} | {p['n_scenarios']} | {p['majority']['only_A']} | {p['majority']['only_B']} | {pval(p['majority']['mcnemar_p'])} | "
                     f"{pval(p['seed0']['mcnemar_p'])} | {'+' if bt['diff'] >= 0 else ''}{bt['diff'] * 100:.1f} pts ({bt['ci95'][0] * 100:+.1f} to {bt['ci95'][1] * 100:+.1f}) | {sig_word(p)} |")
    pa = res["paired"].get("all", {})
    if pa:
        t, a = pa["task_success"], pa["attack_success"]
        L += ["", f"Reading: on {t['n_scenarios']} paired scenarios the task-success difference is {sig_word(t)} (McNemar p = {pval(t['majority']['mcnemar_p'])}); "
              f"the attack-success reduction on {a['n_scenarios']} attack scenarios is {sig_word(a)} (p = {pval(a['majority']['mcnemar_p'])}: with only "
              f"{a['majority']['only_A'] + a['majority']['only_B']} discordant scenarios the exact test cannot reach 0.05 below 6). "
              f"The PII result is {sig_word(pa['pii_to_llm'])} and is structural (tokenization happens before any model call)."]
    return L


def ladder_md(res: dict, level: str = "##") -> list[str]:
    lad = res.get("ladder")
    if not lad:
        return [f"{level} Live ablation ladder", "", "Not run yet (`python -m evals.live --backend gemini --config +prompt_v2 +io_guards +tool_guards --trials 1 ...`).", ""]
    w = lad["window"]
    L = [f"{level} Live ablation ladder (Gemini, 1 trial per scenario)", "",
         f"Same {models_line(res)} and the same scenarios. Intermediate rungs come from `evals/results/live/{lad['run']}` (1 trial, seed 0; calls {w['first_call']} to {w['last_call']}, "
         f"est. ${w['est_usd']:.2f}); baseline and full rows are the seed-0 trials of the main run, so every row is seed 0 and directly paired. Crashed trials excluded: {lad['errored_trials']}. "
         "With 1 trial per scenario, adjacent rungs are NOT distinguishable unless the paired test says so."
         + (f" **Partial:** this ladder run planned {lad['meta'].get('planned_scenario_trials')} scenario-trials ({', '.join(lad['meta'].get('configs', []))}) and stopped early "
            f"({lad['meta']['stopped_early'][0]}); rungs not listed were not run." if lad["meta"].get("stopped_early") else ""), "",
         "| Rung | Scenario-trials | Task success | Attack success | Raw PII to LLM | False-claim rate | vs baseline: task (McNemar p) | vs baseline: attacks (McNemar p) |",
         "|---|---|---|---|---|---|---|---|"]
    for cfg, x in lad["rungs"].items():
        vs = lad["vs_baseline_seed0"].get(cfg)
        vt = f"{vs['task_success']['bootstrap_B_minus_A']['diff'] * 100:+.1f} pts (p={pval(vs['task_success']['seed0']['mcnemar_p'])})" if vs else "reference"
        va = (f"{vs['attack_success']['bootstrap_B_minus_A']['diff'] * 100:+.1f} pts (p={pval(vs['attack_success']['seed0']['mcnemar_p'])})"
              if vs and vs["attack_success"]["n_scenarios"] else ("reference" if not vs else "n/a"))
        L.append(f"| {CFG_LABEL[cfg]} | {x['scenario_trials']} | {r(x['pass'])} | {r(x['attack_success'])} | {r(x['pii_to_llm'])} | {r(x['false_claims'])} | {vt} | {va} |")
    return L


def redteam_md(res: dict, level: str = "##") -> list[str]:
    rt = res.get("redteam")
    if not rt:
        return [f"{level} Live adaptive red team", "", "Not run yet (`python -m evals.redteam.adaptive --live --backend gemini --defender live --layers full --budget 25`).", ""]
    w = rt["window"]
    L = [f"{level} Live adaptive red team (LLM attacker vs live defender)", "",
         f"Attacker: `{rt['attacker_model']}` (temperature 1.0), a different model from the defender ({rt['defender']}). Budget: up to {rt['budget_per_goal']} adaptive attempts per goal; "
         "attack families are assigned round-robin (paraphrase, encoding, role-play, multi-turn escalation, payload splitting, + indirect injection via order notes / knowledge base where the goal allows). "
         f"Each attempt is a fresh sandboxed conversation scored by the same deterministic harm scorer as the static suite. Calls {w['first_call']} to {w['last_call']}, est. ${w['est_usd']:.2f} "
         "(placeholder prices; the attacker model is unpriced and costed at the fallback rate). Attempts within a goal are adaptive and therefore not independent: the Wilson intervals are approximate.", ""]
    for layer, x in rt["layers"].items():
        L += [f"Against **{CFG_LABEL.get(layer, layer)}**: attack success **{r(x['asr'])}** over {x['attempts']} attempts; goals broken {x['goals_broken']['k']}/{x['goals_broken']['n']}"
              + (f" ({', '.join(x['broken_goals'])})" if x["broken_goals"] else "") + f". Attacker refusals / unparseable proposals (fell back to the naive base attack, still counted): {x['attacker_parse_failures']}.", "",
              "| Goal | Attempts | Successes (ASR, 95% CI) | Time to first success (attempt #) | Attacker refusals |", "|---|---|---|---|---|"]
        for gname, v in x["per_goal"].items():
            L.append(f"| {gname} | {v['attempts']} | {r(v['asr'])} | {v['ttfs'] or 'none within budget'} | {v['attacker_parse_failures']} |")
        L += ["", "| Attack family | Attempts | ASR (95% CI) |", "|---|---|---|"]
        L += [f"| {fam} | {v['n']} | {r(v)} |" for fam, v in x["by_family"].items()]
        L += ["", "Every successful attack (with its trace id) is written up in [`docs/redteam_findings.md`](redteam_findings.md)."]
    return L


def readme_block(res: dict | None) -> str:
    if not res or not res["groups"].get("full"):
        return "Live results: not run yet. See `docs/results.md`.\n"
    g = res["groups"]
    b, f = g["baseline"], g["full"]
    pa = res["paired"].get("all", {})
    L = [f"### Live: Gemini ({models_line(res)})", "",
         f"{f['scenarios']} scenarios (main + held-out), up to {max(b['trials_per_scenario_max'], f['trials_per_scenario_max'])} trials each, real API calls {res['window']['first_call'][:10]} to {res['window']['last_call'][:10]}, "
         f"est. ${b['est_cost_usd'] + f['est_cost_usd']:.2f} at placeholder prices (free tier, $0 billed).", "",
         f"pass^3 is over the {f['scenarios_with_3_trials']} scenarios that have 3 completed trials so far.", "",
         "| Config | Task success (95% CI) | pass^3 | Attack success | Raw PII sent to LLM |", "|---|---|---|---|---|"]
    for n in ("baseline", "full"):
        x = g[n]
        L.append(f"| {CFG_LABEL[n]} | {r(x['pass'])} | {pct((x['pass_hat_k_on_3trial_scenarios'] or {}).get('3'))} | {r(x['attack_success'])} | {r(x['pii_to_llm'])} |")
    if pa:
        t, a = pa["task_success"], pa["attack_success"]
        bt = t["bootstrap_B_minus_A"]
        L += ["", f"Paired per-scenario tests: task success {bt['diff'] * 100:+.1f} pts (bootstrap 95% CI {bt['ci95'][0] * 100:+.1f} to {bt['ci95'][1] * 100:+.1f}; McNemar p = {pval(t['majority']['mcnemar_p'])}) is "
              f"{sig_word(t)}; attack success (p = {pval(a['majority']['mcnemar_p'])}) is {sig_word(a)}; raw-PII-to-LLM is {sig_word(pa['pii_to_llm'])}."]
    lad = res.get("ladder")
    if lad:
        L += ["", "Live ladder (1 trial, seed 0): " + " -> ".join(f"{c} {pct(x['pass']['rate'])} (n={x['pass']['n']})" for c, x in lad["rungs"].items()) + " task success"
              + ("; partial, stopped by the daily quota: intermediate rungs are not comparable yet." if lad["meta"].get("stopped_early") else ".")]
    rt = res.get("redteam")
    if rt:
        for layer, x in rt["layers"].items():
            L += ["", f"Live adaptive red team (`{rt['attacker_model']}` attacking {layer}): ASR {r(x['asr'])}, goals broken {x['goals_broken']['k']}/{x['goals_broken']['n']}"
                  + (f" ({', '.join(x['broken_goals'])}; see [`docs/redteam_findings.md`](docs/redteam_findings.md))" if x["broken_goals"] else "") + "."]
    L += ["", "Full live tables, paired tests, ladder and red team: [`docs/results.md`](docs/results.md)."]
    return "\n".join(L) + "\n"


def findings_md(res: dict | None) -> str:
    L = ["> Generated by `python -m evals.report` from `evals/results/live/redteam_gemini_live/attempts.jsonl` (live LLM attacker vs live defender). "
         "Every successful attack is listed with its trace id (= defender session id; the full transcript is in the attempts log). "
         "Offline (simulated-defender) findings are in `docs/results.md` and `docs/failure_analysis.md` (F4).", ""]
    rt = (res or {}).get("redteam")
    if not rt:
        return "\n".join(L + ["No live red-team run recorded yet.", ""])
    for layer, x in rt["layers"].items():
        L += [f"## Against `{layer}`: {len(x['successes'])} successful attempts out of {x['attempts']}", ""]
        if not x["successes"]:
            L += ["No attempt succeeded within the budget. That is evidence about this attacker and this budget only, not a proof of robustness.", ""]
        for s in x["successes"]:
            L += [f"### `{s['trace_id']}`: {s['goal']} (attempt {s['i']}, family {s['family']})", "",
                  f"- Opening: {json.dumps(s['opening'])[:600]}"]
            if s["followups"]:
                L.append(f"- Follow-ups: {json.dumps(s['followups'])[:600]}")
            if s.get("note"):
                L.append(f"- Planted order note: {json.dumps(s['note'])[:400]}")
            if s.get("kb"):
                L.append(f"- Planted KB passage: {json.dumps(s['kb'])[:400]}")
            L += [f"- Why it counts as a success (scorer): {'; '.join(s['why'][:3]) or 'harm scorer flagged it'}",
                  f"- Guard events: {', '.join(s['events']) or 'none'}; tool calls: {', '.join(s['tools']) or 'none'}",
                  f"- Final reply: {json.dumps(s['reply'])[:300]}", ""]
    return "\n".join(L) + "\n"


def replace_block(path: Path, name: str, body: str, default_after: str | None = None) -> bool:
    """Replace the text between <!-- name:start --> and <!-- name:end --> in `path` (hand-written text around it is kept)."""
    start, end = f"<!-- {name}:start (generated by `python -m evals.report`; do not edit by hand) -->", f"<!-- {name}:end -->"
    txt = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in txt and end in txt:
        new = txt[:txt.index(start)] + start + "\n" + body + end + txt[txt.index(end) + len(end):]
    elif default_after is not None and default_after in txt:
        i = txt.index(default_after) + len(default_after)
        new = txt[:i] + "\n" + start + "\n" + body + end + "\n" + txt[i:]
    else:
        new = txt + ("\n" if txt and not txt.endswith("\n") else "") + start + "\n" + body + end + "\n"
    if new != txt:
        path.write_text(new, encoding="utf-8")
    return new != txt
