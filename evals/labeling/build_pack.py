"""Build the judge-calibration labeling pack: ~60 transcripts, stratified, blind.

    python -m evals.labeling.build_pack            # -> evals/labeling/pack.jsonl, key.jsonl, label.html
    python -m evals.labeling.build_pack --n 60 --seed 7

Sources (read-only; nothing here re-runs a model or calls an API):
  live      evals/results/live/gemini_main_heldout/trials.jsonl   real Gemini trials (main + held-out, baseline + full config)
  sim-main  evals/results/full.json + baseline.json               simulated model, main scenarios (baseline adds failures)
  sim-held  evals/results/heldout/full.json                       simulated model, held-out scenarios
Quotas default to 24 / 18 / 18. Within a source, items are drawn round-robin over (category, automatic pass/fail) strata in a
seeded random order, so every category present is covered and both automatic outcomes appear even where failures are rare.

Blindness: pack.jsonl (what the labeler sees) has NO automatic verdicts. Those go to key.jsonl (judge verdict, harness pass/fail,
failure reasons), which only calibrate.py reads. A label is keyed by `item_id`, which is stable for a given source trial.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
SOURCES = {
    "live": [RESULTS / "live" / "gemini_main_heldout" / "trials.jsonl"],
    "sim-main": [RESULTS / "full.json", RESULTS / "baseline.json"],
    "sim-held": [RESULTS / "heldout" / "full.json"],
}
QUOTA = {"live": 24, "sim-main": 18, "sim-held": 18}
PACK, KEY, PAGE = HERE / "pack.jsonl", HERE / "key.jsonl", HERE / "label.html"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        out = []
        for ln in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue                                                  # a partially written last line of a running eval
        return out
    d = json.loads(path.read_text(encoding="utf-8"))
    return [{**t, "config_name": t.get("config_name") or d.get("config"), "set": t.get("set") or d.get("scenario_set")} for t in d["trials"]]


def _scenarios() -> dict[str, dict]:
    out = {}
    for sub in ("scenarios", "scenarios_heldout"):
        for f in sorted((HERE.parent / sub).glob("*.json")):
            for s in json.loads(f.read_text(encoding="utf-8")):
                out[s["id"]] = s
    return out


def item_id(source: str, t: dict) -> str:
    raw = f"{source}|{t.get('config_name')}|{t.get('set')}|{t['scenario_id']}|{t.get('seed')}|{t.get('model', '')}"
    return "L-" + hashlib.sha256(raw.encode()).hexdigest()[:10]


def stratified(trials: list[dict], n: int, rng: random.Random) -> list[dict]:
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for t in trials:
        strata[(t.get("category", "?"), bool(t.get("passed")))].append(t)
    for v in strata.values():
        rng.shuffle(v)
    # failures first in each round: they are the rare, informative class for a judge
    keys = sorted(strata, key=lambda k: (k[1], k[0]))
    rng.shuffle(keys)
    keys.sort(key=lambda k: k[1])
    picked: list[dict] = []
    while len(picked) < n and any(strata.values()):
        for k in keys:
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop())
    return picked


def build(n_total: int | None = None, seed: int = 7) -> dict:
    rng = random.Random(seed)
    scns = _scenarios()
    quota = dict(QUOTA)
    if n_total:
        scale = n_total / sum(QUOTA.values())
        quota = {k: max(1, round(v * scale)) for k, v in QUOTA.items()}
    pack, key, seen = [], [], set()
    for source, paths in SOURCES.items():
        trials = [t for p in paths for t in _load(p) if t.get("transcript")]
        for t in stratified(trials, quota[source], rng):
            iid = item_id(source, t)
            if iid in seen:
                continue
            seen.add(iid)
            s = scns.get(t["scenario_id"], {})
            pack.append({"item_id": iid, "scenario_id": t["scenario_id"], "category": t.get("category"),
                         "customer_id": s.get("customer_id"), "transcript": t["transcript"], "tool_calls": t.get("tool_calls", []),
                         "final_status": t.get("status"), "end_state": t.get("end_state", {}),
                         "policy_expected": s.get("expected", {}), "attack": (s.get("safety") or {}).get("attack")})
            key.append({"item_id": iid, "source": source, "model": t.get("model") or ("simulated" if source != "live" else ""),
                        "config": t.get("config_name"), "set": t.get("set"), "scenario_id": t["scenario_id"], "seed": t.get("seed"),
                        "category": t.get("category"), "auto_judge": t.get("judge"), "auto_passed": bool(t.get("passed")),
                        "auto_why": t.get("why", []), "replies": t.get("replies", [])})
    order = list(range(len(pack)))
    rng.shuffle(order)                                                     # interleave sources: the labeler cannot infer the source
    pack, key = [pack[i] for i in order], [key[i] for i in order]
    PACK.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in pack), encoding="utf-8")
    KEY.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in key), encoding="utf-8")
    PAGE.write_text(render_html(pack), encoding="utf-8")
    summary = defaultdict(lambda: {"n": 0, "auto_pass": 0, "auto_fail": 0})
    for k in key:
        summary[k["source"]]["n"] += 1
        summary[k["source"]]["auto_pass" if k["auto_passed"] else "auto_fail"] += 1
    return {"items": len(pack), "by_source": dict(summary), "categories": len({p["category"] for p in pack})}


def render_html(pack: list[dict]) -> str:
    rubric = (HERE / "RUBRIC.md").read_text(encoding="utf-8") if (HERE / "RUBRIC.md").exists() else ""
    data = json.dumps(pack, ensure_ascii=False).replace("<", "\\u003c")
    return TEMPLATE.replace("__DATA__", data).replace("__RUBRIC__", html.escape(rubric))


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Judge Calibration Labels</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#57534e;--card:#fff;--line:#e7e5e4;--ok:#15803d;--bad:#b91c1c;--acc:#1d4ed8}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--card:#292524;--line:#44403c;--ok:#4ade80;--bad:#f87171;--acc:#93c5fd}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.msg{margin:6px 0;white-space:pre-wrap}.customer b{color:var(--acc)}.agent b{color:var(--mut)}
pre{white-space:pre-wrap;font-size:12px;background:var(--bg);padding:8px;border-radius:6px;overflow-x:auto}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer;margin:4px 6px 4px 0}
button.pass{border-color:var(--ok)}button.fail{border-color:var(--bad)}button.on{outline:3px solid var(--acc)}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.mut{color:var(--mut);font-size:13px}
details{margin:8px 0}textarea,input{font:inherit;width:100%;box-sizing:border-box;padding:6px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
</style></head><body><main>
<h1>Judge calibration: label transcripts</h1>
<p class="mut">Blind pack: automatic verdicts are hidden. Labels stay in this browser until you press <b>Export labels.jsonl</b>; save that file as
<code>evals/labeling/labels.jsonl</code>, then run <code>python -m evals.labeling.calibrate</code>.</p>
<div class="card bar"><label>Rater name <input id="rater" placeholder="e.g. ayush" style="width:180px"></label>
<span id="prog" class="mut"></span><button id="prev">Previous</button><button id="next">Next unlabeled</button><button id="exp">Export labels.jsonl</button></div>
<details class="card"><summary>Rubric</summary><pre>__RUBRIC__</pre></details>
<div id="item" class="card"></div>
</main><script>
const PACK=__DATA__;const KEY='sa-labels-v1';let i=0;
const load=()=>{try{return JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){return {}}};
const save=o=>{try{localStorage.setItem(KEY,JSON.stringify(o))}catch(e){}};
let L=load();
try{document.getElementById('rater').value=localStorage.getItem(KEY+'-rater')||''}catch(e){}
document.getElementById('rater').oninput=e=>{try{localStorage.setItem(KEY+'-rater',e.target.value)}catch(_){}}
function el(t,c,txt){const e=document.createElement(t);if(c)e.className=c;if(txt!==undefined)e.textContent=txt;return e}
function render(){const it=PACK[i],box=document.getElementById('item');box.innerHTML='';const lab=L[it.item_id]||{};
 box.appendChild(el('div','mut',`Item ${i+1}/${PACK.length} · ${it.item_id} · category ${it.category} · customer ${it.customer_id} · final status ${it.final_status}`+(it.attack?` · attack: ${it.attack}`:'')));
 it.transcript.forEach(m=>{const d=el('div','msg '+m.role);d.appendChild(el('b','',m.role+': '));d.appendChild(document.createTextNode(m.text));box.appendChild(d)});
 const g=el('details');g.open=true;g.appendChild(el('summary','','Ground truth (system of record) and what policy expected'));
 g.appendChild(el('pre','',JSON.stringify({tool_calls:it.tool_calls,end_state_actual:it.end_state,policy_expected:it.policy_expected},null,1)));box.appendChild(g);
 const q=el('div','bar');q.appendChild(el('b','','Reply label (rubric R1-R5):'));
 [['pass','pass'],['fail','fail']].forEach(([v,c])=>{const b=el('button',c+(lab.label===v?' on':''),v.toUpperCase());b.onclick=()=>set('label',v);q.appendChild(b)});box.appendChild(q);
 const o=el('div','bar');o.appendChild(el('b','','Optional - case handled correctly end to end (O1)?'));
 [['yes','pass'],['no','fail'],['unsure','']].forEach(([v,c])=>{const b=el('button',c+(lab.outcome_ok===v?' on':''),v);b.onclick=()=>set('outcome_ok',v);o.appendChild(b)});box.appendChild(o);
 const n=el('textarea');n.rows=2;n.placeholder='Note (optional): why fail?';n.value=lab.note||'';n.oninput=e=>set('note',e.target.value,true);box.appendChild(n);
 const done=PACK.filter(p=>(L[p.item_id]||{}).label).length;document.getElementById('prog').textContent=`${done}/${PACK.length} labeled`}
function set(k,v,quiet){const id=PACK[i].item_id;L[id]=Object.assign(L[id]||{},{[k]:v});save(L);if(!quiet)render()}
document.getElementById('prev').onclick=()=>{i=(i-1+PACK.length)%PACK.length;render()};
document.getElementById('next').onclick=()=>{for(let k=1;k<=PACK.length;k++){const j=(i+k)%PACK.length;if(!(L[PACK[j].item_id]||{}).label){i=j;break}}render()};
document.getElementById('exp').onclick=()=>{const r=document.getElementById('rater').value.trim()||'rater1';
 const lines=PACK.filter(p=>(L[p.item_id]||{}).label).map(p=>JSON.stringify(Object.assign({item_id:p.item_id,rater:r},L[p.item_id])));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\n')+(lines.length?'\n':'')],{type:'application/x-ndjson'}));a.download='labels.jsonl';a.click()};
render();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m evals.labeling.build_pack")
    ap.add_argument("--n", type=int, default=None, help="total items (default 60: live 24 / sim-main 18 / sim-held 18)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    print(json.dumps(build(a.n, a.seed), indent=1))


if __name__ == "__main__":
    main()
