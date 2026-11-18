"""RAG evaluation.

  python -m evals.retrieval_eval             # legacy 12-question BM25 regression floor (what `make retrieval` always printed)
  python -m evals.retrieval_eval --report    # full dev/test study -> docs/retrieval_results.md + evals/retrieval/results.json
  supportagent kb eval                       # same as --report

The study (see evals/retrieval/PROVENANCE.md): ~170 LLM-authored questions over the policy corpus, stratified dev/test
split. Every selectable knob (RRF k, fusion weight, pool size, rerank weight, abstention thresholds) is chosen on DEV only;
TEST is evaluated once with the frozen choices. Nothing in the generated report is typed by hand.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from supportagent.config import POLICY_DIR, ROOT
from supportagent.rag import hybrid as H
from supportagent.rag.embeddings import EmbedderUnavailable, HashingEmbedder, make_embedder
from supportagent.rag.hybrid import HybridIndex
from supportagent.rag.rerank import CrossEncoderReranker, HeuristicReranker, RerankerUnavailable
from supportagent.rag.retriever import get_index, load_chunks

from .retrieval import metrics as M
from .retrieval.dataset import HERE as DATA_DIR
from .retrieval.dataset import assign_splits, load_questions

OUT_MD = ROOT / "docs" / "retrieval_results.md"
OUT_JSON = DATA_DIR / "results.json"
DEFAULT_EMBEDDERS = ("hash", "fastembed:BAAI/bge-small-en-v1.5")
ORIGINAL_DOCS = ("refund_policy.md", "escalation_policy.md", "subscription_policy.md")

# ---------------------------------------------------------------------------------------------------------
# Legacy regression floor (kept: tests/test_retriever.py and evals/report.py import `run`)
# ---------------------------------------------------------------------------------------------------------
QA = [
    ("how long do I have to return something", "refund_policy#standard-returns"),
    ("can I return a digital course", "refund_policy#digital-goods"),
    ("my item arrived broken what can I do", "refund_policy#defective-or-wrong-items"),
    ("are clearance items refundable", "refund_policy#final-sale"),
    ("can I cancel an order that has not shipped", "refund_policy#cancellations"),
    ("my package never arrived", "refund_policy#lost-packages"),
    ("do refunds over 100 dollars need approval", "refund_policy#limits"),
    ("when does my subscription cancellation take effect", "subscription_policy#cancellation"),
    ("annual plan refund window", "subscription_policy#annual-refunds"),
    ("are monthly plans refundable", "subscription_policy#monthly-plans"),
    ("customer has many refunds recently", "escalation_policy#refund-abuse-review"),
    ("customer asks for a policy exception", "escalation_policy#exception-requests"),
]
KNOWN_MISSES = ("my package never arrived", "customer has many refunds recently")


def run(k: int = 3) -> dict:
    idx = get_index(str(POLICY_DIR))
    hits, rr = 0, 0.0
    miss = []
    for q, ref in QA:
        refs = [r["ref"] for r in idx.search(q, k)]
        if ref in refs:
            hits += 1
            rr += 1 / (refs.index(ref) + 1)
        else:
            miss.append((q, ref, refs))
    return {"n": len(QA), f"recall@{k}": hits / len(QA), "mrr": rr / len(QA), "misses": miss}


# ---------------------------------------------------------------------------------------------------------
# Systems under test
# ---------------------------------------------------------------------------------------------------------
@contextmanager
def _applied(index: HybridIndex, params: dict):
    saved = {k: getattr(index, k) for k in params}
    for k, v in params.items():
        setattr(index, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(index, k, v)


@dataclass
class System:
    name: str
    mode: str                       # bm25 | bm25+ | dense | hybrid
    index: HybridIndex
    params: dict = field(default_factory=dict)
    rule: str = "cov"               # abstention rule: cov (lexical), cos (dense), both (hybrid)
    cov_key: str = "coverage"
    thresholds: tuple = ()

    def refs(self, query: str, limit: int = 10) -> list[str]:
        with _applied(self.index, self.params):
            return [c["ref"] for c in self.index.rank(query, self.mode, limit)]

    def abstains(self, signals: dict) -> bool:
        if self.rule == "cov":
            return signals[self.cov_key] < self.thresholds[0]
        if self.rule == "cos":
            return signals["cosine"] < self.thresholds[0]
        return signals["coverage"] < self.thresholds[0] and signals["cosine"] < self.thresholds[1]


def _per_question(refs: list[str], gold: list[str]) -> dict:
    return {"r1": M.recall_at_k(refs, gold, 1), "r3": M.recall_at_k(refs, gold, 3), "r5": M.recall_at_k(refs, gold, 5),
            "rr": M.reciprocal_rank(refs, gold, 10), "rr3": M.reciprocal_rank(refs, gold, 3), "ndcg5": M.ndcg_at_k(refs, gold, 5)}


def _dev_objective(sys: System, dev: list[dict]) -> float:
    pq = [_per_question(sys.refs(q["question"]), q["gold"]) for q in dev if q["gold"]]
    return (M.mean([p["rr"] for p in pq]) + M.mean([p["r3"] for p in pq])) / 2


def tune_hybrid(index: HybridIndex, dev: list[dict]) -> tuple[dict, list]:
    """Grid search on DEV only. Defaults are tried first and only strictly better candidates replace them."""
    grid = [(H.RRF_K, H.W_DENSE, H.POOL)] + [(k, w, p) for k in (10, 30, 60) for w in (0.5, 1.0, 2.0) for p in (10, 20)]
    best, best_score, log = None, -1.0, []
    for k, w, p in dict.fromkeys(grid):
        params = {"rrf_k": k, "w_dense": w, "pool": p, "reranker": None}
        s = _dev_objective(System("t", "hybrid", index, params), dev)
        log.append((params | {"reranker": None}, s))
        if s > best_score + 1e-12:
            best, best_score = params, s
    return best, log


def tune_abstain(sys: System, dev: list[dict]) -> tuple:
    """Pick abstention thresholds maximizing F1 of the abstain decision (precision and recall of "no answer") on DEV (centroid of ties)."""
    sig = [(sys.index.confidence(q["question"]), not q["gold"]) for q in dev]

    def cands(key):
        v = sorted({round(s[key], 6) for s, _ in sig})
        return [v[0] - 1e-3] + [(a + b) / 2 for a, b in zip(v, v[1:])] + [v[-1] + 1e-3]

    def ba(th):
        sys.thresholds = th
        return M.f1_abstain([(sys.abstains(s), n) for s, n in sig])

    if sys.rule == "both":
        grid = [(a, b) for a in cands("coverage") for b in cands("cosine")]
    else:
        grid = [(a,) for a in cands("cosine" if sys.rule == "cos" else sys.cov_key)]
    scored = [(ba(th), th) for th in grid]
    top = max(s for s, _ in scored)
    tied = [th for s, th in scored if abs(s - top) < 1e-12]
    centroid = [sum(t[i] for t in tied) / len(tied) for i in range(len(tied[0]))]
    return min(tied, key=lambda t: sum((t[i] - centroid[i]) ** 2 for i in range(len(t))))


# ---------------------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------------------
def evaluate(sys: System, qs: list[dict]) -> dict:
    ans = [q for q in qs if q["gold"]]
    pq = [_per_question(sys.refs(q["question"]), q["gold"]) for q in ans]
    out = {"n": len(qs), "n_answerable": len(ans), "_pq": pq, "_ids": [q.get("id") for q in ans]}
    for key in ("r1", "r3", "r5", "rr", "ndcg5"):
        out[key] = M.bootstrap_ci([p[key] for p in pq], M.mean)
    rows = [(sys.abstains(sys.index.confidence(q["question"])), not q["gold"]) for q in qs]
    out["na_precision"] = M.bootstrap_ci(rows, M.no_answer_precision)
    out["na_recall"] = M.bootstrap_ci(rows, M.no_answer_recall)
    out["false_abstain"] = M.bootstrap_ci(rows, M.false_abstain_rate)
    out["n_abstained"] = sum(a for a, _ in rows)
    out["n_no_answer"] = sum(n for _, n in rows)
    return out


def _f(t, pct=False) -> str:
    p, lo, hi = t
    if p != p:
        return "n/a"
    s = (lambda x: f"{x * 100:.0f}%") if pct else (lambda x: f"{x:.2f}")
    return f"{s(p)} [{s(lo)}, {s(hi)}]" if lo == lo else s(p)


def _table(systems: list[System], res: dict) -> list[str]:
    L = ["| system | recall@1 | recall@3 | recall@5 | MRR@10 | nDCG@5 | no-answer precision | no-answer recall |", "|---|---|---|---|---|---|---|---|"]
    for s in systems:
        r = res[s.name]
        L.append(f"| {s.name} | {_f(r['r1'])} | {_f(r['r3'])} | {_f(r['r5'])} | {_f(r['rr'])} | {_f(r['ndcg5'])} | "
                 f"{_f(r['na_precision'], True)} ({r['n_abstained']} abstained) | {_f(r['na_recall'], True)} |")
    return L


def _ranks(sys: System, query: str, gold: str) -> str:
    refs = sys.refs(query, 53)
    return f"rank {refs.index(gold) + 1}" if gold in refs else "not retrieved"


def _original_corpus_dir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="sa-orig-policies-"))
    for n in ORIGINAL_DOCS:
        shutil.copy(POLICY_DIR / n, d / n)
    return d


def _embedder_for(spec: str, directory: Path, shared: dict):
    if spec == "hash":
        return HashingEmbedder(idf_texts=[f"{c['section']}. {c['text']}" for c in load_chunks(directory)])
    if spec not in shared:
        shared[spec] = make_embedder(spec)
    return shared[spec]


def build_systems(directory: Path, specs: tuple[str, ...], shared: dict, skipped: list, tuned: dict | None = None,
                  dev: list[dict] | None = None, tune_log: dict | None = None) -> list[System]:
    """Create all systems over `directory`. When `dev` is given, hyper-parameters and thresholds are tuned on it; otherwise
    `tuned` (name -> (params, thresholds)) is reused so a second corpus is evaluated with FROZEN choices."""
    systems: list[System] = []
    base: HybridIndex | None = None

    def add(name, mode, index, params, rule, cov_key="coverage"):
        s = System(name, mode, index, params, rule, cov_key)
        if tuned is not None and name in tuned:
            s.params, s.thresholds = tuned[name]
        elif dev is not None:
            s.thresholds = tune_abstain(s, dev)
        systems.append(s)
        return s

    for spec in specs:
        try:
            emb = _embedder_for(spec, directory, shared)
        except EmbedderUnavailable as e:
            if (spec, str(e)) not in skipped:
                skipped.append((spec, str(e)))
            continue
        idx = HybridIndex(directory, embedder=emb)
        tag = emb.name
        if base is None:
            base = idx
            add("BM25 (shipped, legacy analyzer)", "bm25", idx, {}, "cov", "legacy_coverage")
            add("BM25+ (Porter + thesaurus + typo repair)", "bm25+", idx, {}, "cov", "coverage")
        add(f"dense only [{tag}]", "dense", idx, {}, "cos")
        hyb = add(f"hybrid RRF [{tag}]", "hybrid", idx, {"reranker": None}, "both")
        if dev is not None and not (tuned and hyb.name in tuned):
            hyb.params, grid_log = tune_hybrid(idx, dev)
            if tune_log is not None:
                tune_log[hyb.name] = grid_log
            hyb.thresholds = tune_abstain(hyb, dev)
        add_rerank(systems, idx, hyb, f"hybrid RRF + heuristic rerank [{tag}]", HeuristicReranker, "tw", (0.1, 0.25, 0.5), tuned, dev, tune_log)
        if spec.startswith("fastembed"):
            try:
                ce = CrossEncoderReranker(backend="fastembed")
                add_rerank(systems, idx, hyb, f"hybrid RRF + cross-encoder rerank [{tag}]", lambda b: _CEBlend(ce, b), "blend",
                           (0.0, 0.2, 0.5), tuned, dev, tune_log)
            except RerankerUnavailable as e:
                if ("cross-encoder", str(e)) not in skipped:
                    skipped.append(("cross-encoder", str(e)))
    return systems


class _CEBlend:
    def __init__(self, ce: CrossEncoderReranker, blend: float):
        self.ce, self.blend, self.name = ce, blend, f"{ce.name}(blend={blend})"

    def rerank(self, query, cands):
        self.ce.blend = self.blend
        return self.ce.rerank(query, cands)


def add_rerank(systems, idx, hyb, name, make, label, grid, tuned, dev, tune_log):
    s = System(name, "hybrid", idx, dict(hyb.params), "both")
    s.thresholds = hyb.thresholds
    if tuned is not None and name in tuned:
        s.params, s.thresholds = tuned[name]
    elif dev is not None:
        best, best_s, log = None, -1.0, []
        for w in grid:
            p = dict(hyb.params, reranker=make(w) if label == "blend" else make(title_weight=w))
            sc = _dev_objective(System("t", "hybrid", idx, p), dev)
            log.append((label, w, sc))
            if sc > best_s + 1e-12:
                best, best_s = p, sc
        s.params = best
        if tune_log is not None:
            tune_log[name] = log
    systems.append(s)


def _describe(params: dict) -> str:
    r = params.get("reranker")
    d = f"rrf_k={params.get('rrf_k', '-')}, w_dense={params.get('w_dense', '-')}, pool={params.get('pool', '-')}"
    return d + (f", rerank={r.name}" if r else "")


# ---------------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------------
def generate(embedders: tuple[str, ...] = DEFAULT_EMBEDDERS, out_md: Path = OUT_MD, out_json: Path = OUT_JSON) -> dict:
    qs = load_questions()
    split = assign_splits(qs)
    dev = [q for q in qs if split[q["id"]] == "dev"]
    test = [q for q in qs if split[q["id"]] == "test"]
    assert not {q["id"] for q in dev} & {q["id"] for q in test}
    refs_exist = {c["ref"] for c in load_chunks(POLICY_DIR)}
    bad = [(q["id"], g) for q in qs for g in q["gold"] if g not in refs_exist]
    assert not bad, f"gold refs missing from corpus: {bad}"

    skipped: list = []
    shared: dict = {}
    tune_log: dict = {}
    systems = build_systems(POLICY_DIR, embedders, shared, skipped, dev=dev, tune_log=tune_log)     # tuning touches DEV only
    res_dev = {s.name: evaluate(s, dev) for s in systems}
    res_test = {s.name: evaluate(s, test) for s in systems}                                          # TEST read after choices are frozen
    bm = systems[0]

    # legacy 12 questions on the original 3-document corpus (reproduces the documented numbers) and on the full corpus
    orig_dir = _original_corpus_dir()
    frozen = {s.name: (s.params, s.thresholds) for s in systems}
    systems_orig = build_systems(orig_dir, embedders, shared, [], tuned=frozen)
    legacy_qs = [{"question": q, "gold": [g]} for q, g in QA]
    leg_full = {s.name: evaluate(s, legacy_qs) for s in systems}
    leg_orig = {s.name: evaluate(s, legacy_qs) for s in systems_orig}
    shutil.rmtree(orig_dir, ignore_errors=True)

    from collections import Counter
    cat_n = {sp: Counter(q["category"] for q in part) for sp, part in (("dev", dev), ("test", test))}
    cats = sorted({q["category"] for q in qs})
    pkgs = {}
    for mod in ("fastembed", "onnxruntime", "sentence_transformers"):
        try:
            pkgs[mod] = __import__(mod).__version__
        except Exception:
            pkgs[mod] = None

    L = ["# Retrieval evaluation (generated)", "",
         "Generated by `python -m evals.retrieval_eval --report` (or `supportagent kb eval`). **Do not edit by hand**; every number below is computed by that script.",
         "Questions and gold labels were authored by Claude (an LLM), **not by humans, and not human-reviewed**: see `evals/retrieval/PROVENANCE.md`.", "",
         "## Setup", "",
         f"- Corpus: {len(refs_exist)} policy sections in {len({r.split('#')[0] for r in refs_exist})} documents (`policies/`), one chunk per `##` section, ref = `file#section`.",
         f"- Questions: {len(qs)} total = {len(dev)} dev + {len(test)} test, stratified by category (alternating by sha256 of the id; no random seed). "
         f"{sum(1 for q in qs if not q['gold'])} have no answer in the corpus; {sum(1 for q in qs if len(q['gold']) > 1)} need more than one section.",
         "- Dev is the only split used to choose anything (RRF k, dense weight, pool size, rerank weight, abstention thresholds). Test is evaluated once with those frozen choices.",
         "- recall@k = share of gold sections found in the top k (multi-section questions have 2 gold refs); MRR@10 = reciprocal rank of the first gold section; nDCG@5 uses binary gains. "
         "These are computed over answerable questions only.",
         "- No-answer precision = of the questions a system refuses to answer (top evidence below a dev-tuned threshold), the share that truly have no answer; "
         "no-answer recall = share of unanswerable questions it refuses. n/a = never abstained.",
         "- Intervals are 95% percentile bootstrap CIs over questions (2000 resamples, fixed seed). With ~"
         f"{len(test)} test questions (only ~{sum(1 for q in test if not q['gold'])} unanswerable) the intervals are wide; differences smaller than the intervals are not evidence.",
         f"- Environment: Python {platform.python_version()}, fastembed {pkgs['fastembed']}, onnxruntime {pkgs['onnxruntime']}, sentence-transformers {pkgs['sentence_transformers']}.", ""]
    L += ["| category | dev | test |", "|---|---|---|"] + [f"| {c} | {cat_n['dev'][c]} | {cat_n['test'][c]} |" for c in cats] + [""]

    L += ["## Systems", ""]
    for s in systems:
        extra = _describe(s.params) if s.mode == "hybrid" else ""
        L.append(f"- **{s.name}** (`{s.mode}`){': ' + extra if extra else ''}; abstain thresholds (dev-tuned for F1 of the abstain decision): {tuple(round(t, 3) for t in s.thresholds)}")
    L += ["", "BM25+ and the lexical half of hybrid use the Porter/lemma analyzer, a hand-written support-domain thesaurus + phrase rules, and corpus-vocabulary spelling repair; "
          "the dense half sees the raw question. The hashing embedder is an offline character n-gram TF-IDF model: it tolerates typos but has no semantic knowledge.", ""]
    if skipped:
        L += ["### Not run", ""] + [f"- `{n}`: {why}" for n, why in skipped] + [""]
    L += ["- API embedders (`voyage`, `openai`) were not run: no API key is configured and they cost money; the interface is covered by an offline mocked test.", ""]

    L += ["## Test split (held out; headline result)", ""] + _table(systems, res_test) + [""]
    L += ["### Paired difference on test: system minus shipped BM25 (95% CI)", "", "| system | d recall@3 | d MRR@10 | d nDCG@5 |", "|---|---|---|---|"]
    for s in systems[1:]:
        cells = []
        for key in ("r3", "rr", "ndcg5"):
            a = [p[key] for p in res_test[s.name]["_pq"]]
            b = [p[key] for p in res_test[bm.name]["_pq"]]
            d, lo, hi = M.paired_diff_ci(a, b)
            cells.append(f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        L.append(f"| {s.name} | " + " | ".join(cells) + " |")
    L += ["", "### Test recall@3 by category (point estimates; n per cell is small)", "", "| system | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for s in systems:
        row = []
        for c in cats:
            ids = [q for q in test if q["category"] == c and q["gold"]]
            if not ids:
                row.append("-")
                continue
            vals = [p["r3"] for p, i in zip(res_test[s.name]["_pq"], res_test[s.name]["_ids"]) if i in {q["id"] for q in ids}]
            row.append(f"{M.mean(vals):.2f} (n={len(vals)})")
        L.append(f"| {s.name} | " + " | ".join(row) + " |")
    L += ["", "## Dev split (used for tuning; optimistic by construction)", ""] + _table(systems, res_dev) + [""]

    L += ["## The two documented lexical misses and the 12-question floor", "",
          "These 12 questions pre-date the study and are NOT part of dev or test. The two misses were known when the query-expansion rules were written, "
          "so a fix here is a sanity check, not independent evidence; the held-out test split above is the evidence.", "",
          "| question (gold) | " + " | ".join(s.name for s in systems) + " |", "|---|" + "---|" * len(systems)]
    for q, g in QA:
        if q in KNOWN_MISSES:
            L.append(f"| {q} ({g.split('#')[1]}) | " + " | ".join(_ranks(s, q, g) for s in systems) + " |")
    L += ["", "Aggregate over all 12 (recall@3 / MRR@3), on the original 3-document corpus the numbers were measured on before (the shipped BM25 row reproduces the previously documented 83.3% recall@3; MRR here is computed from the top 3 only, as `run()` does) and on the current expanded corpus:", "",
          "| system | original 3-doc corpus | expanded corpus |", "|---|---|---|"]
    for s in systems:
        o, f_ = leg_orig.get(s.name), leg_full[s.name]
        fmt = lambda r: f"recall@3 {r['r3'][0]:.1%}, MRR@3 {M.mean([p['rr3'] for p in r['_pq']]):.2f}"          # noqa: E731
        L.append(f"| {s.name} | {fmt(o) if o else 'not run'} | {fmt(f_)} |")
    L += ["", "## Caveats", "",
          "- Authored by the same model that wrote the retrieval code, with no human labelling; gold labels can be wrong or arguable, and phrasing may echo the author's own vocabulary.",
          "- ~170 questions over 53 sections: CIs are wide and many cells are tiny. No claim beyond what the intervals support.",
          "- The thesaurus and phrase rules were written by the same author after the question set existed (not blind); only numeric knobs were tuned on dev, but rule authoring itself is a leak the split cannot remove.",
          "- Corpus is synthetic and small; absolute numbers say little about a real knowledge base.", ""]
    md = "\n".join(L)
    out_md.write_text(md, encoding="utf-8", newline="\n")

    t = res_test
    summary = []
    for s in systems:
        if s.name.startswith(("BM25 (", "hybrid RRF [", "dense only")):
            r = t[s.name]
            summary.append(f"{s.name}: recall@3 {r['r3'][0]:.1%} [{r['r3'][1]:.1%}, {r['r3'][2]:.1%}], MRR {r['rr'][0]:.2f} [{r['rr'][1]:.2f}, {r['rr'][2]:.2f}]")
    payload = {
        "n_dev": len(dev), "n_test": len(test), "skipped": skipped,
        "systems": {s.name: {"mode": s.mode, "params": _describe(s.params), "thresholds": list(s.thresholds)} for s in systems},
        "test": {n: {k: v for k, v in r.items() if not k.startswith("_")} for n, r in res_test.items()},
        "dev": {n: {k: v for k, v in r.items() if not k.startswith("_")} for n, r in res_dev.items()},
        "readme_line": " | ".join(summary),
        "lib_defaults": {"rrf_k": H.RRF_K, "w_dense": H.W_DENSE, "pool": H.POOL},
        "tuned_hybrid": {s.name: _describe(s.params) for s in systems if s.mode == "hybrid"},
    }
    out_json.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8", newline="\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", action="store_true", help="run the dev/test study and write docs/retrieval_results.md")
    ap.add_argument("--embedders", default=",".join(DEFAULT_EMBEDDERS), help="comma separated: hash,fastembed[:model],st[:model]")
    a = ap.parse_args(argv)
    if a.report:
        p = generate(tuple(x for x in a.embedders.split(",") if x))
        print(f"wrote {OUT_MD}\nwrote {OUT_JSON}\n{p['readme_line']}")
        return 0
    print(json.dumps(run(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
