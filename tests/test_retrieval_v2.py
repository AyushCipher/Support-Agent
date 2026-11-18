"""Hybrid retrieval (ADR-0012): stemmer, RRF, embedders, expansion, rerank, flag wiring, dataset integrity. All offline."""
import io
import json
import warnings

import pytest

from evals.retrieval import metrics as M
from evals.retrieval.dataset import assign_splits, load_questions
from supportagent.config import POLICY_DIR
from supportagent.mcp.client import build_registry
from supportagent.rag.embeddings import (ApiEmbedder, CachingEmbedder, EmbedderUnavailable, FastEmbedEmbedder, HashingEmbedder, cosine,
                                         make_embedder)
from supportagent.rag.expansion import CallableRewriter, ChainExpander, ThesaurusExpander, VocabCorrector
from supportagent.rag.hybrid import HybridIndex, get_retriever, rrf_fuse
from supportagent.rag.rerank import HeuristicReranker
from supportagent.rag.retriever import PolicyIndex, load_chunks
from supportagent.rag.text import porter_stem, tokenize


# ----------------------------------------------------------------------------------------------- text
@pytest.mark.parametrize("word,stem", [("caresses", "caress"), ("ponies", "poni"), ("agreed", "agre"), ("hopping", "hop"), ("relational", "relat"),
                                       ("refundable", "refund"), ("cancellation", "cancel"), ("replacement", "replac"), ("sized", "size")])
def test_porter_stemmer(word, stem):
    assert porter_stem(word) == stem


def test_lemmas_and_legacy_analyzer_frozen():
    assert tokenize("bought", "porter") == tokenize("buy", "porter") and tokenize("sent", "porter") == tokenize("send", "porter")
    assert tokenize("refunds shipping", "legacy") == ["refund", "shipp"]            # legacy suffix stripper must not change


def test_default_bm25_unchanged():
    """The shipped retriever is still the legacy-analyzer BM25 and still returns the same shape."""
    idx = PolicyIndex(POLICY_DIR)
    r = idx.search("how long do I have to return something", 2)
    assert r[0]["ref"] == "refund_policy#standard-returns" and set(r[0]) == {"doc", "section", "ref", "text", "score"}


# ----------------------------------------------------------------------------------------------- fusion
def test_rrf_fuse_math():
    fused = rrf_fuse([[0, 1, 2], [2, 1, 3]], k=60)
    d = {i: s for s, i in fused}
    assert d[1] == pytest.approx(1 / 62 + 1 / 62)
    assert d[0] == pytest.approx(1 / 61) and d[2] == pytest.approx(1 / 63 + 1 / 61)
    assert [i for _, i in fused][:2] == [2, 1]
    w = {i: s for s, i in rrf_fuse([[0], [1]], k=60, weights=[1.0, 3.0])}
    assert w[1] == pytest.approx(3 / 61) and w[0] == pytest.approx(1 / 61)


# ----------------------------------------------------------------------------------------------- embedders
def test_hashing_embedder_deterministic_and_typo_tolerant():
    texts = [c["section"] + ". " + c["text"] for c in load_chunks(POLICY_DIR)]
    a, b = HashingEmbedder(idf_texts=texts), HashingEmbedder(idf_texts=texts)
    q = "warrenty claim"
    assert a.embed([q], "query") == b.embed([q], "query")                               # deterministic across instances
    va, vb, vc = a.embed(["warrenty claim", "warranty claim", "shipping rates"], "query")
    assert cosine(va, vb) > cosine(va, vc) + 0.3                                         # the typo still lands near the real word
    assert cosine(va, va) == pytest.approx(1.0)


def test_caching_embedder_calls_inner_once():
    calls = []

    class Inner:
        name = "x"

        def embed(self, texts, kind="document"):
            calls.append(list(texts))
            return [[1.0, 0.0] for _ in texts]
    c = CachingEmbedder(Inner())
    c.embed(["a", "b"], "query")
    c.embed(["a"], "query")
    assert calls == [["a", "b"]]


def test_api_embedder_skipped_without_key(monkeypatch):
    for v in ("VOYAGE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    for prov in ("voyage", "openai"):
        with pytest.raises(EmbedderUnavailable):
            ApiEmbedder(prov)
    with pytest.raises(EmbedderUnavailable):
        make_embedder("voyage")


def test_api_embedder_request_and_parse_with_mocked_transport():
    seen = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=None):
        seen["url"], seen["body"], seen["auth"] = req.full_url, json.loads(req.data), req.get_header("Authorization")
        return Resp(json.dumps({"data": [{"index": 1, "embedding": [0.0, 2.0]}, {"index": 0, "embedding": [3.0, 0.0]}]}).encode())

    e = ApiEmbedder("voyage", api_key="k", opener=opener)
    out = e.embed(["a", "b"], "query")
    assert seen["url"].startswith("https://api.voyageai.com") and seen["auth"] == "Bearer k" and seen["body"]["input_type"] == "query"
    assert out == [[1.0, 0.0], [0.0, 1.0]]                                               # reordered by index and normalized


def test_auto_embedder_always_resolves_and_explicit_missing_is_loud():
    assert make_embedder("auto").name                                                    # never raises: falls back to hashing
    with pytest.raises(EmbedderUnavailable):
        make_embedder("nonsense")
    try:
        import fastembed  # noqa: F401
    except ImportError:
        with pytest.raises(EmbedderUnavailable):
            FastEmbedEmbedder()


# ----------------------------------------------------------------------------------------------- expansion / rerank
def test_expansion_hooks():
    t = ThesaurusExpander()
    assert "shipment" in t.expand("my package is late").split()
    assert "customs" not in t.expand("the customer wants help").split()                    # Porter would conflate customer/customs; the lookup must not
    vc = VocabCorrector({"refund", "digital", "courses"})
    assert "refund" in vc.expand("refnd for digital corses").split() and "courses" in vc.expand("refnd for digital corses").split()
    assert CallableRewriter(lambda q: "extra terms").expand("hello") == "hello extra terms"
    assert ChainExpander(CallableRewriter(lambda q: "a"), CallableRewriter(lambda q: "b")).expand("q") == "q a b"


def test_heuristic_reranker_prefers_title_match():
    cands = [{"ref": "x", "section": "Other", "text": "nothing here", "score": 1.0, "fused": 1.0},
             {"ref": "y", "section": "Lost Packages", "text": "refund within 14 days", "score": 0.95, "fused": 0.95},
             {"ref": "z", "section": "Else", "text": "unrelated", "score": 0.0, "fused": 0.0}]
    out = HeuristicReranker(title_weight=0.5).rerank("lost packages 14 days", cands)
    assert out[0]["ref"] == "y"


# ----------------------------------------------------------------------------------------------- hybrid index + flag wiring
@pytest.fixture(scope="module")
def hybrid():
    return HybridIndex(POLICY_DIR, embedder="hash")


def test_hybrid_refs_are_citable_and_identical_across_modes(hybrid):
    refs = {c["ref"] for c in hybrid.chunks}
    assert all("#" in r for r in refs) and len(refs) >= 45
    for mode in ("bm25", "bm25+", "dense", "hybrid"):
        for r in hybrid.rank("how long do I have to return something", mode, 5):
            assert r["ref"] in refs
    assert hybrid.search("how long do I have to return something", 1)[0]["ref"] == "refund_policy#standard-returns"


def test_hybrid_beats_bm25_on_typos_and_vocabulary_gap(hybrid):
    assert hybrid.rank("wat is the refnd window for digital corses", "bm25", 1)[0]["ref"] != "refund_policy#digital-goods"
    assert hybrid.rank("wat is the refnd window for digital corses", "hybrid", 1)[0]["ref"] == "refund_policy#digital-goods"
    top3 = [r["ref"] for r in hybrid.rank("my package never arrived", "hybrid", 3)]
    assert "refund_policy#lost-packages" in top3                                         # documented lexical miss (F6)


def test_abstain_hook(hybrid):
    h = HybridIndex(POLICY_DIR, embedder=hybrid.embedder, abstain=(0.99, 0.99))
    assert h.search("anything at all about refunds", 3) == []
    assert HybridIndex(POLICY_DIR, embedder=hybrid.embedder).search("refund window", 3)


def test_flag_default_is_bm25_and_hybrid_is_opt_in(monkeypatch):
    monkeypatch.delenv("SA_RETRIEVAL_MODE", raising=False)
    assert isinstance(get_retriever(str(POLICY_DIR)), PolicyIndex)
    monkeypatch.setenv("SA_RETRIEVAL_MODE", "hybrid")
    monkeypatch.setenv("SA_EMBEDDER", "hash")
    assert isinstance(get_retriever(str(POLICY_DIR)), HybridIndex)
    monkeypatch.setenv("SA_EMBEDDER", "voyage")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert isinstance(get_retriever(str(POLICY_DIR)), PolicyIndex)                    # clean fallback, not a crash
    assert any("falling back to bm25" in str(x.message) for x in w)


def test_kb_server_default_and_hybrid_flag(db, monkeypatch):
    monkeypatch.delenv("SA_RETRIEVAL_MODE", raising=False)
    base, err = build_registry(db).call("kb__search_policy", {"query": "How long do I have to return something?", "top_k": 2}, role="policy")
    assert not err and base["results"][0]["ref"] == "refund_policy#standard-returns" and "30 days" in base["results"][0]["text"]
    monkeypatch.setenv("SA_RETRIEVAL_MODE", "hybrid")
    monkeypatch.setenv("SA_EMBEDDER", "hash")
    hy, err = build_registry(db).call("kb__search_policy", {"query": "my package never arrived", "top_k": 3}, role="policy")
    assert not err and "refund_policy#lost-packages" in [r["ref"] for r in hy["results"]]


# ----------------------------------------------------------------------------------------------- dataset + metrics
def test_dataset_integrity_and_split():
    qs = load_questions()
    refs = {c["ref"] for c in load_chunks(POLICY_DIR)}
    assert len(qs) >= 100
    assert all(g in refs for q in qs for g in q["gold"])
    cats = {q["category"] for q in qs}
    assert {"direct", "paraphrase", "typo", "vocab_mismatch", "multi_section", "no_answer"} <= cats
    assert any(len(q["gold"]) > 1 for q in qs) and any(not q["gold"] for q in qs)
    sp = assign_splits(qs)
    assert sp == assign_splits(list(reversed(qs)))                                       # independent of file order
    dev = [q for q in qs if sp[q["id"]] == "dev"]
    test = [q for q in qs if sp[q["id"]] == "test"]
    assert len(dev) + len(test) == len(qs) and abs(len(dev) - len(test)) <= len(cats)
    for c in cats:                                                                        # stratified: every category appears in both splits
        assert any(q["category"] == c for q in dev) and any(q["category"] == c for q in test)


def test_tuning_only_sees_dev(monkeypatch):
    """The study's tuner functions receive the dev list only; assert generate() never passes a test id to them."""
    import evals.retrieval_eval as E
    qs = load_questions()
    sp = assign_splits(qs)
    seen = []
    orig = E.tune_hybrid

    def spy(index, dev):
        seen.extend(q["id"] for q in dev)
        return orig(index, dev)
    monkeypatch.setattr(E, "tune_hybrid", spy)
    systems = E.build_systems(POLICY_DIR, ("hash",), {}, [], dev=[q for q in qs if sp[q["id"]] == "dev"])
    assert seen and all(sp[i] == "dev" for i in seen) and systems


def test_metrics():
    assert M.recall_at_k(["a", "b", "c"], ["b", "z"], 3) == 0.5
    assert M.reciprocal_rank(["a", "b"], ["b"]) == 0.5 and M.reciprocal_rank(["a"], ["q"]) == 0.0
    assert M.ndcg_at_k(["a", "b"], ["a", "b"], 5) == pytest.approx(1.0)
    assert 0 < M.ndcg_at_k(["x", "a"], ["a"], 5) < 1
    p, lo, hi = M.bootstrap_ci([1.0] * 20 + [0.0] * 20, M.mean, B=500)
    assert p == 0.5 and 0.3 < lo < 0.5 < hi < 0.7
    rows = [(True, True), (True, False), (False, True), (False, False)]
    assert M.no_answer_precision(rows) == 0.5 and M.no_answer_recall(rows) == 0.5
    assert M.no_answer_precision([(False, True)]) != M.no_answer_precision([(False, True)])   # NaN when it never abstains
