from evals.retrieval_eval import run
from supportagent.config import POLICY_DIR
from supportagent.rag.retriever import PolicyIndex


def test_chunks_have_citable_refs():
    idx = PolicyIndex(POLICY_DIR)
    assert all("#" in c["ref"] for c in idx.chunks) and len(idx.chunks) >= 12


def test_retrieval_quality_floor():
    r = run(3)
    assert r["recall@3"] >= 0.8 and r["mrr"] >= 0.7        # known lexical misses are documented (F6); this is a regression floor


def test_top_hit():
    assert PolicyIndex(POLICY_DIR).search("how long do I have to return something", 1)[0]["ref"] == "refund_policy#standard-returns"
