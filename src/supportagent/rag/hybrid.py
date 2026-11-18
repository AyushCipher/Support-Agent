"""Hybrid retrieval: BM25 (+ query expansion) and a dense embedder, fused with reciprocal rank fusion (RRF).

    score(chunk) = sum over rankers r of  weight_r / (rrf_k + rank_r(chunk))

Opt-in (ADR-0012). The shipped default stays lexical BM25; set SA_RETRIEVAL_MODE=hybrid (or dense | bm25+) to switch the
kb MCP server, and SA_EMBEDDER to pick the dense backend (auto | hash | fastembed[:model] | st[:model] | voyage | openai).
Citable refs (`<file>#<section>`) are identical across all modes because every mode ranks the same chunk list.
"""
from __future__ import annotations

import os
import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from .embeddings import CachingEmbedder, Embedder, EmbedderUnavailable, cosine, make_embedder
from .expansion import ChainExpander, ThesaurusExpander, VocabCorrector
from .rerank import Reranker
from .retriever import PolicyIndex, get_index, load_chunks

MODES = ("bm25", "bm25+", "dense", "hybrid")
# Hybrid hyper-parameters. Chosen on the DEV split only (evals/retrieval_eval.py prints the dev-selected values and flags drift).
RRF_K = 60
W_BM25 = 1.0
W_DENSE = 1.0
POOL = 10


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K, weights: list[float] | None = None) -> list[tuple[float, int]]:
    """Reciprocal rank fusion over ranked lists of chunk indices. Returns (fused score, index), best first."""
    weights = weights or [1.0] * len(rankings)
    s: dict[int, float] = defaultdict(float)
    for r, w in zip(rankings, weights):
        for pos, idx in enumerate(r, 1):
            s[idx] += w / (k + pos)
    return sorted(((v, i) for i, v in s.items()), key=lambda x: (-x[0], x[1]))


class DenseIndex:
    def __init__(self, chunks: list[dict], embedder: Embedder):
        self.chunks, self.embedder = chunks, embedder
        self.vecs = embedder.embed([f"{c['section']}. {c['text']}" for c in chunks], "document")
        self._memo: dict[str, list[tuple[float, int]]] = {}

    def rank(self, query: str) -> list[tuple[float, int]]:
        if query not in self._memo:
            q = self.embedder.embed([query], "query")[0]
            self._memo[query] = sorted(((cosine(q, v), i) for i, v in enumerate(self.vecs)), reverse=True)
        return self._memo[query]


class HybridIndex:
    def __init__(self, directory: Path, embedder: Embedder | str = "auto", expander="default", reranker: Reranker | None = None,
                 mode: str = "hybrid", rrf_k: int = RRF_K, w_bm25: float = W_BM25, w_dense: float = W_DENSE, pool: int = POOL,
                 abstain: tuple[float, float] | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode, self.rrf_k, self.w_bm25, self.w_dense, self.pool = mode, rrf_k, w_bm25, w_dense, pool
        self.reranker, self.abstain = reranker, abstain
        self.chunks = load_chunks(Path(directory))
        self.legacy = PolicyIndex(directory, chunks=self.chunks)                       # the shipped BM25, untouched
        self.bm25 = PolicyIndex(directory, analyzer="porter", chunks=self.chunks)      # improved stemming/lemmas
        self.embedder = make_embedder(embedder, [f"{c['section']}. {c['text']}" for c in self.chunks]) if isinstance(embedder, str) else embedder
        self.embedder = CachingEmbedder(self.embedder)
        self.dense = DenseIndex(self.chunks, self.embedder)
        if expander == "default":
            expander = ChainExpander(VocabCorrector(self.bm25.vocabulary()), ThesaurusExpander())
        self.expander = expander
        self._xmemo: dict[str, str] = {}
        self._cmemo: dict[str, dict] = {}

    # -- ranking -----------------------------------------------------------------------------------------
    def _expand(self, query: str) -> str:
        if not self.expander:
            return query
        if query not in self._xmemo:
            self._xmemo[query] = self.expander.expand(query)
        return self._xmemo[query]

    def _wrap(self, scored: list[tuple[float, int]]) -> list[dict]:
        return [{**self.chunks[i], "score": s, "fused": s} for s, i in scored]

    def rank(self, query: str, mode: str | None = None, limit: int = 10) -> list[dict]:
        mode = mode or self.mode
        if mode == "bm25":
            return self._wrap(self.legacy.rank(query))[:limit]
        if mode == "bm25+":
            return self._wrap(self.bm25.rank(self._expand(query)))[:limit]
        if mode == "dense":
            return self._wrap(self.dense.rank(query))[:limit]
        lex = [i for _, i in self.bm25.rank(self._expand(query))[: self.pool]]
        den = [i for _, i in self.dense.rank(query)[: self.pool]]
        fused = self._wrap(rrf_fuse([lex, den], self.rrf_k, [self.w_bm25, self.w_dense]))
        if self.reranker and fused:
            head = self.reranker.rerank(query, fused[: self.pool])
            fused = head + fused[self.pool:]
        return fused[:limit]

    def confidence(self, query: str) -> dict:
        """Two independent evidence signals used for no-answer abstention: idf-weighted BM25 term coverage and top dense cosine."""
        if query in self._cmemo:
            return self._cmemo[query]
        d = self.dense.rank(query)
        self._cmemo[query] = {"coverage": self.bm25.coverage(query), "legacy_coverage": self.legacy.coverage(query), "cosine": d[0][0] if d else 0.0}
        return self._cmemo[query]

    def should_abstain(self, query: str) -> bool:
        if not self.abstain:
            return False
        c = self.confidence(query)
        return c["coverage"] < self.abstain[0] and c["cosine"] < self.abstain[1]

    def search(self, query: str, k: int = 3) -> list[dict]:
        if self.should_abstain(query):
            return []
        return [{x: v for x, v in c.items() if x != "fused"} | {"score": round(c["score"], 4)} for c in self.rank(query, limit=k)]


@lru_cache(maxsize=4)
def _hybrid(directory: str, mode: str, embedder: str) -> HybridIndex:
    return HybridIndex(Path(directory), embedder=embedder, mode=mode)


def get_retriever(directory: str, mode: str | None = None):
    """Factory behind the kb MCP server. mode None -> $SA_RETRIEVAL_MODE -> 'bm25'.
    'bm25' returns the original dependency-free index, so default behaviour (and the eval ladder) is unchanged.
    Any failure to build a non-default retriever degrades to BM25 with a warning instead of breaking the agent."""
    mode = (mode or os.getenv("SA_RETRIEVAL_MODE") or "bm25").strip().lower()
    if mode == "bm25":
        return get_index(directory)
    if mode not in MODES:
        warnings.warn(f"unknown SA_RETRIEVAL_MODE {mode!r}; using bm25", stacklevel=2)
        return get_index(directory)
    try:
        return _hybrid(directory, mode, os.getenv("SA_EMBEDDER", "auto"))
    except (EmbedderUnavailable, OSError) as e:
        warnings.warn(f"retrieval mode {mode!r} unavailable ({e}); falling back to bm25", stacklevel=2)
        return get_index(directory)
