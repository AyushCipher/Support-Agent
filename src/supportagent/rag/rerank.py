"""Rerank hook: reorder the fused candidate pool before truncating to top-k.

A reranker is any object with `name` and `rerank(query, candidates) -> candidates` (best first, `score` updated).
Candidates are the chunk dicts produced by the retriever and carry `fused` (the pre-rerank score).

  HeuristicReranker      dependency-free: boosts candidates whose heading shares stems with the query and whose text
                         contains the query's numbers (e.g. "200 dollars", "14 days").
  CrossEncoderReranker   optional: a cross-encoder model (fastembed's ONNX `TextCrossEncoder` or sentence-transformers'
                         `CrossEncoder`); raises RerankerUnavailable if neither package/model can be loaded.
"""
from __future__ import annotations

import re
from typing import Protocol

from .text import tokenize


class RerankerUnavailable(RuntimeError):
    pass


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]: ...


def _minmax(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    return [0.5] * len(xs) if hi - lo < 1e-12 else [(x - lo) / (hi - lo) for x in xs]


class HeuristicReranker:
    def __init__(self, title_weight: float = 0.25, number_weight: float = 0.1):
        self.title_weight, self.number_weight = title_weight, number_weight
        self.name = f"heuristic(title={title_weight},num={number_weight})"

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return candidates
        q_terms = set(tokenize(query, "porter"))
        q_nums = set(re.findall(r"\b\d+\b", query))
        base = _minmax([c.get("fused", c["score"]) for c in candidates])
        out = []
        for c, b in zip(candidates, base):
            title = set(tokenize(c["section"], "porter"))
            t = len(title & q_terms) / len(title) if title else 0.0
            n = (len(q_nums & set(re.findall(r"\b\d+\b", c["text"]))) / len(q_nums)) if q_nums else 0.0
            out.append({**c, "score": round(b + self.title_weight * t + self.number_weight * n, 4)})
        return sorted(out, key=lambda c: c["score"], reverse=True)


class CrossEncoderReranker:
    def __init__(self, model: str | None = None, backend: str = "auto", blend: float = 0.2):
        self.blend = blend                      # weight of the first-stage (fused) score kept in the final score
        self._score = None
        errs = []
        if backend in ("auto", "fastembed"):
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                m = TextCrossEncoder(model_name=model or "Xenova/ms-marco-MiniLM-L-6-v2")
                self._score = lambda q, docs: [float(x) for x in m.rerank(q, docs)]
                self.name = f"cross-encoder:fastembed:{model or 'Xenova/ms-marco-MiniLM-L-6-v2'}"
            except Exception as e:
                errs.append(f"fastembed: {e}")
        if self._score is None and backend in ("auto", "st"):
            try:
                from sentence_transformers import CrossEncoder
                m2 = CrossEncoder(model or "cross-encoder/ms-marco-MiniLM-L-6-v2")
                self._score = lambda q, docs: [float(x) for x in m2.predict([(q, d) for d in docs])]
                self.name = f"cross-encoder:st:{model or 'cross-encoder/ms-marco-MiniLM-L-6-v2'}"
            except Exception as e:
                errs.append(f"sentence-transformers: {e}")
        if self._score is None:
            raise RerankerUnavailable("; ".join(errs) or f"unknown backend {backend!r}")

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return candidates
        ce = _minmax(self._score(query, [f"{c['section']}. {c['text']}" for c in candidates]))
        fz = _minmax([c.get("fused", c["score"]) for c in candidates])
        out = [{**c, "score": round((1 - self.blend) * a + self.blend * b, 4)} for c, a, b in zip(candidates, ce, fz)]
        return sorted(out, key=lambda c: c["score"], reverse=True)
