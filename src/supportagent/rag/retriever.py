"""Dependency-free BM25 retriever over policy markdown, chunked by H2 heading, with citable refs.

This is the default retriever (ADR-0001: core stays stdlib-only). `analyzer="legacy"` reproduces the original
tokenization exactly; `analyzer="porter"` uses the improved stemmer/lemmas from `rag.text`. Hybrid (BM25 + dense,
RRF fusion) lives in `rag.hybrid` and is opt-in (ADR-0012).
"""
from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .text import STOP, slug, tokenize  # noqa: F401  (STOP re-exported for backwards compatibility)
from .text import legacy_stem as _stem  # noqa: F401


def _tok(s: str) -> list[str]:
    return tokenize(s, "legacy")


_slug = slug


def load_chunks(directory: Path) -> list[dict]:
    """One chunk per H2 section. ref = '<file stem>#<slug of heading>' (the citable reference)."""
    chunks: list[dict] = []
    for f in sorted(Path(directory).glob("*.md")):
        section, buf = "overview", []
        for line in f.read_text(encoding="utf-8").splitlines() + ["## __end__"]:
            if line.startswith("## "):
                text = " ".join(buf).strip()
                if text:
                    chunks.append({"doc": f.stem, "section": section, "ref": f"{f.stem}#{slug(section)}", "text": text})
                section, buf = line[3:].strip(), []
            elif not line.startswith("# "):
                buf.append(line.strip())
    return chunks


class PolicyIndex:
    def __init__(self, directory: Path, k1: float = 1.5, b: float = 0.75, analyzer: str = "legacy", chunks: list[dict] | None = None):
        self.k1, self.b, self.analyzer = k1, b, analyzer
        self.chunks: list[dict] = chunks if chunks is not None else load_chunks(Path(directory))
        self.tfs = [Counter(self.tok(c["section"] + " " + c["text"])) for c in self.chunks]
        self.dl = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.dl) / max(1, len(self.dl))
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.idf_max = math.log(1 + (n + 0.5) / 0.5)          # idf of a term that appears in no chunk

    def tok(self, s: str) -> list[str]:
        return tokenize(s, self.analyzer)

    def vocabulary(self) -> set[str]:
        """Surface (unstemmed) words of the corpus; used by spelling correction."""
        from .text import words
        return {w for c in self.chunks for w in words(c["section"] + " " + c["text"])}

    def rank(self, query: str) -> list[tuple[float, int]]:
        """All chunks with a positive BM25 score, best first, as (score, chunk index)."""
        q = self.tok(query)
        scored = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return scored

    def coverage(self, query: str) -> float:
        """Top BM25 score divided by its theoretical ceiling for this query (idf-weighted share of query terms matched), in [0, 1].
        Unknown query terms count at the maximum idf, so an off-topic query scores low."""
        r = self.rank(query)
        terms = set(self.tok(query))
        ceil = sum(self.idf.get(t, self.idf_max) * (self.k1 + 1) for t in terms)
        return (r[0][0] / ceil) if r and ceil else 0.0

    def search(self, query: str, k: int = 3) -> list[dict]:
        return [{**self.chunks[i], "score": round(s, 3)} for s, i in self.rank(query)[:k]]


@lru_cache(maxsize=4)
def get_index(directory: str) -> PolicyIndex:
    return PolicyIndex(Path(directory))
