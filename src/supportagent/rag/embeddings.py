"""Pluggable dense backends for hybrid retrieval. All optional; the core stays stdlib-only (ADR-0001, ADR-0012).

  HashingEmbedder           deterministic offline fallback: TF-IDF over stemmed words + character n-grams, signed feature
                            hashing, L2-normalised. It handles typos and morphology but has NO semantic knowledge
                            (it will not know that "parcel" ~ "shipment"). It exists so CI stays offline and deterministic.
  FastEmbedEmbedder         local ONNX model via `pip install fastembed` (default BAAI/bge-small-en-v1.5)
  SentenceTransformerEmbedder  local model via `pip install sentence-transformers` (default all-MiniLM-L6-v2)
  ApiEmbedder               Voyage / OpenAI embeddings over HTTPS (stdlib urllib). Unavailable without a key.
                            Anthropic has no embeddings endpoint of its own and points to Voyage AI.

`make_embedder("auto")` picks the best installed local model and silently falls back to the hashing embedder.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from collections import Counter
from typing import Protocol

from .text import STOP, lemma_stem, words


class EmbedderUnavailable(RuntimeError):
    """Raised when an explicitly requested backend cannot be used (missing package, model, or API key)."""


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]: ...


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------- offline fallback
class HashingEmbedder:
    def __init__(self, dim: int = 2048, ngrams: tuple[int, ...] = (3, 4), idf_texts: list[str] | None = None):
        self.dim, self.ngrams = dim, ngrams
        self.name = f"hash-ngram-{dim}"
        self.idf: dict[int, float] = {}
        self.n_docs = 0
        if idf_texts:
            self.fit(idf_texts)

    def _features(self, text: str) -> Counter:
        feats: Counter = Counter()
        for w in words(text):
            if w in STOP:
                continue
            feats[("w", lemma_stem(w))] += 1
            padded = f"<{w}>"
            for n in self.ngrams:
                if len(padded) >= n:
                    for i in range(len(padded) - n + 1):
                        feats[("c", padded[i:i + n])] += 1
        return feats

    def _slot(self, feat: tuple) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(f"{feat[0]}:{feat[1]}".encode(), digest_size=8).digest(), "big")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def fit(self, texts: list[str]) -> None:
        df: Counter = Counter()
        for t in texts:
            df.update({self._slot(f)[0] for f in self._features(t)})
        self.n_docs = len(texts)
        self.idf = {s: math.log((1 + self.n_docs) / (1 + d)) + 1.0 for s, d in df.items()}

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        default_idf = math.log(1 + self.n_docs) + 1.0 if self.n_docs else 1.0
        for feat, tf in self._features(text).items():
            slot, sign = self._slot(feat)
            weight = 1.5 if feat[0] == "w" else 1.0          # whole (stemmed) words count a bit more than fragments
            v[slot] += sign * weight * (1 + math.log(tf)) * self.idf.get(slot, default_idf)
        return _norm(v)

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        return [self._vec(t) for t in texts]


# ---------------------------------------------------------------- optional local models
class FastEmbedEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding
        except Exception as e:                                          # ImportError, or a broken onnxruntime install
            raise EmbedderUnavailable(f"fastembed is not usable: {e}") from e
        try:
            self._m = TextEmbedding(model_name=model)
        except Exception as e:                                          # offline: model weights cannot be downloaded
            raise EmbedderUnavailable(f"fastembed model {model!r} could not be loaded: {e}") from e
        self.name = f"fastembed:{model}"

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        gen = self._m.query_embed(texts) if kind == "query" else self._m.passage_embed(texts)
        return [_norm([float(x) for x in v]) for v in gen]


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise EmbedderUnavailable(f"sentence-transformers is not usable: {e}") from e
        try:
            self._m = SentenceTransformer(model)
        except Exception as e:
            raise EmbedderUnavailable(f"sentence-transformers model {model!r} could not be loaded: {e}") from e
        self.name = f"sentence-transformers:{model}"

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        return [[float(x) for x in v] for v in self._m.encode(list(texts), normalize_embeddings=True)]


# ---------------------------------------------------------------- hosted APIs (skipped without a key)
class ApiEmbedder:
    """Voyage AI or OpenAI embeddings. Costs money per call: only constructed when the key env var is set.
    `opener` is injectable so tests can run the request/response handling without a network."""

    PROVIDERS = {
        "voyage": ("https://api.voyageai.com/v1/embeddings", "VOYAGE_API_KEY", "voyage-3.5-lite"),
        "openai": ("https://api.openai.com/v1/embeddings", "OPENAI_API_KEY", "text-embedding-3-small"),
    }

    def __init__(self, provider: str = "voyage", model: str | None = None, api_key: str | None = None, opener=None, timeout: float = 20.0):
        if provider not in self.PROVIDERS:
            raise EmbedderUnavailable(f"unknown embedding provider {provider!r}")
        self.url, env, default_model = self.PROVIDERS[provider]
        self.provider, self.model, self.timeout = provider, model or default_model, timeout
        self.key = api_key or os.getenv(env, "")
        if not self.key:
            raise EmbedderUnavailable(f"{env} is not set; the {provider} embedder is skipped")
        self._open = opener or urllib.request.urlopen
        self.name = f"api:{provider}:{self.model}"

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        body: dict = {"model": self.model, "input": list(texts)}
        if self.provider == "voyage":
            body["input_type"] = "query" if kind == "query" else "document"
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        with self._open(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode())
        rows = sorted(payload["data"], key=lambda d: d.get("index", 0))
        return [_norm([float(x) for x in r["embedding"]]) for r in rows]


class CachingEmbedder:
    """Memoizes embeddings per (kind, text): repeated queries (and hyper-parameter sweeps in the eval) cost nothing."""

    def __init__(self, inner: Embedder):
        self.inner, self.name, self._cache = inner, inner.name, {}

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        todo = [t for t in dict.fromkeys(texts) if (kind, t) not in self._cache]
        if todo:
            for t, v in zip(todo, self.inner.embed(todo, kind)):
                self._cache[(kind, t)] = v
        return [self._cache[(kind, t)] for t in texts]


# ---------------------------------------------------------------- factory
def make_embedder(spec: str = "auto", idf_texts: list[str] | None = None) -> Embedder:
    """spec: auto | hash | fastembed[:model] | st[:model] | voyage[:model] | openai[:model].

    'auto' tries local models and falls back to the offline hashing embedder; explicit specs raise EmbedderUnavailable
    instead of silently degrading, so a mis-configured deployment is visible."""
    kind, _, model = spec.partition(":")
    if kind == "hash":
        return HashingEmbedder(idf_texts=idf_texts)
    if kind == "fastembed":
        return FastEmbedEmbedder(model) if model else FastEmbedEmbedder()
    if kind in ("st", "sentence-transformers"):
        return SentenceTransformerEmbedder(model) if model else SentenceTransformerEmbedder()
    if kind in ("voyage", "openai"):
        return ApiEmbedder(kind, model or None)
    if kind == "auto":
        for ctor in (FastEmbedEmbedder, SentenceTransformerEmbedder):
            try:
                return ctor()
            except EmbedderUnavailable:
                continue
        return HashingEmbedder(idf_texts=idf_texts)
    raise EmbedderUnavailable(f"unknown embedder spec {spec!r}")
