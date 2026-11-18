"""Ranking metrics and bootstrap confidence intervals (stdlib only)."""
from __future__ import annotations

import math
import random
from typing import Callable, Sequence


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of gold refs found in the top-k (multi-section questions have several gold refs)."""
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def reciprocal_rank(ranked: Sequence[str], gold: Sequence[str], cutoff: int = 10) -> float:
    for i, r in enumerate(ranked[:cutoff], 1):
        if r in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(ranked[:k]) if r in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def bootstrap_ci(items: Sequence, stat: Callable[[Sequence], float], B: int = 2000, seed: int = 7, alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap over questions. Returns (point estimate, lo, hi). Resamples where stat is NaN are dropped."""
    n = len(items)
    point = stat(items)
    if n == 0:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    pop = range(n)
    vals = []
    for _ in range(B):
        v = stat([items[i] for i in rng.choices(pop, k=n)])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    vals.sort()
    return point, vals[int(alpha / 2 * len(vals))], vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def paired_diff_ci(a: Sequence[float], b: Sequence[float], B: int = 2000, seed: int = 11) -> tuple[float, float, float]:
    """Bootstrap CI for mean(a - b) over the same questions (paired)."""
    return bootstrap_ci([x - y for x, y in zip(a, b)], mean, B, seed)


def abstain_counts(abstained: Sequence[bool], no_answer: Sequence[bool]) -> tuple[int, int, int, int]:
    tp = sum(a and n for a, n in zip(abstained, no_answer))
    fp = sum(a and not n for a, n in zip(abstained, no_answer))
    fn = sum((not a) and n for a, n in zip(abstained, no_answer))
    tn = sum((not a) and (not n) for a, n in zip(abstained, no_answer))
    return tp, fp, fn, tn


def no_answer_precision(rows: Sequence[tuple[bool, bool]]) -> float:
    """rows: (abstained, truly_no_answer). Of the questions the system refused to answer, the share that really had no answer. NaN if it never abstains."""
    tp, fp, _, _ = abstain_counts([r[0] for r in rows], [r[1] for r in rows])
    return tp / (tp + fp) if tp + fp else float("nan")


def no_answer_recall(rows: Sequence[tuple[bool, bool]]) -> float:
    tp, _, fn, _ = abstain_counts([r[0] for r in rows], [r[1] for r in rows])
    return tp / (tp + fn) if tp + fn else float("nan")


def false_abstain_rate(rows: Sequence[tuple[bool, bool]]) -> float:
    _, fp, _, tn = abstain_counts([r[0] for r in rows], [r[1] for r in rows])
    return fp / (fp + tn) if fp + tn else float("nan")


def balanced_accuracy(rows: Sequence[tuple[bool, bool]]) -> float:
    r, f = no_answer_recall(rows), false_abstain_rate(rows)
    return (r + (1 - f)) / 2


def f1_abstain(rows: Sequence[tuple[bool, bool]]) -> float:
    p, r = no_answer_precision(rows), no_answer_recall(rows)
    p = 0.0 if p != p else p
    r = 0.0 if r != r else r
    return 2 * p * r / (p + r) if p + r else 0.0
