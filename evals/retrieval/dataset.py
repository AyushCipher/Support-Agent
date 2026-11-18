"""Labeled retrieval questions (evals/retrieval/questions.jsonl) and the dev/test split.

PROVENANCE: every question and gold label was written by Claude (an LLM), not by a human annotator, and has not been
human-reviewed. See evals/retrieval/PROVENANCE.md.

The split is deterministic and stratified: inside each category, questions are ordered by sha256("sa-retrieval-v1:" + id)
and alternate dev, test, dev, test ... No random seed, and no peeking at any metric, is involved. Hyper-parameters,
abstention thresholds and query-expansion choices may be selected on DEV only; TEST is read once for the final report.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
QUESTIONS = HERE / "questions.jsonl"
SPLIT_SALT = "sa-retrieval-v1:"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    qs = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [q["id"] for q in qs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    return qs


def assign_splits(questions: list[dict]) -> dict[str, str]:
    by_cat: dict[str, list[str]] = {}
    for q in questions:
        by_cat.setdefault(q["category"], []).append(q["id"])
    out: dict[str, str] = {}
    for ids in by_cat.values():
        for pos, qid in enumerate(sorted(ids, key=lambda i: hashlib.sha256((SPLIT_SALT + i).encode()).hexdigest())):
            out[qid] = "dev" if pos % 2 == 0 else "test"
    return out


def split_questions(questions: list[dict], name: str) -> list[dict]:
    sp = assign_splits(questions)
    return [q for q in questions if sp[q["id"]] == name]
