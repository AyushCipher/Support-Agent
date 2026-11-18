"""Sampled ONLINE evaluation: no ground truth in production, so score a sample of real sessions with reference-free checks
(claim verification, leak scan, guardrail-event rates, escalation rates) and an optional LLM judge.

  python -m evals.online_eval --state runs/state.db --rate 0.2
Wire this to a cron/queue; alert when violation rates move.
"""
from __future__ import annotations

import argparse
import random
import sqlite3

from supportagent.guardrails.output import OutputGuard
from supportagent.guardrails.patterns import SECRET_RE
from supportagent.harness.checkpoint import CheckpointStore


def evaluate_session(s) -> dict:
    guard = OutputGuard()
    replies = [m.content for m in s.messages if m.role == "assistant" and m.content]
    issues = []
    for r in replies:
        v = guard.check(s, r, s.evidence)
        issues += v.findings
        if SECRET_RE.search(r):
            issues.append("leak:secret")
    ev = [e["type"] for e in s.events]
    return {"session": s.id, "issues": sorted(set(issues)), "tainted": s.tainted, "escalated": s.evidence["escalated"],
            "blocked": ev.count("tool_blocked"), "approvals": ev.count("approval_requested"), "cost_usd": s.cost_usd}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="runs/state.db")
    ap.add_argument("--rate", type=float, default=0.2)
    a = ap.parse_args()
    store = CheckpointStore(sqlite3.connect(a.state, check_same_thread=False))
    rows = store.list(limit=1000)
    sample = [r for r in rows if random.random() < a.rate]
    res = [evaluate_session(store.load(r["id"])) for r in sample]
    bad = [r for r in res if r["issues"]]
    print(f"sampled {len(res)}/{len(rows)} sessions; {len(bad)} with issues ({len(bad) / max(1, len(res)):.1%})")
    for r in bad[:10]:
        print("  ", r)


if __name__ == "__main__":
    main()
