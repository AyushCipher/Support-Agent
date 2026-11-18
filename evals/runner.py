"""Eval runner: scenarios x configs x N trials, parallel, fully isolated (fresh DB + fresh state per trial).

  python -m evals.runner --config full --trials 8
  python -m evals.runner --all --trials 8          # ablation ladder
  python -m evals.runner --all --set heldout --trials 8   # held-out set (results in evals/results/heldout/)
  python -m evals.live --config full ...                  # robust live runs (resume, cache, backoff, cost cap): see evals/live.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from supportagent.config import PRESETS, AgentConfig
from supportagent.guardrails.approval import ApprovalService
from supportagent.harness.checkpoint import CheckpointStore
from supportagent.harness.runtime import Runtime
from supportagent.mcp.client import build_registry
from supportagent.memory.store import CustomerMemory
from supportagent.obs.tracing import MemorySink, Tracer
from supportagent.prompts import registry as prompts
from supportagent.llm.router import make_llm
from supportagent.security.audit import AuditLog
from supportagent.shop.db import CUSTOMERS, ORDERS, build_seed_db

from .fixtures import FaultyRegistry, apply_setup
from .judge import RuleJudge
from .metrics import aggregate
from .scoring import read_end_state, score_end_state, score_reply, score_safety, score_trajectory
from .simulator import ScriptedUser

HERE = Path(__file__).parent
RESULTS = HERE / "results"
EMAILS = {c[0]: c[2] for c in CUSTOMERS}
PII_VALUES = [c[2] for c in CUSTOMERS] + sorted({o[11] for o in ORDERS})      # emails + street addresses that must not reach the LLM


class Recorder:
    """Wraps an LLM and records every byte sent to the provider, so we can measure PII egress to the model vendor."""
    def __init__(self, inner, sink: list):
        self.inner, self.sink, self.name = inner, sink, inner.name

    def complete(self, system, messages, tools, max_tokens=1024, tool_choice=None):
        self.sink.append(system + "\n" + "\n".join(m.content + json.dumps([t.arguments for t in m.tool_calls]) for m in messages))
        return self.inner.complete(system, messages, tools, max_tokens, tool_choice)


def load_scenarios(only: list[str] | None = None, heldout: bool = False) -> list[dict]:
    """heldout=True loads evals/scenarios_heldout/ (authored separately; never used to tune guards)."""
    out = []
    for f in sorted((HERE / ("scenarios_heldout" if heldout else "scenarios")).glob("*.json")):
        out += json.loads(f.read_text())
    return [s for s in out if not only or s["id"] in only]


def run_trial(scn: dict, cfg: AgentConfig, seed: int, seed_db: str, judge=None, llm_maker=None) -> dict:
    """llm_maker(cfg, tier, seed) -> LLM. Default: the backend named in cfg (scripted/anthropic). Live runs pass a wrapper
    that adds caching, backoff and cost accounting (evals/live.py)."""
    make = llm_maker or make_llm
    tmp = Path(tempfile.mkdtemp(prefix="sa-eval-"))
    try:
        db = str(tmp / "shop.db")
        shutil.copy(seed_db, db)
        apply_setup(db, scn.get("setup"))
        registry = build_registry(db, cfg.allow_anon_server)
        if scn.get("faults"):
            registry = FaultyRegistry(registry, scn["faults"])
        state = sqlite3.connect(":memory:", check_same_thread=False)
        reviewer = scn["expected"]["reviewer_decision"] == "approve"
        approvals = ApprovalService(state, lambda rec: (reviewer, "approved by reviewer" if reviewer else "does not meet criteria"))
        sink = MemorySink()
        sent: list[str] = []
        extra = {}
        if llm_maker is not None and getattr(cfg, "verifier", False):       # a live run must not silently use the simulated verifier
            extra["verifier_llm_factory"] = lambda: Recorder(make(cfg, "verifier", seed), sent)
        rt = Runtime(cfg, registry, CheckpointStore(state), approvals, Tracer([sink]), CustomerMemory(state),
                     llm_factory=lambda tier, seed: Recorder(make(cfg, tier, seed), sent), **extra)
        s = rt.new_session(scn["customer_id"], EMAILS[scn["customer_id"]], seed=seed, session_id=f"{scn['id']}-{seed}")
        user = ScriptedUser(scn["persona"]) if not getattr(cfg, "llm_user", False) else None
        transcript, replies, t0 = [], [], time.time()
        msg, status = user.next(None), "active"
        while msg:
            transcript.append({"role": "customer", "text": msg})
            r = rt.handle_user_message(s, msg)
            status = r.status
            transcript.append({"role": "agent", "text": r.reply})
            replies.append(r.reply)
            if r.status not in ("done", "awaiting_user"):
                break
            msg = user.next(r.reply)
        latency = (time.time() - t0) * 1000
        end = read_end_state(db)
        calls = [tc.name for m in s.messages if m.role == "assistant" for tc in m.tool_calls
                 if tc.name not in ("handoff", "classify_intent")]
        end_ok, end_why = score_end_state(scn["expected"], end)
        traj_ok, traj_why = score_trajectory(scn["trajectory"], calls)
        harm_ok, honest_ok, safe_why = score_safety(scn, end, s.events, replies)
        reply_ok, reply_why = score_reply(scn, replies, s.evidence["refund_amounts"])
        appr_ok = scn["expected"]["approval_requested"] in (None, any(e["type"] == "approval_requested" for e in s.events))
        pii_hits = sorted({v for v in PII_VALUES if any(v in blob for blob in sent)})
        audit_ok = AuditLog(state).verify()[0] and AuditLog(sqlite3.connect(db)).verify()[0]
        trial = {
            "scenario_id": scn["id"], "category": scn["category"], "seed": seed, "attack": scn["safety"]["attack"],
            "end_state_ok": end_ok, "trajectory_ok": traj_ok, "safety_ok": harm_ok, "honesty_ok": honest_ok, "reply_ok": reply_ok, "approval_ok": appr_ok,
            "attack_success": bool(scn["safety"]["attack"]) and not harm_ok,
            "why": end_why + traj_why + safe_why + reply_why + ([] if appr_ok else ["approval expectation not met"]),
            "tool_calls": calls, "transcript": transcript, "replies": replies, "end_state": end, "status": status,
            "event_types": [e["type"] for e in s.events], "events": s.events, "models": s.models_used,
            "pii_to_llm": bool(pii_hits), "pii_hits": pii_hits, "audit_chain_ok": audit_ok, "cost_usd": s.cost_usd, "tokens": s.tokens, "steps": s.step, "latency_ms": latency,
        }
        trial["judge"] = (judge or RuleJudge()).judge(scn, trial)["verdict"]
        # Outcome-based success (tau-bench style): DB end-state + safety + reply + approval expectation.
        # Trajectory discipline (no blocked/forbidden attempts, required tools) is reported separately.
        trial["passed"] = end_ok and harm_ok and honest_ok and reply_ok and appr_ok
        trial["prompt_hashes"] = s.prompt_hashes
        return trial
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _work(args):
    scn, cfg, seed, seed_db = args
    return run_trial(scn, cfg, seed, seed_db)


def run_config(cfg: AgentConfig, trials: int, workers: int, only: list[str] | None = None, quiet: bool = False, out_name: str | None = None,
               heldout: bool = False, results_dir: Path | None = None) -> dict:
    scns = load_scenarios(only, heldout)
    seed_db = tempfile.mkdtemp(prefix="sa-seed-") + "/seed.db"
    build_seed_db(seed_db)
    jobs = [(s, cfg, seed, seed_db) for s in scns for seed in range(trials)]
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_work, jobs, chunksize=8))
    else:
        results = [_work(j) for j in jobs]
    agg = aggregate(results, trials)
    out = {"config": asdict(cfg), "prompt_hashes": prompts.manifest(cfg.prompt_set), "backend": cfg.backend,
           "scenario_set": "heldout" if heldout else "main", "model": "simulated (ScriptedLLM)" if cfg.backend == "scripted" else cfg.backend,
           "summary": agg, "trials": results, "wall_seconds": round(time.time() - t0, 1)}
    rd = results_dir or (RESULTS / "heldout" if heldout else RESULTS)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / f"{out_name or cfg.name.replace('+', 'plus_')}.json").write_text(json.dumps(out, indent=1, default=str))
    if not quiet:
        a = agg
        print(f"[{cfg.name:>13}] success {a['task_success']:.1%}  ASR {a['attack_success_rate']:.1%}  safety-viol {a['safety_violation_rate']:.1%}  "
              f"pass^{trials} {a['pass_hat_k'][str(trials)]:.1%}  cost/run ${a['avg_cost_usd']:.4f}  ({out['wall_seconds']}s)")
    return out


def main() -> None:
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full", choices=list(PRESETS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--backend", default="scripted", choices=["scripted", "anthropic"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--set", default="main", choices=["main", "heldout"], help="heldout = evals/scenarios_heldout/ (results in evals/results/heldout/)")
    a = ap.parse_args()
    names = list(PRESETS) if a.all else [a.config]
    for n in names:
        cfg = PRESETS[n].with_(backend=a.backend, llm_backoff_s=1.0 if a.backend == "anthropic" else 0.0)
        run_config(cfg, a.trials, a.workers, a.only, heldout=a.set == "heldout")


if __name__ == "__main__":
    main()
