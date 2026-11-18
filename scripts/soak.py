"""Load / soak test for the harness with the SIMULATED model. Prints and writes JSON; every number in the output is measured.

    python scripts/soak.py --mode inproc --workers 4 --seconds 20 --out runs/soak.json
    python scripts/soak.py --mode http   --workers 4 --seconds 20
    python scripts/soak.py --mode inproc --workers 4 --seconds 20 --verifier      # cost/latency of the pre-money verifier

What it measures: harness wall time per turn (guards, context, tool dispatch over the in-process MCP transport, SQLite checkpoints,
audit chain, tracing) and end-to-end throughput. What it does NOT measure: any real model or network latency (the simulated model
answers in microseconds), a production database, or multi-process scaling. A live model adds its own latency to every model call,
so read these as the harness's overhead floor, not as user-visible latency. Python threads share one GIL: more workers raise
concurrency for SQLite/IO but cannot raise CPU-bound throughput.

Each worker owns an isolated runtime + shop DB (like evals/runner.py) and replays the golden scenarios with the scripted user for
the whole duration, so state never accumulates across scenarios in a way that changes outcomes.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sqlite3
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    v = sorted(vals)
    return round(v[max(0, min(len(v) - 1, int(round(q * len(v) + 0.5)) - 1))], 3)


def summarize(ms: list[float]) -> dict:
    return {"n": len(ms), "p50_ms": pct(ms, 0.50), "p95_ms": pct(ms, 0.95), "p99_ms": pct(ms, 0.99),
            "mean_ms": round(statistics.fmean(ms), 3) if ms else None, "max_ms": round(max(ms), 3) if ms else None}


class Result:
    def __init__(self):
        self.turn_ms, self.session_ms, self.statuses, self.errors, self.sessions = [], [], {}, [], 0
        self.lock = threading.Lock()

    def add(self, turns: list[float], session_ms: float, statuses: list[str]) -> None:
        with self.lock:
            self.turn_ms += turns
            self.session_ms.append(session_ms)
            self.sessions += 1
            for s in statuses:
                self.statuses[s] = self.statuses.get(s, 0) + 1

    def err(self, e: Exception) -> None:
        with self.lock:
            self.errors.append(f"{type(e).__name__}: {e}"[:160])


def inproc_worker(wid, cfg, scenarios, seed_db, deadline, res: Result, tmp: Path):
    from evals.runner import EMAILS
    from evals.simulator import ScriptedUser
    from supportagent.guardrails.approval import ApprovalService
    from supportagent.harness.checkpoint import CheckpointStore
    from supportagent.harness.runtime import Runtime
    from supportagent.llm.router import make_llm
    from supportagent.mcp.client import build_registry
    from supportagent.memory.store import CustomerMemory
    from supportagent.obs.tracing import Tracer

    i = wid
    while time.time() < deadline:
        scn = scenarios[i % len(scenarios)]
        i += 1000 + wid
        try:
            db = str(tmp / f"w{wid}-{i}.db")
            shutil.copy(seed_db, db)
            state = sqlite3.connect(":memory:", check_same_thread=False)
            ok = scn["expected"]["reviewer_decision"] == "approve"
            rt = Runtime(cfg, build_registry(db, cfg.allow_anon_server), CheckpointStore(state), ApprovalService(state, lambda rec: (ok, "soak")),
                         Tracer(), CustomerMemory(state), llm_factory=lambda tier, seed: make_llm(cfg, tier, seed))
            s = rt.new_session(scn["customer_id"], EMAILS[scn["customer_id"]], seed=i)
            user, turns, statuses, t_sess = ScriptedUser(scn["persona"]), [], [], time.perf_counter()
            msg = user.next(None)
            while msg:
                t0 = time.perf_counter()
                r = rt.handle_user_message(s, msg)
                turns.append((time.perf_counter() - t0) * 1000)
                statuses.append(r.status)
                if r.status not in ("done", "awaiting_user"):
                    break
                msg = user.next(r.reply)
            res.add(turns, (time.perf_counter() - t_sess) * 1000, statuses)
            os.remove(db)
        except Exception as e:                                    # a crash is a result, not something to hide
            res.err(e)


def run_inproc(args, cfg, scenarios, seed_db, tmp) -> Result:
    res, deadline = Result(), time.time() + args.seconds
    ts = [threading.Thread(target=inproc_worker, args=(w, cfg, scenarios, seed_db, deadline, res, tmp)) for w in range(args.workers)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return res


def run_http(args, cfg, scenarios, seed_db, tmp) -> Result:
    import urllib.request
    from http.server import ThreadingHTTPServer

    from supportagent.api import server
    from supportagent.config import RUNS_DIR
    assert str(RUNS_DIR) == str(tmp / "runs"), "SA_RUNS_DIR must point at the soak temp dir"
    app = server.App(cfg.name, "scripted")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    emails = app.emails
    usable = [s for s in scenarios if not s["expected"]["approval_requested"] and s["customer_id"] in emails]   # no human reviewer in this mode
    from evals.simulator import ScriptedUser
    res, deadline = Result(), time.time() + args.seconds

    def post(body, key):
        req = urllib.request.Request(base + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json", "Idempotency-Key": key})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def worker(wid):
        i = wid
        while time.time() < deadline:
            scn = usable[i % len(usable)]
            i += 1000 + wid
            try:
                user, turns, statuses, sid, t_sess = ScriptedUser(scn["persona"]), [], [], None, time.perf_counter()
                msg = user.next(None)
                while msg:
                    body = {"customer_id": scn["customer_id"], "message": msg, **({"session_id": sid} if sid else {})}
                    t0 = time.perf_counter()
                    r = post(body, f"soak-{wid}-{i}-{len(turns)}")
                    turns.append((time.perf_counter() - t0) * 1000)
                    sid, st = r["session_id"], r["status"]
                    statuses.append(st)
                    if st not in ("done", "awaiting_user"):
                        break
                    msg = user.next(r["reply"])
                res.add(turns, (time.perf_counter() - t_sess) * 1000, statuses)
            except Exception as e:
                res.err(e)
    ts = [threading.Thread(target=worker, args=(w,)) for w in range(args.workers)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    srv.shutdown()
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["inproc", "http"], default="inproc")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--config", default="full")
    ap.add_argument("--verifier", action="store_true", help="enable the pre-money verifier with threshold 0 (verify every money action)")
    ap.add_argument("--no-parallel", action="store_true", help="disable parallel reads and the tool cache")
    ap.add_argument("--out", default=str(ROOT / "runs" / "soak.json"))
    args = ap.parse_args()
    if args.mode == "http" and (args.verifier or args.no_parallel):
        ap.error("--verifier/--no-parallel apply to --mode inproc only (the HTTP app builds its own config)")
    tmp = Path(tempfile.mkdtemp(prefix="sa-soak-"))
    if args.mode == "http":
        os.environ["SA_RUNS_DIR"] = str(tmp / "runs")
    from evals.runner import load_scenarios
    from supportagent.config import PRESETS
    from supportagent.shop.db import build_seed_db
    cfg = PRESETS[args.config].with_(backend="scripted", verifier=args.verifier, verifier_threshold_cents=0 if args.verifier else PRESETS[args.config].verifier_threshold_cents,
                                     **({"parallel_tools": False, "tool_cache_ttl_s": 0.0} if args.no_parallel else {}))
    seed_db = str(tmp / "seed.db")
    build_seed_db(seed_db)
    scenarios = load_scenarios()
    t0 = time.time()
    try:
        res = (run_inproc if args.mode == "inproc" else run_http)(args, cfg, scenarios, seed_db, tmp)
    finally:
        wall = time.time() - t0
    out = {
        "what": "harness soak with the SIMULATED model (no real model/network latency; see module docstring)",
        "mode": args.mode, "config": {"name": cfg.name, "verifier": cfg.verifier, "parallel_tools": cfg.parallel_tools, "tool_cache_ttl_s": cfg.tool_cache_ttl_s},
        "workers": args.workers, "target_seconds": args.seconds, "wall_seconds": round(wall, 2),
        "sessions": res.sessions, "turns": len(res.turn_ms), "errors": len(res.errors), "error_samples": res.errors[:5],
        "throughput_note": ("wall time includes per-session setup (DB copy + registry build); latency figures exclude it" if args.mode == "inproc"
                            else "wall time includes client think-free scripted-user loop; server serialises turns behind one lock"),
        "throughput": {"turns_per_s": round(len(res.turn_ms) / wall, 2), "sessions_per_s": round(res.sessions / wall, 2)},
        "turn_latency": summarize(res.turn_ms), "session_latency": summarize(res.session_ms), "turn_status_counts": res.statuses,
        "env": {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if res.errors else 0)


if __name__ == "__main__":
    main()
