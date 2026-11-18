"""Run tau-bench retail (test split) with supportagent and/or tau-bench's plain ToolCallingAgent as the agent.

  python -m taubench_sa.run --agents supportagent baseline --tasks 0-4 --trials 1 --workers 4 --rpm 60 --max-requests 2000

Resume: every finished episode is appended to <out>/results.jsonl and skipped on the next run. Cache: every model response is stored
under results/cache/ keyed by (role, model, agent|task|trial, prompt, messages, tools), so a re-run of an interrupted episode replays
its prefix for free. Pairing: for each (task, trial) the requested agents run back to back, so a stopped run still has matched pairs.
Episodes that crash on infrastructure (provider errors after retries) go to errors.jsonl and are excluded from rates (counted in the report).
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .llm import CTX, Budget, CachedGemini, LiteLLMShim, RunStop, install_litellm_shim

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"


def parse_tasks(spec: list[str]) -> list[int]:
    out: list[int] = []
    for s in spec:
        for part in s.split(","):
            if "-" in part:
                a, b = part.split("-")
                out += list(range(int(a), int(b) + 1))
            elif part:
                out.append(int(part))
    return out


def jsonl_writer(path: Path):
    lock = threading.Lock()

    def w(obj: dict) -> None:
        with lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")
    return w


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out


def make_user_env(get_env, user_model: str, task_index: int):
    """tau-bench retail env whose LLM user simulator is created WITHOUT tau-bench's eager reset() at construction (that call has no
    instruction and its output is discarded; skipping it saves one request per episode and does not change any episode)."""
    from tau_bench.envs.user import BaseUserSimulationEnv, LLMUserSimulationEnv

    class LazyLLMUser(LLMUserSimulationEnv):
        def __init__(self, model: str, provider: str) -> None:       # noqa: D401 - same state as the parent, minus the eager reset()
            BaseUserSimulationEnv.__init__(self)
            self.messages, self.model, self.provider, self.total_cost = [], model, provider, 0.0

    env = get_env("retail", user_strategy="human", user_model=user_model, task_split="test", user_provider="gemini", task_index=task_index)
    env.user = LazyLLMUser(user_model, "gemini")
    return env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=["supportagent", "baseline"], choices=["supportagent", "baseline"])
    ap.add_argument("--tasks", nargs="+", default=["0-4"])
    ap.add_argument("--shuffle-seed", type=int, default=None, help="run tasks in a seeded random order (a run stopped by quota then covers a spread of tasks)")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--trial-start", type=int, default=1, help="first trial index (e.g. 2 to add trials 2..N to an existing run)")
    ap.add_argument("--fast-model", default="gemini-3.5-flash-lite")
    ap.add_argument("--strong-model", default="gemini-3.5-flash")
    ap.add_argument("--baseline-model", default="gemini-3.5-flash-lite")
    ap.add_argument("--user-model", default="gemini-3.6-flash")
    ap.add_argument("--user-thinking", default=None, help="Gemini thinking level for the USER SIMULATOR only (e.g. low); default = model default")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rpm", type=float, default=60.0, help="process-wide request pacing (0 = off)")
    ap.add_argument("--max-requests", type=int, default=2000)
    ap.add_argument("--out", default="retail_test")
    a = ap.parse_args(argv)

    out = RESULTS / a.out
    out.mkdir(parents=True, exist_ok=True)
    cache = RESULTS / "cache"
    budget = Budget(a.max_requests)
    log_calls = jsonl_writer(out / "llm_calls.jsonl")
    write_result, write_error = jsonl_writer(out / "results.jsonl"), jsonl_writer(out / "errors.jsonl")

    def make_llm(model: str, role: str) -> CachedGemini:
        return CachedGemini(model, role, budget, cache, rpm=a.rpm, log=log_calls, thinking=a.user_thinking if role == "user" else None)

    install_litellm_shim(LiteLLMShim(make_llm))
    from .adapter import categorize, import_tau, run_baseline_episode, run_supportagent_episode, supportagent_config
    get_env, Action = import_tau()

    tiers = {"fast": a.fast_model, "strong": a.strong_model, "verifier": a.strong_model}
    labels = {"supportagent": f"supportagent[{a.fast_model}->{a.strong_model}]", "baseline": f"tau-tool-calling[{a.baseline_model}]"}
    done = {(r["agent"], r["task_id"], r["trial"]) for r in read_jsonl(out / "results.jsonl")}
    order = parse_tasks(a.tasks)
    if a.shuffle_seed is not None:
        import random
        random.Random(a.shuffle_seed).shuffle(order)
    jobs = [(t, i, ag) for t in range(a.trial_start, a.trial_start + a.trials) for i in order for ag in a.agents
            if (labels[ag], i, t) not in done]
    print(f"{len(jobs)} episodes to run ({len(done)} already in {out.name}/results.jsonl)", flush=True)
    stop = threading.Event()
    stop_reason: list[str] = []
    counts = {"completed": 0, "errored": 0}
    t0 = time.time()

    def job(trial: int, i: int, ag: str) -> None:
        if stop.is_set():
            return
        label = labels[ag]
        CTX.run_key, CTX.calls_by_model = f"{label}|{i}|{trial}", {}
        env = make_user_env(get_env, a.user_model, i)
        ts = time.time()
        try:
            if ag == "supportagent":
                res = run_supportagent_episode(env, i, lambda tier, seed: make_llm(tiers[tier], "agent"), Action, supportagent_config())
            else:
                res = run_baseline_episode(env, i, a.baseline_model, Action)
        except RunStop as e:
            stop.set()
            stop_reason.append(str(e))
            return
        except Exception as e:                                          # infrastructure failure: excluded from rates, counted
            counts["errored"] += 1
            write_error({"agent": label, "task_id": i, "trial": trial, "error": f"{type(e).__name__}: {e}"[:400],
                         "trace": traceback.format_exc()[-1500:], "ts": round(time.time(), 1)})
            print(f"  ERROR {label} task {i} trial {trial}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            return
        if res["termination"] == "infra_error":
            counts["errored"] += 1
            write_error({"agent": label, "task_id": i, "trial": trial, "error": "llm_unavailable (all tiers failed) -> harness handoff", "ts": round(time.time(), 1)})
            return
        gt = [x.model_dump() for x in env.task.actions]
        res.update(agent=label, task_id=i, trial=trial, category=categorize(res, gt, bool(env.task.outputs)), user_model=a.user_model,
                   wall_s=round(time.time() - ts, 1), ts=round(time.time(), 1), gt_actions=gt, outputs=env.task.outputs)
        write_result(res)
        counts["completed"] += 1
        print(f"  {label:48s} task {i:3d} trial {trial}: reward={res['reward']:.0f} {res['termination']:10s} {res['category']:28s} "
              f"calls={res['n_agent_calls']} ({res['wall_s']}s)  [req={budget.requests} hits={budget.cache_hits}]", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futs = [pool.submit(job, *j) for j in jobs]
        for f in as_completed(futs):
            f.result()
    meta = {"args": vars(a), "labels": labels, "planned_this_run": len(jobs), "completed_this_run": counts["completed"],
            "errored_this_run": counts["errored"], "not_run_this_run": len(jobs) - counts["completed"] - counts["errored"],
            "stopped": stop_reason[:1], "network_requests_this_run": budget.requests, "cache_hits_this_run": budget.cache_hits,
            "tokens_this_run": budget.tokens, "wall_seconds": round(time.time() - t0, 1),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with (out / "run_meta.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
