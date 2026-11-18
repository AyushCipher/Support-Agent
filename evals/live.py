"""Robust live-model eval runner (`make eval-live`).  Works with a real key (Anthropic or Gemini) or, with --dry-run, against a
local mock HTTP server that answers with the ScriptedLLM through the SAME client code path (SDK/urllib -> HTTP -> adapter).

Robustness features
  * resume      every finished trial is appended to <run-dir>/trials.jsonl; a re-run skips trials already there
  * disk cache  every model response is cached by sha256(backend, model, seed, system, messages, tools, ...); a crashed or
                repeated trial replays for free and deterministically (cache hits are not counted as requests or cost)
  * backoff     429 / 5xx / overloaded -> exponential backoff + jitter, Retry-After honoured; daily-quota exhaustion stops the run cleanly
  * caps        --max-usd (estimated from PLACEHOLDER prices in config.PRICING) and --max-requests (network requests); the run stops
                cleanly when a cap is hit and the partial results are still reported
  * logs        trials.jsonl (one line per trial, full transcript + events), llm_calls.jsonl (one line per model call),
                errors.jsonl (trials that crashed: excluded from rates, counted in the report), run_meta.json
  * pacing      --rpm N spaces requests (free tiers)
Paired design: for each (scenario, seed) every requested config runs back to back, so an interrupted run still has matched pairs.

  python -m evals.live --backend gemini --set heldout --config baseline full --trials 1 --rpm 15 --max-requests 400
  python -m evals.live --dry-run --backend anthropic --set main --trials 1 --mock-fault 7:429
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from supportagent.config import MODEL_TIERS, PRESETS
from supportagent.harness.types import LLMResponse, ToolCall, Usage
from supportagent.llm.base import TransientLLMError
from supportagent.obs.cost import cost_usd
from supportagent.shop.db import build_seed_db

from .runner import load_scenarios, run_trial

HERE = Path(__file__).parent
LIVE_DIR = HERE / "results" / "live"


class BudgetExceeded(Exception):
    """--max-usd or --max-requests reached; the run stops cleanly."""


class Budget:
    def __init__(self, max_usd: float | None, max_requests: int | None):
        self.max_usd, self.max_requests = max_usd, max_requests
        self.usd = 0.0
        self.requests = 0                  # network requests (cache hits excluded)
        self.cache_hits = 0
        self.retries = 0
        self.lock = threading.Lock()

    def check(self) -> None:
        with self.lock:
            if self.max_usd is not None and self.usd >= self.max_usd:
                raise BudgetExceeded(f"--max-usd {self.max_usd} reached (estimated ${self.usd:.4f})")
            if self.max_requests is not None and self.requests >= self.max_requests:
                raise BudgetExceeded(f"--max-requests {self.max_requests} reached")

    def add(self, usd: float, requests: int) -> None:
        with self.lock:
            self.usd += usd
            self.requests += requests


def _key(backend: str, model: str, seed: int, system, messages, tools, max_tokens, tool_choice) -> str:
    blob = json.dumps({"b": backend, "m": model, "s": seed, "sys": system, "msgs": [m.to_dict() for m in messages],
                       "tools": [(t.name, t.public_schema()) for t in tools], "mt": max_tokens, "tc": tool_choice}, sort_keys=True, default=str)
    # Ticket / refund ids are random per run (uuid in the MCP servers): normalise them or a replay would miss the cache after the first write.
    blob = re.sub(r"TCK-[A-Z0-9]{6}", "TCK-X", blob)
    blob = re.sub(r"RFN-[0-9a-f]{8}", "RFN-X", blob)
    return hashlib.sha256(blob.encode()).hexdigest()


class LiveLLM:
    """Adds caching, backoff, request/cost accounting and call logging around any provider adapter."""
    def __init__(self, inner, backend: str, seed: int, budget: Budget, cache_dir: Path, calls_log, ctx: threading.local,
                 sleep=time.sleep, max_attempts: int = 5, backoff_base: float = 2.0, backoff_cap: float = 60.0):
        self.inner, self.backend, self.seed, self.budget, self.cache_dir = inner, backend, seed, budget, cache_dir
        self.calls_log, self.ctx, self.sleep = calls_log, ctx, sleep
        self.max_attempts, self.backoff_base, self.backoff_cap = max_attempts, backoff_base, backoff_cap
        self.name = inner.name

    def _log(self, **kw) -> None:
        kw.update(ts=round(time.time(), 2), trial=getattr(self.ctx, "trial", None), model=self.name)
        self.calls_log(kw)

    def complete(self, system, messages, tools, max_tokens=1024, tool_choice=None) -> LLMResponse:
        key = _key(self.backend, self.name, self.seed, system, messages, tools, max_tokens, tool_choice)
        f = self.cache_dir / key[:2] / f"{key}.json"
        if f.exists():
            d = json.loads(f.read_text())
            with self.budget.lock:
                self.budget.cache_hits += 1
            self._log(cache="hit", in_tok=d["usage"][0], out_tok=d["usage"][1])
            return LLMResponse(d["text"], [ToolCall(c["id"], c["name"], c["arguments"]) for c in d["calls"]], Usage(*d["usage"]), d["model"], d["stop"])
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            self.budget.check()
            before = getattr(self.inner, "requests", None)
            t0 = time.time()
            try:
                resp = self.inner.complete(system, messages, tools, max_tokens, tool_choice)
            except TransientLLMError as e:
                last = e
                n = (getattr(self.inner, "requests", 0) - before) if before is not None else 1
                self.budget.add(0.0, max(1, n))
                with self.budget.lock:
                    self.budget.retries += 1
                cause = getattr(e, "__cause__", None)
                ra = None
                try:
                    ra = float(getattr(getattr(cause, "response", None), "headers", {}).get("retry-after"))
                except (TypeError, ValueError):
                    pass
                delay = min(self.backoff_cap, ra if ra is not None else self.backoff_base * (2 ** attempt)) * (1.0 if ra is not None else 0.75 + 0.5 * random.random())
                self._log(cache="miss", error=str(e)[:100], retry_in=round(delay, 2))
                self.sleep(delay)
                continue
            n = (getattr(self.inner, "requests", 0) - before) if before is not None else 1
            usd = cost_usd(resp.model, resp.usage)
            self.budget.add(usd, max(1, n))
            f.parent.mkdir(parents=True, exist_ok=True)
            # unique tmp name per process+thread: two workers (or two processes) writing the same key must not collide on one .tmp
            # (OneDrive/Windows raised PermissionError on the shared name); losing the replace race is harmless, the content is equivalent.
            tmp = f.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({"text": resp.text, "calls": [c.to_dict() for c in resp.tool_calls], "usage": [resp.usage.input_tokens, resp.usage.output_tokens],
                                       "model": resp.model, "stop": resp.stop_reason}))
            try:
                tmp.replace(f)
            except PermissionError:
                tmp.unlink(missing_ok=True)
                if not f.exists():
                    raise
            self._log(cache="miss", in_tok=resp.usage.input_tokens, out_tok=resp.usage.output_tokens, usd=round(usd, 6), latency_s=round(time.time() - t0, 2), http_requests=n)
            return resp
        raise last  # type: ignore[misc]


# ---------------------------------------------------------------------------------------------- providers
def make_inner(backend: str, model: str, seed: int, a, mock_url: str | None):
    if backend == "gemini":
        from supportagent.llm.gemini_client import GeminiLLM
        return GeminiLLM(model, api_key=None if not mock_url else "mock-key", base_url=mock_url, rpm=0 if mock_url else a.rpm, max_retries=a.client_retries,
                         backoff_base=a.backoff_base, extra_headers={"x-sa-seed": str(seed)} if mock_url else None, thinking=a.gemini_thinking)
    import anthropic
    from supportagent.llm.anthropic_client import AnthropicLLM
    client = anthropic.Anthropic(api_key="mock-key" if mock_url else os.environ["ANTHROPIC_API_KEY"], base_url=mock_url, max_retries=0, timeout=a.timeout,
                                 default_headers={"x-sa-seed": str(seed)} if mock_url else None)
    return AnthropicLLM(model, client=client)


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
            pass                                   # a torn last line from a crash: ignore, the trial will be re-run
    return out


def default_models(backend: str, a) -> dict:
    if backend == "gemini":
        ef, es = os.environ.get("SA_MODEL_FAST", ""), os.environ.get("SA_MODEL_STRONG", "")
        fast = a.fast_model or (ef if ef.startswith("gemini") else "gemini-3.5-flash-lite")
        strong = a.strong_model or (es if es.startswith("gemini") else "gemini-3.5-flash")
    else:
        fast, strong = a.fast_model or MODEL_TIERS["fast"], a.strong_model or MODEL_TIERS["strong"]
    return {"fast": fast, "strong": strong, "verifier": strong}


def ordered_scenarios(sets: list[str], limit: int | None, only: list[str] | None, trials: int) -> list[tuple[str, dict]]:
    """Category-interleaved order (and round-robin across sets when several are requested), so a run that stops early still covers
    every category and both sets rather than only the first files."""
    per_set: list[list[tuple[str, dict]]] = []
    for st in sets:
        scns = load_scenarios(only, heldout=(st == "heldout"))
        by_cat: dict[str, list[dict]] = {}
        for s in scns:
            by_cat.setdefault(s["category"], []).append(s)
        inter, i = [], 0
        while any(i < len(v) for v in by_cat.values()):
            inter += [(st, v[i]) for v in by_cat.values() if i < len(v)]
            i += 1
        per_set.append(inter)
    out: list[tuple[str, dict]] = []
    for k in range(max((len(x) for x in per_set), default=0)):
        out += [x[k] for x in per_set if k < len(x)]
    return out[:limit] if limit else out


def run_live(a) -> dict:
    sets = ["main", "heldout"] if a.set == "both" else [a.set]
    backend = a.backend
    mock, mock_url = None, None
    if a.dry_run:
        from .mock_llm_server import MockLLMServer
        faults = [{"every": int(x.split(":")[0]), "status": int(x.split(":")[1]), "retry_after": "0"} for x in (a.mock_fault or [])]
        mock = MockLLMServer(faults).start()
        mock_url = mock.url
    elif backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set: live Anthropic run is PENDING (use --dry-run to validate the pipeline offline).")
        return {"status": "pending_no_key"}
    elif backend == "gemini":
        from supportagent.llm.gemini_client import load_api_key
        if not load_api_key():
            print("GEMINI_API_KEY is not set: live Gemini run is PENDING.")
            return {"status": "pending_no_key"}
    models = default_models(backend, a)
    run_dir = Path(a.run_dir) if a.run_dir else LIVE_DIR / (("dryrun_" if a.dry_run else "") + f"{backend}_{'+'.join(sets)}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = LIVE_DIR / "cache" / (("mock_" if a.dry_run else "") + backend)
    budget = Budget(a.max_usd, a.max_requests)
    trials_log, calls_log, errors_log = (jsonl_writer(run_dir / n) for n in ("trials.jsonl", "llm_calls.jsonl", "errors.jsonl"))
    done = {(t["config_name"], t["set"], t["scenario_id"], t["seed"]) for t in read_jsonl(run_dir / "trials.jsonl")}
    ctx = threading.local()
    seed_db = Path(__import__("tempfile").mkdtemp(prefix="sa-seed-")) / "seed.db"
    build_seed_db(seed_db)

    def maker(cfg, tier, seed):
        return LiveLLM(make_inner(backend, models.get(tier, models["strong"]), seed, a, mock_url), backend + ("-mock" if mock_url else ""), seed, budget, cache_dir,
                       calls_log, ctx, sleep=(lambda s: time.sleep(min(s, a.max_sleep))), backoff_base=a.backoff_base)

    cfgs = [PRESETS[n].with_(backend="anthropic" if backend == "anthropic" else "gemini", llm_backoff_s=a.backoff_base, timeout_s=a.turn_timeout_s,
                             max_cost_usd=max(PRESETS[n].max_cost_usd, 0.5)) for n in a.config]
    jobs = [(st, scn, seed, cfg) for st, scn in ordered_scenarios(sets, a.limit_scenarios, a.only, a.trials) for seed in range(a.trials) for cfg in cfgs]
    planned = len(jobs)
    todo = [j for j in jobs if (j[3].name, j[0], j[1]["id"], j[2]) not in done]
    stop = threading.Event()
    reason: list[str] = []
    counts = {"completed": 0, "errors": 0}
    lock = threading.Lock()

    def work(job):
        if stop.is_set():
            return
        st, scn, seed, cfg = job
        ctx.trial = f"{cfg.name}|{st}|{scn['id']}|{seed}"
        try:
            budget.check()
            t = run_trial(scn, cfg, seed, str(seed_db), llm_maker=maker)
        except BudgetExceeded as e:
            reason.append(str(e)); stop.set(); return
        except Exception as e:                      # QuotaExhausted, provider 4xx, persistent 5xx ...
            if type(e).__name__ == "QuotaExhausted":
                reason.append(f"quota exhausted: {e}"); stop.set(); return
            errors_log({"config_name": cfg.name, "set": st, "scenario_id": scn["id"], "seed": seed, "error": f"{type(e).__name__}: {str(e)[:300]}"})
            with lock:
                counts["errors"] += 1
            return
        sim = [m for m in t.get("models", []) if str(m).startswith("scripted")]
        if sim and not a.dry_run:                    # a live trial must never be answered (even partly) by the simulator
            errors_log({"config_name": cfg.name, "set": st, "scenario_id": scn["id"], "seed": seed, "error": f"simulated model in a live trial: {sim}"})
            with lock:
                counts["errors"] += 1
            return
        t.update(config_name=cfg.name, set=st, model=f"{backend}:{models['fast']}/{models['strong']}")
        trials_log(t)
        with lock:
            counts["completed"] += 1

    t0 = time.time()
    try:
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(work, todo))
    except KeyboardInterrupt:
        stop.set(); reason.append("interrupted")
    meta = {"backend": backend, "dry_run": a.dry_run, "models": models, "sets": sets, "configs": a.config, "trials_per_scenario": a.trials, "planned_scenario_trials": planned,
            "already_done_before_this_run": planned - len(todo), "completed_this_run": counts["completed"], "errored_this_run": counts["errors"],
            "stopped_early": reason[:1], "network_requests_this_run": budget.requests, "cache_hits_this_run": budget.cache_hits, "retries_this_run": budget.retries,
            "estimated_usd_this_run": round(budget.usd, 5), "pricing_note": "estimated from PLACEHOLDER per-token prices in config.PRICING",
            "wall_seconds": round(time.time() - t0, 1), "mock_requests": mock.requests if mock else None, "mock_faults_sent": mock.errors_sent if mock else None,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    if mock:
        mock.stop()
    print(json.dumps(meta, indent=1))
    return {"status": "ok", "meta": meta, "run_dir": str(run_dir)}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="anthropic", choices=["anthropic", "gemini"])
    ap.add_argument("--set", default="main", choices=["main", "heldout", "both"])
    ap.add_argument("--config", nargs="+", default=["full"], choices=list(PRESETS))
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--limit-scenarios", type=int, default=None, help="first N scenarios in category-interleaved order")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-usd", type=float, default=None, help="stop when the ESTIMATED spend reaches this (placeholder prices)")
    ap.add_argument("--max-requests", type=int, default=None, help="stop after this many network requests (cache hits are free)")
    ap.add_argument("--rpm", type=float, default=0.0, help="requests/minute pacing (Gemini free tier: 15)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--dry-run", action="store_true", help="answer through a local mock HTTP server backed by ScriptedLLM (no key, no cost)")
    ap.add_argument("--mock-fault", nargs="*", help="e.g. 7:429 11:529  (every 7th request -> 429)")
    ap.add_argument("--fast-model"), ap.add_argument("--strong-model")
    ap.add_argument("--gemini-thinking", default=None, help="e.g. minimal | low | budget:0")
    ap.add_argument("--backoff-base", type=float, default=2.0)
    ap.add_argument("--max-sleep", type=float, default=120.0)
    ap.add_argument("--client-retries", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--turn-timeout-s", type=float, default=900.0)
    return ap


def main() -> None:
    a = build_parser().parse_args()
    r = run_live(a)
    sys.exit(0 if r["status"] in ("ok", "pending_no_key") else 1)


if __name__ == "__main__":
    main()
