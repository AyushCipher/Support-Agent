"""TTL cache for idempotent READ tools, and the bounded parallel fetcher (ADR 0014).

Cache contract
  key        = (tool, canonical args, customer). The customer is part of the key AND of invalidation, so one customer's data can
               never be served to another and a write for customer X drops exactly X's entries.
  what       = only tools whose ToolSpec.risk == 'read' on first-party servers. Unannotated third-party tools default to the risky
               tier (mcp/client.build_registry) and are therefore never cached. Errors are never cached.
  stale data = ANY write attempt for the customer (success or failure: a failed write may still have partially applied) invalidates
               that customer's entries immediately, in this process, whichever session performed it. Writes made by OTHER processes
               are only bounded by the TTL. The cache is advisory: ToolGuard's policy verification and the verifier fetch ground
               truth with direct registry calls that bypass it, so a stale entry can mislead the *model* but never a money decision.
  races      = a read that was IN FLIGHT when a write invalidated the customer must not repopulate the cache with pre-write data
               (two sessions of one customer in different worker threads). Each customer has a generation counter bumped by every
               invalidation; callers snapshot `generation()` before fetching and pass it to `put`, which drops the value if it moved.
  values     = stored as JSON text and re-parsed on read, so callers cannot mutate cached state.
"""
from __future__ import annotations

import concurrent.futures as cf
import contextvars
import json
import threading
import time
from typing import Any, Callable


class ToolCache:
    def __init__(self, ttl_s: float = 30.0, max_entries: int = 512, clock: Callable[[], float] = time.monotonic):
        self.ttl_s, self.max_entries, self.clock = ttl_s, max_entries, clock
        self._d: dict[tuple, tuple[float, str]] = {}
        self._gen: dict[str, int] = {}
        self._lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "puts": 0, "invalidations": 0, "evictions": 0, "stale_puts_dropped": 0}

    @property
    def enabled(self) -> bool:
        return self.ttl_s > 0

    @staticmethod
    def key(tool: str, args: dict, customer: str) -> tuple:
        return (tool, json.dumps(args, sort_keys=True, default=str), customer)

    def get(self, tool: str, args: dict, customer: str) -> Any | None:
        if not self.enabled:
            return None
        k = self.key(tool, args, customer)
        with self._lock:
            hit = self._d.get(k)
            if hit and self.clock() - hit[0] <= self.ttl_s:
                self.stats["hits"] += 1
                return json.loads(hit[1])
            if hit:
                del self._d[k]
            self.stats["misses"] += 1
        return None

    def generation(self, customer: str) -> int:
        with self._lock:
            return self._gen.get(customer, 0)

    def put(self, tool: str, args: dict, customer: str, data: Any, generation: int | None = None) -> None:
        """`generation`: the value of generation(customer) taken BEFORE the fetch started. If a write invalidated the customer since,
        the fetched value may predate that write and is dropped (review finding R3)."""
        if not self.enabled:
            return
        with self._lock:
            if generation is not None and self._gen.get(customer, 0) != generation:
                self.stats["stale_puts_dropped"] += 1
                return
            if len(self._d) >= self.max_entries:
                oldest = min(self._d, key=lambda x: self._d[x][0])
                del self._d[oldest]
                self.stats["evictions"] += 1
            self._d[self.key(tool, args, customer)] = (self.clock(), json.dumps(data, default=str))
            self.stats["puts"] += 1

    def invalidate_customer(self, customer: str) -> int:
        with self._lock:
            self._gen[customer] = self._gen.get(customer, 0) + 1
            doomed = [k for k in self._d if k[2] == customer]
            for k in doomed:
                del self._d[k]
            self.stats["invalidations"] += len(doomed)
        return len(doomed)

    def __len__(self) -> int:
        return len(self._d)


class ParallelFetcher:
    """Run independent read fetches concurrently with bounded concurrency, a per-call timeout and circuit-breaker integration.

    Results are returned in INPUT order regardless of completion order, so the transcript is deterministic. A call whose server's
    breaker is open fails fast without occupying a worker. A timed-out call is recorded as a breaker failure; its thread cannot be
    killed (Python), so it may finish in the background - harmless for reads, which is exactly why only reads are parallelised.
    """
    def __init__(self, registry, max_workers: int = 4, timeout_s: float = 10.0):
        self.registry, self.max_workers, self.timeout_s = registry, max(1, max_workers), timeout_s

    def _breaker(self, tool: str):
        c = self.registry.clients.get(tool.split("__", 1)[0])
        return getattr(c, "breaker", None)

    def fetch_all(self, jobs: list[dict]) -> list[dict]:
        """jobs: [{'tool','args','role','session_id'}] -> [{'data','err','ms','timed_out'}] in the same order."""
        out: list[dict | None] = [None] * len(jobs)
        runnable: list[int] = []
        for i, j in enumerate(jobs):
            b = self._breaker(j["tool"])
            if b is not None and not b.allow():
                out[i] = {"data": {"error": "tool_unavailable", "message": f"{j['tool'].split('__', 1)[0]} circuit open"}, "err": True, "ms": 0.0, "timed_out": False}
            else:
                runnable.append(i)
        if not runnable:
            return out  # type: ignore[return-value]
        ex = cf.ThreadPoolExecutor(min(self.max_workers, len(runnable)), thread_name_prefix="tool")
        try:
            def run(j):
                t0 = time.perf_counter()
                data, err = self.registry.call(j["tool"], j["args"], role=j["role"], timeout=self.timeout_s, session_id=j["session_id"])
                return {"data": data, "err": bool(err), "ms": (time.perf_counter() - t0) * 1000, "timed_out": False}
            futs = {i: ex.submit(contextvars.copy_context().run, run, jobs[i]) for i in runnable}
            waves = -(-len(runnable) // self.max_workers)              # calls queued behind busy workers get their own timeout window
            deadline = time.monotonic() + self.timeout_s * waves
            for i, f in futs.items():
                try:
                    out[i] = f.result(timeout=max(0.0, deadline - time.monotonic()))
                except cf.TimeoutError:
                    b = self._breaker(jobs[i]["tool"])
                    if b is not None:
                        b.fail()
                    out[i] = {"data": {"error": "tool_unavailable", "message": f"{jobs[i]['tool']} timed out after {self.timeout_s}s"},
                              "err": True, "ms": self.timeout_s * 1000, "timed_out": True}
                except Exception as e:                                  # a crashing worker is a tool error, never a harness crash
                    out[i] = {"data": {"error": "tool_unavailable", "message": f"{type(e).__name__}"}, "err": True, "ms": 0.0, "timed_out": False}
        finally:
            ex.shutdown(wait=False)
        return out  # type: ignore[return-value]
