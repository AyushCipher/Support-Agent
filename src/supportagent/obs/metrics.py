"""Metrics derived from trace spans: one signal definition shared by Prometheus (/metrics) and the offline SLO evaluator.

`span_samples(span)` is the single place that decides what a span means for reliability (was the turn a success? was a tool call
an infrastructure error or a deliberate policy block?). The in-process `MetricsSink` and `ops/slo.py` both use it, so what
Prometheus alerts on and what `supportagent slo eval` reports cannot drift apart.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

GOOD_STATUSES = ("done", "awaiting_approval", "awaiting_user")        # healthy flow (a pending approval is not a failure)
BAD_STATUSES = ("escalated_budget", "degraded_handoff")                # the agent gave up / could not reach a model
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
LATENCY_TARGET_S = 8.0                                                 # must be one of LATENCY_BUCKETS (Prometheus needs the le="8" bucket)


@dataclass
class Sample:
    ts: float
    kind: str                       # turn | tool | verifier
    ok: bool | None = None          # None = excluded from the ratio (e.g. operator kill switch)
    value: float = 0.0              # turn: latency seconds
    cost: float = 0.0
    resolved: bool = False
    labels: dict = field(default_factory=dict)


def turn_ok(status: str, safe_fallback: bool) -> bool | None:
    """Task-success PROXY (not ground truth: nothing here knows whether the customer was actually helped).
    good = reached a healthy state without the safe-fallback reply; bad = gave up / model unreachable / fallback; killed = excluded."""
    if status == "killed":
        return None
    if status in BAD_STATUSES or safe_fallback:
        return False
    return status in GOOD_STATUSES


def span_samples(span: dict) -> list[Sample]:
    a, name = span.get("attrs", {}), span.get("name")
    end = float(span.get("end") or span.get("start") or 0.0)
    if name == "agent.turn" and "turn.status" in a and not a.get("turn.dry_run"):
        status, fb = a["turn.status"], bool(a.get("turn.safe_fallback"))
        return [Sample(end, "turn", turn_ok(status, fb), max(0.0, end - float(span.get("start") or end)), float(a.get("turn.cost_usd", 0.0)),
                       status == "done" and not fb, {"status": status})]
    if name == "tool.call" and "tool.outcome" in a:
        return [Sample(end, "tool", a["tool.outcome"] != "error", 0.0, 0.0, False, {"outcome": a["tool.outcome"], "cached": bool(a.get("tool.cached"))})]
    if name == "verifier.call":
        return [Sample(end, "verifier", a.get("verifier.status") == "agree", float(a.get("verifier.latency_ms", 0.0)) / 1000.0,
                       float(a.get("verifier.cost_usd", 0.0)), False, {"status": a.get("verifier.status", "")})]
    return []


class MetricsRegistry:
    """Tiny thread-safe Prometheus text-format registry (counters, histograms, gauges). No dependencies."""
    def __init__(self):
        self._lock = threading.Lock()
        self.counters: dict[tuple, float] = {}
        self.hist: dict[tuple, dict] = {}
        self.help: dict[str, str] = {}

    @staticmethod
    def _lbl(labels: dict) -> str:
        return "{" + ",".join(f'{k}="{v}"' for k, v in sorted(labels.items())) + "}" if labels else ""

    def inc(self, name: str, labels: dict | None = None, by: float = 1.0, help_: str = "") -> None:
        with self._lock:
            k = (name, self._lbl(labels or {}))
            self.counters[k] = self.counters.get(k, 0.0) + by
            self.help.setdefault(name, help_)

    def observe(self, name: str, value: float, labels: dict | None = None, buckets=LATENCY_BUCKETS, help_: str = "") -> None:
        with self._lock:
            h = self.hist.setdefault((name, self._lbl(labels or {})), {"b": buckets, "counts": [0] * len(buckets), "sum": 0.0, "n": 0})
            for i, b in enumerate(h["b"]):
                if value <= b:
                    h["counts"][i] += 1
            h["sum"] += value
            h["n"] += 1
            self.help.setdefault(name, help_)

    def render(self) -> str:
        with self._lock:
            out: list[str] = []
            seen: set[str] = set()
            for (name, lbl), v in sorted(self.counters.items()):
                if name not in seen:
                    seen.add(name)
                    out += [f"# HELP {name} {self.help.get(name, '')}", f"# TYPE {name} counter"]
                out.append(f"{name}{lbl} {v:g}")
            for (name, lbl), h in sorted(self.hist.items()):
                if name not in seen:
                    seen.add(name)
                    out += [f"# HELP {name} {self.help.get(name, '')}", f"# TYPE {name} histogram"]
                inner = lbl[1:-1] + "," if lbl else ""
                for b, c in zip(h["b"], h["counts"]):
                    out.append(f'{name}_bucket{{{inner}le="{b:g}"}} {c}')
                out.append(f'{name}_bucket{{{inner}le="+Inf"}} {h["n"]}')
                out += [f"{name}_sum{lbl} {h['sum']:.6f}", f"{name}_count{lbl} {h['n']}"]
        return "\n".join(out) + "\n"


class MetricsSink:
    """Tracer sink: turns finished spans into Prometheus metrics (same classification as the offline SLO evaluator)."""
    def __init__(self, registry: MetricsRegistry | None = None):
        self.reg = registry or MetricsRegistry()

    def emit(self, span: dict) -> None:
        for s in span_samples(span):
            if s.kind == "turn":
                outcome = "excluded" if s.ok is None else ("good" if s.ok else "bad")
                self.reg.inc("supportagent_turns_total", {"status": s.labels["status"], "outcome": outcome}, help_="Agent turns by final status and SLO outcome")
                if s.ok is not None:                                  # kill-switch turns are excluded from latency SLIs too
                    self.reg.observe("supportagent_turn_duration_seconds", s.value, help_="Harness wall time per agent turn")
                self.reg.inc("supportagent_cost_usd_total", by=s.cost, help_="Notional model cost (USD) across turns")
                if s.resolved:
                    self.reg.inc("supportagent_resolutions_total", help_="Turns that ended in a verified reply (done, no safe fallback)")
            elif s.kind == "tool":
                self.reg.inc("supportagent_tool_calls_total", {"outcome": s.labels["outcome"]}, help_="Tool calls by outcome (error = infrastructure failure; blocked = policy)")
                if s.labels.get("cached"):
                    self.reg.inc("supportagent_tool_cache_hits_total", help_="Read-tool calls served from the TTL cache")
            elif s.kind == "verifier":
                self.reg.inc("supportagent_verifier_calls_total", {"status": s.labels["status"]}, help_="Independent pre-money verifier verdicts")
