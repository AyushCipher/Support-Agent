"""SLOs, a rolling-window evaluator, multi-window burn-rate alerts, and generators for Prometheus rules + a Grafana dashboard (ADR 0015).

Two ways to run the SAME definitions:
  1. offline / cron:   `supportagent slo eval`   -> reads runs/traces.jsonl + runs/state.db, evaluates, notifies sinks (this module)
  2. Prometheus:       `render_prometheus_rules()` -> ops/prometheus_rules.yml, over the metrics the API exports on /metrics

Ratio SLOs (task success, latency, tool errors) use the Google SRE multi-window, multi-burn-rate method: an alert fires only when
the error-budget burn rate exceeds a factor over BOTH a long window (it is real) and a short window (it is still happening).
Threshold SLOs (guard-block rate, cost per resolution, approval backlog age, kill switch) are plain rolling-window ceilings.

HONESTY: the objectives below are illustrative defaults chosen for this repo, not values derived from real traffic. The 'task
success' SLI is a PROXY (turn ended in a healthy state, no safe-fallback), not a measurement of customer outcomes. Tune both from
real data before paging anyone.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..obs.metrics import LATENCY_TARGET_S, Sample, span_samples
from .alerts import Alert


# ---------------------------------------------------------------------------------------------------------------- definitions
@dataclass(frozen=True)
class BurnRule:
    long_s: float
    short_s: float
    factor: float
    severity: str


# SRE-workbook defaults: 14.4x over 1h/5m burns 2% of a 30-day budget (page); 6x over 6h/30m burns 5% (page); 1x over 3d/6h burns 10% (ticket).
DEFAULT_RULES = (BurnRule(3600, 300, 14.4, "page"), BurnRule(21600, 1800, 6.0, "page"), BurnRule(259200, 21600, 1.0, "ticket"))


@dataclass(frozen=True)
class SLO:
    name: str
    description: str
    kind: str                         # ratio | threshold
    objective: float = 0.0            # ratio: required good fraction
    good: str = ""                    # ratio predicate: turn_ok | turn_fast | tool_ok
    target: float = 0.0               # turn_fast: seconds; threshold: ceiling
    window_s: float = 3600.0          # threshold rolling window
    severity: str = "ticket"
    min_events: int = 20              # below this a window has no verdict (no data never pages)
    unit: str = ""


DEFAULT_SLOS: tuple[SLO, ...] = (
    SLO("task_success", "Turns that end in a healthy state without a safe-fallback (success PROXY)", "ratio", 0.95, "turn_ok", unit="ratio"),
    SLO("turn_latency", f"Turns completing within {LATENCY_TARGET_S:g}s (i.e. p95 <= {LATENCY_TARGET_S:g}s)", "ratio", 0.95, "turn_fast", LATENCY_TARGET_S, unit="s"),
    SLO("tool_error_rate", "Tool calls that do not fail with an infrastructure error (policy blocks are not errors)", "ratio", 0.99, "tool_ok", unit="ratio"),
    SLO("guard_block_rate", "Share of tool calls blocked by guardrails (anomaly: attack wave or a regressed prompt)", "threshold", 0, "", 0.15, 3600, "ticket", 20, "ratio"),
    SLO("cost_per_resolution", "Notional model USD per resolved turn", "threshold", 0, "", 0.02, 3600, "ticket", 20, "usd"),
    SLO("approval_backlog_age", "Age in seconds of the oldest pending human approval", "threshold", 0, "", 4 * 3600, 0, "page", 1, "s"),
    SLO("kill_switch_global", "Global kill switch is ON (agent deliberately off)", "threshold", 0, "", 0, 0, "page", 1, "bool"),
    SLO("kill_switch_money", "Money-actions kill switch is ON", "threshold", 0, "", 0, 0, "ticket", 1, "bool"),
)


# ---------------------------------------------------------------------------------------------------------------- math
def burn_rate(bad: int, total: int, objective: float) -> float | None:
    """burn = observed error rate / error budget. 1.0 consumes the budget exactly over the SLO period.
    None when there is no data. With a 100% objective (zero budget) any error is an infinite burn."""
    if total <= 0:
        return None
    err = bad / total
    budget = 1.0 - objective
    if budget <= 0:
        return math.inf if bad else 0.0
    return err / budget


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile, q in [0, 1]."""
    if not values:
        return None
    v = sorted(values)
    return v[max(0, min(len(v) - 1, math.ceil(q * len(v)) - 1))]


def _window(samples: list[Sample], kind: str, now: float, seconds: float) -> list[Sample]:
    lo = now - seconds
    return [s for s in samples if s.kind == kind and lo < s.ts <= now]


def _predicate(slo: SLO, s: Sample) -> bool | None:
    if slo.good == "turn_ok":
        return s.ok
    if slo.good == "turn_fast":
        return None if s.ok is None else s.value <= slo.target
    if slo.good == "tool_ok":
        return s.ok
    raise ValueError(slo.good)


def _counts(slo: SLO, win: list[Sample]) -> tuple[int, int]:
    verdicts = [v for v in (_predicate(slo, s) for s in win) if v is not None]
    return sum(1 for v in verdicts if not v), len(verdicts)


@dataclass
class SLOStatus:
    slo: SLO
    ok: bool
    value: float | None                      # ratio: compliance over the longest window; threshold: current value
    windows: dict = field(default_factory=dict)
    firing: list[Alert] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.slo.name, "description": self.slo.description, "kind": self.slo.kind, "objective": self.slo.objective or None,
                "ok": self.ok, "value": self.value, "windows": self.windows, "firing": [a.key for a in self.firing], **self.extra}


class SLOEvaluator:
    def __init__(self, slos: tuple[SLO, ...] = DEFAULT_SLOS, rules: tuple[BurnRule, ...] = DEFAULT_RULES):
        self.slos, self.rules = slos, rules

    def evaluate(self, samples: list[Sample], gauges: dict | None = None, now: float | None = None) -> list[SLOStatus]:
        now, gauges = time.time() if now is None else now, gauges or {}
        return [self._ratio(s, samples, now) if s.kind == "ratio" else self._threshold(s, samples, gauges, now) for s in self.slos]

    def firing(self, statuses: list[SLOStatus]) -> list[Alert]:
        return [a for st in statuses for a in st.firing]

    # ---- ratio SLOs: multi-window burn rate
    def _ratio(self, slo: SLO, samples: list[Sample], now: float) -> SLOStatus:
        kind = "tool" if slo.good == "tool_ok" else "turn"
        cache: dict[float, tuple[int, int]] = {}

        def counts(seconds: float) -> tuple[int, int]:
            if seconds not in cache:
                cache[seconds] = _counts(slo, _window(samples, kind, now, seconds))
            return cache[seconds]

        def burn(seconds: float) -> float | None:
            bad, total = counts(seconds)
            return burn_rate(bad, total, slo.objective) if total >= slo.min_events else None

        windows, alerts = {}, []
        for secs in sorted({w for r in self.rules for w in (r.long_s, r.short_s)}):
            bad, total = counts(secs)
            windows[f"{secs:g}s"] = {"events": total, "bad": bad, "error_rate": (bad / total) if total else None, "burn_rate": burn(secs)}
        for r in self.rules:
            bl, bs = burn(r.long_s), burn(r.short_s)
            if bl is not None and bs is not None and bl >= r.factor and bs >= r.factor:
                alerts.append(Alert(f"{slo.name}:burn:{r.factor:g}", slo.name, r.severity,
                                    f"{slo.name}: error budget burning {bl:.1f}x (long {r.long_s / 3600:g}h) and {bs:.1f}x (short {r.short_s / 60:g}m); "
                                    f"threshold {r.factor:g}x of a {slo.objective:.1%} objective",
                                    details={"long_burn": bl, "short_burn": bs, "factor": r.factor, "objective": slo.objective}, ts=now))
        longest = max(r.long_s for r in self.rules)
        bad, total = counts(longest)
        extra = {}
        if slo.good == "turn_fast":
            lat = [s.value for s in _window(samples, "turn", now, 3600) if s.ok is not None]
            extra["p95_s_1h"] = percentile(lat, 0.95)
            extra["p50_s_1h"] = percentile(lat, 0.50)
        return SLOStatus(slo, not alerts, (1 - bad / total) if total else None, windows, alerts, extra)

    # ---- threshold SLOs
    def _threshold(self, slo: SLO, samples: list[Sample], gauges: dict, now: float) -> SLOStatus:
        value: float | None = None
        if slo.name == "guard_block_rate":
            win = _window(samples, "tool", now, slo.window_s)
            value = (sum(1 for s in win if s.labels.get("outcome") == "blocked") / len(win)) if len(win) >= slo.min_events else None
        elif slo.name == "cost_per_resolution":
            win = _window(samples, "turn", now, slo.window_s)
            resolved = sum(1 for s in win if s.resolved)
            value = (sum(s.cost for s in win) / resolved) if resolved >= slo.min_events else None
        elif slo.name == "approval_backlog_age":
            value = gauges.get("approval_oldest_pending_s")
        elif slo.name == "kill_switch_global":
            value = float(gauges["kill_switch_global"]) if "kill_switch_global" in gauges else None
        elif slo.name == "kill_switch_money":
            value = float(gauges["kill_switch_money"]) if "kill_switch_money" in gauges else None
        breach = value is not None and value > slo.target
        alerts = [Alert(f"{slo.name}:threshold", slo.name, slo.severity,
                        f"{slo.name} = {value:.4g}{slo.unit if slo.unit in ('s',) else ''} exceeds {slo.target:g}: {slo.description}",
                        details={"value": value, "threshold": slo.target}, ts=now)] if breach else []
        return SLOStatus(slo, not breach, value, {}, alerts)


# ---------------------------------------------------------------------------------------------------------------- data sources
def load_samples(trace_path: str | Path, now: float | None = None, lookback_s: float = 3 * 86400 + 3600) -> list[Sample]:
    """Read runs/traces.jsonl (JsonlSink output) into samples, skipping unparsable lines and spans older than `lookback_s`."""
    now = time.time() if now is None else now
    p, out = Path(trace_path), []
    if not p.exists():
        return out
    with p.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                span = json.loads(line)
            except ValueError:
                continue
            if float(span.get("end") or 0) >= now - lookback_s:
                out += span_samples(span)
    return out


def load_gauges(conn: sqlite3.Connection, now: float | None = None) -> dict:
    """Point-in-time gauges from the state DB: oldest pending approval age and kill-switch state."""
    now = time.time() if now is None else now
    g: dict = {}
    try:
        r = conn.execute("SELECT MIN(created_at) FROM approvals WHERE status='pending'").fetchone()
        g["approval_oldest_pending_s"] = max(0.0, now - r[0]) if r and r[0] else 0.0
    except sqlite3.Error:
        pass
    try:
        from .flags import FlagStore
        f = FlagStore(conn)
        g["kill_switch_global"], g["kill_switch_money"] = int(f.killed("global")), int(f.get("kill_switch.money", False) is True)
    except sqlite3.Error:
        pass
    return g


# ---------------------------------------------------------------------------------------------------------------- Prometheus + Grafana
def _dur(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 86400}d" if s % 86400 == 0 else f"{s // 3600}h" if s % 3600 == 0 else f"{s // 60}m"


def _err_expr(slo: SLO, w: str) -> str:
    if slo.good == "turn_ok":
        return (f'sum(rate(supportagent_turns_total{{outcome="bad"}}[{w}])) / sum(rate(supportagent_turns_total{{outcome=~"good|bad"}}[{w}]))')
    if slo.good == "turn_fast":
        return (f'1 - (sum(rate(supportagent_turn_duration_seconds_bucket{{le="{LATENCY_TARGET_S:g}"}}[{w}])) '
                f'/ sum(rate(supportagent_turn_duration_seconds_count[{w}])))')
    return f'sum(rate(supportagent_tool_calls_total{{outcome="error"}}[{w}])) / sum(rate(supportagent_tool_calls_total[{w}]))'


_THRESHOLD_EXPR = {
    "guard_block_rate": 'sum(rate(supportagent_tool_calls_total{outcome="blocked"}[1h])) / sum(rate(supportagent_tool_calls_total[1h]))',
    "cost_per_resolution": "sum(rate(supportagent_cost_usd_total[1h])) / sum(rate(supportagent_resolutions_total[1h]))",
    "approval_backlog_age": "supportagent_approvals_oldest_pending_age_seconds",
    "kill_switch_global": "supportagent_kill_switch_global",
    "kill_switch_money": "supportagent_kill_switch_money",
}


def render_prometheus_rules(slos: tuple[SLO, ...] = DEFAULT_SLOS, rules: tuple[BurnRule, ...] = DEFAULT_RULES) -> str:
    wins = sorted({w for r in rules for w in (r.long_s, r.short_s)})
    o = ["# GENERATED by `python -m supportagent.ops.slo --rules` from ops/slo.py. Do not edit by hand (a test fails on drift).",
         "# Not validated with promtool in this repo's CI; run `promtool check rules` before loading.", "groups:",
         "  - name: supportagent-sli-recording", "    rules:"]
    ratio = [s for s in slos if s.kind == "ratio"]
    for s in ratio:
        for w in wins:
            o += [f"      - record: supportagent:sli_error_ratio:rate{_dur(w)}", f"        expr: {_err_expr(s, _dur(w))}", f"        labels: {{slo: {s.name}}}"]
    o += ["      - record: supportagent:turn_latency_p95_seconds:5m",
          "        expr: histogram_quantile(0.95, sum by (le) (rate(supportagent_turn_duration_seconds_bucket[5m])))",
          "  - name: supportagent-slo-alerts", "    rules:"]
    for s in ratio:
        budget = 1 - s.objective
        for r in rules:
            thr = f"{r.factor:g} * {budget:.6g}"
            o += [f"      - alert: SupportAgent_{_camel(s.name)}_BudgetBurn_{_dur(r.long_s)}",
                  f"        expr: supportagent:sli_error_ratio:rate{_dur(r.long_s)}{{slo=\"{s.name}\"}} > ({thr}) and "
                  f"supportagent:sli_error_ratio:rate{_dur(r.short_s)}{{slo=\"{s.name}\"}} > ({thr})",
                  f"        labels: {{severity: {r.severity}, slo: {s.name}}}",
                  "        annotations:",
                  f"          summary: \"{s.name}: error budget burning >{r.factor:g}x ({_dur(r.long_s)} and {_dur(r.short_s)} windows)\"",
                  f"          runbook: \"docs/ops_runbook.md#alert-{s.name.replace('_', '-')}\""]
    for s in (x for x in slos if x.kind == "threshold"):
        expr = _THRESHOLD_EXPR[s.name]
        if s.name.startswith("kill_switch"):
            cond, hold = f"{expr} == 1", "0m"
        else:
            cond, hold = f"{expr} > {s.target:g}", ("0m" if s.name == "approval_backlog_age" else "15m")
        o += [f"      - alert: SupportAgent_{_camel(s.name)}", f"        expr: {cond}", f"        for: {hold}",
              f"        labels: {{severity: {s.severity}, slo: {s.name}}}", "        annotations:", f"          summary: \"{s.description}\"",
              f"          runbook: \"docs/ops_runbook.md#alert-{s.name.replace('_', '-')}\""]
    return "\n".join(o) + "\n"


def _camel(name: str) -> str:
    return "".join(p.capitalize() for p in name.split("_"))


def render_grafana_dashboard(slos: tuple[SLO, ...] = DEFAULT_SLOS) -> str:
    ds = {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}
    panels: list[dict] = []

    def panel(title, kind, targets, x, y, w=8, h=7, unit="short", thresholds=None):
        pid = len(panels) + 1
        fc = {"defaults": {"unit": unit}, "overrides": []}
        if thresholds:
            fc["defaults"]["thresholds"] = {"mode": "absolute", "steps": [{"color": "green", "value": None}] + [{"color": c, "value": v} for v, c in thresholds]}
        panels.append({"id": pid, "type": kind, "title": title, "datasource": ds, "gridPos": {"x": x, "y": y, "w": w, "h": h}, "fieldConfig": fc,
                       "targets": [{"refId": chr(65 + i), "expr": e, "legendFormat": lg, "datasource": ds} for i, (e, lg) in enumerate(targets)]})

    panel("Turn latency p95 (5m)", "stat", [("supportagent:turn_latency_p95_seconds:5m", "p95")], 0, 0, 6, 5, "s", [(LATENCY_TARGET_S, "red")])
    panel("Oldest pending approval", "stat", [("supportagent_approvals_oldest_pending_age_seconds", "age")], 6, 0, 6, 5, "s", [(3600, "orange"), (4 * 3600, "red")])
    panel("Kill switch (global / money)", "stat", [("supportagent_kill_switch_global", "global"), ("supportagent_kill_switch_money", "money")], 12, 0, 6, 5, "short", [(1, "red")])
    panel("Pending approvals", "stat", [("supportagent_approvals_pending", "pending")], 18, 0, 6, 5)
    y = 5
    for i, s in enumerate(x for x in slos if x.kind == "ratio"):
        panel(f"{s.name}: error ratio vs budget {1 - s.objective:.3g}", "timeseries",
              [(f'supportagent:sli_error_ratio:rate{w}{{slo="{s.name}"}}', f"{w} window") for w in ("5m", "1h", "6h")] +
              [(f"vector({1 - s.objective:.6g})", "budget")], (i % 3) * 8, y + (i // 3) * 7, 8, 7, "percentunit")
    y += 7
    panel("Turns by outcome", "timeseries", [('sum by (status) (rate(supportagent_turns_total[5m]))', "{{status}}")], 0, y, 8, 7, "ops")
    panel("Tool calls by outcome", "timeseries", [('sum by (outcome) (rate(supportagent_tool_calls_total[5m]))', "{{outcome}}")], 8, y, 8, 7, "ops")
    panel("Cost per resolution (1h, notional USD)", "timeseries", [(_THRESHOLD_EXPR["cost_per_resolution"], "usd/resolution")], 16, y, 8, 7, "currencyUSD")
    y += 7
    panel("Guard-block rate (1h)", "timeseries", [(_THRESHOLD_EXPR["guard_block_rate"], "blocked / calls")], 0, y, 8, 7, "percentunit")
    panel("Verifier verdicts", "timeseries", [('sum by (status) (rate(supportagent_verifier_calls_total[15m]))', "{{status}}")], 8, y, 8, 7, "ops")
    panel("Read-tool cache hits", "timeseries", [('rate(supportagent_tool_cache_hits_total[5m])', "hits/s")], 16, y, 8, 7, "ops")
    dash = {"__inputs": [{"name": "DS_PROMETHEUS", "label": "Prometheus", "type": "datasource", "pluginId": "prometheus", "pluginName": "Prometheus"}],
            "title": "supportagent - SLOs", "uid": "supportagent-slo", "schemaVersion": 39, "version": 1, "editable": True, "tags": ["supportagent", "slo"],
            "time": {"from": "now-6h", "to": "now"}, "refresh": "30s", "panels": panels, "templating": {"list": []}}
    return json.dumps(dash, indent=1) + "\n"


# ---------------------------------------------------------------------------------------------------------------- CLI helpers
def format_report(statuses: list[SLOStatus]) -> str:
    rows = []
    for st in statuses:
        v = "n/a" if st.value is None else f"{st.value:.4g}"
        flag = "OK   " if st.ok else "ALERT"
        rows.append(f"{flag} {st.slo.name:<22} value={v:<10} {st.slo.description}" + (f"  [{', '.join(a.key for a in st.firing)}]" if st.firing else ""))
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m supportagent.ops.slo")
    ap.add_argument("--rules", action="store_true", help="print Prometheus rules")
    ap.add_argument("--dashboard", action="store_true", help="print Grafana dashboard JSON")
    a = ap.parse_args(argv)
    print(render_prometheus_rules() if a.rules else render_grafana_dashboard(), end="")


if __name__ == "__main__":
    main()
