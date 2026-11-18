"""`supportagent retention ...` and `supportagent slo ...` subcommands (registered from cli.py)."""
from __future__ import annotations

import json
import sqlite3
import sys

from ..config import RUNS_DIR

OPS_COMMANDS = ("retention", "slo")


def register(sub) -> None:
    r = sub.add_parser("retention", help="data retention: show policy / purge+archive expired data (supports --dry-run)")
    r.add_argument("action", choices=["run", "show"])
    r.add_argument("--dry-run", action="store_true", help="report what would be purged/archived; change nothing")
    r.add_argument("--policy", default=None, help="JSON policy file (default: built-in, or $SA_RETENTION_POLICY)")
    r.add_argument("--json", action="store_true")
    s = sub.add_parser("slo", help="SLO evaluation over traces + state, alert sinks, Prometheus rules, Grafana dashboard")
    s.add_argument("action", choices=["eval", "rules", "dashboard"])
    s.add_argument("--traces", default=None, help="traces JSONL (default runs/traces.jsonl)")
    s.add_argument("--webhook", default=None, help="POST alerts to this URL")
    s.add_argument("--alerts-file", default=None, help="append alerts to this JSONL file")
    s.add_argument("--no-notify", action="store_true", help="evaluate and print only; do not touch alert state or sinks")
    s.add_argument("--json", action="store_true")


def _state() -> sqlite3.Connection:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(RUNS_DIR / "state.db", isolation_level=None)


def run(a) -> int:
    if a.cmd == "retention":
        from .retention import RetentionJob, load_policy, report_text
        policy = load_policy(a.policy)
        if a.action == "show":
            print(json.dumps(policy, indent=1))
            return 0
        rep = RetentionJob(_state(), policy, RUNS_DIR).run(dry_run=a.dry_run)
        print(json.dumps(rep, indent=1) if a.json else report_text(rep))
        return 0
    from . import slo
    if a.action == "rules":
        sys.stdout.write(slo.render_prometheus_rules())
        return 0
    if a.action == "dashboard":
        sys.stdout.write(slo.render_grafana_dashboard())
        return 0
    from .alerts import AlertManager, FileSink, LogSink, WebhookSink
    now_conn = _state()
    ev = slo.SLOEvaluator()
    statuses = ev.evaluate(slo.load_samples(a.traces or RUNS_DIR / "traces.jsonl"), slo.load_gauges(now_conn))
    print(json.dumps([s.to_dict() for s in statuses], indent=1, default=str) if a.json else slo.format_report(statuses))
    if not a.no_notify:
        sinks = [LogSink()] + ([FileSink(a.alerts_file)] if a.alerts_file else []) + ([WebhookSink(a.webhook)] if a.webhook else [])
        sent = AlertManager(sinks, state_path=RUNS_DIR / "alert_state.json").process(ev.firing(statuses))
        print(f"notifications sent: {len(sent)}")
    return 1 if any(not s.ok for s in statuses) else 0
