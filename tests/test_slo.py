"""SLOs, burn-rate math, multi-window alerting, alert de-duplication and sinks, Prometheus/Grafana artifacts (ADR 0015)."""
import json
import math
import re
import sqlite3
from pathlib import Path

import pytest

from supportagent.guardrails.approval import ApprovalService
from supportagent.llm.scripted import ScriptedLLM
from supportagent.obs.metrics import MetricsRegistry, MetricsSink, Sample, span_samples, turn_ok
from supportagent.obs.tracing import JsonlSink, MemorySink, Tracer
from supportagent.ops import slo
from supportagent.ops.alerts import Alert, AlertManager, FileSink, LogSink, WebhookSink
from supportagent.ops.flags import FlagStore

NOW = 1_000_000.0
H = 3600.0
OPS_DIR = Path(slo.__file__).parent


def turn(ts, ok=True, latency=0.5, cost=0.01, resolved=None):
    return Sample(ts, "turn", ok, latency, cost, ok if resolved is None else resolved, {"status": "done" if ok else "escalated_budget"})


def tool(ts, outcome="ok"):
    return Sample(ts, "tool", outcome != "error", 0.0, 0.0, False, {"outcome": outcome, "cached": False})


def spread(n, bad, end=NOW, span=H, mk=turn):
    """n samples evenly spread over the last `span` seconds ending at `end`; the first `bad` of them are failures."""
    return [mk(end - span * (i + 0.5) / n, ok=(i >= bad)) if mk is turn else mk(end - span * (i + 0.5) / n, "error" if i < bad else "ok") for i in range(n)]


# ------------------------------------------------------------------------------------------------------ burn-rate math
def test_burn_rate_definition():
    assert burn_close(slo.burn_rate(5, 100, 0.95), 1.0)             # 5% errors against a 5% budget = exactly on budget
    assert burn_close(slo.burn_rate(10, 100, 0.95), 2.0)
    assert burn_close(slo.burn_rate(72, 100, 0.95), 14.4)           # 72% errors burns a 5% budget at 14.4x
    assert slo.burn_rate(0, 100, 0.99) == 0.0
    assert slo.burn_rate(0, 0, 0.99) is None                        # no data is not "healthy" and not "burning"
    assert slo.burn_rate(1, 10, 1.0) == math.inf and slo.burn_rate(0, 10, 1.0) == 0.0     # a 100% objective has no budget


def burn_close(a, b):
    return a is not None and abs(a - b) < 1e-9


def test_percentile_nearest_rank():
    assert slo.percentile([], 0.95) is None and slo.percentile([5], 0.95) == 5
    assert slo.percentile(list(range(1, 101)), 0.95) == 95 and slo.percentile(list(range(1, 101)), 0.5) == 50


def status_of(statuses, name):
    return next(s for s in statuses if s.slo.name == name)


# ------------------------------------------------------------------------------------------------------ multi-window logic
def test_fast_burn_fires_only_when_long_and_short_windows_both_exceed():
    ev = slo.SLOEvaluator()
    # 80% of turns failing for the whole last hour -> burn 16x on 1h AND on 5m (> 14.4): page
    st = status_of(ev.evaluate(spread(2000, 1600), now=NOW), "task_success")
    keys = {a.key for a in st.firing}
    assert "task_success:burn:14.4" in keys and not st.ok
    assert all(a.severity == "page" for a in st.firing if a.key.endswith("14.4"))
    assert st.windows["3600s"]["burn_rate"] > 14.4 and st.windows["300s"]["burn_rate"] > 14.4


def test_old_incident_that_has_recovered_does_not_page():
    """Long window still shows the burn (an hour ago it was bad) but the short window is clean: the alert must NOT fire."""
    bad_then = [turn(NOW - 3000 + i, ok=False) for i in range(300)]
    good_now = [turn(NOW - 250 + i) for i in range(100)]
    st = status_of(slo.SLOEvaluator().evaluate(bad_then + good_now, now=NOW), "task_success")
    assert st.windows["3600s"]["burn_rate"] >= 14.4 and st.windows["300s"]["burn_rate"] == 0.0
    assert "task_success:burn:14.4" not in {a.key for a in st.firing}


def test_short_blip_does_not_trip_the_fast_page_rule():
    healthy = [turn(NOW - 3000 + i * 10) for i in range(300)]                          # an hour of good traffic
    blip = [turn(NOW - 60 + i * 0.2, ok=False) for i in range(300)]                            # one bad minute
    st = status_of(slo.SLOEvaluator().evaluate(healthy + blip, now=NOW), "task_success")
    assert st.windows["300s"]["burn_rate"] >= 14.4 and st.windows["3600s"]["burn_rate"] < 14.4
    assert "task_success:burn:14.4" not in {a.key for a in st.firing}                    # the fast page needs the 1h window too


def test_slow_burn_files_a_ticket_not_a_page():
    # 6% errors steadily for 3 days: burn 1.2x on 3d and 6h -> ticket-level only
    samples = spread(600, 36, span=3 * 86400)
    st = status_of(slo.SLOEvaluator().evaluate(samples, now=NOW), "task_success")
    assert {(a.key, a.severity) for a in st.firing} == {("task_success:burn:1", "ticket")}


def test_no_traffic_or_too_little_traffic_never_alerts():
    ev = slo.SLOEvaluator()
    assert not ev.firing(ev.evaluate([], now=NOW))
    tiny = [turn(NOW - 10, ok=False) for _ in range(5)]                                   # 100% errors but only 5 events (< min_events=20)
    assert not ev.firing(ev.evaluate(tiny, now=NOW))


def test_killed_turns_are_excluded_from_success_and_latency():
    assert turn_ok("killed", False) is None and turn_ok("done", True) is False and turn_ok("awaiting_approval", False) is True
    assert turn_ok("degraded_handoff", False) is False and turn_ok("escalated_budget", False) is False
    samples = [Sample(NOW - i, "turn", None, 99.0, 0.0, False, {"status": "killed"}) for i in range(100)]
    assert not slo.SLOEvaluator().firing(slo.SLOEvaluator().evaluate(samples, now=NOW))


def test_latency_slo_is_a_p95_target_expressed_as_a_ratio():
    fast = [turn(NOW - i * 10, latency=1.0) for i in range(100)]
    slow = [turn(NOW - i * 10 - 5, latency=slo.LATENCY_TARGET_S + 1) for i in range(100)]
    st = status_of(slo.SLOEvaluator().evaluate(fast, now=NOW), "turn_latency")
    assert st.ok and st.extra["p95_s_1h"] == 1.0
    st2 = status_of(slo.SLOEvaluator().evaluate(fast + slow, now=NOW), "turn_latency")
    assert not st2.ok and st2.extra["p95_s_1h"] > slo.LATENCY_TARGET_S                      # half the turns are over target: p95 is over too


def test_tool_error_rate_counts_infrastructure_errors_not_policy_blocks():
    blocked = [tool(NOW - i, "blocked") for i in range(200)]
    assert status_of(slo.SLOEvaluator().evaluate(blocked, now=NOW), "tool_error_rate").ok
    errors = spread(200, 40, mk=tool)                                                       # 20% errors vs a 1% budget
    assert not status_of(slo.SLOEvaluator().evaluate(errors, now=NOW), "tool_error_rate").ok


def test_guard_block_rate_threshold():
    mix = [tool(NOW - i, "blocked" if i % 4 == 0 else "ok") for i in range(100)]            # 25% blocked > 15%
    assert not status_of(slo.SLOEvaluator().evaluate(mix, now=NOW), "guard_block_rate").ok
    calm = [tool(NOW - i, "blocked" if i % 20 == 0 else "ok") for i in range(100)]            # 5%
    st = status_of(slo.SLOEvaluator().evaluate(calm, now=NOW), "guard_block_rate")
    assert st.ok and abs(st.value - 0.05) < 1e-9


def test_cost_per_resolution_threshold():
    cheap = [turn(NOW - i, cost=0.005) for i in range(50)]
    assert status_of(slo.SLOEvaluator().evaluate(cheap, now=NOW), "cost_per_resolution").ok
    # 50 resolved turns at $0.01 plus 50 unresolved at $0.03: (0.5 + 1.5) / 50 resolved = $0.04 per resolution > $0.02
    mixed = [turn(NOW - i, cost=0.01) for i in range(50)] + [turn(NOW - 100 - i, ok=False, cost=0.03) for i in range(50)]
    st = status_of(slo.SLOEvaluator().evaluate(mixed, now=NOW), "cost_per_resolution")
    assert not st.ok and abs(st.value - 0.04) < 1e-9


def test_backlog_age_and_kill_switch_use_gauges():
    ev = slo.SLOEvaluator()
    st = ev.evaluate([], {"approval_oldest_pending_s": 5 * 3600, "kill_switch_global": 1, "kill_switch_money": 1}, now=NOW)
    assert {a.key: a.severity for a in ev.firing(st)} == {"approval_backlog_age:threshold": "page", "kill_switch_global:threshold": "page",
                                                           "kill_switch_money:threshold": "ticket"}
    ok = ev.evaluate([], {"approval_oldest_pending_s": 600, "kill_switch_global": 0, "kill_switch_money": 0}, now=NOW)
    assert not ev.firing(ok)
    assert not ev.firing(ev.evaluate([], {}, now=NOW))                                      # unknown gauge = no verdict, never a false page


def test_gauges_from_state_db():
    conn = sqlite3.connect(":memory:", isolation_level=None)
    ap = ApprovalService(conn)
    rec = ap.request("S1", "orders__issue_refund", {"a": 1}, "over_auto_cap", "x")
    conn.execute("UPDATE approvals SET created_at=? WHERE id=?", (NOW - 7200, rec["id"]))
    FlagStore(conn).set_kill("money", True)
    g = slo.load_gauges(conn, now=NOW)
    assert abs(g["approval_oldest_pending_s"] - 7200) < 1 and g["kill_switch_money"] == 1 and g["kill_switch_global"] == 0
    ap.decide(rec["id"], True, "r")
    assert slo.load_gauges(conn, now=NOW)["approval_oldest_pending_s"] == 0.0


# ------------------------------------------------------------------------------------------------------ alert dedup + sinks
class Rec:
    name = "rec"

    def __init__(self):
        self.got = []

    def send(self, a):
        self.got.append((a.key, a.state, a.severity))


def firing(key="task_success:burn:14.4", sev="page"):
    return Alert(key, key.split(":")[0], sev, f"{key} is firing")


def test_alert_dedup_resolve_and_renotify():
    now, rec = [0.0], Rec()
    m = AlertManager([rec], renotify_s=100, clock=lambda: now[0])
    assert len(m.process([firing()])) == 1
    now[0] = 50
    assert m.process([firing()]) == [] and m.stats["suppressed"] == 1                       # same alert, inside the window: suppressed
    now[0] = 150
    assert len(m.process([firing()])) == 1                                                  # re-notify after renotify_s
    now[0] = 160
    out = m.process([])
    assert [(a.key, a.state) for a in out] == [("task_success:burn:14.4", "resolved")]       # one resolved message
    assert m.process([]) == []                                                              # and nothing more
    assert rec.got == [("task_success:burn:14.4", "firing", "page"), ("task_success:burn:14.4", "firing", "page"), ("task_success:burn:14.4", "resolved", "page")]


def test_severity_escalation_bypasses_dedup_and_distinct_alerts_are_independent():
    rec = Rec()
    m = AlertManager([rec], renotify_s=10_000, clock=lambda: 0.0)
    m.process([firing("a:threshold", "ticket"), firing("b:threshold", "page")])
    assert len(rec.got) == 2
    out = m.process([firing("a:threshold", "page"), firing("b:threshold", "page")])         # a escalated ticket -> page; b unchanged
    assert [(a.key, a.severity) for a in out] == [("a:threshold", "page")]


def test_dedup_state_survives_restart_via_state_file(tmp_path):
    rec, st = Rec(), tmp_path / "alert_state.json"
    AlertManager([rec], 1000, st, clock=lambda: 0.0).process([firing()])
    again = AlertManager([rec], 1000, st, clock=lambda: 10.0)                              # e.g. the next cron tick: a new process
    assert again.process([firing()]) == [] and len(rec.got) == 1
    assert [a.state for a in again.process([])] == ["resolved"]


def test_a_failing_sink_never_blocks_the_others_or_the_evaluator(tmp_path):
    class Boom:
        name = "boom"

        def send(self, a):
            raise OSError("down")
    rec, f = Rec(), FileSink(tmp_path / "alerts.jsonl")
    m = AlertManager([Boom(), rec, f, LogSink()], clock=lambda: 5.0)
    assert len(m.process([firing()])) == 1 and m.stats["sink_errors"] == 1 and rec.got
    line = json.loads((tmp_path / "alerts.jsonl").read_text().splitlines()[0])
    assert line["key"] == "task_success:burn:14.4" and line["state"] == "firing"


def test_webhook_sink_payload_and_injectable_transport():
    sent = []
    w = WebhookSink("https://hooks.example/x", post=lambda url, body, t: sent.append((url, json.loads(body), t)))
    w.send(firing())
    url, body, timeout = sent[0]
    assert url == "https://hooks.example/x" and "FIRING" in body["text"] and body["alert"]["severity"] == "page" and timeout == 3.0


# ------------------------------------------------------------------------------------------------------ metrics from real traces
def test_runtime_spans_become_samples_and_metrics_and_slo_data(make_rt, tmp_path):
    rt = make_rt(llm_factory=lambda t, s: ScriptedLLM("strong", s))
    reg = MetricsRegistry()
    jsonl = tmp_path / "traces.jsonl"
    rt.tracer.sinks += [MetricsSink(reg), JsonlSink(jsonl)]
    s = rt.new_session("C001", "alice@example.com")
    for text in ("Where is ORD-1006?", "Refund ORD-1001 please", "Refund ORD-1003 please"):
        rt.handle_user_message(s, text)
    text = reg.render()
    for name in ("supportagent_turns_total", "supportagent_turn_duration_seconds_bucket", "supportagent_tool_calls_total",
                 "supportagent_cost_usd_total", "supportagent_resolutions_total"):
        assert name in text, name
    assert 'supportagent_tool_calls_total{outcome="blocked"}' in text or 'outcome="ok"' in text      # outcomes are labelled
    samples = [x for sp in rt.sink.spans for x in span_samples(sp)]
    assert {x.kind for x in samples} == {"turn", "tool"} and sum(1 for x in samples if x.kind == "turn") == 3
    assert any(x.resolved for x in samples)
    loaded = slo.load_samples(jsonl, now=max(x.ts for x in samples) + 1)                              # same data, read back from the JSONL sink
    assert len(loaded) == len(samples)
    now = max(x.ts for x in samples)
    st = slo.SLOEvaluator().evaluate(loaded, {}, now=now + 1)
    assert status_of(st, "task_success").windows["3600s"]["events"] == 3


def test_load_samples_survives_garbage_and_old_spans(tmp_path):
    p = tmp_path / "t.jsonl"
    old = {"name": "agent.turn", "start": 1.0, "end": 2.0, "attrs": {"turn.status": "done"}}
    new = {"name": "agent.turn", "start": NOW - 1, "end": NOW, "attrs": {"turn.status": "done", "turn.cost_usd": 0.01}}
    p.write_text("not json\n" + json.dumps(old) + "\n" + json.dumps(new) + "\n{truncated", encoding="utf-8")
    got = slo.load_samples(p, now=NOW + 1)
    assert len(got) == 1 and got[0].ok is True
    assert slo.load_samples(tmp_path / "missing.jsonl") == []


def test_prometheus_text_format_is_well_formed():
    reg = MetricsRegistry()
    reg.inc("x_total", {"a": "b"}, help_="h")
    reg.observe("lat_seconds", 0.3, help_="l")
    out = reg.render()
    assert '# TYPE x_total counter' in out and 'x_total{a="b"} 1' in out
    assert 'lat_seconds_bucket{le="0.5"} 1' in out and 'lat_seconds_bucket{le="0.25"} 0' in out and 'lat_seconds_bucket{le="+Inf"} 1' in out
    assert "lat_seconds_count 1" in out


# ------------------------------------------------------------------------------------------------------ generated artifacts
def test_committed_prometheus_rules_and_dashboard_match_the_generator():
    assert (OPS_DIR / "prometheus_rules.yml").read_text(encoding="utf-8") == slo.render_prometheus_rules()
    assert (OPS_DIR / "grafana_dashboard.json").read_text(encoding="utf-8") == slo.render_grafana_dashboard()


def test_prometheus_rules_cover_every_slo_and_reference_only_exported_metrics():
    rules = slo.render_prometheus_rules()
    for s in slo.DEFAULT_SLOS:
        assert s.name in rules, s.name
    assert rules.count("(") == rules.count(")") and rules.count("{") == rules.count("}") and rules.count("[") == rules.count("]")
    for r in slo.DEFAULT_RULES:                                                                   # every (long, short) pair appears as an alert condition
        assert f"rate{slo._dur(r.long_s)}" in rules and f"rate{slo._dur(r.short_s)}" in rules
    exported = {"supportagent_turns_total", "supportagent_turn_duration_seconds_bucket", "supportagent_turn_duration_seconds_count",
                "supportagent_tool_calls_total", "supportagent_cost_usd_total", "supportagent_resolutions_total",
                "supportagent_approvals_oldest_pending_age_seconds", "supportagent_kill_switch_global", "supportagent_kill_switch_money",
                "supportagent_approvals_pending", "supportagent_verifier_calls_total", "supportagent_tool_cache_hits_total"}
    used = set(re.findall(r"supportagent_[a-z_]+", rules + slo.render_grafana_dashboard()))
    used = {u for u in used if not u.startswith("supportagent_sli") and u not in ("supportagent_slo",)}
    assert used <= exported, used - exported


def test_burn_thresholds_in_rules_are_factor_times_budget():
    rules = slo.render_prometheus_rules()
    assert "(14.4 * 0.05)" in rules and "(6 * 0.05)" in rules and "(14.4 * 0.01)" in rules       # task_success/latency 95%, tool errors 99%


def test_grafana_dashboard_is_importable_json():
    d = json.loads(slo.render_grafana_dashboard())
    assert d["__inputs"][0]["name"] == "DS_PROMETHEUS" and d["uid"] and d["panels"]
    ids = [p["id"] for p in d["panels"]]
    assert len(ids) == len(set(ids)) and all(p["targets"] and p["gridPos"]["w"] > 0 for p in d["panels"])
    assert all(t["expr"] for p in d["panels"] for t in p["targets"])


def test_rules_file_parses_as_yaml_when_pyyaml_is_available():
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load((OPS_DIR / "prometheus_rules.yml").read_text(encoding="utf-8"))
    assert {g["name"] for g in doc["groups"]} == {"supportagent-sli-recording", "supportagent-slo-alerts"}
    alerts = [r for g in doc["groups"] for r in g["rules"] if "alert" in r]
    assert len({a["alert"] for a in alerts}) == len(alerts) and all(a["labels"]["severity"] in ("page", "ticket") for a in alerts)


def test_slo_cli_eval_and_exit_code(tmp_path, monkeypatch, capsys):
    from supportagent import cli
    from supportagent.ops import cli as ops_cli
    monkeypatch.setattr(ops_cli, "RUNS_DIR", tmp_path)
    tr = tmp_path / "traces.jsonl"
    import time
    now = time.time()
    tr.write_text("\n".join(json.dumps({"name": "agent.turn", "start": now - 2 - i * 0.01, "end": now - 1 - i * 0.01, "attrs": {"turn.status": "escalated_budget"}})
                            for i in range(100)), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        cli.main(["slo", "eval", "--alerts-file", str(tmp_path / "a.jsonl")])
    assert e.value.code == 1                                                                      # alerts firing -> non-zero (cron/CI friendly)
    out = capsys.readouterr().out
    assert "ALERT" in out and "task_success" in out
    assert any(json.loads(line)["slo"] == "task_success" for line in (tmp_path / "a.jsonl").read_text().splitlines())
    with pytest.raises(SystemExit) as e2:
        cli.main(["slo", "eval", "--alerts-file", str(tmp_path / "a.jsonl")])                      # second tick: deduplicated
    assert len((tmp_path / "a.jsonl").read_text().splitlines()) == len({json.loads(x)["key"] for x in (tmp_path / "a.jsonl").read_text().splitlines()})
    assert e2.value.code == 1
