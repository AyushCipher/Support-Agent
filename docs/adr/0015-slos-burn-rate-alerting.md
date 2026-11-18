# ADR-0015: SLOs with multi-window burn-rate alerts and one signal definition

**Status:** accepted. Written retroactively during the review; the code predates it.

**Context.** Alerts should page on real, ongoing error-budget burn and not on blips, and the same definitions must work without running Prometheus in development.

**Decision.** `obs/metrics.span_samples` is the single definition of what a span means (turn good / bad / excluded; tool infrastructure error vs policy block). Both the in-process `/metrics` exporter and the offline evaluator use it. Ratio SLOs (task-success proxy, turn latency as "fraction under the target", tool errors) use the SRE-workbook multi-window rule: fire only when the burn rate exceeds the factor on BOTH a long and a short window (14.4x 1h/5m page, 6x 6h/30m page, 1x 3d/6h ticket), with a minimum event count per window so no data never pages. Threshold SLOs cover guard-block rate, cost per resolution, approval backlog age and kill switches. `ops/slo.py` generates `prometheus_rules.yml` and a Grafana dashboard; a test fails on drift, and another checks every alert's runbook link resolves (finding R11). `ops/alerts.py` de-duplicates notifications with persisted state; an alert counts as notified only when a sink accepted it (finding R4).

**Consequences.**
- (+) Offline `supportagent slo eval` and Prometheus agree by construction.
- (-) Objectives are illustrative, not derived from traffic; `task_success` is a proxy, not a customer outcome.
- (-) The rules file is not validated by `promtool` in CI.
- (-) The offline evaluator fires at `>=` the factor while PromQL uses `>`; this only differs at exact equality.
- (-) An SLO whose windows drop below `min_events` stops firing and is reported resolved; low-traffic services see resolve/fire flapping.
