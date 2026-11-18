"""Alert model, pluggable sinks (log / webhook / file) and a de-duplicating AlertManager.

The evaluator (ops/slo.py) is stateless: every run reports everything that is firing *now*. The manager turns that into
notifications: one message when an alert starts firing, optional re-notification after `renotify_s`, an immediate message on a
severity escalation, and one 'resolved' message when it stops. State can be persisted to a small JSON file so a cron-driven
`supportagent slo eval` does not re-page on every tick. Sinks are best-effort: a failing sink never breaks evaluation or the
other sinks (failures are counted and logged).
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

log = logging.getLogger("supportagent.alerts")
SEVERITY_RANK = {"info": 0, "ticket": 1, "page": 2}


@dataclass
class Alert:
    key: str                         # stable identity for dedup, e.g. "task_success:burn:14.4"
    slo: str
    severity: str                    # info | ticket | page
    summary: str
    state: str = "firing"            # firing | resolved
    details: dict = field(default_factory=dict)
    ts: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class AlertSink(Protocol):
    def send(self, alert: Alert) -> None: ...


class LogSink:
    name = "log"

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or log

    def send(self, alert: Alert) -> None:
        lvl = logging.ERROR if alert.severity == "page" and alert.state == "firing" else logging.WARNING if alert.state == "firing" else logging.INFO
        self.logger.log(lvl, "ALERT %s [%s/%s] %s", alert.state.upper(), alert.severity, alert.key, alert.summary)


class FileSink:
    """Append-only JSONL: easy to tail, ship, or feed a ticketing integration."""
    name = "file"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(alert.to_dict(), default=str) + "\n")


class WebhookSink:
    """POST JSON (Slack-compatible `text` plus the full alert). `post` is injectable for tests; default is stdlib urllib."""
    name = "webhook"

    def __init__(self, url: str, timeout_s: float = 3.0, post: Callable[[str, bytes, float], None] | None = None):
        self.url, self.timeout_s, self._post = url, timeout_s, post or self._urllib_post

    @staticmethod
    def _urllib_post(url: str, body: bytes, timeout: float) -> None:
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:        # noqa: S310 - operator-configured URL
            r.read()

    def send(self, alert: Alert) -> None:
        icon = {"firing": "FIRING", "resolved": "RESOLVED"}[alert.state]
        body = json.dumps({"text": f"[{icon}] ({alert.severity}) {alert.summary}", "alert": alert.to_dict()}, default=str).encode()
        self._post(self.url, body, self.timeout_s)


class AlertManager:
    def __init__(self, sinks: list[AlertSink], renotify_s: float = 3600.0, state_path: str | Path | None = None,
                 clock: Callable[[], float] = time.time):
        self.sinks, self.renotify_s, self.clock = sinks, renotify_s, clock
        self.state_path = Path(state_path) if state_path else None
        self.active: dict[str, dict] = {}                  # key -> {"severity", "last_sent"}
        self.stats = {"sent": 0, "suppressed": 0, "sink_errors": 0, "undelivered": 0}
        if self.state_path and self.state_path.exists():
            try:
                self.active = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                self.active = {}

    def _emit(self, alert: Alert) -> bool:
        """-> True if at least one sink accepted the alert."""
        alert.ts = alert.ts or self.clock()
        delivered = False
        for s in self.sinks:
            try:
                s.send(alert)
                delivered = True
            except Exception as e:                           # a broken sink must not break the others or the evaluator
                self.stats["sink_errors"] += 1
                log.error("alert sink %s failed: %s: %s", getattr(s, "name", type(s).__name__), type(e).__name__, e)
        self.stats["sent" if delivered else "undelivered"] += 1
        return delivered

    def process(self, firing: list[Alert]) -> list[Alert]:
        """Reconcile what is firing now with what was already notified. Returns the notifications actually sent."""
        now, sent = self.clock(), []
        current = {a.key: a for a in firing}
        for key, a in current.items():
            prev = self.active.get(key)
            escalated = prev is not None and SEVERITY_RANK.get(a.severity, 0) > SEVERITY_RANK.get(prev["severity"], 0)
            if prev is None or escalated or now - prev["last_sent"] >= self.renotify_s:
                a.ts = now
                if self._emit(a):
                    sent.append(a)
                    self.active[key] = {"severity": a.severity, "last_sent": now}
                # else: NOT recorded as notified, so the next evaluation retries instead of suppressing a page that nobody received
                # for `renotify_s` (review finding R4: every sink down used to look exactly like a delivered page)
            else:
                self.stats["suppressed"] += 1
        for key in [k for k in self.active if k not in current]:
            prev = self.active[key]
            r = Alert(key, key.split(":", 1)[0], prev["severity"], f"{key} recovered", "resolved", {}, now)
            if self._emit(r):                                # keep it active until the resolution is actually delivered
                self.active.pop(key)
                sent.append(r)
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_name(self.state_path.name + ".tmp")
            tmp.write_text(json.dumps(self.active), encoding="utf-8")
            os.replace(tmp, self.state_path)                 # atomic: a crash mid-write cannot leave a truncated state file (R10)
        return sent
