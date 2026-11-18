"""Tracing: lightweight spans with OpenTelemetry GenAI-style attributes. Sinks: memory (evals), JSONL (local), OTLP (optional)."""
from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
_stack: contextvars.ContextVar[tuple] = contextvars.ContextVar("span_stack", default=())


class MemorySink:
    def __init__(self):
        self.spans: list[dict] = []

    def emit(self, span: dict) -> None:
        self.spans.append(span)


class JsonlSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, span: dict) -> None:
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(span, default=str) + "\n")


class OTelSink:
    """Exports finished spans to an OTLP endpoint (Jaeger/Tempo/Honeycomb). Needs `pip install supportagent[otel]`."""
    def __init__(self, service: str = "supportagent", endpoint: str | None = None):
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        provider = TracerProvider(resource=Resource.create({"service.name": service}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        self._tracer = provider.get_tracer("supportagent")
        self._trace = trace

    def emit(self, span: dict) -> None:
        sp = self._tracer.start_span(span["name"], start_time=int(span["start"] * 1e9))
        for k, v in span["attrs"].items():
            sp.set_attribute(k, v if isinstance(v, (str, int, float, bool)) else json.dumps(v, default=str))
        sp.set_attribute("session.id", span["trace_id"])
        sp.end(end_time=int(span["end"] * 1e9))


class Tracer:
    def __init__(self, sinks: list | None = None):
        self.sinks = sinks if sinks is not None else []

    @contextmanager
    def trace(self, trace_id: str, name: str = "session.turn", **attrs):
        tok = _trace_id.set(trace_id)
        try:
            with self.span(name, **attrs) as sp:
                yield sp
        finally:
            _trace_id.reset(tok)

    @contextmanager
    def span(self, name: str, **attrs):
        sid = uuid.uuid4().hex[:12]
        stack = _stack.get()
        sp = {"trace_id": _trace_id.get() or "none", "span_id": sid, "parent_id": stack[-1] if stack else None,
              "name": name, "start": time.time(), "end": None, "status": "ok", "attrs": dict(attrs)}
        tok = _stack.set(stack + (sid,))
        try:
            yield sp
        except Exception as e:
            sp["status"] = "error"
            sp["attrs"]["error"] = f"{type(e).__name__}: {e}"[:200]
            raise
        finally:
            _stack.reset(tok)
            sp["end"] = time.time()
            for s in self.sinks:
                try:
                    s.emit(sp)
                except Exception:
                    pass


def erase_spans(path: str | Path, session_ids: set[str] | list[str], customer_id: str = "") -> int:
    """Delete-on-request for the local trace file (security gap G3: traces were outside erasure, kept until age-based retention).

    Drops every span whose trace_id is one of the customer's sessions (or its shadow `<sid>~shadow`) or that carries
    attrs['customer.id'] == customer_id. No archive is written (an erasure must not leave a copy). Unparsable lines are kept.
    Spans appended concurrently are preserved by copying the tail written after the snapshot; a span written between that copy and
    os.replace can still be lost (same residual race as retention, documented in ops_runbook.md). Returns the number of spans removed.
    OTLP backends (Jaeger/Tempo) keep their own copy: purge those by trace id with the backend's own tooling."""
    import os
    p = Path(path)
    if not p.exists():
        return 0
    sids = {str(x) for x in session_ids}
    sids |= {f"{x}~shadow" for x in sids}
    size = p.stat().st_size
    with p.open("rb") as f:
        data = f.read(size)
    keep, removed = [], 0
    for raw in data.splitlines(keepends=True):
        try:
            sp = json.loads(raw)
            hit = sp.get("trace_id") in sids or (customer_id and (sp.get("attrs") or {}).get("customer.id") == customer_id)
        except (ValueError, AttributeError):
            hit = False
        if hit:
            removed += 1
        else:
            keep.append(raw)
    if not removed:
        return 0
    with p.open("rb") as f:
        f.seek(size)
        tail = f.read()
    tmp = p.with_suffix(p.suffix + ".erase-tmp")
    tmp.write_bytes(b"".join(keep) + tail)
    os.replace(tmp, p)
    return removed
