"""Per-scenario world setup and fault injection for the held-out suite.

* apply_setup(db, setup): insert extra orders / subscriptions (dates are relative to the frozen clock NOW) into the fresh
  per-trial database BEFORE the agent starts. Existing scenarios have no `setup`, so they are unaffected.
* FaultyRegistry: wraps the MCP registry and makes chosen tools fail at the registry boundary, mirroring the error shape
  MCPRegistry.call() itself returns when a server times out / is down. It does not touch transports or the guard code.
    mode "timeout"              -> {"error": "tool_unavailable"} (nothing executed)
    mode "error"                -> {"error": "internal_error"}   (nothing executed)
    mode "timeout_after_commit" -> the call REALLY executes, then the caller is told it timed out (lost response)
  `times` = how many calls fail before the tool recovers (use a large number for a persistent outage).
"""
from __future__ import annotations

from supportagent.shop.db import connect, iso


def apply_setup(db_path: str, setup: dict | None) -> None:
    if not setup:
        return
    c = connect(db_path)
    try:
        for o in setup.get("orders", []):
            c.execute("INSERT INTO orders VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (o["id"], o["customer"], o["status"], iso(o["placed"]),
                       iso(o["delivered"]) if o.get("delivered") is not None else None,
                       iso(days_ahead=o["est"]) if o.get("est") is not None else None,
                       o["category"], o["total"], o["ship"], o["item"], o.get("notes", ""), o.get("ship_to", "1 Test St"),
                       o.get("refunded", 0)))
        for s in setup.get("subscriptions", []):
            c.execute("INSERT INTO subscriptions VALUES(?,?,?,?,?,?,?,?,?)",
                      (s["id"], s["customer"], s["plan"], s["status"], s["interval"], s["price"], iso(s["renewed"]),
                       iso(days_ahead=s["period_end"]), s.get("refunded", 0)))
    finally:
        c.close()


class FaultyRegistry:
    def __init__(self, inner, faults: list[dict]):
        self.inner = inner
        self.faults = [dict(f, left=f.get("times", 1)) for f in faults]
        self.injected: list[str] = []

    def specs(self):
        return self.inner.specs()

    def call(self, full_name, args, role="system", timeout=None, session_id=None):
        for f in self.faults:
            if f["tool"] == full_name and f["left"] > 0:
                f["left"] -= 1
                self.injected.append(f"{full_name}:{f['mode']}")
                server = full_name.split("__", 1)[0]
                if f["mode"] == "timeout":
                    return {"error": "tool_unavailable", "message": f"{server} failed: MCPTimeout"}, True
                if f["mode"] == "error":
                    return {"error": "internal_error", "message": f"{server} upstream returned 500"}, True
                if f["mode"] == "timeout_after_commit":
                    self.inner.call(full_name, args, role=role, timeout=timeout, session_id=session_id)
                    return {"error": "tool_unavailable", "message": f"{server} failed: MCPTimeout"}, True
                raise ValueError(f"unknown fault mode {f['mode']}")
        return self.inner.call(full_name, args, role=role, timeout=timeout, session_id=session_id)

    def __getattr__(self, name):
        return getattr(self.inner, name)
