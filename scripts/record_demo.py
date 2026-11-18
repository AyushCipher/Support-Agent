"""Record the `make demo` flows as a reproducible terminal artifact: docs/demo/demo.cast (asciinema v2), demo.svg, demo.txt.

    python scripts/record_demo.py            # or: make demo-record

Offline and deterministic: scripted (simulated) model, a fresh temp runs dir, the real HTTP server on an ephemeral port, real
requests. Random ids (session / approval / refund ids) are replaced by stable placeholders (S-1, AP-1, RFN-1...) and session tokens
are elided, so re-recording produces a byte-identical artifact. The cast's timing is SYNTHETIC (fixed typing/printing delays for
readability), not measured latency. Flows: auto-approved refund, human-approval pause -> reviewer approves -> resume,
cross-customer (IDOR) denial, indirect prompt injection quarantined + session tainted, and session-id hijack refused.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo"
W, H = 110, 40


class Ids:
    """Stable placeholders for random ids so the artifact is reproducible."""
    def __init__(self):
        self.maps: dict[str, dict[str, str]] = {}

    def sub(self, text: str) -> str:
        for pat, pre in ((r"S-[0-9a-f]{10}", "S-"), (r"APR-[0-9a-f]{6,}", "APR-"), (r"RFN-[0-9a-f]{8}", "RFN-"), (r"TCK-[0-9a-f]{6,}", "TCK-")):
            m = self.maps.setdefault(pre, {})
            text = re.sub(pat, lambda x, m=m, pre=pre: m.setdefault(x.group(0), f"{pre}{len(m) + 1}"), text)
        return re.sub(r"st1\.[A-Za-z0-9_-]+", "st1.<elided>", text)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sa-demo-"))
    os.environ.update({"SA_RUNS_DIR": str(tmp), "SA_ADMIN_TOKEN": "demo-reviewer-token", "SA_BACKEND": "scripted"})
    for k in ("SA_ENV", "SA_VAULT_KEY", "SA_PUBLIC_DEMO"):
        os.environ.pop(k, None)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    from http.server import ThreadingHTTPServer

    from supportagent.api import server
    server.RUNS_DIR = tmp
    app = server.App("full", "scripted")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    ids, frames, t = Ids(), [], [0.0]
    lines: list[str] = []

    def emit(text: str, delay: float = 0.05) -> None:
        text = ids.sub(text)
        t[0] += delay
        frames.append([round(t[0], 3), "o", text.replace("\n", "\r\n")])
        lines.append(text)

    def typed(cmd: str) -> None:
        cmd = ids.sub(cmd)                                             # mask ids BEFORE chunking the keystrokes
        emit("\x1b[32m$\x1b[0m ", 0.6)
        for i in range(0, len(cmd), 6):
            emit(cmd[i:i + 6], 0.03)
        emit("\n", 0.2)

    def say(title: str) -> None:
        emit(f"\n\x1b[1;36m== {title}\x1b[0m\n", 0.8)

    def call(path: str, body: dict | None = None, headers: dict | None = None) -> tuple[int, dict]:
        h = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None, headers=h,
                                     method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def show(code: int, obj, keys=None) -> None:
        if keys and isinstance(obj, dict):
            obj = {k: obj[k] for k in keys if k in obj and obj[k] is not None}
        emit(f"\x1b[2mHTTP {code}\x1b[0m " + json.dumps(obj, indent=1, ensure_ascii=False) + "\n", 0.4)

    def chat(cid: str, msg: str, sid: str | None = None, tok: str | None = None) -> dict:
        body = {"customer_id": cid, "message": msg, **({"session_id": sid} if sid else {})}
        hdr = {"X-Session-Token": tok} if tok else {}
        typed(f"curl -s localhost:8080/api/chat {'-H X-Session-Token:$TOK ' if tok else ''}-d '{json.dumps(body)}'")
        code, out = call("/api/chat", body, hdr)
        show(code, out, ["session_id", "status", "reply", "approval_id", "error"])
        return out

    def events(sid: str, tok: str, want: tuple[str, ...]) -> dict:
        typed(f"curl -s localhost:8080/api/sessions/{sid} -H X-Session-Token:$TOK | jq '.tainted, .events'")
        code, s = call(f"/api/sessions/{sid}", headers={"X-Session-Token": tok})
        ev = [e for e in s.get("events", []) if e["type"] in want]
        show(code, {"tainted": s.get("tainted"), "events": ev})
        return s

    emit("\x1b[1msupportagent demo\x1b[0m - offline: simulated model, real harness, real HTTP API, fresh state\n", 0.2)
    say("1. Refund within policy and under the $100 auto cap -> executed automatically")
    r1 = chat("C001", "Refund ORD-1001 please, I changed my mind")

    say("2. High-value refund -> the run PAUSES for a human reviewer")
    r2 = chat("C001", "Refund ORD-1005 please, the espresso machine. Changed my mind.")
    typed("curl -s localhost:8080/api/approvals -H 'Authorization: Bearer $REVIEWER' | jq '.pending'")
    code, ap = call("/api/approvals", headers={"Authorization": "Bearer demo-reviewer-token"})
    show(code, [{k: p[k] for k in ("id", "tool", "reason", "status") if k in p} for p in ap.get("pending", [])])
    if ap.get("pending"):
        aid = ap["pending"][0]["id"]
        typed(f"curl -s -X POST localhost:8080/api/approvals/{aid}/decide -H 'Authorization: Bearer $REVIEWER' -d '{{\"approve\": true}}'")
        code, d = call(f"/api/approvals/{aid}/decide", {"approve": True, "note": "verified receipt"},
                       {"Authorization": "Bearer demo-reviewer-token"})
        show(code, {"approval": {k: d.get("approval", {}).get(k) for k in ("id", "status", "reviewer")}, "status": d.get("status"),
                    "reply": d.get("reply")})

    say("3. Cross-customer access (IDOR): Bob asks to refund Dan's order")
    chat("C002", "Refund ORD-4001 please")

    say("4. Indirect prompt injection hidden in an order note (Bob, ORD-2001)")
    r4 = chat("C002", "Can you check on ORD-2001?")
    if r4.get("session_id"):
        events(r4["session_id"], r4.get("session_token", ""), ("tool_output_injection", "input_injection_flagged", "tool_blocked"))

    say("5. Session hijack: someone who learned Alice's session id, without her session token")
    if r1.get("session_id"):
        chat("C002", "Refund ORD-1002 to me", sid=r1["session_id"])

    say("Audit chain")
    typed("curl -s localhost:8080/api/admin/audit/verify -H 'Authorization: Bearer $REVIEWER'")
    code, v = call("/api/admin/audit/verify", headers={"Authorization": "Bearer demo-reviewer-token"})
    show(code, {k: {"ok": x["ok"], "entries": x["entries"]} for k, x in v.items()})
    httpd.shutdown()
    _ = r2

    OUT.mkdir(parents=True, exist_ok=True)
    header = {"version": 2, "width": W, "height": H, "title": "supportagent demo (offline, simulated model)",
              "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"}}
    (OUT / "demo.cast").write_text(json.dumps(header) + "\n" + "".join(json.dumps(f) + "\n" for f in frames), encoding="utf-8")
    plain = re.sub(r"\x1b\[[0-9;]*m", "", "".join(lines))
    (OUT / "demo.txt").write_text(plain, encoding="utf-8")
    (OUT / "demo.svg").write_text(render_svg("".join(lines)), encoding="utf-8")
    print(f"wrote {OUT / 'demo.cast'} ({len(frames)} frames, {t[0]:.1f}s synthetic), demo.svg, demo.txt")
    return 0


PALETTE = {"32": "#4ade80", "36": "#67e8f9", "2": "#a8a29e"}


def render_svg(ansi: str) -> str:
    """Static render of the whole session (not animated): one <text> row per line, ANSI bold/colour mapped to tspans."""
    rows = ansi.split("\n")
    lh, cw, pad = 17, 7.8, 16
    width = int(max(W, max((len(re.sub(r"\x1b\[[0-9;]*m", "", r)) for r in rows), default=0)) * cw + 2 * pad)
    height = len(rows) * lh + 2 * pad + 24
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" '
           'aria-label="supportagent terminal demo">',
           f'<rect width="100%" height="100%" rx="8" fill="#1c1917"/>',
           '<circle cx="18" cy="14" r="5" fill="#f87171"/><circle cx="34" cy="14" r="5" fill="#fbbf24"/><circle cx="50" cy="14" r="5" fill="#4ade80"/>',
           f'<g font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="13" fill="#e7e5e4" xml:space="preserve">']
    for i, row in enumerate(rows):
        y = pad + 24 + i * lh
        spans, style = [], {}
        for part in re.split(r"(\x1b\[[0-9;]*m)", row):
            m = re.fullmatch(r"\x1b\[([0-9;]*)m", part)
            if m:
                codes = m.group(1).split(";")
                style = {} if codes in (["0"], [""]) else {**style, **{("fill" if c in PALETTE else "bold"): c for c in codes}}
                continue
            if part:
                attrs = (f' fill="{PALETTE[style["fill"]]}"' if "fill" in style else "") + (' font-weight="bold"' if style.get("bold") == "1" else "")
                spans.append(f"<tspan{attrs}>{html.escape(part)}</tspan>")
        out.append(f'<text x="{pad}" y="{y}">{"".join(spans)}</text>')
    out.append("</g></svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
