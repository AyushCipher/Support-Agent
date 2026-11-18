from __future__ import annotations

import argparse
import sqlite3
import sys

from .config import PRESETS, RUNS_DIR


def _chat(args) -> None:
    from .guardrails.approval import ApprovalService
    from .harness.checkpoint import CheckpointStore
    from .harness.runtime import Runtime
    from .mcp.client import build_registry
    from .memory.store import CustomerMemory
    from .obs.tracing import JsonlSink, Tracer
    from .shop.db import CUSTOMERS, build_seed_db
    RUNS_DIR.mkdir(exist_ok=True)
    db = RUNS_DIR / "cli_shop.db"
    build_seed_db(db)
    cfg = PRESETS[args.config].with_(backend=args.backend)
    state = sqlite3.connect(":memory:", check_same_thread=False)

    def reviewer(rec: dict) -> tuple[bool, str]:               # interactive human-in-the-loop in the terminal
        print(f"\n  [APPROVAL NEEDED] {rec['summary']}\n  reason: {rec['reason']}")
        ok = input("  approve? [y/N] ").strip().lower() == "y"
        return ok, "terminal reviewer"

    rt = Runtime(cfg, build_registry(str(db), cfg.allow_anon_server, mode=args.mcp), CheckpointStore(state),
                 ApprovalService(state, reviewer), Tracer([JsonlSink(RUNS_DIR / "traces.jsonl")]), CustomerMemory(state))
    email = {c[0]: c[2] for c in CUSTOMERS}[args.customer]
    s = rt.new_session(args.customer, email)
    print(f"Chatting as {args.customer} ({cfg.name}, {cfg.backend}). Ctrl-D to quit.")
    while True:
        try:
            text = input("\nyou> ")
        except EOFError:
            break
        r = rt.handle_user_message(s, text)
        print(f"agent> {r.reply}")
        print(f"  [{s.role} | tokens={s.tokens} cost=${s.cost_usd:.4f} | events: {', '.join(e['type'] for e in s.events[-4:])}]")


def _anchor_file():
    import os
    return os.getenv("SA_ANCHOR_FILE") or RUNS_DIR / "audit_anchors.jsonl"


def _privacy(a) -> None:
    from .security.audit import AuditLog
    from .security.crypto import default_protector
    from .security.privacy import erase_customer
    state = sqlite3.connect(RUNS_DIR / "state.db", isolation_level=None)
    emails: tuple[str, ...] = ()
    shop = AuditLog(sqlite3.connect(RUNS_DIR / "shop.db", isolation_level=None)) if (RUNS_DIR / "shop.db").exists() else None
    if (RUNS_DIR / "shop.db").exists():
        row = sqlite3.connect(RUNS_DIR / "shop.db").execute("SELECT email FROM customers WHERE id=?", (a.customer,)).fetchone()             if _has_customers(RUNS_DIR / "shop.db") else None
        emails = (row[0],) if row else ()
    r = erase_customer(state, a.customer, AuditLog(state), shop, default_protector(), requested_by=a.requested_by, extra_emails=emails,
                       traces_path=RUNS_DIR / "traces.jsonl")
    print(f"erased {a.customer}: {r['sessions_redacted']} sessions redacted, {r['memory_rows_deleted']} memory rows deleted, "
          f"{r['approvals_redacted']} approvals redacted, {r['keys_shredded']} keys shredded, {r['trace_spans_deleted']} trace spans deleted; "
          f"tombstone {r.get('tombstone_hash', '')[:16]}...")
    print("audit chain preserved: run `supportagent audit verify` (business records and customer id on audit rows are retained by design)")


def _has_customers(path) -> bool:
    c = sqlite3.connect(path)
    try:
        return c.execute("SELECT 1 FROM sqlite_master WHERE name='customers'").fetchone() is not None
    finally:
        c.close()


def _admin(a) -> None:
    from .ops.flags import FlagStore
    from .security.audit import AuditLog
    RUNS_DIR.mkdir(exist_ok=True)
    state = sqlite3.connect(RUNS_DIR / "state.db", isolation_level=None)
    flags, audit = FlagStore(state), AuditLog(state)
    if a.cmd == "flags":
        if a.action == "set":
            import json as _j
            try:
                val = _j.loads(a.value)
            except ValueError:
                val = a.value
            flags.set(a.key, val, "cli")
        for k, v in flags.all().items():
            print(f"{k} = {v}")
    elif a.cmd == "kill":
        flags.set_kill(a.scope, a.state == "on", "cli")
        print(f"kill_switch.{a.scope} = {a.state}")
    elif a.cmd == "audit":
        shop = AuditLog(sqlite3.connect(RUNS_DIR / "shop.db")) if (RUNS_DIR / "shop.db").exists() else None
        chains = (("agent runtime chain", "runtime", audit), ("business chain", "business", shop))
        if a.action == "anchor":
            from .security.anchors import AnchorLog, AnchorSigner, anchor_chain
            alog, signer = AnchorLog(a.anchor_file or _anchor_file()), AnchorSigner.from_env()
            for name, key, log in chains:
                if log is not None:
                    r = anchor_chain(key, log, alog, signer)
                    print(f"anchored {name}: seq {r['seq']} head {r['head'][:16]}... ({r['alg']}/{r['kid']}) -> {alog.path}")
            print("ship this file to write-once storage (S3 Object Lock etc.): see docs/compliance.md")
        elif a.action == "verify":
            bad_any = False
            for name, key, log in chains:
                if log is not None:
                    ok, bad, n = log.verify()
                    print(f"{name}: {'OK' if ok else f'TAMPERED at seq {bad}'} ({n} entries, head {log.head()[:16]}...)")
                    if a.anchors:
                        from .security.anchors import AnchorLog, AnchorVerifier, verify_anchors
                        rep = verify_anchors(key, log, AnchorLog(a.anchor_file or _anchor_file()), AnchorVerifier.from_env())
                        print(f"  anchors: {'OK' if rep['ok'] else 'FAILED'} ({rep['anchors']} verified anchors)")
                        for pr in rep["problems"]:
                            print(f"    - {pr}")
                        bad_any |= not rep["ok"]
            if bad_any:
                sys.exit(2)
        else:
            print(audit.replay(a.session))
            if shop:
                print("--- business-side (orders/tickets servers):")
                print(shop.replay(a.session))
    elif a.cmd == "shadow-report":
        print(flags.shadow_report())


def _kb(a) -> None:
    if a.action == "search":
        from .config import POLICY_DIR
        from .rag.hybrid import get_retriever
        if not a.query:
            sys.exit("usage: supportagent kb search QUERY [--mode hybrid]")
        for r in get_retriever(str(POLICY_DIR), a.mode).search(a.query, 3):
            print(f"{r['score']:<8} {r['ref']}\n         {r['text'][:140]}")
        return
    try:                                                   # the eval suite lives beside the package in the repo checkout
        from evals.retrieval_eval import OUT_JSON, OUT_MD, generate
    except ImportError:
        sys.exit("kb eval needs the repository checkout: run it from the repo root (PYTHONPATH=src:.)")
    p = generate(tuple(x for x in a.embedders.split(",") if x))
    print(f"wrote {OUT_MD}\nwrote {OUT_JSON}\n{p['readme_line']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="supportagent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chat", help="interactive terminal chat")
    c.add_argument("--customer", default="C001")
    c.add_argument("--config", default="full", choices=list(PRESETS))
    c.add_argument("--backend", default="scripted", choices=["scripted", "anthropic", "gemini"])
    c.add_argument("--mcp", default="inprocess", choices=["inprocess", "stdio"])
    sv = sub.add_parser("serve", help="HTTP API + chat/approval UIs")
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("--config", default="full", choices=list(PRESETS))
    sv.add_argument("--backend", default=None, choices=["scripted", "anthropic", "gemini"])
    fl = sub.add_parser("flags", help="list/set feature flags (prompt.stable, prompt.canary_version, prompt.canary_percent, prompt.shadow_version)")
    fl.add_argument("action", choices=["list", "set"])
    fl.add_argument("key", nargs="?")
    fl.add_argument("value", nargs="?")
    k = sub.add_parser("kill", help="kill switch: stop the agent (global) or just money actions (money)")
    k.add_argument("state", choices=["on", "off"])
    k.add_argument("--scope", default="global", choices=["global", "money"])
    au = sub.add_parser("audit", help="verify the tamper-evident chain, or replay one session")
    au.add_argument("action", choices=["verify", "replay", "anchor"])
    au.add_argument("session", nargs="?")
    au.add_argument("--anchors", action="store_true", help="verify: also compare each chain head against the signed anchor log (detects tail truncation)")
    au.add_argument("--anchor-file", default=None, help="anchor log path (default $SA_ANCHOR_FILE or runs/audit_anchors.jsonl)")
    pv = sub.add_parser("privacy", help="data-subject requests")
    pv.add_argument("action", choices=["delete"])
    pv.add_argument("--customer", required=True)
    pv.add_argument("--requested-by", default="cli")
    sub.add_parser("shadow-report", help="agreement between live and shadow prompt versions")
    kb = sub.add_parser("kb", help="knowledge base: `kb eval` regenerates docs/retrieval_results.md; `kb search QUERY` tries a query")
    kb.add_argument("action", choices=["eval", "search"])
    kb.add_argument("query", nargs="?")
    kb.add_argument("--mode", default=None, choices=["bm25", "bm25+", "dense", "hybrid"], help="search: retrieval mode (default $SA_RETRIEVAL_MODE or bm25)")
    kb.add_argument("--embedders", default="hash,fastembed:BAAI/bge-small-en-v1.5", help="eval: comma separated dense backends to evaluate")
    from .ops.cli import OPS_COMMANDS, register as _register_ops
    _register_ops(sub)                                  # `retention` and `slo` live in ops/cli.py
    a = ap.parse_args(argv)
    if a.cmd in OPS_COMMANDS:
        from .ops.cli import run as _run_ops
        sys.exit(_run_ops(a))
    if a.cmd == "kb":
        _kb(a)
        return
    if a.cmd in ("flags", "kill", "audit", "shadow-report"):
        _admin(a)
    elif a.cmd == "privacy":
        RUNS_DIR.mkdir(exist_ok=True)
        _privacy(a)
    elif a.cmd == "chat":
        _chat(a)
    elif a.cmd == "serve":
        from .api.server import serve
        serve(port=a.port, config=a.config, backend=a.backend)


if __name__ == "__main__":
    main(sys.argv[1:])
