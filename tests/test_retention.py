"""Data retention job: dry-run, purge/archive per class, safety rules, audit tombstones, idempotency (ADR 0016)."""
import gzip
import json
import sqlite3

import pytest

from supportagent.guardrails.approval import ApprovalService
from supportagent.harness.checkpoint import CheckpointStore, Session
from supportagent.harness.types import Message, ToolCall
from supportagent.ops.dedup import RequestDedup
from supportagent.ops.flags import FlagStore
from supportagent.ops.retention import PURGED, DEFAULT_POLICY, PolicyError, RetentionJob, load_policy
from supportagent.security.audit import AuditLog

DAY = 86400.0
NOW = 10_000_000_000.0


@pytest.fixture
def world(tmp_path):
    conn = sqlite3.connect(":memory:", isolation_level=None)
    store, audit, flags, approvals = CheckpointStore(conn), AuditLog(conn), FlagStore(conn), ApprovalService(conn)

    def add(sid, age_days, status="done", customer="C001", vault=True):
        s = Session(sid, customer, "alice@example.com", status=status,
                    messages=[Message("user", "refund ORD-1001, my email is <EMAIL_1>"), Message("assistant", "", [ToolCall("t", "orders__get_order", {"order_id": "ORD-1001"})]),
                              Message("tool", "secret tool text", tool_call_id="t")],
                    vault={"<EMAIL_1>": "alice@example.com"} if vault else {}, summary="customer asked for a refund", cost_usd=0.01)
        if status == "awaiting_approval":
            s.pending = {"approval_id": "A1", "call": {}, "args": {}, "remaining": []}
        store.save(s)
        conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (NOW - age_days * DAY, sid))
        return s

    def job(policy=None, **kw):
        return RetentionJob(conn, policy or load_policy(), tmp_path, clock=lambda: NOW, **kw)
    return conn, store, audit, flags, approvals, add, job, tmp_path


def raw(conn, sid):
    return json.loads(conn.execute("SELECT state FROM sessions WHERE id=?", (sid,)).fetchone()[0])


def test_policy_defaults_and_validation(tmp_path):
    assert set(load_policy()) == {"transcripts", "vault", "checkpoints", "traces", "audit", "approvals", "request_dedup"}
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"vault": {"days": 7}}))
    assert load_policy(p)["vault"] == {"days": 7, "action": "purge"} and load_policy()["vault"]["days"] == DEFAULT_POLICY["vault"]["days"]
    for bad in ({"nonsense": {"days": 1}}, {"vault": {"days": -1}}, {"vault": {"action": "burn"}}, {"audit": {"action": "purge"}}):
        with pytest.raises(PolicyError):
            load_policy(overrides=bad)


def test_dry_run_reports_but_changes_nothing(world):
    conn, store, audit, flags, approvals, add, job, tmp = world
    add("old", 400)
    add("mid", 100)
    before = (conn.execute("SELECT id, state, updated_at FROM sessions ORDER BY id").fetchall(), audit.head())
    rep = job().run(dry_run=True)
    assert rep["dry_run"] and rep["classes"]["checkpoints"]["eligible"] == 1 and rep["classes"]["transcripts"]["eligible"] == 1
    assert all(c["acted"] == 0 for c in rep["classes"].values())
    assert (conn.execute("SELECT id, state, updated_at FROM sessions ORDER BY id").fetchall(), audit.head()) == before     # no rows, no tombstones
    assert not (tmp / "archive").exists()                                                                                 # no archive files


def test_each_class_acts_by_age_and_session_stays_loadable(world):
    conn, store, audit, flags, approvals, add, job, tmp = world
    add("fresh", 5)
    add("vault_old", 40)                          # > 30d vault, < 90d transcripts
    add("transcript_old", 100)                    # > 90d transcripts, < 180d checkpoints
    add("gone", 200)                              # > 180d: whole row archived + deleted
    rep = job().run()
    c = rep["classes"]
    assert (c["vault"]["acted"], c["transcripts"]["acted"], c["checkpoints"]["acted"]) == (2, 1, 1)
    assert store.load("fresh").vault and store.load("fresh").messages[0].content.startswith("refund")                    # untouched
    v = store.load("vault_old")
    assert v.vault == {} and v.messages[0].content.startswith("refund") and v.retention["vault_purged_at"] == NOW         # PII keys gone, text kept
    t = store.load("transcript_old")                                                                                      # still a valid Session
    assert all(m.content == PURGED for m in t.messages) and t.messages[1].tool_calls[0].arguments == {} and t.summary == ""
    assert t.messages[1].tool_calls[0].name == "orders__get_order" and t.cost_usd == 0.01 and t.vault == {}               # shape + metrics survive
    with pytest.raises(KeyError):
        store.load("gone")
    assert conn.execute("SELECT updated_at FROM sessions WHERE id='vault_old'").fetchone()[0] == NOW - 40 * DAY           # age is not reset by the purge


def test_archive_is_gzip_jsonl_without_pii_and_policy_purge_writes_none(world):
    conn, store, audit, flags, approvals, add, job, tmp = world
    add("gone", 200)
    rep = job().run()
    path = rep["classes"]["checkpoints"]["archive"]
    lines = gzip.open(path, "rt", encoding="utf-8").read().splitlines()
    rec = json.loads(lines[0])
    assert rec["id"] == "gone" and rec["archived_by"] == "retention" and "alice@example.com" not in lines[0] and rec["vault"] == {}
    assert "refund ORD-1001" not in lines[0]                                                                              # security's whitelist redaction
    add("gone2", 200)
    pol = load_policy(overrides={"checkpoints": {"action": "purge"}})
    assert job(pol).run()["classes"]["checkpoints"]["archive"] is None and len(list((tmp / "archive").glob("checkpoints-*"))) == 1


def test_unfinished_and_held_sessions_are_never_touched(world):
    conn, store, audit, flags, approvals, add, job, tmp = world
    for sid, st in (("a", "active"), ("b", "awaiting_approval"), ("c", "awaiting_user")):
        add(sid, 900, status=st)
    add("held", 900)
    add("held_customer", 900, customer="C009")
    flags.set("retention.legal_hold", ["held", "C009"])
    rep = job().run()
    c = rep["classes"]
    assert c["checkpoints"]["acted"] == 0 and c["checkpoints"]["skipped_nonterminal"] == 3 and c["checkpoints"]["skipped_hold"] == 2
    assert {r[0] for r in conn.execute("SELECT id FROM sessions")} == {"a", "b", "c", "held", "held_customer"}
    assert store.load("b").pending is not None and store.load("held").messages[0].content != PURGED


def test_tombstones_are_audit_chained_and_the_chain_still_verifies(world):
    conn, store, audit, flags, approvals, add, job, tmp = world
    add("gone", 200)
    add("t_old", 100)
    n0 = audit.verify()[2]
    job().run()
    assert audit.verify()[0] and audit.verify()[2] == n0 + 3                       # transcripts + vault + checkpoints tombstones
    ts = [e for e in audit.entries() if e["action"] == "retention_purge"]
    by = {e["detail"]["class"]: e["detail"] for e in ts}
    assert by["checkpoints"]["count"] == 1 and by["checkpoints"]["ids_sample"] == ["gone"] and by["checkpoints"]["archive"].endswith(".jsonl.gz")
    assert by["transcripts"]["ids_sample"] == ["t_old"] and len(by["transcripts"]["ids_sha256"]) == 64
    assert all(e["agent"] == "system:retention" for e in ts)
    assert "alice" not in json.dumps(ts)                                                                                  # tombstones hold no personal data


def test_second_run_is_idempotent(world):
    conn, store, audit, flags, approvals, add, job, tmp = world
    add("t_old", 100)
    add("gone", 200)
    first = job().run()
    second = job().run()
    assert first["classes"]["transcripts"]["acted"] == 1 and second["classes"]["transcripts"]["acted"] == 0 and second["classes"]["checkpoints"]["acted"] == 0
    assert second["classes"]["vault"]["eligible"] == 0


def test_session_modified_during_the_job_is_skipped_not_clobbered(world):
    conn, store, audit, flags, approvals, add, job, tmp = world
    add("racy", 100)
    j = job()
    real_execute = conn.execute

    class Racing:                                                                       # a runtime saves the session between our read and our write
        def __getattr__(self, n):
            return getattr(conn, n)

        def execute(self, sql, *a):
            if sql.startswith("UPDATE sessions SET state"):
                real_execute("UPDATE sessions SET updated_at=? WHERE id='racy'", (NOW,))
            return real_execute(sql, *a)
    j.conn = Racing()
    rep = j.run()
    assert rep["classes"]["transcripts"]["acted"] == 0 and rep["classes"]["transcripts"]["skipped_changed"] == 1
    assert store.load("racy").messages[0].content != PURGED


def test_traces_are_archived_and_rewritten_keeping_fresh_and_unparsable_lines(world):
    conn, store, audit, flags, approvals, add, job, tmp = world
    old, new = {"name": "x", "start": NOW - 40 * DAY, "end": NOW - 40 * DAY}, {"name": "y", "start": NOW - DAY, "end": NOW - DAY}
    (tmp / "traces.jsonl").write_text(json.dumps(old) + "\nGARBAGE LINE\n" + json.dumps(new) + "\n", encoding="utf-8")
    dry = job().run(dry_run=True)["classes"]["traces"]
    assert dry["eligible"] == 1 and "x" in (tmp / "traces.jsonl").read_text()
    rep = job().run()["classes"]["traces"]
    kept = (tmp / "traces.jsonl").read_text().splitlines()
    assert rep["acted"] == 1 and '"y"' in kept[1] and kept[0] == "GARBAGE LINE" and len(kept) == 2
    assert json.loads(gzip.open(rep["archive"], "rt").read())["name"] == "x"


def test_audit_is_archive_only_and_the_live_chain_is_untouched(world):
    conn, store, audit, flags, approvals, add, job, tmp = world
    for i in range(3):
        audit.append("x", ts=NOW - 400 * DAY + i)
    audit.append("recent", ts=NOW - DAY)
    head, n = audit.head(), audit.verify()[2]
    rep = job().run()["classes"]["audit"]
    assert rep["eligible"] == 3 and rep["acted"] == 0 and audit.head() == head and audit.verify() == (True, None, n)      # nothing deleted
    rows = [json.loads(x) for x in gzip.open(rep["archive"], "rt").read().splitlines()]
    assert [r["action"] for r in rows] == ["x"] * 3 and rows[1]["prev_hash"] == rows[0]["hash"]                          # chain links preserved
    again = job().run()["classes"]["audit"]
    assert again["eligible"] == 0                                                                                         # already exported: not exported twice


def test_pending_approvals_survive_and_decided_ones_expire(world):
    conn, store, audit, flags, approvals, add, job, tmp = world
    old_pending = approvals.request("S", "orders__issue_refund", {}, "over_auto_cap", "x")["id"]
    old_done = approvals.request("S", "orders__issue_refund", {}, "over_auto_cap", "y")["id"]
    approvals.decide(old_done, True, "r")
    conn.execute("UPDATE approvals SET created_at=?, decided_at=CASE WHEN status='pending' THEN NULL ELSE ? END", (NOW - 500 * DAY, NOW - 500 * DAY))
    rep = job().run()["classes"]["approvals"]
    assert rep["acted"] == 1
    assert [r[0] for r in conn.execute("SELECT id FROM approvals")] == [old_pending]


def test_request_dedup_rows_expire(world):
    conn, store, audit, flags, approvals, add, job, tmp = world
    d = RequestDedup(conn, clock=lambda: NOW - 5 * DAY)
    d.begin("k", "fp")
    d.complete("k", {"ok": 1})
    assert job().run()["classes"]["request_dedup"]["acted"] == 1 and conn.execute("SELECT COUNT(*) FROM api_requests").fetchone()[0] == 0


def test_cli_retention_run_and_dry_run(tmp_path, monkeypatch, capsys):
    from supportagent import cli
    from supportagent.ops import cli as ops_cli
    monkeypatch.setattr(ops_cli, "RUNS_DIR", tmp_path)
    c = sqlite3.connect(tmp_path / "state.db", isolation_level=None)
    store = CheckpointStore(c)
    store.save(Session("old", "C001", "a@b.c", status="done", messages=[Message("user", "hi")]))
    c.execute("UPDATE sessions SET updated_at=?", (1.0,))
    c.close()
    with pytest.raises(SystemExit) as e:
        cli.main(["retention", "run", "--dry-run"])
    assert e.value.code == 0 and "DRY RUN" in capsys.readouterr().out
    assert sqlite3.connect(tmp_path / "state.db").execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
    with pytest.raises(SystemExit):
        cli.main(["retention", "run"])
    assert "APPLIED" in capsys.readouterr().out
    assert sqlite3.connect(tmp_path / "state.db").execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_composes_with_privacy_erasure(world):
    """Erased (security/privacy) sessions are still valid input to the retention job: nothing crashes, the tombstone chain stays intact."""
    from supportagent.security.privacy import erase_customer
    conn, store, audit, flags, approvals, add, job, tmp = world
    add("s1", 200)
    erase_customer(conn, "C001", audit)
    rep = job().run()
    assert rep["classes"]["checkpoints"]["acted"] == 1 and audit.verify()[0]
