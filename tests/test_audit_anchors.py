"""Audit head anchoring: tail truncation (the documented gap of a bare hash chain) is now detected."""
import json
import os
import sqlite3
import subprocess
import sys

import pytest

from supportagent.security.anchors import AnchorError, AnchorLog, AnchorSigner, AnchorVerifier, anchor_chain, verify_anchors
from supportagent.security.audit import AuditLog, _digest, GENESIS


def make_log(n=6):
    conn = sqlite3.connect(":memory:")
    log = AuditLog(conn)
    for i in range(n):
        log.append(f"act{i}", agent="agent:billing", session="S1", customer="C001", detail={"i": i})
    return conn, log


@pytest.fixture
def hmac_ctx(tmp_path):
    s = AnchorSigner.from_env(secret="anchor-secret")                                  # stdlib-only mode: no [crypto] needed
    return AnchorLog(tmp_path / "anchors.jsonl"), s, AnchorVerifier(secret="anchor-secret")


def test_anchor_then_verify_ok_and_grows(hmac_ctx):
    alog, signer, ver = hmac_ctx
    _, log = make_log()
    rec = anchor_chain("runtime", log, alog, signer)
    assert rec["seq"] == 6 and rec["entries"] == 6 and rec["head"] == log.head() and rec["sig"] and rec["ts"]
    assert verify_anchors("runtime", log, alog, ver)["ok"]
    log.append("later", agent="agent:billing")
    anchor_chain("runtime", log, alog, signer)
    r = verify_anchors("runtime", log, alog, ver)
    assert r["ok"] and r["anchors"] == 2 and r["latest_anchor"]["seq"] == 7


def test_tail_truncation_is_detected_by_anchors_but_not_by_the_bare_chain(hmac_ctx):
    alog, signer, ver = hmac_ctx
    conn, log = make_log()
    anchor_chain("runtime", log, alog, signer)
    conn.execute("DELETE FROM audit_chain WHERE seq>=5")                              # attacker drops the last two rows
    assert log.verify()[0] is True                                                    # the chain alone cannot see it
    r = verify_anchors("runtime", log, alog, ver)
    assert r["ok"] is False and r["truncated"] is True
    assert any("TAIL TRUNCATION" in p and "seq 6" in p for p in r["problems"])
    log.append("cover-up", agent="agent:billing")                                     # ...and appending afterwards does not hide it
    assert verify_anchors("runtime", log, alog, ver)["truncated"] is True


def test_full_chain_rewrite_with_recomputed_hashes_is_detected(hmac_ctx):
    alog, signer, ver = hmac_ctx
    conn, log = make_log()
    anchor_chain("runtime", log, alog, signer)
    # attacker rewrites row 6 and recomputes a *consistent* chain: internal verify passes, but the signed head no longer matches
    prev = log.hash_at(5)
    row = conn.execute("SELECT ts,agent,session,customer,action,detail FROM audit_chain WHERE seq=6").fetchone()
    new_detail = json.dumps({"i": 999}, sort_keys=True)
    h = _digest(prev, row[0], row[1], row[2], row[3], row[4], new_detail)
    conn.execute("UPDATE audit_chain SET detail=?, hash=? WHERE seq=6", (new_detail, h))
    assert log.verify()[0] is True
    r = verify_anchors("runtime", log, alog, ver)
    assert not r["ok"] and any("REWRITTEN" in p for p in r["problems"])


def test_anchor_log_tampering_is_detected(hmac_ctx, tmp_path):
    alog, signer, ver = hmac_ctx
    _, log = make_log()
    for _ in range(3):
        log.append("x", agent="a")
        anchor_chain("runtime", log, alog, signer)
    lines = alog.path.read_text().splitlines()
    # 1) edit an anchor's head: signature no longer matches
    forged = json.loads(lines[1])
    forged["head"] = "f" * 64
    alog.path.write_text("\n".join([lines[0], json.dumps(forged, sort_keys=True, separators=(",", ":")), lines[2]]) + "\n")
    assert any("bad signature" in p for p in verify_anchors("runtime", log, alog, ver)["problems"])
    # 2) silently remove a middle anchor: the prev-link breaks
    alog.path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert any("broken link" in p for p in verify_anchors("runtime", log, alog, ver)["problems"])
    # 3) anchors signed with a different key are not accepted
    other = AnchorLog(tmp_path / "other.jsonl")
    anchor_chain("runtime", log, other, AnchorSigner.from_env(secret="attacker-key"))
    r = verify_anchors("runtime", log, other, ver)
    assert not r["ok"] and any("bad signature" in p for p in r["problems"]) and r["anchors"] == 0


def test_no_anchors_is_a_failure_not_a_silent_pass(hmac_ctx):
    alog, _, ver = hmac_ctx
    _, log = make_log()
    r = verify_anchors("runtime", log, alog, ver)
    assert not r["ok"] and "no valid anchors" in r["problems"][0]


def test_refuses_to_anchor_an_already_tampered_chain(hmac_ctx):
    alog, signer, _ = hmac_ctx
    conn, log = make_log()
    conn.execute("UPDATE audit_chain SET agent='evil' WHERE seq=2")
    with pytest.raises(AnchorError, match="fails verification"):
        anchor_chain("runtime", log, alog, signer)


def test_chains_are_anchored_independently(hmac_ctx):
    alog, signer, ver = hmac_ctx
    _, a = make_log(3)
    _, b = make_log(5)
    anchor_chain("runtime", a, alog, signer)
    anchor_chain("business", b, alog, signer)
    assert verify_anchors("runtime", a, alog, ver)["anchors"] == 1 and verify_anchors("business", b, alog, ver)["ok"]
    assert not verify_anchors("business", a, alog, ver)["ok"]                         # wrong chain against business anchors


def test_empty_chain_anchor_is_genesis(hmac_ctx):
    alog, signer, ver = hmac_ctx
    log = AuditLog(sqlite3.connect(":memory:"))
    assert anchor_chain("runtime", log, alog, signer)["head"] == GENESIS and verify_anchors("runtime", log, alog, ver)["ok"]


def test_eddsa_anchors_verify_with_public_key_only(tmp_path, monkeypatch):
    pytest.importorskip("cryptography")
    monkeypatch.delenv("SA_ANCHOR_SECRET", raising=False)
    key = tmp_path / "anchor_key.json"
    signer = AnchorSigner.from_env(key_file=key)                                      # generates a key file on first use
    assert signer.alg == "EdDSA" and key.exists()
    alog = AnchorLog(tmp_path / "a.jsonl")
    conn, log = make_log()
    anchor_chain("runtime", log, alog, signer)
    pub_file = tmp_path / "anchor_pub.json"
    pub_file.write_text(json.dumps({"x": signer.public["x"]}))                        # what you hand to the auditing machine
    assert AnchorVerifier.from_env(pub_file).public == signer.public and set(json.loads(pub_file.read_text())) == {"x"}
    ver = AnchorVerifier.from_env(pub_file)                                           # public half only: cannot forge
    assert "d" not in signer.public and verify_anchors("runtime", log, alog, ver)["ok"]
    conn.execute("DELETE FROM audit_chain WHERE seq=6")
    assert verify_anchors("runtime", log, alog, ver)["truncated"]
    monkeypatch.setenv("SA_ENV", "production")
    with pytest.raises(AnchorError, match="refusing to auto-generate"):
        AnchorSigner.from_env(key_file=tmp_path / "missing.json")


def test_cli_anchor_and_verify_detects_truncation(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    conn = sqlite3.connect(runs / "state.db", isolation_level=None)
    log = AuditLog(conn)
    for i in range(5):
        log.append(f"act{i}", agent="agent:billing", session="S1", customer="C001")
    env = {**os.environ, "SA_RUNS_DIR": str(runs), "SA_ANCHOR_SECRET": "cli-secret", "PYTHONPATH": os.pathsep.join(["src", "."])}
    run = lambda *a: subprocess.run([sys.executable, "-m", "supportagent.cli", "audit", *a], capture_output=True, text=True, env=env, timeout=60)   # noqa: E731
    out = run("anchor")
    assert out.returncode == 0 and "anchored agent runtime chain: seq 5" in out.stdout
    ok = run("verify", "--anchors")
    assert ok.returncode == 0 and "anchors: OK (1 verified anchors)" in ok.stdout
    conn.execute("DELETE FROM audit_chain WHERE seq>=4")
    bad = run("verify", "--anchors")
    assert bad.returncode == 2 and "TAIL TRUNCATION" in bad.stdout and "agent runtime chain: OK" in bad.stdout   # chain alone says OK
    conn.close()
