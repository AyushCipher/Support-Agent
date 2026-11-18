"""PII vault / memory encryption at rest (AES-GCM, SA_VAULT_KEY) and delete-on-request via crypto-shredding."""
import json
import os
import sqlite3
import subprocess
import sys

import pytest

from supportagent.guardrails.pii import PIIVault, new_vault
from supportagent.llm.scripted import ScriptedLLM
from supportagent.memory.store import CustomerMemory
from supportagent.security.audit import AuditLog
from supportagent.security.crypto import CryptoUnavailable, KeyShredded, Protector, VaultKeyMissing, subject_for
from supportagent.security.privacy import erase_customer

crypto = pytest.importorskip("cryptography")


@pytest.fixture
def prot(tmp_path):
    return Protector.from_env(tmp_path / "keys", passphrase="correct horse battery staple")


# ---- encryption at rest
def test_vault_values_are_ciphertext_in_state_and_roundtrip(prot):
    state = new_vault()
    v = PIIVault(state, "alice@example.com", customer_id="C001", protector=prot)
    tok = v.tokenize_text("mail alice@example.com or 415-555-0142", own=True)[0]
    raw = json.dumps(state)
    assert "alice@example.com" not in raw and "415-555-0142" not in raw                # nothing sensitive in the persisted state
    assert all(x.startswith("enc:v1:") for x in state["map"].values())
    assert v.token_for("EMAIL", "alice@example.com") == "<EMAIL_1>"                    # blind index keeps tokens stable
    assert PIIVault(state, "alice@example.com", protector=prot).detokenize_own(tok)[0] == "mail alice@example.com or 415-555-0142"
    assert state["subj"] == subject_for("C001")


def test_ciphertext_is_authenticated_and_bound_to_its_location(prot):
    state = new_vault()
    v = PIIVault(state, "alice@example.com", customer_id="C001", protector=prot)
    v.tokenize_text("alice@example.com dan@example.com", own=True)
    t1, t2 = state["map"]["<EMAIL_1>"], state["map"]["<EMAIL_2>"]
    state["map"]["<EMAIL_1>"], state["map"]["<EMAIL_2>"] = t2, t1                      # swap ciphertexts between tokens
    with pytest.raises(ValueError, match="decryption failed"):
        v.detokenize_own("<EMAIL_1>")
    state["map"]["<EMAIL_1>"] = t1[:-4] + ("AAAA" if not t1.endswith("AAAA") else "BBBB")      # flip bits
    with pytest.raises(ValueError, match="decryption failed"):
        v.detokenize_own("<EMAIL_1>")


def test_plaintext_dev_mode_unchanged_without_key(monkeypatch):
    monkeypatch.delenv("SA_VAULT_KEY", raising=False)
    monkeypatch.delenv("SA_ENV", raising=False)
    state = new_vault()
    PIIVault(state, "alice@example.com").tokenize_text("alice@example.com", own=True)
    assert state["map"]["<EMAIL_1>"] == "alice@example.com" and "subj" not in state


def test_production_fails_closed_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("SA_VAULT_KEY", raising=False)
    monkeypatch.setenv("SA_ENV", "production")
    with pytest.raises(VaultKeyMissing, match="SA_VAULT_KEY"):
        PIIVault(new_vault(), "alice@example.com")
    with pytest.raises(VaultKeyMissing):
        CustomerMemory(sqlite3.connect(":memory:"))


def test_missing_crypto_extra_is_a_clear_error(monkeypatch, tmp_path):
    import supportagent.security.crypto as cr
    monkeypatch.setattr(cr, "crypto_available", lambda: False)
    with pytest.raises(CryptoUnavailable, match=r"supportagent\[crypto\]"):
        Protector.from_env(tmp_path / "k", passphrase="x")


def test_wrong_master_key_cannot_open_data_keys(tmp_path):
    a = Protector.from_env(tmp_path / "keys", passphrase="right key")
    sealed = a.seal("c-C001", "secret", "ctx")
    b = Protector.from_env(tmp_path / "keys", passphrase="wrong key")                  # same key dir, different master key
    with pytest.raises(ValueError):
        b.open("c-C001", sealed, "ctx")


def test_memory_is_encrypted_and_recalls(prot):
    conn = sqlite3.connect(":memory:")
    mem = CustomerMemory(conn, prot)
    mem.remember("C001", "refund_issued", "ORD-1001 $37.50 to alice@example.com", "tool:orders__issue_refund")
    assert "alice@example.com" not in conn.execute("SELECT value FROM customer_memory").fetchone()[0]
    assert mem.recall("C001") == ["refund_issued: ORD-1001 $37.50 to alice@example.com"]


# ---- delete-on-request
def _populate(db_path, prot):
    """A realistic runtime state: a session checkpoint with an encrypted vault, memory, an approval and an audit chain."""
    from supportagent.guardrails.approval import ApprovalService
    from supportagent.harness.checkpoint import CheckpointStore, Session
    from supportagent.harness.types import Message
    conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    store, mem, audit, approvals = CheckpointStore(conn), CustomerMemory(conn, prot), AuditLog(conn), ApprovalService(conn)
    s = Session(id="S-1", customer_id="C001", customer_email="alice@example.com")
    v = PIIVault(s.vault, s.customer_email, customer_id="C001", protector=prot)
    tok = v.tokenize_text("my phone is 415-555-0142", own=True)[0]
    s.messages = [Message("user", tok + " refund ORD-1001"), Message("assistant", "Done, refunded to alice@example.com")]
    s.summary = "alice@example.com asked about a refund"
    s.events = [{"type": "tool_call", "detail": "alice@example.com"}]
    store.save(s)
    other = Session(id="S-2", customer_id="C002", customer_email="bob@example.com")
    PIIVault(other.vault, other.customer_email, customer_id="C002", protector=prot).tokenize_text("bob@example.com", own=True)
    store.save(other)
    mem.remember("C001", "refund_issued", "ORD-1001 $37.50", "tool:orders__issue_refund")
    mem.remember("C002", "refund_issued", "ORD-1002 $10.00", "tool:orders__issue_refund")
    approvals.request("S-1", "orders__issue_refund", {"order_id": "ORD-1001", "email": "alice@example.com"}, "over cap", "Refund for alice@example.com")
    audit.append("session_started", agent="agent:triage", session="S-1", customer="C001")
    audit.append("tool_call", agent="agent:billing", session="S-1", customer="C001", detail={"tool": "issue_refund"})
    audit.append("session_started", agent="agent:triage", session="S-2", customer="C002")
    return conn, store, mem, audit


def test_crypto_shredding_makes_deleted_data_unrecoverable_and_keeps_audit_intact(tmp_path, prot):
    conn, store, mem, audit = _populate(tmp_path / "state.db", prot)
    head_before, n_before = audit.head(), audit.verify()[2]
    hashes_before = [e["hash"] for e in audit.entries()]
    backup = sqlite3.connect(tmp_path / "backup.db")                                  # an attacker-held / operator backup taken BEFORE erasure
    conn.backup(backup)
    backup_vault = json.loads(backup.execute("SELECT state FROM sessions WHERE id='S-1'").fetchone()[0])["vault"]
    assert backup_vault["map"] and PIIVault(backup_vault, "alice@example.com", protector=prot).detokenize_own("<PHONE_1>")[0] == "415-555-0142"

    rep = erase_customer(conn, "C001", audit, None, prot, requested_by="dpo")
    assert rep["sessions_redacted"] == 1 and rep["memory_rows_deleted"] == 1 and rep["keys_shredded"] >= 1 and rep["approvals_redacted"] == 1

    # live data: redacted
    s = store.load("S-1")
    assert s.customer_email == "" and s.vault == {} and s.summary == "" and all(m.content in ("", "[erased]") for m in s.messages)
    raw = conn.execute("SELECT state FROM sessions WHERE id='S-1'").fetchone()[0]
    assert "alice@example.com" not in raw and "415-555-0142" not in raw and "ORD-1001" not in raw
    assert mem.recall("C001") == [] and mem.recall("C002") == ["refund_issued: ORD-1002 $10.00"]         # others untouched
    ap = conn.execute("SELECT args, summary, status FROM approvals").fetchone()
    assert ap[0] == "{}" and "alice" not in ap[1] and ap[2] == "denied"                # pending approval cancelled + redacted

    # the backup is dead: ciphertext survives but the key does not - even a holder of the master key cannot read it
    fresh = Protector.from_env(tmp_path / "keys", passphrase="correct horse battery staple")
    assert PIIVault(backup_vault, "alice@example.com", protector=fresh).detokenize_own("<PHONE_1>")[0] == "<PHONE_1>"   # unresolved, not leaked
    with pytest.raises(KeyShredded):
        fresh.open(backup_vault["subj"], backup_vault["map"]["<PHONE_1>"], "vault|<PHONE_1>")
    old_mem = backup.execute("SELECT value FROM customer_memory WHERE customer_id='C001'").fetchone()[0]
    with pytest.raises(KeyShredded):
        fresh.open(subject_for("C001"), old_mem, "memory|C001|refund_issued")
    # a different customer's key and data are unaffected
    assert PIIVault(json.loads(store.conn.execute("SELECT state FROM sessions WHERE id='S-2'").fetchone()[0])["vault"], "bob@example.com",
                    protector=fresh).detokenize_own("<EMAIL_1>")[0] == "bob@example.com"

    # audit chain: not rewritten, still verifies, and now carries a tombstone with no personal data in it
    ok, bad, n = audit.verify()
    assert ok and bad is None and n == n_before + 1
    entries = audit.entries()
    assert [e["hash"] for e in entries[:n_before]] == hashes_before and entries[-1]["action"] == "privacy_erasure"   # history byte-identical
    assert audit.hash_at(n_before) is not None and head_before != audit.head()
    tomb = entries[-1]
    assert tomb["agent"] == "human:dpo" and tomb["customer"] == "C001" and "alice@example.com" not in json.dumps(tomb)
    assert tomb["detail"]["sessions_redacted"] == 1 and "business records" in tomb["detail"]["retained"][0]


def test_erasure_does_not_break_anchors_taken_before_it(tmp_path, prot):
    from supportagent.security.anchors import AnchorLog, AnchorSigner, AnchorVerifier, anchor_chain, verify_anchors
    conn, _, _, audit = _populate(tmp_path / "state.db", prot)
    alog, signer = AnchorLog(tmp_path / "a.jsonl"), AnchorSigner.from_env(secret="s")
    anchor_chain("runtime", audit, alog, signer)
    erase_customer(conn, "C001", audit, None, prot)
    assert verify_anchors("runtime", audit, alog, AnchorVerifier(secret="s"))["ok"]    # tombstone appended, history untouched


def test_erase_is_idempotent_and_works_in_plaintext_dev_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("SA_VAULT_KEY", raising=False)
    monkeypatch.delenv("SA_ENV", raising=False)
    conn, store, mem, audit = _populate(tmp_path / "plain.db", None)
    assert "415-555-0142" in conn.execute("SELECT state FROM sessions WHERE id='S-1'").fetchone()[0]     # dev mode is plaintext
    r1 = erase_customer(conn, "C001", audit, None, None)
    assert r1["keys_shredded"] == 0 and r1["memory_rows_deleted"] == 1
    assert "415-555-0142" not in conn.execute("SELECT state FROM sessions WHERE id='S-1'").fetchone()[0]
    r2 = erase_customer(conn, "C001", audit, None, None)                               # second request: nothing left, still safe
    assert r2["memory_rows_deleted"] == 0 and audit.verify()[0]


def test_runtime_end_to_end_with_encryption_then_erase(make_rt, monkeypatch, tmp_path):
    monkeypatch.setenv("SA_VAULT_KEY", "end-to-end passphrase")
    monkeypatch.setenv("SA_KEY_DIR", str(tmp_path / "keys"))
    rt = make_rt(llm_factory=lambda t, s: ScriptedLLM("strong", s))
    s = rt.new_session("C001", "alice@example.com")
    assert rt.handle_user_message(s, "Refund ORD-1001 please, call me on 415-555-0142").status == "done"
    state_json = rt.state.execute("SELECT state FROM sessions WHERE id=?", (s.id,)).fetchone()[0]
    vault = json.loads(state_json)["vault"]
    assert vault["map"] and all(v.startswith("enc:v1:") for v in vault["map"].values())          # real runtime path is encrypted
    assert "415-555-0142" not in state_json and "ORD-1001" in state_json
    from supportagent.security.crypto import default_protector
    erase_customer(rt.state, "C001", rt.audit, None, default_protector())
    after = rt.state.execute("SELECT state FROM sessions WHERE id=?", (s.id,)).fetchone()[0]
    assert "alice@example.com" not in after and "ORD-1001" not in after and json.loads(after)["vault"] == {}
    assert rt.audit.verify()[0] and rt.audit.entries()[-1]["action"] == "privacy_erasure"


def test_cli_privacy_delete_preserves_chain(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    conn, _, mem, audit = _populate(runs / "state.db", None)
    conn.close()
    env = {k: v for k, v in os.environ.items() if k not in ("SA_VAULT_KEY", "SA_ENV")}
    env.update({"SA_RUNS_DIR": str(runs), "PYTHONPATH": os.pathsep.join(["src", "."])})
    run = lambda *a: subprocess.run([sys.executable, "-m", "supportagent.cli", *a], capture_output=True, text=True, env=env, timeout=60)   # noqa: E731
    out = run("privacy", "delete", "--customer", "C001", "--requested-by", "dpo")
    assert out.returncode == 0, out.stderr
    assert "erased C001: 1 sessions redacted, 1 memory rows deleted, 1 approvals redacted" in out.stdout
    v = run("audit", "verify")
    assert "agent runtime chain: OK (4 entries" in v.stdout                            # 3 original + 1 tombstone, chain intact
    c = sqlite3.connect(runs / "state.db")
    assert c.execute("SELECT COUNT(*) FROM customer_memory WHERE customer_id='C001'").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM customer_memory WHERE customer_id='C002'").fetchone()[0] == 1
