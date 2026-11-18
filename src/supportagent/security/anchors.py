"""Audit head anchoring: closes the documented tail-truncation gap of the hash chain (ADR-0011).

A hash chain proves *internal* consistency; deleting the last N rows leaves a perfectly valid shorter chain. So we
periodically commit the head to a place the attacker cannot rewrite and later check the live chain against it:

  supportagent audit anchor              -> appends one signed record per chain to the anchor log (JSON lines)
  supportagent audit verify --anchors    -> chain integrity AND "does every anchored (seq, head) still exist, unmodified?"

Anchor record (one line):  {v, chain, ts, seq, entries, head, prev, alg, kid, sig}
  * `prev` = SHA-256 of the previous anchor line, so editing or removing a *middle* anchor is detected locally.
  * `sig`  = Ed25519 (needs the [crypto] extra; key in SA_ANCHOR_KEY_FILE) or HMAC-SHA256 (SA_ANCHOR_SECRET, stdlib only)
    over the canonical record, so an attacker who can write the file but not the key cannot forge anchors.
The local file is append-only by convention only. Truncating the *anchor log's* tail is exactly the same problem one level up,
which is why anchors must also be shipped to write-once storage (S3 Object Lock, transparency log, a second host) - see
docs/compliance.md. What this module guarantees: anything anchored at least once *outside the attacker's reach* can no longer
be silently removed or rewritten.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from pathlib import Path

from .audit import GENESIS, AuditLog
from .crypto import CryptoUnavailable, b64, crypto_available, production_mode, unb64


class AnchorError(Exception): ...


def _canon(rec: dict) -> bytes:
    return json.dumps({k: v for k, v in rec.items() if k != "sig"}, sort_keys=True, separators=(",", ":")).encode()


class AnchorSigner:
    """Signs anchors. Ed25519 when `cryptography` is installed (verify with the public key only), else HMAC with SA_ANCHOR_SECRET."""

    def __init__(self, alg: str, kid: str, sign, public: dict):
        self.alg, self.kid, self._sign, self.public = alg, kid, sign, public

    def sign(self, data: bytes) -> str:
        return self._sign(data)

    @classmethod
    def from_env(cls, key_file: str | os.PathLike | None = None, secret: str | None = None) -> "AnchorSigner":
        secret = secret if secret is not None else os.getenv("SA_ANCHOR_SECRET", "")
        if secret:
            k = secret.encode()
            kid = "hmac-" + hashlib.sha256(k).hexdigest()[:8]
            return cls("HS256", kid, lambda d: b64(hmac.new(k, d, hashlib.sha256).digest()), {"alg": "HS256", "kid": kid})
        if not crypto_available():
            raise CryptoUnavailable('anchor signing needs a key: set SA_ANCHOR_SECRET (HMAC, stdlib) or install the optional extra: '
                                    'pip install "supportagent[crypto]" (Ed25519)')
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        path = Path(key_file or os.getenv("SA_ANCHOR_KEY_FILE") or _default_key_file())
        if path.exists():
            priv = Ed25519PrivateKey.from_private_bytes(unb64(json.loads(path.read_text())["d"]))
        else:
            if production_mode():
                raise AnchorError(f"anchor key file {path} missing; refusing to auto-generate a signing key in production")
            priv = Ed25519PrivateKey.generate()
            path.parent.mkdir(parents=True, exist_ok=True)
            x0 = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            raw_d = priv.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump({"d": b64(raw_d), "x": b64(x0)}, f)
        x = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        kid = "ed-" + hashlib.sha256(x).hexdigest()[:8]
        return cls("EdDSA", kid, lambda d: b64(priv.sign(d)), {"alg": "EdDSA", "kid": kid, "x": b64(x)})


class AnchorVerifier:
    """Verifies anchor signatures. Ed25519: needs only the public key (`x`), so verification can run on an untrusted host."""

    def __init__(self, public: dict | None = None, secret: str | None = None):
        self.public, self.secret = public or {}, secret

    @classmethod
    def from_signer(cls, s: AnchorSigner) -> "AnchorVerifier":
        return cls(s.public)

    @classmethod
    def from_env(cls, key_file: str | os.PathLike | None = None) -> "AnchorVerifier":
        secret = os.getenv("SA_ANCHOR_SECRET", "")
        if secret:
            return cls(secret=secret)
        path = Path(key_file or os.getenv("SA_ANCHOR_KEY_FILE") or _default_key_file())
        if not path.exists():
            raise AnchorError(f"anchor key {path} not found: cannot verify anchors (set SA_ANCHOR_KEY_FILE or SA_ANCHOR_SECRET)")
        data = json.loads(path.read_text())
        if "x" in data and "d" not in data:                      # public-only key file: enough to verify, useless for forging
            return cls({"alg": "EdDSA", "kid": "ed-" + hashlib.sha256(unb64(data["x"])).hexdigest()[:8], "x": data["x"]})
        return cls.from_signer(AnchorSigner.from_env(path))      # existing private key: derives the public half, never creates one

    def verify(self, rec: dict) -> bool:
        sig, alg = rec.get("sig", ""), rec.get("alg")
        try:
            if alg == "HS256":
                if not self.secret:
                    return False
                return hmac.compare_digest(sig, b64(hmac.new(self.secret.encode(), _canon(rec), hashlib.sha256).digest()))
            if alg == "EdDSA" and self.public.get("alg") == "EdDSA" and rec.get("kid") == self.public.get("kid"):
                from cryptography.exceptions import InvalidSignature
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
                try:
                    Ed25519PublicKey.from_public_bytes(unb64(self.public["x"])).verify(unb64(sig), _canon(rec))
                    return True
                except InvalidSignature:
                    return False
        except Exception:
            return False
        return False


class AnchorLog:
    """Append-only JSON-lines file. One O_APPEND write per record; the record chain (`prev`) detects edits to old lines."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    def lines(self) -> list[str]:
        if not self.path.exists():
            return []
        return [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def last_hash(self) -> str:
        ls = self.lines()
        return hashlib.sha256(ls[-1].encode()).hexdigest() if ls else GENESIS

    def append(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n").encode()
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)


def anchor_chain(name: str, audit: AuditLog, log: AnchorLog, signer: AnchorSigner, now: float | None = None) -> dict:
    """Commit the current head of `audit` to the anchor log. Refuses to anchor a chain that already fails verification."""
    ok, bad, _ = audit.verify()
    if not ok:
        raise AnchorError(f"refusing to anchor chain '{name}': it already fails verification at seq {bad}")
    seq, n, head = audit.tip()
    rec = {"v": 1, "chain": name, "ts": round(now if now is not None else time.time(), 3), "seq": seq, "entries": n, "head": head,
           "prev": log.last_hash(), "alg": signer.alg, "kid": signer.kid}
    rec["sig"] = signer.sign(_canon(rec))
    log.append(rec)
    return rec


def verify_anchors(name: str, audit: AuditLog, log: AnchorLog, verifier: AnchorVerifier) -> dict:
    """Compare `audit` against every anchor recorded for chain `name`.

    Returns {ok, anchors, problems[], truncated, latest_anchor}. Problems are specific: bad anchor-log link/signature, tail
    truncation (anchored seq beyond the chain end), or rewrite (anchored seq present but its hash differs).
    """
    problems: list[str] = []
    truncated = False
    prev, mine = GENESIS, []
    for i, line in enumerate(log.lines(), 1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"anchor log line {i}: not valid JSON (tampered)")
            prev = hashlib.sha256(line.encode()).hexdigest()
            continue
        if rec.get("prev") != prev:
            problems.append(f"anchor log line {i}: broken link to previous anchor (an anchor was edited or removed)")
        if not verifier.verify(rec):
            problems.append(f"anchor log line {i}: bad signature (forged or edited anchor)")
        prev = hashlib.sha256(line.encode()).hexdigest()
        if rec.get("chain") == name and verifier.verify(rec):
            mine.append(rec)
    ok, bad, _ = audit.verify()
    if not ok:
        problems.append(f"chain '{name}' fails integrity verification at seq {bad}")
    tip_seq = audit.tip()[0]
    last_seq = 0
    for rec in mine:
        seq = int(rec["seq"])
        if seq < last_seq:
            problems.append(f"anchor seq went backwards ({last_seq} -> {seq})")
        last_seq = max(last_seq, seq)
        if seq == 0:
            continue
        h = audit.hash_at(seq)
        if h is None:
            if seq > tip_seq:
                truncated = True
                problems.append(f"TAIL TRUNCATION: anchor at {rec['ts']} commits seq {seq} but chain ends at seq {tip_seq}")
            else:                                       # chain grew again after the cut (autoincrement skips the removed seq)
                truncated = True
                problems.append(f"TAIL TRUNCATION: anchored seq {seq} is missing from the chain although later rows exist (rows deleted, then appended over)")
        elif h != rec["head"]:
            problems.append(f"REWRITTEN: anchored seq {seq} has a different hash than the signed anchor")
    if not mine:
        problems.append(f"no valid anchors for chain '{name}' (run `supportagent audit anchor`; cannot prove no truncation)")
    return {"ok": not problems, "anchors": len(mine), "problems": problems, "truncated": truncated,
            "latest_anchor": mine[-1] if mine else None}


def _default_key_file() -> Path:
    from ..config import RUNS_DIR
    return Path(RUNS_DIR) / "anchor_key.json"
