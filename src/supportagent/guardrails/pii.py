"""Reversible PII tokenization: scrub PII before it reaches an external LLM provider, restore it only where it is safe.

  ingress  (customer text, tool results)  -> <EMAIL_1>, <PHONE_1>, <ADDR_1>, <NAME_1>   (model, traces, checkpoints see tokens)
  egress   (final reply to the customer)  -> restore ONLY tokens proven to belong to the authenticated customer
  tool args produced by the model         -> restore (internal systems need real values)

Ownership: a value is "own" if it came from the customer's own message, equals their profile email, or sits in the same
record as their profile email. Tokens for other customers' data are never restored to the customer - a second line of
defense behind server-side authorization (see tests/test_pii.py::test_foreign_pii_never_restored_to_customer).
Cards/SSNs are hard-redacted upstream (never restorable). The vault lives in the session state and is checkpointed.

Encryption at rest (ADR-0011): when SA_VAULT_KEY is set (or a `protector` is passed) every vault value is sealed with
AES-256-GCM under a per-subject data key, and the reverse index stores a keyed hash instead of the value, so the persisted
checkpoint contains no raw PII. In production mode (SA_ENV=production) a missing key is a hard error (fail closed). Deleting the
subject's key (`supportagent privacy delete`) makes every copy of the ciphertext - backups, WAL, replicas - unrecoverable.
Without a key (dev) behavior is unchanged: plaintext values in session state.
"""
from __future__ import annotations

import re
from typing import Any

from ..security.crypto import KeyShredded, Protector, default_protector, subject_for
from .patterns import EMAIL_RE, PHONE_RE

TOKEN_RE = re.compile(r"<(EMAIL|PHONE|ADDR|NAME)_(\d+)>")
FIELD_KIND = {"customer_email": "EMAIL", "email": "EMAIL", "ship_to": "ADDR", "address": "ADDR", "name": "NAME",
              "customer_name": "NAME", "phone": "PHONE"}
FREE_TEXT_KEYS = {"notes", "body", "summary", "subject", "message", "text"}


def new_vault() -> dict:
    return {"map": {}, "rev": {}, "own": [], "n": {}}


ERASED = "[erased]"


class PIIVault:
    def __init__(self, state: dict, own_email: str = "", customer_id: str = "", protector: Protector | None = None):
        for k, v in new_vault().items():
            state.setdefault(k, v)
        self.s, self.own_email = state, (own_email or "").lower()
        self.prot = protector if protector is not None else default_protector()      # None = plaintext dev mode; raises in prod w/o key
        if self.prot is not None and not state.get("subj"):
            state["subj"] = subject_for(customer_id, own_email)                      # key-store subject, recorded for crypto-shredding
        self.subj = state.get("subj", "")

    def _store(self, tok: str, value: str) -> str:
        return self.prot.seal(self.subj, value, f"vault|{tok}") if self.prot else value

    def _plain(self, tok: str) -> str | None:
        """Plaintext for a token, or None if unknown / its key was shredded."""
        v = self.s["map"].get(tok)
        if v is None or self.prot is None or not v.startswith("enc:v1:"):
            return v
        try:
            return self.prot.open(self.subj, v, f"vault|{tok}")
        except KeyShredded:
            return None

    def _rev_key(self, kind: str, value: str) -> str:
        return f"{kind}|h:{self.prot.blind_index(self.subj, value)}" if self.prot else f"{kind}|{value}"

    def token_for(self, kind: str, value: str, own: bool = False) -> str:
        rk = self._rev_key(kind, value)
        tok = self.s["rev"].get(rk)
        if tok is None:
            n = self.s["n"][kind] = self.s["n"].get(kind, 0) + 1
            tok = f"<{kind}_{n}>"
            self.s["map"][tok] = self._store(tok, value)
            self.s["rev"][rk] = tok
        if (own or (kind == "EMAIL" and value.lower() == self.own_email)) and tok not in self.s["own"]:
            self.s["own"].append(tok)
        return tok

    def tokenize_text(self, text: str, own: bool = False, phone: bool = True) -> tuple[str, list[str]]:
        kinds: list[str] = []

        def em(m: re.Match) -> str:
            kinds.append("email")
            return self.token_for("EMAIL", m.group(0), own)

        text = EMAIL_RE.sub(em, text)
        if phone:
            def ph(m: re.Match) -> str:
                kinds.append("phone")
                return self.token_for("PHONE", m.group(0).strip(), own)
            text = PHONE_RE.sub(ph, text)
        return text, kinds

    def tokenize_json(self, node: Any, key: str = "", record_own: bool = False) -> Any:
        if isinstance(node, dict):
            email = str(node.get("customer_email") or node.get("email") or "").lower()
            own_rec = record_own or (bool(email) and email == self.own_email)
            return {k: self.tokenize_json(v, k, own_rec) for k, v in node.items()}
        if isinstance(node, list):
            return [self.tokenize_json(x, key, record_own) for x in node]
        if isinstance(node, str) and node:
            kind = FIELD_KIND.get(key)
            if kind:
                return self.token_for(kind, node, record_own)
            return self.tokenize_text(node, own=record_own, phone=key in FREE_TEXT_KEYS)[0]
        return node

    def detokenize_own(self, text: str) -> tuple[str, list[str]]:
        unresolved: list[str] = []

        def sub(m: re.Match) -> str:
            tok = m.group(0)
            if tok in self.s["own"]:
                plain = self._plain(tok)
                if plain is not None:
                    return plain
            unresolved.append(tok)
            return tok
        return TOKEN_RE.sub(sub, text), unresolved

    def detokenize_value(self, v: Any) -> Any:
        if isinstance(v, str):
            return TOKEN_RE.sub(lambda m: self._plain(m.group(0)) or m.group(0), v)
        if isinstance(v, dict):
            return {k: self.detokenize_value(x) for k, x in v.items()}
        if isinstance(v, list):
            return [self.detokenize_value(x) for x in v]
        return v
