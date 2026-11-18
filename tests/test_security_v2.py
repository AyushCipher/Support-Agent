"""Asymmetric agent identities (EdDSA + JWKS, rotation, verify-only servers). Requires the optional [crypto] extra."""
import hashlib
import hmac
import json

import pytest

pytest.importorskip("cryptography")

from supportagent.mcp.client import build_registry                                   # noqa: E402
from supportagent.security import asymmetric as asym                                  # noqa: E402
from supportagent.security.asymmetric import IdentityIssuer, JwksVerifier, KeyRing, make_identity   # noqa: E402
from supportagent.security.crypto import CryptoUnavailable                            # noqa: E402
from supportagent.security.identity import TokenError, _b64, _unb64                   # noqa: E402


def parts(tok):
    h, c, s = tok.split(".")
    return json.loads(_unb64(h)), json.loads(_unb64(c)), s


def resign_payload(tok, **changes):
    h, c, s = tok.split(".")
    claims = {**json.loads(_unb64(c)), **changes}
    return f"{h}.{_b64(json.dumps(claims).encode())}.{s}"


@pytest.fixture
def iss():
    return IdentityIssuer(KeyRing(overlap_s=120))


def test_eddsa_token_carries_kid_and_claims_and_verifies_with_public_key_only(iss):
    tok = iss.issue("billing", "S1", ["orders:read"], "orders", now=1000)
    h, c, _ = parts(tok)
    assert h["alg"] == "EdDSA" and h["kid"] == iss.keyring.active["kid"]
    assert c["sub"] == "agent:billing" and c["aud"] == "orders" and c["exp"] - c["iat"] == 60 and c["jti"]
    v = JwksVerifier(iss.jwks())                                                      # a verifier built from the public JWKS alone
    assert v.verify(tok, "orders", now=1010)["sid"] == "S1"
    assert not hasattr(v, "issue") and v.can_mint is False
    assert all(set(k) <= {"kty", "crv", "use", "alg", "kid", "x", "x-verify-until"} for k in iss.jwks()["keys"])   # no private 'd'


def test_tampered_token_rejected(iss):
    v = iss.verifier()
    tok = iss.issue("orders", "S1", ["orders:read"], "orders")
    with pytest.raises(TokenError, match="bad signature"):
        v.verify(resign_payload(tok, scope="refunds:issue"), "orders")                # privilege escalation edit
    h, c, s = tok.split(".")
    flipped = _b64(bytes([_unb64(s)[0] ^ 1]) + _unb64(s)[1:])
    with pytest.raises(TokenError, match="bad signature"):
        v.verify(f"{h}.{c}.{flipped}", "orders")
    with pytest.raises(TokenError, match="malformed"):
        v.verify("garbage", "orders")
    with pytest.raises(TokenError, match="bad signature|malformed"):
        v.verify(f"{h}.{c}.!!!", "orders")


def test_wrong_audience_expired_and_ttl_ceiling(iss):
    v = iss.verifier()
    t = iss.issue("orders", "S1", [], "orders", ttl=60, now=1000)
    assert v.verify(t, "orders", now=1059)
    with pytest.raises(TokenError, match="expired"):
        v.verify(t, "orders", now=1060)
    with pytest.raises(TokenError, match="wrong audience"):
        v.verify(iss.issue("orders", "S1", [], "orders"), "kb")
    with pytest.raises(TokenError, match="ttl too long"):                              # even a validly signed 1h token is refused
        v.verify(iss.issue("orders", "S1", [], "orders", ttl=3600), "orders")


def test_algorithm_confusion_and_none_are_rejected(iss):
    v = iss.verifier()
    pub_raw = _unb64(iss.jwks()["keys"][0]["x"])
    h = _b64(json.dumps({"alg": "HS256", "typ": "JWT", "kid": iss.keyring.active["kid"]}).encode())
    c = _b64(json.dumps({"iss": "supportagent", "aud": "orders", "exp": 9999999999, "iat": 1, "jti": "x", "scope": "refunds:issue"}).encode())
    hs = _b64(hmac.new(pub_raw, f"{h}.{c}".encode(), hashlib.sha256).digest())       # classic: sign HS256 with the PUBLIC key
    with pytest.raises(TokenError, match="unsupported alg"):
        v.verify(f"{h}.{c}.{hs}", "orders", now=5)
    hn = _b64(json.dumps({"alg": "none", "kid": "x"}).encode())
    with pytest.raises(TokenError, match="unsupported alg"):
        v.verify(f"{hn}.{c}.", "orders", now=5)
    h_nokid = _b64(json.dumps({"alg": "EdDSA"}).encode())
    with pytest.raises(TokenError, match="missing kid"):
        v.verify(f"{h_nokid}.{c}.AAAA", "orders", now=5)


def test_unknown_key_and_forged_signature_with_stolen_kid(iss):
    v = iss.verifier()
    other = IdentityIssuer(KeyRing())
    with pytest.raises(TokenError, match="unknown kid"):
        v.verify(other.issue("billing", "S", ["refunds:issue"], "orders"), "orders")  # attacker's own key pair
    forged = other.issue("billing", "S", ["refunds:issue"], "orders")
    h, c, s = forged.split(".")
    h2 = _b64(json.dumps({"alg": "EdDSA", "typ": "JWT", "kid": iss.keyring.active["kid"]}).encode())   # claim the real kid
    with pytest.raises(TokenError, match="bad signature"):
        v.verify(f"{h2}.{c}.{s}", "orders")


def test_rotation_overlap_then_retirement():
    ring = KeyRing(overlap_s=120)
    iss = IdentityIssuer(ring)
    old_kid = ring.active["kid"]
    old = iss.issue("orders", "S1", ["orders:read"], "orders", ttl=300, now=1000)      # long-ish token, minted before rotation
    new_kid = iss.rotate(now=1010)
    assert new_kid != old_kid and ring.active["kid"] == new_kid
    new = iss.issue("orders", "S1", ["orders:read"], "orders", ttl=300, now=1020)
    assert parts(new)[0]["kid"] == new_kid
    snapshot_during = JwksVerifier(ring.jwks(now=1050))                               # JWKS served during the overlap
    assert {k["kid"] for k in snapshot_during._fetch()["keys"]} == {old_kid, new_kid}
    assert snapshot_during.verify(old, "orders", now=1050)["sid"] == "S1"             # rotated key still verifies in the window
    assert snapshot_during.verify(new, "orders", now=1050)
    with pytest.raises(TokenError, match="key retired"):                              # stale JWKS copy: deadline still enforced
        snapshot_during.verify(old, "orders", now=1010 + 120)
    assert snapshot_during.verify(new, "orders", now=1130)                            # new key unaffected
    after = JwksVerifier(ring.jwks(now=1200))                                         # JWKS published after the window
    assert [k["kid"] for k in after._fetch()["keys"]] == [new_kid]
    with pytest.raises(TokenError, match="unknown kid"):
        after.verify(old, "orders", now=1200)                                         # retired key rejected


def test_live_verifier_follows_rotation_via_refetch(iss):
    v = iss.verifier()                                                                # in-process: JWKS fetched via callable
    before = iss.issue("orders", "S", [], "orders")
    assert v.verify(before, "orders")
    iss.rotate()
    after = iss.issue("orders", "S", [], "orders")
    assert v.verify(after, "orders") and v.verify(before, "orders")                  # unknown kid triggers a re-fetch; old key in overlap


def test_keyring_persists_and_reloads(tmp_path):
    p = tmp_path / "ring.json"
    a = IdentityIssuer(KeyRing(path=p))
    tok = a.issue("orders", "S", [], "orders")
    b = IdentityIssuer(KeyRing(path=p))                                               # restart: same keys
    assert b.verify(tok, "orders") and b.keyring.active["kid"] == a.keyring.active["kid"]
    if hasattr(__import__("os"), "chmod") and __import__("os").name == "posix":
        assert oct(p.stat().st_mode & 0o777) == "0o600"


def test_registry_default_is_asymmetric_and_servers_cannot_mint(db):
    reg = build_registry(db)
    assert reg.identity.mode == "eddsa"
    srv = reg.clients["orders"].transport.server
    assert srv.identity.can_mint is False and not hasattr(srv.identity, "issue")     # server holds public keys only
    call = lambda tok, args: srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "get_order", "arguments": args, "_meta": {"token": tok}}})["result"]   # noqa: E731
    a = {"order_id": "ORD-1001", "acting_customer_id": "C001"}
    good = reg.identity.issue("orders", "S9", ["orders:read"], "orders")
    assert not call(good, a)["isError"]
    assert "replay" in call(good, a)["content"][0]["text"]                            # replay protection preserved
    assert "wrong audience" in call(reg.identity.issue("orders", "S9", ["orders:read"], "kb"), a)["content"][0]["text"]
    attacker = IdentityIssuer(KeyRing())                                              # a compromised server cannot invent credentials
    assert "unknown kid" in call(attacker.issue("billing", "S9", ["refunds:issue"], "orders"), a)["content"][0]["text"]
    assert "invalid identity" in call(resign_payload(good, scope="refunds:issue", jti="fresh-jti"), a)["content"][0]["text"]


def test_stdio_servers_receive_only_public_keys(db):
    ident = make_identity()
    env = ident.server_env()
    assert set(env) == {"MCP_JWKS"} and "d" not in json.loads(env["MCP_JWKS"])["keys"][0]


def test_hs256_is_an_explicit_dev_fallback(monkeypatch):
    b = make_identity(alg="hs256")
    assert b.mode == "hs256" and b.issuer.verify(b.issuer.issue("orders", "S", [], "orders"), "orders")
    assert make_identity(jwt_secret="shared").mode == "hs256"                         # explicit secret = explicit opt-in
    monkeypatch.setenv("SA_ENV", "production")
    with pytest.raises(CryptoUnavailable, match="production"):
        make_identity(alg="hs256")
    with pytest.raises(ValueError):
        make_identity(alg="rs512")


def test_missing_cryptography_gives_a_clear_error_or_dev_fallback(monkeypatch, capsys):
    import supportagent.security.crypto as cr
    monkeypatch.setattr(cr, "crypto_available", lambda: False)
    monkeypatch.setattr(asym, "crypto_available", lambda: False)
    with pytest.raises(CryptoUnavailable, match=r"supportagent\[crypto\]"):
        make_identity(alg="eddsa")
    dev = make_identity()                                                             # auto + dev mode: loud HS256 fallback
    assert dev.mode == "hs256" and "WARNING" in capsys.readouterr().err
    monkeypatch.setenv("SA_ENV", "production")
    with pytest.raises(CryptoUnavailable):
        make_identity()                                                               # production never silently downgrades


def test_rs256_external_tokens_verify_via_same_machinery():
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    nums = priv.public_key().public_numbers()
    i2b = lambda n: _b64(n.to_bytes((n.bit_length() + 7) // 8, "big"))               # noqa: E731
    jwks = {"keys": [{"kty": "RSA", "kid": "idp-1", "n": i2b(nums.n), "e": i2b(nums.e), "alg": "RS256"}]}
    h = _b64(json.dumps({"alg": "RS256", "kid": "idp-1"}).encode())
    c = _b64(json.dumps({"iss": "https://idp.example", "aud": "support-desk", "sub": "alice", "exp": 2000, "iat": 1000}).encode())
    sig = _b64(priv.sign(f"{h}.{c}".encode(), padding.PKCS1v15(), hashes.SHA256()))
    v = JwksVerifier(jwks, issuer="https://idp.example", allowed_algs=("RS256",))
    assert v.verify(f"{h}.{c}.{sig}", "support-desk", now=1500, max_ttl=None)["sub"] == "alice"
    with pytest.raises(TokenError, match="bad issuer"):
        JwksVerifier(jwks, issuer="https://evil", allowed_algs=("RS256",)).verify(f"{h}.{c}.{sig}", "support-desk", now=1500, max_ttl=None)
    with pytest.raises(TokenError, match="unsupported alg"):
        JwksVerifier(jwks, issuer="https://idp.example").verify(f"{h}.{c}.{sig}", "support-desk", now=1500, max_ttl=None)   # EdDSA-only verifier


def test_replay_guard_is_bounded_by_ttl():
    g = asym.ReplayGuard()
    assert g.check_and_add("a", 100, now=10) and not g.check_and_add("a", 100, now=11)
    assert g.check_and_add("a", 200, now=101)                                         # expired entries are forgotten
