"""Reviewer desk + admin API behind real auth: static token (constant-time), OIDC-style JWT via JWKS, CSRF-safe approval POSTs,
and the public JWKS endpoint."""
import importlib
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from supportagent.security import adminauth
from supportagent.security.identity import _b64


@pytest.fixture
def desk(tmp_path, monkeypatch):
    monkeypatch.setenv("SA_RUNS_DIR", str(tmp_path))
    import supportagent.llm.scripted as sc
    monkeypatch.setitem(sc.PARAMS, "fast", {"diligence": 1.0, "gullibility": 0.0, "sloppiness": 0.0, "misroute": 0.0, "leak": 0.0})
    import supportagent.config as c
    importlib.reload(c)
    from supportagent.api import server as srvmod
    importlib.reload(srvmod)
    app = srvmod.App("full")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srvmod.make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", monkeypatch
    httpd.shutdown()


def call(base, path, body=None, token=None, headers=None, json_ct=True):
    h = dict(headers or {})
    if token:
        h["Authorization"] = f"Bearer {token}"
    if body is not None and json_ct:
        h.setdefault("Content-Type", "application/json")
    r = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None, headers=h,
                               method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def start_refund_approval(base):
    _, r = call(base, "/api/chat", {"customer_id": "C001", "message": "Refund ORD-1005 please, it's the espresso machine"})
    assert r["status"] == "awaiting_approval"
    return r


def test_reviewer_desk_is_disabled_until_configured_then_requires_the_token(desk):
    base, mp = desk
    mp.delenv("SA_ADMIN_TOKEN", raising=False)
    mp.delenv("SA_OIDC_JWKS", raising=False)
    assert call(base, "/api/approvals")[0] == 403                                      # previously open to anyone
    assert call(base, "/api/approvals/APR-x/decide", {"approve": True})[0] == 403
    mp.setenv("SA_ADMIN_TOKEN", "desk-token")
    assert call(base, "/api/approvals")[0] == 401                                      # no credential
    assert call(base, "/api/approvals", token="wrong")[0] == 401
    assert call(base, "/api/approvals", headers={"Authorization": "desk-token"})[0] == 401          # not a Bearer credential
    assert call(base, "/api/approvals", headers={"Authorization": "Basic ZGVzay10b2tlbg=="})[0] == 401
    assert call(base, "/api/approvals", token="desk-token")[0] == 200
    assert call(base, "/api/chat", {"customer_id": "C001", "message": "Where is ORD-1006?"})[0] == 200   # customer chat stays public


def test_static_token_uses_constant_time_compare(desk, monkeypatch):
    base, mp = desk
    mp.setenv("SA_ADMIN_TOKEN", "desk-token")
    seen = []
    real = adminauth.hmac.compare_digest
    monkeypatch.setattr(adminauth.hmac, "compare_digest", lambda a, b: seen.append((a, b)) or real(a, b))
    assert call(base, "/api/approvals", token="desk-token")[0] == 200
    assert seen and isinstance(seen[0][0], bytes)                                      # hmac.compare_digest, on bytes


def test_approval_decision_is_csrf_safe_attributed_and_once_only(desk):
    base, mp = desk
    mp.setenv("SA_ADMIN_TOKEN", "desk-token")
    r = start_refund_approval(base)
    aid = call(base, "/api/approvals", token="desk-token")[1]["pending"][0]["id"]
    path = f"/api/approvals/{aid}/decide"
    # cross-site form post: no bearer header can be attached and a form cannot send application/json
    assert call(base, path, {"approve": True})[0] == 401
    assert call(base, path, {"approve": True}, token="desk-token", json_ct=False,
                headers={"Content-Type": "application/x-www-form-urlencoded"})[0] == 415
    # browser-initiated cross-origin request is refused even with a stolen token
    host = base.split("//")[1]
    assert call(base, path, {"approve": True}, token="desk-token", headers={"Origin": "https://evil.example"})[0] == 403
    assert call(base, "/api/admin/kill", {"on": True}, token="desk-token", headers={"Origin": "https://evil.example"})[0] == 403
    assert call(base, "/api/approvals", token="desk-token")[1]["pending"], "nothing was decided by the refused requests"
    code, d = call(base, path, {"approve": True, "reviewer": "forged-name"}, token="desk-token", headers={"Origin": f"http://{host}"})
    assert code == 200 and d["status"] == "done" and d["approval"]["reviewer"] == "admin"      # identity from auth, not from the body
    assert call(base, path, {"approve": True}, token="desk-token")[0] == 409                    # no double-resume
    assert call(base, "/api/approvals/APR-nope/decide", {"approve": True}, token="desk-token")[0] == 404
    assert r["session_id"]


def test_approvals_ui_has_sign_in_and_sends_header_not_cookie(desk):
    base, _ = desk
    html = urllib.request.urlopen(base + "/approvals").read().decode()
    assert "Authorization" in html and "sessionStorage" in html and "document.cookie" not in html and "reviewer:'desk'" not in html


def test_jwks_endpoint_serves_public_keys_only(desk):
    pytest.importorskip("cryptography")
    base, _ = desk
    code, jwks = call(base, "/.well-known/jwks.json")
    assert code == 200 and jwks["keys"] and jwks["keys"][0]["kty"] == "OKP" and jwks["keys"][0]["kid"]
    assert not ({"d", "p", "q", "k"} & set(jwks["keys"][0]))                           # no private material


# ---- OIDC-style bearer tokens through the same JWKS machinery
@pytest.fixture
def oidc(tmp_path, desk):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    base, mp = desk
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    n = priv.public_key().public_numbers()
    b = lambda i: _b64(i.to_bytes((i.bit_length() + 7) // 8, "big"))                  # noqa: E731
    jwks = tmp_path / "idp-jwks.json"
    jwks.write_text(json.dumps({"keys": [{"kty": "RSA", "kid": "idp-1", "alg": "RS256", "n": b(n.n), "e": b(n.e)}]}))
    mp.delenv("SA_ADMIN_TOKEN", raising=False)
    mp.setenv("SA_OIDC_JWKS", str(jwks))
    mp.setenv("SA_OIDC_ISSUER", "https://idp.example")
    mp.setenv("SA_OIDC_AUDIENCE", "support-desk")
    import time as t

    def mint(**over):
        now = int(t.time())
        claims = {"iss": "https://idp.example", "aud": "support-desk", "sub": "alice@corp", "iat": now, "exp": now + 600, "roles": ["reviewer"], **over}
        h = _b64(json.dumps({"alg": "RS256", "kid": "idp-1"}).encode())
        c = _b64(json.dumps(claims).encode())
        return f"{h}.{c}.{_b64(priv.sign(f'{h}.{c}'.encode(), padding.PKCS1v15(), hashes.SHA256()))}"
    adminauth._OIDC_CACHE.clear()
    return base, mp, mint


def test_oidc_bearer_is_accepted_and_names_the_human_reviewer(oidc):
    base, mp, mint = oidc
    assert call(base, "/api/approvals", token=mint())[0] == 200
    start_refund_approval(base)
    tok = mint()
    aid = call(base, "/api/approvals", token=tok)[1]["pending"][0]["id"]
    code, d = call(base, f"/api/approvals/{aid}/decide", {"approve": False, "note": "no"}, token=tok)
    assert code == 200 and d["approval"]["reviewer"] == "oidc:alice@corp" and d["approval"]["status"] == "denied"
    assert call(base, "/api/admin/flags", token=tok)[0] == 200                         # same auth guards the admin API


def test_oidc_rejects_bad_tokens(oidc):
    base, mp, mint = oidc
    bad = {
        "wrong audience": mint(aud="other-app"),
        "wrong issuer": mint(iss="https://evil.example"),
        "expired": mint(exp=1000, iat=900),
        "tampered": mint().rsplit(".", 1)[0] + "." + _b64(b"x" * 256),
        "hs256 confusion": _b64(b'{"alg":"HS256","kid":"idp-1"}') + "." + mint().split(".")[1] + "." + _b64(b"sig"),
        "alg none": _b64(b'{"alg":"none"}') + "." + mint().split(".")[1] + ".",
    }
    for name, tok in bad.items():
        assert call(base, "/api/approvals", token=tok)[0] == 401, name
    mp.setenv("SA_OIDC_ROLE", "reviewer")
    assert call(base, "/api/approvals", token=mint())[0] == 200
    assert call(base, "/api/approvals", token=mint(roles=["intern"]))[0] == 403         # authenticated but not a reviewer
    assert call(base, "/api/approvals")[0] == 401


def test_static_token_and_oidc_can_coexist(oidc):
    base, mp, mint = oidc
    mp.setenv("SA_ADMIN_TOKEN", "break-glass")
    assert call(base, "/api/approvals", token="break-glass")[0] == 200 and call(base, "/api/approvals", token=mint())[0] == 200
    assert call(base, "/api/approvals", token="nope")[0] == 401
