# ADR-0017: Session-bound customer tokens and a production startup gate

**Status:** accepted

**Context.** The threat model listed customer authentication as out of scope: `/api/chat` trusted `customer_id` and accepted any `session_id`, so knowing a session id was enough to act inside another customer's conversation (with that customer's authorization) and to read it through `/api/sessions/<id>`. Separately, nothing stopped a deploy from running with the offline simulator, plaintext PII or unauthenticated reviewers, and `SA_BACKEND=gemini` silently fell back to the simulator in the API (review finding R1).

**Decision.**
1. A new session returns `session_token = HMAC-SHA256(SA_SESSION_SECRET, "v1|<session>|<customer>")`. Continuing a session and reading it require that token (`X-Session-Token` header, or `session_token` in the chat body). The session's own customer id is bound into the MAC and must match any `customer_id` in the request. Missing, wrong and foreign tokens all return the same 404. Reviewers and admins read any transcript with their bearer credential. Stateless: there is no token table, and rotating the secret revokes every token.
2. Optional customer identity at session creation: an OIDC JWT (`SA_CUSTOMER_OIDC_JWKS`, `_ISSUER`, `_AUDIENCE`) whose `customer_id` (or `sub`) claim must equal the requested customer.
3. `security/prodcheck.py` refuses to start with `SA_ENV=production` unless: live backend with its key and matching model names; full preset; no anonymous MCP; `SA_VAULT_KEY` and `cryptography`; `SA_SESSION_SECRET`; reviewer auth; EdDSA agent identities; and customer OIDC or the explicit `SA_PUBLIC_DEMO=1` opt-in. Messages name variables, never values.

**Consequences.**
- (+) Session ids are no longer capabilities; session hijack is closed at the API.
- (+) Insecure production configs fail fast, with every problem listed at once.
- (-) Without customer OIDC the demo still trusts the customer picker; that is now an explicit, named opt-in.
- (-) Tokens are bearer tokens that expire only on secret rotation; serve over HTTPS only (both platform configs force it).
