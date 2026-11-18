# ADR-0007: Cryptographic, audience-bound agent identities

**Status:** accepted

**Context.** Static per-role tokens prove "some billing agent" called, but not *which session*, and they never expire.

**Decision.** The registry mints a JWT per tool call: `sub=agent:<role>`, `sid`, `aud=<server>`, `scope`, 60 s `exp`, unique `jti`. Servers verify signature/audience/expiry/scope, reject `jti` replays, strip any client-supplied identity and inject the verified one for audit attribution. Static tokens are dead in JWT mode.

**Consequences.** Every business-side audit row names the agent and session, so "who refunded this?" has a cryptographic answer; a token stolen from one server cannot be used on another or after a minute. Cost: one HMAC per call (microseconds).

**Trade-off.** HS256 with a shared secret keeps the runtime dependency-free, but any verifier can also mint. Production: RS256/EdDSA + JWKS (servers verify only), key rotation, and workload identity (mTLS/SPIFFE). `alg:none` and algorithm confusion are explicitly rejected and tested.
