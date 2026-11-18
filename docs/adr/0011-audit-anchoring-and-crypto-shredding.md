# ADR-0011: Audit head anchoring, PII encryption at rest, and erasure by crypto-shredding

**Status:** accepted.

## Part 1 - Anchoring the audit chain

**Context.** The SHA-256 hash chain detects edits and mid-chain deletions but not deletion of the *last N rows*: the remainder is a valid shorter chain.

**Decision.** `supportagent audit anchor` appends one record per chain to an append-only JSON-lines anchor log (`SA_ANCHOR_FILE`, default `runs/audit_anchors.jsonl`): `{chain, ts, seq, entries, head, prev, alg, kid, sig}`. `prev` is the hash of the previous anchor line (edits or removals of middle anchors are detected); `sig` is Ed25519 (`SA_ANCHOR_KEY_FILE`, needs `[crypto]`) or HMAC-SHA256 (`SA_ANCHOR_SECRET`, stdlib only). `audit verify --anchors` checks: anchor-log links and signatures; every anchored `(seq, head)` still exists with the same hash (missing means **tail truncation**; different hash means **rewrite**, including a fully recomputed, internally consistent chain); no anchors is a failure, not a pass. Exit code 2 on any failure so cron/CI can alert. Anchoring refuses a chain that already fails verification.

**Honest limit.** A local file is append-only by convention. An attacker with host access can truncate the anchor log's tail or steal the signing key. Anchoring only *closes* the gap if anchors are shipped to storage the attacker cannot rewrite and the verifier's public key is held elsewhere. Run `audit anchor` on a schedule (for example every 5 minutes) and ship each new line; recipes are in `docs/compliance.md` ("Shipping anchors to write-once storage"). Rows written after the last shipped anchor are still at risk.

## Part 2 - Encryption at rest

**Decision.** `SA_VAULT_KEY` (any string; use 32+ random bytes) goes through scrypt (N=2^15) with a random per-deployment salt to give a master key (KEK). One random 256-bit data key (DEK) per data subject is wrapped by the KEK (AES-256-GCM) in `<SA_KEY_DIR | runs/keys>/<subject>.key`. Every PII-vault value and every long-term-memory value is sealed with AES-256-GCM under the subject's DEK, with its location (`vault|<token>`, `memory|<customer>|<key>`) as AAD, so ciphertexts cannot be swapped between slots and tampering fails closed. The vault's reverse index stores a keyed hash (blind index) instead of the value. Because the vault lives in the session state, the checkpoint is encrypted wherever it is stored. `SA_ENV=production` without `SA_VAULT_KEY` raises at startup (fail closed); without the `[crypto]` extra a configured key raises `CryptoUnavailable` with the install hint. With no key in dev, behavior is unchanged (plaintext).

## Part 3 - Delete on request

**Decision.** `supportagent privacy delete --customer C001` (`security/privacy.py`):
1. **crypto-shred**: delete the customer's DEKs (memory key plus the vault key recorded in each of their sessions). Every copy of the ciphertext in backups, WAL files and replicas becomes unreadable, even to someone holding the master key;
2. redact every retained transcript (message text and tool-call args; all non-whitelisted session fields are blanked, so fields added later are erased by default);
3. delete memory rows; redact and deny that customer's approval records;
4. append a `privacy_erasure` **tombstone** to the audit chain (and the business chain) with counts and no personal data. The chain is never rewritten, so `audit verify` and previously taken anchors still pass.

**Deliberately retained** (for the DPO to confirm the legal basis): shop business records (orders, refunds, tickets) and the pseudonymous customer id on audit rows (proves the erasure; fraud/finance audit). If policy requires the id to go as well, move it behind a mapping table outside the chain so chain rows hold only an opaque reference.

**Not solved.** `Session.customer_email` is still plaintext in the checkpoint JSON while the session lives (needs a harness change to seal it; it is blanked on erasure). Trace files (`traces.jsonl`) hold tokenized text but are not purged. Key files must live outside the backed-up volume (or their backups must honour deletion), otherwise shredding is not final. A running server may hold a DEK in memory for up to the cache TTL (5 s). No consent records, retention TTLs, or data-subject access/export flow. Detection of PII is still regex/field based.
