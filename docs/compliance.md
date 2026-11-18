# Compliance notes (not a certification)

This repository demonstrates **controls**; it is not SOC 2 / GDPR / HIPAA certified and nothing here is legal advice. The table maps common control objectives to what exists and what an auditor would still ask for.

| Objective | Control in this repo | Evidence | Still needed in a real deployment |
|---|---|---|---|
| Access control, least privilege | Per-role tool allowlists + per-role token scopes enforced server-side; hidden params; third-party tools unreachable until granted | `tests/test_mcp.py`, `tests/test_security.py` | Reviewer RBAC beyond one optional role, access reviews. SSO: the admin API and reviewer desk accept OIDC bearer JWTs via JWKS (`SA_OIDC_*`) but this is not tested against a real IdP |
| Individual accountability | Agent identity (JWT `sub`/`sid`) + customer id on every audit row | audit attribution tests | Reviewer decisions are attributed to the authenticated principal (`admin` or `oidc:<sub>`); flag changes through the admin API are still labelled `admin-api` |
| Integrity of logs | SHA-256 hash chain; `audit verify` | tamper tests | Scheduled anchoring plus shipping to write-once storage (below), retention policy |
| Data minimization toward processors | PII tokenization before any LLM call; cards/SSN hard-redacted; checkpoints tokenized | PII egress metric 0% in final config | DPA with the LLM vendor, regional routing, KMS-held master key (vault and memory are AES-256-GCM under `SA_VAULT_KEY`; ADR-0011) |
| Change management | Content-hashed prompts recorded per session; eval-gated promotion; canary + shadow; rollback = one flag | promote/canary/shadow tests | Approval workflow for promotions |
| Incident response | Kill switches, replayable sessions, runbook | kill-switch tests | On-call, alerting, post-incident template |
| Data subject rights | `supportagent privacy delete --customer <id>`: crypto-shreds the customer's keys (vault, memory), redacts retained transcripts and approval records, appends a `privacy_erasure` tombstone; audit chain and earlier anchors still verify (ADR-0011) | `tests/test_privacy.py` (deleted data unrecoverable from a pre-erasure backup, audit verify passes) | Access/export request flow, retention TTLs, consent records; DPO sign-off on retained items: shop business records and the pseudonymous customer id on audit rows; trace files; keep key files out of backups |

## Shipping anchors to write-once storage

`supportagent audit anchor` appends one signed line per chain to `SA_ANCHOR_FILE` (default `runs/audit_anchors.jsonl`). The local file alone does not stop an attacker who owns the host, so run it on a schedule and push every new line off-box to storage that cannot be rewritten. Keep the verification key (the Ed25519 public key, or the HMAC secret) somewhere other than the host being audited.

```sh
# cron, every 5 minutes: anchor, then ship the newest line(s)
*/5 * * * *  cd /srv/supportagent && supportagent audit anchor && ./ship_anchor.sh
```

S3 Object Lock (bucket created with `--object-lock-enabled-for-bucket`; COMPLIANCE-mode retention means nobody, including root, can delete or shorten it):

```sh
# ship_anchor.sh - one immutable object per anchor line (runtime + business chain = last 2 lines)
tail -n 2 runs/audit_anchors.jsonl | while read -r LINE; do
  KEY="anchors/$(date -u +%Y/%m/%d/%H%M%S)-$(echo "$LINE" | sha256sum | cut -c1-8).json"
  echo "$LINE" | aws s3api put-object --bucket my-audit-anchors --key "$KEY" --body /dev/stdin \
    --object-lock-mode COMPLIANCE --object-lock-retain-until-date "$(date -u -d '+7 years' +%Y-%m-%dT%H:%M:%SZ)"
done
```

Equivalent options: Azure immutable blob storage (locked time-based retention policy), GCS bucket lock with a retention policy, an append-only transparency log (Rekor, Trillian), a second host that only accepts appends, or a mailbox the operator cannot edit. Local hardening that helps but is not a substitute: `chattr +a runs/audit_anchors.jsonl` (Linux; clearing it needs root) and running the anchor job as a different OS user.

To audit from a clean machine: download the objects, concatenate them in key order into one file, take the live `state.db` / `shop.db`, and run `SA_ANCHOR_FILE=<concatenated file> SA_ANCHOR_KEY_FILE=<public-key file> supportagent audit verify --anchors`, where the public-key file holds only `{"x": "<public key from the signing key file>"}` (no private half needed to verify). Exit code 2 with `TAIL TRUNCATION` or `REWRITTEN` lines means the live chain no longer contains what was anchored. Rows written after the last shipped anchor are not protected, so shorten the interval to shrink that window.
