# ADR-0016: Age-based data retention with tombstones

**Status:** accepted. Written retroactively during the review; the code predates it.

**Context.** Transcripts, PII vaults, checkpoints, traces, approvals and request-dedup rows grow forever. Storage limitation needs per-class TTLs, but the audit chain must stay verifiable.

**Decision.** `supportagent retention run [--dry-run]` applies a per-class policy (purge, or archive-then-purge) to FINISHED sessions only, skips legal holds, uses optimistic concurrency (`WHERE updated_at=?`) so a session that moved on is skipped and retried next run, archives checkpoints through the privacy module's redaction whitelist, and appends one `retention_purge` tombstone per class to the audit chain. Audit retention is archive-only (the chain cannot be truncated safely yet). Age-based vault purge deletes the vault ciphertext and the session's profile email but never destroys keys: keys are per customer and shared by that customer's newer data. Delete-on-request (crypto-shredding, which now also deletes the customer's trace spans) stays in `security/privacy.py`.

**Consequences.**
- (+) Dry-run writes nothing (no archives, no tombstones, no deletes); every applied purge is provable.
- (-) The traces file is rewritten in place; a span appended between the tail copy and `os.replace` can be lost.
- (-) Spans exported to OTLP backends are outside this job.
