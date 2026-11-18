# ADR-0014: Parallel read tools and a TTL read cache

**Status:** accepted. Written retroactively during the review; the code predates it.

**Context.** Models often request several independent lookups in one response. Running them serially adds latency; repeated identical reads within and across sessions cost MCP round trips.

**Decision.** Consecutive READ tool calls (first-party, `risk == "read"`, allowed for the role) run concurrently in `ParallelFetcher` (bounded workers, per-call timeout, circuit-breaker aware) and are recorded in the original order, so transcripts stay deterministic. Writes are barriers and always run alone. `ToolCache` caches successful first-party reads for `tool_cache_ttl_s` (default 30 s), keyed by (tool, canonical args, customer). Any write attempt for the customer (success or failure, including harness-side escalations, finding R9) invalidates that customer's entries and bumps a per-customer generation; a read that was in flight across an invalidation is not stored (finding R3). Values are stored as JSON text and re-sanitized and re-tokenized on every serve. Money decisions never read the cache (ToolGuard and the verifier call the registry directly). Exactly-once for money comes from harness-derived idempotency keys enforced server-side under `BEGIN IMMEDIATE` with a UNIQUE key, not from the cache.

**Consequences.**
- (+) Lower latency on multi-lookup turns, fewer MCP calls.
- (-) Writes made by OTHER processes are only bounded by the TTL: a stale entry can mislead the model's wording for up to 30 s, never a money decision.
- (-) Cached values hold raw (pre-tokenization) data in process memory for the TTL.
- (-) A timed-out read thread cannot be killed in Python; it finishes in the background (harmless for reads, which is why only reads are parallelised).
