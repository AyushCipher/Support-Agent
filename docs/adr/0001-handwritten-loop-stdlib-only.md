# ADR-0001: Hand-written agent loop; stdlib-only runtime

**Status:** accepted

**Context.** Frameworks (LangGraph, Agents SDKs) are productive, but they hide exactly the parts this project exists to demonstrate and test: budgets, retries, checkpoint boundaries, guard placement, approval pauses.

**Decision.** Write the loop (~400 lines, `harness/runtime.py`) with no agent framework and no runtime dependencies. LLM providers sit behind a 1-method protocol (`complete`); the Anthropic SDK is an optional extra.

**Consequences.**
- (+) Every decision point is a place we can put a guard, a span, a checkpoint and a test. Crash-resume and exactly-once refunds are properties we can *prove* in tests.
- (+) Runs anywhere (CI, air-gapped), starts in milliseconds; 2,000 eval trials in ~2 s.
- (+) The offline simulated model plugs into the same interface, so the real code path is what's evaluated.
- (−) We re-implement things frameworks give for free (message conversion, streaming is *not* implemented). For a team product, adopt a framework for the commodity parts and keep this harness's guard/approval/checkpoint contracts.

**Alternatives.** LangGraph (good checkpointing, but guard placement is implicit in graph edges); Temporal (right answer at scale — see ADR-0006 — overkill for one process).
