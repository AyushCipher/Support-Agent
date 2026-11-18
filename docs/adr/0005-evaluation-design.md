# ADR-0005: Evaluation design

**Status:** accepted

**Decisions.**
- **Outcome-based success** (tau-bench style): final DB state vs ground truth + no harm + honest reply + approval expectation. Trajectory discipline and guard-block counts are *separate* metrics (F7).
- **Harm vs honesty** are separate: attack success counts only things an attacker profits from; false claims are tracked on their own.
- **Fresh database and state per trial**; 8 trials per scenario so **pass^k** (all-k-succeed) exposes unreliability that pass@1 hides (baseline 60.6% pass^1 but 30% pass^8).
- **Ablation ladder** (baseline → +prompt → +I/O guards → +tool guards → +HITL/routing) so each layer's contribution is measured, not asserted.
- **Simulated model for CI** with seeded failure modes: deterministic, free, and the eval code is tested like software (`test_every_scenario_passes_with_a_perfect_model` proves scenarios are solvable). **Never present those numbers as model quality.**
- **Judges are measured, not trusted**: rule judge by default; LLM judge behind the same interface; `calibrate.py` reports Cohen's κ against *human* labels (`make label`).
- **CI gate**: absolute floors (success, ASR, pass^4, cost, per-category) + a regression budget against a committed baseline.
- **Online evals**: no ground truth in production, so sample sessions and run reference-free checks (claim verification, leak scan, guard/approval/escalation rates).
- **When a score saturates, fix the dataset** (F8); keep "before" artifacts as evidence.

**Known gaps.** Judge calibration awaits human labels; attacker suite is fixed (no adaptive red-teaming); live-model numbers need `make eval-live`.
