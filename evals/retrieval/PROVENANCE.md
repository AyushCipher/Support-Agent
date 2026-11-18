# Provenance of `questions.jsonl`

**Authored by Claude (an LLM). Not human-labeled. Not human-reviewed.**

- Each line is `{id, question, gold, category}`. `gold` is the list of policy sections (`<file>#<section slug>`) that a correct answer must draw on; an empty list means the corpus does not answer the question (the system should abstain).
- Gold labels were assigned by the same model that wrote the questions, by reading the policy text in `policies/`. They have not been checked by a person. Some labels are arguable (for example, a "delivered but missing" question could also cite the lost-package refund rule); treat small metric differences with suspicion.
- Categories: `direct` (strong word overlap), `paraphrase`, `vocab_mismatch` (little shared wording), `typo` (misspellings), `multi_section` (two sections needed; recall is the share of gold sections found), `no_answer` (off-topic or near-topic questions the corpus does not cover).
- The 12 questions in `evals/retrieval_eval.py` (`QA`) pre-date this file, are not part of it, and are reported separately as a regression floor.
- The policy documents added for this study (shipping, warranty, privacy requests, fraud and chargebacks, billing and payments, customer service standards) are fictional and were also written by Claude. The refund, escalation and subscription policies were left unchanged.

## Split protocol

`dataset.assign_splits` orders the questions of each category by `sha256("sa-retrieval-v1:" + id)` and alternates `dev`, `test`. It uses no random seed and no metric. Anything that can be selected (RRF `k`, dense weight, pool size, rerank weight, abstention thresholds) is selected on `dev`. `test` is read once, after those choices are frozen, by `python -m evals.retrieval_eval --report`.

## Known limitations

- Same author for questions, gold labels, thesaurus rules and retrieval code. The thesaurus (`rag/expansion.py`) was written after the questions existed, so it is not blind to them even though only numeric knobs were tuned on `dev`.
- About 170 questions: bootstrap confidence intervals are wide, and per-category cells have only a handful of questions.
- Do not add questions to `test` after looking at results, and do not move questions between splits.
