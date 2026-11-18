# Judge calibration labeling pack

60 transcripts for a human to label, so the judges' agreement with a human can finally be measured (Cohen's kappa). Until you label them, every report says **NOT YET MEASURED**; nothing here fills labels automatically.

## What is in the pack
| Source | Items | From |
|---|---|---|
| `live` | 24 | real Gemini trials, `evals/results/live/gemini_main_heldout/trials.jsonl` (main + held-out sets, baseline + full configs) |
| `sim-main` | 18 | simulated model, main scenarios (`results/full.json` + `baseline.json`, so failures exist) |
| `sim-held` | 18 | simulated model, held-out scenarios (`results/heldout/full.json`) |

Sampling is stratified by (category, automatic pass/fail) within each source, seeded (`--seed 7`), so 18 categories and both automatic outcomes appear. Items are shuffled across sources. The pack is **blind**: `pack.jsonl` and `label.html` show the transcript, the tool calls, the actual end state and what policy expected, but no automatic verdict. Verdicts live in `key.jsonl`; do not open it until you are done.

Files: `RUBRIC.md` (read first), `pack.jsonl`, `key.jsonl`, `label.html`, `label.py`, `calibrate.py`, `build_pack.py`. Your output: `labels.jsonl`.

## Steps (about 45-60 minutes)
1. Read `RUBRIC.md` (5 rules for the reply label, 1 optional outcome question).
2. Label, either way:
   - **Browser (recommended):** open `evals/labeling/label.html` (double-click; works offline). Enter your name, label each item PASS/FAIL (optional: outcome yes/no/unsure, a note). Progress is kept in the browser. When done click **Export labels.jsonl** and save it as `evals/labeling/labels.jsonl`.
   - **Terminal:** `python -m evals.labeling.label --rater <yourname>` (or `make label-pack-cli RATER=<yourname>`). Resumable; appends to `labels.jsonl`.
3. Compute agreement: `python -m evals.labeling.calibrate` (or `make calibrate-pack`). It prints Cohen's kappa with a 95% bootstrap CI for human vs rule judge, human vs the stored judge verdict, and (if you answered the outcome question) human vs the harness's end-to-end pass/fail, overall and per source (live vs simulated). Full results go to `evals/labeling/calibration.json`.
4. Optional, stronger: ask a second person to label the same pack with a different rater name and append their rows to the same `labels.jsonl`; calibrate then also reports inter-rater kappa (the ceiling any judge can reach).

## Reading the result
- Fewer than 50 labeled items is reported with a warning; with 60 items the CI is still wide. Report the interval, not just the point estimate.
- A kappa that is high on `sim-*` but low on `live` means the rule judge was tuned to the simulator's phrasing.
- `evals/calibrate.py` (`make calibrate`) is the older path over `results/full.json` only; this pack reuses its kappa and bootstrap code, so both compute kappa identically.

## Rebuilding
`python -m evals.labeling.build_pack` (or `make label-pack`) regenerates the pack from the current results. The live trial file was still being written by the live-eval run when this pack was built (157 trials at build time), so a rebuild may sample different items. Item ids are stable per source trial, but **labels made against an old pack only match items that are still in the new one**: finish labeling before rebuilding.
