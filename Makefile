ifeq ($(OS),Windows_NT)
PY ?= python
export PYTHONPATH := src;.
else
PY ?= python3
export PYTHONPATH := src:.
endif
# live knobs: BACKEND=anthropic|gemini  MAXUSD=1.0  TRIALS=3  RPM=15 (Gemini free tier)
BACKEND ?= anthropic
MAXUSD ?= 1.0
TRIALS ?= 3
RPM ?= 0

.PHONY: install test eval eval-ci eval-heldout eval-heldout-ci heldout-dataset redteam redteam-live eval-live eval-live-dry eval-live-report results report dataset retrieval label calibrate online serve chat demo lint docker clean

install:            ## editable install with dev extras
	$(PY) -m pip install -e ".[dev]"

test:               ## unit + integration tests (offline, ~3s)
	$(PY) -m pytest -q

dataset:            ## regenerate golden scenarios
	$(PY) -m evals.build_dataset

eval:               ## full ablation ladder, 8 trials/scenario (simulated model, ~15s)
	$(PY) -m evals.runner --all --trials 8 --workers 1
	$(PY) -m evals.report

eval-ci:            ## regression gate used in CI
	$(PY) -m evals.ci_gate --trials 4

heldout-dataset:    ## regenerate the held-out scenarios (evals/scenarios_heldout/)
	$(PY) -m evals.build_heldout

eval-heldout:       ## held-out set, full ladder, 8 trials (simulated model)
	$(PY) -m evals.runner --all --set heldout --trials 8 --workers 1
	$(PY) -m evals.report

eval-heldout-ci:    ## held-out regression gate used in CI (floors + regression budget vs evals/baseline_heldout.json)
	$(PY) -m evals.ci_gate --heldout --trials 4

redteam:            ## adaptive red-team, offline mutation search (simulated defender), 200 attempts/goal/layer
	$(PY) -m evals.redteam.adaptive --budget 200 --repeats 1 --workers 4
	$(PY) -m evals.report

redteam-live:       ## LLM attacker (needs a key): BACKEND=gemini|anthropic
	$(PY) -m evals.redteam.adaptive --live --backend $(BACKEND) --budget 15 --max-usd $(MAXUSD)

eval-live:          ## REAL models (needs ANTHROPIC_API_KEY or GEMINI_API_KEY): resumable, cached, backoff, cost cap. BACKEND=gemini RPM=15 for Gemini
	$(PY) -m evals.live --backend $(BACKEND) --set both --config baseline full --trials $(TRIALS) --workers 2 --rpm $(RPM) --max-usd $(MAXUSD)
	$(PY) -m evals.live_report

eval-live-dry:      ## validates the whole live pipeline offline: real client code -> local mock HTTP server -> ScriptedLLM (no key, no cost)
	$(PY) -m evals.live --dry-run --backend $(BACKEND) --set both --config baseline full --trials 1 --limit-scenarios 30 --mock-fault 7:429 11:503 --backoff-base 0

eval-live-report:   ## docs/live_results.md from the raw live JSONL (kept separate from simulated results)
	$(PY) -m evals.live_report

results:            ## regenerate docs/results.md from the JSON results (main ladder + held-out + adaptive red-team)
	$(PY) -m evals.report

report:
	$(PY) -m evals.report

retrieval:
	$(PY) -m evals.retrieval_eval

label:              ## label 50 transcripts for judge calibration
	$(PY) -m evals.label_tool 50

calibrate:          ## judge-vs-human agreement (Cohen's kappa)
	$(PY) -m evals.calibrate

online:             ## sampled production eval over runs/state.db
	$(PY) -m evals.online_eval --state runs/state.db --rate 0.2

serve:              ## chat UI http://localhost:8080  reviewer desk /approvals
	$(PY) -m supportagent.cli serve --port 8080

chat:               ## terminal chat with inline approvals
	$(PY) -m supportagent.cli chat --customer C001

demo:
	bash scripts/demo.sh

lint:
	ruff check src evals tests

docker:
	docker compose up --build

clean:
	rm -rf runs/*.db runs/*.jsonl .pytest_cache **/__pycache__

# ---- review/deploy work-stream additions
.PHONY: demo-record label-pack label-pack-cli calibrate-pack
demo-record:        ## reproducible terminal recording of the demo flows -> docs/demo/demo.cast (asciinema v2), demo.svg, demo.txt
	$(PY) scripts/record_demo.py

label-pack:         ## (re)build the blind 60-transcript judge-calibration pack -> evals/labeling/{pack,key}.jsonl + label.html
	$(PY) -m evals.labeling.build_pack

label-pack-cli:     ## label the pack in the terminal (RATER=yourname)
	$(PY) -m evals.labeling.label --rater $(RATER)

calibrate-pack:     ## Cohen's kappa: human labels (evals/labeling/labels.jsonl) vs judges; NOT YET MEASURED until labeled
	$(PY) -m evals.labeling.calibrate
