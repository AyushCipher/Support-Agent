"""Labeling pack for judge calibration (evals/labeling): blind, stratified, and calibrate never invents a number.
Synthetic labels below are written to a TEMP file only, to exercise the arithmetic; the real labels.jsonl is left for a human."""
import json

import pytest

from evals.labeling import build_pack, calibrate, label


def _pack():
    if not build_pack.PACK.exists():
        pytest.skip("pack not built (python -m evals.labeling.build_pack)")
    return label.read_jsonl(build_pack.PACK), {k["item_id"]: k for k in label.read_jsonl(build_pack.KEY)}


def test_pack_is_blind_and_stratified():
    pack, key = _pack()
    assert 50 <= len(pack) <= 70 and set(key) == {p["item_id"] for p in pack}
    for it in pack:                                                       # no automatic verdict leaks into what the labeler sees
        assert not {"judge", "passed", "why", "auto_judge", "auto_passed", "auto_why", "model"} & set(it)
    sources = {k["source"] for k in key.values()}
    assert sources == {"live", "sim-main", "sim-held"}
    for s in sources:                                                     # both automatic outcomes represented per source
        assert {k["auto_passed"] for k in key.values() if k["source"] == s} == {True, False}
    assert len({p["category"] for p in pack}) >= 10
    page = (build_pack.HERE / "label.html").read_text(encoding="utf-8")
    data = page.split("const PACK=", 1)[1].split(";const KEY", 1)[0]
    assert "<" not in data and len(json.loads(data)) == len(pack)        # embedded data cannot break out of the <script>


def test_stratified_sampler_covers_rare_failures():
    import random
    trials = [{"category": c, "passed": True, "i": i} for c in "abc" for i in range(30)] + [{"category": "a", "passed": False, "i": 99}]
    got = build_pack.stratified(trials, 4, random.Random(0))
    assert any(not t["passed"] for t in got) and {t["category"] for t in got} == {"a", "b", "c"}


def test_calibrate_says_not_measured_without_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "LABELS", tmp_path / "labels.jsonl")
    res = calibrate.compute()
    assert res["measured"] is False and "NOT YET MEASURED" in res["line"]


def test_calibrate_arithmetic_on_synthetic_labels(tmp_path, monkeypatch):
    pack, key = _pack()
    lab = tmp_path / "labels.jsonl"
    rows = [{"item_id": i, "rater": "synthetic", "label": "pass" if k["auto_passed"] else "fail",
             "outcome_ok": "yes" if k["auto_passed"] else "no"} for i, k in key.items()]
    lab.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(calibrate, "LABELS", lab)
    res = calibrate.compute()
    o = res["agreement"]["all"]["outcome_vs_harness"]
    assert res["measured"] and o["n"] == len(key) and o["kappa"] == 1.0 and o["accuracy"] == 1.0   # labels copied from harness -> perfect
    assert res["agreement"]["all"]["rule_judge"]["n"] > 0 and set(res["agreement"]) >= {"all", "live"}


def test_terminal_labeler_appends_resumably(tmp_path, monkeypatch):
    pack, _ = _pack()
    monkeypatch.setattr(label, "LABELS", tmp_path / "labels.jsonl")
    answers = iter(["p", "y", "looks right", "f", "", "", "q"])
    assert label.main(["--rater", "t"], ask=lambda _: next(answers)) == 0
    rows = label.read_jsonl(tmp_path / "labels.jsonl")
    assert [r["label"] for r in rows] == ["pass", "fail"] and rows[0]["outcome_ok"] == "yes" and rows[0]["note"] == "looks right"
    answers = iter(["q"])
    label.main(["--rater", "t"], ask=lambda _: next(answers))             # resumes after the two labeled items
    assert len(label.read_jsonl(tmp_path / "labels.jsonl")) == 2
