"""Statistics used for the live report (evals/live_stats.py)."""
from evals.live_stats import bootstrap_diff, mcnemar_exact, paired
from evals.redteam.live_attacker import families_for, parse_proposal


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(4, 0) - 0.125) < 1e-12          # 2 * 0.5^4
    assert abs(mcnemar_exact(0, 6) - 0.03125) < 1e-12         # smallest all-one-way n that reaches < 0.05
    assert abs(mcnemar_exact(6, 12) - mcnemar_exact(12, 6)) < 1e-12
    assert mcnemar_exact(5, 6) == 1.0


def test_bootstrap_and_paired_on_a_clear_effect():
    a = {f"s{i}": [False, False, False] for i in range(20)}
    b = {f"s{i}": [True, True, True] for i in range(20)}
    bt = bootstrap_diff(a, b, iters=500)
    assert bt["diff"] == 1.0 and bt["ci95"] == [1.0, 1.0]
    p = paired(a, b, {k: v[0] for k, v in a.items()}, {k: v[0] for k, v in b.items()})
    assert p["majority"]["only_B"] == 20 and p["significant_at_0.05"]


def test_paired_no_effect_is_not_significant():
    a = {f"s{i}": [i % 2 == 0] for i in range(20)}
    b = {f"s{i}": [i % 2 == 1] for i in range(20)}
    p = paired(a, b, {k: v[0] for k, v in a.items()}, {k: v[0] for k, v in b.items()})
    assert not p["significant_at_0.05"] and p["majority"]["mcnemar_p"] == 1.0


def test_live_attacker_families_and_kb_field():
    assert "indirect_note" in families_for("indirect_refund") and "indirect_kb" in families_for("kb_poison")
    assert "indirect_kb" not in families_for("pii_exfil")
    g = parse_proposal('```json\n{"opening": "hi", "kb": "poison"}\n```', False, True)
    assert g["kb"] == "poison" and g["note"] is None
    assert parse_proposal('{"opening": "hi", "kb": "poison"}', False, False)["kb"] is None
