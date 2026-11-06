"""The v58 patience exercise: early stopping replayed on recorded score series, the spread
of draws over the arms, the notebook balance and the phase-1 selection under patience."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from MIL_CREDA_EEG import patience_exercise as px

TOOL = Path(__file__).resolve().parents[1] / "tools" / "patience_exercise_v58.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("patience_exercise_v58", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stop_fires_patience_epochs_after_the_last_strict_improvement():
    series = [0.5, 0.6, 0.6, 0.6, 0.6]          # ties are not improvements
    assert px.patience_stop(series, 2) == {"stopEpoch": 4, "bestEpoch": 2,
                                          "bestScore": 0.6, "reachedCap": False}


def test_a_series_that_keeps_improving_reaches_the_cap():
    out = px.patience_stop([0.1, 0.2, 0.3, 0.4], 2)
    assert out == {"stopEpoch": 4, "bestEpoch": 4, "bestScore": 0.4, "reachedCap": True}


def test_a_stop_due_exactly_at_the_last_epoch_counts_as_reaching_the_cap():
    out = px.patience_stop([0.5, 0.4, 0.4], 2)
    assert out["stopEpoch"] == 3 and out["reachedCap"] is True and out["bestEpoch"] == 1


def test_nan_epochs_never_improve():
    out = px.patience_stop([math.nan, math.nan, 0.3, 0.2, 0.1, 0.0, 0.0], 3)
    assert out["bestEpoch"] == 3 and out["stopEpoch"] == 6


def test_patience_must_be_positive_and_series_non_empty():
    with pytest.raises(ValueError):
        px.patience_stop([0.1], 0)
    with pytest.raises(ValueError):
        px.patience_stop([], 3)


def test_outcomes_report_every_patience_and_the_score_lost_against_the_full_argmax():
    series = [0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9]
    out = px.trial_outcomes(series, (2, 10))
    assert out["full"] == {"bestEpoch": 7, "bestScore": 0.9}
    assert out["patience2"]["stopEpoch"] == 4 and out["patience2"]["scoreLoss"] == pytest.approx(0.4)
    assert out["patience10"]["reachedCap"] and out["patience10"]["scoreLoss"] == 0


def test_ten_draws_spread_four_three_three_and_the_extra_draw_rotates():
    for index in range(3):
        arms = [px.transfer_of_draw(k, index) for k in range(10)]
        counts = sorted(arms.count(t) for t in px.TRANSFERS)
        assert counts == [3, 3, 4]
        assert arms.count(px.TRANSFERS[index]) == 4
    with pytest.raises(ValueError):
        px.transfer_of_draw(10, 0)


def test_assign_keeps_every_trial_once_and_balances_the_loads():
    trials = [{"phase": 2, "name": f"s{i % 3}", "group": "common", "transfer": "A",
               "draw": i, "seconds": float(10 + i)} for i in range(12)]
    bins = px.assign(trials, 4)
    flat = sorted((t["name"], t["draw"]) for b in bins for t in b)
    assert flat == sorted((t["name"], t["draw"]) for t in trials)
    loads = [sum(t["seconds"] for t in b) for b in bins]
    assert max(loads) - min(loads) <= max(t["seconds"] for t in trials)


def test_distribution_counts_trials_that_reached_the_cap():
    rows = [{"outcomes": px.trial_outcomes(s)} for s in
            ([0.1] * 100, [i / 100 for i in range(100)])]
    d = px.distribution(rows, 10)
    assert d["trials"] == 2 and d["reachedCap"] == 1
    assert d["stopEpoch"]["min"] == 11 and d["stopEpoch"]["max"] == 100


def test_select_takes_the_best_score_under_patience_and_its_best_epoch(tool):
    def row(draw, series):
        return {"phase": 1, "name": "common", "transfer": "L1->L11", "draw": draw,
                "params": {"lr1": draw}, "outcomes": px.trial_outcomes(series)}
    late_peak = [0.5] * 20 + [0.99] + [0.5] * 79   # unreachable under patience 10
    steady = [0.6] * 5 + [0.7] + [0.6] * 94
    out = tool.select([row(0, late_peak), row(1, steady),
                       {"phase": 1, "name": "common", "transfer": "L1->L11", "draw": 2,
                        "error": "boom"}])
    assert out == [{"group": "common", "transfer": "L1->L11", "params": {"lr1": 1},
                    "epochs": 6, "draw": 1, "score": 0.7}]


def test_unit_seconds_at_fixed_caps_match_the_campaign_cost(tool):
    model = tool.cost_model()
    steps = tool.unit_seconds(model, lambda p, n, a: (1, 1))
    total = sum(sum(v) for v in steps.values())
    probe = __import__("json").loads(tool.T4_PROBE.read_text())
    costs = tool.rec.campaign_cost(probe, model["transfers"], model["core"], model["k2Only"])
    # the same terms; the step accounting adds the final and K3 units' own load
    extra_setup = len(steps["final"]) + len(steps["k3"]) - len(model["transfers"])
    assert total == pytest.approx(costs["coef1"] + costs["coef2"] + costs["fixed"]
                                  + extra_setup * model["table"]["setup"], rel=1e-9)
