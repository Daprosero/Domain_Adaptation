"""The v58 timing extrapolation (`tools/timing_v58_record.py`): the campaign's epoch
accounting and the schema-1 record whose one-formula estimate is the extrapolated total."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "timing_v58_record.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("timing_v58_record", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WINDOWS = {"source_train": 10, "source_valid": 5, "target_train": 10, "target_valid": 5}
REFIT = {"source_train": 15, "source_valid": 0, "target_train": 15, "target_valid": 0}


def probe(tool, epoch=1.0, source_pass=0.0, setup=0.0):
    entry = {"steadyEpochSeconds": [epoch], "error": None}
    return {"windows": WINDOWS,
            "phase1": {g: dict(entry) for g in tool.GROUPS},
            "phase2": {s: dict(entry) for s in tool.STRATEGIES},
            "sourcePass": {g: {"seconds": source_pass} for g in tool.GROUPS},
            "setup": {"loadBagsSeconds": setup, "searchDataSeconds": 0.0}}


def test_epoch_counts_follow_the_campaign_units(tool):
    costs = tool.campaign_cost(probe(tool), {"A": {"inner": WINDOWS, "refit": WINDOWS}},
                               core=set(), k2_only=set())
    # per fold: 3 groups x (60 + 1) + upper bound 4; 9 strategies x 60 + 2 x 9 refits
    assert costs["epochs1"] == 4 * (3 * 61 + 4)
    assert costs["epochs2"] == 4 * (9 * 60 + 18)


def test_k3_adds_variants_on_core_and_one_ablation_on_k2_only(tool):
    one = {"inner": WINDOWS, "refit": WINDOWS}
    plain = tool.campaign_cost(probe(tool), {"A": one, "B": one}, set(), set())
    k3 = tool.campaign_cost(probe(tool), {"A": one, "B": one}, {"A"}, {"B"})
    assert k3["epochs1"] - plain["epochs1"] == 4 * 4
    assert k3["epochs2"] - plain["epochs2"] == 4 * (6 + 1)


def test_epochs_scale_by_windows_and_refits_by_training_windows(tool):
    base = tool.campaign_cost(probe(tool), {"A": {"inner": WINDOWS, "refit": WINDOWS}},
                              set(), set())
    doubled = {k: 2 * v for k, v in WINDOWS.items()}
    double = tool.campaign_cost(probe(tool), {"A": {"inner": doubled, "refit": WINDOWS}},
                                set(), set())
    # phase-1 search epochs double, the refit epoch (refit windows) does not
    assert double["coef1"] - base["coef1"] == pytest.approx(4 * (3 * 60 + 3))
    refit = tool.campaign_cost(probe(tool), {"A": {"inner": WINDOWS, "refit": REFIT}},
                               set(), set())
    assert refit["coef1"] - base["coef1"] == pytest.approx(4 * 4 * (15 - 10) / 15)


def test_the_record_estimate_is_the_extrapolated_total(tool):
    costs = tool.campaign_cost(probe(tool, epoch=2.0, source_pass=3.0, setup=0.5),
                               {"A": {"inner": WINDOWS, "refit": REFIT}}, {"A"}, set())
    record = tool.timing_record(costs, "cpu", "2026-11-06T00:00:00+00:00")
    row, full = record["measurements"][0], record["scales"]["full"]
    estimate = (row["secondsPerUnit"] * full["epochs"] * full["folds"] * full["seeds"]
                * full["arms"] + row["setupSeconds"] * full["folds"] * full["seeds"])
    assert estimate == pytest.approx(costs["coef1"] + costs["coef2"] + costs["fixed"])
    assert record["schema"] == 1 and row["host"] == "local" and row["unit"] == "epoch"


def test_a_failed_measurement_is_refused(tool):
    bad = probe(tool)
    bad["phase2"]["COT"] = {"steadyEpochSeconds": [], "error": "RuntimeError: mps"}
    with pytest.raises(ValueError, match="mps"):
        tool.campaign_cost(bad, {"A": {"inner": WINDOWS, "refit": WINDOWS}}, set(), set())
