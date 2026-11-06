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


def test_a_phase_one_adaptmil_trial_registers_its_own_unit():
    trials = [{"phase": 1, "name": "AdaptMIL"}, {"phase": 1, "name": "common"},
              {"phase": 1, "name": "MIL-CREDA"}, {"phase": 2, "name": "CASUAL"}]
    assert px.registrations(trials) == ["AdaptMIL", "CASUAL"]
    from MIL_CREDA_EEG import campaign, model_store
    campaign.register(px.registrations([{"phase": 1, "name": "AdaptMIL"}]))
    model_store.require_registered("AdaptMIL")


def test_gpu_witness_records_parameter_devices_steps_and_epoch_stamps():
    import torch
    from MIL_CREDA_EEG import two_phase
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    original = two_phase.validation_record
    with px.GpuWitness(two_phase, torch) as w:
        for _ in range(3):
            opt.zero_grad()
            model(torch.ones(1, 2)).sum().backward()
            opt.step()
            assert two_phase.validation_record is not original
            w.stamps.append(0.0)
    assert two_phase.validation_record is original
    ev = w.evidence()
    assert ev["optimizerSteps"] == 3 and ev["parameterDevices"] == {"cpu": 6}
    assert ev["allParametersOnCuda"] is False and ev["epochsStamped"] == 3


def _gpu_trial(**ev):
    base = {"allParametersOnCuda": True, "cudaMaxMemoryAllocatedMB": 120.0,
            "medianEpochSeconds": 0.7, "parameterDevices": {"cuda:0": 10}}
    return {"phase": 2, "name": "COT", "transfer": "L1->L11", "draw": 0,
            "gpuEvidence": {**base, **ev}}


def test_certify_accepts_a_notebook_whose_every_trial_trained_on_cuda(tool):
    result = {"device": {"cudaIsAvailable": True, "cudaDeviceName0": "Tesla T4"}}
    assert tool.certify(result, [_gpu_trial(), _gpu_trial()])["certified"] is True


def test_certify_refuses_missing_evidence_cpu_parameters_or_cpu_speed(tool):
    gpu = {"device": {"cudaIsAvailable": True, "cudaDeviceName0": "Tesla T4"}}
    assert not tool.certify({"device": {"cudaAvailable": True, "name": "Tesla T4"}},
                            [{"phase": 1, "name": "common", "transfer": "A", "draw": 0}]
                            )["certified"]
    out = tool.certify(gpu, [_gpu_trial(allParametersOnCuda=False,
                                        parameterDevices={"cpu": 4})])
    assert not out["certified"] and "parameters on" in out["reasons"][0]
    assert not tool.certify(gpu, [_gpu_trial(medianEpochSeconds=9.0)])["certified"]
    assert not tool.certify(gpu, [_gpu_trial(cudaMaxMemoryAllocatedMB=0.0)])["certified"]
