"""xpilot's opt-in phase-2 search: one TPE study per unit, scored by phase2_select.centroid_score
on the target adapt half and held-out source test bags, its best configuration used by every seed.
"""
from __future__ import annotations

import copy
import glob
import json
import math
from pathlib import Path

import pytest
import torch

from MIL_CREDA_EEG import campaign, phase2_select, remote_units, spectral_trunk, two_phase
from MIL_CREDA_EEG import phase2_trace as P
from MIL_CREDA_EEG import v63_setup as V
from MIL_CREDA_EEG import xpilot as XP

RUN_CONFIG = Path(__file__).resolve().parents[1] / "tools/kaggle/sm-dev-e3-s15-r0/run-config.json"
DEVICE = torch.device("cpu")
SETTINGS = {"lr2": 2e-3, "epochs2": 2, "lambda_global": 2.5, "lambda_local": 0.3, "tau_local": 4.0}


def _raw_config():
    return json.load(open(RUN_CONFIG))["run"]["kwargs"]["config"]


class Forbidden(list):
    """A bag list the search must never open."""

    def _refuse(self, *a, **k):
        raise AssertionError("the phase-2 search touched the target test bags")

    __iter__ = __len__ = __getitem__ = __bool__ = __contains__ = _refuse


class NoLabel:
    """A target bag whose label the search must never read."""

    def __init__(self, bag):
        object.__setattr__(self, "_bag", bag)

    def __getattr__(self, name):
        if name == "label":
            raise AssertionError("the phase-2 search read a target label")
        return getattr(object.__getattribute__(self, "_bag"), name)


@pytest.fixture(scope="module")
def unit():
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(XP.SS.SPACE, "epochs", ("cat", (2,)))
        cfg = remote_units.resolve_inputs(campaign.check_config(_raw_config()), environ={})
        campaign.register(cfg["strategies"])
        remote_units.set_deterministic()
        built = XP.build_sides(cfg, {}, experiment="sim", rotation=0, setting="device", shift=1.0,
                               n_subjects=20)
        with spectral_trunk.installed(), XP.window_samples(built["samples"]):
            params, score = XP.search_phase1(built["sides"], built["cfg"], DEVICE, 1, 0)
    return {**built, "params1": params, "score1": score}


@pytest.fixture
def trunk(unit, monkeypatch):
    monkeypatch.setitem(XP.SS.SPACE, "epochs", ("cat", (2,)))
    monkeypatch.setitem(XP.SPACE2, "epochs2", ("cat", (1, 2)))
    with spectral_trunk.installed(), XP.window_samples(unit["samples"]):
        yield unit


def _state():
    return (dict(P.PHASE_TWO), two_phase.LAMBDA_GLOBAL, two_phase.LAMBDA_LOCAL,
            dict(two_phase.TERM_SELECTIONS))


# ------------------------------------------------------------------ settings


def test_settings_are_patched_inside_and_restored_after_even_on_error():
    before = _state()
    with XP.phase2_settings(**SETTINGS):
        assert P.PHASE_TWO["lr2"] == 2e-3 and P.PHASE_TWO["epochs2"] == 2 and P.PHASE_TWO["tauLocal"] == 4.0
        assert P.PHASE_TWO["wd2"] == before[0]["wd2"]
        assert two_phase.term_coefficients(two_phase.BOTH_TERMS) == (2.5, 0.3)
        assert (two_phase.LAMBDA_GLOBAL, two_phase.LAMBDA_LOCAL) == (2.5, 0.3)
        assert two_phase.TERM_SELECTIONS["global term only"] == before[3]["global term only"]
        hyper = P._hyper({"schedule": {"bagsPerStep": 8}})
        assert hyper["lr2"] == 2e-3 and hyper["tauLocal"] == 4.0 and hyper["epochs2"] == 2
        g, l = torch.tensor(1.5, requires_grad=True), torch.tensor(-0.5, requires_grad=True)
        total = two_phase.phase_two_objective({"global": g, "local": l})
        assert float(total.detach()) == pytest.approx(2.5 * 1.5 + 0.3 * -0.5)
    assert _state() == before
    with pytest.raises(RuntimeError):
        with XP.phase2_settings(**SETTINGS):
            raise RuntimeError("boom")
    assert _state() == before


def test_patched_values_reach_the_phase_two_objective(trunk, monkeypatch):
    sides, cfg = trunk["sides"], trunk["cfg"]
    taus, coefficients, lrs = [], [], []
    # traced_path runs under v63_setup.mil_creda_rules: Eq. 39 is V._objective -> V._obj, the
    # original two_phase.phase_two_objective, which reads two_phase.TERM_SELECTIONS
    real_terms, real_objective, real_adam = P.M.adaptation_terms, V._obj, torch.optim.Adam

    def terms(model, pairs_s, labels, pairs_t, tau):
        taus.append(tau)
        return real_terms(model, pairs_s, labels, pairs_t, tau)

    def objective(t, selection=two_phase.BOTH_TERMS):
        out = real_objective(t, selection)
        expected = 2.5 * t["global"] + 0.3 * t["local"]
        coefficients.append((two_phase.term_coefficients(selection),
                             bool(torch.allclose(out.detach(), expected.detach().to(out.dtype)))))
        return out

    def adam(params, lr, **k):
        lrs.append(lr)
        return real_adam(params, lr=lr, **k)

    monkeypatch.setattr(P.M, "adaptation_terms", terms)
    monkeypatch.setattr(V, "_obj", objective)
    monkeypatch.setattr(torch.optim, "Adam", adam)
    base = XP.phase2_base(sides, cfg, DEVICE, 1, 0, trunk["params1"], trunk["unit"])
    XP.phase2_trial(base, sides, cfg, DEVICE, 1, trunk["params1"], SETTINGS)
    steps = math.ceil(len(sides["target"]) / cfg["schedule"]["bagsPerStep"])
    assert len(taus) == SETTINGS["epochs2"] * steps and set(taus) == {4.0}
    assert set(coefficients) == {((2.5, 0.3), True)} and len(coefficients) == len(taus)
    assert lrs[-1] == 2e-3


# ------------------------------------------------------------------ the trial's phase 2


def test_light_trial_ends_in_the_same_model_as_run_rotation(trunk):
    sides, cfg = trunk["sides"], trunk["cfg"]
    searched = (trunk["params1"], trunk["score1"], 1)
    with XP.phase2_settings(**SETTINGS):
        out = XP.run_unit(sides, cfg, DEVICE, 1, 0, searched, trunk["unit"], log=lambda *a, **k: None)
    epochs = out["objectives"][XP.OBJECTIVE]["epochs"]
    last = epochs[str(SETTINGS["epochs2"] - 1)]
    base = XP.phase2_base(sides, cfg, DEVICE, 1, 0, trunk["params1"], trunk["unit"])
    model = XP.phase2_trial(base, sides, cfg, DEVICE, 1, trunk["params1"], SETTINGS)["model"]
    for key, bags in (("targetTest", sides["target_test"]), ("targetTrain", sides["target"])):
        y, p = P._subjects(bags, P._probabilities(model, bags, "target", DEVICE))
        assert [[int(a), P._s(b, 4)] for a, b in zip(y, p)] == last[key]["rows"]


# ------------------------------------------------------------------ the search


def test_search_never_reads_target_test_bags_or_target_labels(trunk, monkeypatch):
    sides, cfg = trunk["sides"], trunk["cfg"]
    blind = {**sides, "target_test": Forbidden(), "target": [NoLabel(b) for b in sides["target"]]}
    calls = []
    real = phase2_select.centroid_score

    def spy(target_reps, target_pseudo, source_reps, source_labels):
        calls.append((len(target_reps), len(target_pseudo), len(source_reps), list(source_labels)))
        return real(target_reps, target_pseudo, source_reps, source_labels)

    monkeypatch.setattr(phase2_select, "centroid_score", spy)
    before = _state()
    found = XP.search_phase2(blind, cfg, DEVICE, 2, 0, trunk["params1"], trunk["unit"])
    assert _state() == before
    n_t, n_s = len(sides["target"]), len(sides["source_test"])
    assert calls == [(n_t, n_t, n_s, [b.label for b in sides["source_test"]])] * 2
    assert set(found["params"]) == {"lr2", "epochs2", "lambda_global", "lambda_local", "tau_local"}
    assert len(found["trials"]) == 2 and 0.0 <= found["score"] <= 1.0
    assert found["score"] == max(t["score"] for t in found["trials"])
    for t in found["trials"]:
        p = t["params"]
        assert 1e-4 <= p["lr2"] <= 1e-2 and p["epochs2"] in (1, 2)
        assert all(0.1 <= p[k] <= 10 for k in ("lambda_global", "lambda_local", "tau_local"))
    assert found["rule"] == "phase2_select.centroid_score on the target adapt half + held-out source test bags"


def test_trial_score_is_centroid_score_of_the_adapted_model(trunk):
    sides, cfg = trunk["sides"], trunk["cfg"]
    base = XP.phase2_base(sides, cfg, DEVICE, 1, 0, trunk["params1"], trunk["unit"])
    trial = XP.phase2_trial(base, sides, cfg, DEVICE, 1, trunk["params1"], SETTINGS)
    model = trial["model"]
    zt = P.representations(model, sides["target"], "target", DEVICE)
    with torch.no_grad():
        dtype = next(model.parameters()).dtype
        pseudo = model.classify(torch.as_tensor(zt, dtype=dtype)).argmax(1).numpy()
    zs = P.representations(model, sides["source_test"], "source", DEVICE)
    expected = phase2_select.centroid_score(zt, pseudo, zs, [b.label for b in sides["source_test"]])
    assert trial["score"] == pytest.approx(expected)


# ------------------------------------------------------------------ smoke


TOP_KEYS = {"experiment", "unit", "rotation", "sim", "half", "objective", "runs", "phaseTwo", "seconds",
            "timing", "environment"}
TIMING_KEYS = {"loadSeconds", "searchSeconds", "seedSeconds", "cpus", "torchThreads", "bags"}


def _smoke(tmp_path, monkeypatch, **kw):
    monkeypatch.setitem(XP.SS.SPACE, "epochs", ("cat", (2,)))
    monkeypatch.setitem(XP.SPACE2, "epochs2", ("cat", (1,)))
    monkeypatch.setitem(P.PHASE_TWO, "epochs2", 1)
    before = _state()
    XP.smoke(_raw_config(), experiment="sim", rotation=0, trials=1, epochs2=1, n_subjects=20,
             setting="device", shift=1.0, workdir=tmp_path, environ={}, **kw)
    (path,) = glob.glob(str(tmp_path / "xpilot-*.json"))
    return json.loads(open(path).read()), before


def test_smoke_search2_off_keeps_its_keys(tmp_path, monkeypatch):
    out, _ = _smoke(tmp_path, monkeypatch)
    assert set(out) == TOP_KEYS and set(out["timing"]) == TIMING_KEYS
    assert "phase2Search" not in out["runs"]["1"]


def test_smoke_search2_records_the_search_and_runs_every_seed_with_its_best(tmp_path, monkeypatch):
    out, before = _smoke(tmp_path, monkeypatch, search2=True, trials2=2, seeds=(1, 2))
    assert _state() == before
    assert set(out) == TOP_KEYS | {"phase2Search"}
    assert set(out["timing"]) == TIMING_KEYS | {"search2Seconds"}
    found = out["phase2Search"]
    assert {"params", "score", "trials", "rule", "pick", "pickedTrial"} <= set(found) and len(found["trials"]) == 2
    best = found["params"]
    assert out["phaseTwo"]["lr2"] == best["lr2"] and out["phaseTwo"]["epochs2"] == best["epochs2"]
    assert out["phaseTwo"]["tauLocal"] == best["tau_local"]
    assert out["phaseTwo"]["lambdaGlobal"] == best["lambda_global"]
    assert out["phaseTwo"]["lambdaLocal"] == best["lambda_local"]
    for run in out["runs"].values():
        assert set(run["objectives"][XP.OBJECTIVE]["epochs"]) == {"-1", *map(str, range(best["epochs2"]))}
