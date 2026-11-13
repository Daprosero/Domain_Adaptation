"""campaign_unit with phase1_params and phase2_fixed: no search runs, the seeds use the given settings."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import optuna
import pytest

from MIL_CREDA_EEG import campaign
from MIL_CREDA_EEG import campaign_unit as CU
from MIL_CREDA_EEG import phase2_trace as P
from MIL_CREDA_EEG import search
from MIL_CREDA_EEG import xcompare as XC
from MIL_CREDA_EEG import xpilot as XP

RUN_CONFIG = Path(__file__).resolve().parents[1] / "tools/kaggle/sm-dev-e3-s15-r0/run-config.json"
P1 = {"lr1": 0.004, "l2": 0.015, "width": 64, "tauAtt1": 4.8, "epochs": 2}
MIL = {"lr2": 1e-3, "epochs2": 1, "lambda_global": 1.0, "lambda_local": 1.0, "tau_local": 1.0}


def _cluda_params():
    campaign.register(["CLUDA"])
    trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)).ask()
    return {**search.suggest(trial, search.phase_two_space("CLUDA")), "epochs2": 1}


def test_fixed_settings_skip_both_searches_and_are_recorded(tmp_path):
    config = json.load(open(RUN_CONFIG))["run"]["kwargs"]["config"]
    cluda = _cluda_params()
    called = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(XP.SS.SPACE, "epochs", ("cat", (2,)))
        mp.setitem(P.PHASE_TWO, "epochs2", P.PHASE_TWO["epochs2"])
        for mod, fn in ((XP, "search_phase1"), (XP, "search_phase2"), (XC, "search_phase2")):
            mp.setattr(mod, fn, lambda *a, _n=f"{mod.__name__}.{fn}", **k: called.append(_n))
        CU.run(config, seeds=(1, 2), trials=1, trials2=1, methods=("CLUDA", "AdaptMIL"), workdir=tmp_path,
               environ={}, phase1_params=P1, phase2_fixed={"MIL-CREDA": MIL, "CLUDA": cluda},
               full_trace=False, experiment="sim", rotation=0, setting="subject", shift=1.0, n_subjects=20)
    assert called == []
    (xp,) = glob.glob(str(tmp_path / "xpilot-*.json"))
    (xc,) = glob.glob(str(tmp_path / "xcompare-*.json"))
    mil, cmp = json.loads(Path(xp).read_text()), json.loads(Path(xc).read_text())
    assert mil["runs"]["1"]["phase1Search"]["params"] == P1 == cmp["phase1Search"]["params"]
    assert mil["phase2Search"]["fixed"] is True and mil["phase2Search"]["params"] == MIL
    assert set(mil["runs"]) == {"1", "2"} and all("objectives" in r for r in mil["runs"].values())
    found = cmp["methods"]["CLUDA"]["phase2Search"]
    assert found["fixed"] is True
    assert {**found["params"], "epochs2": found["epochs2Given"]} == cluda
    assert set(cmp["methods"]["CLUDA"]["runs"]) == {"1", "2"}
    assert "skipped" in cmp["methods"]["AdaptMIL"]
    sel = json.loads((tmp_path / "phase1-selection.json").read_text())
    assert sel["params"] == P1 and sel["rule"].startswith("given")


def test_a_method_missing_from_phase2_fixed_is_refused(tmp_path):
    config = json.load(open(RUN_CONFIG))["run"]["kwargs"]["config"]
    with pytest.raises(ValueError, match="phase2_fixed"):
        CU.run(config, seeds=(1,), trials=1, trials2=1, methods=("CLUDA",), workdir=tmp_path, environ={},
               phase1_params=P1, phase2_fixed={"MIL-CREDA": MIL},
               experiment="sim", rotation=0, setting="subject", shift=1.0, n_subjects=20)
