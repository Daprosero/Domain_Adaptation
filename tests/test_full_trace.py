"""full_trace=False: the same end model and predict as the full trace, with only the final epoch kept."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from MIL_CREDA_EEG import phase2_trace as P
from MIL_CREDA_EEG import xpilot as XP

RUN_CONFIG = Path(__file__).resolve().parents[1] / "tools/kaggle/sm-dev-e3-s15-r0/run-config.json"


def _run(tmp, full):
    config = json.load(open(RUN_CONFIG))["run"]["kwargs"]["config"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(XP.SS.SPACE, "epochs", ("cat", (2,)))
        mp.setitem(P.PHASE_TWO, "epochs2", P.PHASE_TWO["epochs2"])
        XP.smoke(config, experiment="sim", rotation=0, setting="device", shift=0.5, n_subjects=20,
                 trials=1, epochs2=3, workdir=tmp, environ={}, full_trace=full)
    (f,) = glob.glob(str(tmp / "xpilot-*.json"))
    return json.loads(Path(f).read_text())


def test_light_run_keeps_predict_and_final_epoch_identical(tmp_path):
    full = _run(tmp_path / "full", True)
    light = _run(tmp_path / "light", False)
    o_full = full["runs"]["1"]["objectives"][XP.OBJECTIVE]
    o_light = light["runs"]["1"]["objectives"][XP.OBJECTIVE]
    assert o_light["predict"]["targetTest"]["rows"] == o_full["predict"]["targetTest"]["rows"]
    assert list(o_light["epochs"]) == ["2"]
    assert o_light["epochs"]["2"]["targetTest"]["rows"] == o_full["epochs"]["2"]["targetTest"]["rows"]
    assert "-1" in o_full["epochs"] and len(o_full["epochs"]) == 4
    assert light["fullTrace"] is False and "fullTrace" not in full
    assert o_light["equivalence"]["note"]   # eq39+hardloc is a new rule: no existing path either way
