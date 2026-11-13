"""search_attention=True: MIL-CREDA's phase-2 search also explores the consensus weight gamma in [0, 1] and
the attention temperature tau_att in [1, 10] (log); the picked values run the seeds; defaults restored."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from MIL_CREDA_EEG import campaign_unit as CU
from MIL_CREDA_EEG import phase2_trace as P
from MIL_CREDA_EEG import xpilot as XP

RUN_CONFIG = Path(__file__).resolve().parents[1] / "tools/kaggle/sm-dev-e3-s15-r0/run-config.json"


def test_attention_is_searched_recorded_and_restored(tmp_path):
    config = json.load(open(RUN_CONFIG))["run"]["kwargs"]["config"]
    before = dict(P.PHASE_TWO_AGGREGATION); space = dict(XP.SPACE2)
    seen = []
    real = XP.run_unit

    def spy(*a, **k):
        seen.append(dict(P.PHASE_TWO_AGGREGATION)); return real(*a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(XP.SS.SPACE, "epochs", ("cat", (2,)))
        mp.setitem(XP.SPACE2, "epochs2", ("cat", (1,)))
        mp.setitem(P.PHASE_TWO, "epochs2", P.PHASE_TWO["epochs2"])
        mp.setattr(XP, "run_unit", spy)
        CU.run(config, seeds=(1,), trials=1, trials2=3, methods=(), workdir=tmp_path, environ={}, pick="cef",
               search_attention=True, full_trace=False, experiment="sim", rotation=0, setting="subject",
               shift=1.0, n_subjects=20)
    (xp,) = glob.glob(str(tmp_path / "xpilot-*.json"))
    s = json.loads(Path(xp).read_text())["phase2Search"]
    for t in s["trials"]:
        assert 0.0 <= t["params"]["gamma"] <= 1.0 and 1.0 <= t["params"]["tau_att"] <= 10.0
    assert seen and seen[0] == {"gamma": s["params"]["gamma"], "temperature": s["params"]["tau_att"]}
    assert P.PHASE_TWO_AGGREGATION == before and dict(XP.SPACE2) == space
    (xc,) = glob.glob(str(tmp_path / "xcompare-*.json"))
    assert json.loads(Path(xc).read_text())["methods"] == {}


def test_default_search_leaves_attention_fixed(tmp_path):
    assert "gamma" not in XP.SPACE2 and P.PHASE_TWO_AGGREGATION == {"gamma": 0.5, "temperature": 1.0}
