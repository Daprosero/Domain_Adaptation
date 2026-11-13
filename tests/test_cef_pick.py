"""pick="cef": the search still follows the centroid score; CREDA cef picks among its trials."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pytest

from MIL_CREDA_EEG import campaign_unit as CU
from MIL_CREDA_EEG import phase2_select
from MIL_CREDA_EEG import phase2_trace as P
from MIL_CREDA_EEG import xpilot as XP

RUN_CONFIG = Path(__file__).resolve().parents[1] / "tools/kaggle/sm-dev-e3-s15-r0/run-config.json"


def test_cef_is_the_effective_number_of_predicted_classes():
    assert phase2_select.cef_score([0, 1, 0, 1]) == pytest.approx(2.0)
    assert phase2_select.cef_score([1, 1, 1, 1]) == pytest.approx(1.0)
    p = np.array([3, 1]) / 4
    assert phase2_select.cef_score([0, 0, 0, 1]) == pytest.approx(np.exp(-(p * np.log(p)).sum()))


def _unit(tmp, pick):
    config = json.load(open(RUN_CONFIG))["run"]["kwargs"]["config"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(XP.SS.SPACE, "epochs", ("cat", (2,)))
        mp.setitem(XP.SPACE2, "epochs2", ("cat", (1,)))
        mp.setitem(P.PHASE_TWO, "epochs2", P.PHASE_TWO["epochs2"])
        CU.run(config, seeds=(1,), trials=1, trials2=4, methods=("CLUDA",), workdir=tmp, environ={},
               pick=pick, full_trace=False, experiment="sim", rotation=0, setting="subject", shift=1.0,
               n_subjects=20)
    (xp,) = glob.glob(str(tmp / "xpilot-*.json"))
    (xc,) = glob.glob(str(tmp / "xcompare-*.json"))
    return json.loads(Path(xp).read_text()), json.loads(Path(xc).read_text())


def test_cef_pick_takes_the_first_trial_with_the_highest_cef(tmp_path):
    mil, cmp = _unit(tmp_path / "cef", "cef")
    s = mil["phase2Search"]
    cefs = [t["cef"] for t in s["trials"]]
    assert s["pick"] == "cef" and s["pickedTrial"] == int(np.argmax(cefs))
    assert s["params"] == s["trials"][s["pickedTrial"]]["params"]
    c = cmp["methods"]["CLUDA"]["phase2Search"]
    assert c["pick"] == "cef" and c["pickedTrial"] == int(np.argmax([v if v is not None else -1 for v in c["trialCef"]]))


def test_cef_pick_leaves_the_search_path_unchanged(tmp_path):
    a, ca = _unit(tmp_path / "centroid", "centroid")
    b, cb = _unit(tmp_path / "cef", "cef")
    assert [t["params"] for t in a["phase2Search"]["trials"]] == [t["params"] for t in b["phase2Search"]["trials"]]
    assert [t["score"] for t in a["phase2Search"]["trials"]] == [t["score"] for t in b["phase2Search"]["trials"]]
    assert ca["methods"]["CLUDA"]["phase2Search"]["trialScores"] == cb["methods"]["CLUDA"]["phase2Search"]["trialScores"]
    assert a["phase2Search"]["pick"] == "centroid"
    assert a["phase2Search"]["pickedTrial"] == int(np.argmax([t["score"] for t in a["phase2Search"]["trials"]]))
