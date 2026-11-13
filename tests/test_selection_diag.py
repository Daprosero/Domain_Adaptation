"""selection_diag replays the campaign's MIL-CREDA phase-2 search and scores each trial."""
import glob
import json
from pathlib import Path

import numpy as np
import pytest

from MIL_CREDA_EEG import phase2_trace as P
from MIL_CREDA_EEG import selection_diag as D
from MIL_CREDA_EEG import xpilot as XP

RUN_CONFIG = Path(__file__).resolve().parents[1] / "tools/kaggle/sm-dev-e3-s15-r0/run-config.json"


def test_margin_is_positive_when_source_bags_sit_on_their_class_centroid():
    t = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]]); q = [0, 0, 1, 1]
    s = np.array([[0.0, 0.1], [5.0, 4.9]]); y = [0, 1]
    assert D.centroid_margin(t, q, s, y) > 0.5
    assert D.centroid_margin(t, q, s, [1, 0]) < -0.5
    assert D.centroid_margin(t, [0, 0, 0, 0], s, y) == -1.0


def test_replay_picks_the_same_trials_as_the_campaign_search(tmp_path):
    config = json.load(open(RUN_CONFIG))["run"]["kwargs"]["config"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(XP.SS.SPACE, "epochs", ("cat", (2,)))
        mp.setitem(XP.SPACE2, "epochs2", ("cat", (1,)))
        mp.setitem(P.PHASE_TWO, "epochs2", P.PHASE_TWO["epochs2"])
        D.run(config, experiment="sim", rotation=0, half=0, trials=1, trials2=3, workdir=tmp_path,
              environ={}, setting="subject", shift=1.0, n_subjects=20)
        sides = XP.build_sides(XP.remote_units.resolve_inputs(XP.campaign.check_config(config), environ={}),
                               {}, experiment="sim", rotation=0, half=0, setting="subject", shift=1.0,
                               n_subjects=20)
        with XP.spectral_trunk.installed(), XP.window_samples(sides["samples"]):
            p1, s1 = XP.search_phase1(sides["sides"], sides["cfg"], XP.torch.device("cpu"), 1, 0)
            found = XP.search_phase2(sides["sides"], sides["cfg"], XP.torch.device("cpu"), 3, 0, p1, sides["unit"])
    (f,) = glob.glob(str(tmp_path / "selection-diag-*.json"))
    rows = json.loads(Path(f).read_text())["trials"]
    assert [r["params"] for r in rows] == [t["params"] for t in found["trials"]]
    assert [r["ruleBA"] for r in rows] == [t["score"] for t in found["trials"]]
    assert all("targetBA" in r and "ruleMargin" in r for r in rows)
