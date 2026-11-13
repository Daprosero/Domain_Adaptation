"""comparator_diag replays each comparator's phase-2 search exactly and evaluates every trial."""
import json
from pathlib import Path

import pytest
import torch

from MIL_CREDA_EEG import comparator_diag as CD
from MIL_CREDA_EEG import phase2_trace as P
from MIL_CREDA_EEG import xcompare as XC
from MIL_CREDA_EEG import xpilot as XP

RUN_CONFIG = Path(__file__).resolve().parents[1] / "tools/kaggle/sm-dev-e3-s15-r0/run-config.json"


def test_replay_matches_the_campaign_search_scores(monkeypatch):
    config = json.load(open(RUN_CONFIG))["run"]["kwargs"]["config"]
    monkeypatch.setitem(XP.SS.SPACE, "epochs", ("cat", (2,)))
    monkeypatch.setitem(XP.SPACE2, "epochs2", ("cat", (1,)))
    monkeypatch.setitem(P.PHASE_TWO, "epochs2", P.PHASE_TWO["epochs2"])
    cfg = XP.remote_units.resolve_inputs(XP.campaign.check_config(config), environ={})
    built = XP.build_sides(cfg, {}, experiment="sim", rotation=0, half=0, setting="subject", shift=1.0, n_subjects=20)
    sides, dev = built["sides"], torch.device("cpu")
    XC.campaign.register(["CLUDA"])
    p1 = {"lr1": 1e-3, "l2": 1e-3, "width": 32, "tauAtt1": 2.0, "epochs": 2}
    with XP.spectral_trunk.installed(), XP.window_samples(built["samples"]):
        base = XC.shared_base(sides, built["cfg"], dev, XC.SEARCH_SEED, 0, p1, built["unit"])
        rows = CD.replay("CLUDA", base, sides, built["cfg"], dev, 3, 0, built["unit"])
        found = XC.search_phase2("CLUDA", base["prepared"], base["context"], source=sides["source"],
                                 source_test=sides["source_test"], target=sides["target"], cfg=built["cfg"],
                                 device=dev, trials2=3, rotation=0, unit=built["unit"], epochs2=None)
    assert [r.get("ruleScore", 0.0) for r in rows] == found["trialScores"]
    assert all("targetBA" in r for r in rows if "error" not in r)
