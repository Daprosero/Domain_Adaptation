"""experiment="balladeer": one BALLADEER condition as source, another as target, subject-disjoint
by the v60 task assignment, both directions over the same subjects."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from MIL_CREDA_EEG import campaign
from MIL_CREDA_EEG import xpilot as XP

RUN_CONFIG = Path(__file__).resolve().parents[1] / "tools/kaggle/sm-dev-e3-s15-r0/run-config.json"


def _sides(direction, rotation):
    config = json.load(open(RUN_CONFIG))["run"]["kwargs"]["config"]
    try:
        campaign.load_bags({})
    except Exception as e:                       # the processed BALLADEER bags are not on this machine
        pytest.skip(f"BALLADEER bags unavailable: {e}")
    return XP.build_sides(config, {}, experiment="balladeer", rotation=rotation, direction=direction)


def test_source_and_target_never_share_a_subject_and_test_is_held_out():
    b = _sides("CogniFit->Robots", 0)
    s, sub = b["sides"], b["sides"]["subjects"]
    src = set(sub["sourceTrain"]) | set(sub["sourceTest"]); tgt = set(sub["targetAdapt"]) | set(sub["targetTest"])
    assert src and tgt and not src & tgt
    assert not set(sub["targetAdapt"]) & set(sub["targetTest"])
    assert {x.condition for x in s["source"]} == {"CogniFit"} and {x.condition for x in s["target_test"]} == {"Robots"}
    assert b["unit"] == "ball:CogniFit->Robots:0" and len(s["s_names"]) == len(s["t_names"]) == 14


def test_the_two_directions_swap_the_same_subjects():
    a, b = _sides("CogniFit->Robots", 1)["sides"]["subjects"], _sides("Robots->CogniFit", 1)["sides"]["subjects"]
    assert set(a["sourceTrain"]) | set(a["sourceTest"]) == set(b["targetAdapt"]) | set(b["targetTest"])
    assert set(a["targetAdapt"]) | set(a["targetTest"]) == set(b["sourceTrain"]) | set(b["sourceTest"])
