"""campaign_unit: MIL-CREDA and the comparators in one notebook, one phase-1 search, progress lines."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from MIL_CREDA_EEG import campaign_unit as CU
from MIL_CREDA_EEG import phase2_trace as P
from MIL_CREDA_EEG import xpilot as XP

RUN_CONFIG = Path(__file__).resolve().parents[1] / "tools/kaggle/sm-dev-e3-s15-r0/run-config.json"


@pytest.fixture(scope="module")
def unit_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("unit")
    config = json.load(open(RUN_CONFIG))["run"]["kwargs"]["config"]
    calls = []
    real = XP.search_phase1
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(XP.SS.SPACE, "epochs", ("cat", (2,)))
        mp.setitem(XP.SPACE2, "epochs2", ("cat", (1,)))
        mp.setitem(P.PHASE_TWO, "epochs2", P.PHASE_TWO["epochs2"])
        mp.setattr(XP, "search_phase1", lambda *a, **k: calls.append(1) or real(*a, **k))
        out = CU.run(config, seeds=(1,), trials=1, trials2=1, methods=("CLUDA", "AdaptMIL"),
                     workdir=tmp, environ={"FORGE_RUN_MODE": "smoke"},
                     experiment="sim", rotation=0, setting="subject", shift=1.0, n_subjects=20)
    return {"out": out, "dir": tmp, "phase1Calls": len(calls)}


def test_phase1_is_searched_once_and_shared_by_both_entries(unit_run):
    assert unit_run["phase1Calls"] == 1
    (xp,) = glob.glob(str(unit_run["dir"] / "xpilot-*.json"))
    (xc,) = glob.glob(str(unit_run["dir"] / "xcompare-*.json"))
    mil, cmp = json.loads(Path(xp).read_text()), json.loads(Path(xc).read_text())
    assert mil["unit"] == cmp["unit"] == unit_run["out"]["ran"][0]
    assert mil["runs"]["1"]["phase1Search"]["params"] == cmp["phase1Search"]["params"]
    assert mil["runs"]["1"]["subjectsSplit"] == cmp["subjectsSplit"]
    assert "phase2Search" in mil and "CLUDA" in cmp["methods"] and "skipped" in cmp["methods"]["AdaptMIL"]


def test_progress_rows_cover_every_stage_and_reach_the_plan(unit_run):
    rows = [json.loads(l) for l in (unit_run["dir"] / "progress.jsonl").read_text().splitlines()]
    labels = " | ".join(r["label"] for r in rows)
    for part in ("phase-1 search trial 1/1", "MIL-CREDA phase-2 search trial 1/1", "MIL-CREDA seed 1",
                 "shared base seed 1", "CLUDA phase-2 search trial 1/1", "CLUDA seed 1"):
        assert part in labels
    assert rows[-1]["done"] == pytest.approx(rows[-1]["total"], rel=0.05)
    assert rows[-1]["leftMin"] == pytest.approx(0.0, abs=0.5)


def test_planned_work_counts_seeds_and_skips(unit_run):
    one = CU.planned_work(20, 20, [1], ("CLUDA",))
    ten = CU.planned_work(20, 20, list(range(10)), ("CLUDA",))
    assert ten - one == pytest.approx(9 * (158 + 31 + 7))
    assert CU.planned_work(20, 20, [1], ("AdaptMIL",)) == pytest.approx(20 * 57 + 20 * 38 + 158)
