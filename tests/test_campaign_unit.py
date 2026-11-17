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
    assert ten - one == pytest.approx(9 * (53 + 55 + 7))
    assert CU.planned_work(20, 20, [1], ("AdaptMIL",)) == pytest.approx(20 * 57 + 20 * 38 + 53 + 55)


def test_parallel_mode_yields_the_same_files_and_methods(tmp_path):
    config = json.load(open(RUN_CONFIG))["run"]["kwargs"]["config"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(XP.SS.SPACE, "epochs", ("cat", (2,)))
        mp.setitem(XP.SPACE2, "epochs2", ("cat", (1,)))
        mp.setitem(P.PHASE_TWO, "epochs2", P.PHASE_TWO["epochs2"])
        out = CU.run(config, seeds=(1,), trials=1, trials2=1, methods=("CLUDA", "CS+CCS", "AdaptMIL"),
                     workdir=tmp_path, environ={"FORGE_RUN_MODE": "smoke"}, parallel=2,
                     experiment="sim", rotation=0, setting="subject", shift=1.0, n_subjects=20)
    assert out["parallel"] == 2
    (xp,) = glob.glob(str(tmp_path / "xpilot-*.json"))
    (xc,) = glob.glob(str(tmp_path / "xcompare-*.json"))
    cmp = json.loads(Path(xc).read_text())
    assert set(cmp["methods"]) == {"CLUDA", "CS+CCS", "AdaptMIL"}
    assert "skipped" in cmp["methods"]["AdaptMIL"]
    assert all("runs" in cmp["methods"][m] for m in ("CLUDA", "CS+CCS"))
    assert {j["job"] for j in cmp["parallel"]["jobs"]} == {"MIL-CREDA", "CLUDA", "CS+CCS"}
    assert all(j["error"] is None for j in cmp["parallel"]["jobs"])
    assert "phase2Search" in json.loads(Path(xp).read_text())
    assert (tmp_path / "stamp.json").exists() or any(tmp_path.glob("*.json"))


def test_one_base_per_seed_is_shared_by_milcreda_and_the_comparators(unit_run):
    (xp,) = glob.glob(str(unit_run["dir"] / "xpilot-*.json"))
    (xc,) = glob.glob(str(unit_run["dir"] / "xcompare-*.json"))
    mil, cmp = json.loads(Path(xp).read_text()), json.loads(Path(xc).read_text())
    run = mil["runs"]["1"]
    assert run["cut"] == cmp["base"]["1"]["cut"]
    assert run["phase1"]["grid"][0]["note"].startswith("shared base")
    assert run["objectives"][XP.OBJECTIVE]["predict"]["targetTest"]["rows"] == cmp["base"]["1"]["predict"]["rows"]


def test_shared_base_gives_milcreda_the_same_result_as_its_own_refit(unit_run, tmp_path):
    config = json.load(open(RUN_CONFIG))["run"]["kwargs"]["config"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(XP.SS.SPACE, "epochs", ("cat", (2,)))
        mp.setitem(XP.SPACE2, "epochs2", ("cat", (1,)))
        mp.setitem(P.PHASE_TWO, "epochs2", P.PHASE_TWO["epochs2"])
        XP.smoke(config, seeds=(1,), trials=1, trials2=1, workdir=tmp_path, environ={}, search2=True,
                 experiment="sim", rotation=0, setting="subject", shift=1.0, n_subjects=20)
    (own,) = glob.glob(str(tmp_path / "xpilot-*.json"))
    (shared,) = glob.glob(str(unit_run["dir"] / "xpilot-*.json"))
    a = json.loads(Path(own).read_text())["runs"]["1"]
    b = json.loads(Path(shared).read_text())["runs"]["1"]
    assert a["cut"] == b["cut"]
    oa, ob = a["objectives"][XP.OBJECTIVE], b["objectives"][XP.OBJECTIVE]
    assert oa["predict"]["targetTest"]["rows"] == ob["predict"]["targetTest"]["rows"]
    last = str(max(int(k) for k in oa["epochs"]))
    assert oa["epochs"][last]["targetTest"]["rows"] == ob["epochs"][last]["targetTest"]["rows"]


def test_a_collapsing_best_trial_falls_back_to_the_next_ranked_one(tmp_path):
    from MIL_CREDA_EEG import v63_setup as V
    from MIL_CREDA_EEG import xcompare as XC
    config = json.load(open(RUN_CONFIG))["run"]["kwargs"]["config"]
    real = XC.shared_base
    first = {}

    def flaky(sides, cfg, device, seed, rotation, params1, unit):
        first.setdefault("params", dict(params1))
        if dict(params1) == first["params"]:
            raise V.V63Error("the between-bag median distance is 0.0")
        return real(sides, cfg, device, seed, rotation, params1, unit)

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(XP.SS.SPACE, "epochs", ("cat", (2,)))
        mp.setitem(XP.SPACE2, "epochs2", ("cat", (1,)))
        mp.setitem(P.PHASE_TWO, "epochs2", P.PHASE_TWO["epochs2"])
        mp.setattr(XC, "shared_base", flaky)
        out = CU.run(config, seeds=(1,), trials=2, trials2=1, methods=("CLUDA",), workdir=tmp_path,
                     environ={}, experiment="sim", rotation=0, setting="subject", shift=1.0, n_subjects=20)
    sel = out["files"]["phase1Selection"]
    assert sel["rank"] == 1 and len(sel["collapsed"]) == 1
    assert sel["collapsed"][0]["params"] == first["params"] == XP.LAST_PHASE1_RANKED[0][0]
    (xp,) = glob.glob(str(tmp_path / "xpilot-*.json"))
    assert json.loads(Path(xp).read_text())["runs"]["1"]["phase1Search"]["params"] == XP.LAST_PHASE1_RANKED[1][0]
    assert json.loads((tmp_path / "phase1-selection.json").read_text())["rank"] == 1
