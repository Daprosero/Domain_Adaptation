"""Stage 04 (ADHD shared pool): work-item split, the stored summary reproducing the paper's
Table tab:ieee-ranking and bootstrap, and the smoke wiring of the training entrypoint."""
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from MIL_CREDA_EEG import repro
from MIL_CREDA_EEG.reproduce import stage04_adhd_pool as M

PAPER = repro.repo_root().parents[1] / "paper" / "main.tex"


def test_units_are_the_80_jobs_and_every_split_covers_them_once():
    assert len(M.UNITS) == 80 and len(set(M.UNITS)) == 80
    assert M.job_name(*M.UNITS[0]) == "ix-r0h0-s0" and M.job_name(*M.UNITS[-1]) == "ix-r4h1-s7"
    for n in (1, 3, 7, 80):
        parts = [repro.portion(M.UNITS, (k, n)) for k in range(n)]
        flat = [x for p in parts for x in p]
        assert sorted(flat) == sorted(M.UNITS) and len(flat) == 80
    assert repro.portion(M.UNITS, None) == list(M.UNITS)


def test_paper_checks_hold_on_the_stored_summary():
    rows = repro.compare(M.paper_checks())
    assert len(rows) >= 90
    assert all(r["within"] for r in rows), [r for r in rows if not r["within"]]


def test_render_writes_the_paper_table(tmp_path):
    out = M.render(out_dir=tmp_path)
    tex = out["tex"].read_text()
    assert r"\label{tab:ieee-ranking}" in tex
    assert r"MIL-CREDA & \textbf{4.10} & \textbf{0.663} & \textbf{0.720} & \textbf{0.325} & -- & 3.75 & 0.820 & -- \\" in tex
    with open(out["csv"]) as fh:
        rows = list(csv.DictReader(fh))
    assert [r["method"] for r in rows][:3] == ["MIL-CREDA", "CCA-LSC", "Phase 1 (source only)"]
    if PAPER.exists():                       # the paper sits beside the repository in the workspace
        lines = PAPER.read_text().splitlines()
        start = lines.index(r"\begin{table*}[!t]", lines.index(r"\label{tab:ieee-ranking}") - 5)
        assert tex.splitlines() == lines[start:start + len(tex.splitlines())]


def test_analysis_from_the_condensed_trials_reproduces_the_stored_summary(tmp_path):
    sd = M.summary_path()
    for f in ("trials.csv", "units.json", "ptest.npz", "phase1_subjects.json"):
        shutil.copy(sd / f, tmp_path / f)
    M.analyse(tmp_path)
    assert json.loads((tmp_path / "ieee_bootstrap.json").read_text()) == json.loads(
        (sd / "ieee_bootstrap.json").read_text())
    assert json.loads((tmp_path / "ieee_pool_summary.json").read_text()) == json.loads(
        (sd / "ieee_pool_summary.json").read_text())


def test_stored_summary_holds_the_inputs_of_the_graph_analysis():
    sd = M.summary_path()
    S = json.loads((sd / "ieee_pool_summary.json").read_text())
    a = np.load(sd / "pick_arrays.npz")
    units = json.loads((sd / "units.json").read_text())
    for u in M.IEEE_UNITS:
        off = a[f"cef|{u}|attOffsets[target_test]"]
        assert len(off) == len(units[u]["targetTestSubjects"]) + 1
        assert off[-1] == len(a[f"cef|{u}|att[target_test]"])
        y, cut = np.array(units[u]["targetTestLabels"]), units[u]["cut"]
        ba = M.balanced_accuracy(y, (a[f"cef|{u}|Ptest"] > cut).astype(int))
        assert abs(ba - S["cefPicks"][f"{u}|MIL-CREDA"]["BA"]) < 1e-4


def test_cef_matches_its_definition():
    assert M.cef(12, 0) == 1.0 and abs(M.cef(6, 6) - 2.0) < 1e-12


def _ieee_root():
    try:
        return repro.ieee_root() if os.environ.get("IEEE_ROOT") else None
    except Exception:
        return None


@pytest.mark.skipif(not os.environ.get("IEEE_ROOT"), reason="needs the IEEE ADHD files (IEEE_ROOT)")
def test_smoke_runs_the_paper_entrypoint_and_condenses(tmp_path):
    done = M.run(smoke=True, out=tmp_path / "raw", ieee_root=_ieee_root())
    assert done[0]["methods"] == {m: 1 for m in M.METHODS}
    d = json.loads(next((tmp_path / "raw").glob("*/ieeepool-*.json")).read_text())
    assert (d["unitKey"], d["epochs2"], d["phase2Seed"], d["baseSeed"]) == ("ieee-r0h0", 2, 1, 1)
    assert d["cut"] == json.loads((M.summary_path() / "units.json").read_text())["ieee-r0h0"]["cut"]
    sd = M.condense(tmp_path / "raw", tmp_path / "summary", partial=True)
    with open(sd / "trials.csv") as fh:
        assert len(list(csv.DictReader(fh))) == len(M.METHODS)
    with pytest.raises(FileNotFoundError):
        M.condense(tmp_path / "raw", tmp_path / "s2")
