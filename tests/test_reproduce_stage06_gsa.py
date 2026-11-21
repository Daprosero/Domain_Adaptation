"""Stage 06 (ADHD graph analysis): split coverage, the paper's numbers from the stored summary, and the
smoke wiring (PLV of one unit from the public data and the stored attention, then a reduced GSA-XGB)."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pytest

from MIL_CREDA_EEG import repro
from MIL_CREDA_EEG.reproduce import stage06_gsa as S


def test_units_are_the_ten_adhd_units_and_every_split_covers_them_once():
    assert S.UNITS == tuple(f"ieee-r{r}h{h}" for r in range(5) for h in range(2))
    assert sorted(S.picks()) == sorted(S.UNITS)
    for n in (1, 2, 3, 10):
        parts = [repro.portion(S.UNITS, (k, n)) for k in range(n)]
        flat = [u for p in parts for u in p]
        assert sorted(flat) == sorted(S.UNITS) and len(flat) == len(S.UNITS)
    assert repro.portion(S.UNITS, None) == list(S.UNITS)


def test_stored_summary_holds_every_input_render_needs():
    d = S.summary_dir()
    g = np.load(d / "graphs.npz")
    assert len(g["labels"]) == S.N_SUBJECTS and len(set(g["subjects"].tolist())) == S.N_SUBJECTS
    for k in S.CONDITIONS:
        assert g[k].shape == (S.N_SUBJECTS, len(S.CH), len(S.CH))
    a = np.load(d / "pick_attention.npz")
    for u, pk in S.picks().items():
        assert int(a[f"{u}__trial"]) == pk["trial"]
        assert len(a[S._key(u, "attOffsets[target_test]")]) - 1 == len(a[S._key(u, "Ptest")])
    assert sum(f.stat().st_size for f in d.parent.rglob("*") if f.is_file()) < 15e6


def test_paper_checks_all_within_tolerance():
    rows = S.paper_checks()
    out = repro.compare(rows)
    assert len(out) == 21
    assert all(r["within"] for r in out), [r for r in out if not r["within"]]


def test_render_writes_both_figures_and_the_tables_matching_the_paper(tmp_path):
    S.render(out_dir=tmp_path)
    for name in ("ieee-gsa", "ieee-pair-descriptors"):
        assert (tmp_path / "figures" / f"{name}.pdf").stat().st_size > 0
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 0
    for name in ("gsa_xgb.tex", "gsa_xgb.csv", "pair_descriptors.tex", "pair_descriptors.csv"):
        assert (tmp_path / "tables" / name).stat().st_size > 0
    tex = (tmp_path / "tables" / "gsa_xgb.tex").read_text()
    assert "0.645" in tex and "0.727" in tex and "[0.008, 0.158]" in tex
    pair = json.loads((tmp_path / "tables" / "pair84.json").read_text())
    assert pair["nPairs"] == 876 and round(pair["descriptorCorrelation"], 3) == 0.988


def _ieee_root():
    r = os.environ.get("IEEE_ROOT")
    return r if r and glob.glob(f"{r}/ADHD_group/*.mat") else None


@pytest.mark.skipif(_ieee_root() is None, reason="needs the IEEE ADHD files (set IEEE_ROOT); no download in tests")
def test_smoke_wiring_reproduces_the_stored_graphs_of_one_unit(tmp_path):
    reports = S.run(split=(7, 10), smoke=True, out=tmp_path / "runs", ieee_root=_ieee_root())
    assert [r["unit"] for r in reports] == ["ieee-r3h1"] and reports[0]["reproduces"]
    out = S.merge(out=tmp_path / "summary", runs=tmp_path / "runs", smoke=True)
    g, stored = np.load(out / "graphs.npz"), np.load(S.summary_dir() / "graphs.npz")
    rows = np.flatnonzero(stored["units"] == "ieee-r3h1")
    assert g["subjects"].tolist() == stored["subjects"][rows].tolist()
    for k in ("all", "top", "bottom", "random 0"):
        np.testing.assert_allclose(g[k], stored[k][rows], rtol=0, atol=1e-7)
    xg = json.loads((out / "xgb84.json").read_text())
    assert set(xg["BA"]) == {"all", "top", "bottom", "random"}
