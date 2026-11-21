"""Stage 05 (attention influence and local alignment): split coverage, merge, render and paper numbers."""
from __future__ import annotations

import glob
import json
import os

import pytest

from MIL_CREDA_EEG import ieee_pool as I
from MIL_CREDA_EEG import repro
from MIL_CREDA_EEG.reproduce import stage05_influence as S


def test_units_cover_the_paper_jobs_once():
    assert len(S.UNITS) == 24 + 3 * 10
    assert len(set(S.UNITS)) == len(S.UNITS)
    assert tuple(S.IEEE) == tuple(I.UNITS)
    assert sorted(S.SYNTHETIC) == sorted(json.loads((S.repro.eeg_dir() / "Manifests" / "pool_aggregation"
                                                     / "milcreda_picks.json").read_text())["units"])


@pytest.mark.parametrize("n", [1, 2, 3, 7, 54])
def test_portions_partition_the_units(n):
    parts = [repro.portion(S.UNITS, (k, n)) for k in range(n)]
    flat = [u for p in parts for u in p]
    assert sorted(flat) == sorted(S.UNITS)
    assert len(flat) == len(set(flat))
    assert repro.portion(S.UNITS, None) == list(S.UNITS)


def test_paper_checks_hold_on_the_stored_summary():
    rows = repro.compare(S.paper_checks())
    assert len(rows) == 32
    assert all(r["within"] for r in rows), [r for r in rows if not r["within"]]


def test_alignment_pair_is_the_papers():
    pair = S.alignment_pair()
    assert pair["unit"] == "ieee-r3h1"
    assert (pair["pairs"]["0"]["target"], pair["pairs"]["0"]["source"]) == ("IEEE-v298", "IEEE-v52p")
    assert (pair["pairs"]["1"]["target"], pair["pairs"]["1"]["source"]) == ("IEEE-v236", "IEEE-v265")


def test_render_writes_the_table_and_figure_from_the_summary(tmp_path):
    out = S.render(out_dir=tmp_path)
    for f in out["figures"] + out["tables"]:
        assert os.path.getsize(f) > 0
    csv = (tmp_path / "tables" / "tab_ieee_best_subject.csv").read_text().splitlines()
    assert csv[0].startswith("windows_removed,")
    highest = csv[2].split(",")
    assert highest[0] == "Highest 25%"
    assert [round(float(v), 3) for v in highest[1:]] == [1.374, 1.105, 1.344, 2.066]
    tex = (tmp_path / "tables" / "tab_ieee_best_subject.tex").read_text()
    assert "\\textbf{1.374}" in tex and "Class-event & -- & --" in tex


def test_merge_joins_portions_and_reports_what_is_missing(tmp_path):
    stored = json.loads((S._summary_dir(None) / "influence.json").read_text())
    raw = tmp_path / "raw"
    for k, unit in enumerate(S.IEEE):  # two "accounts" wrote the cef units into the same folder
        f = S.raw_path("cef", unit, raw)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({**stored["cef"][unit], "reproduces": True, "seconds": 1.0}))
    out = S.merge(out=tmp_path / "summary", raw=raw)
    merged = json.loads((out / "influence.json").read_text())
    assert sorted(merged["cef"]) == sorted(S.IEEE) and not merged["synthetic"] and not merged["oracle"]
    status = json.loads((out / "status.json").read_text())
    assert len(status["missing"]) == len(S.UNITS) - len(S.IEEE)
    st = json.loads((out / "statistics.json").read_text())
    assert round(st["ADHD cef"]["top"]["removed"], 3) == 1.374


def _ieee_available() -> bool:
    given = os.environ.get("IEEE_ROOT")
    return bool(given and glob.glob(f"{given}/ADHD_group/*.mat"))


@pytest.mark.skipif(not _ieee_available(), reason="needs IEEE_ROOT with the public ADHD files")
def test_smoke_wiring_measures_one_adhd_unit(tmp_path):
    out = S.run(smoke=True, out=tmp_path)
    d = json.loads(S.raw_path(*S.SMOKE_ITEM, out).read_text())
    assert d["unitKey"] == S.SMOKE_ITEM[1] and d["strict"] is False
    assert len(d["rows"]) == 12 and {"top", "bottom", "random"} <= set(d["rows"][0])
