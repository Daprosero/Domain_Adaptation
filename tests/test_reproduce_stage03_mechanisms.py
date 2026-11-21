"""Stage 03 (synthetic mechanisms): its units, its stored summary against the paper, and its wiring."""
import json
import os

import numpy as np
import pytest

from MIL_CREDA_EEG import repro
from MIL_CREDA_EEG import pool_aggregation as PA
from MIL_CREDA_EEG.reproduce import stage03_mechanisms as S

SUMMARY = S.stage_dir() / "summary"


def test_the_units_are_the_24_shared_pool_units_in_a_fixed_order():
    assert len(S.UNITS) == 24 and S.UNITS == sorted(S.UNITS) and S.FIGURE_UNIT in S.UNITS
    assert S.FIGURE_UNIT == PA.FIGURE_UNIT and S.FIGURE_METHOD == PA.FIGURE_METHOD


@pytest.mark.parametrize("n", [1, 2, 3, 5, 24, 30])
def test_the_portions_cover_every_unit_once(n):
    parts = [repro.portion(S.UNITS, (k, n)) for k in range(n)]
    flat = [u for p in parts for u in p]
    assert sorted(flat) == sorted(S.UNITS) and len(flat) == len(set(flat))
    assert repro.portion(S.UNITS, None) == S.UNITS


def test_the_stored_picks_are_the_current_draw_for_every_unit():
    """The v82 picks under inputs/ pass the entrypoint's own refusal check (the tracked Manifests do not)."""
    picks = json.loads((S.inputs_dir() / "milcreda_picks.json").read_text())
    assert sorted(picks["units"]) == S.UNITS
    for u in S.UNITS:
        PA.checked_pick(u, picks)
    comp = json.loads((S.inputs_dir() / "comparator_picks.json").read_text())
    assert PA.checked_pick(S.FIGURE_UNIT, comp, S.FIGURE_METHOD)["trial"] == 43
    assert all((S.inputs_dir() / "bases" / f"{u}-s1.pt").exists() for u in S.UNITS)


def test_the_stored_records_are_the_campaign_runs_that_reproduced_their_picks():
    recs = S.load_records()
    assert sorted(recs) == S.UNITS
    assert all(r["reproduces"] and r["commit"] == "d75c6c2b48c96766909447ced6c602c193677936" for r in recs.values())
    assert all(r["epochs2"] == 100 and r["evalSubjects"] == 120 and r["procs"] == 4 for r in recs.values())


def test_the_stored_statistics_are_what_the_records_give():
    stored, again = S.load_statistics(), S.statistics(SUMMARY)
    assert json.loads(json.dumps(again)) == stored


def test_every_paper_number_is_reproduced_from_the_stored_summary():
    rows = repro.compare(S.paper_checks())
    assert len(rows) > 90
    bad = [r for r in rows if not r["within"]]
    assert not bad, bad


def test_render_writes_the_paper_tables_and_figures(tmp_path):
    out = S.render(out_dir=tmp_path)
    names = {os.path.basename(p) for p in out["figures"] + out["tables"]}
    assert names == {"synthetic-attention-signal.pdf", "synthetic-attention-signal.png", "synthetic-alignment.pdf",
                     "synthetic-alignment.png", "synthetic-attention-variants.tex", "synthetic-attention-variants.csv",
                     "synthetic-term-ablation.tex", "synthetic-term-ablation.csv"}
    att = (tmp_path / "tables" / "synthetic-attention-variants.tex").read_text()
    assert "Rel.\\ + consensus & \\textbf{0.72} & ref. & \\textbf{2.71} & \\textbf{0.57}$^{*}$ & 0.63$^{*}$ \\\\" in att
    assert "Mean & 0.52 & 0.01/+0.38 & 4.60 & -- & -- \\\\" in att
    term = (tmp_path / "tables" / "synthetic-term-ablation.tex").read_text()
    for line in ("None & 0.576 & 0.038 \\\\", "Global only & 0.717 & 0.453 \\\\", "Local only & 0.669 & 0.155 \\\\",
                 "Global + local & \\textbf{0.721} & ref. \\\\"):
        assert line in term


def test_merge_refuses_an_incomplete_set_of_portions(tmp_path):
    (tmp_path / f"am-{S.UNITS[0]}").mkdir()
    with pytest.raises(FileNotFoundError, match="24 of 24|23 of 24"):
        S.merge(out=tmp_path / "summary", raw=tmp_path)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("REPRO_SLOW") != "1", reason="about 70 s of CPU; set REPRO_SLOW=1")
def test_the_smoke_run_drives_the_paper_entrypoint(tmp_path):
    done = S.run(smoke=True, out=tmp_path)
    assert [d["unit"] for d in done] == [S.UNITS[0]] and done[0]["variants"] == ["own", "mean"]
    rec = json.loads((tmp_path / f"am-{S.UNITS[0]}" / f"aggregation-{S.UNITS[0]}.json").read_text())
    pick = json.loads((S.inputs_dir() / "milcreda_picks.json").read_text())["units"][S.UNITS[0]]
    assert rec["trial"] == pick["trial"] and rec["epochs2"] == 1 and rec["evalSubjects"] == 6
    assert np.isfinite(rec["families"]["aggregation"]["own"]["BApool"])
