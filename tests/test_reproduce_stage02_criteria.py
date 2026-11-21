"""Stage 02 (synthetic criteria pool): work split, the analysis ported from the v82 campaign, and the
paper's numbers rendered from the tracked summary."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest

from MIL_CREDA_EEG import repro
from MIL_CREDA_EEG.reproduce import stage02_criteria as S

CAMPAIGN = repro.eeg_dir() / "Results" / "v82-pool24"          # untracked; present on the campaign machine
FIGURE_DATA = Path(__file__).resolve().parents[3] / "paper" / ".paper-writing" / "figures" / "synthetic-map-methods"


def test_units_are_the_192_sx_jobs_and_every_split_covers_them_once():
    assert len(S.UNITS) == 192 and len(set(S.UNITS)) == 192
    assert {u for u, _ in S.UNITS} == set(S.UNIT_KEYS) and len(S.UNIT_KEYS) == 24
    for n in (1, 2, 3, 7, 8):
        parts = [repro.portion(S.UNITS, (k, n)) for k in range(n)]
        flat = [x for p in parts for x in p]
        assert sorted(flat) == sorted(S.UNITS) and len(flat) == 192
    assert repro.portion(S.UNITS, None) == list(S.UNITS)


def test_paper_checks_hold_from_the_stored_summary():
    rows = S.paper_checks()
    assert len(rows) >= 30
    bad = [r for r in rows if abs(float(r[1]) - float(r[2])) > r[3]]
    assert not bad, bad


def test_derived_summaries_rebuild_from_the_stored_trials(tmp_path):
    for f in ("criteria_pool_trials.json.gz", "phase1_bases.json"):
        (tmp_path / f).write_bytes((S.summary_dir() / f).read_bytes())
    S.derive(tmp_path)
    for f in ("selection.json", "methods.json", "map_cells.json"):
        assert json.loads((tmp_path / f).read_text()) == json.loads((S.summary_dir() / f).read_text()), f


def test_render_writes_the_paper_table_rows_and_the_figure(tmp_path):
    out = S.render(out_dir=tmp_path)
    tex = (tmp_path / "tables" / "selection-dev.tex").read_text()
    assert ("cef & \\textbf{0.721} & 0.707 & 0.666 & \\textbf{0.700} & \\textbf{0.688} & 0.688 & \\textbf{0.684} & "
            "\\textbf{0.655} & 0.576 & \\textbf{0.689} & -- & \\textbf{4.71} \\\\") in tex
    assert "SND & 0.499 & 0.506 & 0.500 & 0.500 & 0.501 & 0.498 & 0.526 & 0.499 &  & 0.504 & 0.065 / +0.20 & 6.64 \\\\" in tex
    assert "Method rank & \\textbf{3.33} & 3.79 & 5.96 & 4.62 & 5.58 & 5.27 & 5.48 & 5.17 & 5.79 &  &  &  \\\\" in tex
    assert (tmp_path / "tables" / "selection-dev.csv").read_text().count("\n") == 13
    assert (tmp_path / "figures" / "synthetic-map-methods.pdf").stat().st_size > 5000
    assert (tmp_path / "figures" / "synthetic-map-methods.png").exists()
    assert out["panels"]["MIL-CREDA"][-1] == [0.021, 0.367, 0.492]        # rows by class effect, 2.5 last


def test_stored_picks_are_the_first_cef_maximum_of_the_stored_trials():
    pools = S._pools(S.load_trials())
    mil = json.loads((S.summary_dir() / "milcreda_picks.json").read_text())["units"]
    comp = json.loads((S.summary_dir() / "comparator_picks.json").read_text())["units"][S.FIGURE_UNIT]
    assert sorted(mil) == sorted(S.UNIT_KEYS) and sorted(comp) == sorted(S.COMPARATORS)
    for u, p in mil.items():
        assert S._first_max(pools[(u, "MIL-CREDA")], "cef")["trial"] == p["trial"]
    for m, p in comp.items():
        assert S._first_max(pools[(S.FIGURE_UNIT, m)], "cef")["trial"] == p["trial"]


def _fake_raw(root: Path) -> None:
    """Two shards of one unit in the criteria_pool.run layout, with tied cef in trial order."""
    rng = np.random.default_rng(0)
    ys = [0, 1] * 6
    for sh, trials in ((0, (0, 1)), (1, (2, 3))):
        stem = f"criteria-d1-s3-r0-k1-sh{sh}of2"
        d = root / f"sx-d1-s3-r0-{sh}"
        (d / stem).mkdir(parents=True)
        rows = {}
        for i in trials:
            pt = rng.dirichlet([1, 1], 20).astype(np.float32)
            pt[:, 0] = np.where(np.arange(20) % 2 == 0, 0.9, 0.1)   # balanced argmax: every trial has cef 2
            pt[:, 1] = 1 - pt[:, 0]
            np.savez_compressed(d / stem / f"MIL-CREDA-{i}.npz", Pt=pt, Zt=rng.normal(size=(20, 4)).astype(np.float32),
                                Ps=rng.dirichlet([1, 1], 12).astype(np.float32),
                                Zs=rng.normal(size=(12, 4)).astype(np.float32))
            rows[str(i)] = {"params": {"lr2": 0.001 * (i + 1)}, "BApool": 0.5 + 0.1 * i, "AUCpool": 0.6,
                            "BAtest12": 0.5, "AUCtest12": 0.5}
        rows["9"] = {"params": {}, "nonFinite": True}
        (d / f"{stem}.json").write_text(json.dumps({"unitKey": "d1-s3-r0", "sourceTestLabels": ys,
                                                    "methods": {"MIL-CREDA": {"trials": rows}}}))


def test_merge_ports_crit3_and_mkpicks_from_raw_outputs(tmp_path, capsys):
    _fake_raw(tmp_path / "raw")
    out = S.merge(raw=tmp_path / "raw", out=tmp_path / "summary")
    with gzip.open(out / "criteria_pool_trials.json.gz", "rt") as f:
        t = json.load(f)
    assert t["criteria"] == list(S.CRITERIA)
    assert [r["trial"] for r in t["rows"]] == [0.0, 1.0, 2.0, 3.0]          # non-finite trial dropped, trial order
    assert set(t["rows"][0]) == {"unit", "method", "trial", "pool", "test", *S.CRITERIA}
    assert all(r["cef"] == pytest.approx(2.0) for r in t["rows"])
    pick = json.loads((out / "milcreda_picks.json").read_text())["units"]["d1-s3-r0"]
    assert pick["trial"] == 0 and pick["params"] == {"lr2": 0.001}           # tie: first maximum in trial order
    assert "derived tables not written" in capsys.readouterr().out
    assert not (out / "selection.json").exists()


def test_merge_without_raw_keeps_the_stored_summary(tmp_path, capsys):
    assert S.merge(raw=tmp_path / "none", out=tmp_path / "s") == tmp_path / "s"
    assert "keeping the stored summary" in capsys.readouterr().out


@pytest.mark.skipif(not CAMPAIGN.exists(), reason="v82 campaign files are only on the campaign machine")
def test_stored_summary_matches_the_v82_campaign_files():
    with gzip.open(S.summary_dir() / "criteria_pool_trials.json.gz", "rb") as f:
        assert f.read() == (CAMPAIGN / "criteria_pool_trials.json").read_bytes()
    for f in ("phase1_bases.json", "milcreda_picks.json", "comparator_picks.json"):
        assert (S.summary_dir() / f).read_bytes() == (CAMPAIGN / f).read_bytes(), f
    sel = json.loads((S.summary_dir() / "selection.json").read_text())
    h3 = json.loads((CAMPAIGN / "h3_pool_summary.json").read_text())
    assert [r["criterion"] for r in sel["table"]] == [r["criterion"] for r in h3["table"]]
    for a, b in zip(sel["table"], h3["table"]):
        assert a["methods"] == b["methods"] and a["meanBA"] == b["meanBA"] and a["meanRank"] == b["meanRank"]
        if "pHolm" in b:
            assert a["pHolm"] == pytest.approx(b["pHolm"], rel=1e-9) and a["cliff"] == b["cliff"]
    met = json.loads((S.summary_dir() / "methods.json").read_text())
    h1 = json.loads((CAMPAIGN / "h1_pool24_summary.json").read_text())
    assert met["table"] == h1["table"] and met["nemenyiCD"] == h1["nemenyiCD"]
    for a, b in zip(met["milcredaVs"], h1["milcredaVs"]):
        assert a["pHolm"] == pytest.approx(b["pHolm"], rel=1e-9) and a["cliffDelta"] == b["cliffDelta"]


@pytest.mark.skipif(not (FIGURE_DATA / "pool24_map_cells.json").exists(), reason="paper workspace not present")
def test_map_cells_equal_the_paper_figure_data():
    a = json.loads((S.summary_dir() / "map_cells.json").read_text())
    b = json.loads((FIGURE_DATA / "pool24_map_cells.json").read_text())
    assert a["dBA"] == b["dBA"] and a["nRotations"] == b["nRotations"]


def test_check_bases_accepts_the_stored_v82_bases_and_refuses_others(tmp_path):
    S.check_bases(S.bases_dir())
    # the older bases still in Manifests/oracle_bases (if any) have a different cut
    old = repro.eeg_dir() / "Manifests" / "oracle_bases" / f"{S.FIGURE_UNIT}-s1.pt"
    if old.exists() and old.read_bytes() != (S.bases_dir() / old.name).read_bytes():
        with pytest.raises(ValueError, match="not the v82 base"):
            S.check_bases(old.parent, [S.FIGURE_UNIT])


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("REPRO_SLOW") != "1", reason="about 60 s of CPU; set REPRO_SLOW=1")
def test_smoke_run_trains_one_trial_and_merges(tmp_path):
    files = S.run(smoke=True, out=tmp_path)
    d = json.loads(Path(files[0]).read_text())
    assert list(d["methods"]["MIL-CREDA"]["trials"]) == ["0"] and d["evalSubjects"] == 12
    out = S.merge(raw=tmp_path, out=tmp_path / "summary")
    with gzip.open(out / "criteria_pool_trials.json.gz", "rt") as f:
        assert len(json.load(f)["rows"]) == 1
