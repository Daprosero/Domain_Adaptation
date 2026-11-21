"""Stage 01 (phase-1 bases): work items, stored summary, render and the paper's numbers."""
import csv
import os

import pytest

from MIL_CREDA_EEG import repro
from MIL_CREDA_EEG.reproduce import stage01_bases as S


def test_units_cover_the_34_bases_once():
    assert len(S.UNITS) == 13
    keys = [k for u in S.UNITS for k in S.base_keys(u)]
    assert len(keys) == len(set(keys)) == 34
    assert sum(k.startswith("ieee") for k in keys) == 10


@pytest.mark.parametrize("n", [1, 2, 3, 5, 13])
def test_portions_partition_the_units(n):
    parts = [repro.portion(S.UNITS, (k, n)) for k in range(n)]
    flat = [u for p in parts for u in p]
    assert sorted(flat, key=S.UNITS.index) == list(S.UNITS)
    assert len(flat) == len(set(flat))


def test_stored_summary_holds_every_base():
    s = S.load_summary()
    assert [r["unitKey"] for r in s["bases"]] == list(S.ALL_KEYS)
    syn = [r for r in s["bases"] if r["family"] == "synthetic"]
    assert all(r["evalPoolBA"] is not None and r["searchScore"] is not None for r in syn)
    assert set(s["adhdSubjects"]) == {k for k in S.ALL_KEYS if k.startswith("ieee")}


def test_paper_checks_hold_exactly_on_the_stored_summary():
    rows = repro.compare(S.paper_checks())
    assert len(rows) == 4 and all(r["within"] and r["tolerance"] == 0 for r in rows)


def test_render_writes_the_table_from_the_summary(tmp_path):
    out = S.render(out_dir=tmp_path)
    with open(out["csv"]) as fh:
        rows = list(csv.DictReader(fh))
    assert [r["unitKey"] for r in rows] == list(S.ALL_KEYS)
    tex = out["tex"].read_text()
    assert tex.count("\\\\") == 34 + 3 and "ieee-r4h1" in tex
    with open(out["numbers"]) as fh:
        nums = list(csv.DictReader(fh))
    assert [float(r["value"]) for r in nums] == [float(r["paper"]) for r in nums]


def test_bases_dir_points_at_tracked_or_regenerated():
    assert S.bases_dir().parts[-2:] == ("Manifests", "oracle_bases")
    assert S.bases_dir(False, "adhd").name == "ieee_pool_bases"
    assert S.bases_dir(True, "adhd").parts[-4:] == ("01_phase1_bases", "regenerated", "bases", "ieee_pool_bases")


def test_regenerated_summary_uses_the_regeneration_tolerance(tmp_path):
    S.write_summary(S.load_summary(), tmp_path, "copy")
    assert all(t == S.REGEN_TOLERANCE for *_, t in S.paper_checks(tmp_path))


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("REPRO_SLOW") != "1" or not os.environ.get("IEEE_ROOT"),
                    reason="about 5 minutes: set REPRO_SLOW=1 and IEEE_ROOT (the public IEEE ADHD folder)")
def test_smoke_wiring(tmp_path):
    recs = S.run(smoke=True, out=tmp_path)
    assert [r["unit"] for r in recs] == ["p1A-r0", "sp1-d1p5-r0"]
    S.merge(tmp_path)
    got = {r["unitKey"] for r in S.load_summary(tmp_path / "summary")["bases"]}
    assert got == {"ieee-r0h0", "ieee-r0h1", "d1p5-s0p5-r0", "d1p5-s1p5-r0", "d1p5-s3-r0"}
    assert S.compare_regenerated(tmp_path / "summary")
