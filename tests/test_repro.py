"""The shared notebook plumbing: account portions, the IEEE manifest and the data root."""
import pytest

from MIL_CREDA_EEG import repro, xds_pilot


def test_portion_none_runs_everything():
    assert repro.portion(range(5)) == [0, 1, 2, 3, 4]
    assert repro.portion(range(5), None) == [0, 1, 2, 3, 4]


def test_portions_cover_every_item_once():
    items = list(range(23))
    parts = [repro.portion(items, (k, 4)) for k in range(4)]
    assert sorted(x for p in parts for x in p) == items
    assert repro.portion(items, "1/4") == parts[1]


@pytest.mark.parametrize("bad", [(4, 4), (-1, 3), (0, 0), "3/2"])
def test_bad_split_is_refused(bad):
    with pytest.raises(ValueError):
        repro.parse_split(bad)


def test_manifest_lists_the_121_ieee_subjects():
    m = repro.manifest()
    files = m["files"]
    assert m["dataset"] == repro.DATASET
    assert len(files) == len(set(files)) == 121
    assert sum(f.startswith("ADHD_group/") for f in files) == 61
    assert sum(f.startswith("Control_group/") for f in files) == 60


def test_ieee_root_uses_a_given_copy(tmp_path, monkeypatch):
    (tmp_path / "ADHD_group").mkdir()
    (tmp_path / "ADHD_group" / "v1p.mat").write_bytes(b"")
    monkeypatch.setenv("IEEE_ROOT", str(tmp_path))
    assert repro.ieee_root() == str(tmp_path)
    assert xds_pilot.find_ieee(str(tmp_path)) == str(tmp_path)


def test_find_ieee_without_kaggle_and_without_download_still_refuses(monkeypatch):
    monkeypatch.setattr(xds_pilot.glob, "glob", lambda *a, **k: [])
    monkeypatch.setenv("MIL_CREDA_NO_DOWNLOAD", "1")
    with pytest.raises(FileNotFoundError):
        xds_pilot.find_ieee(None)


def test_find_ieee_outside_kaggle_falls_back_to_the_public_dataset(monkeypatch):
    monkeypatch.setattr(xds_pilot.glob, "glob", lambda *a, **k: [])
    monkeypatch.delenv("MIL_CREDA_NO_DOWNLOAD", raising=False)
    monkeypatch.setattr(repro, "ieee_root", lambda: "/downloaded")
    assert xds_pilot.find_ieee(None) == "/downloaded"


def test_compare_flags_out_of_tolerance(capsys):
    rows = repro.compare([("BA", 0.70, 0.71, 0.02), ("AUC", 0.60, 0.70, 0.02)])
    assert [r["within"] for r in rows] == [True, False]
    assert "NO" in capsys.readouterr().out
