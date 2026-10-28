"""Experiments plan v05: the reproducible entry point, on a synthetic mini-release.

The notebook calls `preprocess.build_processed`; the command line runs
`python -m MIL_CREDA_EEG.preprocess`. Both must write the same outputs from
the same raw tree.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from MIL_CREDA_EEG import modality, preprocess

REPO = Path(__file__).resolve().parents[1]


def _cgx_file(path: Path, seconds: float, seed: int) -> None:
    rate = 500.0
    n = int(seconds * rate)
    rng = np.random.default_rng(seed)
    columns = (["timestamps"] + [f"{c}(uV)" for c in modality.CGX_CHANNELS]
               + ["A2(uV)", "ExG 1(uV)", "ExG 2(uV)", "ACC32(mg)", "Packet Counter", "TRIGGER"])
    data = np.column_stack(
        [1000 + np.arange(n) / rate] + [rng.normal(scale=20, size=n) for _ in modality.CGX_CHANNELS]
        + [rng.normal(size=n), np.full(n, -833333.3125), np.full(n, -833333.3125),
           np.zeros(n), np.arange(n) % 128, np.zeros(n)])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        np.savetxt(handle, data, delimiter=",", fmt="%.6f")


def _emotiv_file(path: Path, seconds: float, seed: int, cq_level: float) -> None:
    rate = 128.0
    n = int(seconds * rate)
    rng = np.random.default_rng(seed)
    ch = modality.EMOTIV_CHANNELS
    columns = (["Timestamp", "EEG.Counter", "EEG.Interpolated"] + [f"EEG.{c}" for c in ch]
               + [f"CQ.{c}" for c in ch] + ["CQ.Overall", "EQ.OVERALL", "PM.Focus.Scaled"])
    focus = np.full(n, np.nan)
    focus[::1280] = 0.5
    data = np.column_stack(
        [1.6e9 + np.arange(n) / rate, np.arange(n) % 128, np.zeros(n)]
        + [4200 + rng.normal(scale=5, size=n) for _ in ch]
        + [np.full(n, cq_level) for _ in ch] + [np.full(n, 100.0), np.full(n, 100.0), focus])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("title:x, sampling rate:eeg_128;mot_32, samples:1\n")
        handle.write(",".join(columns) + "\n")
        np.savetxt(handle, data, delimiter=",", fmt="%.6f")


def _mini_release(root: Path) -> Path:
    raw = root / "raw"
    _cgx_file(raw / "UB0001/SlacklineLvl1/1000/UB0001_EEG_CGX_2023_01_01T00.00.00.csv", 21, 1)
    _emotiv_file(raw / "UB0001/AttentionRobotsDesktop/1001/UB0001_EPOCX_1_2023.md.pm.bp.csv",
                 21, 2, 4.0)
    _emotiv_file(raw / "UB0002/Cognifit/2000/UB0002_EPOCX_1_2023.md.pm.bp.csv", 21, 3, 0.0)
    _cgx_file(raw / "UB0003/SlacklineLvl6/3000/UB0003_EEG_CGX_2023_01_02T00.00.00.csv", 13, 4)
    (raw / "users_demographics.json").write_text(json.dumps([
        {"user": "UB0001", "diagnosed": "yes", "gender": 1, "age": 9, "group": 1},
        {"user": "UB0002", "diagnosed": "no", "gender": 2, "age": 11, "group": 2},
        {"user": "UB0003", "diagnosed": "undetermined", "gender": 1, "age": 10, "group": 1},
    ]), encoding="utf-8")
    return raw


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("mini")
    raw = _mini_release(root)
    function_out = root / "by_function"
    result = preprocess.build_processed(raw, function_out, workers=1, expected_subjects=3)
    cli_out = root / "by_cli"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    run = subprocess.run([sys.executable, "-m", "MIL_CREDA_EEG.preprocess", "--raw", str(raw),
                          "--out", str(cli_out), "--workers", "2", "--expected-subjects", "3"],
                         cwd=root, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    return {"raw": raw, "function": function_out, "cli": cli_out, "result": result,
            "stdout": run.stdout}


def test_the_cli_and_the_notebook_entry_function_write_identical_outputs(built):
    a, b = built["function"], built["cli"]
    for name in preprocess.OUTPUT_FILES:
        assert (a / name).read_text(encoding="utf-8") == (b / name).read_text(encoding="utf-8"), name
    records_a = sorted(p.name for p in (a / "records").glob("*.json"))
    assert records_a == sorted(p.name for p in (b / "records").glob("*.json"))
    for name in records_a:
        assert (json.loads((a / "records" / name).read_text())
                == json.loads((b / "records" / name).read_text()))
    signals = sorted(p.name for p in (a / "signals").glob("*.npz"))
    assert signals == sorted(p.name for p in (b / "signals").glob("*.npz"))
    assert len(signals) == 3                     # four records, one excluded by contact
    for name in signals:
        with np.load(a / "signals" / name) as x, np.load(b / "signals" / name) as y:
            assert sorted(x.files) == sorted(y.files)
            for key in x.files:
                np.testing.assert_array_equal(x[key], y[key])
    assert json.loads(built["stdout"])["excluded"] == 1


def test_the_mini_release_goes_through_every_rule(built):
    manifest = built["result"]["manifest"].set_index("record_id")
    excluded = manifest[manifest["excluded"]]
    assert list(excluded["exclusion_step"]) == ["2.5 contacto"]
    assert excluded.index[0].startswith("UB0002__Cognifit")
    undetermined = manifest[manifest["undetermined"]]
    assert len(undetermined) == 1 and not undetermined["excluded"].any()
    cgx = manifest[manifest["family"] == "CGX"]
    assert sorted(cgx["n_windows"]) == [2, 4]    # 13 s and 21 s, minus the first window
    with np.load(built["function"] / manifest.loc[cgx.index[0], "npz"]) as z:
        assert z["signal"].shape[1:] == (29, 512)
        assert not {"A2", "ExG 1", "ExG 2"} & set(z["channels"])


def test_rebuild_clears_only_the_files_this_pipeline_writes(built, tmp_path):
    out = tmp_path / "out"
    preprocess.build_processed(built["raw"], out, workers=1, expected_subjects=3)
    stray = out / "notes.txt"
    stray.write_text("keep me", encoding="utf-8")
    (out / "signals" / "notes.txt").write_text("keep me too", encoding="utf-8")
    first = (out / "manifest.csv").read_text(encoding="utf-8")
    result = preprocess.build_processed(built["raw"], out, workers=1, expected_subjects=3)
    assert result["cleared"] == 4 + 4 + 3        # CSVs, record JSONs, npz files
    assert stray.read_text(encoding="utf-8") == "keep me"
    assert (out / "signals" / "notes.txt").is_file()
    assert (out / "manifest.csv").read_text(encoding="utf-8") == first


def test_a_raw_tree_without_demographics_or_with_the_wrong_subject_count_refuses(built, tmp_path):
    with pytest.raises(preprocess.RawDataError, match="Se esperaban 158"):
        preprocess.check_raw(built["raw"])
    with pytest.raises(preprocess.RawDataError, match="No existe"):
        preprocess.check_raw(tmp_path / "nowhere", expected_subjects=3)
    empty = tmp_path / "raw"
    (empty / "UB0001").mkdir(parents=True)
    with pytest.raises(preprocess.RawDataError, match="users_demographics.json"):
        preprocess.check_raw(empty, expected_subjects=1)
    assert preprocess.check_raw(built["raw"], expected_subjects=3)["eeg_records"] == 4


def test_a_zip_whose_md5_differs_refuses(tmp_path):
    assert preprocess.check_zip(tmp_path)["verified"] is False
    (tmp_path / "BALLADEER_ADHD_DATASET.zip").write_bytes(b"not the release")
    with pytest.raises(preprocess.RawDataError, match="MD5"):
        preprocess.check_zip(tmp_path)
    good = preprocess.file_md5(tmp_path / "BALLADEER_ADHD_DATASET.zip")
    assert preprocess.check_zip(tmp_path, expected=good)["verified"] is True
