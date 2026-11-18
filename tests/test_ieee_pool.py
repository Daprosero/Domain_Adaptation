"""The IEEE shared-pool runner: its draw, its sharding, its bases and one short real run.

The IEEE ADHD recordings are not on every machine (`xds_pilot.find_ieee` reads `IEEE_ROOT` or
/kaggle/input), so the real-run tests write a tiny fixture in the IEEE layout (ADHD_group/ and
Control_group/ .mat files, 19 channels at 128 Hz) under pytest's temporary directory and point
`IEEE_ROOT` at it. Phase 1 is fitted on it at a 1-epoch setting, only to have a base to start from.
"""
import json

import numpy as np
import pytest
import scipy.io
from sklearn.metrics import cohen_kappa_score

from MIL_CREDA_EEG import criteria_pool as C
from MIL_CREDA_EEG import ieee_pool as I
from MIL_CREDA_EEG import xds_pilot as X

UNIT = "ieee-r0h0"
CONFIG = {"device": "cpu", "nTrials": 20, "rehearsal": True,
          "remoteInputs": {"processed": "daprosero/balladeer-existing-local"},
          "schedule": {"bagsPerStep": 8, "epochs1": 100, "epochs2": 100},
          "strategies": ["AdaptMIL", "UDA-DDA", "CCA-LSC", "CASUAL", "COT", "CS+CCS",
                         "CLUST — variante source-free", "CLUDA", "MIL-CREDA"],
          "transfers": ["L1->L6"]}
PHASE1 = {"epochs": 3, "l2": 0.015, "lr1": 0.004, "tauAtt1": 4.8, "width": 64}
METHODS = ("MIL-CREDA", "UDA-DDA")
SIDES = ("target", "target_test", "source_test", "source")


def test_the_ten_units_are_five_rotations_by_two_halves():
    assert I.UNITS == tuple(f"ieee-r{r}h{h}" for r in range(5) for h in range(2))
    assert I.unit_of("ieee-r3h1") == (3, 1)
    with pytest.raises(ValueError, match="ieee-r"):
        I.unit_of("d1p5-s1p5-r0")
    assert I.base_file(UNIT, "/x").as_posix() == "/x/ieee-r0h0-s1.pt"
    assert I.BASES.name == "ieee_pool_bases"


def test_the_draws_are_the_synthetic_pool_draws_at_phase_two_seed_one():
    assert I.PHASE2_SEED == 1 and I.METHODS == C.METHODS
    assert I.EPOCHS2 == C.EPOCHS2 == 100 and I.TRIALS == C.TRIALS == 100
    for m in I.METHODS:
        assert I.draw(m) == C.draw(m, 1)


def test_eight_shards_partition_the_hundred_trials_in_twelve_or_thirteen():
    parts = [list(I.shard_trials(s)) for s in range(I.SHARDS)]
    assert I.SHARDS == 8
    assert [i for p in parts for i in p] == list(range(100))
    assert {len(p) for p in parts} == {12, 13}


def test_the_recorded_phase_one_settings_are_read_from_the_rehearsal_run_config(tmp_path):
    d = tmp_path / "acct" / "papersmith-ie-r2h1-cef"
    d.mkdir(parents=True)
    kw = {"phase1_params": PHASE1, "config": CONFIG, "rotation": 2, "half": 1, "experiment": "ieee"}
    (d / "run-config.json").write_text(json.dumps({"run": {"kwargs": kw}}))
    assert I.recorded_settings(tmp_path, 2, 1) == (PHASE1, CONFIG)
    with pytest.raises(FileNotFoundError, match="ie-r0h0"):
        I.recorded_settings(tmp_path, 0, 0)
    (d / "run-config.json").write_text(json.dumps({"run": {"kwargs": {**kw, "half": 0}}}))
    with pytest.raises(ValueError, match="half"):
        I.recorded_settings(tmp_path, 2, 1)


def _fixture(root, per_class=30, seconds=12):
    rng = np.random.default_rng(0)
    t = np.arange(seconds * X.RATE) / X.RATE
    for group, label in (("ADHD_group", 1), ("Control_group", 0)):
        (root / group).mkdir(parents=True)
        for i in range(per_class):
            x = rng.normal(0, 10, (len(t), len(X.IEEE_CHANNELS)))
            # on half the channels only: the instance rule removes the channel mean
            x[:, :10] += (2.0 * label + 1.0 + rng.normal(0, 1.5)) * np.sin(2 * np.pi * 10.0 * t)[:, None]
            scipy.io.savemat(root / group / f"{'a' if label else 'c'}{i:02d}.mat", {"x": x})


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("ieee")
    _fixture(root / "data")
    mp = pytest.MonkeyPatch()
    mp.setenv("IEEE_ROOT", str(root / "data"))
    bases = root / "bases"
    bases.mkdir()
    path = I.build_base(0, 0, PHASE1, CONFIG, bases)
    common = dict(shard=0, shards=1, trials=2, methods=METHODS, bases=bases, epochs=2, procs=1, threads=1)
    out = I.run(UNIT, workdir=root / "with", **common)
    plain = I.run(UNIT, workdir=root / "without", mid=False, **common)
    yield {"root": root, "base": path, "bases": bases, "out": out, "plain": plain, "common": common}
    mp.undo()


def test_the_base_keeps_the_make_bases_layout_and_the_phase_one_prediction(world):
    import torch
    from MIL_CREDA_EEG import spectral_trunk
    assert world["base"].name == "ieee-r0h0-s1.pt"
    with spectral_trunk.installed():
        b = torch.load(world["base"], weights_only=False)
    for k in ("unitKey", "unit", "seed", "rotation", "phase1", "unitKwargs", "config", "cut", "arm", "context",
              "sourceTest", "predict", "commit", "torch", "base", "prepared"):
        assert k in b, k
    assert (b["unitKey"], b["unit"], b["seed"], b["rotation"]) == (UNIT, "ieee:0:0", 1, 0)
    assert b["unitKwargs"] == {"half": 0} and b["phase1"] == PHASE1 and b["config"] == CONFIG
    pred = b["predict"]
    for k in ("BA", "AUC", "kappa", "rows", "subjects", "probabilities", "labels"):
        assert k in pred, k
    assert len(pred["subjects"]) == len(pred["probabilities"]) == len(pred["labels"]) == len(pred["rows"])
    hard = (np.asarray(pred["probabilities"]) > b["cut"]).astype(int)
    assert np.isclose(pred["kappa"], cohen_kappa_score(pred["labels"], hard), atol=1e-4)


def test_a_short_real_run_writes_every_declared_key_and_array(world):
    out = world["out"]
    assert out == {"file": "ieeepool-ieee-r0h0-sh0of1.json", "methods": {m: 2 for m in METHODS}}
    rec = json.loads((world["root"] / "with" / out["file"]).read_text())
    for k in ("unitKey", "unit", "shard", "shards", "trials", "epochs2", "cut", "baseCommit", "baseSeed",
              "phase2Seed", "schedule", "targetTestLabels", "targetTestSubjects", "targetSubjects",
              "sourceTestLabels", "sourceTestSubjects", "sourceSubjects", "procs", "threads", "methods"):
        assert k in rec, k
    assert (rec["unitKey"], rec["baseSeed"], rec["phase2Seed"], rec["epochs2"]) == (UNIT, 1, 1, 2)
    arr = world["root"] / "with" / "ieeepool-ieee-r0h0-sh0of1"
    for m in METHODS:
        for i in ("0", "1"):
            row = rec["methods"][m]["trials"][i]
            assert row["params"] == I.draw(m)[int(i)]
            if row.get("nonFinite"):
                continue
            for k in ("BAtest", "AUCtest", "kappaTest", "stepTotals", "seconds"):
                assert k in row, k
            z = np.load(arr / f"{m.split(' ')[0]}-{i}.npz")
            for k in ("Pt", "Zt", "Ps", "Zs", "Ptest", "Zsrc", "Ztest"):
                assert k in z.files, k
            assert len(z["Ptest"]) == len(rec["targetTestSubjects"]) == len(z["Ztest"])
            assert len(z["Zsrc"]) == len(rec["sourceSubjects"]) and len(z["Ps"]) == len(rec["sourceTestLabels"])
            assert len(z["Pt"]) == len(rec["targetSubjects"])
            for side in SIDES:
                off, w = z[f"attOffsets[{side}]"], z[f"att[{side}]"]
                assert off[0] == 0 and off[-1] == len(w)
            mid = {k for k in z.files if k.startswith("Zmid")}
            assert mid == ({"Zmid[target]", "Zmid[target_test]", "Zmid[source]"} if m == "MIL-CREDA" else set())
            assert ("mid" in row) == (m == "MIL-CREDA")


def test_the_attention_offsets_follow_the_bag_sizes(world):
    import torch
    from MIL_CREDA_EEG import spectral_trunk
    from MIL_CREDA_EEG import xpilot as XP
    from MIL_CREDA_EEG import campaign, remote_units
    with spectral_trunk.installed():
        b = torch.load(world["base"], weights_only=False)
    cfg = remote_units.resolve_inputs(campaign.check_config(b["config"]), environ={})
    sides = XP.build_sides(cfg, {"IEEE_ROOT": str(world["root"] / "data")}, experiment="ieee", rotation=0,
                           half=0)["sides"]
    z = np.load(world["root"] / "with" / "ieeepool-ieee-r0h0-sh0of1" / "MIL-CREDA-0.npz")
    for side in SIDES:
        sizes = [int(x.windows.shape[0]) for x in sides[side]]
        assert z[f"attOffsets[{side}]"].tolist() == np.concatenate([[0], np.cumsum(sizes)]).tolist()
        w = z[f"att[{side}]"]
        for a, c in zip(z[f"attOffsets[{side}]"][:-1], z[f"attOffsets[{side}]"][1:]):
            assert np.isclose(w[a:c].sum(), 1.0, atol=1e-4)


def test_the_mid_snapshot_leaves_the_final_outputs_identical(world):
    a = json.loads((world["root"] / "with" / world["out"]["file"]).read_text())
    b = json.loads((world["root"] / "without" / world["plain"]["file"]).read_text())
    arr_a = world["root"] / "with" / "ieeepool-ieee-r0h0-sh0of1"
    arr_b = world["root"] / "without" / "ieeepool-ieee-r0h0-sh0of1"
    for i in ("0", "1"):
        ra, rb = a["methods"]["MIL-CREDA"]["trials"][i], b["methods"]["MIL-CREDA"]["trials"][i]
        assert "mid" not in rb and ra["mid"]["epoch"] == 1
        drop = ("seconds", "mid")
        assert {k: v for k, v in ra.items() if k not in drop} == {k: v for k, v in rb.items() if k not in drop}
        za, zb = np.load(arr_a / f"MIL-CREDA-{i}.npz"), np.load(arr_b / f"MIL-CREDA-{i}.npz")
        assert set(zb.files) == set(za.files) - {"Zmid[target]", "Zmid[target_test]", "Zmid[source]"}
        for k in zb.files:
            assert np.array_equal(za[k], zb[k]), k


def test_kappa_and_ba_are_the_cut_predictions_of_the_saved_test_probabilities(world):
    from sklearn.metrics import balanced_accuracy_score
    rec = json.loads((world["root"] / "with" / world["out"]["file"]).read_text())
    y = np.asarray(rec["targetTestLabels"])
    for m in METHODS:
        row = rec["methods"][m]["trials"]["0"]
        if row.get("nonFinite"):
            continue
        p = np.load(world["root"] / "with" / "ieeepool-ieee-r0h0-sh0of1" / f"{m.split(' ')[0]}-0.npz")["Ptest"]
        hard = (p > rec["cut"]).astype(int)
        po = np.mean(hard == y)
        pe = np.mean(hard) * np.mean(y) + np.mean(1 - hard) * np.mean(1 - y)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
        assert np.isclose(row["kappaTest"], kappa, atol=1e-4)
        assert np.isclose(row["BAtest"], balanced_accuracy_score(y, hard), atol=1e-4)


def test_a_second_call_resumes_without_retraining(world):
    again = I.run(UNIT, workdir=world["root"] / "with", **world["common"])
    assert again == world["out"]


def test_worker_processes_reproduce_the_single_process_rows(world):
    two = I.run(UNIT, workdir=world["root"] / "two", **{**world["common"], "procs": 2})
    assert two == world["out"]
    a = json.loads((world["root"] / "with" / two["file"]).read_text())
    b = json.loads((world["root"] / "two" / two["file"]).read_text())
    assert b["procs"] == 2
    for m in METHODS:
        for i in ("0", "1"):
            ra, rb = a["methods"][m]["trials"][i], b["methods"][m]["trials"][i]
            assert {k: v for k, v in ra.items() if k != "seconds"} == {k: v for k, v in rb.items() if k != "seconds"}


def test_the_bases_command_fits_each_unit_at_its_recorded_settings(world, tmp_path):
    d = tmp_path / "rehearsal" / "acct" / "papersmith-ie-r0h0-cef"
    d.mkdir(parents=True)
    kw = {"phase1_params": PHASE1, "config": CONFIG, "rotation": 0, "half": 0, "experiment": "ieee"}
    (d / "run-config.json").write_text(json.dumps({"run": {"kwargs": kw}}))
    I.main(["bases", str(tmp_path / "rehearsal"), str(tmp_path / "out"), "--units", UNIT])
    import torch
    from MIL_CREDA_EEG import spectral_trunk
    with spectral_trunk.installed():
        a = torch.load(tmp_path / "out" / "ieee-r0h0-s1.pt", weights_only=False)
        b = torch.load(world["base"], weights_only=False)
    assert a["phase1"] == PHASE1 and a["cut"] == b["cut"] and a["predict"] == b["predict"]
