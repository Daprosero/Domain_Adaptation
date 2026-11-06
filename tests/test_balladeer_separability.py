"""Experiments plan v50, Secs. 4.3 and 5.1: domain separability re-measured, and the core rule."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from MIL_CREDA_EEG import assignment, audit_domains, bag_data, partition, separability
from MIL_CREDA_EEG.modality import CGX_CHANNELS, EMOTIV_CHANNELS


def _means(**pairs: float) -> pd.DataFrame:
    """Per-transfer mean AUC (reading 'all'); keys like 'level:L1:L6'."""
    rows = []
    for key, value in pairs.items():
        arm, source, target = key.split("__")
        rows.append({"arm": arm, "source": source, "target": target, "auc_mean": value})
    return pd.DataFrame(rows)


def test_core_is_the_pair_with_the_highest_mean_of_both_directions():
    table = _means(level__L1__L6=0.70, level__L6__L1=0.40,      # pair mean 0.55
                   level__L1__L11=0.60, level__L11__L1=0.60,    # pair mean 0.60  <- core
                   level__L6__L11=0.50, level__L11__L6=0.52)    # pair mean 0.51
    roles = separability.core_control(table)
    by = {(r.source, r.target): r for r in roles.itertuples()}
    assert by[("L1", "L11")].role == by[("L11", "L1")].role == "core"
    assert {by[k].role for k in [("L1", "L6"), ("L6", "L1"), ("L6", "L11"),
                                 ("L11", "L6")]} == {"control"}
    assert by[("L1", "L6")].pair_auc == pytest.approx(0.55)
    # The single highest direction (L1->L6, 0.70) does not decide: the pair mean does.


def test_single_pair_arms_are_core():
    table = _means(task__Robots__CogniFit=0.51, task__CogniFit__Robots=0.50,
                   device__L6__Robots=0.99, device__Robots__L6=0.98)
    roles = separability.core_control(table)
    assert set(roles["role"]) == {"core"}


def test_a_tie_for_core_refuses():
    table = _means(level__L1__L6=0.6, level__L6__L1=0.6, level__L1__L11=0.6,
                   level__L11__L1=0.6, level__L6__L11=0.5, level__L11__L6=0.5)
    with pytest.raises(separability.SeparabilityError, match="empate"):
        separability.core_control(table)


def test_a_missing_direction_refuses():
    table = _means(level__L1__L6=0.6, level__L1__L11=0.6, level__L11__L1=0.6)
    with pytest.raises(separability.SeparabilityError, match="dirección"):
        separability.core_control(table)


def test_the_instrument_is_the_audits():
    assert separability.N_PERMUTATIONS == audit_domains.N_PERMUTATIONS == 200
    assert separability.SEED == audit_domains.SEED
    assert separability.CHANNELS["level"] == list(CGX_CHANNELS)
    assert separability.CHANNELS["task"] == list(EMOTIV_CHANNELS)


def test_device_arm_is_measured_on_the_common_channels():
    common = separability.CHANNELS["device"]
    assert common == audit_domains.common_channels() and len(common) == 12
    assert all(c in CGX_CHANNELS and c in EMOTIV_CHANNELS for c in common)


# ------------------------------------------------------------- on bags


def _bag(root: Path, subject: str, condition: str, label: str, fold: int, alpha: float,
         channels: list[str], seed: int) -> bag_data.Bag:
    rng = np.random.default_rng(seed)
    t = np.arange(512) / 128.0
    n = 12
    x = rng.normal(scale=0.5, size=(n, len(channels), 512))
    x += alpha * np.sin(2 * np.pi * 10 * t)[None, None, :]
    path = root / f"{subject}-{condition}.npz"
    np.savez(path, signal=x.astype(np.float32), channels=np.array(channels))
    return bag_data.Bag(subject=subject, condition=condition, record_id=f"{subject}-{condition}",
                        label=label, sex="M", fold=fold, device="CGX",
                        channels=tuple(channels), n_windows=n, path=path)


def _world(root: Path, alpha_target: float):
    """Level arm L1 vs L6: one condition each, 3 subjects per fold x label x side."""
    channels = list(CGX_CHANNELS)
    bags, folds, labels = [], {}, {}
    holders = {"L1": set(), "L6": set()}
    k = 0
    for fold in range(partition.N_FOLDS):
        for label in ("yes", "no"):
            for condition, alpha in (("L1", 0.0), ("L6", alpha_target)):
                for _ in range(3):
                    k += 1
                    s = f"UB{k:04d}"
                    folds[s], labels[s] = fold, label
                    holders[condition].add(s)
                    bags.append(_bag(root, s, condition, label, fold, alpha, channels, k))
    rotations = partition.rotations(folds)
    bagset = bag_data.BagSet(bags=tuple(bags), partition_sha256="h" * 64, rotations=rotations)
    a = assignment.assign(assignment.Transfer("level", "L1", "L6"), holders, folds, labels,
                          rotations, partition_sha256="h" * 64)
    return bagset, a


def test_source_and_target_are_the_rotation_training_subjects(tmp_path):
    bags, a = _world(tmp_path, 0.0)
    for r in range(partition.N_FOLDS):
        design = separability.design(a, bags, r, "all")
        sources = set(design["groups"][design["y"] == 1])
        targets = set(design["groups"][design["y"] == 0])
        view = a.rotation(r)
        # Plan v50: a rotation's training subjects are its three non-test folds.
        assert sources == view["source_nontest"]
        assert targets == view["target_nontest"]
        assert {a.folds[s] for s in sources | targets} == set(view["non_test_folds"])
        held_out = view["source_test"] | view["target_test"]
        assert not (sources | targets) & held_out
        assert design["X"].shape == (len(sources) + len(targets),
                                     len(CGX_CHANNELS) * len(audit_domains.BANDS))


def test_both_directions_of_a_pair_use_the_same_subjects_swapped(tmp_path):
    bags, a = _world(tmp_path, 0.0)
    holders = {c: bags.holders(c) for c in ("L1", "L6")}
    b = assignment.assign(assignment.Transfer("level", "L6", "L1"), holders, a.folds,
                          a.labels, a.rotations, partition_sha256="h" * 64)
    for r in range(partition.N_FOLDS):
        forward = separability.design(a, bags, r, "all")
        backward = separability.design(b, bags, r, "all")
        assert set(forward["groups"][forward["y"] == 1]) == \
            set(backward["groups"][backward["y"] == 0])
        assert set(forward["groups"][forward["y"] == 0]) == \
            set(backward["groups"][backward["y"] == 1])


def test_within_class_readings_hold_one_label_each(tmp_path):
    bags, a = _world(tmp_path, 0.0)
    for reading in ("yes", "no"):
        design = separability.design(a, bags, 0, reading)
        assert {a.labels[s] for s in design["groups"]} == {reading}


def test_separable_domains_read_high_and_identical_ones_near_chance(tmp_path):
    (tmp_path / "far").mkdir()
    (tmp_path / "same").mkdir()
    far, a = _world(tmp_path / "far", 3.0)
    same, b = _world(tmp_path / "same", 0.0)
    hi = separability.measure(a, far, 0, "all", n_permutations=20)
    lo = separability.measure(b, same, 0, "all", n_permutations=20)
    assert hi["auc_mean"] > 0.95 and hi["perm_p"] < 0.1
    assert lo["auc_mean"] < 0.8
    assert hi["n_source"] == hi["n_target"] == 3 * 3 * 2   # three non-test folds x two labels


def test_bag_features_match_the_audit_window_features(tmp_path):
    bags, _ = _world(tmp_path, 1.0)
    bag = bags.bags[0]
    w = bag.windows()
    x = w.transpose(0, 2, 1).reshape(-1, w.shape[1])
    expected = audit_domains.record_features(
        audit_domains.window_features(x, np.ones(w.shape[1], bool), transient_windows=0))
    got = separability.bag_features(bag, list(CGX_CHANNELS))
    np.testing.assert_allclose(got, expected.ravel())


def test_results_land_in_the_dashboard_layout_and_record_the_partition(tmp_path):
    rows = pd.DataFrame([{"arm": "task", "transfer": "Robots->CogniFit", "source": "Robots",
                          "target": "CogniFit", "rotation": r, "reading": "all",
                          "auc_mean": 0.6, "auc_sd": 0.1, "perm_p": 0.01,
                          "n_source": 30, "n_target": 18} for r in range(5)]
                        + [{"arm": "task", "transfer": "CogniFit->Robots", "source": "CogniFit",
                            "target": "Robots", "rotation": r, "reading": "all",
                            "auc_mean": 0.7, "auc_sd": 0.1, "perm_p": 0.01,
                            "n_source": 30, "n_target": 18} for r in range(5)])
    out = tmp_path / "separability"
    paths = separability.write_results(rows, out, partition_sha256="h" * 64, extra={})
    assert paths["table"] == out / "tables" / "separability.csv"
    assert paths["summary"] == out / "summary.json"
    table = pd.read_csv(paths["table"])
    assert set(table["role"]) == {"core"}
    assert table.loc[table.transfer == "Robots->CogniFit", "auc_rotation_mean"].iloc[0] == \
        pytest.approx(0.6)
    summary = json.loads(paths["summary"].read_text())
    assert summary["partition_sha256"] == "h" * 64
    assert summary["revision"] == separability.__provenance__["revision"]
    assert summary["core"]["task"] == ["CogniFit->Robots", "Robots->CogniFit"]


def test_a_result_with_another_partition_hash_is_refused_on_load(tmp_path):
    survivors = pd.DataFrame([{"subject": f"UB{i:04d}", "label": "yes" if i % 2 else "no",
                               "sex": "M"} for i in range(1, 21)])
    part = tmp_path / "folds.json"
    partition.freeze_partition(survivors, part)
    out = tmp_path / "separability"
    rows = pd.DataFrame([{"arm": "device", "transfer": t, "source": s, "target": g,
                          "rotation": 0, "reading": "all", "auc_mean": 0.9, "auc_sd": 0.0,
                          "perm_p": 0.01, "n_source": 1, "n_target": 1}
                         for t, s, g in (("L6->Robots", "L6", "Robots"),
                                         ("Robots->L6", "Robots", "L6"))])
    separability.write_results(rows, out, partition_sha256="0" * 64, extra={})
    with pytest.raises(partition.PartitionMismatch):
        separability.load_results(out, part)
    separability.write_results(rows, out, partition_sha256=partition.current_partition_hash(part),
                               extra={})
    assert separability.load_results(out, part)["summary"]["core"]["device"]


def test_separability_is_measured_over_the_four_rotations(tmp_path):
    bags, _ = _world(tmp_path, 0.5)
    transfers = [assignment.Transfer("level", "L1", "L6"), assignment.Transfer("level", "L6", "L1")]
    rows, seconds = separability.measure_transfers(bags, transfers, n_permutations=5, workers=1)
    assert len(bags.rotations) == partition.N_FOLDS == 4
    for _, g in rows.groupby(["transfer", "reading"]):
        assert sorted(g["rotation"]) == [0, 1, 2, 3]
    assert len(rows) == 2 * 4 * len(separability.READINGS)
    for r in range(4):
        assert len(bags.rotations[str(r)]["non_test_folds"]) == 3
    assert set(seconds) == {"level:L1->L6", "level:L6->L1"}


def test_the_eligibility_report_adds_the_pooled_target_totals(tmp_path):
    bags, _ = _world(tmp_path, 0.0)
    transfers = [assignment.Transfer("level", "L1", "L6"), assignment.Transfer("level", "L6", "L1")]
    assignments = [assignment.assign_from_bags(t, bags) for t in transfers]
    paths = separability.write_assignment_tables(assignments, tmp_path / "out")
    table = pd.read_csv(paths["eligibility"])
    assert set(table["scope"]) == {"fold", "pooled_transfer", "pooled_pair"}
    per_fold = table[table.scope == "fold"]
    assert len(per_fold) == 2 * 2 * partition.N_FOLDS * 2
    pooled = table[table.scope == "pooled_transfer"].set_index(["transfer", "label"])["n"]
    assert pooled[("L1->L6", "yes")] == 3 * partition.N_FOLDS
    pair = table[table.scope == "pooled_pair"].set_index("label")["n"]
    assert set(table.loc[table.scope == "pooled_pair", "transfer"]) == {"L1|L6"}
    assert pair["yes"] == pair["no"] == 6 * partition.N_FOLDS
    summary = separability.eligibility_summary(assignments)
    assert summary["transfers"]["level:L1->L6"]["pooled_target"] == {"no": 12, "yes": 12}
    assert summary["transfers"]["level:L1->L6"]["min_per_fold"]["target"] == {"no": 3, "yes": 3}
    assert summary["pairs"]["level:L1|L6"]["pooled_target"] == {"no": 24, "yes": 24}
    assert summary["pairs"]["level:L1|L6"]["by_direction"]["L6->L1"] == {"no": 12, "yes": 12}


def test_the_per_fold_minimum_is_a_declared_limitation_with_no_verdict(tmp_path):
    bags, _ = _world(tmp_path, 0.0)
    transfers = [assignment.Transfer("level", "L1", "L6"), assignment.Transfer("level", "L6", "L1")]
    assignments = [assignment.assign_from_bags(t, bags) for t in transfers]
    summary = separability.eligibility_summary(assignments)
    entry = summary["transfers"]["level:L1->L6"]
    assert set(entry) == {"transfer", "arm", "min_per_fold", "pooled_target"}
    assert "passes" not in entry and "failing" not in entry and "minimum" not in entry
    assert entry["min_per_fold"]["source"] == {"no": 3, "yes": 3}
    assert summary["limitation"] == (
        "El mínimo por fold queda como limitación declarada de la selección.")
    assert separability.__provenance__["revision"].endswith("v55.md")
