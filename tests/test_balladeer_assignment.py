"""Experiments plan v38, Secs. 4.3 and 6.2: domain assignment, leakage guard, probe."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from MIL_CREDA_EEG import assignment, bag_data, partition

SEED = partition.PARTITION_SEED


def _population(n_per_cell_common: int = 7, singles_a: int = 1, singles_b: int = 1,
                unrelated: int = 1) -> dict:
    """Subjects per (fold, label): commons hold A and B, singles one, unrelated none."""
    holders = {"A": set(), "B": set()}
    folds, labels = {}, {}
    k = 0
    for fold in range(partition.N_FOLDS):
        for label in ("yes", "no"):
            for kind, count in (("AB", n_per_cell_common), ("A", singles_a),
                                ("B", singles_b), ("", unrelated)):
                for _ in range(count):
                    k += 1
                    s = f"UB{k:04d}"
                    folds[s], labels[s] = fold, label
                    for c in kind:
                        holders[c].add(s)
    rotations = partition.rotations(folds)
    return {"holders": holders, "folds": folds, "labels": labels, "rotations": rotations}


def _assign(pop: dict, source: str = "A", target: str = "B", arm: str = "level",
            seed: int = SEED) -> assignment.Assignment:
    return assignment.assign(assignment.Transfer(arm, source, target), pop["holders"],
                             pop["folds"], pop["labels"], pop["rotations"], seed=seed,
                             partition_sha256="h" * 64)


def test_the_ten_transfers_are_the_declared_ones():
    names = [(t.arm, t.source, t.target) for t in assignment.TRANSFERS]
    assert len(names) == len(set(names)) == 10
    level = {(s, t) for a, s, t in names if a == "level"}
    assert level == {(s, t) for s in ("L1", "L6", "L11") for t in ("L1", "L6", "L11") if s != t}
    assert {(s, t) for a, s, t in names if a == "task"} == {("Robots", "CogniFit"),
                                                           ("CogniFit", "Robots")}
    assert {(s, t) for a, s, t in names if a == "device"} == {("L6", "Robots"),
                                                             ("Robots", "L6")}


def test_the_assignment_is_total_and_single_valued():
    pop = _population()
    a = _assign(pop)
    for subject in pop["folds"]:
        assert a.side(subject) in ("source", "target", "excluded")
    assert set(a.sides) == set(pop["folds"])
    parts = [a.subjects("source"), a.subjects("target"), a.subjects("excluded")]
    assert sum(len(p) for p in parts) == len(pop["folds"])
    assert set().union(*parts) == set(pop["folds"])
    with pytest.raises(assignment.AssignmentError):
        a.side("UB9999")


def test_source_and_target_are_disjoint_for_every_transfer_and_rotation():
    pop = _population()
    for source, target in (("A", "B"), ("B", "A")):
        a = _assign(pop, source, target)
        assert not a.subjects("source") & a.subjects("target")
        for r in range(partition.N_FOLDS):
            view = a.rotation(r)
            src = set().union(*(view[f"source_{k}"] for k in ("train", "validation", "test")))
            tgt = set().union(*(view[f"target_{k}"] for k in ("train", "validation", "test")))
            assert not src & tgt


def test_a_leaking_assignment_fails_its_own_assertion():
    pop = _population()
    a = _assign(pop)
    subject = next(iter(a.subjects("source")))
    with pytest.raises(assignment.AssignmentError, match="fuga"):
        assignment.check_disjoint({"source": {subject}, "target": {subject}}, a.transfer)


def test_the_side_is_fixed_across_rotations():
    pop = _population()
    a = _assign(pop)
    for subject in pop["folds"]:
        seen = set()
        for r in range(partition.N_FOLDS):
            view = a.rotation(r)
            for side in ("source", "target"):
                if any(subject in view[f"{side}_{k}"] for k in ("train", "validation", "test")):
                    seen.add(side)
        assert len(seen) <= 1
        assert seen == ({a.side(subject)} - {"excluded"})


def test_single_condition_subjects_go_to_their_side():
    pop = _population()
    a = _assign(pop, "B", "A")
    only_a = pop["holders"]["A"] - pop["holders"]["B"]
    only_b = pop["holders"]["B"] - pop["holders"]["A"]
    neither = set(pop["folds"]) - pop["holders"]["A"] - pop["holders"]["B"]
    assert all(a.side(s) == "target" and a.condition_of(s) == "A" for s in only_a)
    assert all(a.side(s) == "source" and a.condition_of(s) == "B" for s in only_b)
    assert all(a.side(s) == "excluded" and a.condition_of(s) is None for s in neither)


def test_commons_equalize_source_and_target_sizes():
    # Unbalanced singles (3 on A, 0 on B per cell): commons fill B first.
    pop = _population(n_per_cell_common=7, singles_a=3, singles_b=0)
    a = _assign(pop)
    for fold in range(partition.N_FOLDS):
        for label in ("yes", "no"):
            cell = [s for s in pop["folds"] if pop["folds"][s] == fold
                    and pop["labels"][s] == label]
            n = Counter(a.side(s) for s in cell)
            assert abs(n["source"] - n["target"]) <= 1
    assert abs(len(a.subjects("source")) - len(a.subjects("target"))) <= 1
    # Balanced singles: commons split in halves.
    b = _assign(_population(n_per_cell_common=8, singles_a=1, singles_b=1))
    assert len(b.subjects("source")) == len(b.subjects("target"))


def test_commons_are_stratified_by_diagnosis_within_each_fold():
    pop = _population(n_per_cell_common=6, singles_a=0, singles_b=0)
    a = _assign(pop)
    for fold in range(partition.N_FOLDS):
        for label in ("yes", "no"):
            commons = [s for s in pop["holders"]["A"] & pop["holders"]["B"]
                       if pop["folds"][s] == fold and pop["labels"][s] == label]
            n = Counter(a.side(s) for s in commons)
            assert n["source"] == n["target"] == 3


def test_the_assignment_is_seeded_by_the_partition_seed():
    pop = _population(n_per_cell_common=8)
    a, b = _assign(pop), _assign(pop)
    assert a.sides == b.sides
    assert assignment.DEFAULT_SEED == partition.PARTITION_SEED
    other = _assign(pop, seed=SEED + 1)
    assert other.sides != a.sides
    counts = lambda x: (len(x.subjects("source")), len(x.subjects("target")))  # noqa: E731
    assert counts(other) == counts(a)


def test_mirrored_transfers_read_the_same_condition_of_each_subject():
    pop = _population(n_per_cell_common=7)
    ab, ba = _assign(pop, "A", "B"), _assign(pop, "B", "A")
    for s in pop["folds"]:
        assert ab.condition_of(s) == ba.condition_of(s)


def _device_population() -> dict:
    pop = _population(n_per_cell_common=5, singles_a=6, singles_b=1)
    holders = {"L6": pop["holders"]["A"], "Robots": pop["holders"]["B"]}
    return {**pop, "holders": holders}


def test_device_arm_commons_go_to_the_low_cost_side_in_both_directions():
    pop = _device_population()
    commons = pop["holders"]["L6"] & pop["holders"]["Robots"]
    for source, target in (("L6", "Robots"), ("Robots", "L6")):
        a = assignment.assign(assignment.Transfer("device", source, target), pop["holders"],
                              pop["folds"], pop["labels"], pop["rotations"],
                              partition_sha256="h" * 64)
        assert all(a.condition_of(s) == "Robots" for s in commons)
        low_cost_side = "target" if target == "Robots" else "source"
        assert all(a.side(s) == low_cost_side for s in commons)


def test_the_probe_reads_only_test_fold_commons():
    pop = _population()
    a = _assign(pop)
    commons = pop["holders"]["A"] & pop["holders"]["B"]
    seen = Counter()
    for r in range(partition.N_FOLDS):
        test_fold = pop["rotations"][str(r)]["test_fold"]
        probe = a.probe_subjects(r)
        assert probe == {s for s in commons if pop["folds"][s] == test_fold}
        seen.update(probe)
    assert set(seen) == commons and set(seen.values()) == {1}


# --------------------------------------------------- the accessor over real bags


def _bags(tmp_path: Path, pop: dict) -> bag_data.BagSet:
    bags = []
    for condition, subjects in pop["holders"].items():
        for s in sorted(subjects):
            bags.append(bag_data.Bag(subject=s, condition=condition, record_id=f"{s}-{condition}",
                                     label=pop["labels"][s], sex="M", fold=pop["folds"][s],
                                     device="CGX", channels=("O1",), n_windows=70,
                                     path=tmp_path / f"{s}-{condition}.npz"))
    return bag_data.BagSet(bags=tuple(bags), partition_sha256="h" * 64,
                           rotations=pop["rotations"])


def test_the_other_condition_is_reached_only_through_the_probe_accessor(tmp_path):
    pop = _population()
    a = _assign(pop)
    bags = _bags(tmp_path, pop)
    for r in range(partition.N_FOLDS):
        view = a.rotation(r)
        for side in ("source", "target"):
            for role in ("train", "validation", "test"):
                for bag in assignment.side_bags(a, bags, side, r, role):
                    assert bag.condition == a.condition_of(bag.subject)
                    assert bag.subject in view[f"{side}_{role}"]
        for s in a.probe_subjects(r):
            other = assignment.probe_bag(a, bags, s, r)
            assert other.subject == s and other.condition != a.condition_of(s)
    single = next(iter(pop["holders"]["A"] - pop["holders"]["B"]))
    with pytest.raises(assignment.ProbeRefused):
        assignment.probe_bag(a, bags, single, 0)
    train_common = next(s for s in pop["holders"]["A"] & pop["holders"]["B"]
                        if pop["folds"][s] in pop["rotations"]["0"]["train_folds"])
    with pytest.raises(assignment.ProbeRefused):
        assignment.probe_bag(a, bags, train_common, 0)


def test_the_eligibility_rule_is_evaluated_and_reported():
    rich = _assign(_population(n_per_cell_common=10, singles_a=0, singles_b=0))
    poor = _assign(_population(n_per_cell_common=6, singles_a=0, singles_b=0))
    ok, short = assignment.eligibility(rich), assignment.eligibility(poor)
    assert assignment.MIN_PER_CLASS_PER_FOLD == 5
    assert ok["passes"] and ok["min_count"] == 5
    assert not short["passes"] and short["min_count"] == 3
    table = short["table"]
    assert set(table.columns) >= {"side", "fold", "label", "n"}
    assert len(table) == 2 * partition.N_FOLDS * 2
    assert short["failing"]  # names every (side, fold, label) under the minimum


def test_the_size_table_reports_every_rotation():
    a = _assign(_population())
    table = assignment.size_table([a])
    assert len(table) == partition.N_FOLDS
    assert {"transfer", "rotation", "source_train", "target_train", "target_test",
            "probe"} <= set(table.columns)
    assert (table["probe"] > 0).all()
    assert np.isfinite(table[["source_train", "target_train"]].to_numpy()).all()
