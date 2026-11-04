"""Domain assignment per transfer, the leakage guard, and the paired probe.

Experiments plan v40, Secs. 4.3, 4.6 and 6.2. Inside each transfer the
assignment is a total, single-valued function that takes every subject of the
partition to exactly one of three destinations: source, target or excluded.
It is fixed per transfer and identical in the four rotations: a source
subject is source in training, validation and test alike, so the target set
of a transfer does not change between rotations.

**The rule (Sec. 4.3), the same in every arm.** A subject holding one of the
two conditions goes to that side; a subject holding neither is excluded.
Subjects holding both (the commons) are assigned so that, within each fold
and for each diagnosis separately, source and target are as equal in size as
possible, with the partition seed: inside each (fold, label) cell the commons
are ordered by `sha256("<seed>:<pair>:<subject>")` and the first `x` go to
the first condition of the pair, `x` chosen to make the cell's two sides as
equal as possible. When the cell's total is odd the extra subject goes to the
side that is behind over the cells of the same label already laid (in fold
order), and on an exact tie to the side the cell's own hash parity names, so
each class's totals stay within one of each other too. The equalization is
per class, not by the side's total size: in the device arm, where the
high-density-only subjects are almost all diagnosed, it sends the diagnosed
commons to the low-cost side and splits the control commons between both
sides. Plan v38's special case (every device-arm common to the low-cost side)
is gone: it left one control per fold on the high-density side.

**One function per unordered pair.** The rule is keyed by the pair of
conditions, not by the direction, so A->B and B->A read the same condition of
every subject and differ only in which side is labelled.

**The pair is pooled (Sec. 4.6).** The primary contrast of a core pair pools
its two directions, and every assigned subject is predicted by the model of
the direction where it is target. `pair_pool` returns that direction for every
assigned subject and asserts that each one is target exactly once across the
two directions; `pair_totals` reports the pooled target sizes per class.

**The guard is an assertion.** `check_disjoint` raises when source and target
share a subject, for the transfer and for every rotation view. A subject's
other condition is never returned by `side_bags`; the only way to it is
`probe_bag`, and only for the test-fold subjects holding both conditions of
the transfer in that rotation (Sec. 6.2): each common subject passes through
the probe exactly once over the four rotations.

**Inner splits (plan v50, Secs. 4.2 and 4.5).** In rotation `r`, with test fold k,
F is the other three frozen folds; `inner_splits(k)` gives the three inner splits of
the nested cross-validation, each validating one fold of F and training on the other
two, so every non-test subject is validated exactly once and the test fold is in none.
They reuse the frozen folds; no new partition is drawn. `side_bags` reads them through
the roles `inner_train` and `inner_validation` (with `split`), and the refit through
`nontest`, every fold of F. The sides stay those of the transfer (Sec. 4.3).

**The per-fold class minimum (plan v50, Sec. 4.3).** No group is excluded by its
number of subjects per class and fold: the minimum is a declared limitation of
the selection, reported, never a pass or fail. `eligibility` reports the
subjects per (side, fold, label), the minimum per side and class over the folds,
and the pooled target totals per class.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from MIL_CREDA_EEG import partition
from MIL_CREDA_EEG.bag_data import Bag, BagSet

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v50.md",
    "sections": ["4.2", "4.3", "4.5", "4.6", "6.2", "5.1", "5.2", "5.3"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo", "K2-emparejamiento"],
    "invariants": [
        "the_ten_transfers_are_the_declared_ones",
        "the_assignment_is_total_and_single_valued",
        "source_and_target_are_disjoint_for_every_transfer_and_rotation",
        "a_leaking_assignment_fails_its_own_assertion",
        "the_side_is_fixed_across_rotations",
        "single_condition_subjects_go_to_their_side",
        "commons_equalize_source_and_target_sizes",
        "commons_are_stratified_by_diagnosis_within_each_fold",
        "the_assignment_is_seeded_by_the_partition_seed",
        "mirrored_transfers_read_the_same_condition_of_each_subject",
        "commons_are_equalized_per_fold_and_diagnosis_in_every_arm",
        "mirrored_transfers_swap_the_sides_of_one_assignment_in_every_arm",
        "a_pair_pools_every_assigned_subject_as_target_exactly_once",
        "a_pair_that_is_not_one_assignment_swapped_refuses",
        "the_pooled_target_totals_are_reported_per_transfer_and_per_pair",
        "the_probe_reads_only_test_fold_commons",
        "the_other_condition_is_reached_only_through_the_probe_accessor",
        "the_per_fold_class_minimum_is_reported_without_a_verdict",
        "the_inner_splits_never_hold_the_test_fold_and_validate_each_non_test_fold_once",
        "the_inner_roles_validate_every_non_test_subject_exactly_once",
    ],
}

SIDES = ("source", "target", "excluded")
ROLES = ("train", "validation", "test")
#: Plan v50, Secs. 4.2 and 4.5: the nested cross-validation's roles. The inner ones need
#: an inner split; `nontest` is every fold but the test fold (the refit's training set).
INNER_ROLES = ("inner_train", "inner_validation")
REFIT_ROLE = "nontest"
DEFAULT_SEED = partition.PARTITION_SEED


class AssignmentError(RuntimeError):
    """The assignment is not a valid function, or it leaks."""


class ProbeRefused(AssignmentError):
    """A subject's other condition was asked for outside the probe."""


def inner_splits(test_fold: int, n_folds: int = partition.N_FOLDS) -> list[dict]:
    """Plan v50, Sec. 4.2: the inner splits of the rotation whose test fold is `test_fold`.

    F is every fold but the test fold, in fold order; split `i` validates F[i] and trains
    on the other two. Each fold of F is validated once and the test fold is in no split.
    """
    if not 0 <= int(test_fold) < n_folds:
        raise AssignmentError(f"test fold {test_fold} is not one of the {n_folds} folds")
    non_test = [k for k in range(n_folds) if k != int(test_fold)]
    return [{"inner": i, "validation_fold": v, "train_folds": [k for k in non_test if k != v]}
            for i, v in enumerate(non_test)]


@dataclass(frozen=True)
class Transfer:
    arm: str
    source: str
    target: str

    @property
    def name(self) -> str:
        return f"{self.source}->{self.target}"

    @property
    def pair(self) -> tuple[str, str]:
        return tuple(sorted((self.source, self.target)))


#: The ten transfers: level 6 ordered pairs, task 2, device 2 (Secs. 5.1-5.3).
TRANSFERS: tuple[Transfer, ...] = (
    *(Transfer("level", s, t) for s in ("L1", "L6", "L11") for t in ("L1", "L6", "L11")
      if s != t),
    Transfer("task", "Robots", "CogniFit"), Transfer("task", "CogniFit", "Robots"),
    Transfer("device", "L6", "Robots"), Transfer("device", "Robots", "L6"),
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_disjoint(sets: Mapping[str, set], transfer: Transfer) -> None:
    """Raise when source and target share a subject (Sec. 4.3)."""
    shared = set(sets["source"]) & set(sets["target"])
    if shared:
        raise AssignmentError(f"fuga en {transfer.arm} {transfer.name}: {sorted(shared)} "
                              f"son fuente y objetivo a la vez.")


@dataclass(frozen=True)
class Assignment:
    transfer: Transfer
    sides: Mapping[str, str]
    conditions: Mapping[str, str | None]
    folds: Mapping[str, int]
    labels: Mapping[str, str]
    rotations: Mapping[str, Mapping]
    commons: frozenset
    seed: int
    partition_sha256: str

    def side(self, subject: str) -> str:
        try:
            return self.sides[subject]
        except KeyError:
            raise AssignmentError(f"{subject} no está en la partición.") from None

    def condition_of(self, subject: str) -> str | None:
        """The one condition of `subject` this transfer reads (None if excluded)."""
        self.side(subject)
        return self.conditions[subject]

    def subjects(self, side: str) -> set[str]:
        if side not in SIDES:
            raise AssignmentError(f"Lado desconocido {side!r}.")
        return {s for s, v in self.sides.items() if v == side}

    def rotation(self, r: int) -> dict:
        """Source and target subjects by role in rotation `r`, guard checked."""
        rot = self.rotations[str(r)]
        fold_sets = {"train": set(rot["train_folds"]), "validation": {rot["validation_fold"]},
                     "test": {rot["test_fold"]}}
        view = {"rotation": int(r), "test_fold": rot["test_fold"],
                "validation_fold": rot["validation_fold"], "train_folds": list(rot["train_folds"])}
        for side in ("source", "target"):
            members = self.subjects(side)
            for role, fs in fold_sets.items():
                view[f"{side}_{role}"] = {s for s in members if self.folds[s] in fs}
        check_disjoint({"source": set().union(*(view[f"source_{k}"] for k in ROLES)),
                        "target": set().union(*(view[f"target_{k}"] for k in ROLES))},
                       self.transfer)
        return view

    def probe_subjects(self, r: int) -> set[str]:
        """Sec. 6.2: the test-fold subjects of rotation `r` holding both conditions."""
        test_fold = self.rotations[str(r)]["test_fold"]
        return {s for s in self.commons if self.folds[s] == test_fold}


def assign(transfer: Transfer, holders: Mapping[str, Iterable[str]],
           folds: Mapping[str, int], labels: Mapping[str, str],
           rotations: Mapping[str, Mapping], seed: int = DEFAULT_SEED,
           partition_sha256: str = "") -> Assignment:
    """The assignment of every partition subject for `transfer` (Sec. 4.3)."""
    first, second = transfer.pair
    held = {c: set(holders.get(c, ())) & set(folds) for c in (first, second)}
    commons = held[first] & held[second]
    condition: dict[str, str | None] = {}
    for s in folds:
        has = [c for c in (first, second) if s in held[c]]
        condition[s] = has[0] if len(has) == 1 else None
    pair_key = f"{first}|{second}"
    running: dict[str, int] = {}                        # per label: #first - #second laid
    cells = sorted({(labels[s], folds[s]) for s in folds})
    for label, fold in cells:
        cell = [s for s in folds if folds[s] == fold and labels[s] == label]
        a = sum(1 for s in cell if condition[s] == first)
        b = sum(1 for s in cell if condition[s] == second)
        common = sorted((s for s in cell if s in commons),
                        key=lambda s: (_hash(f"{seed}:{pair_key}:{s}"), s))
        n = len(common)
        twice = b - a + n
        x = twice // 2
        lead = running.get(label, 0)
        if twice % 2:
            behind_first = lead < 0 or (
                lead == 0 and int(_hash(f"{seed}:{pair_key}:{fold}:{label}"), 16) % 2)
            x += 1 if behind_first else 0
        x = min(max(x, 0), n)
        for k, s in enumerate(common):
            condition[s] = first if k < x else second
        running[label] = lead + (a + x) - (b + n - x)
    sides = {s: ("excluded" if c is None else "source" if c == transfer.source else "target")
             for s, c in condition.items()}
    out = Assignment(transfer=transfer, sides=dict(sorted(sides.items())),
                     conditions=condition, folds=dict(folds), labels=dict(labels),
                     rotations=rotations, commons=frozenset(commons), seed=int(seed),
                     partition_sha256=partition_sha256)
    check_disjoint({"source": out.subjects("source"), "target": out.subjects("target")},
                   transfer)
    return out


def assign_from_bags(transfer: Transfer, bags: BagSet,
                     doc: Mapping | None = None, seed: int = DEFAULT_SEED) -> Assignment:
    """`assign` over a bag set; the universe is the partition's subjects."""
    if doc is not None:
        if doc[partition.HASH_FIELD] != bags.partition_sha256:
            raise partition.PartitionMismatch("Las bolsas y la partición dada difieren.")
        folds = dict(doc["folds"])
        labels = {r["subject"]: r["label"] for r in doc["fingerprint"]["survivors"]}
    else:
        folds, labels = bags.folds(), bags.labels()
    holders = {c: bags.holders(c) for c in (transfer.source, transfer.target)}
    return assign(transfer, holders, folds, labels, bags.rotations, seed,
                  bags.partition_sha256)


# ------------------------------------------------------------- the pair (Sec. 4.6)


def pair_pool(a: Assignment, b: Assignment) -> dict[str, str]:
    """Every assigned subject of a pair -> the direction in which it is target.

    `a` and `b` must be the two directions of one pair under one partition.
    Raises when an assigned subject is target in both directions or in
    neither, or when the two disagree on who is excluded.
    """
    if b.transfer != Transfer(a.transfer.arm, a.transfer.target, a.transfer.source):
        raise AssignmentError(f"{a.transfer.name} y {b.transfer.name} no son las dos "
                              f"direcciones de un par.")
    if a.partition_sha256 != b.partition_sha256 or set(a.sides) != set(b.sides):
        raise AssignmentError(f"Las dos direcciones del par {a.transfer.pair} no comparten "
                              f"la partición.")
    pooled, wrong = {}, []
    for s in sorted(a.sides):
        excluded = (a.side(s) == "excluded", b.side(s) == "excluded")
        if all(excluded):
            continue
        targets = [d.transfer.name for d in (a, b) if d.side(s) == "target"]
        if any(excluded) or len(targets) != 1:
            wrong.append(f"{s} ({a.side(s)}/{b.side(s)})")
            continue
        pooled[s] = targets[0]
    if wrong:
        raise AssignmentError(f"En el par {a.transfer.pair}, cada sujeto asignado debe ser "
                              f"objetivo exactamente una vez entre las dos direcciones: "
                              f"{', '.join(wrong)}.")
    return pooled


def _label_counts(subjects: Iterable[str], labels: Mapping[str, str]) -> dict[str, int]:
    names = sorted(set(labels.values()))
    subjects = list(subjects)
    return {label: sum(1 for s in subjects if labels[s] == label) for label in names}


def pair_totals(a: Assignment, b: Assignment) -> dict:
    """The pooled target sizes per class of a pair, overall and by direction."""
    pooled = pair_pool(a, b)
    return {"arm": a.transfer.arm, "pair": "|".join(a.transfer.pair),
            "pooled_target": _label_counts(pooled, a.labels),
            "by_direction": {d.transfer.name: _label_counts(d.subjects("target"), d.labels)
                             for d in (a, b)}}


def pairs(assignments: Iterable[Assignment]) -> list[tuple[Assignment, Assignment]]:
    """The two directions of every pair among `assignments`, in pair order."""
    by = {(x.transfer.arm, x.transfer.source, x.transfer.target): x for x in assignments}
    out = []
    for (arm, source, target), x in sorted(by.items()):
        if (source, target) == tuple(sorted((source, target))) and (arm, target, source) in by:
            out.append((x, by[(arm, target, source)]))
    return out


# ------------------------------------------------------------- accessors


def _inner_folds(a: Assignment, r: int, role: str, split: int | None) -> set[int]:
    """The folds of `role` in rotation `r`: an inner split's, or all of F for the refit."""
    test = int(a.rotations[str(r)]["test_fold"])
    if role == REFIT_ROLE:
        if split is not None:
            raise AssignmentError(f"the {REFIT_ROLE!r} role takes no inner split")
        return {k for k in range(partition.N_FOLDS) if k != test}
    if split is None:
        raise AssignmentError(f"the {role!r} role needs an inner split (0, 1 or 2)")
    splits = inner_splits(test)
    if not 0 <= int(split) < len(splits):
        raise AssignmentError(f"inner split {split} is not one of {len(splits)}")
    chosen = splits[int(split)]
    folds = ({chosen["validation_fold"]} if role == "inner_validation"
             else set(chosen["train_folds"]))
    if test in folds:
        raise AssignmentError(f"rotation {r}: the test fold {test} entered an inner split")
    return folds


def side_bags(a: Assignment, bags: BagSet, side: str, r: int, role: str, *,
              split: int | None = None) -> list[Bag]:
    """The bags of `side` in rotation `r` and `role`, each in its assigned condition.

    `role` is one of `ROLES` (the frozen rotation's own roles), `INNER_ROLES` with an
    inner `split`, or `REFIT_ROLE` (every non-test fold)."""
    if side not in ("source", "target") or role not in (*ROLES, *INNER_ROLES, REFIT_ROLE):
        raise AssignmentError(f"Lado o rol desconocido: {side!r}, {role!r}.")
    view = a.rotation(r)                       # the guard is checked on every read
    if role in ROLES:
        if split is not None:
            raise AssignmentError(f"the {role!r} role takes no inner split")
        subjects = sorted(view[f"{side}_{role}"])
    else:
        folds = _inner_folds(a, r, role, split)
        subjects = sorted(s for s in a.subjects(side) if a.folds[s] in folds)
    return [bags.get(s, a.conditions[s]) for s in subjects]


def probe_bag(a: Assignment, bags: BagSet, subject: str, r: int) -> Bag:
    """Sec. 6.2: the other condition of a probe subject; refused for anyone else."""
    if subject not in a.probe_subjects(r):
        raise ProbeRefused(f"{subject} no es sujeto de la sonda en la rotación {r} de "
                           f"{a.transfer.name}: su otra condición no se lee.")
    first, second = a.transfer.pair
    other = second if a.conditions[subject] == first else first
    return bags.get(subject, other)


# ------------------------------------------------------------- reports


def eligibility(a: Assignment) -> dict:
    """Subjects per (side, fold, label), the per-fold minimum per side and class, and the
    pooled target totals: a declared limitation (Sec. 4.3), no verdict."""
    labels = sorted(set(a.labels.values()))
    rows = []
    for side in ("source", "target"):
        members = a.subjects(side)
        for fold in sorted(set(a.folds.values())):
            for label in labels:
                n = sum(1 for s in members if a.folds[s] == fold and a.labels[s] == label)
                rows.append({"arm": a.transfer.arm, "transfer": a.transfer.name, "side": side,
                             "fold": fold, "label": label, "n": n})
    table = pd.DataFrame(rows)
    min_per_fold = {side: {label: int(g["n"].min()) for label, g in t.groupby("label")}
                    for side, t in table.groupby("side")}
    return {"transfer": a.transfer.name, "arm": a.transfer.arm, "table": table,
            "min_per_fold": min_per_fold,
            "pooled_target": _label_counts(a.subjects("target"), a.labels)}


def size_table(assignments: Iterable[Assignment]) -> pd.DataFrame:
    """Per transfer and rotation: subjects per side and role, and the probe size."""
    rows = []
    for a in assignments:
        for r in sorted(a.rotations, key=int):
            view = a.rotation(int(r))
            row = {"arm": a.transfer.arm, "transfer": a.transfer.name, "rotation": int(r),
                   "source": len(a.subjects("source")), "target": len(a.subjects("target")),
                   "excluded": len(a.subjects("excluded")), "commons": len(a.commons)}
            for side in ("source", "target"):
                for role in ROLES:
                    row[f"{side}_{role}"] = len(view[f"{side}_{role}"])
            row["probe"] = len(a.probe_subjects(int(r)))
            rows.append(row)
    return pd.DataFrame(rows)
