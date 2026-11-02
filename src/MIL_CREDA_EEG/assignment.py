"""Domain assignment per transfer, the leakage guard, and the paired probe.

Experiments plan v38, Secs. 4.3 and 6.2. Inside each transfer the assignment
is a total, single-valued function that takes every subject of the partition
to exactly one of three destinations: source, target or excluded. It is fixed
per transfer and identical in the five rotations: a source subject is source
in training, validation and test alike, so the target set of a transfer does
not change between rotations.

**The rule (Sec. 4.3).** A subject holding one of the two conditions goes to
that side; a subject holding neither is excluded. Subjects holding both (the
commons) are assigned so that source and target are as equal in size as
possible, stratified by diagnosis within each fold, with the partition seed:
inside each (fold, label) cell the commons are ordered by
`sha256("<seed>:<pair>:<subject>")` and the first `x` go to the first
condition of the pair, `x` chosen to make the cell's two sides as equal as
possible. When the cell's total is odd the extra subject goes to the side
that is behind over the cells already laid (cells in (label, fold) order),
and on an exact tie to the side the cell's own hash parity names. In the
device arm the commons go to the low-cost side (EPOC X) in both directions,
as the plan declares: halving them would leave about three subjects of that
side per class in each test fold.

**One function per unordered pair.** The rule is keyed by the pair of
conditions, not by the direction, so A->B and B->A read the same condition of
every subject and differ only in which side is labelled.

**The guard is an assertion.** `check_disjoint` raises when source and target
share a subject, for the transfer and for every rotation view. A subject's
other condition is never returned by `side_bags`; the only way to it is
`probe_bag`, and only for the test-fold subjects holding both conditions of
the transfer in that rotation (Sec. 6.2): each common subject passes through
the probe exactly once over the five rotations.

**Eligibility (Sec. 4.3).** Each side needs at least five subjects per class
in every fold. `eligibility` evaluates it and reports every (side, fold,
label) cell under the minimum; it never excludes anything by itself.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from MIL_CREDA_EEG import partition
from MIL_CREDA_EEG.bag_data import Bag, BagSet

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v38.md",
    "sections": ["4.3", "6.2", "5.1", "5.2", "5.3"],
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
        "device_arm_commons_go_to_the_low_cost_side_in_both_directions",
        "the_probe_reads_only_test_fold_commons",
        "the_other_condition_is_reached_only_through_the_probe_accessor",
        "the_eligibility_rule_is_evaluated_and_reported",
    ],
}

SIDES = ("source", "target", "excluded")
ROLES = ("train", "validation", "test")
DEFAULT_SEED = partition.PARTITION_SEED
#: Sec. 4.3: at least five subjects per class per fold.
MIN_PER_CLASS_PER_FOLD = 5
#: The low-cost device's conditions (EPOC X), where the device arm's commons go.
LOW_COST = frozenset({"Robots", "CogniFit"})


class AssignmentError(RuntimeError):
    """The assignment is not a valid function, or it leaks."""


class ProbeRefused(AssignmentError):
    """A subject's other condition was asked for outside the probe."""


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
    if transfer.arm == "device":
        low = [c for c in (first, second) if c in LOW_COST]
        if len(low) != 1:
            raise AssignmentError(f"El brazo de dispositivo necesita un lado de bajo costo: "
                                  f"{transfer.pair}.")
        for s in commons:
            condition[s] = low[0]
    else:
        pair_key = f"{first}|{second}"
        running = 0                                     # (#first - #second) laid so far
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
            if twice % 2:
                behind_first = running < 0 or (
                    running == 0 and int(_hash(f"{seed}:{pair_key}:{fold}:{label}"), 16) % 2)
                x += 1 if behind_first else 0
            x = min(max(x, 0), n)
            for k, s in enumerate(common):
                condition[s] = first if k < x else second
            running += (a + x) - (b + n - x)
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


# ------------------------------------------------------------- accessors


def side_bags(a: Assignment, bags: BagSet, side: str, r: int, role: str) -> list[Bag]:
    """The bags of `side` in rotation `r` and `role`, each in its assigned condition."""
    if side not in ("source", "target") or role not in ROLES:
        raise AssignmentError(f"Lado o rol desconocido: {side!r}, {role!r}.")
    subjects = sorted(a.rotation(r)[f"{side}_{role}"])
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


def eligibility(a: Assignment, minimum: int = MIN_PER_CLASS_PER_FOLD) -> dict:
    """Subjects per (side, fold, label) and whether every cell reaches `minimum`."""
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
    failing = table[table["n"] < minimum]
    return {"transfer": a.transfer.name, "arm": a.transfer.arm,
            "passes": bool(failing.empty), "min_count": int(table["n"].min()),
            "minimum": int(minimum), "table": table,
            "failing": failing[["side", "fold", "label", "n"]].to_dict("records")}


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
