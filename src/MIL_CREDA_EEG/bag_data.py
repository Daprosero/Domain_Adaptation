"""Bags: one subject in one condition, read from the processed data.

Experiments plan v60, Secs. 3.2, 4.2 and 5.1-5.3. A bag is a pair of subject
and condition, where a condition is one recording context (task, device and
level). Each bag lives in exactly one domain, because each condition has one
device. A bag carries its windows (Sec. 3.1, already cut by `windows`), the
per-window marks that travel with them (Secs. 2.3-2.5, 2.8, 2.1), its label,
its sex and the fold the frozen partition gave its subject.

**Only what preprocessing kept.** The bags are the records of
`MIL-CREDA-EEG/Data/processed/manifest.csv` with `excluded == False`; EPOC+ and
every other exclusion were decided there (Secs. 2.10 and 4.3) and are never
re-decided here.

**The partition is read, never made.** The fold of every subject comes from
`partition.load_partition`, by default from the one partition in force,
`partition.PARTITION_RELATIVE` (the four-fold `folds_4_v50.json`); a partition of
another fold count or format, such as the retained five-fold `folds.json` or the
v40 four-fold `folds_4.json`, is refused by `partition.check_current` before any
bag is built. A caller that recorded a
partition hash has it checked with `partition.check_result`. A kept subject
the partition does not name is refused: it would mean the processed data and
the frozen partition disagree.

**Conditions of the three arms.** Level arm (Sec. 5.1): the three CGX
Slackline levels, each its own condition. Task arm (Sec. 5.2): EPOC X Robots
and EPOC X CogniFit. Device arm (Sec. 5.3): CGX Slackline medium level (L6)
and EPOC X Robots. Windows and marks are loaded from the record's `.npz` only
when asked for, so a bag set over the whole cohort costs no memory until used.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from MIL_CREDA_EEG import partition

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v60.md",
    "sections": ["3.2", "4.2", "5.1", "5.2", "5.3"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo", "K2-emparejamiento"],
    "invariants": [
        "bags_are_built_only_from_non_excluded_records",
        "a_bag_is_one_subject_in_one_condition",
        "two_records_for_one_bag_are_refused",
        "a_bag_carries_its_windows_marks_label_sex_and_fold",
        "a_bag_carries_the_fold_of_the_frozen_partition",
        "a_partition_hash_mismatch_is_refused",
        "a_subject_outside_the_partition_is_refused",
        "arm_conditions_are_the_declared_ones",
        "the_bags_read_the_current_four_fold_partition_by_default",
        "a_five_fold_partition_is_refused_as_not_current",
    ],
}

#: Repository root (src/MIL_CREDA_EEG/bag_data.py -> parents[2]).
REPOSITORY = Path(__file__).resolve().parents[2]
PROCESSED_RELATIVE = Path("MIL-CREDA-EEG") / "Data" / "processed"

#: Condition name -> (manifest `condition`, manifest `device`).
CONDITIONS: dict[str, tuple[str, str]] = {
    "L1": ("Slackline-L1/CGX", "CGX"),
    "L6": ("Slackline-L6/CGX", "CGX"),
    "L11": ("Slackline-L11/CGX", "CGX"),
    "Robots": ("Robots/EMOTIV", "EPOCX"),
    "CogniFit": ("Cognifit/EMOTIV", "EPOCX"),
}
#: The conditions each arm reads (Secs. 5.1-5.3).
ARM_CONDITIONS: dict[str, tuple[str, ...]] = {
    "level": ("L1", "L6", "L11"),
    "task": ("Robots", "CogniFit"),
    "device": ("L6", "Robots"),
}
#: Keys of a record's .npz that are not per-window marks.
NOT_MARKS = frozenset({"signal", "channels", "record_id", "subject", "condition", "device",
                       "family", "rate", "target_rate", "gap_exceeds_window",
                       "contact_meets_rule"})


class BagDataError(RuntimeError):
    """The processed data cannot be read as bags."""


@dataclass(frozen=True)
class Bag:
    """One subject in one condition. Windows and marks load on demand."""

    subject: str
    condition: str
    record_id: str
    label: str
    sex: str
    fold: int
    device: str
    channels: tuple[str, ...]
    n_windows: int
    path: Path

    def windows(self) -> np.ndarray:
        """(windows, channels, 512) float32: the bag's instances."""
        with np.load(self.path, allow_pickle=False) as z:
            return z["signal"]

    def marks(self) -> dict[str, np.ndarray]:
        """The per-window side data stored beside the signal."""
        with np.load(self.path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files if k not in NOT_MARKS}


@dataclass(frozen=True)
class BagSet:
    """Every bag of the cohort, with the partition they were read under."""

    bags: tuple[Bag, ...]
    partition_sha256: str
    rotations: Mapping[str, Mapping]
    _index: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        for bag in self.bags:
            key = (bag.subject, bag.condition)
            if key in self._index:
                raise BagDataError(f"Dos registros para la bolsa {key}.")
            self._index[key] = bag

    def get(self, subject: str, condition: str) -> Bag:
        try:
            return self._index[(subject, condition)]
        except KeyError:
            raise BagDataError(f"No hay bolsa para {subject} en {condition}.") from None

    def holders(self, condition: str) -> set[str]:
        """Subjects with a kept bag in `condition`."""
        return {s for s, c in self._index if c == condition}

    def folds(self) -> dict[str, int]:
        return {b.subject: b.fold for b in self.bags}

    def labels(self) -> dict[str, str]:
        return {b.subject: b.label for b in self.bags}


def _condition_of(condition: str, device: str) -> str | None:
    for name, (cond, dev) in CONDITIONS.items():
        if (cond, dev) == (condition, device):
            return name
    return None


def bags_from_manifest(manifest: pd.DataFrame, processed_root: str | Path,
                       doc: Mapping) -> BagSet:
    """The bags of the kept records of the declared conditions, folds from `doc`."""
    processed_root = Path(processed_root)
    kept = manifest[~manifest["excluded"].astype(bool)]
    folds = doc["folds"]
    survivors = {r["subject"]: r for r in doc["fingerprint"]["survivors"]}
    bags, seen = [], set()
    for row in kept.itertuples(index=False):
        name = _condition_of(row.condition, row.device)
        if name is None:
            continue
        if row.subject not in folds:
            raise BagDataError(f"El sujeto {row.subject} tiene un registro conservado pero no "
                               f"está en la partición congelada.")
        known = survivors[row.subject]
        if (known["label"], known["sex"]) != (str(row.label), str(row.sex)):
            raise BagDataError(f"{row.subject}: etiqueta o sexo del manifiesto difieren de la "
                               f"partición.")
        key = (row.subject, name)
        if key in seen:
            raise BagDataError(f"Dos registros conservados para la bolsa {key} "
                               f"({row.subject}, {name}).")
        seen.add(key)
        path = processed_root / str(row.npz)
        with np.load(path, allow_pickle=False) as z:
            channels = tuple(str(c) for c in z["channels"])
        bags.append(Bag(subject=row.subject, condition=name, record_id=row.record_id,
                        label=str(row.label), sex=str(row.sex), fold=int(folds[row.subject]),
                        device=row.device, channels=channels, n_windows=int(row.n_windows),
                        path=path))
    bags.sort(key=lambda b: (b.condition, b.subject))
    return BagSet(bags=tuple(bags), partition_sha256=doc[partition.HASH_FIELD],
                  rotations=doc["rotations"])


def default_partition_path() -> Path:
    """The partition in force: the one constant every reader takes it from."""
    return REPOSITORY / partition.PARTITION_RELATIVE


def load_bags(processed_root: str | Path | None = None,
              partition_path: str | Path | None = None,
              expected_partition_sha256: str | None = None) -> BagSet:
    """Read the processed manifest and the frozen partition; build the bags.

    `expected_partition_sha256`, when given, is the hash a caller recorded: a
    mismatch with the partition in force raises `partition.PartitionMismatch`.
    """
    processed_root = Path(processed_root or REPOSITORY / PROCESSED_RELATIVE)
    partition_path = Path(partition_path or default_partition_path())
    doc = partition.check_current(partition.load_partition(partition_path))
    if expected_partition_sha256 is not None:
        partition.check_result(expected_partition_sha256, partition_path)
    manifest = pd.read_csv(processed_root / "manifest.csv", low_memory=False)
    return bags_from_manifest(manifest, processed_root, doc)
