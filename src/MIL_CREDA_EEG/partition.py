"""The four-fold partition over subjects, generated once and frozen.

Experiments plan v40, Sec. 4.2. Preprocessing ends here: the subjects that
survive every one of its steps (at least one record not excluded) are split
into four folds, once, stratified by label and by sex, always over subjects
and never over instances. Rotation `r` takes fold `r` as test, fold
`(r + 1) mod 4` as validation and the other two as training, so each fold
is test exactly once and validation exactly once.

**The order needs no random generator.** Inside each label x sex stratum the
subjects are ordered by `sha256("<seed>:<subject>")`, the strata are laid one
after the other in sorted order, and position `i` goes to fold `i mod n`.
That spreads each stratum evenly (its counts per fold differ by at most one)
and keeps the fold sizes within one of each other, and the result depends on
the subject ids, the seed and this file only -- not on a library's generator.

**The partition is frozen.** The file stores what produced it -- the sorted
survivors with their label and sex, the seed and the code version (the
sha256 of this file) -- and its own sha256 over a canonical serialization
without that field. `freeze_partition` recomputes it on every run: identical
continues without writing; different stops with `PartitionChanged`, naming
what changed. Regenerating needs `regenerate=True` AND a new file name, and
never overwrites a file. Every result that depends on the partition records
`partition_sha256`; `check_result` rejects one whose hash is not the current
partition's.

**Which file is current.** `PARTITION_RELATIVE` (`folds_4.json`) is the one
partition in force, and every reader (`bag_data`, `assignment`,
`separability`, the preprocessing entry point and its notebook) takes it from
here. The five-fold file frozen under plan v18 (`FIVE_FOLD_RELATIVE`,
`folds.json`) stays in the repository untouched as the record of what v18
results used: plan v40 replaced it after a measurement (five folds left test
folds with one target-side control), and it was regenerated under a new name
exactly as Sec. 4.2 prescribes. Under this code it refuses as current twice:
`check_current` names its fold count, and `freeze_partition` names its code
version.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from MIL_CREDA_EEG.audit_domains import SEED

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v40.md",
    "sections": ["4.2", "2.10"],
    "experiments": ["4.2"],
    "invariants": [
        "survivors_are_the_subjects_with_a_kept_record",
        "folds_are_stratified_by_label_and_sex_and_rotate",
        "two_runs_write_the_same_file_byte_for_byte",
        "a_changed_survivor_set_refuses_naming_it",
        "a_changed_seed_refuses_naming_it",
        "regeneration_needs_the_flag_and_a_new_name_and_never_overwrites",
        "a_result_with_another_partition_hash_is_rejected_on_load",
        "a_tampered_file_is_refused",
        "four_folds_rotate_with_one_test_one_validation_and_two_training_folds",
        "the_fold_count_is_a_parameter_recorded_in_the_document",
        "the_current_partition_is_one_declared_four_fold_file",
        "the_five_fold_file_is_kept_and_refused_as_current",
        "the_committed_four_fold_partition_reruns_byte_for_byte",
    ],
}

N_FOLDS = 4
#: The package's declared seed (`audit_domains.SEED`), reused for the partition.
PARTITION_SEED = SEED
STRATA = ("label", "sex")
FORMAT = "balladeer-folds/v1"
HASH_FIELD = "partition_sha256"
#: The partition in force, relative to the repository root: every reader takes it from here.
PARTITION_RELATIVE = Path("MIL-CREDA") / "Partition" / "folds_4.json"
#: The five-fold partition of plan v18, kept untouched and no longer current.
FIVE_FOLD_RELATIVE = Path("MIL-CREDA") / "Partition" / "folds.json"
PLAN = "experiments-plan-experimental-mil-creda-sobre-balladeer-v40.md, Sec. 4.2"


class PartitionError(RuntimeError):
    """The partition file cannot be used or written as asked."""


class PartitionChanged(PartitionError):
    """The recomputed partition differs from the frozen one."""

    def __init__(self, message: str, changes: list[str]):
        super().__init__(message)
        self.changes = changes


class PartitionMismatch(PartitionError):
    """A result was produced under a partition other than the current one."""


# ------------------------------------------------------------- inputs


def survivors(manifest: pd.DataFrame) -> pd.DataFrame:
    """Subjects with at least one record not excluded, with label and sex, sorted."""
    kept = manifest[~manifest["excluded"].astype(bool)]
    table = (kept[["subject", "label", "sex"]].drop_duplicates()
             .sort_values("subject").reset_index(drop=True))
    if table["subject"].duplicated().any():
        raise PartitionError("Un sujeto aparece con dos etiquetas o dos sexos en el manifiesto.")
    return table


def code_version() -> str:
    """The sha256 of this file's source: any change to the rule changes it."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _rows(survivors: pd.DataFrame) -> list[dict]:
    table = survivors[["subject", "label", "sex"]].astype(str).sort_values("subject")
    if table["subject"].duplicated().any():
        raise PartitionError("La lista de sobrevivientes repite un sujeto.")
    return table.to_dict("records")


# ------------------------------------------------------------- the rule


def _order_key(seed: int, subject: str) -> str:
    return hashlib.sha256(f"{seed}:{subject}".encode("utf-8")).hexdigest()


def assign_folds(rows: list[dict], seed: int, n_folds: int = N_FOLDS) -> dict[str, int]:
    """Subject -> fold, stratified by label x sex, sorted by subject id."""
    strata: dict[tuple, list[str]] = {}
    for row in rows:
        strata.setdefault(tuple(row[k] for k in STRATA), []).append(row["subject"])
    sequence = []
    for key in sorted(strata):
        sequence += sorted(strata[key], key=lambda s: (_order_key(seed, s), s))
    folds = {subject: i % n_folds for i, subject in enumerate(sequence)}
    return dict(sorted(folds.items()))


def rotations(folds: Mapping[str, int], n_folds: int = N_FOLDS) -> dict[str, dict]:
    """Rotation r: test fold r, validation fold r+1 mod n, training the rest."""
    out = {}
    for r in range(n_folds):
        test, validation = r, (r + 1) % n_folds
        train = [k for k in range(n_folds) if k not in (test, validation)]
        out[str(r)] = {
            "test_fold": test, "validation_fold": validation, "train_folds": train,
            "test": sorted(s for s, k in folds.items() if k == test),
            "validation": sorted(s for s, k in folds.items() if k == validation),
            "train": sorted(s for s, k in folds.items() if k in train),
        }
    return out


# ------------------------------------------------------------- the document


def _canonical(doc: Mapping) -> bytes:
    body = {k: v for k, v in doc.items() if k != HASH_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def partition_hash(doc: Mapping) -> str:
    """sha256 over the canonical serialization of everything but the hash itself."""
    return hashlib.sha256(_canonical(doc)).hexdigest()


def serialize(doc: Mapping) -> bytes:
    """The exact bytes written to disk."""
    return (json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def build_partition(survivors: pd.DataFrame, seed: int = PARTITION_SEED,
                    n_folds: int = N_FOLDS) -> dict:
    """The full partition document, with its fingerprint and its own hash."""
    rows = _rows(survivors)
    folds = assign_folds(rows, seed, n_folds)
    doc = {
        "format": FORMAT,
        "plan": PLAN,
        "n_folds": int(n_folds),
        "strata": list(STRATA),
        "fingerprint": {"survivors": rows, "seed": int(seed), "code_version": code_version()},
        "folds": folds,
        "rotations": rotations(folds, n_folds),
    }
    doc[HASH_FIELD] = partition_hash(doc)
    return doc


def load_partition(path: str | Path) -> dict:
    """The stored partition, after checking that its own hash still holds."""
    path = Path(path)
    if not path.is_file():
        raise PartitionError(f"No existe la partición {path}.")
    doc = json.loads(path.read_text(encoding="utf-8"))
    stored, recomputed = doc.get(HASH_FIELD), partition_hash(doc)
    if stored != recomputed:
        raise PartitionError(f"El hash guardado en {path} ({stored}) no coincide con su "
                             f"contenido ({recomputed}): el archivo fue editado a mano.")
    return doc


def check_current(doc: Mapping) -> Mapping:
    """Refuse a partition that is not of the current rule's shape (format, fold count)."""
    if doc.get("format") != FORMAT:
        raise PartitionError(f"Formato de partición {doc.get('format')!r}; el vigente es "
                             f"{FORMAT!r}.")
    if doc.get("n_folds") != N_FOLDS:
        raise PartitionError(f"La partición tiene {doc.get('n_folds')} folds y la vigente "
                             f"({PARTITION_RELATIVE}) tiene {N_FOLDS}: no es la partición "
                             f"vigente.")
    return doc


def current_partition_hash(path: str | Path = PARTITION_RELATIVE) -> str:
    """The hash of the partition in force, verified against its content."""
    return load_partition(path)[HASH_FIELD]


def describe_changes(stored: Mapping, fresh: Mapping) -> list[str]:
    """What differs between two fingerprints, in words."""
    old, new = stored.get("fingerprint", {}), fresh["fingerprint"]
    changes = []
    old_rows = {r["subject"]: r for r in old.get("survivors", [])}
    new_rows = {r["subject"]: r for r in new["survivors"]}
    gone, came = sorted(set(old_rows) - set(new_rows)), sorted(set(new_rows) - set(old_rows))
    if gone or came:
        changes.append(f"conjunto de sobrevivientes: salen {gone or 'ninguno'}, "
                       f"entran {came or 'ninguno'}")
    moved = sorted(s for s in set(old_rows) & set(new_rows) if old_rows[s] != new_rows[s])
    if moved:
        detail = ", ".join(f"{s} {old_rows[s]['label']}/{old_rows[s]['sex']} -> "
                           f"{new_rows[s]['label']}/{new_rows[s]['sex']}" for s in moved)
        changes.append(f"etiqueta o sexo: {detail}")
    if old.get("seed") != new["seed"]:
        changes.append(f"semilla de partición: {old.get('seed')} -> {new['seed']}")
    if stored.get("n_folds") != fresh["n_folds"]:
        changes.append(f"número de folds: {stored.get('n_folds')} -> {fresh['n_folds']}")
    if old.get("code_version") != new["code_version"]:
        changes.append(f"versión del código: {old.get('code_version')} -> {new['code_version']}")
    if not changes:
        changes.append("el contenido difiere con la misma huella (formato o regla)")
    return changes


def _write_new(path: Path, doc: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:            # never overwrites
        handle.write(serialize(doc))


def freeze_partition(survivors: pd.DataFrame, path: str | Path, seed: int = PARTITION_SEED,
                     regenerate: bool = False, new_name: str | None = None) -> dict:
    """Create, confirm or regenerate the frozen partition; never overwrites.

    Returns `{"status": "created" | "unchanged" | "regenerated", "path", "doc"}`.
    """
    path = Path(path)
    fresh = build_partition(survivors, seed)
    if regenerate or new_name is not None:
        if not regenerate:
            raise PartitionError("Un nombre nuevo sin la bandera de regeneración no regenera: "
                                 "pasá regenerate=True y new_name.")
        if not new_name:
            raise PartitionError("Regenerar la partición exige un nombre nuevo de archivo "
                                 "(new_name); la anterior se conserva.")
        target = path.parent / Path(new_name).name
        if target.exists() or target == path:
            raise PartitionError(f"{target} ya existe: regenerar nunca sobrescribe una "
                                 f"partición; elegí otro nombre.")
        _write_new(target, fresh)
        return {"status": "regenerated", "path": target, "doc": fresh}
    if not path.exists():
        _write_new(path, fresh)
        return {"status": "created", "path": path, "doc": fresh}
    stored = load_partition(path)
    if serialize(stored) == serialize(fresh):
        return {"status": "unchanged", "path": path, "doc": stored}
    changes = describe_changes(stored, fresh)
    raise PartitionChanged(
        f"La partición recalculada difiere de la congelada en {path}: " + "; ".join(changes)
        + ". Para regenerarla, pasá la bandera de regeneración y un nombre nuevo.", changes)


def check_result(result: Mapping | str, path: str | Path = PARTITION_RELATIVE) -> str:
    """Accept a result only if it records the hash of the partition in force."""
    recorded = result if isinstance(result, str) else result.get(HASH_FIELD)
    current = current_partition_hash(path)
    if not recorded:
        raise PartitionMismatch(f"El resultado no registra {HASH_FIELD}; la partición "
                                f"vigente es {current}.")
    if recorded != current:
        raise PartitionMismatch(f"El resultado se produjo con la partición {recorded} y la "
                                f"vigente es {current}: se rechaza.")
    return current


# ------------------------------------------------------------- summaries


def fold_table(doc: Mapping) -> pd.DataFrame:
    """Subjects per fold by label x sex, with the fold total."""
    rows = pd.DataFrame(doc["fingerprint"]["survivors"])
    rows["fold"] = rows["subject"].map(doc["folds"])
    table = rows.pivot_table(index="fold", columns=list(STRATA), values="subject",
                             aggfunc="count", fill_value=0)
    table.columns = [f"{label}/{sex}" for label, sex in table.columns]
    table["total"] = table.sum(axis=1)
    return table


def rotation_table(doc: Mapping) -> pd.DataFrame:
    """Per rotation: its folds and how many subjects each role holds."""
    return pd.DataFrame([{
        "rotación": int(r), "prueba": v["test_fold"], "validación": v["validation_fold"],
        "entrenamiento": v["train_folds"], "sujetos prueba": len(v["test"]),
        "sujetos validación": len(v["validation"]), "sujetos entrenamiento": len(v["train"]),
    } for r, v in sorted(doc["rotations"].items(), key=lambda kv: int(kv[0]))])
