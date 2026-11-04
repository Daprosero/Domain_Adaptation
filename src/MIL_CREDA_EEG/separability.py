"""Domain separability re-measured on the assigned subjects, and the core rule.

Experiments plan v50, Sec. 5.1 (and 5.2, 5.3 for their arms; Sec. 4.3 for the
eligibility report). The audit's
separabilities were provisional: same subjects on both sides, no reading
within class, and both Emotiv models together. They are measured again with
the audit's own instrument (`audit_domains`: the same log band-power features,
the same subject-grouped logistic classifier, the same 200 permutations inside
the grouped design and the same seed), so that only the composition of
subjects changes.

**What each measurement compares.** Per transfer and rotation, the source
subjects in that rotation's two training folds against the target subjects
in the same training folds, each in the one condition the assignment reads
for it (Sec. 4.3). That is exactly what phase-2 adaptation sees: validation
and test subjects of either side never enter. Because the assignment is one
function per pair of conditions, the two directions of a pair use the same
subject sets with the domain labels swapped, which is why the rule below
averages them. Two readings: all subjects, and
within each class (`yes`, `no`). Only EPOC X enters on the low-cost side (the
bags never hold EPOC+). The device arm is measured on the twelve channels both
montages name identically (`audit_domains.common_channels`), for measuring
only and never for the model (Sec. 3.3); the level arm uses the full CGX
montage and the task arm the full Emotiv one.

**Features from the bags.** A bag's windows are the processed signal (Sec.
2.7: band-pass 0.2-45 Hz on the continuous record, then 128 Hz, first and
partial last windows dropped), which is what the audit's instrument computed
from the raw files. `bag_features` lays the windows end to end and calls
`audit_domains.window_features` with no transient to drop (the bag already
dropped it), then `record_features` (median over windows). A channel with no
variance over the whole record carries no feature, as in the audit.

**The core/control rule (Sec. 5.1), declared before measuring.** The
separability of a pair of conditions is the mean of its two directions, each
direction being the mean over the four rotations of the reading over all
subjects. In each arm the pair with the highest separability is core, with
both its transfers, and the other pairs are control, however small the
difference. The task and device arms hold one pair each, so both their
transfers are core. An exact tie at the top refuses instead of choosing.

**Results** land in `MIL-CREDA/Results/separability/` in the dashboard layout:
`tables/separability.csv` (per transfer and reading: mean and sd over the
rotations, with the pair separability and the role), `tables/
separability_rotations.csv` (every rotation), `tables/assignment_sizes.csv`,
`tables/eligibility.csv` (per side and fold, `scope == "fold"`, plus the
pooled target totals per class of each transfer, `pooled_transfer`, and of
each pair with both directions, `pooled_pair`, Sec. 4.6) and `summary.json`,
which records the same eligibility and the partition hash `load_results`
checks with `partition.check_result`. The rotations are the partition's: four
under the current four-fold partition (`partition.PARTITION_RELATIVE`).

**Eligibility is a declared limitation, not a verdict (Sec. 4.3).** No group of
subjects of the transfer arms is excluded by its number of subjects per class
and fold. The eligibility report gives, per transfer, side and class, the
minimum over the folds (`min_per_fold`) and the pooled target totals, with
`ELIGIBILITY_LIMITATION`; it carries no pass/fail field and no threshold
(`eligibility_summary` drops the ones `assignment.eligibility` computes).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from MIL_CREDA_EEG import assignment, audit_domains, partition
from MIL_CREDA_EEG.assignment import Assignment, Transfer
from MIL_CREDA_EEG.bag_data import REPOSITORY, Bag, BagSet
from MIL_CREDA_EEG.modality import CGX_CHANNELS, EMOTIV_CHANNELS

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v50.md",
    "sections": ["5.1", "5.2", "5.3", "4.3", "4.6"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": [
        "the_instrument_is_the_audits",
        "device_arm_is_measured_on_the_common_channels",
        "bag_features_match_the_audit_window_features",
        "source_and_target_are_the_rotation_training_subjects",
        "both_directions_of_a_pair_use_the_same_subjects_swapped",
        "within_class_readings_hold_one_label_each",
        "separable_domains_read_high_and_identical_ones_near_chance",
        "core_is_the_pair_with_the_highest_mean_of_both_directions",
        "single_pair_arms_are_core",
        "a_tie_for_core_refuses",
        "a_missing_direction_refuses",
        "results_land_in_the_dashboard_layout_and_record_the_partition",
        "a_result_with_another_partition_hash_is_refused_on_load",
        "separability_is_measured_over_the_four_rotations",
        "the_eligibility_report_adds_the_pooled_target_totals",
        "the_per_fold_minimum_is_a_declared_limitation_with_no_verdict",
    ],
}

N_PERMUTATIONS = audit_domains.N_PERMUTATIONS
SEED = audit_domains.SEED
READINGS = ("all", "yes", "no")
#: The channels each arm is measured on.
CHANNELS: dict[str, list[str]] = {
    "level": list(CGX_CHANNELS),
    "task": list(EMOTIV_CHANNELS),
    "device": audit_domains.common_channels(),
}
RESULTS_RELATIVE = Path("MIL-CREDA") / "Results" / "separability"
#: Sec. 4.3: the per-fold minimum is reported as a limitation of the selection, verbatim.
ELIGIBILITY_LIMITATION = "El mínimo por fold queda como limitación declarada de la selección."
#: What `assignment.eligibility` computes against the retired threshold; never reported.
_VERDICT_FIELDS = ("passes", "failing", "minimum", "min_count", "table")


class SeparabilityError(RuntimeError):
    """The separability table cannot be read as the rule needs."""


# ------------------------------------------------------------- features


@lru_cache(maxsize=None)
def _features(path: str, bag_channels: tuple[str, ...], wanted: tuple[str, ...]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        w = np.asarray(z["signal"], dtype=float)
    x = w.transpose(0, 2, 1).reshape(-1, w.shape[1])
    finite = np.isfinite(x)
    usable = finite.any(axis=0) & (np.nanstd(np.where(finite, x, np.nan), axis=0) > 0)
    windows = audit_domains.window_features(x, usable, transient_windows=0)
    windows = audit_domains.features_on(windows, list(bag_channels), list(wanted))
    return audit_domains.record_features(windows).ravel()


def bag_features(bag: Bag, channels: list[str]) -> np.ndarray:
    """Record-level log band power of a bag on `channels` (NaN where absent)."""
    return _features(str(bag.path), tuple(bag.channels), tuple(channels))


# ------------------------------------------------------------- one measurement


def design(a: Assignment, bags: BagSet, r: int, reading: str,
           channels: list[str] | None = None) -> dict:
    """X, y (1 = source), subject groups for one transfer, rotation and reading."""
    if reading not in READINGS:
        raise SeparabilityError(f"Lectura desconocida {reading!r}.")
    channels = channels or CHANNELS[a.transfer.arm]
    keep = (lambda b: True) if reading == "all" else (lambda b: b.label == reading)  # noqa: E731
    source = [b for b in assignment.side_bags(a, bags, "source", r, "train") if keep(b)]
    target = [b for b in assignment.side_bags(a, bags, "target", r, "train") if keep(b)]
    chosen = source + target
    X = np.vstack([bag_features(b, channels)[None, :] for b in chosen])
    y = np.array([1] * len(source) + [0] * len(target))
    groups = np.array([b.subject for b in chosen])
    return {"X": X, "y": y, "groups": groups}


def measure(a: Assignment, bags: BagSet, r: int, reading: str,
            n_permutations: int = N_PERMUTATIONS, seed: int = SEED) -> dict:
    """The audit's separability for one transfer, rotation and reading."""
    d = design(a, bags, r, reading)
    result = audit_domains.separability(d["X"], d["y"], d["groups"], n_permutations, seed)
    return {"arm": a.transfer.arm, "transfer": a.transfer.name, "source": a.transfer.source,
            "target": a.transfer.target, "rotation": int(r), "reading": reading,
            "n_source": int((d["y"] == 1).sum()), "n_target": int((d["y"] == 0).sum()),
            "n_features": int(d["X"].shape[1]), **result,
            "fold_aucs": ",".join(f"{v:.3f}" for v in result["fold_aucs"])}


def _transfer_rows(args: tuple) -> tuple[list[dict], float]:
    transfer, bags, n_permutations, seed = args
    start = time.perf_counter()
    a = assignment.assign_from_bags(transfer, bags)
    rows = [measure(a, bags, int(r), reading, n_permutations, seed)
            for r in sorted(bags.rotations, key=int) for reading in READINGS]
    return rows, time.perf_counter() - start


def measure_transfers(bags: BagSet, transfers: Iterable[Transfer] = assignment.TRANSFERS,
                      n_permutations: int = N_PERMUTATIONS, seed: int = SEED,
                      workers: int = 4) -> tuple[pd.DataFrame, dict[str, float]]:
    """Every rotation and reading of every transfer; seconds per transfer."""
    transfers = list(transfers)
    jobs = [(t, bags, n_permutations, seed) for t in transfers]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            out = list(pool.map(_transfer_rows, jobs, chunksize=1))
    else:
        out = list(map(_transfer_rows, jobs))
    rows = [row for chunk, _ in out for row in chunk]
    seconds = {f"{t.arm}:{t.name}": s for t, (_, s) in zip(transfers, out)}
    return pd.DataFrame(rows), seconds


# ------------------------------------------------------------- the rule


def core_control(table: pd.DataFrame) -> pd.DataFrame:
    """Sec. 5.1: per arm, the pair with the highest mean of both directions is core."""
    out = []
    for arm, rows in table.groupby("arm", sort=True):
        direction = {(r.source, r.target): float(r.auc_mean) for r in rows.itertuples()}
        pairs = sorted({tuple(sorted(k)) for k in direction})
        pair_auc = {}
        for p in pairs:
            a, b = p
            if (a, b) not in direction or (b, a) not in direction:
                raise SeparabilityError(f"Al par {p} del brazo {arm} le falta una dirección.")
            pair_auc[p] = (direction[(a, b)] + direction[(b, a)]) / 2
        top = max(pair_auc.values())
        winners = [p for p, v in pair_auc.items() if v == top]
        if len(winners) > 1:
            raise SeparabilityError(f"Hay un empate en el brazo {arm} entre {winners}: la regla "
                                    f"no elige núcleo.")
        for (s, t), v in sorted(direction.items()):
            p = tuple(sorted((s, t)))
            out.append({"arm": arm, "source": s, "target": t, "transfer": f"{s}->{t}",
                        "pair": f"{p[0]}|{p[1]}", "pair_auc": pair_auc[p],
                        "role": "core" if p == winners[0] else "control"})
    return pd.DataFrame(out)


# ------------------------------------------------------------- results


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """Per transfer and reading: mean and sd over rotations, with the core rule."""
    keys = ["arm", "transfer", "source", "target", "reading"]
    agg = (rows.groupby(keys, sort=True)
           .agg(auc_rotation_mean=("auc_mean", "mean"), auc_rotation_sd=("auc_mean", "std"),
                perm_p_median=("perm_p", "median"), perm_p_max=("perm_p", "max"),
                n_rotations=("rotation", "nunique"), n_source_mean=("n_source", "mean"),
                n_target_mean=("n_target", "mean"))
           .reset_index())
    overall = agg[agg["reading"] == "all"].rename(columns={"auc_rotation_mean": "auc_mean"})
    roles = core_control(overall[["arm", "source", "target", "auc_mean"]])
    return agg.merge(roles[["arm", "transfer", "pair", "pair_auc", "role"]],
                     on=["arm", "transfer"], how="left")


def write_results(rows: pd.DataFrame, out_dir: str | Path, partition_sha256: str,
                  extra: Mapping) -> dict[str, Path]:
    """Write the tables and the summary in the dashboard layout."""
    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    table = aggregate(rows)
    paths = {"table": tables / "separability.csv",
             "rotations": tables / "separability_rotations.csv",
             "summary": out_dir / "summary.json"}
    table.to_csv(paths["table"], index=False)
    rows.sort_values(["arm", "transfer", "reading", "rotation"]).to_csv(paths["rotations"],
                                                                        index=False)
    overall = table[table["reading"] == "all"]
    summary = {
        "revision": __provenance__["revision"],
        partition.HASH_FIELD: partition_sha256,
        "instrument": {"features": "audit_domains log band power", "bands": list(
            audit_domains.BANDS), "classifier": "median imputation + scaling + L2 logistic, C=1",
            "cv": f"StratifiedGroupKFold({audit_domains.N_SPLITS}) by subject",
            "n_permutations": int(rows["n_permutations"].iloc[0])
            if "n_permutations" in rows else N_PERMUTATIONS, "seed": SEED},
        "core": {arm: sorted(g.loc[g.role == "core", "transfer"])
                 for arm, g in overall.groupby("arm")},
        "control": {arm: sorted(g.loc[g.role == "control", "transfer"])
                    for arm, g in overall.groupby("arm")},
        "pairs": {f"{r.arm}:{r.pair}": round(float(r.pair_auc), 4)
                  for r in overall.drop_duplicates(["arm", "pair"]).itertuples()},
        **dict(extra),
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
    return paths


def eligibility_table(assignments: Iterable[Assignment]) -> pd.DataFrame:
    """Per side and fold, then pooled target totals per transfer and per pair."""
    assignments = list(assignments)
    per_fold = pd.concat([assignment.eligibility(a)["table"] for a in assignments])
    per_fold.insert(0, "scope", "fold")
    pooled = [{"scope": "pooled_transfer", "arm": a.transfer.arm, "transfer": a.transfer.name,
               "side": "target", "label": label, "n": n}
              for a in assignments for label, n in assignment.eligibility(a)[
                  "pooled_target"].items()]
    for a, b in assignment.pairs(assignments):
        totals = assignment.pair_totals(a, b)
        pooled += [{"scope": "pooled_pair", "arm": totals["arm"], "transfer": totals["pair"],
                    "side": "target", "label": label, "n": n}
                   for label, n in totals["pooled_target"].items()]
    table = pd.concat([per_fold, pd.DataFrame(pooled)], ignore_index=True)
    table["fold"] = table["fold"].astype("Int64")
    return table


def eligibility_summary(assignments: Iterable[Assignment]) -> dict:
    """Per transfer the class minimums per side over the folds and the pooled target
    totals, and the pooled totals per pair; a declared limitation, no verdict."""
    assignments = list(assignments)
    transfers = {f"{a.transfer.arm}:{a.transfer.name}": {
        k: v for k, v in assignment.eligibility(a).items() if k not in _VERDICT_FIELDS}
        for a in assignments}
    pairs = {}
    for a, b in assignment.pairs(assignments):
        totals = assignment.pair_totals(a, b)
        pairs[f"{totals['arm']}:{totals['pair']}"] = {k: v for k, v in totals.items()
                                                      if k not in ("arm", "pair")}
    return {"transfers": transfers, "pairs": pairs, "limitation": ELIGIBILITY_LIMITATION}


def write_assignment_tables(assignments: Iterable[Assignment],
                            out_dir: str | Path) -> dict[str, Path]:
    """Subjects per side, role and rotation, and the eligibility cells."""
    assignments = list(assignments)
    tables = Path(out_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    paths = {"sizes": tables / "assignment_sizes.csv", "eligibility": tables / "eligibility.csv"}
    assignment.size_table(assignments).to_csv(paths["sizes"], index=False)
    eligibility_table(assignments).to_csv(paths["eligibility"], index=False)
    return paths


def load_results(out_dir: str | Path, partition_path: str | Path) -> dict:
    """The stored results, refused when their partition is not the one in force."""
    out_dir = Path(out_dir)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    partition.check_result(summary, partition_path)
    return {"summary": summary,
            "table": pd.read_csv(out_dir / "tables" / "separability.csv"),
            "rotations": pd.read_csv(out_dir / "tables" / "separability_rotations.csv")}


def run(bags: BagSet, out_dir: str | Path | None = None,
        transfers: Iterable[Transfer] = assignment.TRANSFERS,
        n_permutations: int = N_PERMUTATIONS, workers: int = 4) -> dict:
    """Assign, measure and write everything; returns the paths and timings."""
    out_dir = Path(out_dir or REPOSITORY / RESULTS_RELATIVE)
    transfers = list(transfers)
    assignments = [assignment.assign_from_bags(t, bags) for t in transfers]
    eligible = eligibility_summary(assignments)
    start = time.perf_counter()
    rows, seconds = measure_transfers(bags, transfers, n_permutations, workers=workers)
    wall = time.perf_counter() - start
    paths = write_assignment_tables(assignments, out_dir)
    paths.update(write_results(rows, out_dir, bags.partition_sha256, extra={
        "eligibility": eligible["transfers"],
        "pairs_pooled": eligible["pairs"],
        "eligibility_limitation": eligible["limitation"],
        "timing": {"wall_seconds": round(wall, 1), "workers": int(workers),
                   "seconds_per_transfer": {k: round(v, 1) for k, v in seconds.items()}},
    }))
    return {"paths": paths, "seconds": seconds, "wall_seconds": wall}
