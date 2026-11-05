"""The internal references: source-only controls and the target-supervised upper bound.

Experiments plan v54, Secs. 4.5 and 7.5, for the experiments A-nivel (Sec. 5.1),
B-tarea (Sec. 5.2) and C-dispositivo (Sec. 5.3).

**The main source-only reference.** The common-group phase-1 checkpoint (the
attention wrapper of Sec. 7.2, `mil_wrapper.CommonMILModel`) applied directly
to the target, with no adaptation: its source trunk, attention and head read
the target bags (`two_phase.evaluate` on the `source` domain). It exists only
in the arms whose input is compatible, level and task.

**One matched source-only control per other phase-1 group** (`MATCHED_GROUPS`):
AdaptMIL with its native aggregator and MIL-CREDA with its
attention, each its own group's phase-1 checkpoint applied to the target the
same way, under the same partition, rotations and seeds. They need no
training: the checkpoints are the stored ones (`model_store`, Sec. 7.3).
AdaptMIL is a placeholder here: any phase-1 checkpoint of that
group is accepted once its architecture is registered in
`model_store.BUILDERS`; until then loading it refuses (`GroupNotRegistered`).

**The device arm.** A 29-channel model cannot read a 14-channel input, so
every source-only control of the device arm is the literal record `NA_RECORD`,
never a number: not an accuracy of zero, not a failure to generalize, and it
enters no statistical or gain comparison (`in_statistics` is False). The record
is decided by the arm before any model or bag is touched; no source-only model
is forced with random target layers or shared channels.

**The target-supervised upper bound.** Supervised on target subjects with their
labels, with the same EEGNet-8,2 and common head, through the same phase-1
trainer (`two_phase.train_source_phase`), and its epoch count chosen by the
nested cross-validation of Sec. 4.5, with no early stopping: at the search seed
it is trained on each of the three inner splits of the rotation's non-test folds
F for the whole `epochs1` cap, recording every epoch's validation; E_ub is the
epoch with the best balanced accuracy of the target validation counts pooled
over the three splits (`search.pooled_series`), the earliest on ties; then it is
refit on the target subjects of all of F for exactly E_ub epochs at the given
seed, with no validation set. It reads only the target side of F: never a
source bag, never a target test bag. It is a reference outside the ranking (`in_ranking` False)
and applies to every arm, the device arm included, because it lives in one
domain.

**Implementation choices the plan does not fix.** Sec. 7.5 does not name the
upper bound's architecture or hyperparameters. The architecture is the common
group's (`mil_wrapper.CommonMILModel`, both trunks built with the target's
channel count; the trunk the trainer fits is the one in the `source_trunk`
slot, which here holds the target-supervised trunk); `lr1` and `wd1` are
inputs with no default, the caller passing those selected for the common group
of the same transfer and rotation; it has no Optuna study of its own. The epoch
cap is an input with no default too (Sec. 7.6). Plan v54 does not state the
upper bound's epoch rule explicitly: the inner-split choice of E_ub above is
this implementation's reading of Sec. 4.5 (no early stopping anywhere, the inner
level chooses every model's epoch count).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from MIL_CREDA_EEG import assignment as assignment_mod
from MIL_CREDA_EEG import bag_data, mil_wrapper, model_store, search, two_phase
from MIL_CREDA_EEG.mil_creda_model import CLASSES

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v54.md",
    "sections": ["4.5", "7.3", "7.5"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": [
        "the_device_arm_controls_return_the_na_record_and_never_a_number",
        "the_main_reference_is_the_common_checkpoint_applied_to_the_target",
        "each_other_phase_one_group_has_a_matched_source_only_control",
        "a_placeholder_group_control_accepts_any_registered_phase_one_checkpoint",
        "the_upper_bound_never_sees_source_labels_nor_target_test_labels",
        "the_upper_bound_is_the_common_architecture_on_the_target_channels",
        "the_upper_bound_epochs_are_chosen_on_the_pooled_inner_splits_and_refit_without_validation",
    ],
}

#: Sec. 7.5, verbatim.
NA_RECORD = "N/A — espacios de entrada heterogéneos"
#: Sec. 7.5: where source-only exists (compatible input), and where it does not.
SAME_CHANNEL_ARMS = ("level", "task")
HETEROGENEOUS_ARMS = ("device",)
MAIN_REFERENCE = mil_wrapper.GROUP
MATCHED_GROUPS = ("AdaptMIL", "MIL-CREDA")
UPPER_BOUND = "target-supervised upper bound"
UPPER_BOUND_MODEL = "upper-bound:target-supervised"
#: The schedule keys the upper bound reads; inputs with no default (Sec. 7.6).
UPPER_BOUND_SCHEDULE = ("epochs1", "bagsPerStep")


class ControlError(RuntimeError):
    """A control was asked for something Sec. 7.5 does not define."""


def _reference(group: str) -> str:
    if group == MAIN_REFERENCE:
        return "main source-only reference"
    if group in MATCHED_GROUPS:
        return "matched source-only control"
    raise ControlError(f"{group!r} is not a phase-1 group; the source-only controls are "
                       f"{[MAIN_REFERENCE, *MATCHED_GROUPS]}")


def not_applicable(transfer: assignment_mod.Transfer, group: str, *, rotation: int,
                   seed: int) -> dict:
    """The device arm's record: a sentence, never a number."""
    return {"transfer": transfer.name, "arm": transfer.arm, "group": group,
            "reference": _reference(group), "rotation": int(rotation), "seed": int(seed),
            "record": NA_RECORD, "applicable": False, "in_statistics": False}


def source_only(transfer: assignment_mod.Transfer, group: str, model,
                target_bags: Sequence[two_phase.BagTensor], *, rotation: int, seed: int,
                device: torch.device | str = "cpu") -> dict:
    """A phase-1 checkpoint of `group` applied directly to the target bags."""
    reference = _reference(group)
    if transfer.arm in HETEROGENEOUS_ARMS:
        return not_applicable(transfer, group, rotation=rotation, seed=seed)
    if transfer.arm not in SAME_CHANNEL_ARMS:
        raise ControlError(f"unknown arm {transfer.arm!r}")
    device = torch.device(device)
    channels = {int(b.windows.shape[1]) for b in target_bags}
    if channels != {model.source_trunk.channels}:
        raise ControlError(f"the {group} checkpoint reads {model.source_trunk.channels} "
                           f"channels and the target bags have {sorted(channels)}")
    G = two_phase.evaluate(model, target_bags, "source", device)
    labels = [b.label for b in target_bags]
    positive = CLASSES.index("yes")
    return {"transfer": transfer.name, "arm": transfer.arm, "group": group,
            "reference": reference, "rotation": int(rotation), "seed": int(seed),
            "applicable": True, "in_statistics": True,
            "balanced_accuracy": two_phase.balanced_accuracy(labels, G.argmax(dim=1).tolist()),
            "predictions": [{"subject": b.subject, "direction": transfer.name,
                             "probability": float(G[k, positive]), "label": int(b.label)}
                            for k, b in enumerate(target_bags)]}


def source_only_from_store(path: str | Path, transfer: assignment_mod.Transfer,
                           target_bags: Sequence[two_phase.BagTensor], *,
                           partition_path: str | Path | None = None,
                           allow_code_mismatch: bool = False,
                           device: torch.device | str = "cpu") -> dict:
    """The control of a stored phase-1 checkpoint, which must belong to `transfer`."""
    meta = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    if meta["transfer"] != transfer.name:
        raise ControlError(f"{path} is a checkpoint of transfer {meta['transfer']}, not "
                           f"{transfer.name}")
    if meta["phase"] != 1:
        raise ControlError(f"{path} is not a phase-1 checkpoint")
    if transfer.arm in HETEROGENEOUS_ARMS:
        return not_applicable(transfer, meta["name"], rotation=meta["rotation"],
                              seed=meta["seed"])
    model, meta = model_store.load_model(path, partition_path=partition_path,
                                         allow_code_mismatch=allow_code_mismatch)
    return source_only(transfer, meta["name"], model.to(torch.device(device)), target_bags,
                       rotation=meta["rotation"], seed=meta["seed"], device=device)


def train_upper_bound(a: assignment_mod.Assignment, bags: bag_data.BagSet, rotation: int,
                      seed: int, hyper: Mapping, schedule: Mapping,
                      device: torch.device | str = "cpu") -> dict:
    """Supervised on the target subjects of F: E_ub chosen on the pooled inner splits at
    the search seed, then refit on all of F for exactly E_ub epochs at `seed`."""
    lr1, wd1 = hyper["lr1"], hyper["wd1"]          # no default: KeyError when absent
    if set(schedule) - {"epochs2"} != set(UPPER_BOUND_SCHEDULE):
        raise ValueError(f"the upper bound's schedule is exactly {list(UPPER_BOUND_SCHEDULE)}"
                         f" (epochs2 tolerated and unread); got {sorted(schedule)}")
    device = torch.device(device)
    nan = float("nan")                 # phase-2 keys: the phase-1 trainer never reads them
    full = {**mil_wrapper.COMMON_PHASE_ONE_FIXED, "lr1": lr1, "wd1": wd1,
            "lr2": nan, "wd2": nan, "tauLocal": nan, "epochs2": 1,
            **{k: schedule[k] for k in UPPER_BOUND_SCHEDULE}}

    def target(role: str, split: int | None = None) -> list[two_phase.BagTensor]:
        return two_phase.bag_tensors(assignment_mod.side_bags(a, bags, "target", rotation,
                                                              role, split=split))

    def fit(train, valid, epochs: int, at_seed: int):
        channels = int(train[0].windows.shape[1])
        torch.manual_seed(at_seed)
        generator = torch.Generator().manual_seed(at_seed)
        model = mil_wrapper.CommonMILModel(channels, channels).to(device)
        out = two_phase.train_source_phase(model, train, valid, {**full, "epochs1": epochs},
                                           generator, device, seed=at_seed, fold=rotation,
                                           transfer=a.transfer.name)
        return model, out

    per_split, validated = [], []
    for i in range(search.N_INNER):
        valid = target("inner_validation", i)
        _, out = fit(target("inner_train", i), valid, full["epochs1"], search.SEARCH_SEED)
        per_split.append(out["validation"])
        validated += [b.subject for b in valid]
    series = search.pooled_series(per_split, label_free=False)["pooledBalancedAccuracy"]
    best = int(np.nanargmax(series))
    train = target(assignment_mod.REFIT_ROLE)
    model, out = fit(train, [], best + 1, seed)
    history = [{**r, "model": UPPER_BOUND_MODEL} for r in out["history"]]
    return {"transfer": a.transfer.name, "arm": a.transfer.arm, "rotation": int(rotation),
            "seed": int(seed), "reference": UPPER_BOUND, "in_ranking": False,
            "partition_sha256": a.partition_sha256, "model": model,
            "hyper": {"lr1": lr1, "wd1": wd1, "epochs1": best + 1,
                      "bagsPerStep": schedule["bagsPerStep"]},
            "subjects": {"train": [b.subject for b in train], "inner_validation": validated},
            "selection": {"rule": "pooled target-validation balanced accuracy over the "
                                  "three inner splits (Sec. 4.5), the earliest on ties",
                          "searchSeed": search.SEARCH_SEED, "epochCap": full["epochs1"],
                          "pooledBalancedAccuracy": series, "epoch": best,
                          "epochs": best + 1},
            "checkpoint": out["checkpoint"], "best_epoch": best,
            "best_valid_balanced_accuracy": float(series[best]),
            "history": history}


def upper_bound_predictions(result: Mapping, target_test: Sequence[two_phase.BagTensor],
                            device: torch.device | str = "cpu") -> list[dict]:
    """The trained upper bound's predictions on the target test fold (evaluation only)."""
    G = two_phase.evaluate(result["model"], target_test, "source", torch.device(device))
    positive = CLASSES.index("yes")
    return [{"subject": b.subject, "direction": result["transfer"],
             "probability": float(G[k, positive]), "label": int(b.label)}
            for k, b in enumerate(target_test)]
