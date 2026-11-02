"""The internal references: source-only controls and the target-supervised upper bound.

Experiments plan v40, Sec. 7.5, for the experiments A-nivel (Sec. 5.1),
B-tarea (Sec. 5.2) and C-dispositivo (Sec. 5.3).

**The main source-only reference.** The common-group phase-1 checkpoint (the
attention wrapper of Sec. 7.2, `mil_wrapper.CommonMILModel`) applied directly
to the target, with no adaptation: its source trunk, attention and head read
the target bags (`two_phase.evaluate` on the `source` domain). It exists only
in the arms whose input is compatible, level and task.

**One matched source-only control per other phase-1 group** (`MATCHED_GROUPS`):
AdaptMIL with its native aggregator, HASD with its own and MIL-CREDA with its
attention, each its own group's phase-1 checkpoint applied to the target the
same way, under the same partition, rotations and seeds. They need no
training: the checkpoints are the stored ones (`model_store`, Sec. 7.3).
AdaptMIL and HASD are placeholders here: any phase-1 checkpoint of those
groups is accepted once its architecture is registered in
`model_store.BUILDERS`; until then loading it refuses (`GroupNotRegistered`).

**The device arm.** A 29-channel model cannot read a 14-channel input, so
every source-only control of the device arm is the literal record `NA_RECORD`,
never a number: not an accuracy of zero, not a failure to generalize, and it
enters no statistical or gain comparison (`in_statistics` is False). The record
is decided by the arm before any model or bag is touched; no source-only model
is forced with random target layers or shared channels.

**The target-supervised upper bound.** Trained on the target TRAINING subjects
of the rotation with their labels, selected (early stopping) on the target
VALIDATION subjects of the same rotation, with the same EEGNet-8,2 and common
head, through the same phase-1 trainer (`two_phase.train_source_phase`). It
reads only those two roles of the target side: never a source bag, never a
target test bag. It is a reference outside the ranking (`in_ranking` False)
and applies to every arm, the device arm included, because it lives in one
domain.

**Implementation choices the plan does not fix.** Sec. 7.5 does not name the
upper bound's architecture or hyperparameters. The architecture is the common
group's (`mil_wrapper.CommonMILModel`, both trunks built with the target's
channel count; the trunk the trainer fits is the one in the `source_trunk`
slot, which here holds the target-supervised trunk); `lr1` and `wd1` are
inputs with no default, the caller passing those selected for the common group
of the same transfer, and the epoch cap and patience are inputs with no default
too (Sec. 7.6).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from MIL_CREDA_EEG import assignment as assignment_mod
from MIL_CREDA_EEG import bag_data, mil_wrapper, model_store, two_phase
from MIL_CREDA_EEG.mil_creda_model import CLASSES

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v40.md",
    "sections": ["7.3", "7.5"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": [
        "the_device_arm_controls_return_the_na_record_and_never_a_number",
        "the_main_reference_is_the_common_checkpoint_applied_to_the_target",
        "each_other_phase_one_group_has_a_matched_source_only_control",
        "a_placeholder_group_control_accepts_any_registered_phase_one_checkpoint",
        "the_upper_bound_never_sees_source_labels_nor_target_test_labels",
        "the_upper_bound_is_the_common_architecture_on_the_target_channels",
    ],
}

#: Sec. 7.5, verbatim.
NA_RECORD = "N/A — espacios de entrada heterogéneos"
#: Sec. 7.5: where source-only exists (compatible input), and where it does not.
SAME_CHANNEL_ARMS = ("level", "task")
HETEROGENEOUS_ARMS = ("device",)
MAIN_REFERENCE = mil_wrapper.GROUP
MATCHED_GROUPS = ("AdaptMIL", "HASD", "MIL-CREDA")
UPPER_BOUND = "target-supervised upper bound"
UPPER_BOUND_MODEL = "upper-bound:target-supervised"
#: The schedule keys the upper bound reads; inputs with no default (Sec. 7.6).
UPPER_BOUND_SCHEDULE = ("epochs1", "patience1", "bagsPerStep")


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
    """Supervised on the target training subjects, selected on target validation."""
    lr1, wd1 = hyper["lr1"], hyper["wd1"]          # no default: KeyError when absent
    if set(schedule) - {"epochs2"} != set(UPPER_BOUND_SCHEDULE):
        raise ValueError(f"the upper bound's schedule is exactly {list(UPPER_BOUND_SCHEDULE)}"
                         f" (epochs2 tolerated and unread); got {sorted(schedule)}")
    device = torch.device(device)
    train = two_phase.bag_tensors(assignment_mod.side_bags(a, bags, "target", rotation,
                                                           "train"))
    valid = two_phase.bag_tensors(assignment_mod.side_bags(a, bags, "target", rotation,
                                                           "validation"))
    nan = float("nan")                 # phase-2 keys: the phase-1 trainer never reads them
    full = {**mil_wrapper.COMMON_PHASE_ONE_FIXED, "lr1": lr1, "wd1": wd1,
            "lr2": nan, "wd2": nan, "tauLocal": nan, "epochs2": 1,
            **{k: schedule[k] for k in UPPER_BOUND_SCHEDULE}}
    channels = int(train[0].windows.shape[1])
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = mil_wrapper.CommonMILModel(channels, channels).to(device)
    out = two_phase.train_source_phase(model, train, valid, full, generator, device,
                                       seed=seed, fold=rotation, transfer=a.transfer.name)
    history = [{**r, "model": UPPER_BOUND_MODEL} for r in out["history"]]
    return {"transfer": a.transfer.name, "arm": a.transfer.arm, "rotation": int(rotation),
            "seed": int(seed), "reference": UPPER_BOUND, "in_ranking": False,
            "partition_sha256": a.partition_sha256, "model": model,
            "hyper": {"lr1": lr1, "wd1": wd1,
                      **{k: schedule[k] for k in UPPER_BOUND_SCHEDULE}},
            "subjects": {"train": [b.subject for b in train],
                         "validation": [b.subject for b in valid]},
            "checkpoint": out["checkpoint"], "best_epoch": out["best_epoch"],
            "best_valid_balanced_accuracy": out["best_valid_balanced_accuracy"],
            "history": history}


def upper_bound_predictions(result: Mapping, target_test: Sequence[two_phase.BagTensor],
                            device: torch.device | str = "cpu") -> list[dict]:
    """The trained upper bound's predictions on the target test fold (evaluation only)."""
    G = two_phase.evaluate(result["model"], target_test, "source", torch.device(device))
    positive = CLASSES.index("yes")
    return [{"subject": b.subject, "direction": result["transfer"],
             "probability": float(G[k, positive]), "label": int(b.label)}
            for k, b in enumerate(target_test)]
