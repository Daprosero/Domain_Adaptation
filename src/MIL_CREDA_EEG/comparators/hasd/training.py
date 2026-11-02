"""HASD two-phase controlled variant; plan §§4.5, 7.3–7.4, 7.7.

Source supervision is the existing shared trainer, not a second implementation.
Phase 2 receives no test bags and never reads a training target label.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import torch

from MIL_CREDA_EEG import assignment, bag_data, model_store, search, two_phase
from .losses import das_loss, sgir_loss, pacr_loss, total_loss
from .model import HASDModel, GROUP
from .prototypes import cluster_assignments, centroids

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v40.md",
    "sections": ["4.5", "7.1", "7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["hasd_bridge_freezes_all_but_target_and_keeps_reference",
                   "hasd_target_training_labels_are_never_read",
                   "hasd_shared_phase_one_restores_checkpoint_and_phase_two_records_selection",
                   "hasd_search_adapter_and_checkpoint_roundtrip",
                   "hasd_runner_uses_shared_assignment_and_heldout_test"],
}

EXTRA_KEYS = ("lambda_sgir", "lambda_pacr", "epsilon", "tau")


@dataclass
class AdaptationContext:
    reference: torch.nn.Module
    frozen_state: dict
    layer_copy: dict


def begin_adaptation(model: HASDModel) -> AdaptationContext:
    """Plan §7.4: reference is the target encoder *after* compatible layer copy."""
    joint = two_phase.bridge(model)
    for p in model.target_trunk.parameters():
        p.requires_grad_(True)
    reference = copy.deepcopy(model.target_trunk).eval().requires_grad_(False)
    return AdaptationContext(reference, joint["frozen_state"], joint["layer_copy"])


def adaptation_step(model: HASDModel, context: AdaptationContext,
                    source: Sequence[two_phase.BagTensor],
                    target: Sequence[two_phase.BagTensor], params: Mapping,
                    device: torch.device) -> dict:
    """Target-to-source bag DAS, within-target SGIR/PACR against fixed phase2 start.

    All geometry is in the shared EEGNet feature space. Prototype membership is
    obtained only from the frozen initial target features, then reused for current
    centroids. No source or target training labels enter any of these terms.
    """
    if not source or not target:
        raise ValueError("HASD adaptation needs nonempty source and target bags")
    model.source_trunk.eval()
    model.attention.eval()
    model.head.eval()
    context.reference.eval()
    with torch.no_grad():
        source_h = model.embed([b.windows.to(device) for b in source], "source")
        source_z = model.represent(model.bag_pairs(source_h))
        before_h = [context.reference(b.windows.to(device)).to(torch.float64) for b in target]
        memberships = [cluster_assignments(h) for h in before_h]
        before = [centroids(h, ids) for h, ids in zip(before_h, memberships)]
        before_a = [model.attention(h) for h in before]
    current_h = model.embed([b.windows.to(device) for b in target], "target")
    current_z = model.represent(model.bag_pairs(current_h))
    after = [centroids(h, ids) for h, ids in zip(current_h, memberships)]
    after_a = [model.attention(h) for h in after]
    das = das_loss(current_z, source_z, epsilon=params["epsilon"], tau=params["tau"])
    sgir = sgir_loss(before, after)
    pacr = pacr_loss(before_a, after_a)
    objective = total_loss(das["loss"], sgir, pacr, params["lambda_sgir"], params["lambda_pacr"])
    return {"objective": objective, "das": das["loss"], "sgir": sgir, "pacr": pacr,
            "transport_mass": das["plan"].sum(), "target_bags": len(target)}


def _snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _check_params(params):
    if set(params) != {"lr2", "wd2", *EXTRA_KEYS}:
        raise ValueError(f"HASD parameters must be lr2, wd2 and {EXTRA_KEYS}")
    for key, value in params.items():
        if not math.isfinite(value) or value < 0 or (key in ("epsilon", "lr2") and value == 0):
            raise ValueError(f"invalid HASD parameter {key}={value}")


def train_phase_two(model: HASDModel, context: AdaptationContext, source_train,
                    target_train, target_valid, params, schedule, *, seed: int,
                    fold: int, transfer: str, device="cpu") -> dict:
    """Whole fixed epoch budget, selecting only with target-validation bag labels."""
    _check_params(params)
    schedule = search.check_schedule(schedule)
    # Reuse the shared strict schedule/key validator; unused phase1 axes are not searched.
    two_phase.check_hyper({**{k: 1. for k in two_phase.PHASE_ONE_KEYS},
                          "tauLocal": 1., "lr2": params["lr2"], "wd2": params["wd2"],
                          **schedule})
    if not target_valid or not target_train or not source_train:
        raise ValueError("nonempty train and validation bags required")
    device = torch.device(device)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.target_trunk.parameters(), lr=params["lr2"],
                                 weight_decay=params["wd2"])
    queues = two_phase.ClassQueues(source_train, model.classes, generator)
    history, best_state, best_epoch, best = [], None, -1, -math.inf
    secondary_state, secondary_best = None, -math.inf
    reference_state = _snapshot(context.reference)
    for epoch in range(schedule["epochs2"]):
        two_phase.set_phase_two_modes(model)
        totals = {k: 0. for k in ("loss", "das", "sgir", "pacr", "transport_mass")}
        batches = two_phase.shuffled_batches(len(target_train), schedule["bagsPerStep"], generator)
        for batch in batches:
            s = [source_train[i] for i in queues.draw(schedule["bagsPerStep"] // model.classes)]
            t = [target_train[i] for i in batch]
            step = adaptation_step(model, context, s, t, params, device)
            if not torch.isfinite(step["objective"]):
                raise RuntimeError("nonfinite HASD objective")
            optimizer.zero_grad()
            step["objective"].backward()
            for p in model.target_trunk.parameters():
                if p.grad is None or not torch.isfinite(p.grad).all():
                    raise RuntimeError("missing or nonfinite HASD target gradient")
            optimizer.step()
            model.constrain("target_trunk")
            totals["loss"] += float(step["objective"].detach())
            for k in ("das", "sgir", "pacr", "transport_mass"):
                totals[k] += float(step[k].detach())
        predictions = two_phase.evaluate(model, target_valid, "target", device)
        accuracy = two_phase.balanced_accuracy([b.label for b in target_valid],
                                               predictions.argmax(1).tolist())
        score = two_phase.label_free_score(predictions)
        history.append({"epoch": epoch, "seed": seed, "fold": fold,
                        "model": "HASD:phase2", "transfer": transfer,
                        **{k: v / len(batches) for k, v in totals.items()},
                        "val_balanced_accuracy": accuracy, "label_free_score": score})
        if accuracy > best:
            best, best_epoch, best_state = accuracy, epoch, _snapshot(model.target_trunk)
        if score > secondary_best:
            secondary_best, secondary_state = score, _snapshot(model.target_trunk)
    model.target_trunk.load_state_dict(best_state)
    model.eval()
    for k, v in two_phase.phase_one_state(model).items():
        if not torch.equal(v, context.frozen_state[k]):
            raise two_phase.FrozenPartsMoved(f"HASD frozen part moved: {k}")
    for k, v in context.reference.state_dict().items():
        if not torch.equal(v, reference_state[k]):
            raise two_phase.FrozenPartsMoved(f"HASD initial target reference moved: {k}")
    return {"checkpoint": {"target_state": best_state, "epoch": best_epoch,
                            "valid_balanced_accuracy": best,
                            "label_free_score": history[best_epoch]["label_free_score"],
                            "hyper": {**params, **schedule}},
            "best_epoch": best_epoch, "epochs": len(history), "history": history,
            "secondary_checkpoint": {"target_state": secondary_state,
                                      "label_free_score": secondary_best}}


def train_search_phase_two(data, rotation, seed, record, params, schedule, device="cpu"):
    """Existing search hook: each trial starts fresh from its own phase1 rotation."""
    state, meta = record
    expected = {"name": GROUP, "phase": 1, "rotation": rotation, "seed": seed,
                "transfer": data.transfer.name, "partition_sha256": data.partition_sha256}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise search.SearchError("HASD needs its own phase1 checkpoint of this transfer/rotation/seed/partition")
    _check_params(params)
    torch.manual_seed(seed)
    channels = data.channels()
    model = HASDModel(channels["source"], channels["target"]).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("target_trunk.") for k in missing):
        raise search.SearchError("HASD phase1 state does not fit the native architecture")
    context = begin_adaptation(model)
    d = data.rotations[rotation]
    result = train_phase_two(model, context, d["source_train"], d["target_train"],
                             d["target_valid"], params, schedule, seed=seed, fold=rotation,
                             transfer=data.transfer.name, device=device)
    return {"strategy": GROUP, "model": model, "result": result,
            "hyper": dict(params), "channels": channels, "layer_copy": context.layer_copy}


def run_transfer_rotation(transfer: assignment.Transfer, rotation: int, seed: int,
                          phase_one_hyper: Mapping, params: Mapping, schedule: Mapping,
                          *, bags: bag_data.BagSet | None = None, device="cpu") -> dict:
    """Wiring only: existing assignment/data/supervision and final test predictions.

    Callers save with shared search.save_phase_one/save_phase_two or model_store;
    no result files or partition writes occur implicitly in this function.
    """
    bags = bags if bags is not None else bag_data.load_bags()
    a = assignment.assign_from_bags(transfer, bags)

    def side(domain, role):
        return two_phase.bag_tensors(assignment.side_bags(a, bags, domain, rotation, role))

    s_train, s_valid = side("source", "train"), side("source", "validation")
    t_train, t_valid = side("target", "train"), side("target", "validation")
    torch.manual_seed(seed)
    device = torch.device(device)
    model = HASDModel(s_train[0].windows.shape[1], t_train[0].windows.shape[1]).to(device)
    phase1 = two_phase.train_source_phase(model, s_train, s_valid, phase_one_hyper,
                 torch.Generator().manual_seed(seed), device, seed=seed, fold=rotation,
                 transfer=transfer.name)
    phase1["history"] = [{**r, "model": "HASD:phase1"} for r in phase1["history"]]
    context = begin_adaptation(model)
    phase2 = train_phase_two(model, context, s_train, t_train, t_valid, params, schedule,
                             seed=seed, fold=rotation, transfer=transfer.name, device=device)
    return {"model": model, "phase1": phase1, "phase2": phase2,
            "partition_sha256": a.partition_sha256, "layer_copy": context.layer_copy,
            "predictions": two_phase.predictions(model, side("target", "test"), transfer.name, device)}
