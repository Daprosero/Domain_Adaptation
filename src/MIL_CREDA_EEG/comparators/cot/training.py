"""COT Algorithm 1 under plan v57 §§4.5, 7.2–7.4, 7.7 (controlled variant).

Phase 2 runs its `epochs2` whole and keeps the last epoch (no early stopping, no
restore); with target validation bags every epoch's validation predictions are
recorded for the search's pooled selection, and without them (the refit) nothing
is evaluated.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

from MIL_CREDA_EEG import eegnet, model_store, search, two_phase
from MIL_CREDA_EEG.comparators.cot.losses import (
    cluster_loss, cot_objective, dual_active, maximize_dual, ot_dual,
    source_cross_entropy, squared_cost,
)
from MIL_CREDA_EEG.comparators.cot.model import COTModel

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v57.md",
    "sections": ["4.5", "7.2", "7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": [
        "cot_phase_two_freezes_shared_parts_and_has_finite_target_gradients",
        "cot_training_is_label_blind_and_reproducible",
        "cot_selection_and_history_use_shared_metrics",
        "cot_rejects_foreign_phase_one_records_and_bad_parameters",
        "cot_search_registration_is_explicit_and_restorable",
        "cot_saved_state_round_trips_with_auxiliaries_and_digest",
        "cot_synthetic_two_phase_smoke",
        "cot_keeps_the_last_epoch_and_records_validation_per_epoch",
        "cot_rejects_leaking_roles_and_empty_batches",
    ],
}

# Paper §5.3 defaults for alpha; remaining numerical defaults are unspecified.
DEFAULT_PARAMS = {"lr2": .001, "wd2": 0., "alphaCluster": .2, "alphaOT": .2,
                  "dualLR": .1, "dualMultiplier": 1.}
DUAL_BASE_STEPS = 20


def source_digest() -> str:
    """Supplement shared model_store.CODE_FILES, which does not enumerate comparators."""
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode()+b"\0"+path.read_bytes()+b"\0")
    return h.hexdigest()


def _params(params, schedule):
    if set(params) != set(DEFAULT_PARAMS):
        raise ValueError(f"COT parameters must be exactly {list(DEFAULT_PARAMS)}")
    for k, v in params.items():
        if not math.isfinite(float(v)) or v < 0:
            raise ValueError(f"invalid {k}: {v}")
    if params["lr2"] <= 0 or params["dualLR"] <= 0:
        raise ValueError("learning rates must be positive")
    if params["dualMultiplier"] not in search.KERNEL_MULTIPLIER.choices:
        raise ValueError("dualMultiplier must follow the shared multiplier rule")
    schedule = search.check_schedule(schedule)
    # Reuse the shared schedule validation; MIL-CREDA-only fields are never read.
    nan = float("nan")
    two_phase.check_hyper({**{k: nan for k in two_phase.PHASE_ONE_KEYS},
                           "lr2": params["lr2"], "wd2": params["wd2"],
                           "tauLocal": nan, **schedule})
    return {**params, **schedule, "cot_source_sha256": source_digest()}


def _snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _validate_record(data, rotation, seed, record):
    state, meta = record
    expected = {"phase": 1, "name": "common", "transfer": data.transfer.name,
                "rotation": rotation, "seed": search.PHASE_ONE_SEED,
                "partition_sha256": data.partition_sha256, "channels": data.channels()}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("COT needs this transfer/rotation/seed's common phase-one record")
    if any(k in state for k in ("target_centers", "potentials")):
        raise ValueError("phase one cannot contain COT auxiliaries")
    return state


def _check_roles(d):
    # A refit (v56 §4.5) has no validation bags; the training roles are never empty.
    for key in ("source_train", "target_train"):
        if not d[key]:
            raise ValueError(f"empty {key}")
    source = {b.subject for key in ("source_train", "source_valid") for b in d[key]}
    target = {b.subject for key in ("target_train", "target_valid") for b in d[key]}
    if source & target:
        raise ValueError("source/target subjects overlap")
    for side in ("source", "target"):
        if ({b.subject for b in d[f"{side}_train"]} &
                {b.subject for b in d[f"{side}_valid"]}):
            raise ValueError("training/validation subjects overlap")


def train_phase_two(data, rotation, seed, record, params, schedule, device="cpu", *,
                    label_free=False):
    """Existing search trainer hook; target-training labels are never accessed.

    Phase one is the shared common-group trainer/checkpoint, not retrained here.
    Exactly `epochs2` epochs run and the last is kept; validation labels are only
    recorded for the search's pooled selection. An empty validation role is a refit.
    """
    hyper = _params(params, schedule)
    state = _validate_record(data, rotation, seed, record)
    device = torch.device(device)
    d = data.rotations[rotation]
    _check_roles(d)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    channels = data.channels()
    model = COTModel(channels["source"], channels["target"]).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("target_trunk.") and
                         k not in ("target_centers", "potentials") for k in missing):
        raise ValueError(f"common checkpoint does not fit: {missing}, {unexpected}")
    joint = two_phase.bridge(model)
    model.initialize_centers(generator)
    frozen_keys = [k for k in model.state_dict() if
                   not k.startswith("target_trunk.") and k not in ("target_centers", "potentials")]
    frozen = {k: model.state_dict()[k].clone() for k in frozen_keys}
    optimizer = torch.optim.SGD([*model.target_trunk.parameters(), model.target_centers],
                                lr=params["lr2"], weight_decay=params["wd2"], momentum=.9)
    history, validation = [], []
    keeper = two_phase.LabelFreeBest(label_free)
    valid_labels = [b.label for b in d["target_valid"]]
    for epoch in range(schedule["epochs2"]):
        two_phase.set_phase_two_modes(model)
        batches = two_phase.shuffled_batches(len(d["target_train"]),
                                             schedule["bagsPerStep"], generator)
        source_batches = two_phase.interleaved_batches(d["source_train"],
                                                       schedule["bagsPerStep"], generator)
        sums = {k: 0. for k in ("loss", "source_ce", "cluster", "ot",
                                "cluster_share", "ot_share")}
        for i, batch in enumerate(batches):
            source = [d["source_train"][j] for j in source_batches[i % len(source_batches)]]
            with torch.no_grad():
                Zs = model.represent(model.bag_pairs(model.embed(
                    [b.windows.to(device) for b in source], "source")))
                ce = source_cross_entropy(model.head.linear(Zs),
                                          torch.tensor([b.label for b in source], device=device))
            # COT Algorithm 1 line 7: no graph to encoder/centers in dual ascent.
            if dual_active(i, len(batches)):
                maximize_dual(squared_cost(model.source_prototypes, model.target_centers),
                              model.potentials, lr=params["dualLR"],
                              steps=round(DUAL_BASE_STEPS*params["dualMultiplier"]))
            Zt = model.represent(model.bag_pairs(model.embed(
                [d["target_train"][j].windows.to(device) for j in batch], "target")))
            cluster = cluster_loss(Zt, model.target_centers)    # Eq. 8, sum over bags
            ot = ot_dual(squared_cost(model.source_prototypes, model.target_centers),
                         model.potentials.detach())             # Eq. 11, lambda fixed
            loss = cot_objective(ce, cluster, ot, params["alphaCluster"], params["alphaOT"])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite COT objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all()
                   for group in optimizer.param_groups for p in group["params"]):
                raise FloatingPointError("missing/non-finite COT primal gradient")
            optimizer.step()
            eegnet.apply_max_norm(model.target_trunk)
            values = {"loss": float(loss.detach()), "source_ce": float(ce),
                      "cluster": float(cluster.detach()), "ot": float(ot.detach())}
            denom = abs(values["source_ce"])+abs(params["alphaCluster"]*values["cluster"])+abs(
                params["alphaOT"]*values["ot"])
            values.update(cluster_share=abs(params["alphaCluster"]*values["cluster"])/max(denom, 1e-30),
                          ot_share=abs(params["alphaOT"]*values["ot"])/max(denom, 1e-30))
            for k in sums:
                sums[k] += values[k]
        accuracy, free = math.nan, math.nan
        if d["target_valid"]:
            G = two_phase.evaluate(model, d["target_valid"], "target", device)
            accuracy = two_phase.balanced_accuracy(valid_labels, G.argmax(1).tolist())
            free = two_phase.label_free_score(G)
            validation.append(two_phase.validation_record(epoch, d["target_valid"], G))
        row = {"epoch": epoch, "seed": seed, "fold": rotation, "model": "COT:phase2",
               "transfer": data.transfer.name, **{k: v/len(batches) for k, v in sums.items()},
               "val_balanced_accuracy": accuracy, "label_free_score": free}
        history.append(row)
        keeper.offer(epoch, free, model)
        if any(not torch.equal(model.state_dict()[k], v) for k, v in frozen.items()):
            raise two_phase.FrozenPartsMoved("COT phase two changed a common frozen tensor")
    two_phase.set_phase_two_modes(model)
    last = len(history) - 1
    checkpoint = {"epoch": last, "valid_balanced_accuracy": accuracy, "label_free_score": free,
                  "hyper": hyper, "state": _snapshot(model)}
    return {"strategy": "COT", "model": model, "channels": channels, "hyper": hyper,
            "layer_copy": joint["layer_copy"], "result": two_phase.with_label_free({
                "checkpoint": checkpoint, "best_epoch": last, "epochs": len(history),
                "history": history, "validation": validation,
                "selection": {"primary": "pooled target-validation balanced accuracy, "
                                         "chosen by the search",
                              "secondary": "pooled entropy of mean minus mean entropy, "
                                           "chosen by the search"}}, keeper)}



def register() -> None:
    """Opt-in search wiring; importing this package never changes global registries."""
    search.register_phase_two("COT", {
        "alphaCluster": search.LOSS_WEIGHT, "alphaOT": search.LOSS_WEIGHT,
        "dualLR": search.LOSS_WEIGHT, "dualMultiplier": search.KERNEL_MULTIPLIER,
    }, train_phase_two)


def load_phase_two(path, *, partition_path=None, allow_code_mismatch=False):
    """Shared store verifies partition/state/code, then verify own source identity."""
    state, meta = model_store.load(path, partition_path=partition_path,
                                   allow_code_mismatch=allow_code_mismatch)
    if meta["phase"] != 2 or meta["name"] != "COT":
        raise ValueError("expected a COT phase-two checkpoint")
    if meta["hyper"].get("cot_source_sha256") != source_digest():
        raise ValueError("COT source digest differs from checkpoint")
    model = COTModel(meta["channels"]["source"], meta["channels"]["target"])
    model.load_state_dict(state)
    # Do NOT call bridge: that would overwrite the trained target encoder.
    for p in [*model.source_trunk.parameters(), *model.attention_parameters(),
              *model.head.parameters()]:
        p.requires_grad_(False)
    model.eval()
    return model, meta
