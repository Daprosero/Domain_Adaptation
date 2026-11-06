"""Controlled UDA-DDA variant: common checkpoint -> target-only MMD + CMMD adaptation.

Plan v56 §7.7: phase 2 runs its `epochs2` whole and keeps the last epoch (no
early stopping, no restore). With target validation bags (an inner split) every
epoch's validation predictions are recorded for the search's pooled selection;
without them (the refit) nothing is evaluated. The paper's MMD-weight and
confidence schedules are rescaled to the epochs this run trains, so a refit of
E2 epochs spans its schedule over E2 epochs.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch

from MIL_CREDA_EEG import eegnet, mil_wrapper, model_store, search, two_phase
from . import losses

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v58.md",
    "sections": ["4.5", "7.1", "7.2", "7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["uda_phase_two_freezes_common_predictor", "uda_target_training_labels_are_unread",
                   "uda_bandwidth_is_the_detached_step_median", "uda_search_hook_uses_common_checkpoint",
                   "uda_keeps_the_last_epoch_and_records_validation_per_epoch",
                   "uda_invalid_inputs_fail_closed",
                   "uda_objective_weights_are_effective", "uda_cmmd_has_no_source_loss_gate",
                   "uda_confidence_thresholds_are_sorted_so_the_schedule_never_inverts"],
}

NAME = "UDA-DDA"
#: mmdWeight is the paper's alpha = 1 (Sec. 4.2); cmmdWeight 1 = Eq. 10's CMMD
#: coefficient with the Eq. 9 gate removed (plan v42 Sec. 7.7); confidenceA/B are
#: two unordered draws whose lower is Eq. 7's middle threshold and higher its late
#: one (here the paper's 0.5 / 0.75); kernelMultiplier 1.0 is unspecified (paper
#: fixes no sigma).
DEFAULTS = {"mmdWeight": 1.0, "cmmdWeight": 1.0, "kernelMultiplier": 1.0,
            "confidenceA": 0.5, "confidenceB": 0.75}
TERMS = ("loss", "source_ce", "mmd", "cmmd", "alpha", "selected", "sigma")


def check_params(params: Mapping) -> dict:
    expected = {"lr2", "wd2", *DEFAULTS}
    if set(params) != expected:
        raise ValueError(f"UDA-DDA parameters must be exactly {sorted(expected)}")
    p = {k: float(v) for k, v in params.items()}
    if not all(math.isfinite(v) for v in p.values()):
        raise ValueError("UDA-DDA parameters must be finite")
    if p["lr2"] <= 0 or p["wd2"] < 0:
        raise ValueError("positive learning rate and nonnegative weight decay required")
    if p["mmdWeight"] < 0 or p["cmmdWeight"] < 0:
        raise ValueError("loss weights must be nonnegative")
    if p["kernelMultiplier"] <= 0:
        raise ValueError("kernelMultiplier must be positive")
    if not (0 <= p["confidenceA"] <= 1 and 0 <= p["confidenceB"] <= 1):
        raise ValueError("confidence thresholds must be in [0,1]")
    return p


def ordered_thresholds(params: Mapping) -> tuple[float, float]:
    """Plan v42 Sec. 7.7: the two draws sorted, lower -> middle stage, higher -> late.

    The schedule 0, middle, late, 1 therefore never inverts.
    """
    low, high = sorted((float(params["confidenceA"]), float(params["confidenceB"])))
    return low, high


def representations(model, bags, domain, device):
    """Plan 7.2: list of (windows, channels, 512) -> (bags, 256) through the common attention."""
    return model.represent(model.bag_pairs(model.embed([b.windows.to(device) for b in bags], domain)))


def adaptation_step(model, source, target, params, alpha, threshold, device):
    """Eq. 10 over BAGS: frozen source CE (diagnostic) + alpha MMD + CMMD, without the
    Eq. 9 source-loss gate on CMMD (plan v42 Sec. 7.7): the source is learned in
    phase 1 and frozen, so CMMD weighs `cmmdWeight` from the start of phase 2.

    sigma is this step's detached median rule over the same train-mode bag
    representations the loss reads (the code's per-batch cadence); a bandwidth
    measured once in eval mode does not match train-mode BatchNorm scales.
    """
    labels = torch.tensor([b.label for b in source], device=device)
    with torch.no_grad():
        zs = representations(model, source, "source", device)
        ce = losses.source_cross_entropy(model.classify(zs), labels)
    zt = representations(model, target, "target", device)
    pseudo, keep = losses.confident_pseudo_labels(model.classify(zt).detach(), threshold)
    sigma = losses.median_bandwidth(zs, zt.detach(), params["kernelMultiplier"])
    marginal = losses.mmd2(zs, zt, sigma)
    conditional = losses.cmmd2(zs, zt[keep], labels, pseudo[keep], model.classes, sigma)
    objective = (ce + params["mmdWeight"] * alpha * marginal
                 + params["cmmdWeight"] * conditional)
    return {"loss": objective, "source_ce": ce, "mmd": marginal, "cmmd": conditional,
            "alpha": torch.tensor(alpha),
            "selected": keep.to(torch.float64).mean(), "sigma": torch.tensor(sigma)}


def adapt(model: mil_wrapper.CommonMILModel, source_train, target_train, target_valid,
          params: Mapping, schedule: Mapping, *, seed: int, fold: int = 0,
          transfer: str = "", device: torch.device | str = "cpu") -> dict:
    """Adapt a selected common phase-1 model; target_train labels are never read.

    Only the target trunk trains (plan 7.3); the paper's epoch schedules are
    rescaled from 100 epochs to `epochs2` (plan 7.4). Exactly `epochs2` epochs run and
    the last is kept; the selection is the search's (plan 4.5). An empty `target_valid`
    is a refit.
    """
    if not isinstance(model, mil_wrapper.CommonMILModel):
        raise ValueError("UDA-DDA requires the shared common MIL phase-1 architecture")
    params = check_params(params)
    schedule = search.check_schedule(schedule)
    # Reuse the shared schedule guard rather than duplicate its integer rules.
    two_phase.check_hyper({**mil_wrapper.COMMON_PHASE_ONE_FIXED, "lr1": 1., "wd1": 0.,
                          "lr2": params["lr2"], "wd2": params["wd2"], "tauLocal": 1.,
                          **schedule})
    if not source_train or not target_train:
        raise ValueError("source training and target training bags required")
    s = {b.subject for b in source_train}
    t = {b.subject for b in target_train}
    v = {b.subject for b in target_valid}
    if s & t or s & v or t & v:
        raise ValueError("training domains and target validation subjects must be disjoint")
    device = torch.device(device)
    model.to(device)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    queues = two_phase.ClassQueues(source_train, model.classes, generator)
    joint = two_phase.bridge(model)
    optimizer = torch.optim.Adam(model.target_trunk.parameters(), lr=params["lr2"],
                                 weight_decay=params["wd2"])
    epochs = schedule["epochs2"]
    middle, late = ordered_thresholds(params)
    history, validation = [], []
    for epoch in range(epochs):
        alpha = losses.mmd_weight_schedule(epoch, epochs)
        threshold = losses.confidence_threshold(epoch, epochs, middle, late)
        two_phase.set_phase_two_modes(model)
        sums = dict.fromkeys(TERMS, 0.)
        batches = two_phase.shuffled_batches(len(target_train), schedule["bagsPerStep"], generator)
        for batch in batches:
            source = [source_train[i] for i in queues.draw(schedule["bagsPerStep"] // model.classes)]
            terms = adaptation_step(model, source, [target_train[i] for i in batch], params,
                                    alpha, threshold, device)
            if not torch.isfinite(terms["loss"]):
                raise FloatingPointError("UDA-DDA objective is not finite")
            optimizer.zero_grad()
            terms["loss"].backward()
            optimizer.step()
            eegnet.apply_max_norm(model.target_trunk)
            for k in sums:
                sums[k] += float(terms[k].detach()) / len(batches)
        accuracy = math.nan
        if target_valid:
            prediction = two_phase.evaluate(model, target_valid, "target", device)
            accuracy = two_phase.balanced_accuracy([b.label for b in target_valid],
                                                   prediction.argmax(1).tolist())
            validation.append(two_phase.validation_record(
                epoch, target_valid, prediction,
                two_phase.represent(model, target_valid, "target", device)))
        history.append({"epoch": epoch, "seed": seed, "fold": fold, "transfer": transfer,
                        "model": f"{NAME}:phase2", **sums, "stage": losses.stage(epoch, epochs),
                        "threshold": threshold,
                        "val_balanced_accuracy": accuracy})
    final_state = copy.deepcopy(model.target_trunk.state_dict())
    if not all(torch.equal(v, two_phase.phase_one_state(model)[k])
               for k, v in joint["frozen_state"].items()):
        raise two_phase.FrozenPartsMoved("UDA-DDA changed a frozen phase-1 part")
    last = len(history) - 1
    return {
            "checkpoint": {"target_state": final_state, "epoch": last,
                           "valid_balanced_accuracy": accuracy,
                           "hyper": {**params, **schedule},
                           "thresholds": {"middle": middle, "late": late}},
            "best_epoch": last, "epochs": len(history), "history": history,
            "validation": validation,
            "final_target_state": final_state, "layer_copy": joint["layer_copy"],
            "selection": two_phase.SELECTION}


def train_phase_two(data, rotation, seed, record, params, schedule, device="cpu"):
    """Shared search trainer hook; reject a checkpoint from a different rotation/group."""
    state, meta = record
    expected = {"phase": 1, "name": "common", "transfer": data.transfer.name,
                "rotation": rotation, "seed": search.PHASE_ONE_SEED,
                "partition_sha256": data.partition_sha256}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise search.SearchError("UDA-DDA needs the common checkpoint of this transfer/rotation/seed/partition")
    torch.manual_seed(seed)
    channels = data.channels()
    model = model_store.build_model("common", channels["source"], channels["target"], meta["hyper"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("target_trunk.") for k in missing):
        raise search.SearchError("incompatible common phase-1 state")
    d = data.rotations[rotation]
    result = adapt(model, d["source_train"], d["target_train"], d["target_valid"], params,
                   schedule, seed=seed, fold=rotation, transfer=data.transfer.name, device=device)

    return {"strategy": NAME, "model": model, "result": result,
            "hyper": result["checkpoint"]["hyper"], "channels": channels,
            "layer_copy": result["layer_copy"]}


def register():
    """Explicit registration: importing this package changes no global search state.

    Plan v42 Sec. 7.7 row: MMD and CMMD weights, kernel scale, and the two
    pseudo-label confidence thresholds drawn from the generic range and sorted
    by the trainer (`ordered_thresholds`); no source-loss gate on CMMD.
    """
    search.register_phase_two(NAME, {"mmdWeight": search.LOSS_WEIGHT,
                                    "cmmdWeight": search.LOSS_WEIGHT,
                                    "kernelMultiplier": search.KERNEL_MULTIPLIER,
                                    "confidenceA": search.CONFIDENCE_THRESHOLD,
                                    "confidenceB": search.CONFIDENCE_THRESHOLD}, train_phase_two)
