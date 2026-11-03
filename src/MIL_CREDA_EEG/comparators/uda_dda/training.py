"""Controlled UDA-DDA variant: common checkpoint -> target-only MMD + CMMD adaptation."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch

from MIL_CREDA_EEG import eegnet, mil_wrapper, model_store, search, two_phase
from . import losses

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v40.md",
    "sections": ["4.5", "7.1", "7.2", "7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["uda_phase_two_freezes_common_predictor", "uda_target_training_labels_are_unread",
                   "uda_bandwidth_is_the_detached_step_median", "uda_search_hook_uses_common_checkpoint",
                   "uda_selection_restores_best_epoch", "uda_invalid_inputs_fail_closed",
                   "uda_objective_weights_are_effective"],
}

NAME = "UDA-DDA"
#: mmdWeight is the paper's alpha = 1 (Sec. 4.2); cmmdWeight 1 = Eq. 10 has no
#: coefficient beyond beta; confidence 0.5 / 0.75 are Eq. 7's middle and late
#: thresholds; kernelMultiplier 1.0 is unspecified (paper fixes no sigma).
DEFAULTS = {"mmdWeight": 1.0, "cmmdWeight": 1.0, "kernelMultiplier": 1.0,
            "confidenceMiddle": 0.5, "confidenceLate": 0.75}
TERMS = ("loss", "source_ce", "mmd", "cmmd", "alpha", "beta", "selected", "sigma")


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
    if not (0 <= p["confidenceMiddle"] <= 1 and 0 <= p["confidenceLate"] <= 1):
        raise ValueError("confidence thresholds must be in [0,1]")
    return p


def representations(model, bags, domain, device):
    """Plan 7.2: list of (windows, channels, 512) -> (bags, 256) through the common attention."""
    return model.represent(model.bag_pairs(model.embed([b.windows.to(device) for b in bags], domain)))


def adaptation_step(model, source, target, params, alpha, threshold, device):
    """Eq. 10 over BAGS: frozen source CE (diagnostic, sets beta) + alpha MMD + beta CMMD.

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
    beta = losses.cmmd_weight(float(ce))
    sigma = losses.median_bandwidth(zs, zt.detach(), params["kernelMultiplier"])
    marginal = losses.mmd2(zs, zt, sigma)
    conditional = losses.cmmd2(zs, zt[keep], labels, pseudo[keep], model.classes, sigma)
    objective = (ce + params["mmdWeight"] * alpha * marginal
                 + params["cmmdWeight"] * beta * conditional)
    return {"loss": objective, "source_ce": ce, "mmd": marginal, "cmmd": conditional,
            "alpha": torch.tensor(alpha), "beta": torch.tensor(beta),
            "selected": keep.to(torch.float64).mean(), "sigma": torch.tensor(sigma)}


def adapt(model: mil_wrapper.CommonMILModel, source_train, target_train, target_valid,
          params: Mapping, schedule: Mapping, *, seed: int, fold: int = 0,
          transfer: str = "", device: torch.device | str = "cpu") -> dict:
    """Adapt a selected common phase-1 model; target_train labels are never read.

    Only the target trunk trains (plan 7.3); the paper's epoch schedules are
    rescaled from 100 epochs to `epochs2` (plan 7.4). Selection: plan 4.5.
    """
    if not isinstance(model, mil_wrapper.CommonMILModel):
        raise ValueError("UDA-DDA requires the shared common MIL phase-1 architecture")
    params = check_params(params)
    schedule = search.check_schedule(schedule)
    # Reuse the shared schedule guard rather than duplicate its integer rules.
    two_phase.check_hyper({**mil_wrapper.COMMON_PHASE_ONE_FIXED, "lr1": 1., "wd1": 0.,
                          "lr2": params["lr2"], "wd2": params["wd2"], "tauLocal": 1.,
                          **schedule})
    if not source_train or not target_train or not target_valid:
        raise ValueError("source training, target training and target validation bags required")
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
    history, best_state, best, best_epoch = [], None, -math.inf, -1
    for epoch in range(epochs):
        alpha = losses.mmd_weight_schedule(epoch, epochs)
        threshold = losses.confidence_threshold(epoch, epochs, params["confidenceMiddle"],
                                                params["confidenceLate"])
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
        prediction = two_phase.evaluate(model, target_valid, "target", device)
        accuracy = two_phase.balanced_accuracy([b.label for b in target_valid],
                                               prediction.argmax(1).tolist())
        history.append({"epoch": epoch, "seed": seed, "fold": fold, "transfer": transfer,
                        "model": f"{NAME}:phase2", **sums, "stage": losses.stage(epoch, epochs),
                        "threshold": threshold,
                        "val_balanced_accuracy": accuracy,
                        "label_free_score": two_phase.label_free_score(prediction)})
        if accuracy > best:
            best, best_epoch = accuracy, epoch
            best_state = copy.deepcopy(model.target_trunk.state_dict())
    final_state = copy.deepcopy(model.target_trunk.state_dict())
    model.target_trunk.load_state_dict(best_state)
    if not all(torch.equal(v, two_phase.phase_one_state(model)[k])
               for k, v in joint["frozen_state"].items()):
        raise two_phase.FrozenPartsMoved("UDA-DDA changed a frozen phase-1 part")
    restored = two_phase.evaluate(model, target_valid, "target", device)
    if two_phase.balanced_accuracy([b.label for b in target_valid], restored.argmax(1).tolist()) != best:
        raise two_phase.RestoreFailed("UDA-DDA restored validation score differs")
    return {"checkpoint": {"target_state": best_state, "epoch": best_epoch,
                            "valid_balanced_accuracy": best,
                            "label_free_score": history[best_epoch]["label_free_score"],
                            "hyper": {**params, **schedule}},
            "best_epoch": best_epoch, "epochs": len(history), "history": history,
            "final_target_state": final_state, "layer_copy": joint["layer_copy"],
            "selection": {"primary": "target-validation balanced accuracy",
                          "best_label_free_epoch": max(range(len(history)),
                                                       key=lambda i: history[i]["label_free_score"])}}


def train_phase_two(data, rotation, seed, record, params, schedule, device="cpu"):
    """Shared search trainer hook; reject a checkpoint from a different rotation/group."""
    state, meta = record
    expected = {"phase": 1, "name": "common", "transfer": data.transfer.name,
                "rotation": rotation, "seed": seed, "partition_sha256": data.partition_sha256}
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

    Plan 7.7 row: MMD and CMMD weights, kernel scale, pseudo-label confidence
    thresholds, each on the generic rule (no axis outside it).
    """
    search.register_phase_two(NAME, {"mmdWeight": search.LOSS_WEIGHT,
                                    "cmmdWeight": search.LOSS_WEIGHT,
                                    "kernelMultiplier": search.KERNEL_MULTIPLIER,
                                    "confidenceMiddle": search.CONFIDENCE_THRESHOLD,
                                    "confidenceLate": search.CONFIDENCE_THRESHOLD}, train_phase_two)
