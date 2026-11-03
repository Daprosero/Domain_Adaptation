"""Controlled CCA-LSC variant: common checkpoint -> warmup -> calibrated bag adaptation."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from MIL_CREDA_EEG import eegnet, mil_wrapper, model_store, search, two_phase
from .losses import (CentroidMemory, calibrated_pseudo, calibration_weights,
                     centroid_alignment, discriminative_alignment, domain_adversarial,
                     estimate_prior, label_shift)

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v44.md",
    "sections": ["4.5", "7.1", "7.2", "7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["cca_phase_two_freezes_common_predictor", "cca_target_training_labels_are_unread",
                   "cca_warmup_precedes_fixed_prior", "cca_search_hook_uses_common_checkpoint",
                   "cca_selection_restores_best_epoch", "cca_invalid_inputs_fail_closed",
                   "cca_objective_weights_are_effective"],
}

NAME = "CCA-LSC"
DEFAULTS = {"centroidWeight": 3.0, "discriminativeWeight": 0.6, "adversarialWeight": 1.0,
            "calibrationHm": 1.5, "priorThreshold": 0.5}
WARMUP_EPOCHS = 3  # paper Sec. 3.4, inside the plan's phase 2, not common phase 1
CENTROID_DECAY = 0.9  # unspecified by CCA-LSC paper


def check_params(params: Mapping) -> dict:
    expected = {"lr2", "wd2", *DEFAULTS}
    if set(params) != expected:
        raise ValueError(f"CCA-LSC parameters must be exactly {sorted(expected)}")
    p = {k: float(v) for k, v in params.items()}
    if not all(math.isfinite(v) for v in p.values()):
        raise ValueError("CCA-LSC parameters must be finite")
    if p["lr2"] <= 0 or p["wd2"] < 0 or p["calibrationHm"] <= 0:
        raise ValueError("positive learning rate/hm and nonnegative weight decay required")
    if any(p[k] < 0 for k in ("centroidWeight", "discriminativeWeight", "adversarialWeight")):
        raise ValueError("loss weights must be nonnegative")
    if not 0 <= p["priorThreshold"] <= 1:
        raise ValueError("priorThreshold must be in [0,1]")
    return p


class AlignmentState(nn.Module):
    """Training auxiliaries only: two EMA centroids and a domain discriminator."""
    def __init__(self, width: int, classes: int):
        super().__init__()
        self.source = CentroidMemory(classes, width, CENTROID_DECAY)
        self.target = CentroidMemory(classes, width, CENTROID_DECAY)
        # Unspecified architecture: minimal linear domain logit, no diagnostic head.
        self.discriminator = nn.Linear(width, 1, dtype=torch.float64)


def representations(model, bags, domain, device):
    """Plan 7.2: list of (windows,channels,512) -> (bags,256)."""
    return model.represent(model.bag_pairs(model.embed([b.windows.to(device) for b in bags], domain)))


def adaptation_step(model, auxiliary, source, target, weights, params, device):
    """Paper Eqs. 9/10 over BAGS; source CE is a frozen diagnostic summand."""
    labels = torch.tensor([b.label for b in source], device=device)
    with torch.no_grad():
        zs = representations(model, source, "source", device)
        ps = model.classify(zs)
        ws = ps.max(1).values
        ce = F.nll_loss(ps.clamp_min(1e-12).log(), labels)
    zt = representations(model, target, "target", device)
    pt = model.classify(zt)
    pseudo, wt = calibrated_pseudo(pt.detach(), weights)
    cs, active_s = auxiliary.source(zs, labels, ws)
    ct, active_t = auxiliary.target(zt, pseudo, wt)
    centroid = centroid_alignment(cs, ct, active_s, active_t)
    discrimination = discriminative_alignment(zs, zt, labels, pseudo, ws, wt)
    adversary = domain_adversarial(auxiliary.discriminator, zs, zt)
    objective = (ce + params["centroidWeight"] * centroid
                 + params["discriminativeWeight"] * discrimination
                 + params["adversarialWeight"] * adversary)
    return {"loss": objective, "centroid": centroid, "discriminative": discrimination,
            "adversarial": adversary, "source_ce": ce}


def adapt(model: mil_wrapper.CommonMILModel, source_train, target_train, target_valid,
          params: Mapping, schedule: Mapping, *, seed: int, fold: int = 0,
          transfer: str = "", warmup_epochs: int = WARMUP_EPOCHS,
          device: torch.device | str = "cpu", label_free: bool = False) -> dict:
    """Adapt a selected common phase-1 model; CPU/GPU placement belongs to caller.

    target_train labels are never read. No target test input exists. Warmup uses
    the same CCA losses without LSC; estimate target prior once at its boundary.
    Caller must budget at least one calibrated epoch after warmup.
    """
    if not isinstance(model, mil_wrapper.CommonMILModel):
        raise ValueError("CCA-LSC requires the shared common MIL phase-1 architecture")
    params = check_params(params)
    schedule = search.check_schedule(schedule)
    # Reuse the shared schedule guard rather than duplicate its integer rules.
    two_phase.check_hyper({**mil_wrapper.COMMON_PHASE_ONE_FIXED, "lr1": 1., "wd1": 0.,
                          "lr2": params["lr2"], "wd2": params["wd2"], "tauLocal": 1.,
                          **schedule})
    if int(warmup_epochs) != warmup_epochs or warmup_epochs < 1 or schedule["epochs2"] <= warmup_epochs:
        raise ValueError("phase 2 must include positive warmup and at least one calibrated epoch")
    if not source_train or not target_train or not target_valid:
        raise ValueError("source training, target training and target validation bags required")
    # Enforce split separation without reading target training labels.
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
    auxiliary = AlignmentState(model.target_trunk.output_dim, model.classes).to(device)
    optimizer = torch.optim.Adam([*model.target_trunk.parameters(),
                                  *auxiliary.discriminator.parameters()],
                                 lr=params["lr2"], weight_decay=params["wd2"])
    labels = torch.tensor([b.label for b in source_train], device=device)
    counts = torch.bincount(labels, minlength=model.classes).to(torch.float64)
    source_prior = counts / counts.sum()  # original train bags, NOT balanced draws
    weights = torch.ones(model.classes, dtype=torch.float64, device=device)
    target_prior, history, best_state, best_aux = None, [], None, None
    best, best_epoch = -math.inf, -1
    keeper = two_phase.LabelFreeBest(label_free)
    for epoch in range(schedule["epochs2"]):
        if epoch == warmup_epochs:
            # evaluate restores modes; reads windows only, no target labels.
            target_prior = estimate_prior(two_phase.evaluate(model, target_train, "target", device),
                                          params["priorThreshold"])
            weights = calibration_weights(label_shift(source_prior, target_prior),
                                          params["calibrationHm"])
        two_phase.set_phase_two_modes(model)
        sums = {k: 0. for k in ("loss", "centroid", "discriminative", "adversarial", "source_ce")}
        batches = two_phase.shuffled_batches(len(target_train), schedule["bagsPerStep"], generator)
        for batch in batches:
            source = [source_train[i] for i in queues.draw(schedule["bagsPerStep"] // model.classes)]
            terms = adaptation_step(model, auxiliary, source, [target_train[i] for i in batch],
                                    weights, params, device)
            if not torch.isfinite(terms["loss"]):
                raise FloatingPointError("CCA-LSC objective is not finite")
            optimizer.zero_grad()
            terms["loss"].backward()
            optimizer.step()
            eegnet.apply_max_norm(model.target_trunk)
            for k in sums:
                sums[k] += float(terms[k].detach()) / len(batches)
        prediction = two_phase.evaluate(model, target_valid, "target", device)
        accuracy = two_phase.balanced_accuracy([b.label for b in target_valid], prediction.argmax(1).tolist())
        score = two_phase.label_free_score(prediction)
        history.append({"epoch": epoch, "seed": seed, "fold": fold, "transfer": transfer,
                        "model": f"{NAME}:phase2", **sums, "calibrated": epoch >= warmup_epochs,
                        "val_balanced_accuracy": accuracy, "label_free_score": score})
        if accuracy > best:
            best, best_epoch = accuracy, epoch
            best_state = copy.deepcopy(model.target_trunk.state_dict())
            best_aux = copy.deepcopy(auxiliary.state_dict())
        keeper.offer(epoch, score, model)
    final_state = copy.deepcopy(model.target_trunk.state_dict())
    model.target_trunk.load_state_dict(best_state)
    auxiliary.load_state_dict(best_aux)
    if not all(torch.equal(v, two_phase.phase_one_state(model)[k])
               for k, v in joint["frozen_state"].items()):
        raise two_phase.FrozenPartsMoved("CCA-LSC changed a frozen phase-1 part")
    restored = two_phase.evaluate(model, target_valid, "target", device)
    if two_phase.balanced_accuracy([b.label for b in target_valid], restored.argmax(1).tolist()) != best:
        raise two_phase.RestoreFailed("CCA-LSC restored validation score differs")
    return two_phase.with_label_free({
            "checkpoint": {"target_state": best_state, "auxiliary_state": best_aux,
                            "epoch": best_epoch, "valid_balanced_accuracy": best,
                            "label_free_score": history[best_epoch]["label_free_score"],
                            "hyper": {**params, **schedule, "warmup_epochs": warmup_epochs}},
            "best_epoch": best_epoch, "epochs": len(history), "history": history,
            "source_prior": source_prior.detach().cpu(), "target_prior": target_prior.detach().cpu(),
            "calibration_weights": weights.detach().cpu(), "final_target_state": final_state,
            "layer_copy": joint["layer_copy"],
            "selection": {"primary": "target-validation balanced accuracy",
                          "best_label_free_epoch": max(range(len(history)),
                                                       key=lambda i: history[i]["label_free_score"])}},
        keeper)


def train_phase_two(data, rotation, seed, record, params, schedule, device="cpu", *,
                    label_free=False):
    """Shared search trainer hook; reject a checkpoint from a different rotation/group."""
    state, meta = record
    expected = {"phase": 1, "name": "common", "transfer": data.transfer.name,
                "rotation": rotation, "seed": seed, "partition_sha256": data.partition_sha256}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise search.SearchError("CCA-LSC needs the common checkpoint of this transfer/rotation/seed/partition")
    torch.manual_seed(seed)
    channels = data.channels()
    model = model_store.build_model("common", channels["source"], channels["target"], meta["hyper"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("target_trunk.") for k in missing):
        raise search.SearchError("incompatible common phase-1 state")
    d = data.rotations[rotation]
    result = adapt(model, d["source_train"], d["target_train"], d["target_valid"], params,
                   schedule, seed=seed, fold=rotation, transfer=data.transfer.name, device=device,
                   label_free=label_free)

    return {"strategy": NAME, "model": model, "result": result,
            "hyper": result["checkpoint"]["hyper"], "channels": channels,
            "layer_copy": result["layer_copy"]}


def register():
    """Explicit registration avoids changing global search state merely by importing."""
    search.register_phase_two(NAME, {"centroidWeight": search.LOSS_WEIGHT,
                                    "discriminativeWeight": search.LOSS_WEIGHT,
                                    "adversarialWeight": search.LOSS_WEIGHT,
                                    "calibrationHm": search.LOSS_WEIGHT,
                                    "priorThreshold": search.CONFIDENCE_THRESHOLD}, train_phase_two)
