"""Target-only phase 2 of the controlled CLUST variant (plan v56 §§7.3–7.4, 7.7).

No source data argument exists. Source weights, common attention and diagnostic
head stay fixed. EMA components are auxiliary, never diagnostic heads. Phase 2
runs its `epochs2` whole and keeps the last epoch (no early stopping, v56 §7.7);
with target validation bags every epoch's validation predictions are recorded for
the search's pooled selection, and without them (the refit) nothing is evaluated.
`secondary_checkpoint` stays a per-run reading (the run's best label-free epoch,
None in a refit) for `clust.save(selection="secondary")`; the campaign selects
the label-free (trial, epoch) on the pooled predictions instead.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from MIL_CREDA_EEG import eegnet, two_phase
from MIL_CREDA_EEG.mil_wrapper import CommonMILModel
from . import losses

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v56.md",
    "sections": ["4.5", "7.2", "7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["clust_adaptation_never_reads_training_labels_or_source_bags",
                   "clust_only_target_encoder_changes_and_all_gradients_are_finite",
                   "clust_teacher_uses_exact_ema_and_never_predicts_diagnosis",
                   "clust_keeps_the_last_epoch_and_records_validation_and_dashboard_history",
                   "clust_rejects_invalid_parameters_and_empty_inputs",
                   "clust_auxiliary_state_round_trips",
                   "clust_teacher_momentum_is_one_minus_the_drawn_complement"],
}
NAME = "CLUST — variante source-free"
PARAMETERS = ("lr2", "wd2", "clusteringWeight", "consistencyWeight", "informationWeight",
              "confidence", "emaMomentumComplement")


def check_parameters(params):
    if set(params) != set(PARAMETERS):
        raise ValueError(f"CLUST parameters must be exactly {PARAMETERS}")
    out = {k: float(v) for k, v in params.items()}
    if not all(math.isfinite(v) for v in out.values()):
        raise ValueError("CLUST parameters must be finite")
    if out["lr2"] <= 0 or any(out[k] < 0 for k in PARAMETERS[1:5]):
        raise ValueError("positive learning rate and nonnegative decay/weights required")
    if not .5 <= out["confidence"] <= 1 or not 0 < out["emaMomentumComplement"] <= 1:
        raise ValueError("confidence in [.5,1] and EMA momentum complement 1 - rho in (0,1] "
                         "required")
    return out


def ema_momentum(params):
    """Plan v42 Sec. 7.7: the search draws 1 - rho log-uniformly; the teacher uses rho."""
    return 1.0 - float(params["emaMomentumComplement"])


def clone_state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


class Auxiliary(nn.Module):
    """Eq.5/13: EMA target encoder and target prototype bank, not inference."""

    def __init__(self, model):
        super().__init__()
        self.teacher = deepcopy(model.target_trunk).requires_grad_(False).eval()
        source = F.normalize(model.head.linear.weight.detach().clone(), dim=-1)
        self.register_buffer("source", source)
        self.register_buffer("target", source.clone())
        self.register_buffer("alpha", source.new_tensor(.5))

    @torch.no_grad()
    def update(self, model, centroids, active, momentum):
        """Eq.5 after optimizer step; float buffers EMA, integer counters copied."""
        for teacher, student in zip(self.teacher.parameters(), model.target_trunk.parameters()):
            teacher.mul_(momentum).add_(student, alpha=1 - momentum)
        for teacher, student in zip(self.teacher.buffers(), model.target_trunk.buffers()):
            if teacher.is_floating_point():
                teacher.mul_(momentum).add_(student, alpha=1 - momentum)
            else:
                teacher.copy_(student)
        self.target.copy_(losses.ema_update(self.target, centroids, active, momentum))


def representations(model, windows):
    """Plan §7.2: common attention at bag level, no window pseudo-labels."""
    return model.represent(model.bag_pairs(model.embed(windows, "target")))


@torch.no_grad()
def teacher_representations(model, auxiliary, windows):
    auxiliary.teacher.eval()
    sizes = [len(w) for w in windows]
    features = auxiliary.teacher(torch.cat(windows)).to(torch.float64)
    return model.represent(model.bag_pairs(list(features.split(sizes))))


def adaptation_step(model, target_bags, params, device):
    """Eqs.2–15, target bag windows only; labels are intentionally unread."""
    auxiliary = model.clust_auxiliary
    windows = [bag.windows.to(device) for bag in target_bags]
    z = F.normalize(representations(model, windows), dim=-1)
    with torch.no_grad():
        teacher_z = F.normalize(teacher_representations(model, auxiliary, windows), dim=-1)
        teacher_p = losses.prototype_probabilities(teacher_z, auxiliary.source)
        membership = losses.confidence_assignments(teacher_p, params["confidence"])
    centroids, counts = losses.target_centroids(z, membership, auxiliary.target)
    feature, semantic, per_class = losses.clustering_terms(z, auxiliary.source, centroids,
                                                         membership)
    target_p = losses.prototype_probabilities(z, auxiliary.target)
    consistency = losses.symmetric_kl(teacher_p, target_p)
    information = losses.information_maximization(target_p)
    total = losses.objective(feature, semantic, consistency, information, auxiliary.alpha,
                             [params[k] for k in PARAMETERS[2:5]])
    return {"objective": total, "feature": feature, "semantic": semantic,
            "consistency": consistency, "information": information,
            "centroids": centroids.detach(), "active": counts > 0,
            "per_class": per_class.detach(), "confident": membership.any(-1).double().mean()}


def adapt_target_phase(model, target_train, target_valid, params, schedule, generator,
                       device="cpu", *, seed=0, fold=0, transfer="", label_free=False):
    """Exactly `epochs2` epochs, the last kept; validation labels only recorded for the
    search's selection, never used to adapt.

    Caller supplies a bridged, phase-1 common model; an empty `target_valid` is a refit.
    No target test input is accepted.
    """
    params = check_parameters(params)
    epochs, batch_size = schedule["epochs2"], schedule["bagsPerStep"]
    if any(isinstance(v, bool) or int(v) != v or v < 1 for v in (epochs, batch_size)):
        raise ValueError("epochs2 and bagsPerStep must be positive integers")
    if not target_train:
        raise ValueError("nonempty target training bags required")
    if not isinstance(model, CommonMILModel):
        raise TypeError("CLUST requires the shared common MIL model")
    frozen = two_phase.phase_one_state(model)
    if any(p.requires_grad for p in [*model.source_trunk.parameters(),
                                    *model.attention_parameters(), *model.head.parameters()]):
        raise ValueError("call the shared bridge before CLUST adaptation")
    device = torch.device(device)
    model.to(device)
    model.clust_auxiliary = Auxiliary(model).to(device)
    momentum = ema_momentum(params)
    optimizer = torch.optim.Adam(model.target_trunk.parameters(), lr=params["lr2"],
                                 weight_decay=params["wd2"])
    history, seconds, validation = [], [], []
    best_free, secondary = -math.inf, None
    labels = [b.label for b in target_valid]  # selection only, never adaptation
    keeper = two_phase.LabelFreeBest(label_free)
    for epoch in range(int(epochs)):
        started = time.perf_counter()
        two_phase.set_phase_two_modes(model)
        sums = {k: 0.0 for k in ("loss", "feature", "semantic", "consistency", "information",
                                 "confident")}
        class_sum = torch.zeros_like(model.clust_auxiliary.source[:, 0])
        batches = two_phase.shuffled_batches(len(target_train), int(batch_size), generator)
        seen = 0
        for batch in batches:
            step = adaptation_step(model, [target_train[i] for i in batch], params, device)
            optimizer.zero_grad(set_to_none=True)
            step["objective"].backward()
            if not torch.isfinite(step["objective"]) or any(
                    p.grad is None or not torch.isfinite(p.grad).all()
                    for p in model.target_trunk.parameters()):
                raise FloatingPointError("nonfinite loss or missing/nonfinite target gradient")
            optimizer.step()
            eegnet.apply_max_norm(model.target_trunk)
            model.clust_auxiliary.update(model, step["centroids"], step["active"], momentum)
            for k in sums:
                sums[k] += len(batch) * float(step["objective" if k == "loss" else k].detach())
            class_sum += len(batch) * step["per_class"]
            seen += len(batch)
        means = {k: v / seen for k, v in sums.items()}
        # Eq.21–23, one detached update at the epoch boundary.
        model.clust_auxiliary.alpha.fill_(losses.dynamic_alpha(
            class_sum.new_tensor(means["feature"]), class_sum / seen))
        ba, free = math.nan, math.nan
        if target_valid:
            g = two_phase.evaluate(model, target_valid, "target", device)
            ba = two_phase.balanced_accuracy(labels, g.argmax(-1).tolist())
            free = two_phase.label_free_score(g)
            validation.append(two_phase.validation_record(epoch, target_valid, g))
        seconds.append(time.perf_counter() - started)
        history.append({"epoch": epoch, "seed": seed, "fold": fold, "transfer": transfer,
                        "model": NAME + ":phase2", **means, "val_balanced_accuracy": ba,
                        "label_free_score": free, "alpha": float(model.clust_auxiliary.alpha)})
        keeper.offer(epoch, free, model)
        if free > best_free:
            best_free, secondary = free, {"epoch": epoch, "state": clone_state(model)}
    current = model.state_dict()
    if any(not torch.equal(v.to(current[k]), current[k]) for k, v in frozen.items()):
        raise two_phase.FrozenPartsMoved("CLUST changed frozen phase-one state")
    model.eval()
    last = history[-1]
    return two_phase.with_label_free({
            "checkpoint": {"state": clone_state(model), "epoch": last["epoch"],
                           "valid_balanced_accuracy": last["val_balanced_accuracy"],
                           "label_free_score": last["label_free_score"],
                           "hyper": params},
            "secondary_checkpoint": None if secondary is None else {
                **secondary, "label_free_score": best_free,
                "valid_balanced_accuracy": history[secondary["epoch"]]["val_balanced_accuracy"],
                "hyper": params},
            "best_epoch": last["epoch"], "epochs": len(history), "history": history,
            "validation": validation,
            "seconds": seconds, "seconds_per_epoch": sum(seconds) / len(seconds),
            "selection": {"primary": "pooled target-validation balanced accuracy, chosen "
                                     "by the search",
                          "secondary": "pooled label-free score, chosen by the search"}},
        keeper)



def restore_model(state, source_channels, target_channels, device="cpu"):
    """Rebuild saved common diagnostic model plus its selected auxiliary state."""
    model = CommonMILModel(source_channels, target_channels).to(device)
    two_phase.bridge(model)
    model.clust_auxiliary = Auxiliary(model).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()
