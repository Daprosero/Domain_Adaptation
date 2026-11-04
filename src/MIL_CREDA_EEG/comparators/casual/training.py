"""CASUAL Algorithm 1 under BALLADEER's frozen common-head protocol.

No import-time registration. register() wires the production search: the
plan v42 Sec. 7.7 CASUAL row draws the VAT radius as a fraction of the typical
window L2 norm, log-uniform in [0.01, 1] (`search.VAT_RADIUS_FRACTION`). The
plan does not say how "typical" is measured; here it is the median L2 norm of
the target-training windows of the rotation, measured once at the start of
phase 2 (`typical_window_norm`), an unspecified choice (REPLICATION.md).

Plan v50 §7.7: phase 2 runs its `epochs2` whole and keeps the last epoch (no
early stopping, no restore). With target validation bags (an inner split) every
epoch's validation predictions are recorded for the search's pooled selection;
without them (the refit on every non-test fold) nothing is evaluated.
"""
from __future__ import annotations

import hashlib
import math
import statistics
from pathlib import Path
from collections.abc import Mapping

import torch
from torch import nn

from MIL_CREDA_EEG import eegnet, model_store, search, two_phase
from MIL_CREDA_EEG.mil_wrapper import CommonMILModel, COMMON_PHASE_ONE_FIXED
from .losses import (adaptation_objective, classification_loss, conditional_entropy,
                     discriminator_loss, joint_features, support_alignment,
                     virtual_adversarial_loss)

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v50.md",
    "sections": ["4.5", "7.1", "7.2", "7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["casual_alternation_updates_only_allowed_parameters",
                   "casual_checkpoint_identity_is_checked",
                   "casual_runs_full_budget_keeps_the_last_epoch_and_records_validation",
                   "casual_training_does_not_read_target_labels",
                   "casual_search_hook_uses_common_group_and_the_radius_fraction_axis",
                   "casual_vat_radius_is_the_fraction_of_the_median_target_training_window_norm",
                   "casual_models_round_trip_with_comparator_digest"],
}

STRATEGY = "CASUAL"
PARAM_KEYS = {"lr2", "wd2", "lambda_align", "lambda_entropy", "lambda_vat",
              "vat_radius_fraction"}


def check_params(params):
    if set(params) != PARAM_KEYS:
        raise ValueError(f"CASUAL parameters must be exactly {sorted(PARAM_KEYS)}")
    for k, v in params.items():
        if not math.isfinite(float(v)) or (v < 0 if k == "wd2" else v <= 0):
            raise ValueError(f"invalid {k}: {v}")
    return dict(params)


def typical_window_norm(bags):
    """The typical window L2 norm of Sec. 7.7: median over every window of `bags`.

    Each window's norm is taken over all its (channels, samples); the median of
    an even count is the mean of the two middle values. Reads windows only.
    """
    norms = [float(torch.linalg.vector_norm(w.detach().to(torch.float64)))
             for b in bags for w in b.windows]
    if not norms:
        raise ValueError("the typical window norm needs at least one window")
    return statistics.median(norms)


class DomainDiscriminator(nn.Sequential):
    """Appendix G small-image instantiation: three linear layers, 512 hidden units."""
    def __init__(self, joint_dim=512):
        super().__init__(nn.Linear(joint_dim, 512, dtype=torch.float64),
                         nn.LeakyReLU(0.2), nn.Linear(512, 512, dtype=torch.float64),
                         nn.LeakyReLU(0.2), nn.Linear(512, 1, dtype=torch.float64))


def _state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _features(model, windows, domain):
    z = model.represent(model.bag_pairs(model.embed(windows, domain)))
    p = model.classify(z)
    return joint_features(z, p), p


def alternating_step(model, discriminator, source, target, params, target_optimizer,
                     discriminator_optimizer, device="cpu", *, vat_radius):
    """Algorithm 1: Eq.17 on detached joints, then Eq.16 through frozen phi.

    Target BagTensor.label is deliberately never accessed. Each source label
    supervises one bag, not its windows. Source CE has no target gradient.
    `vat_radius` is absolute: the drawn fraction times the typical window norm.
    """
    params = check_params(params)
    two_phase.set_phase_two_modes(model)
    sw = [b.windows.to(device) for b in source]
    tw = [b.windows.to(device) for b in target]
    with torch.no_grad():
        js, ps = _features(model, sw, "source")
    jt, pt = _features(model, tw, "target")
    target_optimizer.zero_grad(set_to_none=True)
    discriminator_optimizer.zero_grad(set_to_none=True)
    dis = discriminator_loss(discriminator(js.detach()), discriminator(jt.detach()), ps, pt.detach())
    dis.backward()
    discriminator_optimizer.step()
    discriminator_optimizer.zero_grad(set_to_none=True)
    modes = [p.requires_grad for p in discriminator.parameters()]
    for p in discriminator.parameters():
        p.requires_grad_(False)
    try:
        # Recompute scores after discriminator step, keeping the target feature graph.
        alignment = support_alignment(discriminator(js).sigmoid(), discriminator(jt).sigmoid())
        entropy = conditional_entropy(pt)
        vat = virtual_adversarial_loss(model, tw, vat_radius)
        ce = classification_loss(ps, torch.tensor([b.label for b in source], device=device))
        objective = adaptation_objective(ce, alignment, entropy, vat,
                                        **{k: params[k] for k in ("lambda_align", "lambda_entropy", "lambda_vat")})
        if not torch.isfinite(objective):
            raise FloatingPointError("nonfinite CASUAL objective")
        objective.backward()
        target_optimizer.step()
        eegnet.apply_max_norm(model.target_trunk)
    finally:
        for p, mode in zip(discriminator.parameters(), modes):
            p.requires_grad_(mode)
    return {k: float(v.detach()) for k, v in {"loss": objective, "discriminator": dis,
            "alignment": alignment, "entropy": entropy, "vat": vat, "source_ce": ce}.items()}


def adapt_target_phase(model, source_train, target_train, target_valid, params, schedule,
                       *, seed=0, fold=0, transfer="", device="cpu", label_free=False):
    """Exactly `epochs2` epochs, the last kept; validation recorded per epoch when given.

    Caller has loaded the common phase-1 state and called shared bridge(). An empty
    `target_valid` is a refit: nothing is evaluated.
    """
    params = check_params(params)
    schedule = search.check_schedule(schedule)
    two_phase.check_hyper({**COMMON_PHASE_ONE_FIXED, "lr1": 0.001, "wd1": 0.,
                           "lr2": params["lr2"], "wd2": params["wd2"], "tauLocal": 1., **schedule})
    if not source_train or not target_train:
        raise ValueError("CASUAL needs nonempty source train and target train")
    if any(p.requires_grad for p in [*model.source_trunk.parameters(), *model.attention_parameters(), *model.head.parameters()]):
        raise ValueError("call the shared two_phase.bridge before CASUAL adaptation")
    frozen = two_phase.phase_one_state(model)
    device = torch.device(device)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    typical = typical_window_norm(target_train)
    vat_radius = params["vat_radius_fraction"] * typical
    discriminator = DomainDiscriminator(model.source_trunk.output_dim * model.classes).to(device)
    optimizer = torch.optim.Adam(model.target_trunk.parameters(), lr=params["lr2"], weight_decay=params["wd2"])
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=params["lr2"], weight_decay=params["wd2"])
    history, validation = [], []
    keeper = two_phase.LabelFreeBest(label_free)
    size = schedule["bagsPerStep"]
    for epoch in range(schedule["epochs2"]):
        source_batches = two_phase.shuffled_batches(len(source_train), size, generator)
        totals = []
        for step, batch in enumerate(two_phase.shuffled_batches(len(target_train), size, generator)):
            s = source_batches[step % len(source_batches)]
            totals.append(alternating_step(model, discriminator, [source_train[i] for i in s],
                                          [target_train[i] for i in batch], params, optimizer,
                                          dis_optimizer, device, vat_radius=vat_radius))
        ba, free = math.nan, math.nan
        if target_valid:
            probabilities = two_phase.evaluate(model, target_valid, "target", device)
            ba = two_phase.balanced_accuracy([b.label for b in target_valid],
                                             probabilities.argmax(1).tolist())
            free = two_phase.label_free_score(probabilities)
            validation.append(two_phase.validation_record(epoch, target_valid, probabilities))
        row = {"epoch": epoch, "seed": seed, "fold": fold, "transfer": transfer,
               "model": "CASUAL:phase2", "val_balanced_accuracy": ba, "label_free_score": free,
               "typical_window_norm": typical, "vat_radius": vat_radius,
               **{k: sum(r[k] for r in totals) / len(totals) for k in totals[0]}}
        history.append(row)
        keeper.offer(epoch, free, model)
    if model_store.state_digest(two_phase.phase_one_state(model)) != model_store.state_digest(frozen):
        raise two_phase.FrozenPartsMoved("CASUAL changed frozen source/attention/head state")
    model.eval()
    last = history[-1]
    checkpoint = {"target_state": _state(model.target_trunk), "epoch": last["epoch"],
                  "valid_balanced_accuracy": last["val_balanced_accuracy"],
                  "label_free_score": last["label_free_score"], "hyper": params,
                  "discriminator_state": _state(discriminator)}
    return two_phase.with_label_free({
            "checkpoint": checkpoint, "history": history, "validation": validation,
            "best_epoch": last["epoch"], "epochs": len(history),
            "selection": {"primary": "pooled target-validation balanced accuracy, chosen "
                                     "by the search",
                          "secondary": "pooled entropy of mean minus mean entropy, chosen "
                                       "by the search"}}, keeper)


def train_phase_two(data, rotation, seed, record, params: Mapping, schedule: Mapping, device="cpu",
                    *, label_free=False):
    """Existing search trainer seam; checkpoint identity checked before loading."""
    state, meta = record
    expected = {"phase": 1, "name": "common", "rotation": rotation, "seed": seed,
                "transfer": data.transfer.name, "partition_sha256": data.partition_sha256}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("CASUAL requires the same transfer/rotation/seed/partition common phase-1 checkpoint")
    channels = data.channels()
    if meta.get("channels") != channels:
        raise ValueError("phase-1 checkpoint channel counts differ from data")
    torch.manual_seed(seed)
    model = CommonMILModel(channels["source"], channels["target"]).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("target_trunk.") for k in missing):
        raise ValueError(f"common checkpoint does not fit: {missing}, {unexpected}")
    joint = two_phase.bridge(model)
    d = data.rotations[rotation]
    result = adapt_target_phase(model, d["source_train"], d["target_train"], d["target_valid"],
                               params, schedule, seed=seed, fold=rotation,
                               transfer=data.transfer.name, device=device,
                               label_free=label_free)
    return {"strategy": STRATEGY, "model": model, "result": result, "hyper": dict(params),

            "channels": channels, "layer_copy": joint["layer_copy"]}


def register():
    """Explicit integration of the production search (plan v42 Sec. 7.7 CASUAL row).

    Alignment, entropy and VAT weights follow the generic rule; the VAT radius
    is drawn on its own axis as a fraction of the typical window norm.
    """
    search.register_phase_two(STRATEGY, {"lambda_align": search.LOSS_WEIGHT,
                              "lambda_entropy": search.LOSS_WEIGHT,
                              "lambda_vat": search.LOSS_WEIGHT,
                              "vat_radius_fraction": search.VAT_RADIUS_FRACTION},
                              train_phase_two)


def code_digest():
    digest = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(p.name.encode() + b"\0" + p.read_bytes())
    return digest.hexdigest()


def save_final_model(root, data, rotation, seed, trained):
    """Reuse store; include comparator digest which shared CODE_FILES does not cover."""
    checkpoint = trained["result"]["checkpoint"]
    return model_store.save(root, transfer=data.transfer, rotation=rotation, seed=seed,
                            phase=2, name=STRATEGY, state=trained["model"].state_dict(),
                            hyper=checkpoint["hyper"], partition_sha256=data.partition_sha256,
                            channels=trained["channels"], extra={"casual_code_digest": code_digest(),
                            "epoch": checkpoint["epoch"], "valid_balanced_accuracy": checkpoint["valid_balanced_accuracy"],
                            "label_free_score": checkpoint["label_free_score"]})


def load_final_model(path, *, partition_path=None):
    state, meta = model_store.load(path, partition_path=partition_path)
    if meta.get("name") != STRATEGY or meta.get("phase") != 2 or meta.get("casual_code_digest") != code_digest():
        raise model_store.CodeVersionMismatch("CASUAL strategy/phase/comparator digest mismatch")
    model = CommonMILModel(meta["channels"]["source"], meta["channels"]["target"])
    model.load_state_dict(state)
    # Do not bridge here: that would overwrite the trained target trunk.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, meta
