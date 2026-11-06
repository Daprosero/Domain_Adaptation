"""CS+CCS controlled variant: common phase-1 checkpoint -> target-only phase 2 (Eq. 90).

Plan v42 Secs. 7.2-7.4 and 7.7: the bag representation of the common MIL wrapper is
the paper's z, the common head's prediction is its y_hat, and only the target
trunk trains. Eq. 90, L = L_CE + lambda D_CS + beta D_CCS, keeps its source
cross-entropy as a summand computed on frozen source bags: with the source
trunk, the attention and the head frozen it carries no gradient, so it is
reported and changes nothing. Target training labels are never read; target
validation labels are only recorded for the search's pooled selection (Sec. 4.5).

Plan v56 §7.7: phase 2 runs its `epochs2` whole and keeps the last epoch (no
early stopping, no restore); without target validation bags (the refit on every
non-test fold) nothing is evaluated.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence

import torch
from torch.nn import functional as F

from MIL_CREDA_EEG import eegnet, mil_wrapper, model_store, search, two_phase
from .divergences import ccs_divergence, cs_divergence, median_bandwidth, normalise

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v57.md",
    "sections": ["4.5", "7.1", "7.2", "7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": [
        "cs_ccs_phase_two_freezes_common_predictor",
        "cs_ccs_target_training_labels_are_unread",
        "cs_ccs_bandwidths_are_fixed_from_training_bags_at_phase_two_start",
        "cs_ccs_prediction_kernel_is_fixed_at_sigma_one_on_normalised_predictions",
        "cs_ccs_objective_is_eq_90_with_frozen_source_ce",
        "cs_ccs_keeps_the_last_epoch_and_records_validation_per_epoch",
        "cs_ccs_invalid_inputs_fail_closed",
        "cs_ccs_search_registration_is_explicit_and_restorable",
        "cs_ccs_search_hook_rejects_foreign_phase_one_records",
        "cs_ccs_saved_state_round_trips",
        "cs_ccs_training_is_reproducible",
    ],
}

NAME = "CS+CCS"
#: Sec. 4.2 / C.5: lambda = beta = 1. Representation kernel: multiplier 1 x median
#: replaces sigma = 1 (plan v42 Sec. 7.7); the prediction kernel keeps the paper's
#: sigma = 1 (`PREDICTION_SIGMA`) and is not searched.
DEFAULTS = {"csWeight": 1.0, "ccsWeight": 1.0, "zKernelMultiplier": 1.0}
#: Plan v42 Sec. 7.7 and the paper's Sec. 4.2: the prediction kernel is fixed at
#: sigma = 1 on the L2-normalised softmax predictions (the median rule gives a
#: near-zero width on the near-uniform early phase-2 predictions).
PREDICTION_SIGMA = 1.0
#: Sec. 4.2 "SGD optimizer"; momentum unspecified by the paper (upstream script default).
MOMENTUM = 0.9
HISTORY_KEYS = ("loss", "source_ce", "cs", "ccs")


def check_params(params: Mapping) -> dict:
    expected = {"lr2", "wd2", *DEFAULTS}
    if set(params) != expected:
        raise ValueError(f"CS+CCS parameters must be exactly {sorted(expected)}")
    p = {k: float(v) for k, v in params.items()}
    if not all(math.isfinite(v) for v in p.values()):
        raise ValueError("CS+CCS parameters must be finite")
    if p["lr2"] <= 0 or p["wd2"] < 0:
        raise ValueError("positive learning rate and nonnegative weight decay required")
    if p["csWeight"] < 0 or p["ccsWeight"] < 0:
        raise ValueError("divergence weights must be nonnegative")
    if p["zKernelMultiplier"] not in search.KERNEL_MULTIPLIER.choices:
        raise ValueError(f"zKernelMultiplier must follow the shared multiplier rule "
                         f"{search.KERNEL_MULTIPLIER.choices}")
    return p


def check_roles(source_train, target_train, target_valid) -> None:
    """Non-empty training roles (a refit has no validation bags), disjoint subjects; reads
    subjects only, never target labels."""
    if not source_train or not target_train:
        raise ValueError("source training and target training bags required")
    s = {b.subject for b in source_train}
    t = {b.subject for b in target_train}
    v = {b.subject for b in target_valid}
    if s & t or s & v or t & v:
        raise ValueError("source, target training and target validation subjects must be disjoint")


def kernel_inputs(model, bags: Sequence[two_phase.BagTensor], domain: str, device):
    """Plan 7.2/7.4: bags -> (normalised z (B, d), normalised y_hat (B, C), raw z)."""
    z = model.represent(model.bag_pairs(model.embed([b.windows.to(device) for b in bags], domain)))
    return normalise(z), normalise(model.classify(z)), z


@torch.no_grad()
def phase_two_bandwidths(model, source_train, target_train, params, device) -> dict:
    """Fixed once after the bridge from the TRAINING bags of both sides (windows only).

    The representation width is the median rule; the prediction width is the
    fixed `PREDICTION_SIGMA` (plan v42 Sec. 7.7). Each bag passes alone in eval mode (BatchNorm statistics untouched); the
    modes are restored. Target validation bags are not read.
    """
    modes = {m: m.training for m in model.modules()}
    model.eval()
    try:
        rows = [kernel_inputs(model, [b], domain, device)
                for domain, bags in (("source", source_train), ("target", target_train))
                for b in bags]
    finally:
        for m, mode in modes.items():
            m.train(mode)
    z = torch.cat([r[0] for r in rows])
    return {"zBandwidth": median_bandwidth(z, params["zKernelMultiplier"]),
            "yBandwidth": PREDICTION_SIGMA}


def cs_ccs_objective(source_ce, cs, ccs, cs_weight: float, ccs_weight: float):
    """Eq. 90: L = L_CE + lambda D_CS + beta D_CCS."""
    return source_ce + cs_weight * cs + ccs_weight * ccs


def adaptation_step(model, source, target, params, bandwidths, device) -> dict:
    """One step over bags: frozen source side without graph, target side with it."""
    labels = torch.tensor([b.label for b in source], device=device)
    with torch.no_grad():
        zs, ys, raw = kernel_inputs(model, source, "source", device)
        source_ce = F.cross_entropy(model.head.linear(raw), labels)
    zt, yt, _ = kernel_inputs(model, target, "target", device)
    cs = cs_divergence(zs, zt, bandwidths["zBandwidth"])
    ccs = ccs_divergence(zs, zt, ys, yt, bandwidths["zBandwidth"], bandwidths["yBandwidth"])
    loss = cs_ccs_objective(source_ce, cs, ccs, params["csWeight"], params["ccsWeight"])
    return {"loss": loss, "source_ce": source_ce, "cs": cs, "ccs": ccs}


def adapt(model: mil_wrapper.CommonMILModel, source_train, target_train, target_valid,
          params: Mapping, schedule: Mapping, *, seed: int, fold: int = 0,
          transfer: str = "", device: torch.device | str = "cpu") -> dict:
    """Phase 2 of a common phase-1 model; CPU/GPU placement belongs to the caller."""
    if not isinstance(model, mil_wrapper.CommonMILModel):
        raise ValueError("CS+CCS requires the shared common MIL phase-1 architecture")
    params = check_params(params)
    schedule = search.check_schedule(schedule)
    # Reuse the shared schedule guard; the MIL-CREDA-only fields are never read.
    two_phase.check_hyper({**mil_wrapper.COMMON_PHASE_ONE_FIXED, "lr1": 1., "wd1": 0.,
                          "lr2": params["lr2"], "wd2": params["wd2"], "tauLocal": 1.,
                          **schedule})
    check_roles(source_train, target_train, target_valid)
    device = torch.device(device)
    model.to(device)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    queues = two_phase.ClassQueues(source_train, model.classes, generator)
    joint = two_phase.bridge(model)
    bandwidths = phase_two_bandwidths(model, source_train, target_train, params, device)
    optimizer = torch.optim.SGD(model.target_trunk.parameters(), lr=params["lr2"],
                                weight_decay=params["wd2"], momentum=MOMENTUM)
    hyper = {**params, **schedule, **bandwidths, "momentum": MOMENTUM}
    valid_labels = [b.label for b in target_valid]
    history, validation = [], []
    for epoch in range(schedule["epochs2"]):
        two_phase.set_phase_two_modes(model)
        sums = dict.fromkeys(HISTORY_KEYS, 0.)
        batches = two_phase.shuffled_batches(len(target_train), schedule["bagsPerStep"], generator)
        for batch in batches:
            source = [source_train[i] for i in queues.draw(schedule["bagsPerStep"] // model.classes)]
            terms = adaptation_step(model, source, [target_train[i] for i in batch], params,
                                    bandwidths, device)
            if not torch.isfinite(terms["loss"]):
                raise FloatingPointError("CS+CCS objective is not finite")
            optimizer.zero_grad(set_to_none=True)
            terms["loss"].backward()
            optimizer.step()
            eegnet.apply_max_norm(model.target_trunk)
            for k in sums:
                sums[k] += float(terms[k].detach()) / len(batches)
        accuracy = math.nan
        if target_valid:
            prediction = two_phase.evaluate(model, target_valid, "target", device)
            accuracy = two_phase.balanced_accuracy(valid_labels, prediction.argmax(1).tolist())
            validation.append(two_phase.validation_record(
                epoch, target_valid, prediction,
                two_phase.represent(model, target_valid, "target", device)))
        history.append({"epoch": epoch, "seed": seed, "fold": fold, "transfer": transfer,
                        "model": f"{NAME}:phase2", **sums, **bandwidths,
                        "val_balanced_accuracy": accuracy})
    final_state = copy.deepcopy(model.target_trunk.state_dict())
    two_phase.set_phase_two_modes(model)
    if not all(torch.equal(v, two_phase.phase_one_state(model)[k])
               for k, v in joint["frozen_state"].items()):
        raise two_phase.FrozenPartsMoved("CS+CCS changed a frozen phase-1 part")
    last = len(history) - 1
    checkpoint = {"epoch": last, "valid_balanced_accuracy": accuracy,
                  "hyper": hyper, "target_state": final_state}
    return {
            "checkpoint": checkpoint, "best_epoch": last, "epochs": len(history),
            "history": history, "validation": validation,
            "bandwidths": bandwidths, "final_target_state": final_state,
            "layer_copy": joint["layer_copy"],
            "selection": two_phase.SELECTION}


def check_record(data, rotation, seed, record) -> dict:
    """The common phase-1 record of exactly this transfer, rotation, seed and partition."""
    state, meta = record
    expected = {"phase": 1, "name": mil_wrapper.GROUP, "transfer": data.transfer.name,
                "rotation": rotation, "seed": search.PHASE_ONE_SEED,
                "partition_sha256": data.partition_sha256,
                "channels": data.channels()}
    wrong = {k: meta.get(k) for k, v in expected.items() if meta.get(k) != v}
    if wrong:
        raise search.SearchError(f"CS+CCS needs the common phase-1 checkpoint of this "
                                 f"transfer/rotation/seed/partition/channels: {wrong}")
    return state


def train_phase_two(data, rotation, seed, record, params, schedule, device="cpu"):
    """Shared search trainer hook (`search.register_phase_two`)."""
    state = check_record(data, rotation, seed, record)
    torch.manual_seed(seed)
    channels = data.channels()
    model = mil_wrapper.CommonMILModel(channels["source"], channels["target"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("target_trunk.") for k in missing):
        raise search.SearchError(f"incompatible common phase-1 state (missing {missing}, "
                                 f"unexpected {unexpected})")
    d = data.rotations[rotation]
    result = adapt(model, d["source_train"], d["target_train"], d["target_valid"], params,
                   schedule, seed=seed, fold=rotation, transfer=data.transfer.name,
                   device=device)

    return {"strategy": NAME, "model": model, "result": result,
            "hyper": result["checkpoint"]["hyper"], "channels": channels,
            "layer_copy": result["layer_copy"]}


def register() -> None:
    """Explicit opt-in; importing this package never changes the search registries.

    Plan v42 Sec. 7.7: one searched kernel axis, the representations'.
    """
    search.register_phase_two(NAME, {"csWeight": search.LOSS_WEIGHT,
                                     "ccsWeight": search.LOSS_WEIGHT,
                                     "zKernelMultiplier": search.KERNEL_MULTIPLIER},
                              train_phase_two)


def load_phase_two(path, *, partition_path=None, allow_code_mismatch=False):
    """A stored CS+CCS phase-2 model after the shared store's refusals; frozen, eval."""
    state, meta = model_store.load(path, partition_path=partition_path,
                                   allow_code_mismatch=allow_code_mismatch)
    if meta["phase"] != 2 or meta["name"] != NAME:
        raise ValueError("expected a CS+CCS phase-two checkpoint")
    model = mil_wrapper.CommonMILModel(meta["channels"]["source"], meta["channels"]["target"])
    model.load_state_dict(state)
    # No bridge here: it would overwrite the trained target trunk.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, meta
