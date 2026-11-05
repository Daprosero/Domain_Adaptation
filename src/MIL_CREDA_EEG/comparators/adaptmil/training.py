"""Controlled source-first / target-only training (plan v50 §§4.5,7.3–7.4,7.7).

Not the paper's align-first / attention-second schedule. Both phases run their
epoch count whole and keep the last epoch (no patience, no restore, v50 §7.7);
with validation bags every epoch's validation predictions are recorded for the
search's pooled selection, and without them (a refit) nothing is evaluated. Shared partition,
assignment, batching, metrics, bridge and model-store machinery are reused.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from MIL_CREDA_EEG import model_store, search, two_phase
from .model import AdaptMILModel
from .objectives import alignment_objective, median_bandwidth, stage_two_objective

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v50.md",
    "sections": ["4.5", "7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["adaptmil_source_pretraining_runs_its_epochs_and_keeps_the_last",
                   "adaptmil_bridge_freezes_all_but_target_and_preserves_buffers",
                   "adaptmil_adaptation_updates_only_target_with_label_blind_loss",
                   "adaptmil_validation_is_recorded_per_epoch_without_training_on_labels",
                   "adaptmil_search_hook_restarts_from_matching_phase_one_state",
                   "adaptmil_store_checks_comparator_code_identity",
                   "adaptmil_training_guards_hyperparameters_and_roles"],
}

GROUP = "AdaptMIL"
PHASE_TWO_KEYS = {"lr2", "wd2", "mmdWeight", "kernelMultiplier"}


def source_digest():
    """The generic store's code_version excludes comparator modules; seal them too."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _roles(*roles, optional=()):
    """Disjoint roles; the training ones nonempty. A validation role named in `optional`
    may be empty: a refit has no validation bags (v50 §4.5)."""
    sets = []
    for k, role in enumerate(roles):
        if not role:
            if k in optional:
                continue
            raise ValueError("training and validation roles must be nonempty")
        subjects = {b.subject for b in role}
        if len(subjects) != len(role):
            raise ValueError("duplicate subject in a role")
        if any(subjects & seen for seen in sets):
            raise ValueError("subjects overlap between training/validation/domain roles")
        sets.append(subjects)


def _optimizer(parameters, lr, wd):
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(wd) or wd < 0:
        raise ValueError("learning rate must be positive and decay nonnegative, both finite")
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=wd)


def _record(epoch, phase, seed, fold, transfer, **values):
    return {"epoch": epoch, "seed": seed, "fold": fold, "transfer": transfer,
            "model": f"{GROUP}:phase{phase}", **{k: float(v) for k, v in values.items()}}


def train_source_phase(model, source_train, source_valid, hyper, generator, device,
                       *, seed=0, fold=0, transfer=""):
    """Plan phase1: native source-bag CE for exactly `epochs1` epochs, the last kept.

    Accepts the shared phase-one hyper schema; phase-two placeholder NaNs and
    neutral common-group attention/sigma fields are unread by this trainer.
    """
    hyper = two_phase.check_hyper(hyper)
    _roles(source_train, source_valid, optional=(1,))
    device = torch.device(device)
    optimizer = _optimizer([*model.source_trunk.parameters(),
                            *model.attention_parameters(), *model.head.parameters()],
                           hyper["lr1"], hyper["wd1"])
    history, validation = [], []
    for epoch in range(hyper["epochs1"]):
        model.train()
        losses = []
        for indices in two_phase.interleaved_batches(source_train, hyper["bagsPerStep"],
                                                     generator):
            batch = [source_train[i] for i in indices]
            loss = stage_two_objective(model, [b.windows.to(device) for b in batch],
                                       torch.tensor([b.label for b in batch], device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.constrain("source_trunk", "head")
            losses.append(float(loss.detach()))
        accuracy = math.nan
        if source_valid:
            predictions = two_phase.evaluate(model, source_valid, "source", device)
            labels = [b.label for b in source_valid]
            accuracy = two_phase.balanced_accuracy(labels, predictions.argmax(1).tolist())
            validation.append(two_phase.validation_record(epoch, source_valid, predictions))
        history.append(_record(epoch, 1, seed, fold, transfer,
                               loss=sum(losses) / len(losses),
                               val_balanced_accuracy=accuracy))
    model.eval()
    last = len(history) - 1
    return {"checkpoint": {"state": two_phase.phase_one_state(model), "sigma": None,
                           "epoch": last, "valid_balanced_accuracy": accuracy,
                           "hyper": dict(hyper)},
            "best_epoch": last, "epochs": len(history), "history": history,
            "best_valid_balanced_accuracy": accuracy, "validation": validation}


def bridge(model):
    """Reuse the plan's layer-copy and freeze bridge, P included in native aggregator."""
    joint = two_phase.bridge(model)
    model._adaptation = True
    # Drop stale phase-one gradients as well as disabling future gradients.
    for p in model.parameters():
        p.grad = None
    return joint


@torch.no_grad()
def _bandwidth(model, source_train, target_train, multiplier, device):
    """Width in the mode adaptation trains in: BatchNorm on batch statistics, dropout
    off, buffers restored -- MIL-CREDA's sigma rule (`two_phase.within_bag_median_distance`).
    An eval-mode width is ~35x smaller on these trunks and saturates the train-mode kernel."""
    modes = {m: m.training for m in model.modules()}
    buffers = {k: v.clone() for k, v in model.state_dict().items()}
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    try:
        source = model.representations([b.windows.to(device) for b in source_train],
                                       "source", mean=True, chunk=two_phase.EVAL_CHUNK)
        target = model.representations([b.windows.to(device) for b in target_train],
                                       "target", mean=True, chunk=two_phase.EVAL_CHUNK)
        return median_bandwidth(source, target, multiplier)
    finally:
        model.load_state_dict(buffers)
        for module, mode in modes.items():
            module.train(mode)


def adapt_target_phase(model, source_train, target_train, target_valid, params, schedule,
                       generator, device, *, seed=0, fold=0, transfer="", label_free=False):
    """Mean-pool MMD for exactly `epochs2` epochs, the last kept; validation per epoch."""
    schedule = search.check_schedule(schedule)
    # Reuse the shared schedule guards without treating AdaptMIL as MIL-CREDA.
    two_phase.check_hyper({**dict.fromkeys(two_phase.PHASE_ONE_KEYS, float("nan")),
                           "lr2": params.get("lr2"), "wd2": params.get("wd2"),
                           "tauLocal": float("nan"), **schedule})
    if set(params) != PHASE_TWO_KEYS:
        raise ValueError(f"AdaptMIL phase2 keys must be {sorted(PHASE_TWO_KEYS)}")
    if (not math.isfinite(params["mmdWeight"]) or params["mmdWeight"] < 0
            or params["kernelMultiplier"] not in (0.5, 0.75, 1.0, 1.5)):
        raise ValueError("invalid mmdWeight or kernelMultiplier")
    _roles(source_train, target_train, target_valid, optional=(2,))
    if (not model._adaptation or any(p.requires_grad != name.startswith("target_trunk.")
                                     for name, p in model.named_parameters())):
        raise ValueError("AdaptMIL bridge must run before phase2; only target parameters train")
    device = torch.device(device)
    frozen = two_phase.phase_one_state(model)
    optimizer = _optimizer(model.target_trunk.parameters(), params["lr2"], params["wd2"])
    sigma = _bandwidth(model, source_train, target_train, params["kernelMultiplier"], device)
    queues = two_phase.ClassQueues(source_train, model.classes, generator)
    history, validation = [], []
    keeper = two_phase.LabelFreeBest(label_free)
    for epoch in range(schedule["epochs2"]):
        model.train()
        losses, mmds = [], []
        for indices in two_phase.shuffled_batches(len(target_train), schedule["bagsPerStep"],
                                                  generator):
            source = [source_train[i].windows.to(device) for i in queues.draw(
                schedule["bagsPerStep"] // model.classes)]
            target = [target_train[i].windows.to(device) for i in indices]
            terms = alignment_objective(model, source, target, params["mmdWeight"], sigma)
            optimizer.zero_grad(set_to_none=True)
            terms["objective"].backward()
            optimizer.step()
            model.constrain("target_trunk")
            losses.append(float(terms["objective"].detach()))
            mmds.append(float(terms["mmd"].detach()))
        accuracy, score = math.nan, math.nan
        if target_valid:
            predictions = two_phase.evaluate(model, target_valid, "target", device)
            accuracy = two_phase.balanced_accuracy([b.label for b in target_valid],
                                                   predictions.argmax(1).tolist())
            score = two_phase.label_free_score(predictions)
            validation.append(two_phase.validation_record(epoch, target_valid, predictions))
        history.append(_record(epoch, 2, seed, fold, transfer, loss=sum(losses)/len(losses),
                               mmd=sum(mmds)/len(mmds), sigma=sigma,
                               val_balanced_accuracy=accuracy, label_free_score=score,
                               mmd_weight=params["mmdWeight"]))
        keeper.offer(epoch, score, model)
    model.eval()
    if model_store.state_digest(two_phase.phase_one_state(model)) != model_store.state_digest(frozen):
        raise two_phase.FrozenPartsMoved("AdaptMIL phase2 changed frozen parameters/buffers")
    last = len(history) - 1
    return two_phase.with_label_free({
            "checkpoint": {"target_state": _state(model.target_trunk), "epoch": last,
                           "valid_balanced_accuracy": accuracy, "label_free_score": score,
                           "sigma": sigma, "hyper": {**params, **schedule}},
            "history": history, "best_epoch": last, "epochs": len(history),
            "validation": validation,
            "selection": {"primary": "pooled target-validation balanced accuracy, chosen "
                                     "by the search",
                          "secondary": "pooled H(mean prediction) - mean H(prediction), "
                                       "chosen by the search"}}, keeper)


def train_phase_one(data, rotation, seed, params, schedule, device="cpu"):
    """Native-CE entry point; same call shape as a phase-one search callback."""
    hyper = search.phase_one_hyper(GROUP, params, schedule)
    d = data.rotations[rotation]
    torch.manual_seed(seed)
    model = AdaptMILModel(**{f"{k}_channels": v for k, v in data.channels().items()}).to(device)
    result = train_source_phase(model, d["source_train"], d["source_valid"], hyper,
                                torch.Generator().manual_seed(seed), device, seed=seed,
                                fold=rotation, transfer=data.transfer.name)
    return {"group": GROUP, "model": model, "result": result, "hyper": hyper,
            "channels": data.channels()}


def train_phase_two(data, rotation, seed, record, params, schedule, device="cpu", *,
                    label_free=False):
    """Existing search.register_phase_two callback; one fresh model per trial/rotation."""
    state, meta = record
    expected = {"phase": 1, "name": GROUP, "transfer": data.transfer.name,
                "rotation": rotation, "seed": search.PHASE_ONE_SEED,
                "partition_sha256": data.partition_sha256}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("AdaptMIL phase1 record does not match transfer/rotation/seed/partition")
    # Code identity is the store's: `model_store.load` refuses a record whose
    # `code_version` (which covers comparators/**) differs; the campaign's shared
    # `search.save_phase_one` writes no comparator-specific digest.
    torch.manual_seed(seed)
    model = AdaptMILModel(**{f"{k}_channels": v for k, v in data.channels().items()}).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("target_trunk.") for k in missing):
        raise ValueError("AdaptMIL phase1 state does not fit the native architecture")
    joint = bridge(model)
    d = data.rotations[rotation]
    result = adapt_target_phase(model, d["source_train"], d["target_train"], d["target_valid"],
                                params, schedule, torch.Generator().manual_seed(seed), device,
                                seed=seed, fold=rotation, transfer=data.transfer.name,
                                label_free=label_free)
    return {"strategy": GROUP, "model": model, "result": result,

            "hyper": {**params, **schedule}, "channels": data.channels(),
            "layer_copy": joint["layer_copy"]}


def save_model(root, data, rotation, seed, phase, trained):
    """Reuse generic model_store; add identity for comparator files it does not hash."""
    checkpoint = trained["result"]["checkpoint"]
    state = checkpoint["state"] if phase == 1 else trained["model"].state_dict()
    # Shared phase-one schemas carry NaNs for unread phase-two knobs. Do not
    # leak those placeholders into the checkpoint's JSON configuration.
    hyper = ({k: trained["hyper"][k] for k in ("lr1", "wd1", *two_phase.SCHEDULE_KEYS)}
             if phase == 1 else trained["hyper"])
    return model_store.save(root, transfer=data.transfer, rotation=rotation, seed=seed,
                            phase=phase, name=GROUP, state=state, hyper=hyper,
                            partition_sha256=data.partition_sha256,
                            channels=trained["channels"], sigma=checkpoint.get("sigma"),
                            extra={"adaptmil_source_digest": source_digest(),
                                   "epoch": checkpoint["epoch"],
                                   "valid_balanced_accuracy": checkpoint["valid_balanced_accuracy"],
                                   "variant": "BALLADEER controlled variant, plan 7.4"})


def load_model(path, *, partition_path=None):
    """Load either phase through the store's partition/code/tensor checks plus our seal."""
    state, meta = model_store.load(path, partition_path=partition_path)
    if meta.get("name") != GROUP or meta.get("adaptmil_source_digest") != source_digest():
        raise model_store.CodeVersionMismatch("missing/foreign AdaptMIL source digest")
    model = AdaptMILModel(meta["channels"]["source"], meta["channels"]["target"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("target_trunk.") for k in missing):
        raise ValueError("stored AdaptMIL state does not fit")
    if meta["phase"] == 2:
        # Do not copy source over the selected target at load time.
        model.requires_grad_(False)
        model.target_trunk.requires_grad_(True)
        model._adaptation = True
    return model.eval(), meta
