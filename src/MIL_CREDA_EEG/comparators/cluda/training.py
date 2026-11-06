"""CLUDA controlled bag adaptation, protocol v56 §§4.5, 7.2–7.4, 7.7.

Official symmetric SupCon, including target-target and source-source pairs.
L81 supersedes the earlier paper-Eq. (3) ruling; see REPLICATION.md.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from MIL_CREDA_EEG import eegnet, model_store, search, two_phase
from .losses import cross_domain_contrastive, source_source_contrastive, contrastive_objective
from .model import CLUDAModel

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v57.md",
    "sections": ["4.5", "7.2", "7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": [
        "cluda_phase_two_freezes_common_state_and_trains_target_and_projector",
        "cluda_source_source_trains_only_projector_and_coefficient_ratio",
        "cluda_argmax_has_no_threshold_and_target_training_labels_are_unread",
        "cluda_last_epoch_validation_and_refit",
        "cluda_rejects_foreign_records_invalid_parameters_and_leaking_roles",
        "cluda_separate_projector_batches_skip_singleton_target_and_require_source_classes",
        "cluda_search_registration_is_explicit",
        "cluda_saved_state_round_trips_and_checks_source_digest",
    ],
}

NAME = "CLUDA"
DEFAULT_PARAMS = {"lr2": .001, "wd2": 0., "contrastiveWeight": 1., "temperature": .07}
#: v56 §7.7 own axis, defined once in `search.OWN_AXES` like CLUST's and CASUAL's.
TEMPERATURE = search.CONTRASTIVE_TEMPERATURE


def source_digest():
    """Comparator identity supplements the shared store's core-code digest."""
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode()+b"\0"+path.read_bytes()+b"\0")
    return h.hexdigest()


def pseudo_labels(logits):
    """Paper Algorithm 1 / official models.py:224-228: argmax, no confidence gate."""
    return logits.detach().argmax(dim=1)


def _params(params, schedule):
    if set(params) != set(DEFAULT_PARAMS):
        raise ValueError(f"CLUDA parameters must be exactly {list(DEFAULT_PARAMS)}")
    p = {k: float(v) for k, v in params.items()}
    if not all(math.isfinite(v) for v in p.values()):
        raise ValueError("CLUDA parameters must be finite")
    if p["lr2"] <= 0 or p["wd2"] < 0 or p["contrastiveWeight"] < 0 or p["temperature"] <= 0:
        raise ValueError("invalid CLUDA learning rate, decay, weight or temperature")
    schedule = search.check_schedule(schedule)
    nan = float("nan")
    two_phase.check_hyper({**{k: nan for k in two_phase.PHASE_ONE_KEYS},
                           "lr2": p["lr2"], "wd2": p["wd2"], "tauLocal": nan, **schedule})
    return {**p, **schedule, "cluda_source_sha256": source_digest()}


def _represent(model, bags, domain, device):
    return model.represent(model.bag_pairs(model.embed([b.windows.to(device) for b in bags], domain)))


def adaptation_step(model, source, target, params, device):
    """Official main.py:91 objective; frozen source, trainable projector.

    Project target separately, then source, as in official models.py:225,232.
    Singleton source domains share one source-only BN forward (declared deviation).
    No source labels become instance labels; target training labels are unread.
    """
    if len(source) < 2 or len(target) < 2:
        raise ValueError("CLUDA adaptation requires at least two bags on each side")
    labels = torch.tensor([b.label for b in source], device=device)
    with torch.no_grad():
        zs = _represent(model, source, "source", device)
        ce = F.cross_entropy(model.head.linear(zs), labels)
    zt = _represent(model, target, "target", device)
    pseudo = pseudo_labels(model.head.linear(zt))
    target_embeddings = model.projector(zt)
    source_embeddings = model.projector(zs)
    domains = [b.subject for b in source]
    contrastive, missing = cross_domain_contrastive(
        source_embeddings, target_embeddings, labels, pseudo, params["temperature"], domains)
    pair_diagnostics = {}
    source_source, missing_source = source_source_contrastive(
        source_embeddings, labels, domains, diagnostics=pair_diagnostics)
    sizes = {subject: domains.count(subject) for subject in dict.fromkeys(domains)}
    n_domains = len(sizes)
    return {"loss": contrastive_objective(ce, contrastive, source_source, params["contrastiveWeight"]),
            "source_ce": ce, "contrastive": contrastive, "source_source": source_source,
            "anchors_without_positive": missing, "anchors": len(source) + n_domains*len(target),
            "source_anchors_without_positive": missing_source,
            "source_anchors": (n_domains-1)*len(source),
            "source_domains": n_domains,
            "source_pairs": pair_diagnostics["pairs"],
            "source_zero_loss_pairs": pair_diagnostics["zero_loss_pairs"],
            "target_pseudo_classes": int(pseudo.unique().numel())}


def train_phase_two(data, rotation, seed, record, params, schedule, device="cpu", *,
                    label_free=False):
    """Shared phase-2 trainer signature. Uses the single PHASE_ONE_SEED checkpoint.

    Exactly epochs2 epochs, last state retained; every inner validation epoch is
    recorded for pooled selection by shared search. Empty validation is refit.
    """
    hyper = _params(params, schedule)
    state, meta = record
    channels = data.channels()
    expected = {"phase": 1, "name": "common", "transfer": data.transfer.name,
                "rotation": rotation, "seed": search.PHASE_ONE_SEED,
                "partition_sha256": data.partition_sha256, "channels": channels}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("CLUDA needs this transfer/rotation/PHASE_ONE_SEED/partition common checkpoint")
    d = data.rotations[rotation]
    for key in ("source_train", "target_train"):
        if not d[key]:
            raise ValueError(f"empty {key}")
    for side in ("source", "target"):
        if {b.subject for b in d[f"{side}_train"]} & {b.subject for b in d[f"{side}_valid"]}:
            raise ValueError("training/validation subjects overlap")
    source_subjects = {b.subject for role in ("source_train", "source_valid") for b in d[role]}
    target_subjects = {b.subject for role in ("target_train", "target_valid") for b in d[role]}
    if source_subjects & target_subjects:
        raise ValueError("source/target subjects overlap")
    if len(d["target_train"]) < 2:
        raise ValueError("CLUDA target training requires at least two bags")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    # Shared helper validates both source classes before constructing any model.
    queues = two_phase.ClassQueues(d["source_train"], len(two_phase.CLASSES), generator)
    device = torch.device(device)
    model = CLUDAModel(channels["source"], channels["target"]).to(device)
    # Reject auxiliary state in phase one rather than inheriting a trained projector.
    if any(k.startswith("projector.") for k in state):
        raise ValueError("common phase one cannot contain a CLUDA projector")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith(("target_trunk.", "projector.")) for k in missing):
        raise ValueError(f"incompatible common checkpoint: {missing}, {unexpected}")
    joint = two_phase.bridge(model)
    frozen = {k: v.clone() for k, v in model.state_dict().items()
              if not k.startswith(("target_trunk.", "projector."))}
    optimizer = torch.optim.Adam([*model.target_trunk.parameters(), *model.projector.parameters()],
                                 lr=hyper["lr2"], weight_decay=hyper["wd2"], betas=(.5, .999))
    history, validation = [], []
    keeper = two_phase.LabelFreeBest(label_free)
    for epoch in range(schedule["epochs2"]):
        two_phase.set_phase_two_modes(model)
        model.projector.train()
        batches = two_phase.shuffled_batches(len(d["target_train"]), schedule["bagsPerStep"], generator)
        sums = dict.fromkeys(("loss", "source_ce", "contrastive", "source_source"), 0.)
        absent, anchors, classes = [], [], []
        source_absent, source_anchors, source_domains = [], [], []
        source_pairs, source_zero_pairs = [], []
        skipped, contributing = 0, 0
        for batch in batches:
            if len(batch) < 2:
                # Retained L78 batching deviation: target projector BN cannot train one bag.
                skipped += 1
                continue
            source = [d["source_train"][j] for j in queues.draw(schedule["bagsPerStep"] // model.classes)]
            contributing += len(batch)
            terms = adaptation_step(model, source, [d["target_train"][j] for j in batch], hyper, device)
            if not torch.isfinite(terms["loss"]):
                raise FloatingPointError("non-finite CLUDA objective")
            optimizer.zero_grad(set_to_none=True)
            terms["loss"].backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all()
                   for group in optimizer.param_groups for p in group["params"]):
                raise FloatingPointError("missing/non-finite CLUDA gradient")
            optimizer.step()
            eegnet.apply_max_norm(model.target_trunk)
            for k in sums:
                sums[k] += float(terms[k].detach())
            absent.append(terms["anchors_without_positive"])
            anchors.append(terms["anchors"])
            classes.append(terms["target_pseudo_classes"])
            source_absent.append(terms["source_anchors_without_positive"])
            source_anchors.append(terms["source_anchors"])
            source_domains.append(terms["source_domains"])
            source_pairs.append(terms["source_pairs"])
            source_zero_pairs.append(terms["source_zero_loss_pairs"])
        accuracy, free = math.nan, math.nan
        if d["target_valid"]:
            G = two_phase.evaluate(model, d["target_valid"], "target", device)
            accuracy = two_phase.balanced_accuracy([b.label for b in d["target_valid"]], G.argmax(1).tolist())
            free = two_phase.label_free_score(G)
            validation.append(two_phase.validation_record(epoch, d["target_valid"], G))
        history.append({"epoch": epoch, "seed": seed, "fold": rotation, "model": "CLUDA:phase2",
                        "transfer": data.transfer.name,
                        **{k: v/len(absent) for k, v in sums.items()},
                        "source_zero_loss_pair_fraction": sum(source_zero_pairs)/sum(source_pairs),
                        "step_source_pairs": source_pairs,
                        "step_source_zero_loss_pairs": source_zero_pairs,
                        "source_anchors_without_positive": sum(source_absent),
                        "source_anchors": sum(source_anchors),
                        "step_source_anchors_without_positive": source_absent,
                        "step_source_anchors": source_anchors,
                        "step_source_domains": source_domains,
                        "skipped_singleton_target_steps": skipped,
                        "target_bags_contributing": contributing,
                        "anchors_without_positive": sum(absent), "anchors": sum(anchors),
                        "step_anchors_without_positive": absent, "step_anchors": anchors,
                        "step_target_pseudo_classes": classes,
                        "val_balanced_accuracy": accuracy, "label_free_score": free})
        keeper.offer(epoch, free, model)
        if any(not torch.equal(model.state_dict()[k], v) for k, v in frozen.items()):
            raise two_phase.FrozenPartsMoved("CLUDA changed a frozen common tensor")
    two_phase.set_phase_two_modes(model)
    model.projector.eval()
    last = len(history)-1
    checkpoint = {"epoch": last, "valid_balanced_accuracy": accuracy, "label_free_score": free,
                  "hyper": hyper, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    return {"strategy": NAME, "model": model, "channels": channels, "hyper": hyper,
            "layer_copy": joint["layer_copy"], "result": two_phase.with_label_free({
                "checkpoint": checkpoint, "best_epoch": last, "epochs": len(history),
                "history": history, "validation": validation,
                "selection": {"primary": "pooled target-validation balanced accuracy, chosen by the search",
                              "secondary": "pooled entropy of mean minus mean entropy, chosen by the search"}}, keeper)}


def register():
    """Explicit opt-in; parent must admit CLUDA and TEMPERATURE in shared registries."""
    search.register_phase_two(NAME, {"contrastiveWeight": search.LOSS_WEIGHT,
                                     "temperature": TEMPERATURE}, train_phase_two)


def load_phase_two(path, *, partition_path=None, allow_code_mismatch=False):
    """Reuse shared store verification; restore auxiliaries without re-bridging target."""
    state, meta = model_store.load(path, partition_path=partition_path,
                                   allow_code_mismatch=allow_code_mismatch)
    if meta["phase"] != 2 or meta["name"] != NAME:
        raise ValueError("expected a CLUDA phase-two checkpoint")
    if meta["hyper"].get("cluda_source_sha256") != source_digest():
        raise ValueError("CLUDA source digest differs from checkpoint")
    model = CLUDAModel(meta["channels"]["source"], meta["channels"]["target"])
    model.load_state_dict(state)
    for p in [*model.source_trunk.parameters(), *model.attention_parameters(), *model.head.parameters()]:
        p.requires_grad_(False)
    model.eval()
    return model, meta
