"""The proof-of-concept hypothesis battery on the toyset: `rollo`, `mlp` (BatchNorm).

`MIL-CREDA/Notebooks/PoC_Hypotheses_Rollo.ipynb` reads everything it runs from
here. Nothing about the method is implemented in this file: the bags are
`bags.build`'s, the arms are `wiring.Arm`/`wiring.MechanismArm`, every phase
helper is `toy_run`'s, and the two training loops are the tutorial's own
(`MIL-CREDA/Notebooks/Toyset.ipynb`, cells "Fase 1" and "Fase 2"), lifted into
functions so that a seed can run in a worker process. What this module adds is:

- **the arms the hypotheses need and the toyset does not declare**: `TG` (global
  term only), `TL` (local term only) and `TGL-NC` (TGL with gamma = 0, the "own
  mechanism without consensus" of protocol section 6.3), plus the four
  reference mechanisms of `wiring.MECHANISMS`;
- **a declared contamination on the toy path**: `bags.build`'s own seam
  (`bags._contaminate`, called with `noise=rate`) with `config.NOISE_PAIRING`
  bent to a three-class donor map for the duration of the draw, because the
  declared pairing is the digit benchmark's ten-class involution;
- **the measurements**: attention mass on contaminated against clean
  instances (K1), the kernel-space partner separation (K2), and the paired
  statistics every verdict is read from.

No `__provenance__`: like the rest of this package, it implements no equation.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import dataclasses
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from MIL_CREDA.attention import bag_embedding
from MIL_CREDA.bag_kernel import bag_kernel, bag_kernel_matrix
from MIL_CREDA.local_term import total_correspondence
from MIL_CREDA_Benchmark import bags, config, harness, toy_bags, toy_run, wiring

#: The one scenario this battery runs (L96: "rollo + mlp BatchNorm").
FAMILY = "rollo"
BACKBONE = toy_run.NORMALIZATIONS["batchnorm"]

#: The full toyset programme: 100/100 epochs, patience 30, 4050 points.
SCALE = toy_run.FULL

#: The hyperparameters, fixed BEFORE this battery and never searched by it.
#: They are the archived search's choice for `rollo`
#: (`MIL-CREDA/Results/Toyset/full/barrido.json`, `hp1`/`hp2`), made on search
#: seeds 100-102 -- disjoint from every seed this battery draws. The archive's
#: results are bound to r34 and are not evidence here; its hyperparameters are
#: only the configuration, the protocol's "hyperparameters chosen for full
#: MIL-CREDA", which every variant then shares.
ARCHIVED_ROLLO_HYPER = {
    "lr1": 0.015290622066240019,
    "weightDecay1": 6.277218548234598e-07,
    "attentionGamma": 0.8633557574655063,
    "attentionTemperature": 0.42588932435340904,
    "dim": 64,
    "tauLocal": 0.27466369735257745,
    "lr2": 0.0026420508173972492,
    "weightDecay2": 1.5227262701506252e-06,
}

#: The archived search's `sigmaMultiplier`, applied by the caller as
#: `toy_run.frozen_source_sigma` asks: to the phase-1 bootstrap width and to
#: the frozen phase-2 width alike.
SIGMA_MULTIPLIER = 0.5

#: The ablation arms: TGL itself and one flag off each.
POC_ARMS = {
    "TGL": toy_run.TOY_ARMS_BY_ID["TGL"],
    "TG": {**toy_run.TOY_ARMS_BY_ID["TGL"], "id": "TG",
           "name": "Toy-MIL-CREDA (global only)",
           "label": "global term only, local term off", "local": False},
    "TL": {**toy_run.TOY_ARMS_BY_ID["TGL"], "id": "TL",
           "name": "Toy-MIL-CREDA (local only)",
           "label": "local term only, global term off", "global": False},
}

#: TGL with gamma = 0 and everything else identical (protocol section 6.3).
NO_CONSENSUS = "TGL-NC"

#: The attention references of K1, reported without criterion.
REFERENCE_MECHANISMS = ("abmil-published", "abmil-gated", "max", "mean")

#: The donor class of each label on the toy path: a cycle, because three
#: classes admit no fixed-point-free involution. Every contaminated instance of
#: a bag labelled `c` is a point of class `NOISE_PAIRING_TOY[c]`.
NOISE_PAIRING_TOY = [1, 2, 0]

#: The subject-shift run's per-pair target shift (`toy_bags.construir`'s
#: `escala_por_par`); the first run used none. See
#: `toy_bags.ESCALA_POR_PAR_DECLARADA` for the scale and its rationale.
SUBJECT_SHIFT_SCALE = toy_bags.ESCALA_POR_PAR_DECLARADA


@dataclasses.dataclass(frozen=True)
class Conditions:
    """The data and step conditions a battery runs under; every field off by
    default, which is runs 1 and 2 exactly (run 2 sets `per_pair_shift`).

    All of them act on the shared material or on the shared phase-2 step, so
    every arm, floor and reference of a seed sees the same ones.
    """

    #: Run 2's independent per-pair target shift (`toy_bags.desplazar_por_par`).
    per_pair_shift: float = 0.0
    #: The Sigma-preserving rotation of `perturbar`, in degrees; `None` is its
    #: declared 10.
    rotation_degrees: float | None = None
    #: The smooth subject field along the roll, as a fraction of a band's width
    #: in `t` (`toy_bags.campo_de_sujeto`); 0 is off.
    subject_field: float = 0.0
    #: Source bags per class in every phase-2 step (with that many target bags
    #: per class): 1 is the toy's draw as it was.
    source_bags_per_class: int = 1
    #: What a contaminated slot holds: `donor` (`bags.build`'s own seam) or
    #: `artifact` (the clean instance plus off-manifold noise, same slots).
    contamination: str = "donor"

    def draw_options(self) -> dict:
        return {"per_pair_shift": self.per_pair_shift,
                "rotation_degrees": self.rotation_degrees,
                "subject_field": self.subject_field}


OFF = Conditions()

#: Run 3's fraction of a band's width the subject field may slide a target
#: point along the roll: below the half-spacing where the nearest same-class
#: source band would stop being the partner (analysis 0.2, condition 4).
SUBJECT_FIELD_FRACTION = 0.45

#: Run 3 (L98 and the parent's fixed values): 25-degree Sigma-preserving
#: rotation, the subject field, and k = 4 source bags per class per step.
RUN3 = Conditions(rotation_degrees=25.0, subject_field=SUBJECT_FIELD_FRACTION,
                  source_bags_per_class=4)

#: Run 3's scale: the test role tripled (6 -> 18 bags per class) at the same
#: density (50 points per bag), train and valid kept at 16 and 5.
CONDITIONS_SCALE = dataclasses.replace(
    toy_run.FULL, name="conditions", bags_per_class=39,
    frac_train=16 / 39, frac_valid=5 / 39, points_per_domain=39 * 3 * 50)

#: The artifact noise's per-coordinate standard deviation, in units of the
#: domain's own clean coordinate spread `esc` (the unit `perturbar` uses): a
#: bad-contact window is of the order of the signal's whole range, so one
#: `esc` puts an artifact off the roll's sheet while keeping it in the
#: domain's box.
ARTIFACT_SCALE = 1.0

#: The artifact noise's stream offset, its own so nothing else moves.
ARTIFACT_STREAM = 5000


def poc_hyper() -> dict:
    """`toy_run.declared_hyper` with the archived `rollo` values over it."""
    return {**toy_run.declared_hyper(SCALE), **ARCHIVED_ROLLO_HYPER}


def spec_for(arm: str) -> dict:
    """The `wiring.Arm` spec an arm trains under."""
    return POC_ARMS["TGL"] if arm == NO_CONSENSUS else POC_ARMS[arm]


def hyper_for(arm: str, hyper: dict) -> dict:
    """The hyper an arm trains under: the no-consensus arm sets gamma to zero."""
    return {**hyper, "attentionGamma": 0.0} if arm == NO_CONSENSUS else dict(hyper)


# ---------------------------------------------------------------- material

@contextlib.contextmanager
def toy_noise_pairing():
    """`config.NOISE_PAIRING` as the three-class donor map, restored on exit."""
    declared = config.NOISE_PAIRING
    config.NOISE_PAIRING = list(NOISE_PAIRING_TOY)
    try:
        yield
    finally:
        config.NOISE_PAIRING = declared


def contamination_mask(bagset: bags.BagSet) -> torch.Tensor:
    """(bags, instances) True where `bags._contaminate` replaced the instance."""
    mask = torch.zeros(tuple(bagset.members.shape), dtype=torch.bool)
    for record in bagset.manifest["contamination"]["bags"]:
        mask[record["bag"], record["slots"]] = True
    return mask


def contaminated_material(familia, seed: int, scale, backbone: str,
                          rate: float) -> tuple[dict, dict]:
    """`toy_run.material_for` with `bags.build(noise=rate)`, and the masks.

    The contamination reaches `config.NOISE_ROLES` (train and eval) of both
    domains, with the bag label unchanged; the clean draw is bit-identical to
    `material_for` because the seam draws its replacements from a stream of
    its own.
    """
    with toy_run.material_bound(familia, scale, backbone), toy_noise_pairing():
        root = Path("/nonexistent-toy-root")
        codes = toy_run.FAMILY_CODES[familia.nombre]
        material = {"source": bags.build(codes[0], root, seed, noise=rate),
                    "target": bags.build(codes[1], root, seed, noise=rate)}
    return material, {side: contamination_mask(b) for side, b in material.items()}


def artifact_material(familia, seed: int, scale, backbone: str, rate: float,
                      noise_scale: float = ARTIFACT_SCALE) -> tuple[dict, dict]:
    """The clean material with the SAME slots `bags._contaminate` picks at
    `rate` replaced by artifacts: the clean instance plus isotropic noise of
    standard deviation `noise_scale * esc`, `esc` the domain's clean spread.

    Both domains, the train and eval roles (`config.NOISE_ROLES`), labels kept.
    """
    clean = toy_run.material_for(familia, seed, scale, backbone)
    if rate == 0:
        return clean, {side: contamination_mask(b) for side, b in clean.items()}
    donor, masks = contaminated_material(familia, seed, scale, backbone, rate)
    out = {}
    for side, X in (("source", familia.Xs), ("target", familia.Xt)):
        base = clean[side]
        esc = float(X.std(0).mean())
        code = toy_run.FAMILY_CODES[familia.nombre][0 if side == "source" else 1]
        rng = np.random.default_rng((seed * 1000 + ord(code)) * 7919 + ARTIFACT_STREAM)
        images = base.images.clone()
        index = base.members[masks[side]]
        noise = rng.normal(0.0, noise_scale * esc, (len(index), images.shape[1]))
        images[index] = images[index] + torch.as_tensor(noise, dtype=images.dtype)
        record = donor[side].manifest["contamination"]
        manifest = {**base.manifest, "contamination": {
            "mode": "artifact", "rate": rate, "noiseSd": noise_scale * esc,
            "instancesPerBag": record["instancesPerBag"], "roles": record["roles"],
            "bags": [{"bag": b["bag"], "role": b["role"], "label": b["label"],
                      "slots": b["slots"]} for b in record["bags"]]}}
        out[side] = dataclasses.replace(base, images=images, manifest=manifest)
    return out, masks


def material_under(familia, seed: int, scale, conditions: Conditions,
                   rate: float) -> tuple[dict, dict]:
    """The material of one seed under `conditions` at contamination `rate`."""
    if conditions.contamination not in ("donor", "artifact"):
        raise ValueError(f"unknown contamination mode {conditions.contamination!r}")
    if rate == 0:
        material = toy_run.material_for(familia, seed, scale, BACKBONE)
        return material, {side: contamination_mask(b) for side, b in material.items()}
    if conditions.contamination == "donor":
        return contaminated_material(familia, seed, scale, BACKBONE, rate)
    return artifact_material(familia, seed, scale, BACKBONE, rate)


# -------------------------------------------------------------------- arms

def build_poc_arm(arm: str, material: dict, scale, hyper: dict,
                  device: torch.device, seed: int) -> wiring.Arm:
    """One arm of this battery, seeded exactly as `toy_run.build_arm` seeds.

    `TGL` IS `toy_run.build_arm`. The other arms reach `wiring.Arm` (or
    `wiring.build_mechanism`) through the same seeding and the same pools, so
    every arm built with one seed draws the same shared parameters.
    """
    if arm == "TGL":
        return toy_run.build_arm("TGL", material, scale, hyper, device, seed)
    toy_run.refuse_retired_hyper(hyper)
    source, target = material["source"], material["target"]
    pools = (harness.pool_of(source, source.train_idx, device),
             harness.pool_of(target, target.train_idx, device))
    torch.manual_seed(seed)
    if arm in REFERENCE_MECHANISMS:
        return wiring.build_mechanism(arm, scale.classes, *pools,
                                      hyper=hyper).to(device)
    return wiring.Arm(spec_for(arm), scale.classes, *pools,
                      hyper=hyper_for(arm, hyper)).to(device)


# ------------------------------------------- the tutorial's two phases, lifted

def source_parameters(model: wiring.Arm) -> list:
    return [p for n, p in model.named_parameters()
            if not n.startswith("target_encoder.")]


def batch(dataset, positions, device: torch.device):
    items = [dataset[int(i)] for i in positions]
    x = torch.stack([item[0] for item in items]).to(device)
    y = torch.tensor([item[1] for item in items], device=device)
    return x, y


def phase_one_model(arm: str, material: dict, scale, hyper: dict,
                    device: torch.device, seed: int):
    """The phase-1 model at the bootstrap width, measured on the source."""
    probe = build_poc_arm(arm, material, scale, hyper, device, seed)
    within = toy_run.embedding_scales(probe, material["source"],
                                      material["source"].train_idx,
                                      device)["withinBag"]
    sigma = SIGMA_MULTIPLIER * within
    return build_poc_arm(arm, material, scale, {**hyper, "kernelSigma": sigma},
                         device, seed), sigma


def train_phase_one(model, source_train, source_valid, scale, hyper: dict,
                    generator: torch.Generator, device: torch.device) -> dict:
    """Tutorial phase 1: Eq. (21) on the source, best validation epoch, frozen."""
    parameters = source_parameters(model)
    optimizer = torch.optim.Adam(parameters, lr=hyper["lr1"] or scale.lr_1,
                                 weight_decay=hyper["weightDecay1"])
    steps = math.ceil(len(source_train) / config.BAGS_PER_STEP)
    best = {"accuracy": -1.0, "epoch": 0, "state": None}
    epochs = 0
    for epoch in range(1, scale.epochs_1 + 1):
        epochs = epoch
        model.train()
        for positions in harness.balanced_batches(source_train.targets, steps,
                                                  generator):
            x, y = batch(source_train, positions, device)
            loss = toy_run.supervised_loss(model, x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.project_constraints()
        accuracy = harness.accuracy(model, source_valid, device)
        if accuracy > best["accuracy"]:
            best = {"accuracy": accuracy, "epoch": epoch,
                    "state": toy_run.shared_phase_one_state(model)}
        if epoch - best["epoch"] >= scale.patience_1:
            break
    model.load_state_dict(best["state"], strict=False)
    for parameter in parameters:
        parameter.requires_grad_(False)
    model.eval()
    return {"epochs": epochs, "bestEpoch": best["epoch"],
            "bestValidAccuracy": best["accuracy"]}


def prepare_phase_two(model_source, material: dict, arm: str, scale,
                      hyper: dict, device: torch.device, seed: int):
    """Frozen width, phase-1 weights loaded, source frozen, theta_t cloned."""
    sigma = SIGMA_MULTIPLIER * toy_run.frozen_source_sigma(
        model_source, material["source"], material["source"].train_idx,
        device)["sigma"]
    model = build_poc_arm(arm, material, scale, {**hyper, "kernelSigma": sigma},
                          device, seed)
    model.load_state_dict(toy_run.shared_phase_one_state(model_source), strict=False)
    for parameter in source_parameters(model):
        parameter.requires_grad_(False)
    toy_run.copy_source_to_target(model)
    return model, sigma


@contextlib.contextmanager
def _phase_two_step(bags_per_step: int):
    """`config.BAGS_PER_STEP` (the target draw per step) for phase 2 only."""
    declared = config.BAGS_PER_STEP
    config.BAGS_PER_STEP = bags_per_step
    try:
        yield
    finally:
        config.BAGS_PER_STEP = declared


def train_phase_two(model, source_train, scale, hyper: dict,
                    generator: torch.Generator, device: torch.device,
                    source_bags_per_class: int = 1) -> dict:
    """Tutorial phase 2: only theta_t moves; the epoch is chosen by `cef`.

    `source_bags_per_class = k` draws k source bags of every class and k * C
    target bags per step, for the same number of EPOCHS: one pass is then
    `per-class count / k` steps, so each domain's bags are seen as often as at
    k = 1 (16 steps x 3 bags = 4 steps x 12 bags per pass). At k = 1 this is
    the tutorial's loop exactly.
    """
    k = source_bags_per_class
    with _phase_two_step(k * scale.classes if k > 1 else config.BAGS_PER_STEP):
        return _train_phase_two(model, source_train, scale, hyper, generator,
                                device, k)


def _train_phase_two(model, source_train, scale, hyper, generator, device, k):
    toy_run.set_phase_two_modes(model)
    optimizer = torch.optim.Adam(model.target_encoder.parameters(),
                                 lr=hyper["lr2"] or scale.lr_2,
                                 weight_decay=hyper["weightDecay2"])
    coefficient = toy_run.FIXED_COEFFICIENT
    per_class = toy_run.steps_of_one_epoch(source_train.targets)
    if per_class % k:
        raise ValueError(f"{per_class} source bags per class do not divide into "
                         f"steps of {k}")
    steps_per_epoch = per_class // k
    best = {"cef": -math.inf, "epoch": -1, "state": None}
    terms = {"global": [], "local": []}
    for step, positions in enumerate(harness.balanced_batches(
            source_train.targets, scale.epochs_2 * steps_per_epoch, generator,
            per_class=k), start=1):
        x, y = batch(source_train, positions, device)
        optimizer.zero_grad()
        record = model.training_step(x, y, coefficient, generator,
                                     ramp_local=coefficient)
        objective = toy_run.phase_two_objective(record)
        objective.backward()
        optimizer.step()
        terms["global"].append(record["contributionGlobal"])
        terms["local"].append(record["contributionLocal"])
        if step % steps_per_epoch:
            continue
        epoch = step // steps_per_epoch
        cef = toy_run.effective_classes(model, device)
        if epoch >= toy_run.MIN_EPOCHS_2 and cef > best["cef"]:
            best = {"cef": cef, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in
                              model.target_encoder.state_dict().items()}}
    model.target_encoder.load_state_dict(best["state"])
    toy_run.set_phase_two_modes(model)
    return {"steps": len(terms["global"]), "stepsPerEpoch": steps_per_epoch,
            "bestEpoch": best["epoch"], "bestCef": best["cef"],
            "globalRange": [min(terms["global"]), max(terms["global"])],
            "localRange": [min(terms["local"]), max(terms["local"])]}


def target_test_accuracy(model, material: dict, device: torch.device) -> float:
    _, _, target_test = bags.roles(material["target"])
    accuracy = harness.target_accuracy(model, target_test, device)
    toy_run.set_phase_two_modes(model)
    return accuracy


def adabn_control_accuracy(model, material: dict, device: torch.device) -> float:
    """The AdaBN reading (information only): theta_t = theta_s with its
    BatchNorm statistics re-estimated on the unlabelled target train role.

    Read on a copy of a model fresh out of `prepare_phase_two`, so nothing the
    battery trains is touched. The statistics are the cumulative average over
    every target train instance (`momentum=None`), with no gradient.
    """
    probe = copy.deepcopy(model)
    norms = [m for m in probe.target_encoder.modules()
             if isinstance(m, torch.nn.BatchNorm1d)]
    if not norms:
        return float("nan")
    for norm in norms:
        norm.reset_running_stats()
        norm.momentum = None
    target = material["target"]
    probe.eval()
    probe.target_encoder.train()
    with torch.no_grad():
        for start in range(0, len(target.train_idx), config.BAGS_PER_STEP):
            chunk = target.train_idx[start:start + config.BAGS_PER_STEP]
            probe.target_embeddings(target.images[target.members[chunk]].to(device))
    _, _, target_test = bags.roles(target)
    return harness.target_accuracy(probe, target_test, device)


# ------------------------------------------------------------ measurements

@torch.no_grad()
def attention_weights(model, bagset: bags.BagSet, positions: torch.Tensor,
                      domain: str, device: torch.device) -> torch.Tensor:
    """(bags, instances) in-bag weights each bag receives, in `eval`, through
    the encoder of `domain` and the arm's own `weights_for`."""
    model.eval()
    embed = model.instance_embeddings if domain == "source" else model.target_embeddings
    rows = []
    for start in range(0, len(positions), config.BAGS_PER_STEP):
        chunk = positions[start:start + config.BAGS_PER_STEP]
        embeddings = embed(bagset.images[bagset.members[chunk]].to(device))
        rows.extend(w for _, w in model.bags_of(embeddings, model.sigma))
    toy_run.set_phase_two_modes(model)
    return torch.stack(rows).cpu()


def attention_contrast(weights: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Per contaminated bag: m * (mean clean weight - mean contaminated weight).

    In units of the uniform weight `1/m`, so uniform pooling reads 0 and a
    positive value means the contaminated instances receive LESS weight.
    Bags with no contaminated (or no clean) instance are left out.
    """
    weights, masks = weights.double(), masks.bool()
    keep = masks.any(dim=1) & (~masks).any(dim=1)
    weights, masks = weights[keep], masks[keep]
    m = weights.shape[1]
    contaminated = (weights * masks).sum(dim=1) / masks.sum(dim=1)
    clean = (weights * ~masks).sum(dim=1) / (~masks).sum(dim=1)
    return m * (clean - contaminated)


@torch.no_grad()
def _kernel_bags(model, bagset, positions, domain, device):
    embed = model.instance_embeddings if domain == "source" else model.target_embeddings
    pairs, Z = [], []
    for start in range(0, len(positions), config.BAGS_PER_STEP):
        chunk = positions[start:start + config.BAGS_PER_STEP]
        embeddings = embed(bagset.images[bagset.members[chunk]].to(device))
        bags_ = model.bags_of(embeddings, model.sigma)
        pairs.extend(model._kernel_pairs(bags_))
        Z.extend(bag_embedding(H, w) for H, w in bags_)
    return pairs, torch.stack(Z)


@torch.no_grad()
def bag_distances(model, bagset_a, positions_a, domain_a, bagset_b,
                  positions_b, domain_b, device: torch.device) -> torch.Tensor:
    """(a, b) distances in H_I between bags, under the bag kernel of Eq. (18)
    at the model's own frozen width: ||Psi_a - Psi_b||."""
    model.eval()
    pairs_a, _ = _kernel_bags(model, bagset_a, positions_a, domain_a, device)
    pairs_b, _ = _kernel_bags(model, bagset_b, positions_b, domain_b, device)
    sigma = model.sigma
    cross = bag_kernel_matrix(pairs_a, pairs_b, sigma)
    self_a = torch.stack([bag_kernel(H, w, H, w, sigma) for H, w in pairs_a])
    self_b = torch.stack([bag_kernel(H, w, H, w, sigma) for H, w in pairs_b])
    squared = self_a[:, None] + self_b[None, :] - 2.0 * cross
    toy_run.set_phase_two_modes(model)
    return squared.clamp_min(0.0).sqrt().cpu()


def partner_separation(D: torch.Tensor, target_labels, target_index,
                       source_labels, source_index,
                       within_class: bool = True) -> torch.Tensor:
    """Per target bag: mean distance to the other source bags minus the
    distance to its own partner (same class, same band index).

    `within_class` restricts "the others" to the partner's own class, so the
    reading is about WHICH bag of the class, not about the class.
    """
    out = []
    for j in range(D.shape[0]):
        same_class = source_labels == target_labels[j]
        own = same_class & (source_index == target_index[j])
        others = (same_class if within_class else torch.ones_like(same_class)) & ~own
        (k,) = torch.nonzero(own, as_tuple=True)
        out.append(D[j, others].mean() - D[j, k[0]])
    return torch.stack(out)


@torch.no_grad()
def partner_reading(model, material: dict, device: torch.device) -> dict:
    """K2 on the test roles: the separation (within class and across), and the
    local correspondence's own conditional mass on the partner (Eq. 35)."""
    source, target = material["source"], material["target"]
    s_pos, t_pos = source.eval_idx, target.eval_idx
    per_class = config.BAGS_PER_CLASS
    s_labels, t_labels = source.labels[s_pos], target.labels[t_pos]
    s_index, t_index = s_pos % per_class, t_pos % per_class
    D = bag_distances(model, target, t_pos, "target", source, s_pos, "source", device)
    within = partner_separation(D, t_labels, t_index, s_labels, s_index, True)
    across = partner_separation(D, t_labels, t_index, s_labels, s_index, False)

    model.eval()
    pairs_s, _ = _kernel_bags(model, source, s_pos, "source", device)
    pairs_t, Z_t = _kernel_bags(model, target, t_pos, "target", device)
    K_st = bag_kernel_matrix(pairs_s, pairs_t, model.sigma)
    G_t = torch.softmax(model.head(Z_t), dim=1)
    masses = []
    for j in range(len(t_pos)):
        pi = total_correspondence(K_st[:, j], s_labels.to(K_st.device), G_t[j],
                                  model.tau_local).cpu()
        same = s_labels == t_labels[j]
        own = same & (s_index == t_index[j])
        total = pi[same].sum()
        masses.append(float(pi[own].sum() / total) if total > 0 else float("nan"))
    toy_run.set_phase_two_modes(model)
    own_d = torch.stack([D[j, (s_labels == t_labels[j]) & (s_index == t_index[j])][0]
                         for j in range(D.shape[0])])
    return {"separationWithin": float(within.mean()),
            "separationAcross": float(across.mean()),
            "ownDistance": float(own_d.mean()),
            "partnerMass": float(np.nanmean(masses)),
            "partnerMassChance": 1.0 / int((s_labels == s_labels[0]).sum()),
            "probeBags": int(len(t_pos))}


# ----------------------------------------------------------- one seed's runs

def _device() -> torch.device:
    return toy_run.dispositivo_del_toyset()


def _two_phases(arm, material, familia, scale, hyper, device, seed,
                phase_one=None, source_bags_per_class: int = 1):
    """Phase 1 (unless handed one), phase 2, evaluation; returns the model."""
    source_train, source_valid, source_test = bags.roles(material["source"])
    if phase_one is None:
        model_source, _ = phase_one_model(arm, material, scale, hyper, device, seed)
        p1 = train_phase_one(model_source, source_train, source_valid, scale,
                             hyper, torch.Generator().manual_seed(seed + 1), device)
        phase_one = (model_source, p1)
    model_source, p1 = phase_one
    model, sigma = prepare_phase_two(model_source, material, arm, scale, hyper,
                                     device, seed)
    floor = target_test_accuracy(model, material, device)
    adabn = adabn_control_accuracy(model, material, device)
    p2 = train_phase_two(model, source_train, scale, hyper,
                         torch.Generator().manual_seed(seed + 2), device,
                         source_bags_per_class)
    record = {"phaseOne": p1, "phaseTwo": p2, "sigmaPhaseTwo": sigma,
              "sourceOnlyTargetAccuracy": floor, "adabnTargetAccuracy": adabn,
              "targetAccuracy": target_test_accuracy(model, material, device),
              "sourceAccuracy": harness.accuracy(model, source_test, device)}
    toy_run.set_phase_two_modes(model)
    return model, record, phase_one


def run_ladder_seed(seed: int, scale=SCALE, per_pair_shift: float = 0.0,
                    conditions: Conditions | None = None, rate: float = 0.0) -> dict:
    """H-task, K2 and K3 for one seed on clean material.

    TGL, TG and TL share ONE phase-1 checkpoint (the protocol: phase 1 has no
    adaptation term) and run phase 2 from it with the same generator seed, so
    they differ only by the flag. TGL-NC trains its own phase 1 (the
    aggregation enters phase 1) from the same initialization.

    `per_pair_shift` adds the per-pair target shift to the draw (off by default).
    `conditions` (run 3) replaces it with a whole `Conditions`; `rate` > 0 runs
    the ladder on material contaminated in `conditions.contamination` mode.
    """
    conditions = conditions or Conditions(per_pair_shift=per_pair_shift)
    k = conditions.source_bags_per_class
    torch.set_num_threads(1)
    start = time.perf_counter()
    device = _device()
    hyper = poc_hyper()
    familia = toy_run.build_family(FAMILY, seed, scale, **conditions.draw_options())
    material, _ = material_under(familia, seed, scale, conditions, rate)
    out = {"seed": seed, "perPairShift": conditions.per_pair_shift, "rate": rate,
           "conditions": dataclasses.asdict(conditions), "arms": {}}
    with toy_run.material_bound(familia, scale, BACKBONE, hyper):
        shared = None
        for arm in ("TGL", "TG", "TL"):
            model, record, shared = _two_phases(arm, material, familia, scale,
                                                hyper, device, seed, shared, k)
            if arm in ("TGL", "TG"):
                record["partner"] = partner_reading(model, material, device)
            out["arms"][arm] = record
        floor_model, _ = prepare_phase_two(shared[0], material, "TGL", scale,
                                           hyper, device, seed)
        out["sourceOnlyPartner"] = partner_reading(floor_model, material, device)
        _, record, _ = _two_phases(NO_CONSENSUS, material, familia, scale, hyper,
                                   device, seed, source_bags_per_class=k)
        out["arms"][NO_CONSENSUS] = record
    out["seconds"] = time.perf_counter() - start
    return out


#: The arms K1 trains at each contamination rate.
K1_ARMS = ("TGL", NO_CONSENSUS) + REFERENCE_MECHANISMS


def run_contamination_seed(seed: int, rate: float, scale=SCALE,
                           arms=K1_ARMS, per_pair_shift: float = 0.0,
                           conditions: Conditions | None = None) -> dict:
    """K1 for one seed and one rate: every arm trains both phases on the
    contaminated material, and its attention is read on the contaminated
    TEST bags of the target (theta_t, primary) and of the source (theta_s)."""
    torch.set_num_threads(1)
    start = time.perf_counter()
    device = _device()
    hyper = poc_hyper()
    conditions = conditions or Conditions(per_pair_shift=per_pair_shift)
    familia = toy_run.build_family(FAMILY, seed, scale, **conditions.draw_options())
    material, masks = material_under(familia, seed, scale, conditions, rate)
    out = {"seed": seed, "rate": rate, "perPairShift": conditions.per_pair_shift,
           "conditions": dataclasses.asdict(conditions), "arms": {}}
    with toy_run.material_bound(familia, scale, BACKBONE, hyper):
        for arm in arms:
            model, record, _ = _two_phases(arm, material, familia, scale, hyper,
                                           device, seed,
                                           source_bags_per_class=conditions.source_bags_per_class)
            for side, domain in (("target", "target"), ("source", "source")):
                bagset = material[side]
                weights = attention_weights(model, bagset, bagset.eval_idx,
                                            domain, device)
                mask = masks[side][bagset.eval_idx]
                contrast = attention_contrast(weights, mask)
                record[f"{side}Contrast"] = float(contrast.mean())
                record[f"{side}ContaminatedMass"] = float(
                    (weights.double() * mask).sum(dim=1).mean())
                record[f"{side}Bags"] = int(contrast.numel())
            out["arms"][arm] = record
    out["contaminatedShare"] = float(masks["target"][material["target"].eval_idx]
                                     .double().mean())
    out["seconds"] = time.perf_counter() - start
    return out


# --------------------------------------------------------------- parallelism

def _worker_init(src: str) -> None:
    if src not in sys.path:
        sys.path.insert(0, src)
    torch.set_num_threads(1)


def _call(job):
    function, args, *rest = job
    return globals()[function](*args, **(rest[0] if rest else {}))


#: A directory of `<job key>.json` outputs `run_many` reuses and fills.
CACHE_ENV = "MIL_CREDA_POC_CACHE"


def _canonical(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"__dataclass__": type(value).__qualname__, **dataclasses.asdict(value)}
    raise TypeError(f"no canonical form for {type(value).__name__}")


def job_key(job: tuple, src: str) -> str:
    """sha256 of the canonical JSON of (function name, args, sorted kwargs, src)."""
    function, args, *rest = job
    kwargs = rest[0] if rest else {}
    text = json.dumps([function, list(args), sorted(kwargs.items()), src],
                      sort_keys=True, separators=(",", ":"), default=_canonical)
    return hashlib.sha256(text.encode()).hexdigest()


def battery_jobs(prereg: dict) -> list[tuple]:
    """Run 3's battery, in the notebook's order: the clean ladder per seed, the
    artifact ladder per seed at the declared rate, then for each rate the donor
    and the artifact contamination runs per seed."""
    scale = CONDITIONS_SCALE
    clean = Conditions(**prereg["conditions"])
    artifact = dataclasses.replace(clean, contamination="artifact")
    rate = prereg["artifact"]["k3Rate"]
    seeds, rates = prereg["seeds"], prereg["contaminationRates"]
    jobs = [("run_ladder_seed", (s,), {"scale": scale, "conditions": clean})
            for s in seeds]
    jobs += [("run_ladder_seed", (s,), {"scale": scale, "conditions": artifact,
                                        "rate": rate}) for s in seeds]
    for r in rates:
        jobs += [("run_contamination_seed", (s, r), {"scale": scale, "conditions": clean})
                 for s in seeds]
        jobs += [("run_contamination_seed", (s, r), {"scale": scale, "conditions": artifact})
                 for s in seeds]
    return jobs


def _store(path: Path, out) -> None:
    partial = path.with_suffix(f".{os.getpid()}.part")
    partial.write_text(json.dumps(out))
    partial.replace(path)


def run_many(jobs: list[tuple], workers: int, src: str) -> list:
    """Run `(function name, args)` or `(function name, args, kwargs)` jobs in
    spawned worker processes.

    Each job is a whole seed (or seed and rate) and depends on nothing but its
    arguments, so the results are the same whatever the worker count.

    When `MIL_CREDA_POC_CACHE` names a directory, a job whose `<job_key>.json`
    is there is read instead of run, and every job run is written there as it
    finishes.
    """
    cache = os.environ.get(CACHE_ENV)
    context = multiprocessing.get_context("spawn")
    if not cache:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, mp_context=context,
                initializer=_worker_init, initargs=(src,)) as pool:
            return list(pool.map(_call, jobs))
    directory = Path(cache)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{job_key(job, src)}.json" for job in jobs]
    outputs = [json.loads(p.read_text()) if p.exists() else None for p in paths]
    missing = [i for i, p in enumerate(paths) if outputs[i] is None]
    if missing:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, mp_context=context,
                initializer=_worker_init, initargs=(src,)) as pool:
            for i, out in zip(missing, pool.map(_call, [jobs[i] for i in missing])):
                _store(paths[i], out)
                outputs[i] = out
    return outputs


# ---------------------------------------------------------------- statistics

def summary(values) -> dict:
    """Mean, sd, a 95 % t-interval of the mean, median and n."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "ci95": [mean - half, mean + half],
            "median": float(np.median(x))}


def paired_wilcoxon(x, y=None, alternative: str = "greater") -> dict:
    """Wilcoxon signed-rank on `x - y` (or `x` alone), one-sided by default.

    Zero differences are dropped (`zero_method="wilcox"`). Reported with the
    rank-biserial correlation and the Hodges-Lehmann estimate with its
    Wilcoxon-based 95 % interval (normal approximation of the critical rank).
    """
    d = np.asarray(x, dtype=float) - (0.0 if y is None else np.asarray(y, dtype=float))
    nonzero = d[d != 0]
    walsh = np.sort([(d[i] + d[j]) / 2 for i in range(len(d)) for j in range(i, len(d))])
    hl = float(np.median(walsh))
    n = len(d)
    k = int(math.floor(n * (n + 1) / 4
                       - 1.959963984540054 * math.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    k = max(k, 0)
    ci = [float(walsh[k]), float(walsh[len(walsh) - 1 - k])]
    if len(nonzero) == 0:
        return {"statistic": float("nan"), "p": 1.0, "rankBiserial": 0.0,
                "hodgesLehmann": hl, "hodgesLehmannCI": ci, "n": n, "nonZero": 0,
                "alternative": alternative}
    result = stats.wilcoxon(nonzero, zero_method="wilcox", alternative=alternative)
    ranks = stats.rankdata(np.abs(nonzero))
    plus, minus = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    return {"statistic": float(result.statistic), "p": float(result.pvalue),
            "rankBiserial": float((plus - minus) / (plus + minus)),
            "hodgesLehmann": hl, "hodgesLehmannCI": ci, "n": n,
            "nonZero": int(len(nonzero)), "alternative": alternative}


def holm(pvalues) -> list[float]:
    """Holm's step-down adjustment, returned in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[index]))
        adjusted[index] = running
    return adjusted.tolist()
