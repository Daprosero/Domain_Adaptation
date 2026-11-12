"""Instrumented MIL-CREDA phase 2 on the spectral encoder: where does the transfer fail?

Diagnostic outside the plan's protocol; writes nothing to the model store. Phase 1 is the
settled general scheme (spectral encoder, 1-45 Hz + common average reference windows, at most
60 per bag, `learned_phase1.HYPER`, explicit L2 0.01, between-bag sigma). Phase 2 is traced
step by step, term by term, under one of three objectives (`OBJECTIVES`):

- "adapt": eta_glob L_glob + eta_loc L_loc; L_src read without gradient, phi frozen, no L2-SP
  (`two_phase.adapt_target_phase` as it stands);
- "eq39": Eq. 39 as written, L_src with gradient (to phi) + L_glob + L_loc, phi trained, mu 0;
- "eq39+l2sp": "eq39" plus mu ||theta_t - theta_t0||^2, mu = 10 (plans v64/v65).
Every variant runs with gamma 0.5 and attention temperature 1, argmax pseudo-labels (Eq. 22),
confidences (Eq. 24) and g (Eq. 35) with gradient, tau_loc 1, gate 1 (none of the v63 fixes).

Three target-trunk initialisations (`ARMS`): "copy" (same device, theta_t0 = theta_s via
`two_phase.bridge`, source standardization), "warm" (device transfer on each domain's full
montage; the source projection columns of every channel name both montages hold are copied,
extra target channels get zero columns, the bias is copied; source statistics on the shared
feature slots and label-free target training statistics on the rest) and "scratch" (same
transfer, theta_t randomly initialised by the seeded build, target training statistics,
mu forced to 0).

Per step: each term's value, the L2 norm of its gradient w.r.t. theta_t and phi, cosines
between term gradients and, as the key readout, the cosine of each term's gradient with the
gradient of an ORACLE target cross-entropy (true target labels of the step's batch; a
diagnosis only, never in the objective or the optimizer). A term absent from the objective is
still evaluated and read out. Per epoch (and at epoch -1, before the first step): target test
and train AUC / BA@0.5 / BA@cut, source test AUC, drift, attention entropy and the geometry of
the bag representations. The cut is chosen once per rotation on the source inner-validation
subjects of a phase-1 model trained on inner split 0 (source labels only).

The replica is checked against the uninstrumented existing path (`two_phase.adapt_target_phase`
under `ablation_pilot.attention_rules`, `ablation_pilot.fix_rules(frozenset())` and
`v63_setup.mil_creda_rules`) on the same seed and data for `EQUIVALENCE_EPOCHS` epochs.
Target labels never enter training, the cut or any selection. Writes phase2-trace.json.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from MIL_CREDA_EEG import ablation_pilot as AP
from MIL_CREDA_EEG import assignment as A
from MIL_CREDA_EEG import campaign, mil_creda_model as M, model_store, remote_units
from MIL_CREDA_EEG import spectral_trunk, two_phase
from MIL_CREDA_EEG import learned_phase1 as L
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG import v63_setup as V
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2

ARMS = ("copy", "warm", "scratch", "copy-own", "warm-own")
#: "-own": the same weights, but the target standardization is the label-free statistics of the
#: target training bags on every slot (per-domain standardization, no learning).
OBJECTIVES = ("adapt", "eq39", "eq39+l2sp", "eq39+prior", "eq39+truePL", "glob", "loc",
              "glob+truePL", "loc+truePL", "oracle-ce")
#: Diagnostic controls (target labels read, never a method): "+truePL" replaces Eq. 22 by the
#: true target labels; "oracle-ce" trains theta_t on Eq. 21 with the true target labels, head
#: frozen (upper bound of what theta_t can reach). "glob"/"loc": one term of Eq. 39 only.
#: "eq39+prior": eq39 with Eq. 22 replaced by prior-matched pseudo-labels. At the start of each
#: epoch the target-train bags are scored on the target path; a bag is labelled positive when
#: its score exceeds the (1 - pi_s) quantile, pi_s the positive share of the SOURCE training
#: bags. No target label is read.
PRIOR_OBJECTIVES = ("eq39+prior",)
#: Phase-1 aggregation, gamma 0 in both (consensus stays out of phase 1); phi trains in both.
#: "neutral": temperature 10, fixed. "searched": the temperature of TAU_GRID with the best
#: (BA@0.5 + AUC) / 2 on the source inner-validation subjects of split 0 (source labels only).
PHASE_ONE_ATTENTION = {"neutral": (10.0,), "searched": (1.0, 2.0, 5.0, 10.0)}
TAU_GRID = PHASE_ONE_ATTENTION["searched"]
SEARCH_RULE = ("each tau of the grid trains on source inner_train split 0 and is scored on source "
               "inner_validation split 0 by (BA at 0.5 + AUC) / 2, subject level; the best tau "
               "(ties to the larger tau) is refitted on the refit role")
PHASE_ONE_L2 = 0.01
#: searched phase-1 settings per rotation, {rotation: {lr1, l2, width, tauAtt1, epochs}} from
#: spectral_search (TPE on source inner splits only); empty means the fixed recipe above
PHASE_ONE_SEARCHED: dict = {}
PHASE_TWO = {"lr2": 1e-3, "wd2": 1e-5, "epochs2": 20, "tauLocal": 1.0}
PHASE_TWO_AGGREGATION = {"gamma": 0.5, "temperature": 1.0}
MU = 10.0
EQUIVALENCE_EPOCHS = 2
CUT_RULE = ("the cut maximizing subject-level BA over source inner-validation subjects (split 0) "
            "of the chosen-tau phase-1 candidate trained on inner_train split 0; midpoints "
            "between sorted scores, ties toward 0.5; chosen once per rotation and applied, "
            "unchanged, to every snapshot")
TERMS = ("src", "glob", "loc", "l2sp", "total")
#: optional read-only hook (xpilot's sim experiment): when True, every snapshot also carries
#: the per-window attention weights of the target bags, read in eval mode without gradient
#: (no dropout draw, no running-statistics update), so training is unchanged; off by default
ATTENTION_EXPORT = False


# ------------------------------------------------------------------ small helpers


def _s(x, k=5):
    """Five significant digits; None for missing or non-finite values."""
    if x is None:
        return None
    x = float(x)
    return float(f"{x:.{k}g}") if np.isfinite(x) else None


def _flat(ts) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1).double() for t in ts]) if ts else torch.zeros(0)


def _norm(ts) -> float:
    return float(_flat(ts).norm())


def _cos(a, b):
    a, b = _flat(a), _flat(b)
    na, nb = float(a.norm()), float(b.norm())
    if na < 1e-30 or nb < 1e-30:
        return None
    return float(a @ b) / (na * nb)


def _grads(value, params):
    if not value.requires_grad:
        return [torch.zeros_like(p) for p in params]
    gs = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, gs)]


def _transfer(name: str):
    return V.BY_NAME.get(name) or {x.name: x for x in A.TRANSFERS}[name]


def _subjects(bags, p):
    per = {}
    for b, q in zip(bags, p):
        per.setdefault(b.subject.split("#")[0], [int(b.label), []])[1].append(float(q))
    return [v[0] for v in per.values()], [float(np.mean(v[1])) for v in per.values()]


def _metrics(y, p, cut):
    y, p = np.asarray(y), np.asarray(p)
    two = len(set(y.tolist())) > 1
    return {"AUC": _s(roc_auc_score(y, p)) if two else None,
            "BA": _s(balanced_accuracy_score(y, p > .5)) if two else None,
            "BAcut": _s(balanced_accuracy_score(y, p > cut)) if two else None,
            "pos": _s(np.mean(p > .5)), "posCut": _s(np.mean(p > cut))}


def _probabilities(model, bags, domain, device):
    G = two_phase.evaluate(model, bags, domain, device)
    return np.nan_to_num(G[:, 1].detach().cpu().numpy().astype(float), nan=0.5)


# ------------------------------------------------------------------ phase 1


def _hyper(cfg, epochs2=None, temperature=10.0, searched=None):
    epochs2 = PHASE_TWO["epochs2"] if epochs2 is None else epochs2
    extra = ({"lr1": float(searched["lr1"]), "epochs1": int(searched["epochs"])} if searched else {})
    return two_phase.check_hyper({**L.HYPER, **extra, "bagsPerStep": cfg["schedule"]["bagsPerStep"],
                                  "attentionGamma": 0.0, "attentionTemperature": float(temperature),
                                  "lr2": PHASE_TWO["lr2"], "wd2": PHASE_TWO["wd2"],
                                  "tauLocal": PHASE_TWO["tauLocal"], "epochs2": int(epochs2)})


def phase_one(source, c_source, c_target, hyper, seed, device, *, transfer="", fold=0, searched=None):
    """learned_phase1's recipe on `source` (built with both channel counts), then the
    between-bag sigma of plan v63/v64 fixed on the model. `searched` carries the
    spectral_search width and L2 of this rotation (lr1, epochs and tau come in `hyper`)."""
    from MIL_CREDA_EEG.spectral_search import width as _width
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    with _width(int(searched["width"]) if searched else spectral_trunk.WIDTH):
        model = model_store.build_model("MIL-CREDA", c_source, c_target, hyper).to(device)
    model.source_trunk.norm = sc.standard_of(model.source_trunk, source).to(device)
    with explicit_l2(float(searched["l2"]) if searched else PHASE_ONE_L2):
        two_phase.train_source_phase(model, source, [], hyper, g, device, seed=seed, fold=fold,
                                     transfer=transfer)
    model._sigma = V.between_bag_sigma(model, source, "source", seed=seed)
    return model


def best_cut(y, p) -> float:
    grid = sorted(set(p))
    cands = [(grid[i] + grid[i + 1]) / 2 for i in range(len(grid) - 1)] or [0.5]
    return float(max(cands, key=lambda t: (balanced_accuracy_score(y, np.asarray(p) > t),
                                           -abs(t - .5))))


# ------------------------------------------------------------------ target-trunk arms


def _shared_slots(source_names, target_names):
    """(target channel, source channel) index pairs of every name both montages hold."""
    return [(ct, list(source_names).index(n)) for ct, n in enumerate(target_names)
            if n in source_names]


def _feature_index(c, k=None):
    """SpectralTrunk.features flattens (C, 10) as c*10 + k."""
    width = 2 * len(spectral_trunk.BANDS)
    return list(range(c * width, (c + 1) * width)) if k is None else c * width + k


def initialise_target(model, arm, target_train, source_names, target_names):
    """Set theta_t0 and the target standardization for `arm` AFTER `two_phase.bridge`."""
    s, t = model.source_trunk, model.target_trunk
    info = {"arm": arm}
    with torch.no_grad():
        own_std = arm.endswith("-own")
        arm = arm[:-4] if own_std else arm
        if arm == "copy":
            if s.channels != t.channels or list(source_names) != list(target_names):
                raise ValueError("the copy arm needs identical montages")
            t.norm = sc.Standard(s.norm.mean.clone(), s.norm.std.clone())
            info["standardization"] = "source training statistics (plan v63: no target re-standardization)"
        elif arm == "warm":
            own = sc.standard_of(t, target_train)
            mean, std = own.mean.clone(), own.std.clone()
            t.project.weight.zero_()
            pairs = _shared_slots(source_names, target_names)
            for ct, cs in pairs:
                ti, si = _feature_index(ct), _feature_index(cs)
                t.project.weight[:, ti] = s.project.weight[:, si]
                mean[ti], std[ti] = s.norm.mean[si], s.norm.std[si]
            t.project.bias.copy_(s.project.bias)
            t.norm = sc.Standard(mean, std)
            info.update({"sharedChannels": [target_names[ct] for ct, _ in pairs],
                         "sourceOnly": [n for n in source_names if n not in target_names],
                         "targetOnly": [n for n in target_names if n not in source_names],
                         "standardization": "source statistics on shared slots, label-free target "
                                            "training statistics on the extra slots"})
        elif arm == "scratch":
            t.norm = sc.standard_of(t, target_train)
            info["standardization"] = "label-free target training statistics"
            info["init"] = "standard nn.Linear init from the seeded build (torch.manual_seed(seed))"
        else:
            raise ValueError(f"arm is one of {ARMS}")
        if own_std:
            t.norm = sc.standard_of(t, target_train)
            info["standardization"] = "label-free target training statistics on every slot"
    return info


@torch.no_grad()
def warm_check(model, bags, source_names, target_names, device):
    """The warm-started target path vs the frozen source path on the same target windows: the
    source trunk reads the shared channels' features at their source slots and its own
    source-only channels at their source mean (standardized value 0)."""
    s, t = model.source_trunk, model.target_trunk
    pairs = _shared_slots(source_names, target_names)
    emb, prob = 0.0, 0.0
    for b in bags:
        x = b.windows.to(device)
        h_t = t(x).double()
        f_t = torch.nan_to_num(t.features(x))
        f_s = s.norm.mean.expand(f_t.shape[0], -1).clone()
        for ct, cs in pairs:
            f_s[:, _feature_index(cs)] = f_t[:, _feature_index(ct)]
        h_s = s.project(s.norm(f_s)).double()
        emb = max(emb, float((h_t - h_s).abs().max()))
        g_t = model.classify(model.represent(model.bag_pairs([h_t])))
        g_s = model.classify(model.represent(model.bag_pairs([h_s])))
        prob = max(prob, float((g_t - g_s).abs().max()))
    return {"maxAbsEmbedding": _s(emb), "maxAbsProbability": _s(prob), "bags": len(bags)}


# ------------------------------------------------------------------ snapshots


def _rbf_mmd(X, Y):
    Z = torch.cat([X, Y])
    D = torch.cdist(Z, Z)
    bw = float(D[~torch.eye(len(Z), dtype=torch.bool)].median()) or 1.0
    K = torch.exp(-D ** 2 / (2 * bw ** 2))
    n = len(X)
    return float(K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean()), bw


def _geometry(model, Zs, ys, Zt, yt):
    ys, yt = np.asarray(ys), np.asarray(yt)
    mu = lambda Z, y, c: Z[torch.from_numpy(y == c)].mean(0)
    mmd, bw = _rbf_mmd(Zs, Zt)
    s0, s1, t0, t1 = mu(Zs, ys, 0), mu(Zs, ys, 1), mu(Zt, yt, 0), mu(Zt, yt, 1)
    w = model.head.linear.weight.detach().double()
    head = (w[1] - w[0]).cpu()
    c = lambda a, b: _cos([a], [b])
    return {"meanDist": _s((Zs.mean(0) - Zt.mean(0)).norm()), "mmd2": _s(mmd), "mmdBandwidth": _s(bw),
            "tc0_sc0": _s((t0 - s0).norm()), "tc1_sc1": _s((t1 - s1).norm()),
            "tc0_sc1": _s((t0 - s1).norm()), "tc1_sc0": _s((t1 - s0).norm()),
            "t0_t1": _s((t1 - t0).norm()), "s0_s1": _s((s1 - s0).norm()),
            "directionCos": _s(c(t1 - t0, s1 - s0)),
            "headCosTarget": _s(c(t1 - t0, head)), "headCosSource": _s(c(s1 - s0, head)),
            "spreadTarget": _s(Zt.std(0).norm()), "spreadSource": _s(Zs.std(0).norm())}


@torch.no_grad()
def _attention_entropy(model, bags, device):
    out = []
    for b in bags:
        (_, w), = model.bag_pairs(model.embed([b.windows.to(device)], "target"))
        w = w.reshape(-1).clamp_min(1e-300)
        out.append(float(-(w * w.log()).sum()) / max(np.log(len(w)), 1e-12))
    return float(np.mean(out))


@torch.no_grad()
def attention_weights(model, bags, device) -> dict:
    """Subject -> in-bag attention weight of every window (target path, eval mode)."""
    modes = {m: m.training for m in model.modules()}
    model.eval()
    try:
        out = {}
        for b in bags:
            (_, w), = model.bag_pairs(model.embed([b.windows.to(device)], "target"))
            out[b.subject] = [_s(v, 4) for v in w.reshape(-1).tolist()]
    finally:
        for m, mode in modes.items():
            m.train(mode)
    return out


def snapshot(model, ctx, device, keep_rows=True):
    """Target test/train metrics, source test AUC, drift, entropy and geometry."""
    out = {}
    for key, bags in (("targetTest", ctx["target_test"]), ("targetTrain", ctx["target_train"])):
        y, p = _subjects(bags, _probabilities(model, bags, "target", device))
        out[key] = _metrics(y, p, ctx["cut"])
        if keep_rows:
            out[key]["rows"] = [[int(a), _s(b, 4)] for a, b in zip(y, p)]
    y, p = _subjects(ctx["source_test"], _probabilities(model, ctx["source_test"], "source", device))
    out["sourceTestAUC"] = _metrics(y, p, ctx["cut"])["AUC"]
    out["driftTheta"] = _s(_norm([p - ctx["theta0"][n] for n, p in model.target_trunk.named_parameters()]))
    out["driftPhi"] = _s(_norm([p - q for p, q in zip(model.attention_parameters(), ctx["phi0"])]))
    out["attentionEntropy"] = _s(_attention_entropy(model, ctx["target_train"], device))
    Zs = two_phase.represent(model, ctx["source_train"], "source", device)
    for key, bags in (("geometryTrain", ctx["target_train"]), ("geometryTest", ctx["target_test"])):
        Zt = two_phase.represent(model, bags, "target", device)
        out[key] = _geometry(model, Zs, [b.label for b in ctx["source_train"]], Zt,
                             [b.label for b in bags])
    if ATTENTION_EXPORT:
        out["attention"] = {"targetTest": attention_weights(model, ctx["target_test"], device),
                            "targetTrain": attention_weights(model, ctx["target_train"], device)}
    return out


# ------------------------------------------------------------------ the two phase-2 paths


@contextlib.contextmanager
def _objective_rules(model, objective, mu):
    """The patches the existing path runs under for `objective` (fix_rules with no fix; for
    eq39 also attention_rules and mil_creda_rules with gate 1)."""
    with AP.fix_rules(frozenset()):
        if objective == "adapt":
            yield
        else:
            with V.mil_creda_rules(model, 0.5, 1.0, mu=mu):   # threshold unused: argmax Eq. 22
                yield


@contextlib.contextmanager
def _prior_rules(prior, state):
    """Prior-matched Eq. 22: positive when the bag's score exceeds the epoch's quantile cut."""
    if prior is None:
        yield
        return
    saved = M.pseudolabel
    M.pseudolabel = lambda g: int(float(g[1]) > state["cut"])
    try:
        yield
    finally:
        M.pseudolabel = saved


MODIFIERS = ("truePL", "affine", "knorm", "prior", "l2sp", "hardloc", "conf", "late", "avg", "cluster", "sep", "im")


def _known(objective: str) -> bool:
    head, *mods = objective.split("+")
    return objective in OBJECTIVES or (head in ("eq39", "glob", "loc") and set(mods) <= set(MODIFIERS))


class AffineStandard(torch.nn.Module):
    """The fixed standardization followed by a learned per-feature scale and shift: the only
    target parameters phase 2 moves under "+affine" (the projection stays at theta_t0)."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        n = base.mean.numel()
        self.scale = torch.nn.Parameter(torch.ones(n))
        self.shift = torch.nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return self.base(x) * self.scale + self.shift


@contextlib.contextmanager
def _knorm_rules(on):
    """Normalized bag kernel k(a,b) / sqrt(k(a,a) k(b,b)): every self-similarity is 1."""
    if not on:
        yield
        return
    saved = M.bag_kernel_matrix

    def normalized(rows, cols, sigma):
        K = saved(rows, cols, sigma)
        dr = torch.stack([saved([r], [r], sigma)[0, 0] for r in rows])
        dc = torch.stack([saved([c], [c], sigma)[0, 0] for c in cols])
        return K / torch.sqrt(dr[:, None] * dc[None, :]).clamp_min(1e-12)

    M.bag_kernel_matrix = normalized
    try:
        yield
    finally:
        M.bag_kernel_matrix = saved


@contextlib.contextmanager
def _true_rules(on, queue):
    """Diagnostic Eq. 22: the true target label of each bag, in batch order."""
    if not on:
        yield
        return
    saved = M.pseudolabel
    M.pseudolabel = lambda g: queue.pop(0)
    try:
        yield
    finally:
        M.pseudolabel = saved


CONF_SHARE = 0.3
KMEANS_ITERATIONS = 20
SEP_WEIGHT, IM_WEIGHT = 1.0, 1.0


def separation_term(model, pairs_s, labels_s, pairs_t, pseudo):
    """CAN-style inter-class discrepancy, to be MAXIMISED: for each ordered pair c != c', the
    squared bag-kernel MMD between target class c (pseudo-labels) and source class c', halved
    so it lies in [0, 1]. The global term only pulls same-class sets together; this one also
    pushes target bags of one class away from source bags of the other class."""
    sigma = model.sigma
    K_ss = M.bag_kernel_matrix(pairs_s, pairs_s, sigma)
    K_st = M.bag_kernel_matrix(pairs_s, pairs_t, sigma)
    K_tt = M.bag_kernel_matrix(pairs_t, pairs_t, sigma)
    ls, lt = labels_s.reshape(-1).to(K_ss.device), torch.as_tensor(pseudo).reshape(-1).to(K_ss.device)
    values = []
    for c in range(model.classes):
        for d in range(model.classes):
            t_idx, s_idx = torch.nonzero(lt == c).reshape(-1), torch.nonzero(ls == d).reshape(-1)
            if c == d or t_idx.numel() == 0 or s_idx.numel() == 0:
                continue
            mmd = (K_tt[t_idx][:, t_idx].mean() + K_ss[s_idx][:, s_idx].mean()
                   - 2 * K_st[s_idx][:, t_idx].mean())
            values.append(mmd / 2)
    return torch.stack(values).mean() if values else torch.zeros((), device=K_ss.device)


def information_term(G_t):
    """SHOT-style information maximisation on the target predictions, to be MINIMISED:
    mean per-bag entropy (confident predictions) minus the entropy of the batch-mean
    prediction (both classes still used). Counters the confidence collapse of the global term."""
    p = G_t.clamp_min(1e-8)
    per_bag = -(p * p.log()).sum(1).mean()
    mean = p.mean(0)
    return per_bag + (mean * mean.log()).sum()


def representations(model, bags, side, device) -> np.ndarray:
    """Bag representations (the classifier's input) with the trunk frozen."""
    model.eval()
    with torch.no_grad():
        pairs = model.bag_pairs(model.embed([b.windows.to(device) for b in bags], side))
        return model.represent(pairs).detach().cpu().numpy().astype(np.float64)


def _unit(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def spherical_kmeans(z, centroids, iterations=KMEANS_ITERATIONS):
    """Cosine k-means on the target, seeded with the source class centroids (SHOT/CAN style):
    cluster k keeps the class of the source centroid it started from. Label-free."""
    zn, c = _unit(z), _unit(centroids)
    assign = np.argmax(zn @ c.T, axis=1)
    for _ in range(iterations):
        c = _unit(np.stack([zn[assign == k].mean(0) if np.any(assign == k) else c[k]
                            for k in range(len(c))]))
        nxt = np.argmax(zn @ c.T, axis=1)
        if np.array_equal(nxt, assign):
            break
        assign = nxt
    return assign


def cluster_labels(model, source, target, device) -> np.ndarray:
    zs = representations(model, source, "source", device)
    zt = representations(model, target, "target", device)
    ys = np.array([b.label for b in source])
    return spherical_kmeans(zt, np.stack([zs[ys == k].mean(0) for k in range(model.classes)]))
LATE_START, LATE_RAMP = 10, 5    # "+late": L_loc off for 10 epochs, then a linear ramp over 5


@contextlib.contextmanager
def _conf_rules(on, state):
    """L_loc over the most confident target bags only: the CONF_SHARE of each batch with the
    largest |g_1 - g_0| of the frozen head (label-free). Read by the hard mass wrapper."""
    if not on:
        yield
        return
    saved = M.local_loss

    def selected(d):
        c = torch.tensor(state["conf"][-len(d):])
        keep = torch.argsort(c, descending=True)[:max(1, int(np.ceil(CONF_SHARE * len(d))))]
        state["conf"].clear()
        state["keep"] = keep.tolist()
        return saved(d[keep.to(d.device)])

    M.local_loss = selected
    try:
        yield
    finally:
        M.local_loss = saved


@contextlib.contextmanager
def _hardloc_rules(on, queue, state=None):
    """Class-conditional Eq. 35 with a hard class: the mass g_j is replaced by the one-hot of
    the true label (queue filled, "+truePL") or of the argmax pseudo-label. No gradient flows
    through the mass, so each target bag is pulled only toward its own class's sources."""
    if not on:
        yield
        return
    saved = M.total_correspondence

    def hard(similarities, source_labels, target_scores, tau):
        g = torch.as_tensor(target_scores).reshape(-1)
        if state is not None:
            state["conf"].append(float((g[1] - g[0]).abs().detach()))
        c = queue.pop(0) if queue else int(torch.argmax(g.detach()))
        return saved(similarities, source_labels, torch.nn.functional.one_hot(
            torch.tensor(c), g.numel()).to(g.dtype).to(g.device), tau)

    M.total_correspondence = hard
    try:
        yield
    finally:
        M.total_correspondence = saved


def existing_path(model, source, target, hyper, seed, device, objective, mu, arm_init):
    """The uninstrumented phase 2: bridge, arm init, then `two_phase.adapt_target_phase`.
    Returns per-step objective values and the final target trunk."""
    steps = []
    attention = AP.attention_rules() if objective != "adapt" else contextlib.nullcontext()
    with attention:
        joint = two_phase.bridge(model)
        arm_init(model)
        inner = two_phase.phase_two_step

        def recorded(*a, **k):
            out = inner(*a, **k)
            steps.append(float(out["objective"].detach()))
            return out

        two_phase.phase_two_step = recorded
        try:
            with _objective_rules(model, objective, mu):
                g = torch.Generator().manual_seed(seed)
                two_phase.adapt_target_phase(model, {"checkpoint": {"sigma": model.sigma}},
                                             joint["frozen_state"], source, target, [], hyper, g,
                                             device)
        finally:
            two_phase.phase_two_step = inner
    return steps, {n: p.detach().clone() for n, p in model.target_trunk.named_parameters()}


def traced_path(model, source, target, target_labels, hyper, seed, device, objective, mu,
                arm_init, ctx, epochs, equivalence_epochs, log=print, prior=None, light=False):
    """The instrumented replica: same batches, same objective, same optimizer; the gradient is
    taken with autograd.grad and handed to Adam, term gradients read on the side.

    `light` (xpilot's phase-2 search): the same updates, so the same end model, without the
    snapshots, the per-step diagnostics or the epoch log. It reads no target label and no
    target test bag (`target_labels` and `ctx["target_test"]` are not used; `ctx` needs only
    "sigma"), and refuses the objectives that read target labels to train."""
    mods = set(objective.split("+")[1:])
    true_pl = "truePL" in mods
    cluster = "cluster" in mods       # Eq. 22 and the hard local mass from target clusters
    sep, im = "sep" in mods, "im" in mods
    affine, knorm = "affine" in mods, "knorm" in mods
    hardloc = "hardloc" in mods
    conf, conf_state = "conf" in mods, {"conf": []}
    late = "late" in mods
    avg, avg_state = "avg" in mods, {"n": 0, "mean": None}
    oracle_ce = objective == "oracle-ce"
    if light and (true_pl or cluster or oracle_ce):
        raise ValueError(f"light phase 2 reads no target label: {objective!r} needs them")
    head = objective.split("+")[0]
    selection = {"glob": "global term only", "loc": "local term only"}.get(head, two_phase.BOTH_TERMS)
    objective = "eq39" if head in ("glob", "loc") or mods & {"truePL", "affine", "knorm", "hardloc", "conf", "late", "avg", "cluster", "sep", "im"} else objective
    train_phi = objective not in ("adapt", "oracle-ce")
    joint = two_phase.bridge(model)
    info = arm_init(model)
    if affine:
        model.target_trunk.norm = AffineStandard(model.target_trunk.norm).to(next(model.parameters()).device)
        info["phaseTwoMoves"] = "per-feature scale and shift of the target standardization only"
    theta0 = {n: p.detach().clone() for n, p in model.target_trunk.named_parameters()}
    phi0 = [p.detach().clone() for p in model.attention_parameters()]
    ctx.update({"theta0": theta0, "phi0": phi0})
    theta = ([model.target_trunk.norm.scale, model.target_trunk.norm.shift] if affine
             else list(model.target_trunk.parameters()))
    phi = model.attention_parameters()
    for p in phi:                       # phi is differentiable for the readout in every variant
        p.requires_grad_(True)
    model.fix_sigma(ctx["sigma"])
    optimizer = torch.optim.Adam(theta + (phi if train_phi else []), lr=hyper["lr2"],
                                 weight_decay=hyper["wd2"])
    queues = two_phase.ClassQueues(source, model.classes, torch.Generator())
    g = torch.Generator().manual_seed(seed)
    queues.generator = g
    per_class = hyper["bagsPerStep"] // model.classes
    record = {"steps": [], "epochs": {}, "init": info}
    if not light:
        record["epochs"]["-1"] = snapshot(model, ctx, device)
    objective_values, theta_at_check = [], None
    cut_state = {"cut": 0.5}
    record["priorCuts"] = []
    pl_queue, loc_queue = [], []
    with _objective_rules(model, "adapt" if oracle_ce else objective, mu), \
            _prior_rules(prior, cut_state), _true_rules(true_pl or cluster, pl_queue), _knorm_rules(knorm), \
            _hardloc_rules(hardloc, loc_queue, conf_state), \
            _conf_rules(conf, conf_state):
        for epoch in range(epochs):
            if prior is not None:   # label-free: target scores and the source prior only
                q = _probabilities(model, target, "target", device)
                cut_state["cut"] = float(np.quantile(q, 1.0 - prior))
                record["priorCuts"].append(_s(cut_state["cut"]))
            if cluster:
                assigned = cluster_labels(model, source, target, device)
                record.setdefault("clusterAccuracy", []).append(
                    _s(float(np.mean(assigned == np.asarray(target_labels)))))
            two_phase.set_phase_two_modes(model)
            for step, batch in enumerate(two_phase.shuffled_batches(len(target), hyper["bagsPerStep"], g)):
                src = [source[i] for i in queues.draw(per_class)]
                tgt = [target[i] for i in batch]
                labels = torch.tensor([b.label for b in src], device=device)
                true_t = None if light else torch.tensor([target_labels[i] for i in batch], device=device)
                pairs_s = model.bag_pairs(model.embed([b.windows.to(device) for b in src], "source"))
                source_ce = M.source_term(model, model.classify(model.represent(pairs_s)), labels)
                pairs_t = model.bag_pairs(model.embed([b.windows.to(device) for b in tgt], "target"))
                pl_queue[:] = [] if light else [int(v) for v in true_t.tolist()]
                if cluster:
                    pl_queue[:] = [int(assigned[i]) for i in batch]
                loc_queue[:] = pl_queue[:] if (true_pl or cluster) else []
                terms = M.adaptation_terms(model, pairs_s, labels, pairs_t, hyper["tauLocal"])
                total = two_phase.phase_two_objective(terms, selection)
                if late:    # delayed, ramped local coefficient
                    ramp = min(1.0, max(0.0, (epoch - LATE_START + 1) / LATE_RAMP))
                    lg, ll = two_phase.term_coefficients(two_phase.BOTH_TERMS)
                    total = lg * terms["global"] + ramp * ll * terms["local"]
                if sep:
                    total = total - SEP_WEIGHT * separation_term(model, pairs_s, labels, pairs_t, terms["pseudo"])
                if im:
                    total = total + IM_WEIGHT * information_term(terms["G_target"])
                if train_phi:
                    total = total + source_ce
                if oracle_ce:
                    total = M.source_term(model, terms["G_target"], true_t)
                if light:       # the update only: the gradient of the total, handed to Adam
                    params = theta + phi
                    G_total = _grads(total, params)
                    optimizer.zero_grad()
                    for i, (p, gr) in enumerate(zip(params, G_total)):
                        p.grad = gr.detach().clone() if (train_phi or i < len(theta)) else None
                    optimizer.step()
                    model.constrain(*(("target_trunk", "attention") if train_phi else ("target_trunk",)))
                    objective_values.append(float(total.detach()))
                    continue
                prox = sum(((p - theta0[n]) ** 2).sum() for n, p in model.target_trunk.named_parameters())
                l2sp_read = MU * prox                     # readout at the plan's mu
                oracle = M.source_term(model, terms["G_target"], true_t)
                values = {"src": source_ce, "glob": terms["global"], "loc": terms["local"],
                          "l2sp": l2sp_read, "total": total}
                params = theta + phi
                G = {k: _grads(v, params) for k, v in {**values, "oracle": oracle}.items()}
                gth = {k: v[:len(theta)] for k, v in G.items()}
                gph = {k: v[len(theta):] for k, v in G.items()}
                before = [p.detach().clone() for p in theta]
                optimizer.zero_grad()
                for i, (p, gr) in enumerate(zip(theta + phi, G["total"])):
                    p.grad = gr.detach().clone() if (train_phi or i < len(theta)) else None
                optimizer.step()
                model.constrain(*(("target_trunk", "attention") if train_phi else ("target_trunk",)))
                objective_values.append(float(total.detach()))
                pseudo = terms["pseudo"].cpu()
                record["steps"].append({
                    "e": epoch, "s": step,
                    "val": {k: _s(v.detach()) for k, v in {**values, "oracle": oracle}.items()},
                    "gTheta": {k: _s(_norm(v)) for k, v in gth.items()},
                    "gPhi": {k: _s(_norm(v)) for k, v in gph.items()},
                    "cosTheta": {"glob_loc": _s(_cos(gth["glob"], gth["loc"])),
                                 "glob_l2sp": _s(_cos(gth["glob"], gth["l2sp"])),
                                 "loc_l2sp": _s(_cos(gth["loc"], gth["l2sp"]))},
                    "oracleTheta": {k: _s(_cos(gth[k], gth["oracle"])) for k in TERMS if k != "src"},
                    "oraclePhi": {k: _s(_cos(gph[k], gph["oracle"])) for k in TERMS if k != "l2sp"},
                    "pl": {"acc": _s(float((pseudo == true_t.cpu()).float().mean())),
                           "pos": _s(float((pseudo == 1).float().mean())),
                           "active": int(terms["active_classes"]),
                           "conf": _s(terms["confidence_mean"]),
                           "accSel": (_s(float((pseudo[conf_state["keep"]] == true_t.cpu()[conf_state["keep"]])
                                               .float().mean())) if conf and conf_state.get("keep") else None)},
                    "upd": _s(_norm([p.detach() - q for p, q in zip(theta, before)]) / max(_norm(before), 1e-30))})
            if epoch + 1 == equivalence_epochs:
                theta_at_check = {n: p.detach().clone() for n, p in model.target_trunk.named_parameters()}
            if light:   # the "avg" snapshot swaps the weights back: no effect on the end model
                continue
            if avg and epoch >= LATE_START + LATE_RAMP:   # running mean of the moved weights
                live = [p.detach().clone() for p in theta + phi]
                avg_state["n"] += 1
                if avg_state["mean"] is None:
                    avg_state["mean"] = [w.clone() for w in live]
                else:
                    avg_state["mean"] = [m + (w - m) / avg_state["n"] for m, w in zip(avg_state["mean"], live)]
                with torch.no_grad():   # snapshot the averaged model, then resume from the live one
                    for p, m in zip(theta + phi, avg_state["mean"]):
                        p.copy_(m)
                    record["epochs"][str(epoch)] = snapshot(model, ctx, device)
                    for p, w in zip(theta + phi, live):
                        p.copy_(w)
            else:
                record["epochs"][str(epoch)] = snapshot(model, ctx, device)
            e = record["epochs"][str(epoch)]
            log(objective, "epoch", epoch, "test", e["targetTest"]["AUC"], e["targetTest"]["BAcut"],
                "train", e["targetTrain"]["AUC"], flush=True)
    two_phase.set_phase_two_modes(model)
    keep = lambda st: {k: v for k, v in st.items() if k not in AP.ATTENTION_KEYS}
    frozen = (keep if train_phi else (lambda st: st))
    if not two_phase._states_equal(frozen(two_phase.phase_one_state(model)), frozen(joint["frozen_state"])):
        raise two_phase.FrozenPartsMoved("the traced phase 2 moved a frozen part")
    return record, objective_values, theta_at_check


# ------------------------------------------------------------------ one rotation


def _epoch_means(steps):
    out = {}
    for e in sorted({s["e"] for s in steps}):
        ss = [s for s in steps if s["e"] == e]
        mean = lambda f: _s(np.mean(v)) if (v := [x for x in map(f, ss) if x is not None]) else None
        out[str(e)] = {
            **{f"val_{k}": mean(lambda s, k=k: s["val"][k]) for k in (*TERMS, "oracle")},
            **{f"gTheta_{k}": mean(lambda s, k=k: s["gTheta"][k]) for k in (*TERMS, "oracle")},
            **{f"gPhi_{k}": mean(lambda s, k=k: s["gPhi"][k]) for k in ("src", "glob", "loc", "total", "oracle")},
            **{f"cos_{k}": mean(lambda s, k=k: s["cosTheta"][k]) for k in ("glob_loc", "glob_l2sp", "loc_l2sp")},
            **{f"oracleTheta_{k}": mean(lambda s, k=k: s["oracleTheta"][k]) for k in ("glob", "loc", "l2sp", "total")},
            **{f"oraclePhi_{k}": mean(lambda s, k=k: s["oraclePhi"][k]) for k in ("src", "glob", "loc", "total")},
            "plAcc": mean(lambda s: s["pl"]["acc"]), "plPos": mean(lambda s: s["pl"]["pos"]),
            "plActive": mean(lambda s: s["pl"]["active"]), "plConf": mean(lambda s: s["pl"]["conf"]),
            "upd": mean(lambda s: s["upd"])}
    return out


def run_rotation(transfer, cfg, a, bags, device, seed, rotation, arm, objectives,
                 phase1_attention="neutral", log=print, sides=None):
    """Phase 1 once (cut model and base model), then every objective from that base.
    `sides` hands the prepared bag lists directly (other datasets: xpilot); otherwise they
    come from the BALLADEER assignment `a`."""
    if sides is None:
        prep = lambda side, role, split=None: L._filtered_whole(
            A.side_bags(a, bags, side, rotation, role, split=split), car=True, reject=None, channels=None)
        names = lambda side: list(A.side_bags(a, bags, side, rotation, "test")[0].channels)
        sides = {"source": prep("source", A.REFIT_ROLE), "source_test": prep("source", "test"),
                 "target": prep("target", A.REFIT_ROLE), "target_test": prep("target", "test"),
                 "inner_train": prep("source", "inner_train", 0),
                 "inner_valid": prep("source", "inner_validation", 0),
                 "s_names": names("source"), "t_names": names("target")}
    source, source_test = sides["source"], sides["source_test"]
    target, target_test = sides["target"], sides["target_test"]
    s_names, t_names = sides["s_names"], sides["t_names"]
    cs, ct = int(source[0].windows.shape[1]), int(target[0].windows.shape[1])
    started = time.time()
    # the phase-1 temperature (searched or fixed) and the decision cut, both from inner split 0:
    # each candidate trains on its source training subjects and is scored on its validation ones
    inner_train, inner_valid = sides["inner_train"], sides["inner_valid"]
    grid = []
    searched = PHASE_ONE_SEARCHED.get(int(rotation)) or PHASE_ONE_SEARCHED.get(str(rotation))
    taus = (float(searched["tauAtt1"]),) if searched else PHASE_ONE_ATTENTION[phase1_attention]
    for tau in taus:
        cand = phase_one(inner_train, cs, ct, _hyper(cfg, temperature=tau, searched=searched), seed, device,
                         transfer=transfer, fold=rotation, searched=searched)
        vy, vp = _subjects(inner_valid, _probabilities(cand, inner_valid, "source", device))
        m = _metrics(vy, vp, 0.5)
        score = ((m["BA"] or 0.0) + (m["AUC"] or 0.0)) / 2
        grid.append({"tau": tau, "BA": m["BA"], "AUC": m["AUC"], "score": _s(score),
                     "cut": best_cut(vy, vp), "_s": score})
        del cand
    best = max(grid, key=lambda g: (g["_s"], g["tau"]))
    for g in grid:
        g.pop("_s")
    tau1, cut = best["tau"], best["cut"]
    hyper = _hyper(cfg, temperature=tau1, searched=searched)
    base = phase_one(source, cs, ct, hyper, seed, device, transfer=transfer, fold=rotation,
                     searched=searched)
    phase1_seconds = time.time() - started
    sigma = base.sigma
    sy, sp = _subjects(source_test, _probabilities(base, source_test, "source", device))
    phase1 = {"sourceTest": _metrics(sy, sp, cut), "mode": phase1_attention,
              "temperature": tau1, "gamma": 0.0, "grid": grid,
              "searchRule": SEARCH_RULE if phase1_attention == "searched" else "fixed tau 10",
              "seconds": round(phase1_seconds, 1)}
    arm_init = lambda m: initialise_target(m, arm, target, s_names, t_names)

    def phase_two_model():
        m = copy.deepcopy(base)
        m.attention_gamma = PHASE_TWO_AGGREGATION["gamma"]
        m.attention_temperature = PHASE_TWO_AGGREGATION["temperature"]
        return m

    # predict: the target path right after the arm init, at the PHASE-1 aggregation
    pm = copy.deepcopy(base)
    two_phase.bridge(pm)
    info = arm_init(pm)
    ctx = {"cut": cut, "sigma": sigma, "source_train": source, "source_test": source_test,
           "target_train": target, "target_test": target_test,
           "theta0": {n: p.detach().clone() for n, p in pm.target_trunk.named_parameters()},
           "phi0": [p.detach().clone() for p in pm.attention_parameters()]}
    predict = snapshot(pm, ctx, device)
    check = {}
    if arm == "warm":
        check = {"test": warm_check(pm, target_test, s_names, t_names, device),
                 "train": warm_check(pm, target, s_names, t_names, device)}
    elif arm == "copy":
        q_t = _probabilities(pm, target_test, "target", device)
        q_s = _probabilities(pm, target_test, "source", device)
        check = {"test": {"maxAbsProbability": _s(np.abs(q_t - q_s).max())}}
    del pm
    out = {"rotation": rotation, "arm": arm, "phase1Attention": phase1_attention,
           "sigma": _s(sigma), "sigmaRule": "v63_setup.between_bag_sigma on the trained source trunk",
           "cut": _s(cut), "cutRule": CUT_RULE, "init": info, "phase1": phase1,
           "subjects": {"source": len(source), "sourceTest": len(source_test),
                        "target": len(target), "targetTest": len(target_test),
                        "innerTrain": len(inner_train), "innerValid": len(inner_valid)},
           "channels": {"source": cs, "target": ct}, "predictEquality": check, "objectives": {}}
    for objective in objectives:
        mu = MU if objective == "eq39+l2sp" and arm != "scratch" else 0.0
        prior = (float(np.mean([b.label for b in source])) if objective in PRIOR_OBJECTIVES else None)
        base_objective = "eq39" if objective in PRIOR_OBJECTIVES else objective
        new_rule = prior is not None or objective not in ("adapt", "eq39", "eq39+l2sp")
        # equivalence: the existing path vs the replica, same seed and data (the prior rule is
        # new, so its replica is checked against plain eq39 only over the epochs it shares: none)
        t2 = time.time()
        ex_hyper = _hyper(cfg, EQUIVALENCE_EPOCHS, tau1)
        ex_steps, ex_theta = ([], None) if new_rule else existing_path(
            phase_two_model(), source, target, ex_hyper, seed, device, objective, mu, arm_init)
        existing_seconds = time.time() - t2
        t3 = time.time()
        model = phase_two_model()
        record, values, theta_check = traced_path(
            model, source, target, [b.label for b in target], hyper, seed, device, base_objective, mu,
            arm_init, ctx, PHASE_TWO["epochs2"], EQUIVALENCE_EPOCHS, log=log, prior=prior)
        n = len(ex_steps)
        equivalence = {"note": "new rule: no existing path to compare", "sourcePrior": _s(prior),
                       "targetPrior": _s(float(np.mean([b.label for b in target]))),
                       "cuts": record["priorCuts"]} if new_rule else {"epochs": EQUIVALENCE_EPOCHS, "steps": n,
                       "maxAbsObjective": _s(np.max(np.abs(np.asarray(ex_steps) - np.asarray(values[:n])))),
                       "maxAbsThetaT": _s(max(float((ex_theta[k] - theta_check[k]).abs().max()) for k in ex_theta)),
                       "existingPath": "two_phase.bridge + adapt_target_phase under ablation_pilot.fix_rules("
                                       "frozenset())" + ("" if objective == "adapt" else
                                                         ", ablation_pilot.attention_rules and "
                                                         f"v63_setup.mil_creda_rules(gate 1, mu {mu:g})")}
        out["objectives"][objective] = {
            "objective": objective, "mu": mu, "prior": _s(prior), "phiTrained": objective != "adapt",
            "sourceGradient": objective != "adapt", "equivalence": equivalence,
            "predict": predict, "epochs": record["epochs"], "epochMeans": _epoch_means(record["steps"]),
            "steps": record["steps"], "cut": out["cut"],
            "seconds": {"existing": round(existing_seconds, 1), "traced": round(time.time() - t3, 1)}}
        log(arm, phase1_attention, objective, "rotation", rotation, "equivalence",
            equivalence.get("maxAbsObjective"), equivalence.get("maxAbsThetaT"), flush=True)
    return out


# ------------------------------------------------------------------ pooled summary


def _pooled(runs, pick):
    y, p, c = [], [], []
    for r in runs:
        for lab, q in pick(r)["rows"]:
            y.append(lab), p.append(q), c.append(r["cut"])
    y, p, c = np.asarray(y), np.asarray(p), np.asarray(c)
    return {"AUC": _s(roc_auc_score(y, p)), "BA": _s(balanced_accuracy_score(y, p > .5)),
            "BAcut": _s(balanced_accuracy_score(y, p > c)), "n": int(len(y))}


def summary(runs):
    epochs = list(runs[0]["epochs"])
    out = {"predict": {k: _pooled(runs, lambda r, k=k: r["predict"][k]) for k in ("targetTest", "targetTrain")}}
    out["perEpoch"] = {}
    for e in epochs:
        row = {k: _pooled(runs, lambda r, k=k, e=e: r["epochs"][e][k]) for k in ("targetTest", "targetTrain")}
        num = lambda f: _s(np.mean([v for v in (f(r) for r in runs) if v is not None])) \
            if any(f(r) is not None for r in runs) else None
        row.update({"sourceTestAUC": num(lambda r: r["epochs"][e]["sourceTestAUC"]),
                    "driftTheta": num(lambda r: r["epochs"][e]["driftTheta"]),
                    "driftPhi": num(lambda r: r["epochs"][e]["driftPhi"]),
                    "attentionEntropy": num(lambda r: r["epochs"][e]["attentionEntropy"]),
                    **{f"geo_{k}": num(lambda r, k=k: r["epochs"][e]["geometryTrain"][k])
                       for k in runs[0]["epochs"][e]["geometryTrain"]}})
        if e != "-1":
            row.update({k: num(lambda r, k=k: r["epochMeans"][e][k]) for k in runs[0]["epochMeans"][e]})
        out["perEpoch"][e] = row
    if all("maxAbsObjective" in r["equivalence"] for r in runs):
        out["equivalence"] = {"maxAbsObjective": max(r["equivalence"]["maxAbsObjective"] for r in runs),
                              "maxAbsThetaT": max(r["equivalence"]["maxAbsThetaT"] for r in runs)}
    else:
        out["equivalence"] = {"prior": [r["equivalence"] for r in runs]}
    return out


# ------------------------------------------------------------------ entry points


def local_config(path="tools/kaggle/rot8-task/run-config.json") -> dict:
    c = json.load(open(path, encoding="utf-8"))["run"]["kwargs"]["config"]
    return {**{k: v for k, v in c.items() if k != "remoteInputs"}, "device": "cpu"}


def smoke(config: Mapping, product=None, *, transfer: str, arm: str,
          objectives: Sequence[str] = OBJECTIVES, phase1_attention: str = "neutral",
          seed: int = 1, rotations: Sequence[int] = (0, 1, 2, 3), workdir=None,
          environ: Mapping[str, str] | None = None) -> dict:
    """Every objective of one (arm, phase-1 attention), phase 1 shared per rotation."""
    objectives = [o for o in objectives if not (arm == "scratch" and o == "eq39+l2sp")]
    if arm not in ARMS or not all(_known(o) for o in objectives) or phase1_attention not in PHASE_ONE_ATTENTION:
        raise ValueError(f"arm in {ARMS}, objectives in {OBJECTIVES}, phase1Attention in "
                         f"{list(PHASE_ONE_ATTENTION)}")
    environ = os.environ if environ is None else environ
    workdir = Path(workdir) if workdir else Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    placement = remote_units.device_placement(config, environ)
    cfg = remote_units.resolve_inputs(campaign.check_config(config), environ=environ)
    campaign.register(cfg["strategies"])
    environment = remote_units.environment_stamp(remote_units.set_deterministic(),
                                                 remote_units.campaign_identity(cfg, environ), placement)
    device = torch.device(cfg.get("device", "cpu"))
    bags = campaign.load_bags(cfg)
    a = A.assign_from_bags(_transfer(transfer), bags)
    started, rots = time.time(), []
    with spectral_trunk.installed():
        for r in rotations:
            rots.append(run_rotation(transfer, cfg, a, bags, device, seed, int(r), arm, objectives,
                                     phase1_attention))
            print(arm, phase1_attention, transfer, "rotation", r, "done",
                  round(time.time() - started), "s", flush=True)
    by = {}
    for o in objectives:
        runs = [r["objectives"][o] for r in rots]
        by[o] = {"summary": summary(runs), "runs": runs}
    phase1 = [{"rotation": r["rotation"], **r["phase1"]} for r in rots]
    meta = [{k: v for k, v in r.items() if k != "objectives"} for r in rots]
    record = {"transfer": transfer, "arm": arm, "phase1Attention": phase1_attention,
              "objectives": objectives, "seed": seed, "rotations": list(rotations),
              "phaseOneSearched": dict(PHASE_ONE_SEARCHED),
              "phaseOneL2": PHASE_ONE_L2, "phaseOneGrid": PHASE_ONE_ATTENTION[phase1_attention],
              "chosenTau": [r["phase1"]["temperature"] for r in rots],
              "phaseTwo": PHASE_TWO, "phaseTwoAggregation": PHASE_TWO_AGGREGATION,
              "bagsPerStep": cfg["schedule"]["bagsPerStep"],
              "oracle": "Eq. 21 on the step's target batch with the true target labels; diagnosis only",
              "seconds": round(time.time() - started), "rotationsMeta": meta,
              "phase1SourceTestPooled": _pooled_phase1(rots), "byObjective": by,
              "environment": environment}
    (workdir / "phase2-trace.json").write_text(json.dumps(record, default=str), encoding="utf-8")
    return {"ran": [transfer], "seconds": record["seconds"]}


def remote(config: Mapping, product=None, *, transfer: str, arm: str, objectives: Sequence[str],
           seed: int = 1, lr2: float | None = None, epochs2: int | None = None,
           phase1_searched: Mapping | None = None,
           rotations: Sequence[int] = (0, 1, 2, 3), workdir=None,
           environ: Mapping[str, str] | None = None) -> dict:
    """Remote entry: one trace with the given phase-2 step and budget, plus the run stamp."""
    environ = os.environ if environ is None else environ
    if lr2 is not None:
        PHASE_TWO["lr2"] = float(lr2)
    if epochs2 is not None:
        PHASE_TWO["epochs2"] = int(epochs2)
    PHASE_ONE_SEARCHED.clear()
    PHASE_ONE_SEARCHED.update({int(k): v for k, v in (phase1_searched or {}).items()})
    workdir = Path(workdir) if workdir else Path.cwd()
    out = smoke(config, product, transfer=transfer, arm=arm, objectives=list(objectives), seed=int(seed),
                rotations=rotations, workdir=workdir, environ=environ)
    mode = environ.get(remote_units.RUN_MODE_ENV)
    if mode:
        environment = json.loads((workdir / "phase2-trace.json").read_text(encoding="utf-8"))["environment"]
        payload = {"mode": mode,
                   "evidence": {"commit": environment["commit"], "sourcesDigest": environment["codeVersion"],
                                "codeVersion": environment["codeVersion"], "unit": transfer, "units": [transfer],
                                "complete": True, "gpu": None, "estimates": {}},
                   "environment": environment, "runs": {"ran": [transfer], "skipped": []}}
        (Path.cwd() / remote_units.STAMP_FILE).write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def _pooled_phase1(rots):
    return {"note": "per-rotation phase-1 source test metrics are in rotationsMeta[].phase1",
            "AUCmean": _s(np.mean([r["phase1"]["sourceTest"]["AUC"] for r in rots
                                   if r["phase1"]["sourceTest"]["AUC"] is not None] or [np.nan]))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--transfer", required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--objectives", default=",".join(OBJECTIVES))
    p.add_argument("--phase1-attention", choices=list(PHASE_ONE_ATTENTION), default="neutral")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rotations", default="0,1,2,3")
    p.add_argument("--epochs2", type=int, default=PHASE_TWO["epochs2"])
    p.add_argument("--lr2", type=float, default=PHASE_TWO["lr2"])
    p.add_argument("--config", default="tools/kaggle/rot8-task/run-config.json")
    p.add_argument("--workdir", required=True)
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    PHASE_TWO["epochs2"] = int(args.epochs2)
    PHASE_TWO["lr2"] = float(args.lr2)
    return smoke(local_config(args.config), transfer=args.transfer, arm=args.arm,
                 objectives=args.objectives.split(","), phase1_attention=args.phase1_attention,
                 seed=args.seed,
                 rotations=[int(r) for r in args.rotations.split(",")], workdir=args.workdir)


if __name__ == "__main__":
    main()
