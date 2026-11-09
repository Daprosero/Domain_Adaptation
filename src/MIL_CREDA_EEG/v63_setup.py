"""Experiments plan v63: the spectral encoder, the data views and MIL-CREDA as it runs.

Plan v63, Secs. 3.2, 4.3, 4.8, 5.3, 7.2, 7.3 and 7.7. Nothing in the v60 modules changes:
every v63 rule that differs from them is installed here, inside `installed()`, and only
while it is active.

**Instances (Sec. 7.3).** Each bag keeps at most `WINDOWS` = 60 windows, taken at a regular
step of ceil(n / 60) over its n useful windows; each window is filtered 1-45 Hz (4th-order
Butterworth, zero phase) and re-referenced to the common average of the channels the
transfer uses: the 12 channels EMOTIV and CGX share on a device transfer (`SHARED`), the
device's own channels otherwise. Channel selection comes before the average reference.

**Encoder (Sec. 7.3).** `spectral_trunk.SpectralTrunk`: log absolute and log relative band
power per channel in five bands, then one learnable linear map to 64 values. Its feature
standardization is fixed from one domain's TRAINING windows and stored as buffers
(`Standard`): a phase-1 training reads the standardization of its own source training bags
(an inner split's, in the search), and every trunk of phase 2, source and target, keeps the
phase-1 checkpoint's source standardization ("no se vuelve a estandarizar el objetivo").
No standardization reads a label.

**Phase 1 (Secs. 7.3, 7.7).** Every group searches `lr1` and an explicit L2 coefficient
(`PHASE_ONE_SPACE`, `phase_one_l2`), with wd1 = 0.
MIL-CREDA runs with neutral aggregation, gamma 0 and attention temperature 10; the common
wrapper and AdaptMIL keep their v60 aggregation (Sec. 7.2 leaves the baselines unchanged).
MIL-CREDA's sigma is the median distance between instances of DIFFERENT source training
bags, measured on the trained source trunk at the end of its phase 1 (`between_bag_sigma`);
it is the checkpoint's sigma, and phase 2 runs at exactly that value.

**MIL-CREDA phase 2 as run (Sec. 7.3).** gamma 0.5 and temperature 1 for source and target;
the target confidences (Eq. 24) enter Eq. 26 as weights without gradient; g enters the
correspondence (Eq. 35) without gradient; tau_loc of each target bag is the standard
deviation of its row of the source-target bag kernel; Eq. 22 cuts at the median of the
bridged model's scores on the unlabelled target training bags, computed once before the
first step; the objective adds mu ||theta_t - theta_t0||^2 (L2-SP, mu = 10) toward the
bridged target trunk; and the shift gate multiplies the adaptation terms (both etas) by
`gate_weight(AUC)` of the label-free domain classifier of Sec. 4.7 in that transfer and
outer fold. Only `lr2` and `wd2` are searched. These rules are thread-local: a phase-2
training of MIL-CREDA in one inner split never reads another split's threshold.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from optuna.distributions import FloatDistribution
from scipy.signal import butter, sosfiltfilt
from torch import nn

from MIL_CREDA_EEG import assignment as A
from MIL_CREDA_EEG import eegnet, mil_creda_model as M, mil_wrapper, model_store, partition, search
from MIL_CREDA_EEG import spectral_trunk, two_phase
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v63.md",
    "sections": ["4.3", "4.8", "5.3", "7.2", "7.3", "7.7"],
    "experiments": ["S0-codificador", "A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": [
        "a_bag_keeps_at_most_60_windows_at_step_ceil_n_over_60",
        "device_transfers_read_the_12_shared_channels_before_the_average_reference",
        "the_sixteen_transfers_of_sec_4_8",
        "the_trunk_standardization_is_fixed_and_reads_no_label",
        "the_gate_maps_auc_half_to_zero_and_one_to_one",
        "mil_creda_phase_two_confidences_and_g_carry_no_gradient",
        "tau_loc_is_the_kernel_row_standard_deviation",
        "the_pseudolabel_cut_is_the_bridged_median_computed_once",
        "the_l2_sp_term_is_mu_times_the_squared_distance_to_the_bridged_trunk",
        "the_phase_one_space_is_lr1_and_l2_and_the_mil_creda_phase_two_space_is_lr2_and_wd2",
        "every_arms_phase_one_adds_the_searched_explicit_l2",
        "the_between_bag_sigma_never_pairs_a_bag_with_itself",
    ],
}

REVISION = __provenance__["revision"]
WINDOWS = 60
BAND = (1.0, 45.0)
RATE = 128
SOS = butter(4, list(BAND), "bandpass", fs=RATE, output="sos")
#: Sec. 5.3: the 12 channels EMOTIV and CGX share.
SHARED = ("F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8")
DOMAINS = ("L1", "L6", "L11", "Robots", "CogniFit")
HIGH_DENSITY, LOW_COST = ("L1", "L6", "L11"), ("Robots", "CogniFit")

#: Sec. 4.8: the six core transfers and every other CGX <-> EMOTIV transfer, 16 in all.
CORE = (A.Transfer("level", "L1", "L11"), A.Transfer("level", "L11", "L1"),
        A.Transfer("task", "Robots", "CogniFit"), A.Transfer("task", "CogniFit", "Robots"),
        A.Transfer("device", "L6", "Robots"), A.Transfer("device", "Robots", "L6"))
TRANSFERS: tuple[A.Transfer, ...] = CORE + tuple(
    A.Transfer("device", s, t)
    for h in HIGH_DENSITY for e in LOW_COST for s, t in ((h, e), (e, h))
    if {h, e} != {"L6", "Robots"})
BY_NAME = {t.name: t for t in TRANSFERS}
#: Sec. 5.1-5.3: which experiment reads a transfer.
EXPERIMENT_OF_ARM = {"level": "A-nivel", "task": "B-tarea", "device": "C-dispositivo"}

#: Sec. 7.3, the table "MIL-CREDA tal como corre".
MU = 10.0
PHASE_ONE_AGGREGATION = {"attentionGamma": 0.0, "attentionTemperature": 10.0}
PHASE_TWO_AGGREGATION = {"attentionGamma": 0.5, "attentionTemperature": 1.0}
#: A phase-1 key `two_phase` requires and v63 never reads: the sigma of MIL-CREDA is the
#: between-bag median, not a multiple of the within-bag one (Sec. 7.7).
UNUSED_SIGMA_MULTIPLIER = 1.0
#: `two_phase.check_hyper` requires tauLocal; v63 replaces it by the per-bag row spread.
UNUSED_TAU_LOCAL = 1.0
#: Sec. 7.7: shared ranges, log scale.
#: Phase 1 searches lr1 and an explicit L2 coefficient added to the loss (`phase_one_l2`):
#: Adam's weight decay barely acts on a loss divided by the source bound (about 18.42), so
#: wd1 is fixed at 0 for every arm and the regularizer is the explicit penalty instead.
PHASE_ONE_SPACE = {"lr1": FloatDistribution(1e-4, 1e-1, log=True),
                   "l2": FloatDistribution(1e-4, 1e-1, log=True)}
PHASE_ONE_WD1 = 0.0
MIL_CREDA_PHASE_TWO_SPACE = {"lr2": FloatDistribution(1e-5, 1e-2, log=True),
                             "wd2": FloatDistribution(1e-7, 1e-2, log=True)}
SIGMA_PAIRS = 400


class V63Error(RuntimeError):
    """A v63 rule was asked for something the plan does not allow."""


def transfer(name: str) -> A.Transfer:
    try:
        return BY_NAME[name]
    except KeyError:
        raise V63Error(f"{name!r} is not one of the 16 transfers of Sec. 4.8") from None


def experiment_of(t: A.Transfer) -> str:
    return EXPERIMENT_OF_ARM[t.arm]


def channels_of(t: A.Transfer) -> tuple[str, ...] | None:
    """Sec. 5.3: the shared channels on a device transfer, every channel otherwise."""
    return SHARED if t.arm == "device" else None


# ------------------------------------------------------------------ instances


def window_step(n: int) -> int:
    return max(1, math.ceil(int(n) / WINDOWS))


def prepare(bag, channels: Sequence[str] | None = None) -> two_phase.BagTensor:
    """One raw `bag_data.Bag` -> its v63 BagTensor (Sec. 7.3)."""
    w = np.asarray(bag.windows() if callable(bag.windows) else bag.windows, dtype=np.float64)
    if channels is not None:
        names = list(bag.channels)
        missing = [c for c in channels if c not in names]
        if missing:
            raise V63Error(f"{bag.subject}/{bag.condition} lacks the shared channels {missing}")
        w = w[:, [names.index(c) for c in channels]]
    w = w[::window_step(len(w))][:WINDOWS]
    w = sosfiltfilt(SOS, w - w.mean(-1, keepdims=True), axis=-1)
    w = w - w.mean(1, keepdims=True)                       # common average reference
    label = bag.label if isinstance(bag.label, (int, np.integer)) else two_phase.CLASSES.index(bag.label)
    return two_phase.BagTensor(bag.subject, bag.condition, int(label),
                               torch.from_numpy(np.ascontiguousarray(w)).float())


class Prepared:
    """A cache of prepared bags per (subject, condition, channel set)."""

    def __init__(self, bags):
        self.bags = bags
        self._cache: dict = {}

    def __call__(self, raw: Sequence, channels: Sequence[str] | None) -> list:
        key_c = tuple(channels) if channels else None
        out = []
        for b in raw:
            key = (b.subject, b.condition, key_c)
            if key not in self._cache:
                self._cache[key] = prepare(b, channels)
            out.append(self._cache[key])
        return out


def rotation_key(rotations: Mapping, r: int):
    return r if r in rotations else str(r)


def search_data(t: A.Transfer, bags, prep: Prepared | None = None,
                rotations: Sequence[int] | None = None) -> search.SearchData:
    """`search.search_data` over the v63 bags: refit view and inner splits of each
    requested rotation; no test-fold bag is read."""
    prep = prep or Prepared(bags)
    a = A.assign_from_bags(t, bags)
    channels = channels_of(t)
    rotations = list(range(partition.N_FOLDS)) if rotations is None else [int(r) for r in rotations]
    side = lambda s, r, role, split=None: prep(A.side_bags(a, bags, s, r, role, split=split),
                                               channels)
    refit = {r: {"source_train": side("source", r, A.REFIT_ROLE), "source_valid": [],
                 "target_train": side("target", r, A.REFIT_ROLE), "target_valid": []}
             for r in rotations}
    inner = {(r, i): {"source_train": side("source", r, "inner_train", i),
                      "source_valid": side("source", r, "inner_validation", i),
                      "target_train": side("target", r, "inner_train", i),
                      "target_valid": side("target", r, "inner_validation", i)}
             for r in rotations for i in range(search.N_INNER)}
    folds = {r: v for r, v in search.rotation_folds(bags.rotations).items() if r in rotations}
    return search.SearchData(transfer=t, partition_sha256=a.partition_sha256,
                             rotations=refit, folds=folds, inner=inner)


def test_bags(t: A.Transfer, bags, rotation: int, prep: Prepared | None = None) -> dict:
    """The test fold of `rotation`: target bags (scored) and source bags (the source side
    of the domain classifier on bag representations, Sec. 4.4)."""
    prep = prep or Prepared(bags)
    a = A.assign_from_bags(t, bags)
    channels = channels_of(t)
    return {s: prep(A.side_bags(a, bags, s, rotation, "test"), channels)
            for s in ("source", "target")}


# ------------------------------------------------------------------ standardization


class Standard(nn.Module):
    """Fixed feature standardization; buffers only, so it travels in the state dict."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def feature_probe(channels: int) -> spectral_trunk.SpectralTrunk:
    """A trunk built by the unpatched constructor, for its fixed `features` only."""
    probe = spectral_trunk.SpectralTrunk.__new__(spectral_trunk.SpectralTrunk)
    _trunk_init(probe, channels)
    return probe


def standard_of(bags: Sequence[two_phase.BagTensor]) -> Standard:
    """Mean and standard deviation of the spectral features of `bags`' windows."""
    if not bags:
        raise V63Error("a standardization needs at least one training bag")
    probe = feature_probe(int(bags[0].windows.shape[1]))
    with torch.no_grad():
        f = torch.nan_to_num(torch.cat([probe.features(b.windows) for b in bags]))
    return Standard(f.mean(0), f.std(0) + 1e-6)


def standard_of_state(state: Mapping[str, torch.Tensor]) -> Standard:
    """The source standardization a stored phase-1 state carries."""
    try:
        return Standard(state["source_trunk.norm.mean"], state["source_trunk.norm.std"])
    except KeyError:
        raise V63Error("the phase-1 state carries no source standardization") from None


_NORM = threading.local()
_NORM_GLOBAL: dict = {"value": None}


def _current_norm() -> Standard | None:
    return getattr(_NORM, "value", None) or _NORM_GLOBAL["value"]


@contextlib.contextmanager
def standardization(norm: Standard, *, thread: bool = False) -> Iterator[None]:
    """Every SpectralTrunk built while the block runs carries `norm` (this thread only
    with `thread=True`, else every thread that set none of its own)."""
    holder = _NORM.__dict__ if thread else _NORM_GLOBAL
    saved = holder.get("value")
    holder["value"] = norm
    try:
        yield
    finally:
        holder["value"] = saved


_trunk_init = spectral_trunk.SpectralTrunk.__init__
_trunk_reset = spectral_trunk.SpectralTrunk.reset_parameters


def _v63_trunk_init(self, channels, samples=eegnet.SAMPLES):
    _trunk_init(self, channels, samples)
    norm = _current_norm()
    if norm is None:
        raise V63Error("a v63 spectral trunk is built only under a training-window "
                       "standardization (`standardization`)")
    if int(norm.mean.numel()) != 2 * len(spectral_trunk.BANDS) * int(channels):
        raise V63Error(f"the standardization holds {norm.mean.numel()} features; a "
                       f"{channels}-channel trunk reads {2 * len(spectral_trunk.BANDS) * channels}")
    self.norm = Standard(norm.mean, norm.std)


def _v63_trunk_reset(self):
    self.project.reset_parameters()


# ------------------------------------------------------------------ MIL-CREDA as run


def gate_weight(domain_auc: float | None) -> float:
    """Sec. 7.3: linear from AUC 0.5 (weight 0) to 1 (full weight), clipped."""
    if domain_auc is None or not math.isfinite(float(domain_auc)):
        raise V63Error("the shift gate needs the domain classifier's AUC of Sec. 4.7")
    return float(min(1.0, max(0.0, (float(domain_auc) - 0.5) / 0.5)))


def between_bag_sigma(model, bags: Sequence[two_phase.BagTensor], domain: str = "source",
                      pairs: int = SIGMA_PAIRS, seed: int = 0) -> float:
    """Median distance between instances of two different bags, `pairs` random pairs."""
    if len(bags) < 2:
        raise V63Error("the between-bag sigma needs two bags")
    g = torch.Generator().manual_seed(int(seed))
    trunk = model.trunk(domain)
    with torch.no_grad():
        device = next(iter(trunk.parameters()), torch.empty(0)).device
        H = [trunk(b.windows.to(device)).double().cpu() for b in bags]
    d = []
    while len(d) < pairs:
        i, j = torch.randint(len(H), (2,), generator=g).tolist()
        if i == j:
            continue
        a = H[i][torch.randint(len(H[i]), (1,), generator=g)]
        b = H[j][torch.randint(len(H[j]), (1,), generator=g)]
        d.append(float(torch.cdist(a, b)))
    value = float(np.median(d))
    if not value > 0:
        raise V63Error(f"the between-bag median distance is {value}")
    return value


_STATE = threading.local()
_conf, _corr, _pl, _obj = (M.confidences, M.total_correspondence, M.pseudolabel,
                           two_phase.phase_two_objective)


def _active() -> dict | None:
    return getattr(_STATE, "value", None)


def _detached_confidences(G):
    return _conf(G).detach() if _active() else _conf(G)


def row_tau(cross: torch.Tensor) -> float:
    """Sec. 7.3: tau_loc of a target bag, the spread of its row of the bag kernel."""
    return float(cross.detach().std()) + 1e-6


def _correspondence(cross, labels, g, tau):
    if _active():
        return _corr(cross, labels, g.detach(), row_tau(cross))
    return _corr(cross, labels, g, tau)


def _pseudolabel(g):
    s = _active()
    return int(float(g[1]) >= s["threshold"]) if s else _pl(g)


def _objective(terms, term_selection=two_phase.BOTH_TERMS):
    s = _active()
    out = _obj(terms, term_selection)
    if not s:
        return out
    out = s["gate"] * out
    prox = sum(((p - s["theta0"][n]) ** 2).sum()
               for n, p in s["model"].target_trunk.named_parameters())
    return out + s["mu"] * prox.to(out.dtype)


@contextlib.contextmanager
def mil_creda_rules(model, threshold: float, gate: float, mu: float = MU) -> Iterator[None]:
    """The phase-2 rules of MIL-CREDA as run, for this thread's training."""
    theta0 = {n: p.detach().clone() for n, p in model.target_trunk.named_parameters()}
    saved = _active()
    _STATE.value = {"threshold": float(threshold), "gate": float(gate), "mu": float(mu),
                    "model": model, "theta0": theta0}
    try:
        yield
    finally:
        _STATE.value = saved


def probabilities(model, bags, domain: str, device) -> np.ndarray:
    G = two_phase.evaluate(model, bags, domain, torch.device(device))
    return np.nan_to_num(G[:, 1].detach().cpu().numpy().astype(float), nan=0.5)


def mil_creda_hyper(meta_hyper: Mapping, params: Mapping, schedule: Mapping) -> dict:
    if set(params) != set(MIL_CREDA_PHASE_TWO_SPACE):
        raise V63Error(f"MIL-CREDA phase 2 searches exactly {list(MIL_CREDA_PHASE_TWO_SPACE)} "
                       f"(Sec. 7.7); got {sorted(params)}")
    return two_phase.check_hyper({**{k: meta_hyper[k] for k in two_phase.PHASE_ONE_KEYS},
                                  **PHASE_TWO_AGGREGATION, **params,
                                  "tauLocal": UNUSED_TAU_LOCAL,
                                  **search.check_schedule(schedule)})


def train_mil_creda_phase_two(data: search.SearchData, rotation: int, seed: int, record,
                              params: Mapping, schedule: Mapping, device="cpu", *,
                              gate: float, term_selection: str = two_phase.BOTH_TERMS,
                              gamma: float | None = None) -> dict:
    """MIL-CREDA's v63 phase 2 from one phase-1 checkpoint (state, meta). The K3 variants
    of Sec. 6.3 change only `term_selection` (global or local term only) or `gamma` (the
    own aggregation without consensus, gamma 0 in phase 2 at the same temperature)."""
    state, meta = record
    sigma1 = meta.get("sigma")
    if sigma1 is None:
        raise V63Error("the MIL-CREDA phase-1 checkpoint carries no sigma")
    device = torch.device(device)
    hyper = mil_creda_hyper(meta["hyper"], params, schedule)
    d = data.rotations[rotation]
    channels = data.channels()
    with standardization(standard_of_state(state), thread=True):
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        model = model_store.build_model("MIL-CREDA", channels["source"], channels["target"],
                                        hyper).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("target_trunk.") for k in missing):
        raise V63Error(f"the phase-1 state does not fit (missing {missing}, unexpected "
                       f"{unexpected})")
    model.fix_sigma(sigma1)
    model.attention_gamma = (PHASE_TWO_AGGREGATION["attentionGamma"] if gamma is None
                             else float(gamma))
    model.attention_temperature = PHASE_TWO_AGGREGATION["attentionTemperature"]
    joint = two_phase.bridge(model)
    threshold = float(np.median(probabilities(model, d["target_train"], "target", device)))
    with mil_creda_rules(model, threshold, gate):
        result = two_phase.adapt_target_phase(
            model, {"checkpoint": {"sigma": sigma1}}, joint["frozen_state"], d["source_train"],
            d["target_train"], d["target_valid"], hyper, generator, device, seed=seed,
            fold=rotation, transfer=data.transfer.name, term_selection=term_selection)
    result["checkpoint"].update({"pseudolabelThreshold": threshold, "gate": gate, "mu": MU,
                                 "termSelection": term_selection,
                                 "attentionGamma": model.attention_gamma})
    return {"strategy": "MIL-CREDA", "model": model, "result": result, "hyper": hyper,
            "channels": channels}


def phase_one_hyper(group: str, params: Mapping, schedule: Mapping) -> dict:
    """The v63 phase-1 key set of `group`: lr1 searched, wd1 = 0 (the searched L2 enters
    the loss through `phase_one_l2`, never the optimizer)."""
    if set(params) != set(PHASE_ONE_SPACE):
        raise V63Error(f"phase 1 searches exactly {list(PHASE_ONE_SPACE)}; got {sorted(params)}")
    if group == "MIL-CREDA":
        fixed = {**PHASE_ONE_AGGREGATION, "sigmaMultiplier": UNUSED_SIGMA_MULTIPLIER}
    elif group in ("common", "AdaptMIL"):
        fixed = dict(mil_wrapper.COMMON_PHASE_ONE_FIXED)
    else:
        raise V63Error(f"{group!r} is not a phase-1 group")
    nan = float("nan")
    return two_phase.check_hyper({**fixed, "lr1": params["lr1"], "wd1": PHASE_ONE_WD1,
                                  "lr2": nan, "wd2": nan, "tauLocal": nan,
                                  **search.check_schedule(schedule)})


def l2_penalty(model, lam: float) -> torch.Tensor:
    """lam x the squared weights of the source trunk and the head, as `explicit_l2`."""
    weights = [p for n, p in model.source_trunk.named_parameters() if "weight" in n]
    return lam * sum((w ** 2).sum() for w in (*weights, model.head.linear.weight))


@contextlib.contextmanager
def phase_one_l2(lam: float) -> Iterator[None]:
    """The searched explicit L2 of every arm's phase 1: `phase_one_sweep.explicit_l2` for
    the common group and MIL-CREDA (`two_phase.source_term`), and the same penalty on
    AdaptMIL's own phase-1 objective. Entered once per trial or refit, in the calling
    thread; the inner splits of a trial share its value."""
    lam = float(lam)
    if not lam >= 0:
        raise V63Error(f"the L2 coefficient is a nonnegative number; got {lam!r}")
    from MIL_CREDA_EEG.comparators.adaptmil import training as adaptmil_training
    plain = adaptmil_training.stage_two_objective

    def penalized(model, bags, labels):
        return plain(model, bags, labels) + l2_penalty(model, lam)

    adaptmil_training.stage_two_objective = penalized
    try:
        with explicit_l2(lam):
            yield
    finally:
        adaptmil_training.stage_two_objective = plain


def train_phase_one(group: str, data: search.SearchData, rotation: int, seed: int,
                    hyper: Mapping, device="cpu", *, sigma: bool = True) -> dict:
    """`search.train_phase_one` under the standardization of this view's source training
    bags; MIL-CREDA then takes the between-bag sigma of its trained source trunk (a search
    trial, at gamma 0, never reads sigma and skips it with `sigma=False`)."""
    src = data.rotations[rotation]["source_train"]
    with standardization(standard_of(src), thread=True):
        out = search.train_phase_one(group, data, rotation, seed, hyper, device)
    if group == "MIL-CREDA" and sigma:
        model = out["model"]
        sigma = between_bag_sigma(model, src, seed=seed)
        model._sigma = sigma                        # replaces the unused within-bag value
        out["result"]["checkpoint"]["sigma"] = sigma
        out["result"]["checkpoint"]["state"] = two_phase.phase_one_state(model)
        out["result"]["sigma"] = {"sigma": sigma, "rule": "median distance between instances "
                                  "of different source training bags (plan v63, Sec. 7.7)"}
    return out


@contextlib.contextmanager
def installed() -> Iterator[None]:
    """The v63 encoder and rules, active while the block runs."""
    patches = [(M, "confidences", _detached_confidences),
               (M, "total_correspondence", _correspondence), (M, "pseudolabel", _pseudolabel),
               (two_phase, "phase_two_objective", _objective),
               (spectral_trunk.SpectralTrunk, "__init__", _v63_trunk_init),
               (spectral_trunk.SpectralTrunk, "reset_parameters", _v63_trunk_reset)]
    saved = [(o, n, getattr(o, n)) for o, n, _ in patches]
    with spectral_trunk.installed():
        for o, n, v in patches:
            setattr(o, n, v)
        try:
            yield
        finally:
            for o, n, v in saved:
                setattr(o, n, v)


# ------------------------------------------------------------------ store


TRAINING_MODULES = ("v63_campaign.py", "v63_filters.py", "v63_records.py", "v63_setup.py")


def code_version() -> str:
    """The v60 code version and the bytes of the v63 modules that train or record
    (`TRAINING_MODULES`); the analysis modules read records and are left out, so adding or
    editing one never changes the identity of a study or a stored model."""
    digest = hashlib.sha256(model_store.code_version().encode())
    for name in TRAINING_MODULES:
        path = Path(__file__).parent / name
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def save_state(path: Path, state: Mapping[str, torch.Tensor], meta: Mapping) -> Path:
    """A state dict and its JSON sidecar (revision v63, partition hash, code version)."""
    path = Path(path)
    if not meta.get(partition.HASH_FIELD):
        raise V63Error("a model is stored with the hash of its partition (Sec. 7.3)")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().clone() for k, v in state.items()}
    tmp = path.with_suffix(".pt.partial")
    torch.save(state, tmp)
    tmp.replace(path)
    sidecar = {**meta, "revision": REVISION, "code_version": code_version(),
               "state_sha256": model_store.state_digest(state)}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=1, sort_keys=True,
                                                    default=str), encoding="utf-8")
    return path


def load_state(path: Path, partition_sha256: str) -> tuple[dict, dict]:
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get(partition.HASH_FIELD) != partition_sha256:
        raise partition.PartitionMismatch(f"{path} was trained on another partition")
    if meta.get("revision") != REVISION:
        raise V63Error(f"{path} was written under {meta.get('revision')!r}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if model_store.state_digest(state) != meta["state_sha256"]:
        raise V63Error(f"{path}: the state does not match its sidecar")
    return state, meta
