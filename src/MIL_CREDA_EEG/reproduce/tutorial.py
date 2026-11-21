"""Helpers of the two MIL-CREDA usage tutorials (not a paper stage: nothing here is merged or rendered).

* `MIL-CREDA/Notebooks/Toyset.ipynb` trains MIL-CREDA on the three 3-D toy families (lunas, toroide,
  rollo). The toy points need an instance encoder of their own (`PointTrunk`, a small MLP), so phase 1
  and phase 2 are assembled here from the paper's own training functions with three classes:
  `two_phase.train_source_phase` and the between-bag bandwidth of `phase2_trace.phase_one`, then
  `phase2_trace.traced_path` with the paper's objective "eq39+hardloc" under the annealed learning rate
  of `lr_schedule`, exactly as `xpilot.phase2_trial` and `criteria_pool.train` call them. The phase-2
  configurations are the paper's seeded random draws (`criteria_pool.draw`) and the label-free pick is
  cef (`phase2_select.cef_score`, first maximum).
* `MIL-CREDA-EEG/Notebooks/00_Tutorial_EEG.ipynb` runs the paper's code unchanged on one simulated
  unit (`xpilot.build_sides`, `xcompare.shared_base`, `criteria_pool.train`) and on one cross-subject
  split of the public IEEE ADHD data.

The paper's values quoted below come from the phase-1 bases and pick manifests the paper campaigns ran
with (archived branch `archive/ieee-pool-slim`, commit 7cd5ba19: `MIL-CREDA-EEG/Manifests/oracle_bases/`,
`ieee_pool_bases/`, `MIL-CREDA-EEG/Results/v82-pool24/milcreda_picks.json` and the tracked
`MIL-CREDA-EEG/Manifests/attention_influence/ieee_cef_picks.json`), so the tutorials need no binary
base file: they refit phase 1 at those settings.
"""
from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn

from MIL_CREDA_EEG import criteria_pool as C
from MIL_CREDA_EEG import eegnet, lr_schedule, mechanisms, phase2_select, repro, spectral_trunk, two_phase
from MIL_CREDA_EEG import oracle_screen as O
from MIL_CREDA_EEG import phase2_trace as P
from MIL_CREDA_EEG import pool_aggregation as PA
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG import v63_setup as V
from MIL_CREDA_EEG import xcompare as XC
from MIL_CREDA_EEG import xpilot as XP
from MIL_CREDA_EEG.mil_creda_model import MILCREDAModel
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2

DEVICE = torch.device("cpu")
#: the schedule block of every paper base's config (batches of eight bags; the epoch counts are set per run)
CONFIG = {"schedule": {"bagsPerStep": 8, "epochs1": 100, "epochs2": 100}}
#: the paper's phase-2 pool: 100 epochs per configuration, seed-1 random draws (`criteria_pool`)
PAPER_EPOCHS2 = C.EPOCHS2
PAPER_TRIALS = C.TRIALS
POOL_SEED = PA.PICK_SEED

#: The synthetic unit of the paper's illustrations (d_C = 1.5, shift 1.5, rotation 0). Its phase-1
#: settings, cut and MIL-CREDA cef pick are those of the paper's base `d1p5-s1p5-r0-s1.pt` and of
#: `v82-pool24/milcreda_picks.json` (trial 34 of the seed-1 draw).
SIM_UNIT = {
    "key": "d1p5-s1p5-r0", "rotation": 0,
    "kwargs": {"setting": "device", "d_target": 1.5, "shift": 1.5, "artifact_rate": 3.0,
               "event_fraction": 0.2, "paired": True, "n_subjects": 60},
    "phase1": {"lr1": 0.004515240679619132, "l2": 0.010735442724225793, "width": 32,
               "tauAtt1": 0.6982789675027884, "epochs": 50},
    "paperCut": 0.5603971519698578,
    "paperPick": {"trial": 34, "BApool": 0.85833, "AUCpool": 0.94222, "BAtest12": 0.91667},
}

#: One cross-subject split of IEEE ADHD/Control Children (rotation 3, half 1). Phase-1 settings and cut
#: from the paper's base `ieee-r3h1-s1.pt`; the cef pick from `ieee_cef_picks.json`.
IEEE_UNIT = {
    "key": "ieee-r3h1", "rotation": 3, "half": 1,
    "phase1": {"lr1": 0.001910635062186717, "l2": 0.0036966191380306624, "width": 128,
               "tauAtt1": 0.9084702910860815, "epochs": 100},
    "paperCut": 0.13641737654690683,
    #: cef of trials 0-21 in the paper's ADHD pool (jobs ix-r3h1-s*, `MIL-CREDA-<trial>.npz`, argmax of Pt)
    "paperCef": (1.9723, 1.9723, 1.8899, 1.9723, 1.9723, 1.9723, 1.9723, 1.9723, 1.9723, 1.9723, 1.5692,
                 1.9723, 1.9723, 1.9723, 1.9723, 1.9723, 1.9723, 1.7548, 1.9723, 1.9723, 1.9723, 2.0),
}
IEEE_PICKS = repro.eeg_dir() / "Manifests" / "attention_influence" / "ieee_cef_picks.json"


# ------------------------------------------------------------------ EEG: the paper's code, unchanged


def ieee_pick(unit_key: str = IEEE_UNIT["key"]) -> dict:
    """The paper's MIL-CREDA cef pick of an ADHD unit: trial, BA, AUC and kappa on the test half."""
    return dict(json.loads(Path(IEEE_PICKS).read_text())["units"][unit_key])


def pool_params(trial: int, seed: int = POOL_SEED) -> dict:
    """Configuration `trial` of the paper's seeded phase-2 pool of MIL-CREDA."""
    return dict(C.draw("MIL-CREDA", seed)[int(trial)])


def sim_unit(eval_subjects: int = O.EVAL_SUBJECTS) -> dict:
    """The simulated unit's subject-wise sides and its pool of fresh evaluation subjects (true labels
    only the simulator has). Simulating 120 + 120 subjects takes about a minute on a laptop CPU."""
    kw = SIM_UNIT["kwargs"]
    built = XP.build_sides(CONFIG, {}, experiment="sim", rotation=SIM_UNIT["rotation"], half=0, **kw)
    pool = O.eval_pool({"unitKwargs": kw}, built["effect"], int(eval_subjects))
    return {**built, "pool": pool, "rotation": SIM_UNIT["rotation"]}


def ieee_unit(root: str | None = None) -> dict:
    """The IEEE split's sides; the data come from the public Kaggle dataset (`repro.ieee_root`)."""
    root = root or repro.ieee_root()
    built = XP.build_sides(CONFIG, {"IEEE_ROOT": root}, experiment="ieee",
                           rotation=IEEE_UNIT["rotation"], half=IEEE_UNIT["half"])
    return {**built, "rotation": IEEE_UNIT["rotation"], "root": root}


@contextlib.contextmanager
def running(unit: dict):
    """The contexts every paper run trains under: the spectral trunk, the unit's window length and the
    light phase-2 trace (as `criteria_pool._setup` enters them)."""
    with spectral_trunk.installed(), XP.window_samples(unit["samples"]), XP.full_trace_mode(False):
        yield


def phase_one(unit: dict, params1: dict, seed: int = 1) -> dict:
    """Phase 1 at the given settings: the cut fitted on an inner split, the source model refitted on all
    source training subjects (`xcompare.shared_base`), and its bridged copy that predicts the target
    with no adaptation. Returns the base record `criteria_pool.train` reads."""
    b = XC.shared_base(unit["sides"], unit["cfg"], DEVICE, int(seed), int(unit["rotation"]), dict(params1),
                       unit["unit"])
    return {**b, "phase1": dict(params1), "rotation": int(unit["rotation"])}


def phase_two(unit: dict, base: dict, params: dict, train_seed: int, epochs: int = PAPER_EPOCHS2):
    """MIL-CREDA's phase 2 from `base` at `params`, exactly as the paper's pool trains one trial
    (`criteria_pool.train`: a dry run counts the optimizer steps, then the annealed full run).
    The attention is trained; every other phase-1 part stays frozen and is checked."""
    with PA.attention_moves(base["base"]):
        model, _ = C.train("MIL-CREDA", dict(params), base, unit["sides"], unit["cfg"], DEVICE,
                           int(train_seed), int(unit["rotation"]), unit["unit"], int(epochs))
    return model


def scores(model, bags, cut: float) -> dict:
    """Subject-level BA at the source cut, AUC and Cohen's kappa on the target path."""
    with torch.no_grad():
        s = XC.snapshot(model, bags, float(cut), DEVICE)
    return {k: s[k] for k in ("BA", "AUC", "kappa")}


def cef(model, bags, domain: str = "target") -> float:
    """The label-free criterion: effective number of predicted classes on the unlabeled bags."""
    with torch.no_grad():
        pseudo = two_phase.evaluate(model, bags, domain, DEVICE).argmax(1).cpu().numpy()
    return float(phase2_select.cef_score(pseudo))


def attention(model, bag, domain: str = "target") -> np.ndarray:
    """The attention weights beta of one bag's windows (they sum to one)."""
    with torch.no_grad():
        return np.asarray(mechanisms.attention_weights(model, bag.windows, domain), dtype=float)


def masks(bag) -> dict:
    """The simulator's per-window ground truth of a simulated bag: artifact and class-event masks."""
    from MIL_CREDA_EEG import eeg_sim as E
    m = E.SIM_MASKS[bag.subject]
    return {"artifact": np.asarray(m["artifact"], bool), "artifactType": list(m["artifactType"]),
            "event": np.asarray(m["event"], bool)}


def illustration_subject(bags):
    """The paper's rule, fixed before reading any weight: the first evaluation subject with ADHD whose
    bag holds class events and both artifact types (blinks and muscle)."""
    for b in bags:
        if int(b.label) != 1:
            continue
        m = masks(b)
        kinds = {k for k, a in zip(m["artifactType"], m["artifact"]) if a and k}
        if m["event"].any() and {"blink", "muscle"} <= kinds:
            return b
    raise ValueError("no ADHD subject with class events and both artifact types")


def mask_aucs(weights, bag) -> dict:
    """AUC of the inverted weight against the artifact mask and of the weight against the event mask
    (`pool_aggregation.mask_aucs`); 0.5 is uniform weighting."""
    m = masks(bag)
    return PA.mask_aucs(weights, m["artifact"], m["event"], int(bag.label))


# ------------------------------------------------------------------ toy set: 3-D points, three classes

FAMILIES = ("lunas", "toroide", "rollo")
TOY_CLASSES = 3
#: the toy instance encoder: 3 -> HIDDEN -> width, tanh
HIDDEN = 32
#: tutorial phase-1 settings for the toy set, inside the paper's phase-1 ranges (learning rate 1e-4 to
#: 1e-2, l2 1e-4 to 1e-1, width {32, 64, 128}, attention temperature 0.01 to 1, epochs {50, 100, 200});
#: fixed rather than searched (the paper runs 20 TPE trials over three inner splits)
TOY_PHASE1 = {"lr1": 1e-2, "l2": 1e-3, "width": 32, "tauAtt1": 0.5, "epochs": 100}
#: the target encoder starts as a copy of the source one, with the target's own label-free
#: standardization (`phase2_trace.initialise_target` arm "copy-own"): the toy target is a translated and
#: rotated copy of the source, and the standardization removes the translation without any label
TOY_ARM = "copy-own"


class PointTrunk(nn.Module):
    """(n, 3) points -> (n, width): a fixed standardization (`norm`, set from the source training
    points as `phase2_trace.phase_one` sets the spectral one) and a small MLP (`project`). It keeps the
    trunk interface the paper's model and trainers read."""

    def __init__(self, channels: int, samples: int = 1, width: int = 32, hidden: int = HIDDEN):
        super().__init__()
        self.channels, self.samples, self.output_dim = int(channels), int(samples), int(width)
        self.register_buffer("masks", torch.zeros(1), persistent=False)   # device probe of standard_of
        self.norm = sc.Standard(torch.zeros(self.channels), torch.ones(self.channels))
        self.project = nn.Sequential(nn.Linear(self.channels, hidden), nn.Tanh(), nn.Linear(hidden, width))

    def constrain(self) -> None:
        """No constraint."""

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(len(x), -1).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.norm(self.features(x)))


@contextlib.contextmanager
def points_installed(width: int):
    """The paper's model builds `PointTrunk`s of `width` while the block runs (as
    `spectral_trunk.installed` swaps in the spectral trunk); the bridge copies `project`."""

    class Trunk(PointTrunk):
        def __init__(self, channels, samples=1):
            super().__init__(channels, samples, width=int(width))

    names = {"EEGNetTrunk": Trunk, "LAYERS": ("project",), "CHANNEL_DEPENDENT": (), "HEAD_MAX_NORM": 1.0}
    saved = {k: getattr(eegnet, k) for k in names}
    for k, v in names.items():
        setattr(eegnet, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(eegnet, k, v)


def toy_bags(points, labels, bag_ids, prefix: str) -> list:
    """One `BagTensor` per (class, band) of a toy domain: the band of a class is its subject."""
    out = []
    for c in sorted(set(int(x) for x in labels)):
        for k in sorted(set(int(x) for x in bag_ids[labels == c])):
            sel = (labels == c) & (bag_ids == k)
            out.append(two_phase.BagTensor(f"{prefix}-c{c}-b{k:02d}", prefix, int(c),
                                           torch.from_numpy(np.ascontiguousarray(points[sel])).float()))
    return out


def toy_unit(name: str, seed: int = 0, rotation: int = 0) -> dict:
    """One toy family's two domains cut into bags and split subject-wise as the paper's synthetic
    units are (`xpilot.cross_dataset_sides`: the source keeps one fifth of its bags per class for test;
    the target is split in five folds per class, `rotation` tests one and adapts on the rest)."""
    from MIL_CREDA_Benchmark import toy_run
    fam = toy_run.build_family(name, int(seed), toy_run.PILOT)
    src = toy_bags(fam.Xs, fam.ys, fam.bs, "SRC")
    tgt = toy_bags(fam.Xt, fam.yt, fam.bt, "TGT")
    sides = XP.cross_dataset_sides(src, tgt, int(rotation))
    return {"name": name, "family": fam, "sides": sides, "rotation": int(rotation),
            "names": [f"ch{i}" for i in range(3)]}


def _toy_hyper(params1: dict) -> dict:
    return P._hyper(CONFIG, temperature=float(params1["tauAtt1"]), searched=params1)


def toy_phase_one(unit: dict, params1: dict = TOY_PHASE1, seed: int = 1) -> dict:
    """`phase2_trace.phase_one` with three classes and the point trunk: L_src on the source for
    `epochs` epochs with the explicit l2 penalty, then the kernel width fixed at the median distance
    between instances of different source bags. Also returns the bridged, target-initialised copy
    (the target encoder starts as a copy of the source one), whose prediction is the no-adaptation
    reference."""
    source = unit["sides"]["source"]
    hyper = _toy_hyper(params1)
    torch.manual_seed(int(seed))
    g = torch.Generator().manual_seed(int(seed))
    with points_installed(params1["width"]):
        model = MILCREDAModel(3, 3, attention_gamma=hyper["attentionGamma"],
                              attention_temperature=hyper["attentionTemperature"], classes=TOY_CLASSES)
        model.source_trunk.norm = sc.standard_of(model.source_trunk, source)
        with explicit_l2(float(params1["l2"])):
            history = two_phase.train_source_phase(model, source, [], hyper, g, DEVICE, seed=int(seed))["history"]
        model._sigma = V.between_bag_sigma(model, source, "source", seed=int(seed))
        prepared = copy.deepcopy(model)
        two_phase.bridge(prepared)
        P.initialise_target(prepared, TOY_ARM, unit["sides"]["target"], unit["names"], unit["names"])
    return {"base": model, "prepared": prepared, "phase1": dict(params1), "sigma": model.sigma,
            "loss": [h["loss"] for h in history]}


def _toy_fit(base: dict, unit: dict, params: dict, epochs: int, seed: int):
    """`xpilot.phase2_trial`'s training on the toy model: a copy of the base, the trial's aggregation
    settings, then `phase2_trace.traced_path` with "eq39+hardloc" (light: reads no target label)."""
    p = dict(params)
    wd2, saved = p.pop("wd2"), P.PHASE_TWO["wd2"]
    P.PHASE_TWO["wd2"] = float(wd2)
    sides = unit["sides"]
    try:
        with XP.phase2_settings(**p, epochs2=int(epochs)), points_installed(base["phase1"]["width"]):
            hyper = _toy_hyper(base["phase1"])
            model = copy.deepcopy(base["base"])
            model.attention_gamma = P.PHASE_TWO_AGGREGATION["gamma"]
            model.attention_temperature = P.PHASE_TWO_AGGREGATION["temperature"]
            model.pooling = P.PHASE_TWO_AGGREGATION.get("pooling", "attention")
            P.traced_path(model, sides["source"], sides["target"], None, hyper, int(seed), DEVICE,
                          XP.OBJECTIVE, 0.0,
                          lambda m: P.initialise_target(m, TOY_ARM, sides["target"], unit["names"], unit["names"]),
                          {"sigma": base["sigma"]}, int(epochs), P.EQUIVALENCE_EPOCHS,
                          log=lambda *a, **k: None, light=True)
    finally:
        P.PHASE_TWO["wd2"] = saved
    return model


def toy_phase_two(base: dict, unit: dict, params: dict, epochs: int, seed: int):
    """One toy phase-2 run as `criteria_pool.train` runs one trial: a dry run of `criteria_pool.DRY_EPOCHS`
    epochs counts the optimizer steps at lr2, then the full run anneals lr2 (1 + 10p)^-0.75."""
    lr2 = float(params["lr2"])
    with PA.attention_moves(base["base"]):
        with lr_schedule.count_steps(lr2) as counts:
            _toy_fit(base, unit, params, C.DRY_EPOCHS, seed)
        totals = [max(1, round(c * int(epochs) / C.DRY_EPOCHS)) for c in counts] or [1]
        with lr_schedule.annealed(lr2, totals):
            return _toy_fit(base, unit, params, epochs, seed)


def toy_scores(model, bags, domain: str) -> dict:
    """Balanced accuracy (argmax over the three classes) and accuracy of a domain's bags."""
    with torch.no_grad():
        G = two_phase.evaluate(model, bags, domain, DEVICE).cpu().numpy()
    y, pred = np.array([int(b.label) for b in bags]), G.argmax(1)
    return {"BA": float(balanced_accuracy_score(y, pred)), "accuracy": float(np.mean(y == pred))}


def representations(model, bags, domain: str) -> np.ndarray:
    """Bag representations z (the classifier's input), one row per bag."""
    with torch.no_grad():
        return two_phase.represent(model, bags, domain, DEVICE).cpu().numpy()


def toy_pool(unit: dict, base: dict, trials: int, epochs: int, seed: int = POOL_SEED) -> list[dict]:
    """The paper's selection protocol on one toy family: the first `trials` configurations of the
    paper's seeded phase-2 draw, each trained `epochs` epochs with the paper's training seed
    (`criteria_pool.train_seed`); per trial its cef on the unlabeled target bags and, for reading only,
    its target-test balanced accuracy. Non-finite trials are kept and flagged, as in the paper."""
    rows = []
    for i, p in enumerate(C.draw("MIL-CREDA", int(seed), int(trials))):
        try:
            model = toy_phase_two(base, unit, p, epochs, C.train_seed(seed, i))
        except FloatingPointError:
            rows.append({"trial": i, "params": p, "nonFinite": True, "cef": None})
            continue
        rows.append({"trial": i, "params": p, "nonFinite": False, "model": model,
                     "cef": cef(model, unit["sides"]["target"]),
                     "test": toy_scores(model, unit["sides"]["target_test"], "target")})
    return rows


def cef_pick(rows) -> int:
    """Index of the label-free pick: the first maximum of cef (a non-finite trial never wins)."""
    return phase2_select.first_best([r["cef"] for r in rows])
