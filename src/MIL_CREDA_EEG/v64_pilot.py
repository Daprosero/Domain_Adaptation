"""Pilot before plan v64: can EEGNet enter MIL-CREDA with sub-bag augmentation?

Diagnostic outside any plan: it reports no result of v63 and writes only under
`Results/v64pilot-vanilla`. Everything is plan v63 (subject-wise partition, phase-2 rules,
between-bag sigma, L2-SP, shift gate) except the trunk, which is EEGNet exactly as
`eegnet.py` defines it (EEGNet-8,2 of the official code) with nothing added: no input
standardization, its own max-norm constraints, no explicit L2, no BatchNorm recalibration.
Hyperparameters are fixed before the run (no search): Adam at lr1 1e-3 (its default),
50 phase-1 epochs; lr2 1e-3, wd2 1e-5, 20 phase-2 epochs.

Three data arms, each split subject-wise (all units of a subject on its own side):
  whole   the v63 bags (<= 60 windows at step ceil(n / 60)).
  sub15   each v63 bag cut into random sub-bags of 15 windows (seeded order).
  blocks  windows with 50% overlap (every stored 4 s window plus the half-shifted window
          between contiguous pairs), cut into ordered contiguous blocks of 15 windows,
          <= 20 per subject spread evenly over the session, a bridge window left out between
          blocks so no two blocks share a sample.
Every unit counts as a "subject" in training, adaptation and test; the analysis aggregates by
the real subject. MIL-CREDA runs with the v63 shift gate of that transfer and rotation (stored
v63 values) and with the gate forced to 1.

A unit id is "<transfer>:<rotation>", e.g. "Robots->L11:0".
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from scipy.signal import sosfiltfilt

from MIL_CREDA_EEG import eegnet, mil_creda_model as M, remote_units, search, two_phase
from MIL_CREDA_EEG import v63_campaign as C
from MIL_CREDA_EEG import v63_setup as V

RESULTS = Path("Results") / "v64pilot-vanilla"
ARMS = ("whole", "sub15", "blocks")
SIZE, CAP = 15, 20
#: EEGNet as originally defined, nothing added: Adam at its default rate in phase 1 and no
#: explicit L2; lr2 and wd2 are MIL-CREDA's own phase-2 values.
FIXED = {"lr1": 1e-3, "l2": 0.0, "lr2": 1e-3, "wd2": 1e-5}
EPOCHS1, EPOCHS2 = 50, 20
BAGS_PER_STEP = 8
#: The v63 shift gate per rotation, read from the stored v63 MIL-CREDA records (seed 1).
GATE_V63 = {
    "Robots->CogniFit": [0.056213017751479244, 0.5292307692307692, 0.0, 0.3538461538461539],
    "CogniFit->Robots": [0.056213017751479244, 0.5292307692307692, 0.0, 0.3538461538461539],
    "L11->Robots": [0.9162257495590829, 0.9926131117266852, 0.9815980629539951, 0.9392014519056262],
    "Robots->L11": [0.9162257495590829, 0.9926131117266852, 0.9815980629539951, 0.9392014519056262],
}


# ------------------------------------------------------------------ data arms


def _label(bag) -> int:
    return int(bag.label if isinstance(bag.label, (int, np.integer))
               else two_phase.CLASSES.index(bag.label))


def overlapped(bag, channels=None) -> torch.Tensor:
    """Every stored window plus the half-shifted window of each contiguous pair, filtered
    and average-referenced as v63 does one window."""
    with np.load(bag.path, allow_pickle=False) as z:
        w = z["signal"].astype(np.float64)
        start = z["window_start_s"]
    if channels is not None:
        names = list(bag.channels)
        w = w[:, [names.index(c) for c in channels]]
    half, out = w.shape[-1] // 2, []
    for k in range(len(w)):
        out.append(w[k])
        if k + 1 < len(w) and abs(start[k + 1] - start[k] - 4.0) < 1e-6:
            out.append(np.concatenate([w[k][:, half:], w[k + 1][:, :half]], axis=-1))
    w = np.stack(out)
    w = sosfiltfilt(V.SOS, w - w.mean(-1, keepdims=True), axis=-1)
    w = w - w.mean(1, keepdims=True)
    return torch.from_numpy(np.ascontiguousarray(w)).float()


def blocks_of(w: torch.Tensor, subject: str, condition: str, label: int,
              size: int = SIZE, cap: int = CAP) -> list:
    """Ordered contiguous blocks of `size` overlapped windows, <= `cap` spread evenly, a
    bridge window left out between consecutive blocks so none shares a sample."""
    m = w.shape[0]
    k = min(cap, (m + 1) // (size + 1))
    starts = np.round(np.linspace(0, m - size, k)).astype(int) if k > 1 else np.array([0])
    if k > 1 and np.diff(starts).min() < size + 1:
        raise V.V63Error("two blocks would share a sample")
    return [two_phase.BagTensor(f"{subject}#{i}", condition, int(label), w[s:s + size])
            for i, s in enumerate(starts)]


def ordered_blocks(bag, channels=None, size: int = SIZE, cap: int = CAP) -> list:
    return blocks_of(overlapped(bag, channels), bag.subject, bag.condition, _label(bag),
                     size, cap)


class BlockPrepared(V.Prepared):
    """`V.Prepared` that expands each raw bag into its ordered blocks."""

    def __call__(self, raw, channels):
        key_c = tuple(channels) if channels else None
        out = []
        for b in raw:
            key = (b.subject, b.condition, key_c)
            if key not in self._cache:
                self._cache[key] = ordered_blocks(b, channels)
            out.extend(self._cache[key])
        return out


def random_subbags(bags, size: int, generator: torch.Generator) -> list:
    out = []
    for b in bags:
        order = torch.randperm(b.windows.shape[0], generator=generator)
        for i, s in enumerate(range(0, len(order) - size + 1, size)):
            out.append(two_phase.BagTensor(f"{b.subject}#{i}", b.condition, b.label,
                                           b.windows[order[s:s + size]]))
    return out


def standardize(view: dict, test: list) -> list:
    """In place: every bag of the view, and the test bags, scaled per channel by the mean
    and std of the source training windows (no target statistic, no label)."""
    x = torch.cat([b.windows for b in view["source_train"]])
    mean, std = x.mean((0, 2), keepdim=True)[0], x.std((0, 2), keepdim=True)[0].clamp_min(1e-6)
    scale = lambda g: [two_phase.BagTensor(b.subject, b.condition, b.label,
                                           (b.windows - mean) / std) for b in g]
    for k in list(view):
        view[k] = scale(view[k])
    return scale(test)


# ------------------------------------------------------------------ v63 rules with EEGNet


@contextlib.contextmanager
def eegnet_rules():
    """The v63 phase-2 rules without the spectral encoder; the spectral feature
    standardization becomes a no-op. EEGNet itself is left exactly as defined."""
    nothing = contextlib.nullcontext
    patches = [(M, "confidences", V._detached_confidences),
               (M, "total_correspondence", V._correspondence), (M, "pseudolabel", V._pseudolabel),
               (two_phase, "phase_two_objective", V._objective),
               (V, "standardization", lambda norm, thread=False: nothing()),
               (V, "standard_of", lambda bags: None), (V, "standard_of_state", lambda s: None)]
    saved = [(o, n, getattr(o, n)) for o, n, _ in patches]
    for o, n, v in patches:
        setattr(o, n, v)
    try:
        yield
    finally:
        for o, n, v in saved:
            setattr(o, n, v)


def _rows(model, bags, domain: str, device) -> list[dict]:
    p = V.probabilities(model, bags, domain, device)
    return [{"subject": b.subject.split("#")[0], "unit": b.subject, "label": int(b.label),
             "p": float(q)} for b, q in zip(bags, p)]


def run_one(t, rotation: int, arm: str, seed: int, bags, preps: Mapping, device,
            epochs1: int = EPOCHS1, epochs2: int = EPOCHS2) -> dict:
    started = time.time()
    data = V.search_data(t, bags, preps[arm == "blocks"], rotations=[rotation])
    test = V.test_bags(t, bags, rotation, preps[arm == "blocks"])["target"]
    view = data.rotations[rotation]
    if arm == "sub15":
        g = torch.Generator().manual_seed(1000 + seed)
        for k in ("source_train", "target_train"):
            view[k] = random_subbags(view[k], SIZE, g)
        test = random_subbags(test, SIZE, g)
    out = {"transfer": t.name, "rotation": rotation, "arm": arm, "seed": seed,
           "trunk": None, "fixed": FIXED, "epochs1": epochs1, "epochs2": epochs2,
           "gateV63": GATE_V63[t.name][rotation],
           "nUnits": {"source": len(view["source_train"]), "target": len(view["target_train"]),
                      "test": len(test)}}
    gates = (("milCreda", GATE_V63[t.name][rotation]), ("milCredaGate1", 1.0))
    return train_and_score(data, view, test, rotation, seed, device, epochs1, epochs2, gates,
                           out, started)


def train_and_score(data, view: Mapping, test: list, rotation: int, seed: int, device,
                    epochs1: int, epochs2: int, gates, out: dict, started: float) -> dict:
    """Phase 1 of the common group (predict) and of MIL-CREDA, then MIL-CREDA's phase 2 at
    each (label, gate) of `gates`; every prediction on `test` lands in `out`."""
    sched1 = {"epochs1": epochs1, "epochs2": epochs2, "bagsPerStep": BAGS_PER_STEP}
    record = None
    for group in ("common", "MIL-CREDA"):
        hyper = V.phase_one_hyper(group, {"lr1": FIXED["lr1"], "l2": FIXED["l2"]}, sched1)
        try:
            with V.phase_one_l2(FIXED["l2"]):
                trained = V.train_phase_one(group, data, rotation, seed, hyper, device)
        except (FloatingPointError, V.V63Error) as error:
            # a collapsed encoder (between-bag sigma 0) is a result, recorded, never retried
            failed = f"failed: {type(error).__name__}: {error}"
            for key in (("predict",) if group == "common"
                        else ("phase1", *(label for label, _ in gates))):
                out[key] = failed
            if group == "common":
                continue
            out["seconds"] = round(time.time() - started, 1)
            return out
        model = trained["model"]
        out["trunk"] = type(model.source_trunk).__name__
        key = "predict" if group == "common" else "phase1"
        out[key] = _rows(model.eval(), test, "source", device)
        out[f"{key}Train"] = _rows(model, view["source_train"], "source", device)
        if group == "MIL-CREDA":
            sigma = V.between_bag_sigma(model, view["source_train"], seed=seed)
            model._sigma = sigma
            record = (two_phase.phase_one_state(model), {"hyper": dict(hyper), "sigma": sigma})
    for label, gate in gates:
        saved, search.PHASE_ONE_SEED = search.PHASE_ONE_SEED, seed
        try:
            o2 = V.train_mil_creda_phase_two(data, rotation, seed, record,
                                             {"lr2": FIXED["lr2"], "wd2": FIXED["wd2"]},
                                             sched1, device, gate=gate)
            out[label] = _rows(o2["model"].eval(), test, "target", device)
        except (FloatingPointError, V.V63Error) as error:   # recorded, never retried
            out[label] = f"failed: {type(error).__name__}: {error}"
        finally:
            search.PHASE_ONE_SEED = saved
    out["seconds"] = round(time.time() - started, 1)
    return out


def record_path(product: Path, t, rotation: int, arm: str, seed: int) -> Path:
    return (product / RESULTS / t.name.replace("->", "-to-") / f"rotation{rotation}"
            / f"{arm}__seed{seed}.json")


def run(config: Mapping, product: str | Path | None = None, *, units: Sequence[str],
        seeds: Sequence[int] = (1, 2, 3), arms: Sequence[str] = ARMS,
        epochs1: int = EPOCHS1, epochs2: int = EPOCHS2, workdir: str | Path | None = None,
        environ: Mapping[str, str] | None = None) -> dict:
    """Every (unit, arm, seed) in order, resumable: a stored record is skipped. Stops before
    the session deadline (`FORGE_SESSION_DEADLINE`) with the step time measured so far."""
    environ = os.environ if environ is None else environ
    product = Path(product) if product else C.DEFAULT_PRODUCT
    workdir = Path(workdir) if workdir else Path.cwd()
    config = {"schedule": {"epochs1": epochs1, "epochs2": epochs2,
                           "bagsPerStep": BAGS_PER_STEP}, **config}
    cfg = C._setup(config, environ)
    deadline = float(environ.get(remote_units.DEADLINE_ENV, "inf"))
    placement = remote_units.device_placement(cfg, environ)
    deterministic = remote_units.set_deterministic()
    identity = {"commit": remote_units.source_commit(environ), "codeVersion": V.code_version(),
                "revision": "v64-pilot"}
    environment = remote_units.environment_stamp(deterministic, identity, placement)
    device = cfg["device"]
    bags = C.campaign.load_bags(cfg)
    preps = {False: V.Prepared(bags), True: BlockPrepared(bags)}
    ran, skipped, longest, started = [], [], 0.0, time.time()
    todo = [(u, a, s) for u in units for a in arms for s in seeds]
    with eegnet_rules():
        for k, (u, arm, seed) in enumerate(todo):
            name, rotation = u.rsplit(":", 1)
            t, rotation = V.transfer(name), int(rotation)
            path = record_path(product, t, rotation, arm, seed)
            if path.is_file():
                skipped.append(f"{u}/{arm}/{seed}")
                continue
            if time.time() + 1.5 * longest + C.DEFAULT_MARGIN > deadline:
                break
            out = run_one(t, rotation, arm, seed, bags, preps, device, epochs1, epochs2)
            longest = max(longest, out["seconds"])
            C.write_record(path, {**out, "environment": environment})
            ran.append(f"{u}/{arm}/{seed}")
            print(u, arm, seed, out["trunk"], out["nUnits"], f"{out['seconds']}s", flush=True)
    remaining = [f"{u}/{a}/{s}" for u, a, s in todo
                 if f"{u}/{a}/{s}" not in ran and f"{u}/{a}/{s}" not in skipped]
    timing = {"longestStepSeconds": longest, "seconds": round(time.time() - started, 1),
              "device": environment.get("device")}
    mode = environ.get(remote_units.RUN_MODE_ENV)
    if mode:
        workdir.mkdir(parents=True, exist_ok=True)
        payload = {"mode": mode,
                   "evidence": {"commit": environment.get("commit"),
                                "sourcesDigest": identity["codeVersion"],
                                "codeVersion": identity["codeVersion"], "unit": units[0],
                                "units": list(units), "complete": not remaining,
                                "estimates": {}},
                   "environment": environment, "timing": timing,
                   "runs": {"ran": ran, "skipped": skipped, "remaining": remaining}}
        (workdir / remote_units.STAMP_FILE).write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8")
    return {"ran": ran, "skipped": skipped, "remaining": remaining, "timing": timing}
