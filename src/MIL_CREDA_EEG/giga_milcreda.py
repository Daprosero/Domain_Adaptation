"""Pilot: MIL-CREDA without its MIL part, on GiGA (Cho et al. 2017) trials, cross-subject.

Diagnostic outside any plan. The encoder (EEGNet, 320 samples = 2.5-5 s at 128 Hz) hands its
instances straight to MIL-CREDA's terms: there is no bag, no attention, no consensus and no
local term. What stays is every piece of MIL-CREDA that is not MIL, called from the same
`MIL_CREDA` functions the bag version calls:

EEGNet is exactly `eegnet.py`'s EEGNet-8,2, nothing added (no input standardization, its own
max-norm constraints, no explicit L2, no BatchNorm recalibration).

Phase 1, the source alone: Eq. 21 normalized by its bound (`source_loss`); Adam at its default
rate; then the encoder and head frozen. sigma is the median distance between source training
instances on the trained encoder (v63's between-bag rule with each instance its own unit),
fixed from here on.

Phase 2, only the target encoder moves (a copy of the source one, made once), only through the
adaptation term: the instance Gaussian kernel (Eq. 2/14) for K_ss, K_st, K_tt; the frozen
head's posteriors of the target instances (Eq. 20 on instances); their confidences (Eq. 24),
detached as in v63; pseudo-labels (Eq. 22) cut at the median of the bridged posteriors over the
target training instances, computed once (v63); the class blocks (Eq. 23), their weighting
(Eq. 26), the activity test (Eq. 27), the balanced mixture (Eq. 28), the score (Eq. 29), the
cosh class loss (Eq. 32) and the global term (Eq. 33); plus v63's L2-SP anchor mu ||theta_t -
theta_t0||^2. Gate 1 (no shift gate value exists here).

Values fixed before the run, the v64 pilot's: lr1 1e-3, 50 epochs; lr2 1e-3, wd2 1e-5, 20
epochs; mu 10; batches of 64 instances. Partition: 5 subject folds (subject i in fold
i % 5); rotation r targets fold r, the other 40 subjects are the source; each target subject's
trials split by index parity, one half adapts, the other is tested, then they swap.
Statistics are per subject.
"""

from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from MIL_CREDA.conditional import conditional_blocks, mixed_matrix, weighted_blocks
from MIL_CREDA.confidence import confidences, pseudolabel
from MIL_CREDA.global_term import (balanced_dependency_score, balanced_mixture,
                                   class_global_loss_cosh, global_loss, is_active)
from MIL_CREDA.kernels import gaussian_kernel
from MIL_CREDA.objective import source_loss
from MIL_CREDA.renyi import quadratic_entropy
from MIL_CREDA_EEG import eegnet, remote_units
from MIL_CREDA_EEG.giga_pilot import FOLDS, find_data, load
from MIL_CREDA_EEG.mil_creda_model import EPSILON_SOURCE, one_hot

RESULTS = Path("Results") / "giga-milcreda"
FIXED = {"lr1": 1e-3, "lr2": 1e-3, "wd2": 1e-5, "mu": 10.0}
EPOCHS1, EPOCHS2, BATCH, SIGMA_PAIRS = 50, 20, 64, 4000


class Net(torch.nn.Module):
    def __init__(self, channels: int, samples: int):
        super().__init__()
        self.trunk = eegnet.EEGNetTrunk(channels, samples)
        self.head = eegnet.EEGNetHead(self.trunk.output_dim, 2)

    def posteriors(self, x, trunk=None):
        z = (trunk or self.trunk)(x).to(torch.float64)
        return z, self.head(z)


def _batches(n: int, size: int, g: torch.Generator):
    order = torch.randperm(n, generator=g)
    return [order[i:i + size] for i in range(0, n, size) if len(order[i:i + size]) > 1]


def _scores(net: Net, trunk, x: torch.Tensor, dev) -> np.ndarray:
    trunk.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            out.append(net.posteriors(x[i:i + 256].to(dev), trunk)[1][:, 1].cpu())
    return torch.cat(out).numpy()


def phase_one(net: Net, xs, ys, dev, g, epochs: int, lr: float | None = None,
              on_epoch=None) -> list[float]:
    params = list(net.trunk.parameters()) + list(net.head.parameters())
    opt = torch.optim.Adam(params, lr=FIXED["lr1"] if lr is None else lr, weight_decay=0.0)
    curve = []
    for _ in range(epochs):
        net.train(); losses = []
        for idx in _batches(len(xs), BATCH, g):
            _, G = net.posteriors(xs[idx].to(dev))
            loss = source_loss(G, one_hot(ys[idx].to(dev), 2), EPSILON_SOURCE)
            opt.zero_grad(); loss.backward(); opt.step()
            net.trunk.constrain(); net.head.constrain()
            losses.append(float(loss.detach()))
        curve.append(float(np.mean(losses)))
        if on_epoch is not None:
            net.eval(); on_epoch(net)
    net.eval()
    for p in params:
        p.requires_grad_(False)
    return curve


def between_instance_sigma(trunk, xs, dev, g) -> float:
    with torch.no_grad():
        Z = torch.cat([trunk(xs[i:i + 256].to(dev)).to(torch.float64) for i in range(0, len(xs), 256)])
    a = torch.randint(0, len(Z), (SIGMA_PAIRS,), generator=g)
    b = torch.randint(0, len(Z), (SIGMA_PAIRS,), generator=g)
    keep = a != b
    value = float((Z[a[keep]] - Z[b[keep]]).norm(dim=1).median())
    if not value > 0:
        raise ValueError(f"the between-instance median distance is {value}")
    return value


def global_term(f_s, y_s, f_t, G_t, sigma: float, threshold: float) -> tuple[torch.Tensor, int]:
    """Eqs. 22-33 on instances: MIL-CREDA's global term with no bag in it."""
    K_ss = gaussian_kernel(f_s, f_s, sigma)
    K_st = gaussian_kernel(f_s, f_t, sigma)
    K_tt = gaussian_kernel(f_t, f_t, sigma)
    w_t = confidences(G_t).detach()                                        # Eq. 24, v63
    pseudo = (G_t[:, 1].detach() >= threshold).long()                       # Eq. 22, v63 cut
    counts = [int((y_s == c).sum()) for c in range(2)]
    losses = []
    for c in range(2):
        s_idx = torch.nonzero(y_s == c).reshape(-1)
        t_idx = torch.nonzero(pseudo == c).reshape(-1)
        if s_idx.numel() == 0 or t_idx.numel() == 0:
            losses.append(None); continue
        K_s_c, K_st_c, K_t_c = conditional_blocks(K_ss, K_st, K_tt, s_idx, t_idx)   # Eq. 23
        K_st_w, K_t_w = weighted_blocks(K_st_c, K_t_c, w_t[t_idx])                   # Eq. 26
        if not is_active(K_s_c, K_t_w, mixed_matrix(K_s_c, K_st_w, K_t_w)):          # Eq. 27
            losses.append(None); continue
        breve_s, breve_t, _ = balanced_mixture(K_s_c, K_st_w, K_t_w)                 # Eq. 28
        score = balanced_dependency_score(K_s_c, K_st_w, K_t_w)                      # Eq. 29
        losses.append(class_global_loss_cosh(score, quadratic_entropy(breve_s),
                                             quadratic_entropy(breve_t)))           # Eq. 32
    return global_loss(losses, counts), sum(l is not None for l in losses)          # Eq. 33


def phase_two(net: Net, xs, ys, xt, dev, g, epochs: int, sigma: float,
              lr: float | None = None, wd: float | None = None, on_epoch=None) -> tuple:
    trunk_t = copy.deepcopy(net.trunk)
    for p in trunk_t.parameters():
        p.requires_grad_(True)
    theta0 = {n: p.detach().clone() for n, p in trunk_t.named_parameters()}
    threshold = float(np.median(_scores(net, trunk_t, xt, dev)))            # bridged, once
    opt = torch.optim.Adam(trunk_t.parameters(), lr=FIXED["lr2"] if lr is None else lr,
                           weight_decay=FIXED["wd2"] if wd is None else wd)
    per_class = [torch.nonzero(ys == c).reshape(-1) for c in range(2)]
    curve = []
    for _ in range(epochs):
        trunk_t.train(); vals, active = [], []
        for j in _batches(len(xt), BATCH, g):
            half = len(j) // 2                                              # both classes present
            i = torch.cat([pc[torch.randint(0, len(pc), (max(1, half),), generator=g)]
                           for pc in per_class])
            with torch.no_grad():
                f_s = net.trunk(xs[i].to(dev)).to(torch.float64)
            f_t, G_t = net.posteriors(xt[j].to(dev), trunk_t)
            term, n_active = global_term(f_s, ys[i].to(dev), f_t, G_t, sigma, threshold)
            prox = sum(((p - theta0[n]) ** 2).sum() for n, p in trunk_t.named_parameters())
            loss = term + FIXED["mu"] * prox.to(term.dtype)
            opt.zero_grad(); loss.backward(); opt.step(); trunk_t.constrain()
            vals.append(float(term.detach())); active.append(n_active)
        curve.append({"global": float(np.mean(vals)), "activeClasses": float(np.mean(active))})
        if on_epoch is not None:
            trunk_t.eval(); on_epoch(trunk_t)
    trunk_t.eval()
    return trunk_t, threshold, curve


def run(config: Mapping, product: str | Path | None = None, *, rotations: Sequence[int],
        seeds: Sequence[int] = (1,), epochs1: int = EPOCHS1, epochs2: int = EPOCHS2,
        data: str | None = None, subjects: Sequence[int] | None = None,
        workdir: str | Path | None = None, environ: Mapping[str, str] | None = None) -> dict:
    environ = os.environ if environ is None else environ
    product = Path(product) if product else Path.cwd()
    workdir = Path(workdir) if workdir else Path.cwd()
    placement = remote_units.device_placement(config, environ)
    deterministic = remote_units.set_deterministic()
    identity = {"commit": remote_units.source_commit(environ), "revision": "giga-milcreda"}
    environment = remote_units.environment_stamp(deterministic, identity, placement)
    dev = torch.device(config.get("device", "cpu"))
    subj = load(find_data(data))
    group = {s: i % FOLDS for i, s in enumerate(sorted(subj))}
    ids = sorted(subj) if subjects is None else [s for s in sorted(subj) if s in set(subjects)]
    started, done = time.time(), []
    try:
        for r in rotations:
            tgt = [s for s in ids if group[s] == r]
            src = [s for s in ids if group[s] != r]
            xs = np.concatenate([subj[s][0] for s in src]); ys_np = np.concatenate([subj[s][1] for s in src])
            norm = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            xs_t, ys = norm(xs), torch.from_numpy(ys_np)
            C, T = xs.shape[1], xs.shape[2]
            for seed in seeds:
                path = product / RESULTS / f"rotation{r}__seed{seed}.json"
                if path.is_file():
                    continue
                t0 = time.time()
                torch.manual_seed(seed); np.random.seed(seed)
                g = torch.Generator().manual_seed(seed)
                net = Net(C, T).to(dev)
                curve1 = phase_one(net, xs_t, ys, dev, g, epochs1)
                sigma = between_instance_sigma(net.trunk, xs_t, dev, g)
                rows, log = [], {"phase1": curve1, "sigma": sigma, "phase2": {}}
                # source-only: the frozen phase-1 model on every target trial
                for s in tgt:
                    for half in (0, 1):
                        x = norm(subj[s][0][half::2])
                        rows += [{"subject": s, "arm": "phase1", "half": half, "y": int(yy),
                                  "p": float(pp)} for yy, pp in
                                 zip(subj[s][1][half::2], _scores(net, net.trunk, x, dev))]
                for half in (0, 1):
                    xa = norm(np.concatenate([subj[s][0][half::2] for s in tgt]))
                    trunk_t, threshold, curve2 = phase_two(net, xs_t, ys, xa, dev, g, epochs2, sigma)
                    log["phase2"][half] = {"threshold": threshold, "curve": curve2}
                    for s in tgt:
                        x = norm(subj[s][0][1 - half::2])
                        rows += [{"subject": s, "arm": "milCreda", "half": 1 - half, "y": int(yy),
                                  "p": float(pp)} for yy, pp in
                                 zip(subj[s][1][1 - half::2], _scores(net, trunk_t, x, dev))]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"rotation": r, "seed": seed, "target": tgt,
                                            "fixed": FIXED, "epochs1": epochs1, "epochs2": epochs2,
                                            "seconds": round(time.time() - t0, 1), "rows": rows,
                                            "log": log, "environment": environment}, default=float))
                done.append(f"{r}/{seed}")
                print(r, seed, "sigma", round(sigma, 3), "seconds", round(time.time() - t0, 1), flush=True)
    finally:
        pass
    timing = {"seconds": round(time.time() - started, 1), "device": environment.get("device")}
    mode = environ.get(remote_units.RUN_MODE_ENV)
    if mode:
        workdir.mkdir(parents=True, exist_ok=True)
        payload = {"mode": mode, "evidence": {"commit": environment.get("commit"),
                                              "unit": f"giga-mc:{list(rotations)}",
                                              "units": [f"giga-mc:{r}" for r in rotations],
                                              "complete": True, "estimates": {}},
                   "environment": environment, "timing": timing, "runs": {"ran": done}}
        (workdir / remote_units.STAMP_FILE).write_text(json.dumps(payload, indent=1, default=str))
    return {"ran": done, "timing": timing}


# ---------------------------------------------------------------- the short tuned test
#
# Nested double search, no target label read. Inside outer rotation r the 40 source subjects
# split once: fold (r + 1) % 5 is the INNER TARGET (10 labelled source subjects acting as new
# subjects), the other 30 the inner source. Phase 1 searches lr1 (and its epoch count) by the
# inner target's accuracy; phase 2 then searches lr2, wd2 (and its epoch count), adapting on
# one parity half of the inner target and scoring the other, both ways, exactly as outside.
# TPE with a fixed seed, the same number of trials per phase; mu stays 10 (v63). The phase-1
# baseline is the phase-1 model at ITS tuned values, the one MIL-CREDA adapts from. The
# selected values are refitted on all 40 source subjects and scored on the outer target.

RESULTS_TUNED = Path("Results") / "giga-milcreda-tuned"
SPACE1 = {"lr1": (1e-4, 1e-2)}
SPACE2 = {"lr2": (1e-5, 1e-2), "wd2": (1e-7, 1e-2)}


def _acc(net: Net, trunk, x, y, dev) -> float:
    return float(((_scores(net, trunk, x, dev) > .5) == y.numpy()).mean())


def run_tuned(config: Mapping, product: str | Path | None = None, *, rotations: Sequence[int],
              seed: int = 1, trials: int = 8, epochs1: int = EPOCHS1, epochs2: int = EPOCHS2,
              data: str | None = None, workdir: str | Path | None = None,
              environ: Mapping[str, str] | None = None) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    environ = os.environ if environ is None else environ
    product = Path(product) if product else Path.cwd()
    workdir = Path(workdir) if workdir else Path.cwd()
    placement = remote_units.device_placement(config, environ)
    deterministic = remote_units.set_deterministic()
    identity = {"commit": remote_units.source_commit(environ), "revision": "giga-milcreda-tuned"}
    environment = remote_units.environment_stamp(deterministic, identity, placement)
    dev = torch.device(config.get("device", "cpu"))
    subj = load(find_data(data))
    ids = sorted(subj)
    group = {s: i % FOLDS for i, s in enumerate(ids)}
    arr = lambda sids, half=None: (
        torch.from_numpy(np.ascontiguousarray(np.concatenate(
            [subj[s][0] if half is None else subj[s][0][half::2] for s in sids]), dtype=np.float32)),
        torch.from_numpy(np.concatenate(
            [subj[s][1] if half is None else subj[s][1][half::2] for s in sids])))
    started, done = time.time(), []
    for r in rotations:
        path = product / RESULTS_TUNED / f"rotation{r}__seed{seed}.json"
        if path.is_file():
            continue
        t0 = time.time()
        tgt = [s for s in ids if group[s] == r]
        src = [s for s in ids if group[s] != r]
        q = (r + 1) % FOLDS
        itgt = [s for s in src if group[s] == q]
        isrc = [s for s in src if group[s] != q]
        xs_i, ys_i = arr(isrc)
        xv, yv = arr(itgt)
        C, T = xs_i.shape[1], xs_i.shape[2]

        def fresh():
            torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
            return Net(C, T).to(dev), g

        # phase 1: lr1 and its epochs by inner-target accuracy
        log1 = []
        def obj1(trial):
            lr1 = trial.suggest_float("lr1", *SPACE1["lr1"], log=True)
            net, g = fresh(); series = []
            phase_one(net, xs_i, ys_i, dev, g, epochs1, lr=lr1,
                      on_epoch=lambda n: series.append(_acc(n, n.trunk, xv, yv, dev)))
            best = int(np.argmax(series))
            trial.set_user_attr("epochs", best + 1)
            log1.append({"lr1": lr1, "series": series})
            return series[best]
        st1 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        st1.optimize(obj1, n_trials=trials)
        lr1, ep1 = st1.best_params["lr1"], st1.best_trial.user_attrs["epochs"]

        # phase 2: lr2, wd2 and epochs, adapting on one inner-target half, scoring the other
        net, g = fresh()
        phase_one(net, xs_i, ys_i, dev, g, ep1, lr=lr1)
        sigma_i = between_instance_sigma(net.trunk, xs_i, dev, g)
        halves = [arr(itgt, h) for h in (0, 1)]
        log2 = []
        def obj2(trial):
            lr2 = trial.suggest_float("lr2", *SPACE2["lr2"], log=True)
            wd2 = trial.suggest_float("wd2", *SPACE2["wd2"], log=True)
            per_half = []
            for h in (0, 1):
                series = []
                xe, ye = halves[1 - h]
                phase_two(net, xs_i, ys_i, halves[h][0], dev, torch.Generator().manual_seed(seed),
                          epochs2, sigma_i, lr=lr2, wd=wd2,
                          on_epoch=lambda tt: series.append(_acc(net, tt, xe, ye, dev)))
                per_half.append(series)
            series = list(np.mean(per_half, axis=0))
            best = int(np.argmax(series))
            trial.set_user_attr("epochs", best + 1)
            log2.append({"lr2": lr2, "wd2": wd2, "series": series})
            return series[best]
        st2 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        st2.optimize(obj2, n_trials=trials)
        lr2, wd2, ep2 = st2.best_params["lr2"], st2.best_params["wd2"], st2.best_trial.user_attrs["epochs"]

        # refit on all 40 source subjects, score the outer target
        xs_t, ys = arr(src)
        net, g = fresh()
        phase_one(net, xs_t, ys, dev, g, ep1, lr=lr1)
        sigma = between_instance_sigma(net.trunk, xs_t, dev, g)
        rows = []
        for s in tgt:
            for h in (0, 1):
                x = torch.from_numpy(np.ascontiguousarray(subj[s][0][h::2], dtype=np.float32))
                rows += [{"subject": s, "arm": "phase1", "half": h, "y": int(yy), "p": float(pp)}
                         for yy, pp in zip(subj[s][1][h::2], _scores(net, net.trunk, x, dev))]
        for h in (0, 1):
            xa, _ = arr(tgt, h)
            trunk_t, threshold, curve2 = phase_two(net, xs_t, ys, xa, dev, g, ep2, sigma, lr=lr2, wd=wd2)
            for s in tgt:
                x = torch.from_numpy(np.ascontiguousarray(subj[s][0][1 - h::2], dtype=np.float32))
                rows += [{"subject": s, "arm": "milCreda", "half": 1 - h, "y": int(yy), "p": float(pp)}
                         for yy, pp in zip(subj[s][1][1 - h::2], _scores(net, trunk_t, x, dev))]
        sel = {"lr1": lr1, "epochs1": ep1, "innerAcc1": st1.best_value,
               "lr2": lr2, "wd2": wd2, "epochs2": ep2, "innerAcc2": st2.best_value}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rotation": r, "seed": seed, "target": tgt, "innerTarget": itgt,
                                    "trials": trials, "selection": sel, "mu": FIXED["mu"],
                                    "seconds": round(time.time() - t0, 1), "rows": rows,
                                    "log": {"phase1": log1, "phase2": log2, "sigma": sigma},
                                    "environment": environment}, default=float))
        done.append(r)
        print(r, sel, "seconds", round(time.time() - t0, 1), flush=True)
    timing = {"seconds": round(time.time() - started, 1), "device": environment.get("device")}
    mode = environ.get(remote_units.RUN_MODE_ENV)
    if mode:
        workdir.mkdir(parents=True, exist_ok=True)
        payload = {"mode": mode, "evidence": {"commit": environment.get("commit"),
                                              "unit": f"giga-tuned:{list(rotations)}",
                                              "units": [f"giga-tuned:{r}" for r in rotations],
                                              "complete": True, "estimates": {}},
                   "environment": environment, "timing": timing, "runs": {"ran": done}}
        (workdir / remote_units.STAMP_FILE).write_text(json.dumps(payload, indent=1, default=str))
    return {"ran": done, "timing": timing}
