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


def phase_one(net: Net, xs, ys, dev, g, epochs: int) -> list[float]:
    params = list(net.trunk.parameters()) + list(net.head.parameters())
    opt = torch.optim.Adam(params, lr=FIXED["lr1"], weight_decay=0.0)
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


def phase_two(net: Net, xs, ys, xt, dev, g, epochs: int, sigma: float) -> tuple:
    trunk_t = copy.deepcopy(net.trunk)
    for p in trunk_t.parameters():
        p.requires_grad_(True)
    theta0 = {n: p.detach().clone() for n, p in trunk_t.named_parameters()}
    threshold = float(np.median(_scores(net, trunk_t, xt, dev)))            # bridged, once
    opt = torch.optim.Adam(trunk_t.parameters(), lr=FIXED["lr2"], weight_decay=FIXED["wd2"])
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
