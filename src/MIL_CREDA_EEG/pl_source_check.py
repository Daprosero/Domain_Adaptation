"""Where phase-2 pseudo-labels could come from: the source head or the target's own structure.

Diagnostic outside the plan's protocol. Per rotation, phase 1 is trained as in
`phase2_trace` (fixed tau 10, gamma 0) and the target path is initialised with the arm. On
every target subject (training and test pooled), with the trunk frozen:

* head: the frozen source head's argmax, and its 30% most confident subjects;
* centroid: nearest source class centroid in the bag representation (cosine);
* cluster: spherical k-means on the target, seeded with the source centroids (SHOT/CAN style);
* ceiling: a 5-fold logistic regression on the target representation with the TRUE target
  labels. It bounds what any label-free rule can recover from this representation.

Only the ceiling reads target labels for fitting; every other rule reads them for scoring.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from MIL_CREDA_EEG import assignment as A
from MIL_CREDA_EEG import campaign, remote_units, spectral_trunk, two_phase
from MIL_CREDA_EEG import learned_phase1 as L
from MIL_CREDA_EEG import phase2_trace as P

CONF_SHARE = 0.3
KMEANS_ITERATIONS = 20


def representations(model, bags, side, device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pairs = model.bag_pairs(model.embed([b.windows.to(device) for b in bags], side))
        return model.represent(pairs).detach().cpu().numpy().astype(np.float64)


def _unit(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def spherical_kmeans(z, centroids, iterations=KMEANS_ITERATIONS):
    zn, c = _unit(z), _unit(centroids)
    assign = np.argmax(zn @ c.T, axis=1)
    for _ in range(iterations):
        new = np.stack([zn[assign == k].mean(0) if np.any(assign == k) else c[k] for k in range(len(c))])
        c = _unit(new)
        nxt = np.argmax(zn @ c.T, axis=1)
        if np.array_equal(nxt, assign):
            break
        assign = nxt
    return assign


def _score(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return {"acc": round(float(np.mean(y == pred)), 3),
            "BA": round(float(balanced_accuracy_score(y, pred)), 3),
            "positive": round(float(np.mean(pred == 1)), 3)}


def run_rotation(transfer, cfg, a, bags, device, seed, rotation, arm):
    prep = lambda side, role: L._filtered_whole(A.side_bags(a, bags, side, rotation, role),
                                                car=True, reject=None, channels=None)
    source = prep("source", A.REFIT_ROLE)
    target = prep("target", A.REFIT_ROLE) + prep("target", "test")
    names = lambda side: list(A.side_bags(a, bags, side, rotation, "test")[0].channels)
    cs, ct = int(source[0].windows.shape[1]), int(target[0].windows.shape[1])
    base = P.phase_one(source, cs, ct, P._hyper(cfg, temperature=10.0), seed, device,
                       transfer=transfer, fold=rotation)
    model = copy.deepcopy(base)
    two_phase.bridge(model)
    P.initialise_target(model, arm, prep("target", A.REFIT_ROLE), names("source"), names("target"))
    zs, zt = representations(model, source, "source", device), representations(model, target, "target", device)
    ys, yt = np.array([b.label for b in source]), np.array([b.label for b in target])
    q = P._probabilities(model, target, "target", device)
    head = (np.asarray(q) > 0.5).astype(int)
    conf = np.abs(np.asarray(q) - 0.5)
    top = np.argsort(-conf)[:max(1, int(np.ceil(CONF_SHARE * len(q))))]
    centroids = np.stack([zs[ys == k].mean(0) for k in (0, 1)])
    nearest = np.argmax(_unit(zt) @ _unit(centroids).T, axis=1)
    cluster = spherical_kmeans(zt, centroids)
    out = {"rotation": rotation, "nTarget": int(len(yt)), "targetPositive": round(float(yt.mean()), 3),
           "head": {**_score(yt, head), "AUC": round(float(roc_auc_score(yt, q)), 3)},
           "headConfident": _score(yt[top], head[top]),
           "centroid": _score(yt, nearest), "cluster": _score(yt, cluster),
           "clusterAgreesWithHead": round(float(np.mean(cluster == head)), 3)}
    folds = min(5, int(np.bincount(yt).min()))
    if folds >= 2:
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
        p = cross_val_predict(probe, zt, yt, cv=cv, method="predict_proba")[:, 1]
        out["ceiling"] = {**_score(yt, (p > 0.5).astype(int)), "AUC": round(float(roc_auc_score(yt, p)), 3),
                          "folds": folds}
    return out


def smoke(config: Mapping, product=None, *, transfers: Sequence[Sequence[str]], seed: int = 1,
          rotations: Sequence[int] = (0, 1, 2, 3), workdir=None,
          environ: Mapping[str, str] | None = None) -> dict:
    """`transfers` holds [transfer, arm] pairs."""
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
    started, results = time.time(), []
    with spectral_trunk.installed():
        for transfer, arm in transfers:
            a = A.assign_from_bags(P._transfer(transfer), bags)
            rots = [run_rotation(transfer, cfg, a, bags, device, int(seed), int(r), arm) for r in rotations]
            results.append({"transfer": transfer, "arm": arm, "seed": int(seed), "rotations": rots})
            print(transfer, arm, json.dumps(rots), flush=True)
    (workdir / "pl-source-check.json").write_text(json.dumps(
        {"results": results, "seconds": round(time.time() - started), "environment": environment},
        default=str), encoding="utf-8")
    mode = environ.get(remote_units.RUN_MODE_ENV)
    units = [t for t, _ in transfers]
    if mode:
        payload = {"mode": mode,
                   "evidence": {"commit": environment["commit"], "sourcesDigest": environment["codeVersion"],
                                "codeVersion": environment["codeVersion"], "unit": units[0], "units": units,
                                "complete": True, "gpu": None, "estimates": {}},
                   "environment": environment, "runs": {"ran": units, "skipped": []}}
        (workdir / remote_units.STAMP_FILE).write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return {"ran": units, "seconds": round(time.time() - started)}
