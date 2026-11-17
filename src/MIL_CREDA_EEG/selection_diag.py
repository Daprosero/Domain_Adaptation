"""Diagnostic of the label-free phase-2 selection rule (not a method step, never selects).

Replays MIL-CREDA's phase-2 search of one campaign unit exactly (same bags, same phase-1
search, same TPE sampler and seed, same objective returned to the sampler, so the same 20
trials) and records, per trial:

* the current rule's score (phase2_select.centroid_score, BA of nearest-centroid assignment of
  the labelled source-test bags; the value the campaign optimised);
* a continuous variant: the mean normalised centroid margin of those source-test bags,
  (d(wrong centroid) - d(right centroid)) / mean(d), which rarely ties;
* the target-test BA at the unit's source cut and AUC, read ONLY to evaluate the rules.

Target labels never reach the sampler, the selection or any model update.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import optuna
import torch

from MIL_CREDA_EEG import campaign, remote_units, spectral_trunk
from MIL_CREDA_EEG import phase2_select
from MIL_CREDA_EEG import phase2_trace as P
from MIL_CREDA_EEG import v63_setup as V
from MIL_CREDA_EEG import xcompare as XC
from MIL_CREDA_EEG import xpilot as XP


def centroid_margin(target_reps, target_pseudo, source_reps, source_labels) -> float:
    t = np.asarray(target_reps, dtype=np.float64)
    q = np.asarray(target_pseudo).astype(int).reshape(-1)
    s = np.asarray(source_reps, dtype=np.float64)
    y = np.asarray(source_labels).astype(int).reshape(-1)
    classes = np.unique(y)
    if len(classes) != 2 or any(not np.any(q == c) for c in classes):
        return -1.0
    cen = np.stack([t[q == c].mean(0) for c in classes])
    dist = np.sqrt(((s[:, None, :] - cen[None, :, :]) ** 2).sum(-1))
    idx = np.searchsorted(classes, y)
    right, wrong = dist[np.arange(len(y)), idx], dist[np.arange(len(y)), 1 - idx]
    return float(np.mean((wrong - right) / (dist.mean() + 1e-12)))


def _instances(model, bags, side, device):
    """Per-window representations (Eq. 13, the head's input space) and each window's bag index."""
    model.eval()
    with torch.no_grad():
        H = model.embed([b.windows.to(device) for b in bags], side)
    Z = torch.cat([h.detach().cpu() for h in H]).numpy().astype(np.float64)
    idx = np.concatenate([np.full(len(h), i) for i, h in enumerate(H)])
    return Z, idx


def _centroid_scores(zt, qt, zs, ys):
    """Nearest-centroid BA and mean normalised margin of labelled source points to target centroids."""
    from sklearn.metrics import balanced_accuracy_score
    classes = np.unique(ys)
    if any(not np.any(qt == c) for c in classes):
        return 0.0, -1.0
    cen = np.stack([zt[qt == c].mean(0) for c in classes])
    d = np.sqrt(((zs[:, None, :] - cen[None]) ** 2).sum(-1))
    ba = float(balanced_accuracy_score(ys, classes[np.argmin(d, 1)]))
    i = np.searchsorted(classes, ys)
    m = float(np.mean((d[np.arange(len(ys)), 1 - i] - d[np.arange(len(ys)), i]) / (d.mean() + 1e-12)))
    return ba, m


def _r(a, k=5):
    return np.round(np.asarray(a, dtype=np.float64), k).tolist()


def run(config, product=None, *, experiment="ieee", rotation=0, half=0, trials=20, trials2=20,
        workdir=None, environ=None, phase1_params=None, artifacts=False, extra=False, **unit) -> dict:
    """`phase1_params` skips the phase-1 search (the campaign's searched parameters for this unit);
    `artifacts` keeps, per trial, the representations and probabilities every label-free validator
    needs (target adapt, source test with labels, target test with labels) for offline scoring."""
    environ = os.environ if environ is None else environ
    workdir = Path(workdir) if workdir else Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 2))
    started = time.time()
    cfg = remote_units.resolve_inputs(campaign.check_config(config), environ=environ)
    built = XP.build_sides(cfg, environ, experiment=experiment, rotation=rotation, half=half, **unit)
    sides, name, cfg = built["sides"], built["unit"], built["cfg"]
    device = torch.device(cfg.get("device", "cpu"))
    rows = []
    with spectral_trunk.installed(), XP.window_samples(built["samples"]):
        if phase1_params is not None:
            ranked = [(dict(phase1_params), None)]
        else:
            params1, score1 = XP.search_phase1(sides, cfg, device, trials, rotation)
            ranked = list(XP.LAST_PHASE1_RANKED) or [(params1, score1)]
        for rank, (p1, s1) in enumerate(ranked):     # the campaign's non-collapsing rule
            try:
                cut = XC.shared_base(sides, cfg, device, XP.SEARCH2_SEED, rotation, p1, name)["cut"]
                base = XP.phase2_base(sides, cfg, device, XP.SEARCH2_SEED, rotation, p1, name)
                break
            except V.V63Error:
                continue
        P.PHASE_ONE_SEARCHED.clear()
        P.PHASE_ONE_SEARCHED[int(rotation)] = p1
        print(f"[{name}] phase 1 rank {rank} {p1}", flush=True)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=XP.SAMPLER_SEED + int(rotation)))
        labels_s = [b.label for b in sides["source_test"]]

        def objective(trial):
            params = XP._suggest2(trial)
            row = {"trial": trial.number, "params": params}
            try:
                out = XP.phase2_trial(base, sides, cfg, device, XP.SEARCH2_SEED, p1, params)
                model = out["model"]
                zt, pseudo = XP._blind_reps(model, sides["target"], "target", device)
                zs, _ = XP._blind_reps(model, sides["source_test"], "source", device)
                row["ruleBA"] = round(float(out["score"]), 4)
                row["ruleMargin"] = round(centroid_margin(zt, pseudo, zs, labels_s), 5)
                y, p = P._subjects(sides["target_test"], P._probabilities(model, sides["target_test"], "target", device))
                m = P._metrics(y, p, cut)
                row["targetBA"], row["targetAUC"] = m["BA"], m["AUC"]
                if extra:                      # larger source sets for the rule: inner validation, windows
                    inner = {b.subject: b for _, va in sides["inner"] for b in va}
                    inner = list(inner.values())
                    zi, _ = XP._blind_reps(model, inner, "source", device)
                    yi = np.array([b.label for b in inner])
                    row["ruleInnerBA"], row["ruleInnerMargin"] = _centroid_scores(
                        zt, np.asarray(pseudo), zi, yi)
                    Ht, it = _instances(model, sides["target"], "target", device)
                    qt = np.asarray(pseudo)[it]
                    for tag, bags in (("SourceTest", sides["source_test"]), ("Inner", inner)):
                        Hs, js = _instances(model, bags, "source", device)
                        ys_i = np.array([bags[k].label for k in js])
                        row[f"inst{tag}BA"], row[f"inst{tag}Margin"] = _centroid_scores(Ht, qt, Hs, ys_i)
                    row["innerBags"] = len(inner)
                if artifacts:
                    ys, ps = P._subjects(sides["source_test"], P._probabilities(model, sides["source_test"], "source", device))
                    _, pa = P._subjects(sides["target"], P._probabilities(model, sides["target"], "target", device))
                    row["art"] = {"zTarget": _r(zt), "pseudoTarget": [int(v) for v in pseudo], "pTarget": _r(pa),
                                  "zSource": _r(zs), "ySource": [int(v) for v in labels_s], "pSource": _r(ps),
                                  "pTest": _r(p), "yTest": [int(v) for v in y]}
                score = row["ruleBA"]
            except V.V63Error as e:
                row["error"] = str(e)
                score = 0.0
            score = score if np.isfinite(score) else 0.0
            rows.append(row)
            print(f"[{name}] trial {trial.number + 1}/{trials2} rule {row.get('ruleBA')} margin "
                  f"{row.get('ruleMargin')} target BA {row.get('targetBA')}", flush=True)
            return score            # the sampler sees exactly what the campaign's search saw

        study.optimize(objective, n_trials=int(trials2))
    out = {"unit": name, "rotation": rotation, "half": half, "phase1": {"rank": rank, "params": p1},
           "cut": float(cut), "trials": rows, "seconds": round(time.time() - started)}
    (workdir / f"selection-diag-{name.replace(':', '-')}.json").write_text(json.dumps(out, default=str))
    mode = environ.get(remote_units.RUN_MODE_ENV)
    if mode:
        env = remote_units.environment_stamp(remote_units.set_deterministic(),
                                             remote_units.campaign_identity(cfg, environ),
                                             remote_units.device_placement(config, environ))
        (workdir / remote_units.STAMP_FILE).write_text(json.dumps(
            {"mode": mode, "evidence": {"commit": env["commit"], "sourcesDigest": env["codeVersion"],
                                        "codeVersion": env["codeVersion"], "unit": name, "units": [name],
                                        "complete": True, "gpu": None, "estimates": {}},
             "environment": env, "runs": {"ran": [name], "skipped": []}}, default=str))
    return {"ran": [name], "seconds": out["seconds"]}
