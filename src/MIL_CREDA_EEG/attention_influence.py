"""Direction-free window influence of MIL-CREDA's attention on one synthetic pool unit.

One notebook = one unit. The unit's frozen seed-1 base and MIL-CREDA's cef pick (`pool_aggregation.checked_pick`)
are trained exactly as the mechanism campaign trains its "own" variant (same training seed, epochs and
annealed schedule); the run refuses to report influence unless its pool balanced accuracy reproduces the
pick's. Per evaluation subject (masks known, `eeg_sim.SIM_MASKS`), the logit of P(class 1) is read with all
windows, with the top 25 % attention windows removed, with the bottom 25 %, and with 10 random 25 % draws;
and, as a check of the instrument, with the class-event windows and the artifact windows removed against 10
random sets of the same size. Influence is the absolute change of the logit; its direction is recorded as
whether the change moved the model toward the subject's true class. Writes `influence-<unit_key>.json`.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from MIL_CREDA_EEG import criteria_pool as C, eeg_sim as E, mechanisms, oracle_screen as O, two_phase, xcompare as XC
from MIL_CREDA_EEG import pool_aggregation as PA

FRAC, DRAWS, METHOD = 0.25, 10, "MIL-CREDA"
#: largest |pool BA - pick BA| accepted as reproducing the pick
TOLERANCE = 1e-9


def logit1(model, X, dev) -> tuple[float, float]:
    """Logit and probability of class 1 for one bag of windows."""
    p = float(two_phase.evaluate(model, [SimpleNamespace(windows=X)], "target", dev)[0, 1])
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p))), p


def subject_rows(model, bags, dev, masks=None, seed: int = 10_000) -> list[dict]:
    """Per bag: the logit change under each removal (signed; influence is its absolute value)."""
    rows = []
    for k, b in enumerate(bags):
        X = b.windows
        n = X.shape[0]
        q = max(1, int(round(FRAC * n)))
        w = np.asarray(mechanisms.attention_weights(model, X, "target"), float)
        order = np.argsort(w)
        base, p0 = logit1(model, X, dev)

        def delta(drop):
            keep = torch.as_tensor(np.setdiff1d(np.arange(n), drop))
            return logit1(model, X[keep], dev)[0] - base

        rng = np.random.default_rng(seed + k)
        row = {"subject": b.subject, "label": int(b.label), "n": n, "p": p0,
               "top": delta(order[-q:]), "bottom": delta(order[:q]),
               "random": [delta(rng.permutation(n)[:q]) for _ in range(DRAWS)]}
        if masks is not None:
            m = masks[b.subject]
            for name in ("event", "artifact"):
                idx = np.flatnonzero(np.asarray(m[name], bool))
                if 0 < len(idx) < n:
                    row[name] = delta(idx)
                    row[name + "Random"] = [delta(rng.permutation(n)[:len(idx)]) for _ in range(DRAWS)]
        rows.append(row)
    return rows


def run(unit_key: str, threads: int | None = 1, workdir=None, bases=None, picks=None,
        eval_subjects: int = O.EVAL_SUBJECTS) -> dict:
    t0 = time.time()
    pick = PA.checked_pick(unit_key, picks)
    trial, params = int(pick["trial"]), dict(pick["params"])
    tseed = C.train_seed(PA.PICK_SEED, trial)
    threads = int(threads or os.environ.get("MIL_CREDA_THREADS") or os.cpu_count() or 2)
    s = C._setup(unit_key, (METHOD,), eval_subjects, bases, threads)
    try:
        with PA.attention_moves(s["b"]["base"]):
            model, totals = C.train(METHOD, params, s["b"], s["sides"], s["cfg"], s["dev"], tseed,
                                    s["rot"], s["name"], C.EPOCHS2)
        with torch.no_grad():
            on_pool = XC.snapshot(model, s["pool"], s["cut"], s["dev"])
            diff = abs(float(on_pool["BA"]) - float(pick["BApool"]))
            out = {"unitKey": unit_key, "trial": trial, "params": params, "trainSeed": tseed, "threads": threads,
                   "pickBApool": pick["BApool"], "BApool": on_pool["BA"], "AUCpool": on_pool["AUC"],
                   "reproduction": diff, "reproduces": diff <= TOLERANCE, "stepTotals": totals,
                   "frac": FRAC, "draws": DRAWS}
            if out["reproduces"]:
                out["rows"] = subject_rows(model, s["pool"], s["dev"], E.SIM_MASKS)
    finally:
        s["stack"].close()
    out["seconds"] = round(time.time() - t0, 1)
    workdir = Path(workdir) if workdir else Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / f"influence-{unit_key}.json").write_text(json.dumps(out))
    if not out["reproduces"]:
        raise RuntimeError(f"{unit_key}: pool BA {on_pool['BA']} does not reproduce the pick's {pick['BApool']}")
    return out


#: ADHD picks of MIL-CREDA per unit, label-free (cef) and oracle, as the v82 ADHD tables report them
IEEE_PICKS = {tag: PA.PICKS.parent.parent / "attention_influence" / f"ieee_{tag}_picks.json" for tag in ("cef", "oracle")}
#: picks hold BA and AUC rounded to 5 decimals
IEEE_TOLERANCE = 1e-4


def run_ieee(unit_key: str, tag: str = "cef", threads: int | None = 1, workdir=None, bases=None,
             ieee_root=None, picks=None) -> dict:
    """The same measurement on one ADHD unit's target-test subjects (no masks), for the cef or the oracle
    pick; refuses unless the retrained pick reproduces the reported test BA and AUC. Writes
    `influence-<unit_key>-<tag>.json`."""
    from MIL_CREDA_EEG import ieee_pool as I
    t0 = time.time()
    pick = json.loads(Path(picks or IEEE_PICKS[tag]).read_text())["units"][unit_key]
    trial = int(pick["trial"])
    params = C.draw(METHOD, I.PHASE2_SEED)[trial]
    tseed = C.train_seed(I.PHASE2_SEED, trial)
    threads = int(threads or os.environ.get("MIL_CREDA_THREADS") or os.cpu_count() or 2)
    s = I._setup(unit_key, [METHOD], bases or I.BASES, threads, ieee_root)
    try:
        model, totals = C.train(METHOD, params, s["b"], s["sides"], s["cfg"], s["dev"], tseed, s["rot"],
                                s["name"], I.EPOCHS2)
        tt = s["sides"]["target_test"]
        with C._wrapper(METHOD, s["b"], params.get("tau_att", 1.0)), torch.no_grad():
            on_test = XC.snapshot(model, tt, s["cut"], s["dev"])
            diff = max(abs(float(on_test["BA"]) - float(pick["BA"])), abs(float(on_test["AUC"]) - float(pick["AUC"])))
            out = {"unitKey": unit_key, "tag": tag, "trial": trial, "params": params, "trainSeed": tseed,
                   "threads": threads, "pickBA": pick["BA"], "pickAUC": pick["AUC"], "BAtest": on_test["BA"],
                   "AUCtest": on_test["AUC"], "reproduction": diff, "reproduces": diff <= IEEE_TOLERANCE,
                   "stepTotals": totals, "frac": FRAC, "draws": DRAWS}
            if out["reproduces"]:
                out["rows"] = subject_rows(model, tt, s["dev"])
    finally:
        s["stack"].close()
    out["seconds"] = round(time.time() - t0, 1)
    workdir = Path(workdir) if workdir else Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / f"influence-{unit_key}-{tag}.json").write_text(json.dumps(out))
    if not out["reproduces"]:
        raise RuntimeError(f"{unit_key} {tag}: test BA/AUC {on_test['BA']}/{on_test['AUC']} do not reproduce "
                           f"the pick's {pick['BA']}/{pick['AUC']}")
    return out


def run_alignment(unit_key: str, threads: int | None = 1, workdir=None, bases=None, ieee_root=None) -> dict:
    """ADHD local-term alignment for one unit: MIL-CREDA's cef pick trained with and without the local term
    (lambda_local = 0), same training seed; bag representations of the target-test and source subjects at
    the start, middle and end of phase 2, and the source subjects' attention-weighted PLV descriptors.
    Refuses unless the with-term run reproduces the reported test BA and AUC. Writes `alignment-<unit_key>.json`."""
    from MIL_CREDA_EEG import ieee_pool as I, v63_connectivity as V
    t0 = time.time()
    pick = json.loads(IEEE_PICKS["cef"].read_text())["units"][unit_key]
    trial = int(pick["trial"])
    params = C.draw(METHOD, I.PHASE2_SEED)[trial]
    tseed = C.train_seed(I.PHASE2_SEED, trial)
    threads = int(threads or os.environ.get("MIL_CREDA_THREADS") or os.cpu_count() or 2)
    s = I._setup(unit_key, [METHOD], bases or I.BASES, threads, ieee_root)
    out = {"unitKey": unit_key, "trial": trial, "params": params, "trainSeed": tseed, "threads": threads,
           "pickBA": pick["BA"], "pickAUC": pick["AUC"]}
    try:
        tt, src = s["sides"]["target_test"], s["sides"]["source"]
        out["subjects"], out["labels"] = [b.subject for b in tt], [int(b.label) for b in tt]
        take = lambda m: {"t": PA.representations(m, tt, "target"), "s": PA.representations(m, src, "source")}
        for tag, q in (("local", dict(params)), ("noLocal", {**params, "lambda_local": 0.0})):
            with PA.mid_snapshot(take, I.EPOCHS2, 0) as st, PA.mid_snapshot(take, I.EPOCHS2, I.EPOCHS2 // 2) as md:
                model, _ = C.train(METHOD, q, s["b"], s["sides"], s["cfg"], s["dev"], tseed, s["rot"],
                                   s["name"], I.EPOCHS2)
            end = take(model)
            out[tag] = {k: {"t": v["t"].tolist(), "s": v["s"].tolist()}
                        for k, v in (("start", st["taken"]), ("mid", md["taken"]), ("end", end))}
            if tag == "local":
                with C._wrapper(METHOD, s["b"], params.get("tau_att", 1.0)), torch.no_grad():
                    on_test = XC.snapshot(model, tt, s["cut"], s["dev"])
                    sa = {b.subject: np.asarray(mechanisms.attention_weights(model, b.windows, "source"), float)
                          for b in src}
                diff = max(abs(float(on_test["BA"]) - float(pick["BA"])), abs(float(on_test["AUC"]) - float(pick["AUC"])))
                out.update(BAtest=on_test["BA"], AUCtest=on_test["AUC"], reproduction=diff,
                           reproduces=diff <= IEEE_TOLERANCE)
                if not out["reproduces"]:
                    break
                out["sourceLabels"] = {b.subject: int(b.label) for b in src}
                out["sourceDescriptors"] = {b.subject: V.descriptors(V.weighted_graph(
                    V.window_plv(b.windows.numpy()), sa[b.subject])).tolist() for b in src}
    finally:
        s["stack"].close()
    out["seconds"] = round(time.time() - t0, 1)
    workdir = Path(workdir) if workdir else Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / f"alignment-{unit_key}.json").write_text(json.dumps(out))
    if not out["reproduces"]:
        raise RuntimeError(f"{unit_key}: alignment run does not reproduce the cef pick ({out['reproduction']})")
    return out


def run_alignment_all(threads: int | None = 1, workdir=None, units=None) -> dict:
    """Every ADHD unit's alignment run, one after another."""
    from MIL_CREDA_EEG import ieee_pool as I
    done = {}
    for u in units or I.UNITS:
        try:
            done[u] = run_alignment(u, threads, workdir)["reproduction"]
        except RuntimeError as e:
            done[u] = str(e)
    Path(workdir or Path.cwd()).joinpath("alignment-done.json").write_text(json.dumps(done))
    return done


def run_ieee_all(tag: str = "cef", threads: int | None = 1, workdir=None, units=None) -> dict:
    """Every ADHD unit for one pick tag, one after another (each run is short)."""
    from MIL_CREDA_EEG import ieee_pool as I
    done = {}
    for u in units or I.UNITS:
        try:
            done[u] = run_ieee(u, tag, threads, workdir)["reproduction"]
        except RuntimeError as e:
            done[u] = str(e)
    Path(workdir or Path.cwd()).joinpath(f"influence-ieee-{tag}-done.json").write_text(json.dumps(done))
    return done
