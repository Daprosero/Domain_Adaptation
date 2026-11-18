"""The synthetic shared-pool protocol on the IEEE ADHD cross-subject units.

One notebook = one unit (`ieee-r{r}h{h}`: rotation r of the five subject folds, target-test half h)
and one shard of the trials. Phase 1 is NOT searched here: every method of every notebook starts
phase 2 from the unit's single base (`ieee-r{r}h{h}-s1.pt`), fitted once with seed 1 by
`xcompare.shared_base` at the TPE-searched phase-1 settings the unit's rehearsal recorded
(`run.kwargs.phase1_params` of `papersmith-ie-r{r}h{h}-cef/run-config.json`, found by
`xpilot.search_phase1`), and frozen with its cut and its phase-1 prediction on the target-test
half (the "Phase 1 (source only)" row). For each method, phase 2 trains the same 100 random settings
the synthetic pool draws (`criteria_pool.draw(method, 1)`: Musgrave et al.'s protocol, validators
select from a shared pool and never guide it), each for 100 epochs under the annealed learning rate
of `criteria_pool.train`, with one training seed that depends on the trial index only
(`criteria_pool.train_seed(1, trial)`), so the shards together equal one unsharded run.

There is no evaluation pool (the simulator's fresh subjects have no IEEE counterpart): each trial
records BA at the base's cut, AUC and Cohen's kappa on the unit's own target-test half, and the
outputs every later ADHD result reads offline, so none of them retrains: the target-adaptation and
source-test outputs (`Pt, Zt, Ps, Zs`, exactly as `criteria_pool` saves them, from which cef and
every label-free criterion are computed), the target-test probabilities, the per-window attention
of every bag of the four sides, the final representations of the source training and target-test
bags, and, for MIL-CREDA, the representations at epoch 50 (`pool_aggregation.mid_snapshot`, which
leaves the run unchanged). cef picks offline; the oracle, the best test-half BA of the 100 trials,
is reported as an upper bound only. No label-free criterion is computed or read here.

Usage:
  python -m MIL_CREDA_EEG.ieee_pool bases <rehearsal_dir> <out_dir> [--units ieee-r0h0 ...]
  python -m MIL_CREDA_EEG.ieee_pool run <unit> <shard> [--shards 8] [--procs 4] [--workdir .]
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import multiprocessing
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from MIL_CREDA_EEG import campaign, mechanisms, remote_units, spectral_trunk
from MIL_CREDA_EEG import criteria_pool as C
from MIL_CREDA_EEG import oracle_screen as O
from MIL_CREDA_EEG import pool_aggregation as PA
from MIL_CREDA_EEG import xcompare as XC
from MIL_CREDA_EEG import xpilot as XP

BASES = O.BASES.parent / "ieee_pool_bases"
METHODS = C.METHODS
EPOCHS2 = C.EPOCHS2
TRIALS = C.TRIALS
#: the synthetic pool's phase-2 seed: the same 100 settings per method, the same training seeds
PHASE2_SEED = 1
SHARDS = 8
UNITS = tuple(f"ieee-r{r}h{h}" for r in range(5) for h in range(2))
SLOWNESS = C.SLOWNESS
#: the four sides whose every bag's attention is saved, with the domain each one is read in
SIDES = {"target": "target", "target_test": "target", "source_test": "source", "source": "source"}
#: the sides whose representations MIL-CREDA saves at mid-run
MID_SIDES = {"target": "target", "target_test": "target", "source": "source"}


def unit_of(unit_key: str) -> tuple[int, int]:
    m = re.fullmatch(r"ieee-r(\d)h([01])", unit_key)
    if not m:
        raise ValueError(f"an IEEE unit is ieee-r<rotation>h<half>; got {unit_key!r}")
    return int(m.group(1)), int(m.group(2))


def base_file(unit_key: str, bases=None) -> Path:
    """Always the seed-1 base: phase 1 is fitted once per unit and frozen."""
    return Path(bases or BASES) / f"{unit_key}-s1.pt"


def draw(method: str, trials: int = TRIALS) -> list[dict]:
    """The synthetic pool's own draw: `criteria_pool.draw(method, 1)`."""
    return C.draw(method, PHASE2_SEED, trials)


def shard_trials(shard: int, shards: int = SHARDS, trials: int = TRIALS) -> range:
    """A balanced partition: shard sizes differ by at most one (100 trials in 8 shards: 12 or 13),
    unlike `criteria_pool.shard_trials`' ceiling split (seven of 13 and one of 9)."""
    t, n = int(trials), int(shards)
    return range(int(shard) * t // n, (int(shard) + 1) * t // n)


def recorded_settings(rehearsal_dir, rotation: int, half: int) -> tuple[dict, dict]:
    """The unit's TPE-searched phase-1 settings and its config, as its rehearsal recorded them."""
    hits = sorted(glob.glob(f"{rehearsal_dir}/*/papersmith-ie-r{int(rotation)}h{int(half)}-cef/run-config.json"))
    if not hits:
        raise FileNotFoundError(f"no papersmith-ie-r{rotation}h{half}-cef/run-config.json under {rehearsal_dir}")
    kw = json.loads(Path(hits[0]).read_text())["run"]["kwargs"]
    for k, v in (("experiment", "ieee"), ("rotation", int(rotation)), ("half", int(half))):
        if k in kw and kw[k] != v:
            raise ValueError(f"{hits[0]} records {k}={kw[k]!r}, not {v!r}")
    return dict(kw["phase1_params"]), dict(kw["config"])


def _environ(ieee_root=None) -> dict:
    root = ieee_root or os.environ.get("IEEE_ROOT")
    return {"IEEE_ROOT": root} if root else {}


def _build(config, rotation, half, ieee_root=None) -> dict:
    cfg = remote_units.resolve_inputs(campaign.check_config(config), environ={})
    return XP.build_sides(cfg, _environ(ieee_root), experiment="ieee", rotation=int(rotation), half=int(half))


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=os.path.dirname(XC.__file__)).stdout.strip()


def build_base(rotation: int, half: int, phase1_params: dict, config: dict, out_dir, ieee_root=None) -> Path:
    """Phase 1 at the recorded settings, seed 1, saved in `make_bases.py`'s layout. `predict` is the
    phase-1 model's own prediction on the target-test half (`xcompare.snapshot`: BA at the cut, AUC,
    kappa, rows), with each subject's name, unrounded probability and label beside it."""
    unit_key = f"ieee-r{int(rotation)}h{int(half)}"
    built = _build(config, rotation, half, ieee_root)
    sides, name, cfg = built["sides"], built["unit"], built["cfg"]
    with spectral_trunk.installed(), XP.window_samples(built["samples"]), XP.full_trace_mode(False):
        t0 = time.time()
        b = XC.shared_base(sides, cfg, torch.device("cpu"), 1, int(rotation), dict(phase1_params), name)
        y, p = XC.P._subjects(sides["target_test"], XC.P._probabilities(b["prepared"], sides["target_test"],
                                                                        "target", torch.device("cpu")))
    subjects = list(dict.fromkeys(x.subject.split("#")[0] for x in sides["target_test"]))
    predict = {**b["predict"], "subjects": subjects, "probabilities": [float(v) for v in p],
               "labels": [int(v) for v in y]}
    rec = {"unitKey": unit_key, "unit": name, "seed": 1, "rotation": int(rotation), "phase1": dict(phase1_params),
           "unitKwargs": {"half": int(half)}, "config": dict(config), "cut": float(b["cut"]), "arm": b["arm"],
           "context": b["context"], "sourceTest": b["sourceTest"], "predict": predict, "commit": _commit(),
           "torch": torch.__version__, "base": b["base"], "prepared": b["prepared"]}
    out = base_file(unit_key, out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rec, out)
    print(unit_key, "cut", round(b["cut"], 4), "predictBA", predict["BA"], round(time.time() - t0, 1), "s",
          flush=True)
    return out


#: the per-process state a worker builds once: base, sides, entered contexts
_STATE: dict = {}


def _setup(unit_key, methods, bases, threads, ieee_root=None) -> dict:
    torch.set_num_threads(max(1, int(threads)))
    rotation, half = unit_of(unit_key)
    with spectral_trunk.installed():
        b = torch.load(base_file(unit_key, bases), weights_only=False)
    if (int(b["rotation"]), int(b["unitKwargs"]["half"])) != (rotation, half):
        raise ValueError(f"base {b['unitKey']} is not unit {unit_key}")
    built = _build(b["config"], rotation, half, ieee_root)
    sides, name, cfg = built["sides"], built["unit"], built["cfg"]
    if name != b["unit"]:
        raise ValueError(f"base {b['unit']} does not match the rebuilt unit {name}")
    campaign.register([m for m in methods if m != "MIL-CREDA"])
    stack = contextlib.ExitStack()
    for ctx in (spectral_trunk.installed(), XP.window_samples(built["samples"]), XP.full_trace_mode(False)):
        stack.enter_context(ctx)
    return {"b": b, "sides": sides, "name": name, "cfg": cfg, "stack": stack, "dev": torch.device("cpu"),
            "rot": rotation, "cut": float(b["cut"])}


def _init_worker(unit_key, methods, bases, threads, ieee_root):
    _STATE.update(_setup(unit_key, methods, bases, threads, ieee_root))


def attention(model, bags, domain: str) -> tuple[np.ndarray, np.ndarray]:
    """Every bag's window weights (`mechanisms.attention_weights`, the primitive under
    `pool_aggregation.attention_rows`, which also reads the simulator's masks the IEEE bags do not
    have), concatenated in bag order, and the offsets of each bag's slice."""
    w = [np.asarray(mechanisms.attention_weights(model, b.windows, domain), dtype=np.float32) for b in bags]
    sizes = [len(x) for x in w]
    return (np.concatenate(w) if w else np.zeros(0, np.float32),
            np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64))


def _mid_take(sides):
    return lambda m: {f"Zmid[{k}]": PA.representations(m, sides[k], d).astype(np.float32)
                      for k, d in MID_SIDES.items()}


def _trial(task) -> tuple:
    """One (method, trial): trains it, saves its arrays, returns its row. Depends on the task only,
    so the process that runs it and the order trials run in change nothing."""
    method, i, p, epochs, arr_dir, mid = task
    s, t0 = _STATE, time.time()
    sides, dev, cut = s["sides"], s["dev"], s["cut"]
    snap_ctx = (PA.mid_snapshot(_mid_take(sides), epochs, epochs // 2) if mid and method == "MIL-CREDA"
                else contextlib.nullcontext(None))
    try:
        with snap_ctx as snap:
            model, totals = C.train(method, p, s["b"], sides, s["cfg"], dev, C.train_seed(PHASE2_SEED, i),
                                    s["rot"], s["name"], epochs)
    except FloatingPointError:     # the trainer's own non-finite refusal
        return method, i, {"params": p, "nonFinite": True, "seconds": round(time.time() - t0, 1)}
    arrays = {}
    with C._wrapper(method, s["b"], p.get("tau_att", 1.0)):
        on_test = XC.snapshot(model, sides["target_test"], cut, dev)
        arrays["Pt"], arrays["Zt"] = C._outputs(model, sides["target"], "target", dev)
        arrays["Ps"], arrays["Zs"] = C._outputs(model, sides["source_test"], "source", dev)
        _, arrays["Zsrc"] = C._outputs(model, sides["source"], "source", dev)
        _, arrays["Ztest"] = C._outputs(model, sides["target_test"], "target", dev)
        arrays["Ptest"] = XC.P._probabilities(model, sides["target_test"], "target", dev).astype(np.float32)
        for side, domain in SIDES.items():
            arrays[f"att[{side}]"], arrays[f"attOffsets[{side}]"] = attention(model, sides[side], domain)
    row = {"params": p, "BAtest": on_test["BA"], "AUCtest": on_test["AUC"], "kappaTest": on_test["kappa"],
           "stepTotals": totals}
    if snap is not None:
        if "taken" not in snap:
            raise RuntimeError(f"{method} trial {i}: the mid snapshot was not taken")
        arrays.update(snap["taken"])
        row["mid"] = {"epoch": epochs // 2, "step": snap.get("step"), "totalSteps": snap.get("totalSteps"),
                      "stepsPerEpoch": snap.get("perEpoch")}
    np.savez_compressed(Path(arr_dir) / f"{method.split(' ')[0]}-{i}.npz", **arrays)
    row["seconds"] = round(time.time() - t0, 1)
    return method, i, row


def run(unit_key: str, shard: int, shards: int = SHARDS, trials: int = TRIALS, methods: Sequence[str] = METHODS,
        workdir=None, bases=None, epochs: int = EPOCHS2, procs: int = 4, threads: int = 1, mid: bool = True,
        ieee_root=None) -> dict:
    """`procs` worker processes, each with `threads` torch threads, share the shard's trials, as in
    `criteria_pool.run`. Every row depends on its (method, trial) only. `mid` False skips
    MIL-CREDA's mid-run snapshot (the final outputs are identical either way)."""
    workdir = Path(workdir) if workdir else Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"methods are among {list(METHODS)}; got {unknown}")
    ieee_root = _environ(ieee_root).get("IEEE_ROOT")
    _STATE.clear()
    _STATE.update(_setup(unit_key, methods, bases, threads, ieee_root))
    b, sides, name = _STATE["b"], _STATE["sides"], _STATE["name"]
    stem = f"ieeepool-{unit_key}-sh{int(shard)}of{int(shards)}"
    out_path, arr_dir = workdir / f"{stem}.json", workdir / stem
    arr_dir.mkdir(exist_ok=True)
    subjects = lambda side: [x.subject for x in sides[side]]
    labels = lambda side: [int(x.label) for x in sides[side]]
    out = json.loads(out_path.read_text()) if out_path.exists() else {
        "unitKey": unit_key, "unit": name, "shard": int(shard), "shards": int(shards), "trials": int(trials),
        "epochs2": int(epochs), "cut": _STATE["cut"], "baseCommit": b["commit"], "baseSeed": 1,
        "phase2Seed": PHASE2_SEED, "schedule": "lr2*(1+10p)^-0.75",
        "protocol": "v81",
        "oracle": "best BA at the base's cut on the target-test half (upper bound only)",
        "phase1Predict": {k: b["predict"].get(k) for k in ("BA", "AUC", "kappa")},
        "targetTestLabels": labels("target_test"), "targetTestSubjects": subjects("target_test"),
        "targetSubjects": subjects("target"), "sourceTestLabels": labels("source_test"),
        "sourceTestSubjects": subjects("source_test"), "sourceSubjects": subjects("source"), "methods": {}}
    out["procs"], out["threads"] = int(procs), int(threads)
    tasks = []
    for method in methods:
        rows = out["methods"].setdefault(method, {"trials": {}})["trials"]
        settings = draw(method, trials)
        tasks += [(method, i, settings[i], int(epochs), str(arr_dir), bool(mid))
                  for i in shard_trials(shard, shards, trials) if str(i) not in rows]
    # the slowest methods first, so no worker is left alone with one at the end
    tasks.sort(key=lambda t: -SLOWNESS.get(t[0], 1.0))

    def record(result):
        method, i, row = result
        out["methods"][method]["trials"][str(i)] = row
        out_path.write_text(json.dumps(out))

    try:
        if int(procs) <= 1:
            for t in tasks:
                record(_trial(t))
        else:
            _STATE["stack"].close()
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(int(procs), initializer=_init_worker,
                          initargs=(unit_key, tuple(methods), bases, threads, ieee_root)) as workers:
                for result in workers.imap_unordered(_trial, tasks, chunksize=1):
                    record(result)
    finally:
        _STATE["stack"].close()
    for method in methods:
        print(unit_key, shard, method, "done", len(out["methods"][method]["trials"]), flush=True)
    out_path.write_text(json.dumps(out))
    return {"file": out_path.name, "methods": {m: len(v["trials"]) for m, v in out["methods"].items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m MIL_CREDA_EEG.ieee_pool")
    sub = ap.add_subparsers(dest="cmd", required=True)
    bp = sub.add_parser("bases", help="fit the seed-1 phase-1 bases at the rehearsal's recorded settings")
    bp.add_argument("rehearsal_dir")
    bp.add_argument("out_dir", nargs="?", default=str(BASES))
    bp.add_argument("--units", nargs="*", default=list(UNITS))
    rp = sub.add_parser("run", help="run one shard of one unit")
    rp.add_argument("unit")
    rp.add_argument("shard", type=int)
    rp.add_argument("--shards", type=int, default=SHARDS)
    rp.add_argument("--trials", type=int, default=TRIALS)
    rp.add_argument("--methods", nargs="*", default=list(METHODS))
    rp.add_argument("--workdir", default=None)
    rp.add_argument("--bases", default=None)
    rp.add_argument("--epochs", type=int, default=EPOCHS2)
    rp.add_argument("--procs", type=int, default=4)
    rp.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "bases":
        for u in a.units:
            r, h = unit_of(u)
            build_base(r, h, *recorded_settings(a.rehearsal_dir, r, h), a.out_dir)
        print("BASES_DONE", " ".join(a.units), flush=True)
    else:
        print(json.dumps(run(a.unit, a.shard, a.shards, a.trials, a.methods, a.workdir, a.bases, a.epochs,
                             a.procs, a.threads)), flush=True)


if __name__ == "__main__":
    main()
