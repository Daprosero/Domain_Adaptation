"""Patience exercise of the v58 search rule: how many epochs a study really needs under
an epoch cap of 100 with early stopping at patience 10 (an exercise, not the campaign).

The rule under study (operator, 2026-11-06): every trial trains up to `CAP` epochs and
stops when its selection score has not improved for `patience` consecutive epochs.
Phase 1's score is the pooled source-validation balanced accuracy, phase 2's the v58
centroid rule (no target label), both pooled per epoch over the three inner splits,
exactly the series `search.run_phase_one_study` / `search.run_phase_two_study` record.

Each trial trains the FULL `CAP` epochs and keeps the whole series; the stop epoch, the
best epoch under patience and whether the cap was reached are computed afterwards
(`patience_stop`), which is identical to running early stopping (the trainers have no
patience of their own and the score of epoch e never depends on later epochs) and gives
every patience at once.

Plan v59 adopted the rule (cap 100, patience 30 in phase 1 and 10 in phase 2) inside
`search` itself (`search.train_inner_splits`), so a study run by this exercise now stops on
the search's own patience; the records it already produced were measured on full series.

A trial is draw k of the campaign's own sampler (`search.SAMPLER_SEED`; the GPSampler's
first 10 trials come from its independent `RandomSampler` with that seed), enqueued into
a one-trial study so the campaign's objective runs unchanged.

Remote entry point: `run(device, trials, phase_one)` on a cloud session, the processed
data mounted as an input dataset (`remote_timing.processed_root`). It writes, in its
working directory, `patience_exercise/trials.jsonl` (one line per finished trial, as it
finishes) and `patience_exercise/result.json`, and `shard.json` under the remote runner.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Sequence

CAP = 100
PATIENCES = (5, 10, 15)
#: One transfer per arm (core transfers of the separability summary), outer fold 0.
TRANSFERS = ("L1->L11", "Robots->CogniFit", "L6->Robots")
TRIALS_PER_NAME = 10
ROTATION = 0
#: The probes' bags per step (Sec. 7.6 leaves it to the pilot).
BAGS_PER_STEP = 8
SCHEDULE = {"epochs1": CAP, "epochs2": CAP, "bagsPerStep": BAGS_PER_STEP}
GROUPS = ("common", "AdaptMIL", "MIL-CREDA")
STRATEGIES = ("MIL-CREDA", "AdaptMIL", "UDA-DDA", "CCA-LSC", "CASUAL", "COT", "CS+CCS",
              "CLUST — variante source-free", "CLUDA")


# ------------------------------------------------------------------ pure helpers


def patience_stop(series: Sequence[float], patience: int) -> dict:
    """Early stopping replayed on a recorded per-epoch score series.

    An epoch improves when its score is strictly greater than the best so far (NaN never
    improves). Training stops at the first epoch that is `patience` epochs after the last
    improvement (or after the start, when nothing improved yet). Epochs are 1-indexed,
    as `search`'s `epochs = best index + 1`. `reachedCap` is true when no stop fired before
    the last recorded epoch."""
    if patience < 1:
        raise ValueError(f"patience is a positive whole number; got {patience!r}")
    if not series:
        raise ValueError("an empty series has no stop epoch")
    best, best_i = -math.inf, -1
    for i, value in enumerate(series):
        v = float(value)
        if not math.isnan(v) and v > best:
            best, best_i = v, i
        elif i - best_i >= patience and i < len(series) - 1:
            return {"stopEpoch": i + 1, "bestEpoch": best_i + 1, "bestScore": _finite(best),
                    "reachedCap": False}
    return {"stopEpoch": len(series), "bestEpoch": best_i + 1, "bestScore": _finite(best),
            "reachedCap": True}


def _finite(v: float):
    return None if not math.isfinite(v) else v


def full_best(series: Sequence[float]) -> dict:
    """The campaign's rule without patience: argmax over every epoch, earliest on ties."""
    values = [float(v) for v in series]
    finite = [(v, i) for i, v in enumerate(values) if not math.isnan(v)]
    if not finite:
        return {"bestEpoch": None, "bestScore": None}
    best = max(v for v, _ in finite)
    return {"bestEpoch": min(i for v, i in finite if v == best) + 1, "bestScore": best}


def trial_outcomes(series: Sequence[float], patiences: Sequence[int] = PATIENCES) -> dict:
    out = {"epochsRecorded": len(series), "full": full_best(series)}
    for p in patiences:
        stop = patience_stop(series, p)
        full = out["full"]["bestScore"]
        stop["scoreLoss"] = (None if full is None or stop["bestScore"] is None
                             else full - stop["bestScore"])
        out[f"patience{p}"] = stop
    return out


def _quantile(values: Sequence[float], q: float) -> float:
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def distribution(rows: Sequence[Mapping], patience: int) -> dict:
    """Over trial rows (each with `outcomes`): the stop and best epochs under `patience`."""
    key = f"patience{patience}"
    stops = [r["outcomes"][key]["stopEpoch"] for r in rows]
    bests = [r["outcomes"][key]["bestEpoch"] for r in rows]
    losses = [r["outcomes"][key]["scoreLoss"] for r in rows
              if r["outcomes"][key]["scoreLoss"] is not None]
    if not stops:
        return {"trials": 0}
    return {"trials": len(stops),
            "reachedCap": sum(bool(r["outcomes"][key]["reachedCap"]) for r in rows),
            "stopEpoch": {"min": min(stops), "p25": _quantile(stops, .25),
                          "median": statistics.median(stops), "mean": statistics.fmean(stops),
                          "p75": _quantile(stops, .75), "p90": _quantile(stops, .9),
                          "max": max(stops)},
            "bestEpoch": {"min": min(bests), "median": statistics.median(bests),
                          "max": max(bests)},
            "scoreLossVsFull": {"max": max(losses) if losses else None,
                                "mean": statistics.fmean(losses) if losses else None,
                                "nonZero": sum(x > 0 for x in losses)}}


def transfer_of_draw(draw: int, name_index: int, n: int = TRIALS_PER_NAME,
                     transfers: Sequence[str] = TRANSFERS) -> str:
    """Draws 0..n-1 of one group/strategy spread over the arms as evenly as possible
    (4/3/3); which arm gets the extra draw rotates with the name's index."""
    k = len(transfers)
    sizes = [n // k + (1 if j < n % k else 0) for j in range(k)]
    edge = 0
    for j, size in enumerate(sizes):
        edge += size
        if draw < edge:
            return transfers[(j + name_index) % k]
    raise ValueError(f"draw {draw} is outside 0..{n - 1}")


def assign(trials: Sequence[Mapping], bins: int, transfer_setup: float = 0.0,
           group_setup: Mapping | None = None) -> list[list[Mapping]]:
    """Longest-processing-time assignment of trials (each with `seconds`) to `bins`
    notebooks; a notebook pays `transfer_setup` once per transfer it holds and, for phase
    2, `group_setup[(group, transfer)]` once per phase-1 refit it needs."""
    group_setup = group_setup or {}
    loads = [0.0] * bins
    keys: list[set] = [set() for _ in range(bins)]
    out: list[list[Mapping]] = [[] for _ in range(bins)]

    def extra(b: int, t: Mapping) -> float:
        cost = 0.0 if ("t", t["transfer"]) in keys[b] else transfer_setup
        g = t.get("group")
        if t["phase"] == 2 and ("g", g, t["transfer"]) not in keys[b]:
            cost += group_setup.get((g, t["transfer"]), 0.0)
        return cost

    for t in sorted(trials, key=lambda x: (-x["seconds"], x["name"], x["draw"])):
        b = min(range(bins), key=lambda j: (loads[j] + extra(j, t) + t["seconds"], j))
        loads[b] += extra(b, t) + t["seconds"]
        keys[b].add(("t", t["transfer"]))
        if t["phase"] == 2:
            keys[b].add(("g", t.get("group"), t["transfer"]))
        out[b].append(dict(t))
    return out


# ------------------------------------------------------------------ remote run


def registrations(trials: Sequence[Mapping]) -> list[str]:
    """The strategies `campaign.register` must wire for `trials`: every phase-2 strategy,
    and a phase-1 group registered by its own comparator unit (AdaptMIL)."""
    names = {t["name"] for t in trials if t["phase"] == 2}
    names |= {t["name"] for t in trials if t["phase"] == 1 and t["name"] in STRATEGIES
              and t["name"] != "MIL-CREDA"}
    return sorted(names)


class GpuWitness:
    """Per trial, what certifies where training ran: the device types of every parameter
    any optimizer stepped (a global optimizer step pre-hook, so every trainer is covered
    however it builds its model), the number of steps, the CUDA peak memory, and the end
    of every epoch (`two_phase.validation_record`, which all twelve trainers call once
    per epoch, wrapped to stamp the clock) to give the measured seconds per epoch."""

    def __init__(self, two_phase, torch):
        self.two_phase, self.torch = two_phase, torch
        self.devices: dict[str, int] = {}
        self.steps = 0
        self.stamps: list[float] = []

    def __enter__(self):
        torch, original = self.torch, self.two_phase.validation_record
        self._original = original

        def stamped(*args, **kwargs):
            out = original(*args, **kwargs)
            self.stamps.append(time.perf_counter())
            return out

        def hook(optimizer, args, kwargs):
            self.steps += 1
            if self.steps <= 3 or self.steps % 500 == 0:
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        key = str(p.device)
                        self.devices[key] = self.devices.get(key, 0) + 1

        self.two_phase.validation_record = stamped
        from torch.optim.optimizer import register_optimizer_step_pre_hook
        self._handle = register_optimizer_step_pre_hook(hook)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc):
        self.two_phase.validation_record = self._original
        self._handle.remove()
        return False

    def evidence(self) -> dict:
        torch = self.torch
        diffs = [b - a for a, b in zip(self.stamps, self.stamps[1:])]
        return {"optimizerSteps": self.steps,
                "parameterDevices": dict(sorted(self.devices.items())),
                "allParametersOnCuda": bool(self.devices) and all(
                    k.startswith("cuda") for k in self.devices),
                "cudaMaxMemoryAllocatedMB": (torch.cuda.max_memory_allocated() / 2**20
                                             if torch.cuda.is_available() else None),
                "epochsStamped": len(self.stamps),
                "medianEpochSeconds": statistics.median(diffs) if diffs else None}


@contextmanager
def _enqueued(search, params: Mapping):
    """Every study `search.make_study` creates meanwhile starts with `params` enqueued."""
    original = search.make_study

    def make(path, name):
        study = original(path, name)
        study.enqueue_trial(dict(params))
        return study

    search.make_study = make
    try:
        yield
    finally:
        search.make_study = original


def _phase_one_record(search, model_store, bag_data, group, data, selection, device, scratch,
                      schedule=SCHEDULE):
    """The outer fold's phase-1 checkpoint of `group`: `selection`'s configuration refit on
    the refit view for exactly its E1 epochs at `search.PHASE_ONE_SEED` (Sec. 4.5)."""
    schedule = search.refit_schedule(schedule, 1, int(selection["epochs"]))
    hyper = search.phase_one_hyper(group, selection["params"], schedule)
    trained = search.train_phase_one(group, data, ROTATION, search.PHASE_ONE_SEED, hyper,
                                     device)
    path = search.save_phase_one(Path(scratch) / "Models", group, data, ROTATION,
                                 search.PHASE_ONE_SEED, trained)
    return model_store.load(path, partition_path=bag_data.default_partition_path())


def run_trial(spec: Mapping, data, record, device, scratch, schedule=SCHEDULE) -> dict:
    """One trial of the campaign's objective, on one-trial study at `scratch`."""
    from MIL_CREDA_EEG import model_store, search
    root = Path(scratch) / f"trial-{spec['phase']}-{spec['draw']}-{model_store.name_slug(spec['name'])}"
    with _enqueued(search, spec["params"]):
        if spec["phase"] == 1:
            study = search.run_phase_one_study(spec["name"], data, ROTATION, schedule,
                                               root=root, n_trials=1, device=device)
        else:
            study = search.run_phase_two_study(spec["name"], data, ROTATION, record, schedule,
                                               root=root, n_trials=1, device=device)
    trial = study.trials[0]
    if trial.params != dict(spec["params"]):
        raise RuntimeError(f"the trial ran {trial.params}, not the enqueued {spec['params']}")
    series = trial.user_attrs[search.PRIMARY_SERIES[spec["phase"]]]
    return {"series": [None if v is None or not math.isfinite(v) else float(v)
                       for v in series],
            "trainLoss": trial.user_attrs.get("trainLoss"), "value": trial.value,
            "campaignBestEpoch": trial.user_attrs.get("bestEpoch")}


def run(device: str = "cuda", trials: Sequence[Mapping] = (),
        phase_one: Sequence[Mapping] = (), out_dir: str = "patience_exercise",
        cap: int = CAP, processed: str | None = None) -> dict:
    """Run `trials` (phase 1 and/or 2 specs) on `device`; `phase_one` holds, per (group,
    transfer) a phase-2 trial needs, the selected {"group", "transfer", "params", "epochs"}."""
    import torch
    from MIL_CREDA_EEG import (assignment, bag_data, campaign, model_store, remote_timing,
                               search, two_phase)
    started = time.time()
    evidence = remote_timing.device_evidence(device)
    evidence["cudaIsAvailable"] = bool(torch.cuda.is_available())
    evidence["cudaDeviceName0"] = (torch.cuda.get_device_name(0)
                                   if torch.cuda.is_available() else None)
    print(f"[patience] device {json.dumps(evidence)}", flush=True)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    lines = target / "trials.jsonl"
    schedule = {**SCHEDULE, "epochs1": int(cap), "epochs2": int(cap)}
    root = Path(processed) if processed else remote_timing.processed_root()
    t0 = time.time()
    bags = bag_data.load_bags(root)
    setup = {"loadBagsSeconds": time.time() - t0, "searchDataSeconds": {},
             "phaseOneRefitSeconds": {}}
    campaign.register(registrations(trials))
    by_name = {t.name: t for t in assignment.TRANSFERS}
    selections = {(s["group"], s["transfer"]): s for s in phase_one}
    datas, records, errors, done = {}, {}, {}, []
    with tempfile.TemporaryDirectory() as scratch:
        for spec in trials:
            key = f"p{spec['phase']}:{spec['name']}:{spec['transfer']}:{spec['draw']}"
            t_start = time.time()
            try:
                if spec["transfer"] not in datas:
                    t1 = time.time()
                    datas[spec["transfer"]] = search.search_data(by_name[spec["transfer"]], bags)
                    setup["searchDataSeconds"][spec["transfer"]] = time.time() - t1
                data = datas[spec["transfer"]]
                record = None
                if spec["phase"] == 2:
                    gkey = (search.PHASE_ONE_GROUP_OF[spec["name"]], spec["transfer"])
                    if gkey not in records:
                        t1 = time.time()
                        records[gkey] = _phase_one_record(search, model_store, bag_data,
                                                          gkey[0], data, selections[gkey],
                                                          device, scratch, schedule)
                        setup["phaseOneRefitSeconds"][f"{gkey[0]}:{gkey[1]}"] = time.time() - t1
                    record = records[gkey]
                with GpuWitness(two_phase, torch) as witness:
                    out = run_trial(spec, data, record, device, scratch, schedule)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                row = {**{k: spec[k] for k in ("phase", "name", "transfer", "draw", "params")},
                       "arm": by_name[spec["transfer"]].arm, **out,
                       "gpuEvidence": witness.evidence(),
                       "outcomes": trial_outcomes([math.nan if v is None else v
                                                   for v in out["series"]]),
                       "seconds": time.time() - t_start}
            except Exception as exc:  # recorded, never hidden
                traceback.print_exc()
                errors[key] = f"{type(exc).__name__}: {exc}"
                row = {**{k: spec[k] for k in ("phase", "name", "transfer", "draw")},
                       "error": errors[key], "seconds": time.time() - t_start}
            with lines.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            done.append(key)
            print(f"[patience] {key} {row.get('seconds'):.1f}s "
                  f"{'ERROR ' + row['error'] if 'error' in row else row['outcomes']['patience10']}",
                  flush=True)
    result = {"device": evidence, "commit": os.environ.get("FORGE_CLONE_COMMIT"),
              "processedRoot": str(root), "schedule": schedule, "setup": setup,
              "trials": done, "errors": errors, "wallSeconds": time.time() - started}
    (target / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    if os.environ.get("FORGE_RUN_MODE"):
        stamp = {"evidence": {"commit": result["commit"], "device": evidence,
                              "trials": str(lines), "wallSeconds": result["wallSeconds"],
                              "errors": errors}}
        if not errors:
            stamp["evidence"]["complete"] = True
        Path("shard.json").write_text(json.dumps(stamp, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"wallSeconds": result["wallSeconds"], "errors": errors}), flush=True)
    return {"wallSeconds": result["wallSeconds"], "errors": errors}


if __name__ == "__main__":
    kwargs = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    run(**kwargs)
