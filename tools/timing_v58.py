"""Short local timing probe of the v58 BALLADEER campaign units, on CPU or MPS.

What it times, through the campaign's own per-split entry points on the real processed
data (`MIL-CREDA-EEG/Data/processed`, read only): for one transfer, outer fold 0 and its
inner split 0, each phase-1 group's training (`search.train_phase_one`, the call a
phase-1 trial makes per inner split) and each strategy's phase-2 training
(`search.train_phase_two`, the call a phase-2 trial makes per inner split, from the
group's phase-1 checkpoint stored and reloaded through `model_store`), for a few epochs;
and the centroid rule's source pass (`search.source_representations`) once per group.

Per-epoch seconds are read without touching any trainer: `two_phase.validation_record`,
which every one of the twelve trainers calls once at the end of each epoch, is wrapped to
stamp the clock and return what it returns. Epoch e's duration is stamp(e) - stamp(e-1)
for e >= 2, so the first epoch (warm-up: allocator, kernels, caches) is excluded; it is
kept apart as `firstEpochSeconds`.

Writes one JSON (the measurements, nothing extrapolated) to `--out`. Never writes into
the product: the phase-1 checkpoints go to a scratch directory (`--scratch`).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import optuna
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from MIL_CREDA_EEG import (assignment, bag_data, campaign, model_store, search,  # noqa: E402
                           two_phase)

PARAM_SEED = 20261106


class Stamps:
    """Wraps `two_phase.validation_record` to stamp the end of every epoch."""

    def __init__(self):
        self.rows: list[tuple[int, float]] = []
        self._original = two_phase.validation_record

    def __enter__(self):
        original = self._original

        def stamped(epoch, *args, **kwargs):
            out = original(epoch, *args, **kwargs)
            self.rows.append((int(epoch), time.perf_counter()))
            return out

        two_phase.validation_record = stamped
        return self

    def __exit__(self, *exc):
        two_phase.validation_record = self._original
        return False

    def epochs(self, start: float) -> dict:
        ends: dict[int, float] = {}
        for epoch, t in self.rows:
            ends[epoch] = max(t, ends.get(epoch, t))
        ordered = sorted(ends)
        steady = [ends[e] - ends[e - 1] for e in ordered if e - 1 in ends]
        first = ends[ordered[0]] - start if ordered else None
        return {"epochsRecorded": ordered, "steadyEpochSeconds": steady,
                "firstEpochSeconds": first}


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _params(space, seed: int) -> dict:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    return search.suggest(study.ask(), space)


def _windows(bags) -> int:
    return int(sum(b.windows.shape[0] for b in bags))


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def _cuda_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 2**20


def _mps_memory_mb() -> float | None:
    if not torch.backends.mps.is_available():
        return None
    return torch.mps.driver_allocated_memory() / 2**20


def _timed(fn, device):
    _sync(device)
    start = time.perf_counter()
    with Stamps() as stamps:
        try:
            out, error = fn(), None
        except Exception as exc:  # recorded, never hidden
            out, error = None, f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        _sync(device)
    total = time.perf_counter() - start
    return out, {"seconds": total, **stamps.epochs(start), "error": error,
                 "peakRssMB": _peak_rss_mb(), "mpsDriverMB": _mps_memory_mb(),
                 "cudaPeakMB": _cuda_memory_mb()}


def run(transfer_name: str, device_name: str, epochs: int, bags_per_step: int,
        scratch: Path, groups=None, strategies=None, phase2_epochs=None,
        processed_root=None) -> dict:
    device = torch.device(device_name)
    started = time.perf_counter()
    bags = bag_data.load_bags(processed_root)
    loaded = time.perf_counter()
    transfer = {t.name: t for t in assignment.TRANSFERS}[transfer_name]
    data = search.search_data(transfer, bags)
    view = data.inner_view(0, 0)
    built = time.perf_counter()
    strategies = list(strategies or model_store.STRATEGIES)
    groups = list(groups or sorted({search.PHASE_ONE_GROUP_OF[s] for s in strategies},
                                   key=model_store.GROUPS.index))
    campaign.register(strategies)
    schedule = {"epochs1": epochs, "epochs2": epochs, "bagsPerStep": bags_per_step}
    split = view.rotations[0]
    refit = data.rotations[0]
    out = {"transfer": transfer_name, "arm": transfer.arm, "device": str(device),
           "torch": torch.__version__, "threads": torch.get_num_threads(),
           "cpuCount": os.cpu_count(), "platform": platform.platform(),
           "python": platform.python_version(),
           "mpsFallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
           "measuredAt": datetime.now(timezone.utc).isoformat(), "epochs": epochs,
           "bagsPerStep": bags_per_step, "rotation": 0, "innerSplit": 0,
           "bags": {role: len(split[role]) for role in split},
           "windows": {role: _windows(split[role]) for role in split},
           "refitBags": {role: len(refit[role]) for role in refit},
           "refitWindows": {role: _windows(refit[role]) for role in refit},
           "setup": {"loadBagsSeconds": loaded - started,
                     "searchDataSeconds": built - loaded},
           "phase1": {}, "phase2": {}, "sourcePass": {}}
    records = {}
    for group in groups:
        params = _params(search.PHASE_ONE_SPACES[group], PARAM_SEED)
        hyper = search.phase_one_hyper(group, params, schedule)
        trained, timing = _timed(lambda: search.train_phase_one(
            group, view, 0, search.PHASE_ONE_SEED, hyper, device), device)
        out["phase1"][group] = {**timing, "params": params}
        if trained is None:
            continue
        path = search.save_phase_one(scratch / "Models", group, view, 0,
                                   search.PHASE_ONE_SEED, trained)
        records[group] = model_store.load(path, partition_path=bag_data.default_partition_path())
        _, timing = _timed(lambda: search.source_representations(
            group if group != "common" else "UDA-DDA", data, 0, records[group], device), device)
        out["sourcePass"][group] = timing
    for strategy in strategies:
        group = search.PHASE_ONE_GROUP_OF[strategy]
        if group not in records:
            out["phase2"][strategy] = {"error": f"no phase-1 checkpoint of {group}"}
            continue
        params = _params(search.phase_two_space(strategy), PARAM_SEED)
        own = {**schedule, "epochs2": (phase2_epochs or {}).get(strategy, epochs)}
        _, timing = _timed(lambda: search.train_phase_two(strategy, view, 0, search.SEARCH_SEED, records[group],
                                                         params, own, device), device)
        out["phase2"][strategy] = {**timing, "group": group, "epochs": own["epochs2"]}
    out["wallSeconds"] = time.perf_counter() - started
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transfer", default="L1->L11")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bags-per-step", type=int, default=8)
    parser.add_argument("--groups", default="")
    parser.add_argument("--strategies", default="")
    parser.add_argument("--phase2-epochs", default="",
                        help="per-strategy phase-2 epochs, e.g. CCA-LSC=4 (CCA-LSC needs its "
                             "3 warm-up epochs plus one calibrated epoch)")
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = run(args.transfer, args.device, args.epochs, args.bags_per_step,
                 Path(args.scratch),
                 [g for g in args.groups.split(",") if g] or None,
                 [s for s in args.strategies.split(",") if s] or None,
                 {k: int(v) for k, v in (item.split("=") for item in
                                         args.phase2_epochs.split(",") if item)})
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("device", "transfer", "wallSeconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
