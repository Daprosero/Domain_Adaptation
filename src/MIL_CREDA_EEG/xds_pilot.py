"""Pilot: cross-dataset ADHD transfer between IEEE (Nasrabadi et al.) and BALLADEER EMOTIV tasks.

Diagnostic outside any plan, no augmentation: every subject is one bag, as in plan v63.
EEGNet under the v63 rules with the v59 corrections and the fixed values of `v64_pilot`; the
training and scoring code is `v64_pilot.train_and_score`, unchanged.

Data. Both datasets read the 10 channels they share (`SHARED_X`, 10-20 names). IEEE: the last
key of each .mat, transposed to C x T (the STF-KernelSHAP loader), 50 Hz notch (Q 30) as that
loader does, then cut into non-overlapping 4 s windows (512 samples at 128 Hz). BALLADEER: the
stored 4 s windows. Then, for both, the v63 instance rule: at most 60 windows at step
ceil(n / 60), 1-45 Hz zero-phase band-pass, common average over the shared channels.

Partition, subject-wise. The two datasets share no subject, so the source trains on ALL its
subjects. The target is split in 4 folds: BALLADEER keeps its frozen partition fold; IEEE is
stratified by group (subjects sorted by file name, group member i in fold i % 4). Rotation r:
the target subjects outside fold r adapt (unlabelled); fold r is tested. Inputs standardized
per channel by the source training windows. MIL-CREDA's phase 2 runs with the gate at 1 (no
v63 shift-gate value exists for these transfers).

A unit id is "<source>-><target>:<rotation>", one of IEEE->Robots, Robots->IEEE,
IEEE->CogniFit, CogniFit->IEEE.
"""

from __future__ import annotations

import glob
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import scipy.io
import torch
from scipy.signal import filtfilt, iirnotch, sosfiltfilt

from MIL_CREDA_EEG import assignment as A
from MIL_CREDA_EEG import remote_units, search, two_phase
from MIL_CREDA_EEG import v63_campaign as C
from MIL_CREDA_EEG import v63_setup as V
from MIL_CREDA_EEG import v64_pilot as P

RESULTS = Path("Results") / "xds-pilot"
#: The STF-KernelSHAP channel order of the IEEE recordings.
IEEE_CHANNELS = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
                 "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz")
SHARED_X = ("F7", "F3", "T7", "P7", "O1", "O2", "P8", "T8", "F4", "F8")
RATE, WINDOW, FOLDS = 128, 512, 4
TRANSFERS = ("IEEE->Robots", "Robots->IEEE", "IEEE->CogniFit", "CogniFit->IEEE")


def find_ieee(root: str | None = None) -> str:
    if root and glob.glob(f"{root}/ADHD_group/*.mat"):
        return root
    hits = sorted(glob.glob("/kaggle/input/**/TDAH/ieee/ADHD_group", recursive=True))
    if not hits:
        raise FileNotFoundError("no IEEE ADHD_group folder under /kaggle/input")
    return str(Path(hits[0]).parent)


def instances(w: np.ndarray) -> torch.Tensor:
    """The v63 instance rule on (n, C, 512) windows already restricted to SHARED_X."""
    w = w[::max(1, math.ceil(len(w) / V.WINDOWS))][:V.WINDOWS]
    w = sosfiltfilt(V.SOS, w - w.mean(-1, keepdims=True), axis=-1)
    w = w - w.mean(1, keepdims=True)
    return torch.from_numpy(np.ascontiguousarray(w)).float()


def ieee_bags(root: str) -> list[dict]:
    b, a = iirnotch(50.0, 30.0, fs=RATE)
    pick = [IEEE_CHANNELS.index(c) for c in SHARED_X]
    out = []
    for group, label in (("ADHD_group", 1), ("Control_group", 0)):
        files = sorted(glob.glob(f"{root}/{group}/*.mat"))
        for i, f in enumerate(files):
            d = scipy.io.loadmat(f)
            x = np.asarray(d[list(d.keys())[-1]], dtype=np.float64).T      # C x T
            if x.shape[0] != len(IEEE_CHANNELS):
                raise ValueError(f"{f}: {x.shape[0]} channels, expected 19")
            x = filtfilt(b, a, x, axis=-1)[pick]
            n = x.shape[1] // WINDOW
            w = x[:, :n * WINDOW].reshape(len(pick), n, WINDOW).transpose(1, 0, 2)
            out.append({"bag": two_phase.BagTensor(f"IEEE-{Path(f).stem}", "IEEE", label,
                                                   instances(w)),
                        "fold": i % FOLDS})
    return out


def balladeer_bags(bags, condition: str) -> list[dict]:
    out = []
    for b in bags.bags:
        if b.condition != condition:
            continue
        w = np.asarray(b.windows(), dtype=np.float64)
        names = list(b.channels)
        w = w[:, [names.index(c) for c in SHARED_X]]
        label = int(b.label if isinstance(b.label, (int, np.integer))
                    else two_phase.CLASSES.index(b.label))
        out.append({"bag": two_phase.BagTensor(b.subject, condition, label, instances(w)),
                    "fold": int(b.fold)})
    return out


def view_of(domains: Mapping[str, list[dict]], name: str, rotation: int):
    source, target = name.split("->")
    tgt = domains[target]
    view = {"source_train": [d["bag"] for d in domains[source]], "source_valid": [],
            "target_train": [d["bag"] for d in tgt if d["fold"] != rotation],
            "target_valid": []}
    test = [d["bag"] for d in tgt if d["fold"] == rotation]
    t = A.Transfer("dataset", source, target)
    data = search.SearchData(transfer=t, partition_sha256=f"xds-pilot:{name}", rotations={rotation: view})
    return t, data, view, test


def record_path(product: Path, name: str, rotation: int, seed: int) -> Path:
    return (product / RESULTS / name.replace("->", "-to-") / f"rotation{rotation}"
            / f"whole__seed{seed}.json")


def run(config: Mapping, product: str | Path | None = None, *, units: Sequence[str],
        seeds: Sequence[int] = (1, 2, 3), epochs1: int = P.EPOCHS1, epochs2: int = P.EPOCHS2,
        ieee: str | None = None, workdir: str | Path | None = None,
        environ: Mapping[str, str] | None = None) -> dict:
    environ = os.environ if environ is None else environ
    product = Path(product) if product else C.DEFAULT_PRODUCT
    workdir = Path(workdir) if workdir else Path.cwd()
    config = {"schedule": {"epochs1": epochs1, "epochs2": epochs2,
                           "bagsPerStep": P.BAGS_PER_STEP}, **config}
    cfg = C._setup(config, environ)
    deadline = float(environ.get(remote_units.DEADLINE_ENV, "inf"))
    placement = remote_units.device_placement(cfg, environ)
    deterministic = remote_units.set_deterministic()
    identity = {"commit": remote_units.source_commit(environ), "codeVersion": V.code_version(),
                "revision": "xds-pilot"}
    environment = remote_units.environment_stamp(deterministic, identity, placement)
    device = cfg["device"]
    bags = C.campaign.load_bags(cfg)
    domains = {"IEEE": ieee_bags(find_ieee(ieee)),
               "Robots": balladeer_bags(bags, "Robots"),
               "CogniFit": balladeer_bags(bags, "CogniFit")}
    counts = {k: {"subjects": len(v), "adhd": sum(d["bag"].label for d in v),
                  "windowsMedian": int(np.median([d["bag"].windows.shape[0] for d in v]))}
              for k, v in domains.items()}
    print("domains", counts, flush=True)
    ran, skipped, longest, started = [], [], 0.0, time.time()
    todo = [(u, s) for u in units for s in seeds]
    with P.eegnet_rules():
        for u, seed in todo:
            name, rotation = u.rsplit(":", 1)
            rotation = int(rotation)
            if name not in TRANSFERS:
                raise ValueError(f"{name!r} is not one of {TRANSFERS}")
            path = record_path(product, name, rotation, seed)
            if path.is_file():
                skipped.append(f"{u}/{seed}")
                continue
            if time.time() + 1.5 * longest + C.DEFAULT_MARGIN > deadline:
                break
            t0 = time.time()
            t, data, view, test = view_of(domains, name, rotation)
            test = P.standardize(view, test)
            out = {"transfer": name, "rotation": rotation, "arm": "whole", "seed": seed,
                   "trunk": None, "fixed": P.FIXED, "epochs1": epochs1, "epochs2": epochs2,
                   "channels": list(SHARED_X), "domains": counts,
                   "nUnits": {"source": len(view["source_train"]),
                              "target": len(view["target_train"]), "test": len(test)}}
            out = P.train_and_score(data, view, test, rotation, seed, device, epochs1, epochs2,
                                    (("milCreda", 1.0),), out, t0)
            longest = max(longest, out["seconds"])
            C.write_record(path, {**out, "environment": environment})
            ran.append(f"{u}/{seed}")
            print(u, seed, out["trunk"], out["nUnits"], f"{out['seconds']}s", flush=True)
    remaining = [f"{u}/{s}" for u, s in todo if f"{u}/{s}" not in ran + skipped]
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
