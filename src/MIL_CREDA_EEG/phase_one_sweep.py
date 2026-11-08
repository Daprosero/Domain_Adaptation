"""Phase-1 sweep outside plan v59's search space: filters, regularization, learning rate.

Diagnostic, not a campaign step. It answers whether phase 1 of MIL-CREDA can learn the
task pair under other settings, reading held-out source and target subjects of every
outer rotation. It patches `eegnet` constants in memory, writes nothing to the model
store and reports no result of the plan. Settings are compared on source subjects only;
target numbers are read, never used to choose.

A unit is `sweep.<config>.<transfer>.seed<s>`, one JSON file under `sweep/` in the
working directory. Stored hyperparameters come from `phase_one_sweep_selections.json`,
the selections the v59 search stored for the two task transfers.

**BatchNorm.** The plan's momentum of 0.01 leaves the running statistics far from the batch
statistics after the few steps these runs take, and evaluation reads them. Every side is
scored after `recalibrate` recomputes them from the training bags, weights untouched.

**Decision rule, fixed before the run.** Per subject, the mean probability over the three
seeds. Each configuration against `stored-40` on held-out source subjects: paired
permutation test on the per-subject correct-class probability, Benjamini-Hochberg over the
configurations, Friedman across them. The target side is read, never tested for choosing.
These source subjects are outer test folds, so a winner here is a lead only: a later plan
must select it again inside the inner splits, and this sweep is never reported as a result.
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
from sklearn.metrics import roc_auc_score

from MIL_CREDA_EEG import assignment as A
from MIL_CREDA_EEG import campaign, eegnet, model_store, remote_units, search, two_phase
from MIL_CREDA_EEG.patience_exercise import GpuWitness

SELECTIONS = Path(__file__).with_name("phase_one_sweep_selections.json")
TRANSFERS = {"RC": "Robots->CogniFit", "CR": "CogniFit->Robots", "L11": "L11->L1"}
SEEDS = (1, 2, 3)
EPOCHS = 40
OUTPUT = "sweep"
BASE = dict(F1=8, D=2, F2=16, KERNEL_LENGTH=64, SEPARABLE_LENGTH=16, POOL1=4, POOL2=8,
            DROPOUT=0.25, HEAD_MAX_NORM=0.25, DEPTHWISE_MAX_NORM=1.0, BN_MOMENTUM=0.01)
OPEN = dict(lr=1e-3, HEAD_MAX_NORM=1.0)
WIDE = dict(F1=16, F2=32)
CONFIGS: dict[str, dict] = {
    "stored-E": dict(epochs="stored"),
    "stored-40": dict(),
    "filt-E": dict(epochs="stored", band="filt"),
    "cap72-E": dict(epochs="stored", cap=72),
    "lr1e-3": dict(lr=1e-3),
    "head1": dict(OPEN),
    "drop0": dict(OPEN, DROPOUT=0.0),
    "drop5": dict(OPEN, DROPOUT=0.5),
    "wd2": dict(OPEN, wd=1e-2),
    "wide": dict(OPEN, **WIDE),
    "wide-k32": dict(OPEN, KERNEL_LENGTH=32, **WIDE),
    "wide-k128": dict(OPEN, KERNEL_LENGTH=128, **WIDE),
    "small-reg": dict(OPEN, wd=1e-2, DROPOUT=0.5, F1=4, F2=8),
    "small": dict(F1=4, D=2, F2=8),
    "tiny": dict(F1=2, D=1, F2=2),
    "d4": dict(OPEN, D=4, F2=32),
    "d1": dict(OPEN, D=1, F2=8),
    "sep32": dict(OPEN, SEPARABLE_LENGTH=32),
    "sep8": dict(OPEN, SEPARABLE_LENGTH=8),
    "pool-fine": dict(OPEN, POOL1=2, POOL2=4),
    "pool-coarse": dict(OPEN, POOL1=8, POOL2=16),
    "depth-free": dict(OPEN, DEPTHWISE_MAX_NORM=10.0),
}


def unit_ids() -> list[str]:
    """Every unit, cheap transfer first within a seed, so a place holds a mix."""
    return [f"sweep.{c}.{t}.seed{s}" for s in SEEDS for t in TRANSFERS for c in CONFIGS]


def parse(unit: str) -> tuple[str, str, int]:
    _, config, transfer, seed = unit.split(".")
    return config, TRANSFERS[transfer], int(seed.removeprefix("seed"))


@contextlib.contextmanager
def patched(spec: Mapping):
    """The `eegnet` constants of one configuration, restored on exit."""
    values = {k: spec.get(k, v) for k, v in BASE.items()}
    saved = {k: getattr(eegnet, k) for k in BASE}
    for key, value in values.items():
        setattr(eegnet, key, value)
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(eegnet, key, value)


SOS = None


def prepared(bags, spec: Mapping) -> list:
    """The input the configuration asks for: 1-45 Hz with common average, or a window cap."""
    global SOS
    if not spec.get("band") and not spec.get("cap"):
        return bags
    out = []
    for b in bags:
        w = b.windows
        if spec.get("cap") and w.shape[0] > spec["cap"]:
            keep = torch.linspace(0, w.shape[0] - 1, spec["cap"]).round().long()
            w = w[keep]
        if spec.get("band") == "filt":
            from scipy.signal import butter, sosfiltfilt
            SOS = SOS if SOS is not None else butter(4, [1, 45], "bandpass", fs=128, output="sos")
            x = w.double().numpy()
            x = sosfiltfilt(SOS, x - x.mean(-1, keepdims=True), axis=-1)
            w = torch.from_numpy((x - x.mean(1, keepdims=True)).copy()).float()
        out.append(two_phase.BagTensor(b.subject, b.condition, b.label, w))
    return out


def balanced_accuracy(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    y, hat = np.array(labels), (np.array(probabilities) >= 0.5).astype(int)
    return float(np.mean([(hat[y == c] == c).mean() for c in (0, 1)]))


def _side(model, bags, device) -> list[dict]:
    p = two_phase.evaluate(model, bags, "source", device)[:, 1].detach().cpu().numpy()
    return [{"subject": b.subject, "label": b.label, "probability": float(q)}
            for b, q in zip(bags, p)]


def recalibrate(model, bags, device: torch.device, chunk: int = 512,
                trunk: str = "source_trunk") -> None:
    """Recompute one trunk's BatchNorm statistics from `bags`; no weight moves."""
    encoder = getattr(model, trunk)
    layers = [m for m in encoder.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    momenta = [m.momentum for m in layers]
    model.eval()
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None                  # a cumulative average over every batch
        layer.train()
    with torch.no_grad():
        for bag in bags:
            for start in range(0, bag.windows.shape[0], chunk):
                encoder(bag.windows[start:start + chunk].to(device))
    for layer, momentum in zip(layers, momenta):
        layer.momentum = momentum
        layer.eval()


def _summary(rows: Sequence[Mapping]) -> dict:
    y, p = [r["label"] for r in rows], [r["probability"] for r in rows]
    if not np.isfinite(p).all():
        return {"ba": None, "auc": None, "n": len(rows), "diverged": True}
    auc = float(roc_auc_score(y, p)) if np.ptp(p) > 0 else 0.5
    return {"ba": round(balanced_accuracy(y, p), 4), "auc": round(auc, 4), "n": len(rows)}


def run_unit(unit: str, cfg: Mapping, bags, device: torch.device,
             epochs_override: int | None = None) -> dict:
    name, transfer_name, seed = parse(unit)
    spec = CONFIGS[name]
    stored = json.loads(SELECTIONS.read_text(encoding="utf-8"))[transfer_name]
    transfer = {t.name: t for t in A.TRANSFERS}[transfer_name]
    a = A.assign_from_bags(transfer, bags)
    groups = ("train", "source", "target")
    sides: dict[str, list] = {k: [] for k in groups}
    curves, started = [], time.time()
    with patched(spec):
        for r in range(4):
            params = dict(stored[str(r)]["params"])
            params["lr1"] = spec.get("lr", params["lr1"])
            params["wd1"] = spec.get("wd", params["wd1"])
            epochs = int(stored[str(r)]["epochs"]) if spec.get("epochs") == "stored" else EPOCHS
            epochs = epochs_override or epochs
            hyper = search.phase_one_hyper("MIL-CREDA", params,
                                           search.refit_schedule(cfg["schedule"], 1, epochs))
            train = prepared(two_phase.bag_tensors(A.side_bags(a, bags, "source", r, A.REFIT_ROLE)), spec)
            source = prepared(two_phase.bag_tensors(A.side_bags(a, bags, "source", r, "test")), spec)
            target = prepared(two_phase.bag_tensors(A.side_bags(a, bags, "target", r, "test")), spec)
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            model = model_store.build_model("MIL-CREDA", int(train[0].windows.shape[1]),
                                            int(target[0].windows.shape[1]), hyper).to(device)
            result = two_phase.train_source_phase(model, train, source, hyper, generator, device,
                                                  seed=seed, fold=r, transfer=transfer_name)
            curves.append([round(float(h["val_balanced_accuracy"]), 4)
                           for h in result["history"]])
            recalibrate(model, train, device)
            for side, group in zip(groups, (train, source, target)):
                sides[side].extend(_side(model, group, device))
            del model, train, source, target
    return {"unit": unit, "config": name, "transfer": transfer_name, "seed": seed,
            "spec": spec, "seconds": round(time.time() - started),
            **{side: _summary(rows) for side, rows in sides.items()},
            "sourceCurves": curves, "predictions": sides}


def run(config: Mapping, product: str | Path | None = None, *,
        units: Sequence[str] | None = None, workdir: str | Path | None = None,
        epochs: int | None = None, environ: Mapping[str, str] | None = None) -> dict:
    """The units named by `FORGE_RUN_UNITS`, or the caller, into `<workdir>/sweep/`."""
    environ = os.environ if environ is None else environ
    workdir = Path(workdir) if workdir else Path.cwd()
    placement = remote_units.device_placement(config, environ)
    cfg = remote_units.resolve_inputs(campaign.check_config(config), environ=environ)
    campaign.register(cfg["strategies"])
    raw = environ.get(remote_units.UNITS_ENV)
    chosen = list(units) if units else json.loads(raw) if raw else unit_ids()
    unknown = sorted(set(chosen) - set(unit_ids()))
    if unknown:
        raise ValueError(f"unknown sweep units: {unknown}")
    environment = remote_units.environment_stamp(remote_units.set_deterministic(),
                                                 remote_units.campaign_identity(cfg, environ),
                                                 placement)
    device = torch.device(cfg.get("device", "cpu"))
    deadline = float(environ.get(remote_units.DEADLINE_ENV, "inf"))
    bags = campaign.load_bags(cfg)
    folder = workdir / OUTPUT
    folder.mkdir(parents=True, exist_ok=True)
    ran, skipped, failed = [], [], {}
    for index, unit in enumerate(chosen):
        path = folder / f"{unit}.json"
        if path.exists():
            skipped.append(unit)
            continue
        if time.time() > deadline - 900:
            (workdir / "partial.json").write_text(json.dumps({"remaining": chosen[index:]}),
                                                  encoding="utf-8")
            break
        try:
            record = run_unit(unit, cfg, bags, device, epochs)
        except Exception as error:             # one bad unit never ends the session
            failed[unit] = repr(error)
            print({"unit": unit, "failed": repr(error)}, flush=True)
            continue
        record["environment"] = {k: environment.get(k) for k in ("commit", "codeVersion", "device")}
        path.write_text(json.dumps(record, default=str), encoding="utf-8")
        print({k: record[k] for k in ("unit", "seconds", "train", "source", "target")}, flush=True)
        ran.append(unit)
    (workdir / "sweep-run.json").write_text(
        json.dumps({"ran": ran, "skipped": skipped, "failed": failed, "environment": environment},
                   indent=1, default=str), encoding="utf-8")
    return {"ran": ran, "skipped": skipped, "failed": failed, "environment": environment}


def smoke(config: Mapping, product: str | Path | None = None, *,
          units: Sequence[str] | None = None, workdir: str | Path | None = None,
          epochs: int | None = 2, budget_seconds: float | None = None,
          environ: Mapping[str, str] | None = None) -> dict:
    """The rehearsal: a short run of `units`, with the evidence `smoke record` reads."""
    environ = os.environ if environ is None else environ
    workdir = Path(workdir) if workdir else Path.cwd()
    chosen = list(units or unit_ids()[:1])
    mode = environ.get(remote_units.RUN_MODE_ENV)
    with GpuWitness(two_phase, torch) if mode else contextlib.nullcontext() as witness:
        outcome = run(config, product, units=chosen, workdir=workdir, epochs=epochs,
                      environ=environ)
    if mode:
        env = outcome["environment"]
        payload = {"mode": mode,
                   "evidence": {"commit": env["commit"], "sourcesDigest": env["codeVersion"],
                                "codeVersion": env["codeVersion"], "unit": chosen[0],
                                "units": chosen, "complete": set(outcome["ran"]) >= set(chosen),
                                "gpu": witness.evidence(), "estimates": {}},
                   "environment": env,
                   "runs": {k: v for k, v in outcome.items() if k != "environment"}}
        (workdir / remote_units.STAMP_FILE).write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8")
    return outcome
