"""Training and measuring. It owns how a run happens, never what a run is.

What each arm computes lives in `wiring`; which material it sees lives in `bags`;
every number that defines the experiment lives in `config`. This file only drives
them and records what came out.

Two schedules are CREDA's own and are applied to every arm without exception: the
warm-up of the balance coefficient and the decay of the learning rate. Applying
them to one side only would add a difference nobody is measuring.

Read every number with the header printed beside it. With fewer than three
repetitions the dispersion is zero, so the threshold is zero and every row
declares a winner from a bare difference; that is stamped rather than hidden,
because the pilot has to exercise the same path the full run will.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import zlib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import torch
import torch.nn as nn

from CREDA.schedules import creda_ramp
from MIL_CREDA_Benchmark import bags, ceiling_record, config, report_digest, tables, wiring
from MIL_CREDA_Benchmark.schedules import milcreda_ramp
from MIL_CREDA_Benchmark.verdict import judge, render, standard_error, tally


# ------------------------------------------------------------------ environment

def resolve_device() -> torch.device:
    """The same preference order `CREDA.training_pipeline` uses."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def environment() -> dict:
    """Where this ran, recorded rather than assumed.

    The guard exists because a run's provenance describes whichever environment
    produced it. It refuses when the repository has its own virtualenv and
    something else is running — that is a mistake worth stopping. On a hosted
    runtime there is no such virtualenv and nothing to compare against, so the
    environment is stamped into the summary instead: a table made in Colab is
    labelled as made in Colab rather than attributed to this machine.
    """
    prefix = Path(sys.prefix).resolve()
    inside = prefix.is_relative_to(config.REPOSITORY)
    if (config.REPOSITORY / ".venv").is_dir() and not inside:
        raise SystemExit(
            f"refusing to run under {prefix}\n"
            f"  this repository has its own virtualenv and the measurement is of "
            f"an environment; run with {config.REPOSITORY}/.venv/bin/python."
        )
    return {
        "interpreter": str(prefix),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "selfHosted": inside,
        "power": power_state(),
        "device": device_class(),
    }


def device_class() -> dict:
    """Which accelerator this actually got, not which one was asked for.

    Requesting a class and receiving it are two obligations, and only the second
    is a fact. A remote service allocates by availability, so a run can ask for
    one accelerator and land on another without a word, and the device class is
    part of a run's own provenance regardless of whether anything times it.

    The name is what the driver reports; `kind` is the backend, which is what
    survives when a platform gives no model name at all.
    """
    if torch.cuda.is_available():
        try:
            return {"name": torch.cuda.get_device_name(0), "kind": "cuda"}
        except Exception:
            return {"name": "cuda", "kind": "cuda"}
    if torch.backends.mps.is_available():
        return {"name": "mps", "kind": "mps"}
    return {"name": platform.processor() or "cpu", "kind": "cpu"}


#: What makes two runs the same machine for the purpose of a cost measurement.
#:
#: `power.charge` is deliberately absent: it moves while a run is going and is not
#: a class of machine. `power.source` is present because throttling on battery is
#: a real difference between arms, which is the reason the stamp exists at all.
ENVIRONMENT_KEYS = ("python", "platform", "torch", "selfHosted")


def environment_key(stamp: dict) -> str:
    """A short handle for one machine, so a run can carry it without the whole stamp.

    Runs reference this; the full stamps live once beside them. Comparing handles
    is what lets a merge group cost dimensions by machine instead of pooling them
    into a mean that describes none of the machines involved.
    """
    device = stamp.get("device") or {}
    power = stamp.get("power") or {}
    material = [str(stamp.get(k)) for k in ENVIRONMENT_KEYS]
    material += [str(device.get("name")), str(device.get("kind")),
                 str(power.get("source"))]
    return hashlib.sha256("|".join(material).encode("utf-8")).hexdigest()[:12]


def power_state() -> dict:
    """Whether the machine was on mains when the run happened.

    Recorded as provenance rather than as a guard against a comparison
    dimension -- time and memory (`seconds`/`peakMiB`) are removed from this
    comparison entirely, so the power state no longer protects either of them
    from a throttled machine. It is kept because it is cheap, best-effort, and
    still a fact about the environment a run happened in.

    Recorded rather than enforced. Refusing to run on battery would stop work that
    is often fine — the ceiling search measures accuracy, which is deterministic
    and does not care. What must not happen is a throttled run being filed
    alongside a clean one with nothing to tell them apart.

    Best-effort and never fatal: an unreadable power state is reported as unknown,
    because a stamp that crashed the run it was documenting would be worse than no
    stamp at all.
    """
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["pmset", "-g", "ps"], capture_output=True,
                                 text=True, timeout=5)
            first = out.stdout.splitlines()[0] if out.stdout else ""
            source = ("mains" if "AC Power" in first
                      else "battery" if "Battery Power" in first else "unknown")
            charge = re.search(r"(\d+)%", out.stdout)
            return {"source": source,
                    "charge": int(charge.group(1)) if charge else None}
        online = Path("/sys/class/power_supply/AC/online")
        if online.exists():
            return {"source": "mains" if online.read_text().strip() == "1"
                    else "battery", "charge": None}
    except Exception:
        pass
    return {"source": "unknown", "charge": None}


# -------------------------------------------------------------------- schedules

def ramp(epoch: int, epochs: int, family: str | None,
         delta: float = config.RAMP_DELTA,
         ceiling: float = config.RAMP_CEILING) -> float:
    """The arm's adaptation coefficient, from its own method's schedule.

    Each family names its own entry point — `creda_ramp` for prior work,
    `milcreda_ramp` for the method — and both are given the same `delta` and the
    same `ceiling` here, explicitly. That is how each method keeps the default it
    was defined with for its own runs while the two arms of this comparison share
    one coefficient: the defaults are never what the benchmark uses.

    Calling one method's schedule for both families would have been shorter and
    would have made MIL-CREDA's coefficient come out of prior work's module,
    which is a dependency nobody declared. The curve is still written once —
    `milcreda_ramp` binds to `creda_ramp` rather than copying it, because two
    implementations of one formula across two arms is the fork this package
    exists not to have. Same numbers by construction, and pinned by a test.

    A floor with no adaptation term passes `None` and gets zero: it has no
    coefficient, and handing it one would suggest a term it does not carry.
    """
    if family is None:
        return 0.0
    schedule = creda_ramp if family == "creda" else milcreda_ramp
    return schedule(epoch, epochs, delta=delta, ceiling=ceiling)


def learning_rate(epoch: int, epochs: int) -> float:
    """One fixed, declared rate for every arm, at every epoch: `config.LR`.

    The decay of CREDA's own `get_eta` (`LR_ALPHA`/`LR_BETA`) is removed: this
    stretch's own decision replaces the dynamic schedule with a single constant,
    not searched, identical for every arm. `epoch`/`epochs` stay in the
    signature so every call site -- which passes them unconditionally, once per
    epoch -- does not have to change, and so a future schedule has the same two
    numbers this one never needed.
    """
    del epoch, epochs
    return config.LR


def balanced_batches(targets: list[int], steps: int, generator: torch.Generator):
    """One bag of every class per step, so no class is ever missing.

    `total_correspondence` refuses to invent a value for a class with no source
    bag, and with ten bags drawn at random out of ten classes one goes missing
    often enough to kill an epoch. Each class keeps its own shuffled queue and
    refills when it runs out, so the draw stays stratified across the epoch and
    the arms all see the same shape of batch.
    """
    queues: dict[int, list[int]] = {c: [] for c in set(targets)}
    by_class: dict[int, list[int]] = {c: [] for c in set(targets)}
    for position, label in enumerate(targets):
        by_class[label].append(position)

    for _ in range(steps):
        batch = []
        for class_id in sorted(by_class):
            if not queues[class_id]:
                pool = by_class[class_id]
                order = torch.randperm(len(pool), generator=generator).tolist()
                queues[class_id] = [pool[i] for i in order]
            batch.append(queues[class_id].pop())
        yield batch


# ------------------------------------------------------------------ the reduction

@dataclass
class Reduction:
    """The bounds a number was obtained under, carried beside it."""

    setting: str = "trained"
    revision: str = config.REVISION
    backbone: str = config.BACKBONE
    instancesPerBag: int = config.INSTANCES_PER_BAG
    bagsPerDomain: int = config.BAGS_PER_DOMAIN
    trainBags: int = config.TRAIN_BAGS
    validBags: int = config.VALID_BAGS
    evalBags: int = config.EVAL_BAGS
    #: Which role the ceiling search read. Recorded because "chosen on material
    #: the verdict never saw" is a claim about the run, not about the code.
    searchRole: str = config.SEARCH_ROLE
    epochs: int = config.EPOCHS
    seeds: list[int] = field(default_factory=lambda: list(config.SEEDS))
    #: La fracción de cada bolsa de ENTRENAMIENTO reemplazada por imágenes de
    #: otras clases. Va en las cotas y no en una bandera de módulo: una tabla de
    #: exactitudes no dice nada sin la tasa que llevaba el material, y una tasa
    #: que sólo viviera en `config` llegaría a la corrida y nunca al registro de
    #: al lado. Los roles de selección y evaluación están limpios en toda tasa.
    labelNoise: float = config.NOISE
    #: Si esta corrida es un ensayo. Decide dónde escribe, y un registro que no
    #: dice que es de ensayo es exactamente cómo un número de piloto termina
    #: citado como resultado. La escala sigue siendo `epochs`/`seeds`.
    pilot: bool = False
    #: Qué forma de corrida es: `"campaign"` (cada transferencia a una tasa) o
    #: `"curve"` (una transferencia a través de cada nivel). Acá y no como
    #: argumento suelto, porque es la tercera coordenada del destino y las tres
    #: tienen que llegar juntas a cada escritor: mientras `kind` viajaba aparte,
    #: la campaña escribía sus corridas en un árbol y su sello en otro.
    kind: str = "campaign"
    #: The neutral each family's searched ceiling is read against.
    rampCeiling: float = config.RAMP_CEILING
    #: Decision 1's one bandwidth, and Eq. (16)/(28)'s three hyperparameters,
    #: stamped beside every other bound. A checkpoint whose manifest recorded
    #: a different value for any of the four than the RUN that produced it --
    #: never than today's bare `config` default, see `hyper_for` below -- was
    #: trained under a different objective, exactly as one trained under an
    #: earlier `revision` was: `latent.load` refuses on a mismatch here the
    #: same way it refuses on a mismatched `revision`.
    #:
    #: Ordinary constructor parameters, and no longer `init=False`. They used
    #: to stamp only `config`'s own bare default because nothing downstream
    #: read anything else: the six-dimensional search
    #: (`search_ceilings_trials`) explores all four per transfer and records
    #: its winners, but `run_one`/`wiring.build` never consumed them -- a
    #: searched value was recorded, never applied, and every campaign trained
    #: at the same declared constant regardless of what the search found.
    #: `hyper_for` is what closes that: it resolves the searched winner for
    #: the transfer actually being run, falling back to these four scalars
    #: -- still `config`'s own defaults, unchanged -- only where nothing was
    #: searched. Keeping them as plain fields is what lets `keep_median` stamp
    #: a PER-TRANSFER copy of this reduction (`dataclasses.replace`) into each
    #: checkpoint's own manifest, rather than the campaign's one shared,
    #: pooled instance.
    kernelSigma: float = field(default_factory=lambda: config.KERNEL_SIGMA)
    attentionGamma: float = field(default_factory=lambda: config.ATTENTION_GAMMA)
    attentionTemperature: float = field(
        default_factory=lambda: config.ATTENTION_TEMPERATURE)
    #: Eq. (28)'s local temperature, the fourth searched dimension `Reduction`
    #: did not carry until now -- `run_one` accepted it only through the
    #: search's own explicit `hyper=` override, with no field here for a
    #: campaign to stamp. Same rule as the three above: `hyper_for`'s winner
    #: first, this scalar (`config.TAU_LOCAL`) only where nothing was searched.
    tauLocal: float = field(default_factory=lambda: config.TAU_LOCAL)
    #: What each family searched and kept for its derivations. Empty until the
    #: search has run, and then carried beside every number it produced — a
    #: coefficient chosen by measurement is part of the bounds, not a detail.
    ceilings: dict = field(default_factory=lambda: dict(config.CEILINGS))
    #: The ceiling of each family on each transfer the search actually measured.
    #: A transfer absent from here inherits `ceilings`, and that fallback is the
    #: declared rule rather than a default: the four transfers the search never
    #: saw run at the winner of the two it did, out of sample.
    ceilingsByTransfer: dict = field(
        default_factory=lambda: {family: dict(picks) for family, picks
                                 in config.CEILINGS_BY_TRANSFER.items()})
    #: The other four searched dimensions' per-transfer picks -- `rampDelta`,
    #: `kernelSigma`, `attentionGamma`, `attentionTemperature`, `tauLocal` --
    #: keyed the same way `ceilingsByTransfer` is (`{family: {label: {dim:
    #: value}}}`). `hyper_for` reads this before falling back to the scalar
    #: fields above, the identical two-reading rule `ceiling_for` already
    #: applies to the coefficient.
    hyperByTransfer: dict = field(default_factory=dict)
    ceilingSearch: dict = field(default_factory=dict)
    rampDelta: float = config.RAMP_DELTA
    device: str = "cpu"
    environment: dict = field(default_factory=dict)

    @property
    def verdicts_meaningful(self) -> bool:
        return len(self.seeds) >= 3

    @staticmethod
    def from_record(d: dict) -> "Reduction":
        """A `Reduction`, rebuilt from a record's own `"reduction"` field.

        `Reduction(**d)` crashes on this input: `kernelSigma`, `attentionGamma`
        and `attentionTemperature` are `init=False`, and `dataclasses` refuses
        any keyword naming a non-init field. That crash is what
        `Benchmark_Report_v1.ipynb` hit rebuilding a `Reduction` from
        `summary["reduction"]` -- `campaign()` writes `asdict(reduction)`, which
        carries all three, straight back at `Reduction(**summary["reduction"])`.

        Dropping them silently would be worse than the crash: a rebuilt
        `Reduction` would report the CURRENT config's stamp regardless of what
        the record actually carries, exactly the confusion `latent.load()` and
        `ceiling_record.stamp_drift` both refuse elsewhere -- a report drawn
        under today's `KERNEL_SIGMA` while describing a run made under a
        different one, with nothing in the object itself to tell the two apart.

        So this refuses on the same disagreement `latent.hyperparameter_drift`
        already checks a checkpoint's manifest against, applied here to a
        record's own `"reduction"` field: a field missing entirely, or present
        and different from what `config` currently carries, is drift, and drift
        refuses. Only once the drift check passes are the three dropped, along
        with any other non-init field a future `Reduction` might add, and the
        remaining keys become the constructor call.
        """
        from MIL_CREDA_Benchmark import latent as _latent

        drift = _latent.hyperparameter_drift({"reduction": d})
        if drift:
            raise SystemExit(
                "refusing to rebuild a Reduction from a record stamped under "
                f"hyperparameters the current config no longer carries: {drift}.\n"
                "  This record was produced under a different `KERNEL_SIGMA`, "
                "`ATTENTION_GAMMA` or `ATTENTION_TEMPERATURE` than the one "
                "`config` declares now -- rebuilding a `Reduction` from it would "
                "silently stamp today's values over a record that measured "
                "something else."
            )
        init_names = {f.name for f in fields(Reduction) if f.init}
        return Reduction(**{k: v for k, v in d.items() if k in init_names})


# --------------------------------------------------------------------- one run

@torch.no_grad()
def accuracy(model: nn.Module, dataset, device: torch.device) -> float:
    """Bag accuracy: the unit both families decide in."""
    model.eval()
    correct = total = 0
    for start in range(0, len(dataset), config.BAGS_PER_STEP):
        items = [dataset[i] for i in range(start, min(start + config.BAGS_PER_STEP,
                                                     len(dataset)))]
        x = torch.stack([item[0] for item in items]).to(device)
        y = torch.tensor([item[1] for item in items], device=device)
        correct += int((model(x).argmax(dim=1) == y).sum())
        total += int(y.numel())
    model.train()
    return correct / total if total else float("nan")


def pool_of(bagset: bags.BagSet, positions: torch.Tensor, device: torch.device) -> wiring.Pool:
    """A role of a domain, moved to the device once rather than batch by batch."""
    return wiring.Pool(
        images=bagset.images.to(device),
        members=bagset.members[positions].to(device),
        labels=bagset.labels[positions].to(device),
    )


def transfer_label(transfer: tuple[str, str]) -> str:
    """The one spelling of a transfer used as a key anywhere.

    Written once because the record, the search's progress lines and the ceiling
    lookup all have to agree on it. Two of them agreeing and the third not would
    make a per-transfer ceiling silently fall back to the pooled one, which is
    the failure that reads as success: the run proceeds and reports a number.
    """
    return f"{transfer[0]}->{transfer[1]}"


def ceiling_for(reduction: Reduction, family: str | None,
                transfer: tuple[str, str]) -> float:
    """The ceiling in force for one family on one transfer.

    Two readings, and which one applies is the rule the report states. On a
    transfer the search measured, the winner of that transfer. On one it never
    saw, the winner pooled over the searched transfers — an out-of-sample
    application, declared as such, because the scalar was not chosen by looking
    at that transfer.

    A family with no adaptation term has no ceiling and gets the neutral; the
    coefficient it multiplies is not in its objective at all.

    **Refuses when `reduction.ceilingSearch` names this family under a stamp
    the CURRENT config disagrees with.** `campaign()` attaches
    `search_record(pilot=reduction.pilot)` there before any run starts — that
    record, and not necessarily the one `reduction.ceilings`/
    `reduction.ceilingsByTransfer` actually came from. The two agree whenever a
    caller built the `Reduction` through `with_ceilings_in_force()`, which reads
    both halves from that same record. They do not have to agree in general: a
    `Reduction` assembled by hand from `config.ceilings_on_record(pilot=...)` at
    one `pilot` value, then run under a different `reduction.pilot` — the
    remote-rehearsal shape, a pilot reduction whose numbers came from the full
    record — carries ceilings this check never looks at, because it only ever
    consults the file matching `reduction.pilot`. This is a backstop for that
    narrower case, not the same check `ceilings_in_force()`/
    `config.ceilings_on_record()` make reached a second way: those refuse on
    the record they actually read from, at the `pilot` they were actually asked
    for; this refuses on the record `campaign()` happened to attach, which is
    not always that one. A bare `Reduction()` carries `ceilingSearch == {}` and
    is unaffected: nothing to check against means nothing refuses, the same as
    `run_smoke()`'s declared neutral, which was never searched at all.
    """
    if family is None:
        return config.RAMP_CEILING
    entry = reduction.ceilingSearch.get(family)
    if entry is not None and ceiling_record.stamp_drift(entry):
        raise SystemExit(
            f"refusing to use {family}'s ceiling: the attached record stamped "
            "it under a revision or hyperparameters the current config no "
            "longer carries.\n"
            "  Re-run `harness.search_ceilings(...)` under today's config, or "
            "delete the stale record to search again."
        )
    pooled = reduction.ceilings.get(family, config.RAMP_CEILING)
    return (reduction.ceilingsByTransfer.get(family, {})
            .get(transfer_label(transfer), pooled))


def hyper_for(reduction: Reduction, family: str | None,
             transfer: tuple[str, str]) -> dict:
    """The other five searched dimensions in force for one family on one
    transfer: `rampDelta`, `kernelSigma`, `attentionGamma`,
    `attentionTemperature`, `tauLocal`.

    The identical two-reading rule `ceiling_for` already applies to the
    coefficient, carried to the five dimensions beside it: on a transfer the
    search measured, that transfer's own winner (`reduction.hyperByTransfer`);
    on one it never saw, or with nothing searched at all, `reduction`'s own
    scalar fields -- which are `config`'s declared constants unless a caller
    overrode them, exactly what every arm trained at before this search
    existed.

    **Read for every arm, `family=None` included.** `ceiling_for` returns the
    neutral for a family with no adaptation term, because the coefficient it
    would multiply is not in that arm's objective at all -- but Eq. (15)/(16)'s
    attention and Decision 1's bandwidth shape EVERY arm's pooling, adaptation
    or not, so the floor has to train under the same transfer's winner an
    adapted arm does. Reading `family`'s own entry first and falling back to
    whichever family `reduction.hyperByTransfer` actually names is what makes
    that work today, with `config.SEARCH_ARMS` naming exactly one family
    (`milcreda`): the floor has no family of its own to look up, and the
    single searched family's winners are what it reads instead.

    This function never refuses on a stale stamp the way `ceiling_for` does:
    `reduction.hyperByTransfer` is populated by the identical call
    (`with_ceilings_in_force`) that populates `ceilingsByTransfer`, at the
    identical `pilot`, so a drifted record is refused there, before either
    dict is ever attached to a `Reduction`.
    """
    pooled = {"rampDelta": reduction.rampDelta, "kernelSigma": reduction.kernelSigma,
              "attentionGamma": reduction.attentionGamma,
              "attentionTemperature": reduction.attentionTemperature,
              "tauLocal": reduction.tauLocal}
    lookup = family if family in reduction.hyperByTransfer else next(
        iter(reduction.hyperByTransfer), None)
    per_transfer = reduction.hyperByTransfer.get(lookup) if lookup else None
    winner = (per_transfer or {}).get(transfer_label(transfer))
    return {**pooled, **(winner or {})}


def run_one(arm_id: str, transfer: tuple[str, str], seed: int,
            reduction: Reduction, device: torch.device,
            material: dict, ceiling: float | None = None,
            role: str = "eval", hyper: dict | None = None) -> dict:
    """One arm, one transfer, one repetition, end to end.

    `ceiling` overrides the family's for this run. The search passes it to walk
    the grid; the campaign passes each family's found value, so every arm derived
    from a family inherits the one that family searched.

    `hyper` overrides Decision 1's bandwidth and Eq. (15)/(16)/(28)'s three
    hyperparameters (`kernelSigma`, `attentionGamma`, `attentionTemperature`,
    `tauLocal`) and the ramp's own growth rate (`rampDelta`) for this run --
    `search_ceilings_trials` is the one caller that passes it explicitly,
    walking its own six-dimensional space, one trial at a time.

    **Omitted, and this is the wiring that changed**: this call resolves
    `hyper_for(reduction, family, transfer)` itself -- the identical
    two-reading rule `ceiling` already gets from `ceiling_for` when a caller
    omits IT, carried to the other five dimensions. A campaign never passes
    `hyper` explicitly and never needed to: what the search found already
    reaches every arm of the transfer it was measured on, through
    `reduction.hyperByTransfer` (populated by `with_ceilings_in_force`, the
    identical call that populates `ceilingsByTransfer`). A `reduction` with
    nothing searched (`hyperByTransfer` empty, the bare `Reduction()` shape)
    resolves to exactly its own scalar fields -- `config`'s declared constants
    unless a caller overrode them -- so a caller with no search record trains
    exactly as every caller did before this override existed.

    `role` is which material the run is judged on. The search reads `valid` and
    the campaign reads `eval`, and they are disjoint by construction — a
    coefficient chosen on the material the verdict rests on would make the
    verdict report a decision it already made, by an amount nobody can subtract
    afterwards.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 9973)

    family = config.ARMS_BY_ID[arm_id]["adaptation"]
    if hyper is None:
        hyper = hyper_for(reduction, family, transfer)

    source, target = material["source"], material["target"]
    source_train, source_valid, source_eval = bags.roles(source)
    target_train, target_valid, target_eval = bags.roles(target)

    # Which role this run is judged on, and it is one or the other rather than
    # both. Measuring both would cost two extra passes per epoch inside the timed
    # region of every run of a campaign, and — the reason that matters more — it
    # would have the search read the evaluation role and then discard it. A role
    # the search cannot see is a stronger guarantee than one it agrees not to use.
    if role == "valid":
        judged_source, judged_target = source_valid, target_valid
    elif role == "eval":
        judged_source, judged_target = source_eval, target_eval
    else:
        raise ValueError(f"unknown role {role!r}; the roles are 'valid' and 'eval'")

    model = wiring.build(
        arm_id, config.CLASSES,
        pool_of(source, source.train_idx, device),
        pool_of(target, target.train_idx, device),
        hyper=hyper,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    steps = -(-config.TRAIN_BAGS // config.BAGS_PER_STEP)
    # `hyper` is never `None` past the resolution above -- `hyper_for` always
    # returns all five keys, `config`'s own defaults where nothing searched --
    # so this reads the resolved value directly rather than defending against
    # an absence that cannot happen here any more.
    ramp_delta = hyper.get("rampDelta", config.RAMP_DELTA)

    curve: list[dict] = []
    epochs_record: list[dict] = []

    model.train()
    for epoch in range(reduction.epochs):
        # The family's ceiling on this transfer, or the one this call was
        # handed. Never a global: each family keeps what it searched, its
        # derivations inherit it, and a transfer the search measured keeps its
        # own pick rather than the pooled one.
        top = (ceiling_for(reduction, family, transfer)
               if ceiling is None else ceiling)
        coefficient = ramp(epoch, reduction.epochs, family, ceiling=top,
                          delta=ramp_delta)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(epoch, reduction.epochs)

        for batch in balanced_batches(source_train.targets, steps, generator):
            items = [source_train[i] for i in batch]
            x = torch.stack([item[0] for item in items]).to(device)
            y = torch.tensor([item[1] for item in items], device=device)
            optimizer.zero_grad()
            step = model.training_step(x, y, coefficient, generator)
            step["loss"].backward()
            optimizer.step()
            curve.append({"epoch": epoch, "ramp": coefficient,
                          "supervised": step["supervised"],
                          "adaptation": step["adaptation"],
                          "contribution": step["contribution"]})

        epochs_record.append({
            "epoch": epoch,
            "sourceAccuracy": accuracy(model, judged_source, device),
            "targetAccuracy": accuracy(model, judged_target, device),
        })

    contributions = [abs(point["contribution"]) for point in curve]
    # The supervised magnitude has to leave this function or it is gone: the curve
    # is discarded at the end of the run and no checkpoint can recover it. Without
    # it `contribution` is a bare number, and "the term commanded nothing" and "the
    # term was scaled to nothing" read identically. Eq. (21) is divided by B_src
    # precisely so the three terms of Eq. (39) can be read against each other, so
    # the ratio is the quantity that normalization exists to make meaningful.
    supervised = [abs(point["supervised"]) for point in curve]
    mean_supervised = sum(supervised) / len(supervised) if supervised else 0.0
    mean_contribution = sum(contributions) / len(contributions) if contributions else 0.0
    # The last epoch's evaluation is the final one; measuring it again would cost
    # two more passes over both evaluation sets in every one of the runs.
    final = epochs_record[-1]
    return {
        "arm": arm_id,
        "transfer": f"{transfer[0]}->{transfer[1]}",
        "seed": seed,
        # Which machine produced this run, on the run and not only on the
        # campaign. A shard is a remote session, and a session that times out and
        # resumes can land on different hardware inside one shard — a stamp held
        # once per file cannot express that, and it is exactly what distributing
        # produces. The full stamps live once beside the runs; this is the handle.
        "env": environment_key(reduction.environment),
        "targetAccuracy": final["targetAccuracy"],
        "sourceAccuracy": final["sourceAccuracy"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "contribution": mean_contribution,
        "supervised": mean_supervised,
        # An arm with no adaptation term reports zero rather than a ratio, because
        # a floor has no share to command and a nan would propagate into the table.
        "adaptationShare": (
            mean_contribution / (mean_supervised + mean_contribution)
            if (mean_supervised + mean_contribution) > 0 else 0.0
        ),
        "curve": curve,
        "epochs": epochs_record,
        "state": model.state_dict() if arm_id in config.CHECKPOINTS else None,
    }


# ----------------------------------------------------- attention mechanisms
#
# Section 4's comparison: which attention mechanism, on the full arm (`G`)
# and nothing else -- never a declared arm's own training path. Mirrors
# `run_one` deliberately close (same roles, same ramp, same optimizer) so a
# difference between two mechanisms is a difference of pooling, exactly the
# property `wiring.MechanismArm`'s own docstring states; it is not `run_one`
# itself because a `mechanism` is not an `arm_id` -- `wiring.build_mechanism`
# takes the former, `config.ARMS_BY_ID` has no entry for it, and threading a
# mechanism string through every branch `run_one` takes on a real arm id
# would read as one function serving two different questions.

#: Every declared arm sharing `G`'s family -- what the search record's
#: winners (`ceiling_for`/`hyper_for`) resolve against, and the ramp's own
#: floor: the comparison trains the COMPLETE method's family, never a
#: declared arm id, so this is the one constant the five mechanisms share
#: rather than a per-mechanism lookup.
MECHANISM_FAMILY = config.ARMS_BY_ID["G"]["adaptation"]


def run_mechanism(mechanism: str, transfer: tuple[str, str], seed: int,
                  reduction: Reduction, device: torch.device,
                  material: dict, role: str = "eval") -> dict:
    """One mechanism, one transfer, one repetition, end to end -- the
    comparison's own `run_one`.

    Trains `G`'s full method (`wiring.build_mechanism`, always) with
    Eq. (15)/(16) replaced by `mechanism`; the ceiling and the other five
    searched dimensions still resolve through `ceiling_for`/`hyper_for`
    against `MECHANISM_FAMILY`, the identical two-reading rule every declared
    arm of that family already gets, so the comparison trains under the same
    searched bounds a real campaign would.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 9973)

    hyper = hyper_for(reduction, MECHANISM_FAMILY, transfer)

    source, target = material["source"], material["target"]
    source_train, source_valid, source_eval = bags.roles(source)
    target_train, target_valid, target_eval = bags.roles(target)

    if role == "valid":
        judged_source, judged_target = source_valid, target_valid
    elif role == "eval":
        judged_source, judged_target = source_eval, target_eval
    else:
        raise ValueError(f"unknown role {role!r}; the roles are 'valid' and 'eval'")

    model = wiring.build_mechanism(
        mechanism, config.CLASSES,
        pool_of(source, source.train_idx, device),
        pool_of(target, target.train_idx, device),
        hyper=hyper,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    steps = -(-config.TRAIN_BAGS // config.BAGS_PER_STEP)
    ramp_delta = hyper.get("rampDelta", config.RAMP_DELTA)

    model.train()
    for epoch in range(reduction.epochs):
        top = ceiling_for(reduction, MECHANISM_FAMILY, transfer)
        coefficient = ramp(epoch, reduction.epochs, MECHANISM_FAMILY,
                          ceiling=top, delta=ramp_delta)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(epoch, reduction.epochs)

        for batch in balanced_batches(source_train.targets, steps, generator):
            items = [source_train[i] for i in batch]
            x = torch.stack([item[0] for item in items]).to(device)
            y = torch.tensor([item[1] for item in items], device=device)
            optimizer.zero_grad()
            step = model.training_step(x, y, coefficient, generator)
            step["loss"].backward()
            optimizer.step()

    return {
        "mechanism": mechanism,
        "transfer": transfer_label(transfer),
        "seed": seed,
        "sourceAccuracy": accuracy(model, judged_source, device),
        "targetAccuracy": accuracy(model, judged_target, device),
    }


def run_mechanism_sweep(reduction: Reduction, device: torch.device,
                        transfers: list | None = None, noise: float = 0.0,
                        progress=print) -> dict:
    """Every mechanism, over `config.VERDICT_TRANSFERS` (or `transfers`) and
    `reduction.seeds` -- the record `tables.MECHANISM_RECORD` names,
    `{"mechanisms": [...], "clean": [...], "noisy": [...]}`.

    Writes to `noisy` when `noise` is non-zero, `clean` otherwise -- called
    twice (once per condition) rather than folding both into one call the
    way `campaign` folds transfers, because the two conditions are two
    separate questions section 4 asks and a partial sweep (clean finished,
    noisy not yet) has to be representable on disk.

    `noise` overwrites `reduction.labelNoise` rather than living beside it,
    the same reconciliation `search_ceilings_trials` already makes for the
    identical reason: two live coordinates for one destination is how a
    write ends up in the wrong tree with nothing raising.
    """
    reduction = replace(reduction, labelNoise=noise)
    drawn = {code: bags.build(code, config.DATA_CACHE, seed, noise)
             for seed in reduction.seeds for code in config.DOMAINS}
    runs: list[dict] = []
    for transfer in (transfers or config.VERDICT_TRANSFERS):
        for seed in reduction.seeds:
            material = {"source": drawn[transfer[0]], "target": drawn[transfer[1]]}
            for mechanism in wiring.MECHANISMS:
                run = run_mechanism(mechanism, transfer, seed, reduction, device,
                                    material, role="eval")
                runs.append(run)
                progress(f"  mechanisms {transfer_label(transfer)} seed {seed} "
                         f"{mechanism}: target={run['targetAccuracy']:.3f}")

    # The destination, through its gate and with every coordinate written. It
    # was `config.PRODUCT / tables.MECHANISM_RECORD`, a fixed path no scale
    # reached, excused in `config.DESTINOS_SIN_COORDENADA` on the ground that
    # section 4 "declares no pilot `Reduction` of its own". That described the
    # absence of a dial, never a property of the record: the moment
    # `run_mechanism_sweep_shard` takes `pilot`, a fixed path writes
    # three-epoch numbers over the full record section 4 is read from —
    # silently, and in exactly the right shape.
    #
    # `rate=0.0` and not `reduction.labelNoise`: the two conditions are
    # multiplexed INSIDE the json (`clean`/`noisy`), which is what the old
    # excuse had right, so the rate is not a coordinate of this destination
    # and the one that is — the scale — now travels. The file NAME comes from
    # the constant its readers already name rather than from a fresh literal:
    # at full scale this is byte for byte the path it always was, and there
    # are no two spellings that can drift apart.
    record_path = (config.results_for(rate=0.0, kind="campaign",
                                      pilot=reduction.pilot)
                   / Path(tables.MECHANISM_RECORD).name)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {"mechanisms": list(wiring.MECHANISMS), "clean": [], "noisy": []}
    if record_path.exists():
        existing = json.loads(record_path.read_text(encoding="utf-8"))
    key = "noisy" if noise else "clean"
    existing["mechanisms"] = list(wiring.MECHANISMS)
    existing[key] = runs
    record_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return existing


# ------------------------------------------------------------------ aggregation

def spread(values: list[float]) -> dict:
    """Mean and dispersion. A bare mean over several seeds hides the only thing
    several seeds were run to reveal."""
    n = len(values)
    mean = sum(values) / n if n else float("nan")
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        deviation = math.sqrt(variance)
    else:
        deviation = 0.0
    return {"mean": mean, "stdev": deviation, "n": n}


def summarize(runs: list[dict]) -> dict:
    """Every dimension of one cell of the grid, across its repetitions."""
    return {dimension: spread([float(r[dimension]) for r in runs])
            for dimension in config.DIMENSIONS}


def ladder_rows(cell: dict, transfer: str, dimensions: dict | None = None) -> list[dict]:
    """One row per rung and dimension, with the arm on the right as `new`.

    `dimensions` defaults to `config.DIMENSIONS` (every declared dimension) --
    `campaign()`'s own call below relies on that default, and so does this
    function's own strictness: a `cell` missing a dimension it was told to
    read still raises `KeyError`, unweakened. A caller reading a POOLED
    grid (perRun dimensions like `seconds`/`peakMiB` were never averaged
    across machines and do not exist there) passes the narrower set it
    actually has -- see `tools/bridge.py::build_summary`.
    """
    dimensions = config.DIMENSIONS if dimensions is None else dimensions
    rows = []
    for left, right, reading in config.LADDER:
        if left not in cell or right not in cell:
            continue
        for dimension, better in dimensions.items():
            rows.append({
                "dimension": f"{left}->{right} {dimension}",
                "rung": f"{left}->{right}",
                "reading": reading,
                "transfer": transfer,
                "metric": dimension,
                "better": better,
                "baseline": cell[left][dimension],
                "new": cell[right][dimension],
            })
    return rows


def paired_across_transfers(grid: dict) -> list[dict]:
    """For each rung, the difference transfer by transfer, then its own spread.

    Comparing raw accuracies pooled over transfers would fold the difficulty of
    each transfer into the dispersion and drown everything. The difference of two
    arms measured on the same transfer, with the same split and the same seeds,
    cancels that difficulty, which is what makes the panorama carry weight even
    when no single transfer resolves anything.
    """
    readings = []
    for left, right, reading in config.LADDER:
        for metric in ("targetAccuracy", "sourceAccuracy"):
            differences = []
            for transfer, cell in grid.items():
                if left in cell and right in cell:
                    # Left minus right, the order the rung is named and read. The
                    # panorama outlives its table in the record, so it carries the
                    # same convention the rung table prints — one artifact with two
                    # opposite signs for the same subtraction is a record that
                    # cannot be read without knowing which function wrote it.
                    differences.append(cell[left][metric]["mean"] - cell[right][metric]["mean"])
            if not differences:
                continue
            statistics = spread(differences)
            # The field still counts transfers where the RIGHT arm came out above;
            # with the subtraction flipped that is now the negative side.
            favouring = sum(1 for d in differences if d < 0)
            readings.append({
                "rung": f"{left}->{right}",
                "reading": reading,
                "metric": metric,
                "meanDifference": statistics["mean"],
                "stdev": statistics["stdev"],
                "transfers": statistics["n"],
                "favouringRight": favouring,
                "favouringLeft": statistics["n"] - favouring,
            })
    return readings


def median_seeds(cell_runs: list[dict], arm_id: str) -> set:
    """Which repetitions of one cell sit closest to its median.

    The selection rule on its own, so a shard and the centre can share it rather
    than each carrying a copy that drifts. Lifted verbatim out of `keep_median`,
    which now calls it — same ordering, same span, same edge behaviour when a cell
    has fewer runs than `CHECKPOINTS` asks for.

    Never the best. The best of thirty is an extreme of thirty draws, and the
    latent space of the luckiest run describes that run rather than the method.
    """
    ordered = sorted(cell_runs, key=lambda r: r["targetAccuracy"])
    middle = len(ordered) // 2
    span = min(config.CHECKPOINTS[arm_id], len(ordered))
    start = max(0, min(middle - span // 2, len(ordered) - span))
    return {run["seed"] for run in ordered[start:start + span]}


def keep_median(cell_runs: list[dict], arm_id: str, transfer: str,
                manifests: dict, reduction: Reduction) -> list[str]:
    """Persist the repetitions closest to the median, and drop the rest.

    Never the best. The best of thirty is an extreme of thirty draws, and the
    latent space of the luckiest run describes that run rather than the method.

    Every checkpoint is written as it is produced and pruned afterwards, because
    which repetition sits at the median is only known once the cell is finished,
    and re-running the chosen ones later would not reproduce them bit for bit on
    a device that does not promise determinism.
    """
    keep = median_seeds(cell_runs, arm_id)

    # El directorio de esta tasa y esta forma, sacado de las mismas cotas bajo
    # las que corrió. Leer `config.MODELS` acá mientras la campaña escribió en
    # otro lado no borraría nada y no promovería nada, en silencio.
    pesos = config.models_for(reduction.labelNoise, reduction.kind,
                             reduction.pilot)

    # The manifest stamps what this CELL's own run actually trained under, not
    # the campaign's one shared, pooled `reduction`. `hyper_for` resolves the
    # five dimensions `run_one` itself resolved for this exact
    # (family, transfer) pair -- the same call, the same record -- so a
    # checkpoint's manifest and the run that produced it can never disagree.
    # `family=None` for a floor still reads the searched family's own winners
    # (`hyper_for`'s own docstring: Eq. (15)/(16)'s attention shapes every
    # arm's pooling, adaptation or not), so this is correct for every declared
    # arm and not only the adapted ones.
    family = config.ARMS_BY_ID.get(arm_id, {}).get("adaptation")
    stamped = replace(reduction, **hyper_for(reduction, family,
                                             tuple(transfer.split("->"))))

    kept: list[str] = []
    for run in cell_runs:
        stem = f"{arm_id}_{transfer.replace('->', '-')}_seed{run['seed']}"
        weights = pesos / f"{stem}.pt"
        if run["seed"] not in keep:
            weights.unlink(missing_ok=True)
            continue
        bags.write_manifest(
            pesos / f"{stem}.manifest.json",
            arm=arm_id, transfer=transfer, seed=run["seed"],
            targetAccuracy=run["targetAccuracy"],
            sourceAccuracy=run["sourceAccuracy"],
            reduction=asdict(stamped),
            **manifests[(transfer, run["seed"])],
        )
        kept.append(str(weights.relative_to(config.REPOSITORY)))
    return kept


#: Where a search in flight keeps what it has measured. Separate from the finished
#: record on purpose: `ceilings.json` existing means the search answered, and a
#: half-filled file under that name would be read as an answer by everything that
#: consumes it, including the campaign's own refusal.
PARTIAL_SUFFIX = ".partial.json"


#: Where a shard's own files live, under the results the campaign already owns.
SHARDS_DIR = "shards"


def shard_paths(shard: str | None, pilot: bool = False,
                noise: float = 0.0, kind: str = "campaign") -> dict:
    """Where one shard writes, so no two shards write to the same place.

    Without this there is one `runs.jsonl`, opened `"w"` and truncated on every
    campaign, and one `ceilings.partial.json` with no locking. Two shards running
    at once would clobber each other, and the loser would be a silent partial
    file rather than an error — the failure mode that reads as a finished run.

    `None` is the single-machine case and keeps every path exactly where it has
    always been. Sharding is an addition, not a relocation: the notebooks, the
    records already written and the `records` declaration all name these paths,
    and moving them would break a working repository to serve one that does not
    exist yet.
    """
    root = config.results_for(noise, kind, pilot)
    if shard is None:
        record = config.ceilings_record_for(pilot)
        # El registro sigue a la raíz cuando la tasa o la forma la movieron, y
        # NO cuando la movió el piloto: la búsqueda de ensayo ya se distingue
        # por su nombre de archivo, y moverla además de árbol la separaría de su
        # parcial, que es el escritor de la búsqueda a medias.
        if root != config.results_for(0.0, "campaign", pilot):
            record = root / record.name
        return {"runs": root / "runs.jsonl",
                "partial": record.with_name(record.stem + PARTIAL_SUFFIX),
                "stamp": root / "shard.json"}
    home = root / SHARDS_DIR / shard
    return {"runs": home / "runs.jsonl",
            "partial": home / f"ceilings{PARTIAL_SUFFIX}",
            "stamp": home / "shard.json"}


def _git_commit(repository: Path) -> str | None:
    """`HEAD`'s commit in the checkout that is stamping this run, or `None`.

    `git` genuinely unavailable — no history, no `.git`, the executable
    missing — is a fact about the checkout, not a failure to raise past. The
    caller omits the `commit` key entirely rather than write a value that
    looks like evidence and is not: that is what lets `completeness()` report
    it `missing` instead of present-and-wrong, and what makes this a stamp a
    non-git checkout can never seal.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository,
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _evidence() -> dict:
    """Everything a stamp can know before the run it describes has happened.

    `outputs` is deliberately absent: the files a shard produces cannot be
    named before the run that produces them, which is exactly why sealing
    them in is a separate, later call.
    """
    evidence: dict = {
        # The same digest a report's own provenance stamp uses over `src/` —
        # reused, not recomputed, so the two halves can never drift apart.
        "codeDigest": report_digest.stamp().split(" ", 1)[1],
        "importsFrom": str(Path(__file__).resolve().parent),
    }
    commit = _git_commit(config.REPOSITORY)
    if commit is not None:
        evidence["commit"] = commit
    return evidence


def write_shard_stamp(shard: str | None, reduction: Reduction) -> Path:
    """The full environment, once per shard, beside the runs that reference it.

    Runs carry a twelve-character handle rather than the whole stamp — repeating
    it on every record would be the same fact written a thousand times, and a
    fact written a thousand times is one that can disagree with itself. This is
    where the handle resolves.
    """
    path = shard_paths(shard, noise=reduction.labelNoise,
                       kind=reduction.kind, pilot=reduction.pilot)["stamp"]
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = reduction.environment or environment()
    path.write_text(json.dumps({
        "shard": shard,
        "env": environment_key(stamp),
        "environment": stamp,
        "seeds": list(reduction.seeds),
        "epochs": reduction.epochs,
        # Planos y de primer nivel porque es la única forma que
        # `disagreements()` puede comparar --- lee `stamp.get(field)` y nunca una
        # ruta con puntos --- y dos shards contaminados a tasas distintas son un
        # experimento distinto, no hardware distinto.
        "labelNoise": reduction.labelNoise,
        "pilot": reduction.pilot,
        "kind": reduction.kind,
        "revision": reduction.revision,
        "ceilings": dict(reduction.ceilings),
        "ceilingsByTransfer": {family: dict(picks) for family, picks
                               in reduction.ceilingsByTransfer.items()},
        # The other five searched dimensions, on the same footing and for the
        # same reason: the search moves six parameters, so a shard straddling a
        # search can disagree on any of them, not only on the ceiling. Empty
        # here is itself the reading that matters -- a record carrying no
        # per-transfer hyperparameters is one `hyper_for` answered from
        # `config`'s own constants, and this is where that shows rather than
        # being a silence two shards could share without meaning the same thing.
        "hyperByTransfer": {family: {label: dict(dims) for label, dims
                                     in picks.items()}
                            for family, picks in reduction.hyperByTransfer.items()},
        "evidence": _evidence(),
    }, indent=2), encoding="utf-8")
    return path


def seal_shard_stamp(shard: str | None, noise: float = 0.0,
                     kind: str = "campaign",
                     pilot: bool = False) -> Path:
    """Add `outputs` to an already-written stamp, atomically, once the run ends.

    Two-phase because `outputs` cannot be known when `write_shard_stamp` runs:
    the files a shard writes are exactly the files this call closes over, once
    they exist. Atomic the same way `jobfolder.py`'s own generation is — a
    scratch copy plus `os.replace` — so a reader hitting the rewrite window
    sees either the whole pre-seal stamp or the whole sealed one, never a
    half-written file.

    A run that dies before this call leaves its stamp unsealed forever: no
    `outputs` key, so `completeness()` reports it missing, so `merge()`
    refuses it. That refusal is the entire enforcement mechanism — there is no
    separate "did it finish" flag a caller could forget to check.
    """
    path = shard_paths(shard, noise=noise, kind=kind, pilot=pilot)["stamp"]
    stamp = json.loads(path.read_text(encoding="utf-8"))
    outputs = sorted(p.name for p in path.parent.iterdir() if p.is_file())
    stamp.setdefault("evidence", {})["outputs"] = outputs
    partial = path.with_name(path.stem + PARTIAL_SUFFIX)
    partial.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    os.replace(partial, path)
    return path


def _partial_path(noise: float = 0.0, kind: str = "campaign",
                  pilot: bool = False) -> Path:
    """Las tres coordenadas y no ninguna. Tomaba cero y devolvía el parcial de la
    campaña limpia para cualquier búsqueda: dos búsquedas a tasas distintas se
    pisaban el archivo a medias, y la que perdía se leía como una que resumía."""
    return shard_paths(None, noise=noise, kind=kind, pilot=pilot)["partial"]


def _read_partial(path: Path | None = None) -> dict:
    """Cells already measured, keyed by family, so a relaunch skips them.

    Keyed by `(seed, transfer)` rather than by position: a relaunch that resumed
    by counting would silently shift if the seed list or the transfer list moved,
    and would then attribute one cell's measurements to another.
    """
    path = path or _partial_path()
    if not path.exists():
        return {}
    stored = json.loads(path.read_text(encoding="utf-8"))
    return {
        family: {(int(seed), label): {float(c): v for c, v in scores.items()}
                 for key, scores in cells.items()
                 for seed, label in [key.split("|", 1)]}
        for family, cells in stored.get("cells", {}).items()
    }


def _partial_stamp_drift(path: Path | None = None) -> dict:
    """`ceiling_record.stamp_drift`, applied to a grid-search partial's own
    top-level stamp rather than to a ceiling record entry.

    Empty when the partial does not exist -- nothing to resume, nothing to
    refuse. `stamp_drift` already treats an absent field as drift, so a
    partial written before `_write_partial` stamped every write (below) comes
    back as drift on all four fields, the same as an old ceiling record does.
    """
    path = path or _partial_path()
    if not path.exists():
        return {}
    stored = json.loads(path.read_text(encoding="utf-8"))
    return ceiling_record.stamp_drift(stored)


def _write_partial(family: str, arm_id: str, cells: dict, minutes: float,
                   progress, path: Path | None = None) -> None:
    """Persist what is measured so far, after every cell."""
    path = path or _partial_path()
    stored = (json.loads(path.read_text(encoding="utf-8"))
              if path.exists() else {"cells": {}, "minutesPerCell": {}})
    stored["cells"].setdefault(family, {}).update({
        f"{seed}|{label}": {str(c): v for c, v in scores.items()}
        for (seed, label), scores in cells.items()})
    stored["minutesPerCell"].setdefault(family, []).append(round(minutes, 2))
    stored["arms"] = {**stored.get("arms", {}), family: arm_id}
    stored["environment"] = environment()
    # Stamped every write, like every other record this repository's search
    # produces: a partial is a ceiling record mid-measurement, bound to the
    # same revision and the same three hyperparameters as the entry it will
    # eventually become, and a relaunch that resumed it under a moved
    # `KERNEL_SIGMA` would silently splice cells measured under two objectives
    # into one entry.
    ceiling_record.stamp(stored)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored, indent=2), encoding="utf-8")


def search_record(pilot: bool | None = None) -> dict | None:
    """The ceiling search's own record, whole, or None when it has not run.

    Read from disk and not carried in memory: `config.CEILINGS` keeps only the
    winners, and the winner alone cannot say whether it was searched at scale or
    whether the seeds agreed on it.

    `pilot=None` significa *el que rige* --- la corrida completa si existe, el
    ensayo si no --- y es el valor por omisión, la misma forma que
    `contamination.level_dir` ya tenía para el registro de la campaña y por el
    mismo motivo: leer el árbol completo a secas es lo que hacía que un ensayo
    entero apareciera como «no hay búsqueda», y «no corrió» y «corrió y no lo
    encontré» son la misma tabla vacía para quien la lee. Un informe de ensayo
    llegaba así a 23 renglones de «no hay nada que mostrar» sobre una búsqueda
    que sí había corrido.

    Quien necesita un archivo Y NO EL OTRO lo dice: `pilot=False` es la lectura
    que GOBIERNA --- `campaign` exige `atRequiredScale` sobre el registro
    completo, y aceptar ahí el del ensayo dejaría que los techos de un ensayo
    gobernaran una campaña real, que es peor que el defecto que esto arregla.
    `pilot=True` es el ensayo y nada más. La omisión es para lo que se muestra,
    nunca para lo que decide.

    **Se le pasa la escala de la corrida, y antes no.** El argumento escrito acá
    era que son dos registros con escalas propias y que reenviarle el `ES_ENSAYO`
    del cuaderno haría que «un informe de ensayo dijera "no hay búsqueda"
    mientras su propia reducción cita techos completos». La conclusión era
    correcta y su premisa se movió: esa reducción citaba techos completos porque
    `config.ceilings_on_record()` no tenía cómo recibir una escala, así que una
    campaña de ensayo consumía la búsqueda COMPLETA. Ahora la recibe, la campaña
    de ensayo corre bajo `ceilings.pilot.json` y el informe que la dibuja
    pregunta por el mismo archivo: las dos mitades dicen lo mismo porque leen lo
    mismo, que es lo que aquel párrafo quería y no podía tener.

    La resolución vive en `config.ceilings_record_at`, una sola vez: la
    preferencia entre los dos archivos estaba escrita acá y allá, y dos
    ortografías de una regla es el defecto, no la regla.
    """
    record, _ = config.ceilings_record_at(pilot)
    if record is None:
        return None
    found = json.loads(record.read_text(encoding="utf-8"))
    # Tagged and not filtered, the same choice `latent.available()` makes for
    # `currentRevision`/`currentHyperparameters`: a record from before
    # `ceiling_record.stamp` existed, or one stamped under a `KERNEL_SIGMA`
    # or revision the current config no longer carries, still comes back --
    # dropping it here would hide from every reader (including this
    # function's own callers below) that it exists at all. `ceilings_in_force`
    # and `ceiling_for` are what actually refuse on this tag; this function
    # only reports it.
    for entry in found.values():
        if isinstance(entry, dict):
            entry["currentStamp"] = not ceiling_record.stamp_drift(entry)
    return found


def search_source_note(pilot: bool | None = None) -> str:
    """De dónde salieron los techos, en una línea, para encabezar la rejilla.

    El gemelo exacto de `contamination.source_note`, que es el mecanismo que
    este repositorio ya usa para que una tabla de ensayo no se pueda leer como
    una completa. No se inventa un segundo aviso: misma forma, mismo lugar
    ---arriba de los números, no al pie--- y la misma regla de que se escribe
    SIEMPRE y no sólo en el caso malo, porque un aviso que aparece únicamente
    cuando algo anda mal no le enseña a nadie qué es lo que vigila, y la primera
    vez que falta se lee como que no había nada que avisar.

    Los hechos salen de `config.ceilings_provenance()`, que ya los computaba
    para estampar la reducción; acá sólo se redactan.

    **Y se le pasa la escala pedida.** El rechazo de arriba ya preguntaba por
    ella y la redacción de abajo leía la vigente, así que un aviso pedido para
    el ENSAYO ---con los dos registros en disco--- pasaba el rechazo y después
    decía «búsqueda completa, 20 épocas» encima de una rejilla de tres. Las dos
    mitades de una misma respuesta contra dos archivos distintos: el aviso que
    existe para que un ensayo no se lea como una corrida completa era el que lo
    hacía.
    """
    if pilot is not None and search_record(pilot=pilot) is None:
        return ("**Sin registro de techos** para la escala pedida: no hay nada "
                "que mostrar, que no es lo mismo que una rejilla vacía.")
    procedencia = config.ceilings_provenance(pilot=pilot)
    if procedencia["source"] == "none":
        return ("**Sin búsqueda de techos.** Ni corrida completa ni ensayo: no "
                "hay nada que mostrar, que no es lo mismo que una rejilla "
                "vacía.")
    if procedencia["source"] == "pilot":
        return (f"**Estos techos son de un ENSAYO** "
                f"({procedencia['epochs']} épocas), porque no hay búsqueda "
                f"completa. El protocolo pide "
                f"{(procedencia.get('requiredScale') or {}).get('epochs')} "
                f"épocas: no se citan como resultados, ni en el informe, ni en "
                f"el resumen, ni en conversación.")
    return f"Búsqueda completa, {procedencia['epochs']} épocas."


def campaign_source_note(ensayo: bool | None) -> str:
    """De qué árbol salieron los números que un lector está por ver.

    El mismo mecanismo que `search_source_note`/`contamination.source_note`,
    aplicado a la campaña que `Benchmark_Results.ipynb` lee: `cargar_corridas()`
    prefiere la corrida completa y cae al ensayo cuando no hay ninguna, la
    misma regla que `search_source_note` aplica a la búsqueda de techos --
    y esa caída es correcta y tiene que ser visible. (El cuaderno que la
    mostraba para los techos, `Benchmark_Search_Report_v1.ipynb`, fue
    retirado; la regla no vivía en él sino en la función de al lado.)
    Escrito siempre, nunca sólo en el caso malo: un aviso que sólo aparece
    cuando algo anda mal no le enseña a nadie qué vigila.

    `ensayo=None` es "todavía no hay corrida en ningún árbol" -- el estado
    que `cargar_corridas()` reporta con `reduccion is None`, y en el que no
    hay épocas ni semillas que citar.
    """
    if ensayo is None:
        return ("**Sin corrida todavía.** Ni completa ni ensayo: no hay "
                "número que este aviso pueda fechar.")
    if ensayo:
        return ("**Estos números son de un ENSAYO**, no de la corrida "
                "completa: no hay campaña a escala completa en disco. No se "
                "citan como resultados, ni en el informe, ni en el resumen, "
                "ni en conversación.")
    return "Corrida completa."


def ceilings_in_force(reduction: Reduction, device: torch.device,
                      progress=print, shard: str | None = None,
                      pilot: bool = False) -> dict[str, float]:
    """The ceilings the campaign will run at: searched once if no record exists.

    The campaign refuses without them, and `config.CEILINGS` is filled at import
    from a file that may not exist yet — so a caller that imports the package,
    searches, and then builds a `Reduction` gets the empty mapping it started
    with and a refusal it cannot read. This is the one call that closes that gap,
    and it reads the record back from disk rather than trusting what the search
    returned, so what the campaign runs at is what the record says.

    An existing record is used as it stands and never re-searched. A record that
    exists means the search answered, and overwriting an answer because a later
    caller wanted a different one is exactly the silent refunding the campaign's
    refusal exists to prevent. Under-scale is not fixed here either: `campaign`
    reads `atRequiredScale` itself and says which record to delete.

    **Devuelve el registro de ESTA escala y no el vigente.** `pilot` decidía a
    qué archivo buscar y escribir, y después la lectura salía por
    `config.ceilings_on_record()` sin coordenada: con los dos registros en
    disco, una corrida de ensayo buscaba en `ceilings.pilot.json`, lo escribía,
    y se llevaba los techos de `ceilings.json` --- otro experimento, a veinte
    épocas --- mientras su propio registro quedaba ahí sin que nada lo leyera.
    Las dos mitades de esta función contra dos archivos distintos, y la que
    manda era la que no llevaba la coordenada.

    **Refuses on a record whose stamp does not match the CURRENT config.**
    `search_record()` tags every family entry with `currentStamp`; a record
    written before `ceiling_record.stamp` existed, or one searched under a
    `KERNEL_SIGMA`/`ATTENTION_GAMMA`/`ATTENTION_TEMPERATURE`/`REVISION` the
    config no longer carries, governs a different objective exactly as a
    checkpoint stamped that way would -- and `latent.load()` refuses on that
    same disagreement rather than analysing it silently. `ceilings.pilot.json`
    on disk today predates this stamp entirely and refuses here by that same
    rule: it is not deleted or rewritten by this change, it is refused, which
    is correct.
    """
    reduction = replace(reduction, pilot=pilot)
    if search_record(pilot=pilot) is None:
        progress("no ceiling record: searching, once, before anything is compared")
        search_ceilings(reduction, device, progress=progress, shard=shard,
                        pilot=pilot)
    record = search_record(pilot=pilot) or {}
    drifted = sorted(family for family, entry in record.items()
                     if isinstance(entry, dict) and not entry.get("currentStamp", False))
    if drifted:
        raise SystemExit(
            "refusing to use ceilings stamped under a different revision or "
            f"hyperparameters: {', '.join(drifted)}.\n"
            "  The record on disk was searched under a `KERNEL_SIGMA`, "
            "`ATTENTION_GAMMA`, `ATTENTION_TEMPERATURE` or `REVISION` the "
            "current config no longer carries -- or predates this stamp "
            "entirely.\n"
            "  Re-run `harness.search_ceilings(...)` under today's config, or "
            "delete the stale record to search again."
        )
    return config.ceilings_on_record(pilot=pilot)


def with_ceilings_in_force(reduction: Reduction, device: torch.device,
                           progress=print, shard: str | None = None) -> Reduction:
    """`reduction`, rebuilt with both readings of the ceiling from the record.

    The one call a notebook should make. `ceilings_in_force` returns the pooled
    mapping alone, and a caller that sets only that leaves `ceilingsByTransfer`
    holding whatever `config` was imported with — empty, if the record did not
    exist yet. Every transfer would then fall back to the pooled winner, which
    is the old behaviour arriving silently: the run proceeds, the numbers look
    ordinary, and the two measured transfers quietly ran at the wrong ceiling.
    """
    # `pilot` de la reducción y no de la firma. `ceilings_in_force` hace
    # `replace(reduction, pilot=pilot)`, así que omitirlo no dejaba la escala
    # librada: la SOBRESCRIBÍA con `False`. Una reducción de ensayo entraba acá
    # y salía buscando ---y escribiendo--- en el registro de la corrida
    # completa, nueve horas y media que nadie pidió, bajo el nombre del archivo
    # que la campaña real consume. Hoy los tres sitios que llaman construyen
    # `Reduction()` sin `pilot`, así que esto no mueve ninguna corrida; lo que
    # cierra es que la próxima que sí lo lleve no se le dé vuelta en silencio.
    #
    # Y la MISMA escala en las dos mitades. El agrupado salía de
    # `ceilings_in_force(pilot=reduction.pilot)` y el pick por transferencia de
    # una llamada pelada: dos resoluciones independientes del registro para las
    # dos mitades de un mismo techo. Hoy las dos nombran `reduction.pilot`, así
    # que la reducción no puede llevar adentro un agrupado de un experimento y
    # un pick de otro --- que en el valor más sensible del cálculo no se ve
    # nunca, porque los dos números existen y son plausibles.
    pooled = ceilings_in_force(reduction, device, progress=progress, shard=shard,
                               pilot=reduction.pilot)
    return replace(reduction, ceilings=pooled,
                   ceilingsByTransfer=config.ceilings_by_transfer_on_record(
                       pilot=reduction.pilot),
                   # The other five searched dimensions' per-transfer picks,
                   # read at the SAME `pilot` and in the SAME call as the
                   # ceiling above -- the identical reason `ceilingsByTransfer`
                   # is read here rather than left at whatever `config` was
                   # imported with: two halves of one search read from two
                   # different records would mix two experiments into the
                   # values `hyper_for` resolves.
                   hyperByTransfer=config.hyper_by_transfer_on_record(
                       pilot=reduction.pilot))


def governs_the_ceilings_record(noise: float = 0.0,
                               transfers: list | None = None) -> bool:
    """Whether the search that just ran is the one whose record the campaign reads.

    The rule was already written beside one writer -- "un ensayo nunca escribe
    donde va la respuesta que la campana consume" -- and applied to the `pilot`
    axis alone. `noise` never got it, so the diagnostic's re-search at rate 0.4
    over a single transfer overwrote the clean six-transfer record and nothing
    raised. At full scale that call destroys a record costing nine and a half
    hours to reproduce.

    Derived from the arguments the writer already holds rather than from a flag
    a caller passes: a flag someone must remember is the same defect one
    indirection further away, and this module has been bitten by exactly that.

    Two ways a search fails to govern, and they are different failures. Under
    contamination the ceiling answers a question the campaign never asked. Over
    fewer transfers it answers the right question about one corner of a record
    whose every other corner it would erase.
    """
    if noise:
        return False
    chosen = [transfer_label(t) for t in (transfers or config.SEARCH_TRANSFERS)]
    return chosen == [transfer_label(t) for t in config.SEARCH_TRANSFERS]


def sellar_techos(found: dict, reduction: Reduction) -> dict:
    """Dónde se midió el techo de cada familia, escrito adentro de su entrada.

    El registro de techos era el único de este repositorio sin sello de entorno.
    Cada `shard.json` lleva intérprete, plataforma, versión de torch y
    dispositivo; los techos no llevaban ninguno de los cuatro, y con la búsqueda
    completa corriendo en un worker eso significa que el escalar que gobierna
    toda la campaña no diría en ningún lado en qué máquina se eligió. Un lector
    posterior no podría distinguir un registro buscado acá de uno buscado allá,
    que es exactamente la distinción que el sello existe para conservar en todos
    los demás registros.

    **Por familia y no una vez arriba**, por dos razones. El registro es un mapa
    plano `familia -> entrada` y todo consumidor lo recorre así --- `for familia,
    entrada in sorted(record.items())` en `tables` ---, de modo que una clave
    hermana se leería como una tercera familia y saldría impresa como una fila.
    Y además es la granularidad correcta: las búsquedas de las dos familias
    pueden repartirse entre workers distintos, y entonces cada entrada dice dónde
    se midió la suya en vez de heredar la de la otra.

    Lleva el mango corto y el sello completo, igual que las corridas: el mango
    para comparar y agrupar sin cargar el sello entero, el sello para que el
    registro viaje solo. Una corrida puede permitirse referenciar un
    `shard.json` de al lado porque viaja con él; este archivo vuelve del worker
    por su cuenta.

    **Y la misma revisión, el mismo sigma y los mismos dos hiperparámetros de
    Eq. (16) que `Reduction` ya estampa en cada checkpoint.** Un techo buscado
    bajo un `KERNEL_SIGMA` distinto gobierna una campaña bajo un objetivo
    distinto, exactamente como un checkpoint entrenado así -- y antes de esto
    el registro de techos no llevaba ninguno de los cuatro, así que ninguna
    campaña podía distinguir un techo medido bajo el sigma vigente de uno
    medido antes de que se moviera. `ceiling_record.stamp` es el mismo
    mecanismo que `latent.HYPERPARAMETER_FIELDS` usa para los checkpoints,
    aplicado acá a la entrada por familia.
    """
    stamp = reduction.environment or environment()
    handle = environment_key(stamp)
    for entry in found.values():
        entry["env"] = handle
        entry["environment"] = stamp
        ceiling_record.stamp(entry)
    return found


def search_ceilings_trials(reduction: Reduction, device: torch.device,
                           progress=print, shard: str | None = None,
                           pilot: bool = False, noise: float = 0.0,
                           transfers: list | None = None) -> dict:
    """El techo de cada familia en cada transferencia, por búsqueda con trials.

    Reemplaza a la rejilla y **no reemplaza lo que se mide**: cada trial llama al
    mismo `run_one` con el mismo rol y el mismo criterio. Lo que cambia es quién
    elige dónde mirar — cinco puntos fijos repetidos sobre tres semillas, contra
    treinta puntos que un GP propone sobre un rango continuo.

    **Una sola semilla, declarada.** Dos trials sobre semillas distintas medirían
    el techo y el sorteo a la vez, que es la confusión que la lectura apareada de
    la rejilla existía para cancelar. Acá se cancela por construcción: los treinta
    trials de una transferencia corren sobre el material idéntico, dibujado una
    vez.

    **Las seis transferencias, cada una con su propia búsqueda.** La rejilla medía
    dos y las otras cuatro heredaban; las dos derrotas significativas de la
    familia `milcreda` están las dos en transferencias heredadas. Con las seis
    medidas la rama agrupada de `ceiling_for` deja de ser alcanzable.

    **La meseta la define el instrumento, no el GP.** `SEARCH_RESOLUTION` es la
    diferencia más chica que el criterio puede expresar sobre el rol de búsqueda:
    con veinte bolsas, una bolsa. Dos techos que difieren en menos que eso no son
    distinguibles por la medición, opine lo que opine el modelo — y usar una
    cantidad ajustada por el GP haría que el ancho de la meseta dependiera de qué
    tan bien ajustó, que es la propiedad equivocada.

    **El registro se cronometra a sí mismo, al lado de la escala que declara.**
    `atRequiredScale` dice si esta búsqueda vale como respuesta; `seconds` dice
    cuánto costó llegar hasta acá. Los dos juntos son lo único que permite
    proyectar lo que cuesta la corrida completa: con `epochs`, `search.trials` y
    `requiredScale` ya en la entrada, la proyección es aritmética sobre los ejes
    que la entrada misma nombra, y sin el tiempo no hay nada que multiplicar. Un
    ensayo que corre en minutos y una búsqueda completa que corre en horas se ven
    idénticos en el registro anterior a este campo, y el número no se recupera
    después: hay que volver a correr la búsqueda entera para medirlo.

    **Toda la búsqueda de la familia, no la suma de sus transferencias.** Los
    `minutes` que ya viven adentro de `perTransfer` cronometran solo el
    `study.optimize` de cada transferencia y dejan afuera el sorteo del material
    — `bags.build` decodifica miles de imágenes una vez por familia, antes del
    primer trial. Una proyección construida sobre esa suma subestima el costo por
    exactamente ese sorteo, que es trabajo real de la búsqueda. `seconds` abarca
    la familia entera y por construcción es mayor que la suma de sus
    transferencias.

    **En segundos, y al nivel de la familia.** En segundos porque es la unidad en
    la que el resto del registro ya cronometra (`perRun` trae `seconds`), así que
    un lector que sume el costo de una búsqueda y el de una campaña no convierte
    nada; los `minutes` de `perTransfer` son para la línea de progreso, que la lee
    una persona. Al nivel de la familia porque el archivo es un mapeo
    familia -> entrada y nada más: una clave total al nivel de arriba rompería
    `config.ceilings_on_record`, que lee `entry["ceiling"]` de *cada* valor. Quien
    quiera el total suma las dos familias, que además cuestan distinto.
    """
    import optuna
    from optuna.samplers import GPSampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    reduction = replace(reduction, labelNoise=noise, pilot=pilot)
    n_trials = config.PILOT_SEARCH_TRIALS if pilot else config.SEARCH_TRIALS
    low, high = config.CEILING_RANGE
    seed = config.SEARCH_SEED
    ruido = config.SEARCH_RESOLUTION
    # The other five dimensions, each with its own declared range -- see
    # `config.RAMP_DELTA_RANGE`/`KERNEL_SIGMA_RANGE`/`ATTENTION_GAMMA_RANGE`/
    # `ATTENTION_TEMPERATURE_RANGE`/`TAU_LOCAL_RANGE` for where every one came
    # from. Not `lambda_glob`/`lambda_loc`: both are the shared ramp
    # (`ramp(...)`, then `total_objective(..., coefficient, coefficient)`), and
    # a second, independent coefficient for each would change Eq. (39) itself
    # rather than search over it.
    hyper_ranges = {
        "rampDelta": (config.RAMP_DELTA_RANGE, True),
        "kernelSigma": (config.KERNEL_SIGMA_RANGE, True),
        "attentionGamma": (config.ATTENTION_GAMMA_RANGE, False),
        "attentionTemperature": (config.ATTENTION_TEMPERATURE_RANGE, True),
        "tauLocal": (config.TAU_LOCAL_RANGE, True),
    }

    found: dict[str, dict] = {}
    for family, arm_id in config.SEARCH_ARMS.items():
        # Arranca antes del sorteo, no antes del primer trial: el sorteo es
        # trabajo de la búsqueda y dejarlo afuera haría que el número proyectado
        # sea menor que el que se va a pagar.
        arrancada = time.perf_counter()
        # El material se dibuja una vez por familia y se reusa en cada trial:
        # `bags.build` decodifica miles de imágenes y rehacerlo por trial costaría
        # treinta sorteos por transferencia para nada. Además es lo que hace que
        # los treinta trials sean comparables entre sí.
        drawn = {code: bags.build(code, config.DATA_CACHE, seed, noise)
                 for code in config.DOMAINS}
        per_transfer: dict[str, dict] = {}
        for transfer in (transfers or config.SEARCH_TRANSFERS):
            label = transfer_label(transfer)
            material = {"source": drawn[transfer[0]], "target": drawn[transfer[1]]}
            started = time.perf_counter()

            def objective(trial, _t=transfer, _m=material, _a=arm_id):
                ceiling = trial.suggest_float("ceiling", low, high, log=True)
                hyper = {
                    dim: trial.suggest_float(dim, *bounds, log=log)
                    for dim, (bounds, log) in hyper_ranges.items()
                }
                run = run_one(_a, _t, seed, reduction, device, _m,
                              ceiling=ceiling, hyper=hyper, role=config.SEARCH_ROLE)
                return run[config.SEARCH_CRITERION]

            # Una semilla por estudio, derivada y no compartida. Con
            # `seed=config.SEARCH_SEED` en los doce, las propuestas de arranque
            # son idénticas y los doce visitan los mismos puntos: medido en el
            # ensayo local, las seis transferencias de `creda` recorrieron los
            # mismos cuatro techos. Cuando la meseta es ancha el ganador es el
            # más chico *visitado*, así que el registro habría mostrado un
            # acuerdo entre transferencias que era artefacto de la semilla.
            # Derivada de `(familia, transferencia)` con CRC32 —determinista
            # entre corridas, a diferencia de `hash()`, que va salado— para que
            # sigan siendo reproducibles y dejen de ser la misma.
            semilla_estudio = (seed + zlib.crc32(f"{family}|{label}".encode())) % (2 ** 31)
            study = optuna.create_study(direction="maximize",
                                        sampler=GPSampler(seed=semilla_estudio))
            study.optimize(objective, n_trials=n_trials)
            visitados = [
                {"ceiling": t.params["ceiling"],
                 **{dim: t.params[dim] for dim in hyper_ranges},
                 "value": t.value}
                for t in study.trials if t.value is not None
            ]
            elegido = ceiling_record.choose(visitados, ruido)
            elegido["minutes"] = (time.perf_counter() - started) / 60
            per_transfer[label] = elegido
            progress(f"  search {family:>8} {label}: techo={elegido['ceiling']:.4g} "
                     f"criterio={elegido['value']:.3f} meseta={len(elegido['plateau'])} "
                     f"[{elegido['minutes']:.1f} min]")

        # El agrupado es un repliegue que con las seis medidas nadie alcanza:
        # `ceiling_for` solo lo usa para una transferencia que la búsqueda no vio.
        # Se calcula igual, con la misma regla sobre las seis elecciones, para que
        # el registro no tenga un campo que nadie sabe de dónde salió.
        agrupado = ceiling_record.choose(
            [{"ceiling": d["ceiling"], "value": d["best"],
              **{dim: d[dim] for dim in hyper_ranges}}
             for d in per_transfer.values()], ruido)
        found[family] = {
            "arm": arm_id,
            "ceiling": agrupado["ceiling"],
            # The five other searched dimensions, pooled the identical way
            # `ceiling` itself is: the smallest-ceiling winner among the
            # per-transfer plateau, and whichever combination of the other
            # four that trial happened to run under -- see `ceiling_record.
            # choose`'s own docstring. `harness.run_one` never resolves these
            # from `reduction` automatically (only the search's explicit
            # `hyper=` reaches them); a searched value here is recorded, not
            # applied, exactly as `KERNEL_SIGMA`'s own declared default is a
            # hand-set number from a one-time measurement rather than
            # something a run reads out of this file.
            "rampDelta": agrupado["rampDelta"],
            "kernelSigma": agrupado["kernelSigma"],
            "attentionGamma": agrupado["attentionGamma"],
            "attentionTemperature": agrupado["attentionTemperature"],
            "tauLocal": agrupado["tauLocal"],
            "criterion": config.SEARCH_CRITERION,
            "role": config.SEARCH_ROLE,
            "epochs": reduction.epochs,
            # Al nivel de la familia, donde `epochs` y `seeds` ya vivian: es el
            # nivel que el lector de escala recorre, y dejarlo solo adentro de
            # `search` lo haria ilegible para el chequeo que compara la escala
            # corrida contra la declarada.
            "trials": n_trials,
            "search": {"kind": "optuna", "sampler": "GPSampler",
                       "trials": n_trials, "seed": seed,
                       "perStudySeed": "crc32(familia|transferencia) + seed",
                       "space": {
                           "ceiling": {"low": low, "high": high, "log": True},
                           **{dim: {"low": bounds[0], "high": bounds[1], "log": log}
                              for dim, (bounds, log) in hyper_ranges.items()},
                       },
                       "resolution": ruido},
            "decidedByFlatRule": agrupado["decidedByFlatRule"],
            "plateau": agrupado["plateau"],
            # `noise` acá es el término de resolución del GP y precede al eje de
            # ruido de etiqueta; `labelNoise` es la tasa a la que se contaminó el
            # material. Dos cantidades que la palabra sola no separa, así que la
            # nueva se lleva el nombre largo y la vieja no cambia bajo quienes ya
            # la leen.
            "noise": ruido,
            "labelNoise": noise,
            "flatRule": ceiling_record.FLAT_RULE,
            "atRequiredScale": (reduction.epochs >= config.FULL_SEARCH_EPOCHS
                                and n_trials >= config.SEARCH_TRIALS),
            "requiredScale": {"epochs": config.FULL_SEARCH_EPOCHS,
                              "trials": config.SEARCH_TRIALS},
            # Al lado de la escala declarada, porque es con ella que se lee: la
            # escala dice si esta respuesta vale, el tiempo dice cuánto costó, y
            # la proyección a `requiredScale` es el producto de los dos. Abarca
            # el sorteo del material además de los trials, así que es mayor que
            # la suma de los `minutes` de `perTransfer` y no se deriva de ellos.
            "seconds": time.perf_counter() - arrancada,
            "transfers": [transfer_label(t) for t in (transfers or config.SEARCH_TRANSFERS)],
            "neutral": config.RAMP_CEILING,
            # La forma que los consumidores viejos ya leen: etiqueta -> float.
            "byTransfer": {k: d["ceiling"] for k, d in per_transfer.items()},
            "perTransfer": per_transfer,
            "inheritanceRule": "ninguna: las seis transferencias se midieron",
        }

    if governs_the_ceilings_record(noise, transfers):
        record = config.ceilings_record_for(pilot)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(sellar_techos(found, reduction), indent=2),
                          encoding="utf-8")
    return found


def search_ceilings(reduction: Reduction, device: torch.device,
                    progress=print, shard: str | None = None,
                    pilot: bool = False, noise: float = 0.0,
                    transfers: list | None = None) -> dict:
    """Each family's ceiling, found on the selection transfers and kept for its
    derivations.

    Despacha al motor que `config.SEARCH_ENGINE` declare. La rejilla queda entera
    y no por cortesía: escribió el registro que gobierna la campaña vigente, y un
    motor que ya no se puede correr es un registro que ya no se puede reproducir.

    A shared ceiling equalizes the coefficient and unequalizes the balance: the
    two objectives sit a factor of B_src apart, so one number puts adaptation at
    most of one objective and a tenth of the other. This equalizes where each
    method operates instead.

    **What keeps the search out of the verdict is the role, not the transfer.**
    The search reads `valid` (20 bags) and the verdict reads `eval` (36); the two
    are disjoint material, and they stay disjoint whichever transfers each looks
    at. This paragraph used to claim the search ran on transfers the verdict never
    saw, and that was false as configured — `SEARCH_TRANSFERS` was a subset of
    `VERDICT_TRANSFERS`, so both searched transfers were also judged. Nothing
    leaked, because the role split was doing the work the sentence credited to the
    transfer split. Saying it correctly matters now: the sentence would otherwise
    be read as forbidding a search over every transfer, which is exactly what
    removes the out-of-sample inheritance.

    At pilot scale this exercises the pipeline and settles nothing: the ramp runs
    on the fraction of training elapsed, so with three epochs it is saturated by
    the second and every ceiling is reached almost immediately. The search is the
    same program at both scales, which is the point; only its answer is worth
    reading at the scale the protocol declares.
    """
    if config.SEARCH_ENGINE == "optuna":
        reduction = replace(
            reduction,
            epochs=(config.PILOT_SEARCH_EPOCHS if pilot else config.SEARCH_EPOCHS),
            seeds=[config.SEARCH_SEED], labelNoise=noise, pilot=pilot)
        return search_ceilings_trials(reduction, device, progress=progress,
                                      shard=shard, pilot=pilot, noise=noise,
                                      transfers=transfers)

    # Its own scale, and not the caller's. The search is one experiment run once
    # at the scale the campaign runs at; borrowing the pilot's three epochs would
    # answer about a landscape nothing else trains in.
    reduction = replace(
        reduction,
        epochs=(config.PILOT_SEARCH_EPOCHS if pilot else config.SEARCH_EPOCHS),
        seeds=list(config.PILOT_SEARCH_SEEDS if pilot else config.SEARCH_SEEDS),
        labelNoise=noise, pilot=pilot)
    grid = list(config.CEILING_GRID)
    # Las tres coordenadas, de la reducción que se acaba de reconciliar arriba.
    # Con `pilot` solo, la re-búsqueda contaminada del diagnóstico escribía --- y
    # después borraba, en la última línea de esta función --- el parcial de la
    # búsqueda limpia: el mismo archivo, dos experimentos.
    partial = shard_paths(shard, noise=reduction.labelNoise,
                          kind=reduction.kind, pilot=reduction.pilot)['partial']
    # Refused before reading a single cell out of the partial, let alone
    # measuring another one: a partial stamped under a moved `KERNEL_SIGMA` (or
    # written before this stamp existed at all) was measured under a different
    # objective than today's, and resuming it would splice cells from two
    # objectives into one entry with nothing in the record to tell them apart.
    _partial_drift = _partial_stamp_drift(partial)
    if _partial_drift:
        raise SystemExit(
            f"refusing to resume {partial}: stamped under a revision or "
            f"hyperparameters the current config no longer carries: "
            f"{_partial_drift}.\n"
            f"  Delete {partial} to search again under today's config."
        )
    measured = _read_partial(partial)
    if measured:
        done = sum(len(cells) for cells in measured.values())
        progress(f"  resuming: {done} cell(s) already measured, skipping them")
        # A partial is a file on disk; the grid is a line in config.py. Nothing
        # keeps them in step, so a grid edited after a partial was written left
        # a bare KeyError at aggregation, hours in. Refuse by name instead,
        # eagerly, before any family measures another cell.
        grid_set = set(grid)
        for family, cells in measured.items():
            stale = sorted({c for scores in cells.values() for c in scores
                            if c not in grid_set})
            if stale:
                names = ", ".join(f"{c:g}" for c in stale)
                raise SystemExit(
                    "refusing to resume a search measured at ceilings the "
                    "grid no longer has:\n"
                    f"  {family} measured {names}, and CEILING_GRID is now "
                    f"{grid}.\n"
                    f"  Delete {partial} to search again under the current "
                    f"grid, or restore {names} to config.CEILING_GRID to "
                    "keep the cells already measured."
                )
    found: dict[str, dict] = {}
    for family, arm_id in config.SEARCH_ARMS.items():
        # Material outermost, ceilings innermost. Two reasons, and the second is
        # the one that matters: `bags.build` decodes thousands of images, so
        # rebuilding it per ceiling costs five draws per seed for nothing — and
        # every ceiling has to be measured on *the same* material for the paired
        # reading below to cancel anything.
        cells: dict[tuple[int, str], dict[float, float]] = {}
        for seed in reduction.seeds:
            drawn = {code: bags.build(code, config.DATA_CACHE, seed, noise)
                     for code in config.DOMAINS}
            for transfer in (transfers or config.SEARCH_TRANSFERS):
                label = transfer_label(transfer)
                material = {"source": drawn[transfer[0]], "target": drawn[transfer[1]]}
                if (seed, label) in measured.get(family, {}):
                    cells[(seed, label)] = measured[family][(seed, label)]
                    progress(f"  search {family:>8} seed={seed} {label}: ya medida")
                    continue
                started = time.perf_counter()
                for ceiling in grid:
                    run = run_one(arm_id, transfer, seed, reduction, device, material,
                                  ceiling=ceiling, role=config.SEARCH_ROLE)
                    cells.setdefault((seed, label), {})[ceiling] = \
                        run[config.SEARCH_CRITERION]
                minutes = (time.perf_counter() - started) / 60
                progress(f"  search {family:>8} seed={seed} {label}: "
                         + "  ".join(f"{c:g}={cells[(seed, label)][c]:.3f}" for c in grid)
                         + f"   [{minutes:.1f} min]")
                # Written now and not at the end. Cutting a grid of hours on its
                # last cell used to lose every one before it — measured the hard
                # way, at 1h37 for nothing. It also makes the file the progress
                # report: a later session opens it and reads what is measured and
                # what is left, which no amount of stdout could give it once the
                # terminal is gone. And the per-cell minutes turn the cost from
                # somebody's estimate into a number, so a twenty-minute stall in
                # the middle — a machine dropping to battery, say — is visible
                # instead of averaged away.
                _write_partial(family, arm_id, cells, minutes, progress, partial)

        # Paired, not pooled. Comparing bare means folds each cell's own
        # difficulty into the dispersion and drowns the effect; centring every
        # cell on its own mean across ceilings cancels that difficulty, because
        # each ceiling was measured on exactly the same material. It is the same
        # reading `paired_across_transfers` gives the ladder.
        centred: dict[float, list[float]] = {c: [] for c in grid}
        pooled: dict[float, list[float]] = {c: [] for c in grid}
        for scores in cells.values():
            middle = sum(scores.values()) / len(scores)
            for ceiling, value in scores.items():
                centred[ceiling].append(value - middle)
                pooled[ceiling].append(value)

        rows = [{"ceiling": c,
                 config.SEARCH_CRITERION: sum(pooled[c]) / len(pooled[c]),
                 "paired": sum(centred[c]) / len(centred[c]),
                 "n": len(pooled[c])}
                for c in grid]
        # The tie rule, declared rather than inherited from `max`. Ties are not a
        # curiosity here: below some point a term is inert and every ceiling under
        # it scores exactly the same, so on that stretch the tie-break is what
        # chooses. The smallest wins — the same outcome for less adaptation is the
        # weaker claim, and a search should not hand a term more weight than the
        # measurement asked for.
        def pick(candidates: list[dict], key: str = "paired") -> dict:
            top = max(r[key] for r in candidates)
            tied = [r for r in candidates if r[key] == top]
            return min(tied, key=lambda r: r["ceiling"])

        best = pick(rows)
        top = best["paired"]
        tied = [r for r in rows if r["paired"] == top]

        # Whether each seed would have picked the same ceiling on its own. An
        # average hides a choice that flips: three seeds landing on three
        # different ceilings and one landing on the same one produce the same
        # winner and are not the same evidence. It costs nothing — the runs are
        # already done — and it is the only thing here that says whether the pick
        # is a finding or a coin.
        per_seed: dict[int, float] = {}
        for seed in reduction.seeds:
            of_seed = [scores for (s, _), scores in cells.items() if s == seed]
            per_seed[seed] = pick(
                [{"ceiling": c,
                  "paired": sum(s[c] - sum(s.values()) / len(s) for s in of_seed)
                            / len(of_seed)}
                 for c in grid])["ceiling"]
        # What each transfer would have picked on its own, by the same paired
        # rule. These govern the two transfers the search measured; the pooled
        # winner governs the four it never saw. Computed here because the runs
        # are already done — the two readings cost nothing to separate now and
        # cannot be recovered from the pooled grid afterwards.
        by_transfer: dict[str, float] = {}
        for label in sorted({lab for _, lab in cells}):
            of_transfer = [scores for (_, lab), scores in cells.items()
                           if lab == label]
            by_transfer[label] = pick(
                [{"ceiling": c,
                  "paired": sum(s[c] - sum(s.values()) / len(s)
                                for s in of_transfer) / len(of_transfer)}
                 for c in grid])["ceiling"]

        found[family] = {
            "arm": arm_id,
            "ceiling": best["ceiling"],
            # The rule, in the record rather than only in the report: a reader
            # holding this file alone can tell an inherited ceiling from a
            # measured one without knowing which transfers were searched.
            "byTransfer": by_transfer,
            "inheritanceRule": "a transfer the search measured runs at its own "
                               "pick; one it never saw runs at `ceiling`, the "
                               "pooled winner, applied out of sample",
            "criterion": config.SEARCH_CRITERION,
            "grid": rows,
            # How many grid points scored the same. A ceiling chosen between four
            # identical scores and one chosen by a real difference are the same
            # number and not the same evidence, and the record has to say which.
            "tied": [r["ceiling"] for r in tied],
            "decidedByTieBreak": len(tied) > 1,
            "tieRule": "smallest ceiling among the tied: the same outcome for less "
                       "adaptation is the weaker claim",
            "comparison": "paired within (seed, transfer): every ceiling measured "
                          "on the same material, so the cell's own difficulty "
                          "cancels instead of drowning the effect",
            "perSeedPick": {str(seed): value for seed, value in per_seed.items()},
            "seedsAgree": len(set(per_seed.values())) == 1,
            "role": config.SEARCH_ROLE,
            "epochs": reduction.epochs,
            "seeds": list(reduction.seeds),
            # Whether this was searched at the scale the verdict requires. Without
            # it the record and the configuration agree with each other and a
            # ceiling found at pilot scale reads as finished — the same failure the
            # pilot stamp exists to prevent, one experiment over.
            "atRequiredScale": (reduction.epochs >= config.FULL_SEARCH_EPOCHS
                                and len(reduction.seeds) >= config.FULL_SEARCH_SEEDS),
            "requiredScale": {"epochs": config.FULL_SEARCH_EPOCHS,
                              "seeds": config.FULL_SEARCH_SEEDS},
            # Sin `seconds`, a diferencia del motor por trials, y no por olvido.
            # Esta búsqueda reanuda: `measured` devuelve celdas que ya estaban en
            # el parcial y que esta corrida no pagó, así que un solo reloj sobre
            # la familia mediría lo que costó *esta sesión* y se leería como lo
            # que costó la búsqueda. Un número que miente sobre lo que nombra es
            # peor que su ausencia, justo donde se lo usaría para proyectar. Lo
            # que esta forma sí puede cronometrar honestamente es la celda, y ya
            # lo hace: `minutesPerCell` en el parcial, una entrada por celda
            # efectivamente medida.

            "transfers": [transfer_label(t) for t in (transfers or config.SEARCH_TRANSFERS)],
            # The neutral it is read against, so a searched value that lands on it
            # confirms the normalization by measurement rather than by argument.
            "neutral": config.RAMP_CEILING,
        }
    # A su propio registro. Un ensayo nunca escribe donde va la respuesta que la
    # campana consume: ese archivo separado es lo que hace admisible el dial de
    # escala de mas arriba, y sin el volveria a ser lo que la version anterior
    # de este modulo prohibia con razon.
    if governs_the_ceilings_record(noise, transfers):
        record = config.ceilings_record_for(pilot)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(sellar_techos(found, reduction), indent=2),
                          encoding="utf-8")
    # The scratch file goes only once the answer exists. Leaving it would let a
    # later relaunch resume from cells that already produced a finished record.
    partial.unlink(missing_ok=True)
    return found


def campaign(reduction: Reduction, device: torch.device,
             arms: list[str] | None = None, progress=print,
             shard: str | None = None, transfers: list | None = None) -> dict:
    """The whole grid: every arm, every declared transfer, every repetition.

    `transfers` narrows it, and the degradation sweep is the reason: that one
    walks every noise level over a SINGLE transfer, which is a different shape
    from a campaign and not a smaller one. `reduction.kind` keeps the two apart
    on disk --- both can stand at the same rate, and `runs.jsonl` is opened
    `"w"`.


    The search runs first and its answer is part of the bounds, not a detail:
    every family trains at the ceiling it found, every derivation inherits its
    family's, and the verdict is read only over the transfers the search never
    saw. Reading it over all six would let the ceiling's own selection material
    back into the number it was chosen to improve.
    """
    arm_ids = arms or [arm["id"] for arm in config.ARMS]

    # Refused HERE, before any mkdir, `runs.jsonl` truncation or training
    # below -- a stamp mismatch discovered later, inside `run_one`'s first
    # call to `ceiling_for`, still cost a truncated `runs.jsonl` (opened `"w"`
    # a few lines down) and a fully trained arm B beforehand. Both records
    # `search_record` this campaign can end up reading are checked: the one at
    # `reduction.pilot` -- what `reduction.ceilingSearch` is attached from,
    # below -- and, when `reduction.pilot` is true, the full one too, because
    # `governs_the_ceilings_record`'s scale guard a few lines down also reads
    # it and a caller may have reached this function without ever calling
    # `with_ceilings_in_force`/`config.ceilings_on_record` (both refuse this
    # same drift on read, but a `Reduction` built from a hand-assembled record
    # dict skips both).
    for _pilot in ({reduction.pilot, False} if reduction.pilot else {False}):
        _record = search_record(pilot=_pilot) or {}
        _drifted = sorted(family for family, entry in _record.items()
                          if isinstance(entry, dict)
                          and not entry.get("currentStamp", False))
        if _drifted:
            raise SystemExit(
                "refusing to run: ceilings stamped under a different revision "
                f"or hyperparameters ({'pilot' if _pilot else 'full'} record "
                f"at {config.ceilings_record_for(_pilot)}): "
                f"{', '.join(_drifted)}.\n"
                "  Re-run `harness.search_ceilings(...)` under today's config, "
                "or delete the stale record to search again."
            )

    # El árbol de ESTA corrida. Era `config.RESULTS` a secas, que es el de la
    # corrida completa y limpia: un ensayo, o un nivel del barrido, creaba el
    # directorio de la campaña completa antes de escribir una sola línea en el
    # suyo --- un directorio vacío donde algo después busca evidencia.
    config.results_for(reduction.labelNoise, reduction.kind,
                       reduction.pilot).mkdir(parents=True, exist_ok=True)
    # Sólo donde algo los va a leer. Un nivel sin lector que igual escribiera
    # 8 GB de pesos dejaría un directorio que nadie abre y nadie borra, que es
    # peor que una ausencia: parece evidencia.
    pesos = config.models_for(reduction.labelNoise, reduction.kind,
                             reduction.pilot)
    if config.keeps_checkpoints(reduction.labelNoise):
        pesos.mkdir(parents=True, exist_ok=True)

    # Consumed, never searched here. A campaign that funded its own coefficient
    # out of the run it is about to report would be choosing and judging in one
    # pass, and the refusal is what makes "searched once, beforehand, at the
    # campaign's own scale" a fact about the record rather than a convention.
    if not reduction.ceilings:
        raise SystemExit(
            "refusing to run without the searched ceilings.\n"
            "  Each family's ceiling is one experiment, run once at "
            f"{config.SEARCH_EPOCHS} epochs, before any campaign.\n"
            "  Run `harness.search_ceilings(...)` and load its record, or pass "
            "`ceilings=` explicitly."
        )

    # The record has per-transfer picks and this `Reduction` does not. That is
    # not a difference of opinion, it is a stale field: `config` was imported
    # before the record existed, so the default was empty and a caller set only
    # `ceilings`. Running anyway would apply the pooled winner to the two
    # transfers the search measured — the old rule, arriving with no sign that
    # anything was skipped. Refuse by name instead.
    # `pilot=reduction.pilot`, o sea el registro del que esta reducción sacó sus
    # techos, y no el vigente. Sin la coordenada este chequeo leía un archivo y
    # la reducción venía del otro: una campaña de ensayo bajo un registro
    # completo con picks se negaba nombrando picks que su propio registro no
    # tiene, y una campaña completa sin `ceilings.json` era medida contra los
    # picks del ENSAYO. No es la guarda de escala --- ésa sigue dos bloques más
    # abajo, pidiendo `pilot=False` por su nombre, y no se mueve.
    on_record = config.ceilings_by_transfer_on_record(pilot=reduction.pilot)
    missing = sorted(family for family, picks in on_record.items()
                     if picks and not reduction.ceilingsByTransfer.get(family))
    if missing:
        raise SystemExit(
            "refusing to run with the per-transfer ceilings left behind:\n"
            f"  the record has picks for {', '.join(missing)} and this run "
            "carries none, so every transfer would fall back to the pooled "
            "winner.\n"
            "  Build the reduction with `harness.with_ceilings_in_force(...)` "
            "rather than setting `ceilings=` alone."
        )
    # And not a ceiling searched below the scale its answer needs. Missing is the
    # obvious failure; this is the quiet one — someone lowers the search to test
    # the pipeline cheaply, `ceilings.json` gets written from three epochs, and
    # every campaign afterwards consumes it without a word.
    # `pilot=False` dicho y no heredado de la firma: ésta es la lectura que
    # GOBIERNA --- dos líneas más abajo se exige `atRequiredScale` sobre ella ---
    # y desde que la omisión significa «el que rige», una llamada pelada le daría
    # el registro del ensayo y esta campaña se negaría por una escala que el
    # ensayo nunca prometió. Un registro de ensayo no gobierna una campaña real:
    # es exactamente la sustitución que este rechazo existe para impedir.
    searched = search_record(pilot=False) or {}
    # The whole record travels into the reduction, here and not at the caller's
    # hand. The winner alone cannot say whether the grid leaned or the tie-break
    # chose, and a field a caller has to remember to fill is one that gets filled
    # on the run somebody was paying attention and left empty on the next.
    #
    # El registro de ESTA corrida, y no `searched`. Las dos lecturas compartían
    # una variable y son dos preguntas distintas: `searched` es la que GOBIERNA
    # ---abajo se le exige `atRequiredScale` y por eso nombra `pilot=False`---
    # y esto es la EVIDENCIA de la que salió el escalar con el que se corrió,
    # que `tables.conclusion_ceilings` imprime al pie de las tablas. Compartida,
    # una campaña de ensayo se sellaba con la rejilla de la búsqueda completa
    # ---o con `{}` cuando esa búsqueda no existía--- mientras corría bajo los
    # techos del ensayo: el informe describía cómo se eligió un número que esa
    # corrida no usó, y las dos cosas son igual de plausibles en la página.
    reduction = replace(reduction,
                        ceilingSearch=search_record(pilot=reduction.pilot) or {})
    under = [family for family, entry in searched.items()
             if not entry.get("atRequiredScale", False)]
    if under:
        raise SystemExit(
            f"refusing to run on ceilings searched below scale: {', '.join(under)}.\n"
            f"  The search needs {config.FULL_SEARCH_EPOCHS} epochs and "
            f"{config.FULL_SEARCH_SEEDS} repetitions; the record says otherwise.\n"
            f"  Re-run the search, or delete {config.CEILINGS_RECORD.name} to start over."
        )
    progress(f"Ceilings in force: {reduction.ceilings}")

    # This shard's own file, so two running at once cannot clobber one another.
    # `None` keeps the single-machine path exactly where it has always been.
    paths = shard_paths(shard, noise=reduction.labelNoise,
                        kind=reduction.kind, pilot=reduction.pilot)
    paths["runs"].parent.mkdir(parents=True, exist_ok=True)
    write_shard_stamp(shard, reduction)
    records = paths["runs"].open("w", encoding="utf-8")

    # Seeds sit outermost so a repetition's material — the stratified draw, the
    # composition of the bags and the split — is built once for all three domains
    # and reused across that seed's transfers. A repetition that reused another
    # seed's bags would vary the initialization and nothing else.
    cells: dict[tuple[str, str], list[dict]] = {}
    manifests: dict[tuple[str, str], dict] = {}
    for seed in reduction.seeds:
        drawn = {code: bags.build(code, config.DATA_CACHE, seed, reduction.labelNoise)
                 for code in config.DOMAINS}
        # Only the transfers the search never looked at. The other two funded the
        # ceiling and cannot also carry the verdict it was chosen to improve.
        for transfer in (transfers or config.VERDICT_TRANSFERS):
            label = f"{transfer[0]}->{transfer[1]}"
            material = {"source": drawn[transfer[0]], "target": drawn[transfer[1]]}
            manifests[(label, seed)] = {"source": material["source"].manifest,
                                        "target": material["target"].manifest}
            of_cell: list[dict] = []
            for arm_id in arm_ids:
                run = run_one(arm_id, transfer, seed, reduction, device, material)
                state = run.pop("state")
                # Descartado en vez de escrito donde no hay lector: la curva de
                # degradación sale de `runs.jsonl`, que todo nivel escribe.
                if state is not None and config.keeps_checkpoints(reduction.labelNoise):
                    stem = f"{arm_id}_{label.replace('->', '-')}_seed{seed}"
                    torch.save({k: v.cpu() for k, v in state.items()},
                               pesos / f"{stem}.pt")
                records.write(json.dumps(run) + "\n")
                records.flush()
                cells.setdefault((label, arm_id), []).append(run)
                of_cell.append(run)
            # One line per (seed, transfer) cell, written once its arms are
            # done, and not one per run: six transfers over thirty seeds is
            # 180 lines where a line per run would be 1800, and 1800 lines is
            # a report nobody reads rather than a sign of life.
            #
            # No timing rides along any more: time and memory (`seconds`/
            # `peakMiB`) are removed from this comparison entirely, so there
            # is nothing left to name as "slowest". `runs.jsonl` keeps every
            # reading either way -- this line is progress, not the record.
            if of_cell:
                targets = [r["targetAccuracy"] for r in of_cell]
                progress(f"  {label} seed {seed}: {len(of_cell)} arms  "
                         f"target {min(targets):.3f}-{max(targets):.3f}")
        del drawn

    records.close()

    grid: dict[str, dict] = {}
    checkpoints: dict[str, list[str]] = {}
    for (label, arm_id), cell_runs in cells.items():
        grid.setdefault(label, {})[arm_id] = summarize(cell_runs)
        if arm_id in config.CHECKPOINTS and config.keeps_checkpoints(reduction.labelNoise):
            checkpoints[f"{arm_id} {label}"] = keep_median(
                cell_runs, arm_id, label, manifests, reduction)

    per_transfer = {label: judge(ladder_rows(cell, label)) for label, cell in grid.items()}

    # Una dimensión `perRun` no tiene media que este módulo esté dispuesto a
    # entregar, así que el informe la lee corrida por corrida --- y esa forma la
    # armaba únicamente `shards.merge`. Una campaña en una sola máquina escribía
    # entonces un resumen sin ella, y la sección de tiempo del informe salía
    # vacía con las corridas en `runs.jsonl` todo el tiempo. Una máquina es un
    # entorno: la mediana entre semillas dentro de él, con su rango, es
    # exactamente lo que esa sección dice que muestra.
    #
    # Import local porque `shards` importa este módulo. Se reusa su función en
    # vez de repetirla acá: dos implementaciones de una misma forma son dos que
    # pueden separarse, y el informe no podría decir cuál de las dos leyó.
    from MIL_CREDA_Benchmark import shards as _shards
    _, _, per_run = _shards.partition(config.DIMENSIONS, _shards.declaration())
    grid_per_run = _shards.per_run_grid(
        [run for cell_runs in cells.values() for run in cell_runs], per_run)

    summary = {
        "kind": "bounded",
        "reduction": asdict(reduction),
        "verdictsMeaningful": reduction.verdicts_meaningful,
        "grid": grid,
        "gridPerRun": grid_per_run,
        "perTransfer": per_transfer,
        "panorama": paired_across_transfers(grid),
        "checkpoints": checkpoints,
        "tally": {label: tally(rows) for label, rows in per_transfer.items()},
    }
    # Al lado de las corridas que resume, y no en `config.RESULTS`. Anclado ahí
    # dejaba un ensayo escribiendo sus corridas en el árbol de piloto y su
    # resumen en el de la corrida completa.
    destino = paths["runs"].parent
    (destino / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    # The same record, where the verification looks for it. A correct summary at a
    # path nobody opens protects nothing: a later session would find no record at all
    # and start over as though this had never run. `targetScale` beside `reduction` is
    # what lets it tell a pilot from a campaign instead of only a revision from a
    # stale one.
    product_results = (config.results_for(0.0, reduction.kind,
                                          reduction.pilot).parent
                       / "Probe_results.json")
    product_results.parent.mkdir(parents=True, exist_ok=True)
    product_results.write_text(json.dumps({
        "kind": summary["kind"],
        "revision": reduction.revision,
        "reduction": asdict(reduction),
        "targetScale": {"epochs": config.FULL_EPOCHS, "seeds": config.FULL_SEEDS},
        "verdictsMeaningful": reduction.verdicts_meaningful,
        "comparison": summary["panorama"],
        "detail": str((destino / "summary.json").relative_to(config.REPOSITORY)),
        "figures": sorted(str(p.relative_to(config.REPOSITORY))
                          for p in destino.rglob("*.pdf")),
    }, indent=2), encoding="utf-8")

    # Sealed last, once every file this shard writes actually exists. A run
    # that raises anywhere above never reaches this line, so its stamp stays
    # unsealed and therefore incomplete — never merge-eligible.
    seal_shard_stamp(shard, noise=reduction.labelNoise,
                     kind=reduction.kind, pilot=reduction.pilot)
    return summary


def run_pilot(epochs: int = config.EPOCHS, seeds: list[int] | None = None) -> dict:
    """One single-machine campaign, headless: exactly what
    `Benchmark_Campaign_v1.ipynb`'s own pilot cells run, callable with plain
    JSON-native arguments instead of from a notebook.

    `ceilings_in_force()` obtains what `campaign()` refuses to run
    without — searching once if `ceilings.json` has no record yet, or
    reusing it unchanged if it does, the exact join
    `test_the_notebook_obtains_the_ceilings_before_it_runs_the_campaign`
    already holds the notebook to. `campaign()` then runs the whole grid,
    never a shard of it — `tools/distribute.py`'s `run_shard()` is the
    fan-out path for splitting the seed axis across several remote
    workers; this is the single-machine counterpart it is not.

    `epochs` is the one dial a caller may cheapen without touching a
    scientific constant: `Reduction`'s own default already reads
    `config.EPOCHS`, so calling this with no override reproduces that
    default exactly, and a caller may reach a cheaper number — for a
    firedrill proving a remote pipeline runs at all, say — by passing
    `epochs=` explicitly rather than editing `config.py`.
    """
    device = resolve_device()
    reduction = Reduction(
        seeds=list(seeds) if seeds is not None else list(config.SEEDS),
        epochs=epochs, device=str(device), environment=environment())
    reduction = with_ceilings_in_force(reduction, device)
    return campaign(reduction, device)


def run_search(shard: str | None = None, pilot: bool = False) -> dict:
    """The ceiling search alone, headless, and nothing after it.

    `search_ceilings()` takes a `Reduction` and a `torch.device`, neither of
    which a caller holding only JSON can build — and a remote job names a
    `module.function` and hands it JSON keyword arguments. So the search had
    a launcher for a notebook and a launcher for a whole campaign, and none
    for itself: the one experiment this repository is currently blocked on
    was the one with no way to ask for it. This is that way.

    Stops at the record. `run_pilot()` searches *and then* runs the grid,
    which is right for a single machine walking the whole thing in one go
    and wrong for a worker asked to settle one question — the campaign is
    73 hours at full scale and the search is under three, and conflating
    them means a job that cannot be sized. `ceilings_in_force()` is what
    both call, so an existing record is reused here exactly as it is there:
    a record that exists means the search answered, and re-answering it
    because a later caller wanted a different answer is the silent
    refunding `campaign()`'s own refusal exists to prevent.

    **There is no `epochs` dial, and `pilot=True` is not one.** The dial this
    function refused had one destination: a ceiling found at pilot scale
    settles nothing and would still have been written to the file the
    full-scale answer goes to, so it would only have looked like it worked.
    That objection was about the destination, and the destination changed —
    `pilot=True` runs at `config.PILOT_SEARCH_EPOCHS` and writes to
    `config.CEILINGS_PILOT_RECORD`, a separate file the full record always
    outranks. It rehearses the program and its answer is still not quotable:
    the ramp runs on the fraction of training elapsed, so at pilot scale it
    saturates by the second epoch and every ceiling is reached almost at
    once. What it settles is whether the search *runs*, which is the only
    question a rehearsal is entitled to answer.

    `shard` names this call's own shard namespace, the same parameter
    `run_smoke()` and `distribute.run_shard()` take, so a search split across
    workers never has two of them writing one partial.
    """
    device = resolve_device()
    # The search's own scale, stated rather than left implied. `search_ceilings`
    # sets these itself, so this is a no-op for behaviour — and it is written
    # anyway, because a `Reduction` that says three epochs while the run it
    # describes does twenty is a record that lies about itself.
    reduction = Reduction(
        seeds=list(config.PILOT_SEARCH_SEEDS if pilot else config.SEARCH_SEEDS),
        epochs=(config.PILOT_SEARCH_EPOCHS if pilot else config.SEARCH_EPOCHS),
        device=str(device), environment=environment())
    ceilings_in_force(reduction, device, shard=shard, pilot=pilot)
    # Read back from disk, never from what the search returned, for the reason
    # `ceilings_in_force` already gives: what the campaign will run at is what
    # the record says, and the record is the thing a later session reads.
    record = search_record(pilot=pilot)
    if record is None:
        raise SystemExit(
            "the search ran and left no record at "
            f"{config.ceilings_record_for(pilot)}. "
            "Nothing downstream can read a ceiling "
            "that was never written down, and a run that reports success "
            "without one would be claiming an answer it cannot show."
        )
    return record


def run_campaign_shard(shard: str | None = None,
                       seeds: list[int] | None = None,
                       pilot: bool = False) -> dict:
    """One shard of the campaign, headless, callable with JSON alone.

    `tools/distribute.py`'s `run_shard()` already does this and a remote worker
    cannot reach it: `tools/` is outside every declared clone path, and that
    module path-imports the forge's own packer, which does not exist inside a
    kernel. So the fan-out this repository's distribution declaration is built
    around — the seed axis split across machines — had no way to be asked for
    from a job. This is that way, and it is the same three lines, placed where a
    clone can see them.

    The seed list is the whole parameterisation of a SHARD, because the seed
    axis is what the declaration says may be split
    (`__benchmark__["distribution"]["axis"]`). `pilot` is a different
    coordinate and not a second way to spell that one: it moves the scale AND
    the destination together, exactly as `run_search(pilot=...)` does.

    **The scale is received, never guessed.** This function used to hardcode
    `config.FULL_SEEDS`/`config.FULL_EPOCHS` and take no dial, and the
    argument written here was that a shard measured at pilot scale is not a
    cheaper shard but a different experiment, which would be merged with the
    others as though it were one. **That argument is still true, and it is
    still the reason the full path below is untouched** — a `pilot=False`
    call builds the identical `Reduction` it always did and writes where it
    always wrote. What it did not establish is the conclusion it was used
    for: a run routed to the PILOT tree is never merged with anything,
    because it is not in the tree anything merges from. That is precisely
    what `barrido_de_ruido` already relies on, one experiment over.

    What the missing dial cost is measured rather than argued: the declared
    flow was walked at pilot scale and this function trained the FULL grid —
    five arms, six transfers, thirty seeds, twenty epochs — for 56 minutes
    before a person killed it by hand. Nothing refused, because nothing here
    could tell a pilot walk from a campaign.

    `pilot=True` carries `config.SEEDS`/`config.EPOCHS` — the two constants
    `config.is_pilot_scale()` reads, and the only two that separate the
    scales — and `pilot=True` on the `Reduction` itself, which is what
    `results_for`/`models_for` route on. It also decides which ceiling record
    `with_ceilings_in_force` reads, through `reduction.pilot`: a pilot
    campaign consumes `ceilings.pilot.json`, which is what `search-pilot`
    produces, and never the full record it has no business consuming.

    `shard` names this call's own namespace and is passed through to
    `campaign()`, which hands it to `shard_paths()`. Without it two shards
    running at once write one `runs.jsonl` and one stamp between them, and the
    loser is a silent partial. Defaulting to `None` is the single-machine case,
    where there is nothing to collide with.

    An explicit `seeds` still wins over the scale's own list, at either
    scale: that is the shard split, and a caller naming one has already said
    which repetitions this machine owns.

    Refuses nothing here that `campaign()` does not already refuse: it still
    demands a ceiling record, and a worker that clones only `src/` does not
    receive one. That is a property of what the job declares it clones, not
    something this function can paper over.
    """
    device = resolve_device()
    reduction = Reduction(
        seeds=list(seeds) if seeds is not None
        else list(config.SEEDS if pilot else config.FULL_SEEDS),
        epochs=config.EPOCHS if pilot else config.FULL_EPOCHS,
        pilot=pilot, device=str(device), environment=environment())
    reduction = with_ceilings_in_force(reduction, device, shard=shard)
    return campaign(reduction, device, shard=shard)


def run_mechanism_sweep_shard(seeds: list[int] | None = None,
                              pilot: bool = False) -> dict:
    """The attention-mechanism comparison, headless, callable with JSON
    alone -- `run_campaign_shard`'s own sibling for Section 4.

    Runs both conditions section 4 reads (`clean`, rate `0.0`, then `noisy`,
    `config.NOISE_REPORTED`) in one call, so a single job submission leaves
    the record complete rather than requiring two separate ones the operator
    would have to remember to both send.

    **The scale is received, never guessed** -- the same dial
    `run_campaign_shard` now takes, and the same reasoning, which is written
    out there rather than restated here. The sentence this docstring used to
    carry -- "full scale always, the same reason `run_campaign_shard` never
    takes a pilot dial: a comparison measured at three epochs is a different
    experiment, not a cheaper one" -- was true about MERGING and was used to
    conclude something about the DIAL. A pilot comparison is a different
    experiment, which is exactly why it is written to a different tree and
    merged with nothing.

    `pilot` here moves the record too, and it has to be said out loud
    because this one does not route through `results_for` the way a campaign
    does: `run_mechanism_sweep` writes one JSON file whose path was a fixed
    string, so a pilot dial alone would have written three-epoch numbers over
    the full record Section 4 is read from. That is the defect this whole
    change exists to close, wearing the opposite mask.
    """
    device = resolve_device()
    reduction = Reduction(
        seeds=list(seeds) if seeds is not None
        else list(config.SEEDS if pilot else config.FULL_SEEDS),
        epochs=config.EPOCHS if pilot else config.FULL_EPOCHS,
        pilot=pilot, device=str(device), environment=environment())
    reduction = with_ceilings_in_force(reduction, device)
    run_mechanism_sweep(reduction, device, noise=0.0)
    return run_mechanism_sweep(reduction, device, noise=config.NOISE_REPORTED)


#: What a smoke run stamps as its ceiling, for both families, so
#: `write_shard_stamp`'s record is honest about what was used rather than
#: leaving it implied by `ramp()`'s own per-call default. `RAMP_CEILING` is
#: not chosen for this run: it is the value every derivation already falls
#: back to before any search has run — `ramp()`'s own default argument, and
#: the neutral of a normalized Eq. (39), declared once in `config.py` and
#: never the search's to find. A smoke run reuses that declared neutral
#: rather than inventing a ceiling of its own.
SMOKE_CEILINGS: dict[str, float] = {family: config.RAMP_CEILING
                                    for family in config.SEARCH_ARMS}

#: The arm and transfer a smoke run exercises. `G` is the complete method —
#: weighting, the learned selector and the local term all fire, which is
#: more of `wiring.build`'s own branching than any lighter arm reaches — and
#: `VERDICT_TRANSFERS[0]` is `M->U`, the same pair `SEARCH_TRANSFERS` leads
#: with. Neither choice is a scientific one: a smoke run reports no accuracy
#: anybody is meant to read, only that the wire from `bags.build` through
#: `wiring.build` to a sealed stamp still carries current.
SMOKE_ARM = "G"


def run_smoke(seed: int = 0, shard: str | None = None,
              checkpoint: bool = False) -> dict:
    """The smallest slice that exercises a real shard's whole wire, and
    nothing past it: one arm, one transfer, one seed, two epochs.

    Every wire a real shard uses, in the same order `campaign()` uses them:
    `resolve_device()`, `bags.build()`, `wiring.build()` (through
    `run_one()`), then `write_shard_stamp()` / `seal_shard_stamp()` around a
    `runs.jsonl` this function writes itself. `campaign()` is not called
    here — it always walks every one of `config.VERDICT_TRANSFERS` for every
    arm it is given, with no argument that narrows it to one transfer — so a
    single-slice smoke does its own minimal version of `campaign()`'s
    bookkeeping instead, over exactly one `run_one()` call.

    Never calls `ceilings_in_force()` or `search_ceilings()` — the one hard
    requirement this function exists to meet. `run_one()` needs a ceiling to
    run at all, and gets one explicitly here, from `reduction.ceilings`
    (`SMOKE_CEILINGS`): the module's own already-declared neutral, not a
    value chosen by outcome and not a shortcut through the search. Skipping
    the search is therefore not a scientific claim about where either
    family's ceiling actually sits — it is a statement that this run is
    plumbing, not a result. A smoke run never writes `summary.json` or
    `Probe_results.json`, and `campaign()` run for real still refuses
    without a `ceilings.json` the search produced; nothing here weakens
    that refusal.

    **That independence is now a property of some rehearsals, not of every
    one, and this paragraph used to claim otherwise.** It read as the
    definition of what a rehearsal is: depend on nothing, so it can run
    before anything exists. The rule that replaced it is narrower and
    stricter — a rehearsal on the worker consumes what the steps before it
    left at FULL scale, so that the step is proven against the inputs the
    real run will open rather than against a stand-in. Reading a declared
    neutral is exactly the stand-in that rule forbids.

    So this function keeps its independence and loses its monopoly.
    `steps.ensayo_remoto` routes here for a step with no predecessor — a
    step whose declared `reads` is empty, which is derived from the chain
    and never a name written down — because there is nothing upstream for
    such a step to consume and what is left to prove is that the wire
    carries current on THIS machine. Every other step's rehearsal runs the
    step's own notebook against its predecessors' full-scale output, and
    refuses when that output is not there yet.

    `shard` names this call's own shard namespace, the same parameter
    `distribute.run_shard()` takes, passed through to `shard_paths()` so a
    smoke rehearsal never collides with a real shard's files sharing the
    same directory. Defaults to `None` — one kernel container runs one
    smoke call, so there is nothing else in that container to collide with.

    `checkpoint`, off by default, writes exactly one `.pt` and its manifest
    through `campaign()`'s own path — `torch.save({k: v.cpu() for k, v in
    state.items()}, ...)` beside a manifest `keep_median()` writes with
    `bags.write_manifest()` — rather than a bespoke save. Phase 2's
    `latent.available()`/`latent.load()` read `config.models_for()` in
    exactly that shape, so a checkpoint proven any other way would not
    prove the path a real campaign actually uses — and both sides get the
    coordinates from this call's own reduction, so they cannot drift
    apart. `config.CHECKPOINTS[SMOKE_ARM]` asks for three per cell; with
    exactly one seed here, `median_seeds()`
    degenerates to the only one there is rather than pruning it away. Off by
    default because most rehearsals only need `runs.jsonl`, and a checkpoint
    costs a model's worth of disk on every call that does not ask for one.
    """
    device = resolve_device()
    transfer = config.VERDICT_TRANSFERS[0]
    reduction = Reduction(
        seeds=[seed], epochs=2, device=str(device),
        environment=environment(), ceilings=dict(SMOKE_CEILINGS),
    )

    # De la reducción de arriba y no de los valores por omisión de
    # `shard_paths`. Coinciden hoy --- este ensayo corre limpio, en forma de
    # campaña y sin marca de piloto --- y coincidían por casualidad: las tres
    # coordenadas estaban escritas dos veces, una en la reducción que sella y
    # otra implícita en el destino, y el día que una se moviera el sello y las
    # corridas se iban a árboles distintos sin que nada lo dijera.
    paths = shard_paths(shard, noise=reduction.labelNoise,
                        kind=reduction.kind, pilot=reduction.pilot)
    paths["runs"].parent.mkdir(parents=True, exist_ok=True)
    write_shard_stamp(shard, reduction)

    drawn = {code: bags.build(code, config.DATA_CACHE, seed)
             for code in {transfer[0], transfer[1]}}
    material = {"source": drawn[transfer[0]], "target": drawn[transfer[1]]}
    run = run_one(SMOKE_ARM, transfer, seed, reduction, device, material)
    state = run.pop("state", None)

    if checkpoint and state is not None:
        pesos = config.models_for(reduction.labelNoise, reduction.kind,
                                  reduction.pilot)
        pesos.mkdir(parents=True, exist_ok=True)
        label = run["transfer"]
        stem = f"{SMOKE_ARM}_{label.replace('->', '-')}_seed{seed}"
        torch.save({k: v.cpu() for k, v in state.items()},
                   pesos / f"{stem}.pt")
        manifests = {(label, seed): {"source": material["source"].manifest,
                                     "target": material["target"].manifest}}
        keep_median([run], SMOKE_ARM, label, manifests, reduction)

    with paths["runs"].open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(run) + "\n")

    seal_shard_stamp(shard, noise=reduction.labelNoise,
                     kind=reduction.kind, pilot=reduction.pilot)
    return {
        "arm": SMOKE_ARM,
        "transfer": run["transfer"],
        "seed": seed,
        "targetAccuracy": run["targetAccuracy"],
        "sourceAccuracy": run["sourceAccuracy"],
    }


def header(reduction: Reduction) -> str:
    lines = [
        f"setting={reduction.setting}  backbone={reduction.backbone}  "
        f"bags={reduction.bagsPerDomain}x{reduction.instancesPerBag}  "
        f"split={reduction.trainBags}/{reduction.evalBags}  "
        f"epochs={reduction.epochs}  seeds={len(reduction.seeds)}  "
        f"device={reduction.device}  revision={reduction.revision}",
    ]
    if not reduction.verdicts_meaningful:
        lines.append(
            f"!! {len(reduction.seeds)} repetition(s): the dispersion is zero, so the "
            f"threshold is zero and every row below declares a winner from a bare "
            f"difference. These are point estimates, not verdicts."
        )
    return "\n".join(lines)


