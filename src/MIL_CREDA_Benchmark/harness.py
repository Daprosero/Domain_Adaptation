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
import tracemalloc
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
import torch.nn as nn

from CREDA.schedules import creda_ramp
from MIL_CREDA_Benchmark import bags, config, report_digest, wiring
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


def synchronize(device: torch.device) -> None:
    """Wait for the device before reading the clock.

    CUDA and MPS queue their kernels, so a timer stopped without this measures
    when the work was submitted rather than when it finished — a wall time that
    looks precise and describes nothing.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def environment() -> dict:
    """Where this ran, recorded rather than assumed.

    The guard exists because wall time and peak memory describe whichever
    environment produced them. It refuses when the repository has its own
    virtualenv and something else is running — that is a mistake worth stopping.
    On a hosted runtime there is no such virtualenv and nothing to compare
    against, so the environment is stamped into the summary instead: a table made
    in Colab is labelled as made in Colab rather than attributed to this machine.
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
    one accelerator and land on another without a word — and `seconds` and
    `peakMiB` are dimensions of the verdict, so grouping them by an environment
    that cannot tell two GPU classes apart is grouping by a label that lies.

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

    `seconds` and `peakMiB` are dimensions of the verdict, not decoration, and a
    measurement describes whichever environment produced it. A laptop that drops
    to battery partway through a long grid throttles, and if that catches some
    arms and not others it is a difference between arms nobody declared — the
    ladder would credit a mechanism with what the power state did.

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
    """`get_eta` of CREDA's pipeline: 1e-3 decaying toward 1.4e-4."""
    p = epoch / epochs
    return config.LR * (1 + config.LR_ALPHA * p) ** (-config.LR_BETA)


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
    #: The neutral each family's searched ceiling is read against.
    rampCeiling: float = config.RAMP_CEILING
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
    ceilingSearch: dict = field(default_factory=dict)
    rampDelta: float = config.RAMP_DELTA
    device: str = "cpu"
    environment: dict = field(default_factory=dict)

    @property
    def verdicts_meaningful(self) -> bool:
        return len(self.seeds) >= 3


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
    """
    if family is None:
        return config.RAMP_CEILING
    pooled = reduction.ceilings.get(family, config.RAMP_CEILING)
    return (reduction.ceilingsByTransfer.get(family, {})
            .get(transfer_label(transfer), pooled))


def run_one(arm_id: str, transfer: tuple[str, str], seed: int,
            reduction: Reduction, device: torch.device,
            material: dict, ceiling: float | None = None,
            role: str = "eval") -> dict:
    """One arm, one transfer, one repetition, end to end.

    `ceiling` overrides the family's for this run. The search passes it to walk
    the grid; the campaign passes each family's found value, so every arm derived
    from a family inherits the one that family searched.

    `role` is which material the run is judged on. The search reads `valid` and
    the campaign reads `eval`, and they are disjoint by construction — a
    coefficient chosen on the material the verdict rests on would make the
    verdict report a decision it already made, by an amount nobody can subtract
    afterwards.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 9973)

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
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    steps = -(-config.TRAIN_BAGS // config.BAGS_PER_STEP)

    curve: list[dict] = []
    epochs_record: list[dict] = []

    tracemalloc.start()
    synchronize(device)
    started = time.perf_counter()
    model.train()
    for epoch in range(reduction.epochs):
        family = config.ARMS_BY_ID[arm_id]["adaptation"]
        # The family's ceiling on this transfer, or the one this call was
        # handed. Never a global: each family keeps what it searched, its
        # derivations inherit it, and a transfer the search measured keeps its
        # own pick rather than the pooled one.
        top = (ceiling_for(reduction, family, transfer)
               if ceiling is None else ceiling)
        coefficient = ramp(epoch, reduction.epochs, family, ceiling=top)
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

    synchronize(device)
    seconds = time.perf_counter() - started
    peak = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    contributions = [abs(point["contribution"]) for point in curve]
    # The supervised magnitude has to leave this function or it is gone: the curve
    # is discarded at the end of the run and no checkpoint can recover it. Without
    # it `contribution` is a bare number, and "the term commanded nothing" and "the
    # term was scaled to nothing" read identically. Eq. (18) is divided by B_src
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
        "seconds": seconds,
        "peakMiB": peak,
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

    kept: list[str] = []
    for run in cell_runs:
        stem = f"{arm_id}_{transfer.replace('->', '-')}_seed{run['seed']}"
        weights = config.MODELS / f"{stem}.pt"
        if run["seed"] not in keep:
            weights.unlink(missing_ok=True)
            continue
        bags.write_manifest(
            config.MODELS / f"{stem}.manifest.json",
            arm=arm_id, transfer=transfer, seed=run["seed"],
            targetAccuracy=run["targetAccuracy"],
            sourceAccuracy=run["sourceAccuracy"],
            reduction=asdict(reduction),
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


def shard_paths(shard: str | None) -> dict:
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
    if shard is None:
        return {"runs": config.RESULTS / "runs.jsonl",
                "partial": config.CEILINGS_RECORD.with_name(
                    config.CEILINGS_RECORD.stem + PARTIAL_SUFFIX),
                "stamp": config.RESULTS / "shard.json"}
    home = config.RESULTS / SHARDS_DIR / shard
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
    path = shard_paths(shard)["stamp"]
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = reduction.environment or environment()
    path.write_text(json.dumps({
        "shard": shard,
        "env": environment_key(stamp),
        "environment": stamp,
        "seeds": list(reduction.seeds),
        "epochs": reduction.epochs,
        "revision": reduction.revision,
        "ceilings": dict(reduction.ceilings),
        "ceilingsByTransfer": {family: dict(picks) for family, picks
                               in reduction.ceilingsByTransfer.items()},
        "evidence": _evidence(),
    }, indent=2), encoding="utf-8")
    return path


def seal_shard_stamp(shard: str | None) -> Path:
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
    path = shard_paths(shard)["stamp"]
    stamp = json.loads(path.read_text(encoding="utf-8"))
    outputs = sorted(p.name for p in path.parent.iterdir() if p.is_file())
    stamp.setdefault("evidence", {})["outputs"] = outputs
    partial = path.with_name(path.stem + PARTIAL_SUFFIX)
    partial.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    os.replace(partial, path)
    return path


def _partial_path() -> Path:
    return shard_paths(None)["partial"]


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored, indent=2), encoding="utf-8")


def search_record() -> dict | None:
    """The ceiling search's own record, whole, or None when it has not run.

    Read from disk and not carried in memory: `config.CEILINGS` keeps only the
    winners, and the winner alone cannot say whether it was searched at scale or
    whether the seeds agreed on it.
    """
    if not config.CEILINGS_RECORD.exists():
        return None
    return json.loads(config.CEILINGS_RECORD.read_text(encoding="utf-8"))


def ceilings_in_force(reduction: Reduction, device: torch.device,
                      progress=print, shard: str | None = None) -> dict[str, float]:
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
    """
    if search_record() is None:
        progress("no ceiling record: searching, once, before anything is compared")
        search_ceilings(reduction, device, progress=progress, shard=shard)
    return config.ceilings_on_record()


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
    pooled = ceilings_in_force(reduction, device, progress=progress, shard=shard)
    return replace(reduction, ceilings=pooled,
                   ceilingsByTransfer=config.ceilings_by_transfer_on_record())


def search_ceilings(reduction: Reduction, device: torch.device,
                    progress=print, shard: str | None = None) -> dict:
    """Each family's ceiling, found on the selection transfers and kept for its
    derivations.

    A shared ceiling equalizes the coefficient and unequalizes the balance: the
    two objectives sit a factor of B_src apart, so one number puts adaptation at
    most of one objective and a tenth of the other. This equalizes where each
    method operates instead.

    It runs on `SEARCH_TRANSFERS` and never on the ones the verdict is read from.
    Choosing by outcome on the material the verdict uses would make the verdict
    read a decision it already made, and splitting the roles by transfer costs no
    bags — the evaluation role keeps all 36 and the resolution the campaign was
    sized for.

    At pilot scale this exercises the pipeline and settles nothing: the ramp runs
    on the fraction of training elapsed, so with three epochs it is saturated by
    the second and every ceiling is reached almost immediately. The search is the
    same program at both scales, which is the point; only its answer is worth
    reading at the scale the protocol declares.
    """
    # Its own scale, and not the caller's. The search is one experiment run once
    # at the scale the campaign runs at; borrowing the pilot's three epochs would
    # answer about a landscape nothing else trains in.
    reduction = replace(reduction, epochs=config.SEARCH_EPOCHS,
                        seeds=list(config.SEARCH_SEEDS))
    grid = list(config.CEILING_GRID)
    partial = shard_paths(shard)['partial']
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
            drawn = {code: bags.build(code, config.DATA_CACHE, seed)
                     for code in config.DOMAINS}
            for transfer in config.SEARCH_TRANSFERS:
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
            "transfers": [transfer_label(t) for t in config.SEARCH_TRANSFERS],
            # The neutral it is read against, so a searched value that lands on it
            # confirms the normalization by measurement rather than by argument.
            "neutral": config.RAMP_CEILING,
        }
    config.CEILINGS_RECORD.parent.mkdir(parents=True, exist_ok=True)
    config.CEILINGS_RECORD.write_text(json.dumps(found, indent=2), encoding="utf-8")
    # The scratch file goes only once the answer exists. Leaving it would let a
    # later relaunch resume from cells that already produced a finished record.
    partial.unlink(missing_ok=True)
    return found


def campaign(reduction: Reduction, device: torch.device,
             arms: list[str] | None = None, progress=print,
             shard: str | None = None) -> dict:
    """The whole grid: every arm, every transfer, every repetition.

    The search runs first and its answer is part of the bounds, not a detail:
    every family trains at the ceiling it found, every derivation inherits its
    family's, and the verdict is read only over the transfers the search never
    saw. Reading it over all six would let the ceiling's own selection material
    back into the number it was chosen to improve.
    """
    arm_ids = arms or [arm["id"] for arm in config.ARMS]
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    config.MODELS.mkdir(parents=True, exist_ok=True)

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
    on_record = config.ceilings_by_transfer_on_record()
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
    searched = search_record() or {}
    # The whole record travels into the reduction, here and not at the caller's
    # hand. The winner alone cannot say whether the grid leaned or the tie-break
    # chose, and a field a caller has to remember to fill is one that gets filled
    # on the run somebody was paying attention and left empty on the next.
    reduction = replace(reduction, ceilingSearch=searched)
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
    paths = shard_paths(shard)
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
        drawn = {code: bags.build(code, config.DATA_CACHE, seed) for code in config.DOMAINS}
        # Only the transfers the search never looked at. The other two funded the
        # ceiling and cannot also carry the verdict it was chosen to improve.
        for transfer in config.VERDICT_TRANSFERS:
            label = f"{transfer[0]}->{transfer[1]}"
            material = {"source": drawn[transfer[0]], "target": drawn[transfer[1]]}
            manifests[(label, seed)] = {"source": material["source"].manifest,
                                        "target": material["target"].manifest}
            for arm_id in arm_ids:
                run = run_one(arm_id, transfer, seed, reduction, device, material)
                state = run.pop("state")
                if state is not None:
                    stem = f"{arm_id}_{label.replace('->', '-')}_seed{seed}"
                    torch.save({k: v.cpu() for k, v in state.items()},
                               config.MODELS / f"{stem}.pt")
                records.write(json.dumps(run) + "\n")
                records.flush()
                cells.setdefault((label, arm_id), []).append(run)
                progress(f"  {arm_id:>2} {label} seed {seed}: "
                         f"target {run['targetAccuracy']:.3f}  "
                         f"source {run['sourceAccuracy']:.3f}  "
                         f"{run['seconds']:.1f}s")
        del drawn

    records.close()

    grid: dict[str, dict] = {}
    checkpoints: dict[str, list[str]] = {}
    for (label, arm_id), cell_runs in cells.items():
        grid.setdefault(label, {})[arm_id] = summarize(cell_runs)
        if arm_id in config.CHECKPOINTS:
            checkpoints[f"{arm_id} {label}"] = keep_median(
                cell_runs, arm_id, label, manifests, reduction)

    per_transfer = {label: judge(ladder_rows(cell, label)) for label, cell in grid.items()}
    summary = {
        "kind": "bounded",
        "reduction": asdict(reduction),
        "verdictsMeaningful": reduction.verdicts_meaningful,
        "grid": grid,
        "perTransfer": per_transfer,
        "panorama": paired_across_transfers(grid),
        "checkpoints": checkpoints,
        "tally": {label: tally(rows) for label, rows in per_transfer.items()},
    }
    (config.RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    # The same record, where the verification looks for it. A correct summary at a
    # path nobody opens protects nothing: a later session would find no record at all
    # and start over as though this had never run. `targetScale` beside `reduction` is
    # what lets it tell a pilot from a campaign instead of only a revision from a
    # stale one.
    product_results = config.PRODUCT / "Results" / "Probe_results.json"
    product_results.parent.mkdir(parents=True, exist_ok=True)
    product_results.write_text(json.dumps({
        "kind": summary["kind"],
        "revision": reduction.revision,
        "reduction": asdict(reduction),
        "targetScale": {"epochs": config.FULL_EPOCHS, "seeds": config.FULL_SEEDS},
        "verdictsMeaningful": reduction.verdicts_meaningful,
        "comparison": summary["panorama"],
        "detail": str((config.RESULTS / "summary.json").relative_to(config.REPOSITORY)),
        "figures": sorted(str(p.relative_to(config.REPOSITORY))
                          for p in config.RESULTS.rglob("*.pdf")),
    }, indent=2), encoding="utf-8")

    # Sealed last, once every file this shard writes actually exists. A run
    # that raises anywhere above never reaches this line, so its stamp stays
    # unsealed and therefore incomplete — never merge-eligible.
    seal_shard_stamp(shard)
    return summary


def run_pilot(epochs: int = config.EPOCHS, seeds: list[int] | None = None) -> dict:
    """One single-machine campaign, headless: exactly what
    `Benchmark_Phase1_Run.ipynb`'s own pilot cells run, callable with plain
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


def run_search(shard: str | None = None) -> dict:
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

    **There is deliberately no `epochs` dial**, and that is the difference
    from `run_pilot()` rather than an omission. `run_pilot()` offers one
    because a pilot is allowed to be cheap — it says so. The search is not:
    `search_ceilings()` sets `config.SEARCH_EPOCHS` and `config.SEARCH_SEEDS`
    on its own reduction and ignores whatever the caller passed, because a
    ceiling found at pilot scale settles nothing and would still be written
    to the same file the full-scale answer goes to. Any dial here would only
    look like it worked.

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
        seeds=list(config.SEARCH_SEEDS), epochs=config.SEARCH_EPOCHS,
        device=str(device), environment=environment())
    ceilings_in_force(reduction, device, shard=shard)
    # Read back from disk, never from what the search returned, for the reason
    # `ceilings_in_force` already gives: what the campaign will run at is what
    # the record says, and the record is the thing a later session reads.
    record = search_record()
    if record is None:
        raise SystemExit(
            "the search ran and left no record at "
            f"{config.CEILINGS_RECORD}. Nothing downstream can read a ceiling "
            "that was never written down, and a run that reports success "
            "without one would be claiming an answer it cannot show."
        )
    return record


def run_campaign_shard(shard: str | None = None,
                       seeds: list[int] | None = None) -> dict:
    """One shard of the campaign, headless, callable with JSON alone.

    `tools/distribute.py`'s `run_shard()` already does this and a remote worker
    cannot reach it: `tools/` is outside every declared clone path, and that
    module path-imports the forge's own packer, which does not exist inside a
    kernel. So the fan-out this repository's distribution declaration is built
    around — the seed axis split across machines — had no way to be asked for
    from a job. This is that way, and it is the same three lines, placed where a
    clone can see them.

    The seed list is the whole parameterisation, because the seed axis is what
    the declaration says may be split (`__benchmark__["distribution"]["axis"]`).
    Everything else is the campaign's own: epochs stay at `config.FULL_EPOCHS`
    for the reason `run_search` takes no dial either — a shard measured at pilot
    scale is not a cheaper shard, it is a different experiment, and it would be
    merged with the others as though it were one.

    `shard` names this call's own namespace and is passed through to
    `campaign()`, which hands it to `shard_paths()`. Without it two shards
    running at once write one `runs.jsonl` and one stamp between them, and the
    loser is a silent partial. Defaulting to `None` is the single-machine case,
    where there is nothing to collide with.

    Refuses nothing here that `campaign()` does not already refuse: it still
    demands a ceiling record, and a worker that clones only `src/` does not
    receive one. That is a property of what the job declares it clones, not
    something this function can paper over.
    """
    device = resolve_device()
    reduction = Reduction(
        seeds=list(seeds) if seeds is not None else list(config.FULL_SEEDS),
        epochs=config.FULL_EPOCHS, device=str(device), environment=environment())
    reduction = with_ceilings_in_force(reduction, device, shard=shard)
    return campaign(reduction, device, shard=shard)


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

    `shard` names this call's own shard namespace, the same parameter
    `distribute.run_shard()` takes, passed through to `shard_paths()` so a
    smoke rehearsal never collides with a real shard's files sharing the
    same directory. Defaults to `None` — one kernel container runs one
    smoke call, so there is nothing else in that container to collide with.

    `checkpoint`, off by default, writes exactly one `.pt` and its manifest
    through `campaign()`'s own path — `torch.save({k: v.cpu() for k, v in
    state.items()}, ...)` beside a manifest `keep_median()` writes with
    `bags.write_manifest()` — rather than a bespoke save. Phase 2's
    `latent.available()`/`latent.load()` read `config.MODELS` in exactly
    that shape, so a checkpoint proven any other way would not prove the
    path a real campaign actually uses. `config.CHECKPOINTS[SMOKE_ARM]`
    asks for three per cell; with exactly one seed here, `median_seeds()`
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

    paths = shard_paths(shard)
    paths["runs"].parent.mkdir(parents=True, exist_ok=True)
    write_shard_stamp(shard, reduction)

    drawn = {code: bags.build(code, config.DATA_CACHE, seed)
             for code in {transfer[0], transfer[1]}}
    material = {"source": drawn[transfer[0]], "target": drawn[transfer[1]]}
    run = run_one(SMOKE_ARM, transfer, seed, reduction, device, material)
    state = run.pop("state", None)

    if checkpoint and state is not None:
        config.MODELS.mkdir(parents=True, exist_ok=True)
        label = run["transfer"]
        stem = f"{SMOKE_ARM}_{label.replace('->', '-')}_seed{seed}"
        torch.save({k: v.cpu() for k, v in state.items()},
                   config.MODELS / f"{stem}.pt")
        manifests = {(label, seed): {"source": material["source"].manifest,
                                     "target": material["target"].manifest}}
        keep_median([run], SMOKE_ARM, label, manifests, reduction)

    with paths["runs"].open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(run) + "\n")

    seal_shard_stamp(shard)
    return {
        "arm": SMOKE_ARM,
        "transfer": run["transfer"],
        "seed": seed,
        "targetAccuracy": run["targetAccuracy"],
        "sourceAccuracy": run["sourceAccuracy"],
        "seconds": run["seconds"],
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


def render_transfer(summary: dict, transfer: str) -> str:
    """One transfer's ladder, in the same table shape the sweep uses."""
    rows = [row for row in summary["perTransfer"][transfer]
            if row["metric"] in ("targetAccuracy", "sourceAccuracy")]
    return f"[{transfer}]\n" + render(rows, {
        "setting": summary["reduction"]["setting"],
        "backbone": summary["reduction"]["backbone"],
        "dataset": transfer,
        "seeds": len(summary["reduction"]["seeds"]),
        "revision": summary["reduction"]["revision"],
    })

