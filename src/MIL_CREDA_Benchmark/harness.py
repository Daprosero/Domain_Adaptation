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

import json
import math
import platform
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from CREDA.schedules import creda_ramp
from MIL_CREDA_Benchmark import bags, config, wiring
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
    }


# -------------------------------------------------------------------- schedules

def ramp(epoch: int, epochs: int, delta: float = config.RAMP_DELTA,
         ceiling: float = config.RAMP_CEILING) -> float:
    """The adaptation coefficient, from CREDA's own schedule module.

    This used to be a second copy of the formula, because `training_pipeline`
    cannot be imported for it — scikit-image, timm, pandas and matplotlib all
    load at its module level. Two copies across the two arms of a comparison is
    the fork this benchmark exists not to have, so the schedule moved to
    `CREDA.schedules` and both families read it from there.
    """
    return creda_ramp(epoch, epochs, delta=delta, ceiling=ceiling)


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
    evalBags: int = config.EVAL_BAGS
    epochs: int = config.EPOCHS
    seeds: list[int] = field(default_factory=lambda: list(config.SEEDS))
    rampCeiling: float = config.RAMP_CEILING
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


def run_one(arm_id: str, transfer: tuple[str, str], seed: int,
            reduction: Reduction, device: torch.device,
            material: dict) -> dict:
    """One arm, one transfer, one repetition, end to end."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 9973)

    source, target = material["source"], material["target"]
    source_train, source_eval = bags.roles(source)
    target_train, target_eval = bags.roles(target)

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
        coefficient = ramp(epoch, reduction.epochs)
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
            "sourceAccuracy": accuracy(model, source_eval, device),
            "targetAccuracy": accuracy(model, target_eval, device),
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


def ladder_rows(cell: dict, transfer: str) -> list[dict]:
    """One row per rung and dimension, with the arm on the right as `new`."""
    rows = []
    for left, right, reading in config.LADDER:
        if left not in cell or right not in cell:
            continue
        for dimension, better in config.DIMENSIONS.items():
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
    ordered = sorted(cell_runs, key=lambda r: r["targetAccuracy"])
    middle = len(ordered) // 2
    span = min(config.CHECKPOINTS[arm_id], len(ordered))
    start = max(0, min(middle - span // 2, len(ordered) - span))
    keep = {run["seed"] for run in ordered[start:start + span]}

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


def campaign(reduction: Reduction, device: torch.device,
             arms: list[str] | None = None, progress=print) -> dict:
    """The whole grid: every arm, every transfer, every repetition."""
    arm_ids = arms or [arm["id"] for arm in config.ARMS]
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    config.MODELS.mkdir(parents=True, exist_ok=True)
    records = (config.RESULTS / "runs.jsonl").open("w", encoding="utf-8")

    # Seeds sit outermost so a repetition's material — the stratified draw, the
    # composition of the bags and the split — is built once for all three domains
    # and reused across that seed's transfers. A repetition that reused another
    # seed's bags would vary the initialization and nothing else.
    cells: dict[tuple[str, str], list[dict]] = {}
    manifests: dict[tuple[str, str], dict] = {}
    for seed in reduction.seeds:
        drawn = {code: bags.build(code, config.DATA_CACHE, seed) for code in config.DOMAINS}
        for transfer in config.TRANSFERS:
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
    return summary


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

