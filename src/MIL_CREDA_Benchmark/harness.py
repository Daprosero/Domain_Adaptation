"""The measuring helpers the toyset shares with the retired digit benchmark.

What each arm computes lives in `wiring`; which material it sees lives in `bags`;
every number that defines the experiment lives in `config`. The training loops,
campaigns, searches and shards that drove the image benchmark are gone; what
stays is what `toy_run` calls: the device, the coefficient schedule, the
class-balanced batch draw, the two accuracy entry points, a role's pool, and
the mean-and-dispersion summary.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from CREDA.schedules import creda_ramp
from MIL_CREDA_Benchmark import bags, config, wiring
from MIL_CREDA_Benchmark.schedules import milcreda_ramp


# ------------------------------------------------------------------ environment

def resolve_device() -> torch.device:
    """The same preference order `CREDA.training_pipeline` uses."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def balanced_batches(targets: list[int], steps: int, generator: torch.Generator,
                     per_class: int = 1):
    """One bag of every class per step, so no class is ever missing.

    `total_correspondence` refuses to invent a value for a class with no source
    bag, and with ten bags drawn at random out of ten classes one goes missing
    often enough to kill an epoch. Each class keeps its own shuffled queue and
    refills when it runs out, so the draw stays stratified across the epoch and
    the arms all see the same shape of batch.

    `per_class` bags of every class per step (default one, which is the draw
    exactly as it was). A queue refills only when empty, so a step repeats a
    bag only if a class's count is not a multiple of `per_class`.
    """
    queues: dict[int, list[int]] = {c: [] for c in set(targets)}
    by_class: dict[int, list[int]] = {c: [] for c in set(targets)}
    for position, label in enumerate(targets):
        by_class[label].append(position)

    for _ in range(steps):
        batch = []
        for class_id in sorted(by_class):
            for _ in range(per_class):
                if not queues[class_id]:
                    pool = by_class[class_id]
                    order = torch.randperm(len(pool), generator=generator).tolist()
                    queues[class_id] = [pool[i] for i in order]
                batch.append(queues[class_id].pop())
        yield batch


# ------------------------------------------------------------------ measuring

@torch.no_grad()
def accuracy(model: nn.Module, dataset, device: torch.device) -> float:
    """Bag accuracy through the SOURCE encoder (`theta_s`): the unit both
    families decide in. Signature and behavior unchanged since Eq. (13) split
    the encoder -- `model(x)` is `Arm.forward`, which stays the source path.
    """
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


@torch.no_grad()
def target_accuracy(model: nn.Module, dataset, device: torch.device) -> float:
    """Bag accuracy through the TARGET encoder (`theta_t`) -- `accuracy`'s
    mirror, calling `model.score_target(x)` in place of `model(x)`.

    On the floor (`B`), `Arm.score_target` falls back to the same source
    encoder `forward` uses, so this reads identically to `accuracy` there;
    doing nothing here (scoring target material through `theta_s`, as
    `accuracy` alone would) is the mixed-up-domain failure this method exists
    to remove -- it would make the 1.99x parameter count buy nothing at
    inference and make a degenerate target encoder invisible.
    """
    model.eval()
    correct = total = 0
    for start in range(0, len(dataset), config.BAGS_PER_STEP):
        items = [dataset[i] for i in range(start, min(start + config.BAGS_PER_STEP,
                                                     len(dataset)))]
        x = torch.stack([item[0] for item in items]).to(device)
        y = torch.tensor([item[1] for item in items], device=device)
        correct += int((model.score_target(x).argmax(dim=1) == y).sum())
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
