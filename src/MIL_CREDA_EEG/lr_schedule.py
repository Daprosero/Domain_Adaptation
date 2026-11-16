"""The annealed phase-2 learning rate shared by every method of the criteria pool.

lr_p = lr2 * (1 + ALPHA * p) ** -BETA, p the fraction of the optimizer's steps already taken: the
schedule of Ganin et al. (DANN) that T-GARNet trains with (eta_0 1e-3, alpha 10, beta 0.75). It is
applied from outside every trainer through a global optimizer step pre-hook, so the eight phase-2
trainers keep their own code: an optimizer is scheduled when its learning rate at its first step
equals the trial's lr2 (a method's inner solvers, such as COT's dual step, run at other rates and
are left alone). The total step count of each scheduled optimizer comes from `count_steps` over a
short dry run of the same trial, scaled to the full epoch budget.
"""
from __future__ import annotations

import contextlib
import math
import weakref

from torch.optim.optimizer import register_optimizer_step_pre_hook

ALPHA, BETA = 10.0, 0.75


class NotScheduled(RuntimeError):
    """A run under `annealed` stepped no optimizer at the trial's lr2."""


def factor(p: float) -> float:
    return (1.0 + ALPHA * min(1.0, max(0.0, p))) ** -BETA


def _is_lr2(opt, lr2: float) -> bool:
    return math.isclose(float(opt.param_groups[0]["lr"]), float(lr2), rel_tol=1e-12, abs_tol=0.0)


@contextlib.contextmanager
def count_steps(lr2: float):
    """Yields a list that ends up holding, in creation order, the steps of every optimizer whose
    first step ran at `lr2`."""
    counts: list[int] = []
    slot: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def hook(opt, args, kwargs):
        if opt not in slot:
            slot[opt] = len(counts) if _is_lr2(opt, lr2) else None
            if slot[opt] is not None:
                counts.append(0)
        if slot[opt] is not None:
            counts[slot[opt]] += 1

    handle = register_optimizer_step_pre_hook(hook)
    try:
        yield counts
    finally:
        handle.remove()


@contextlib.contextmanager
def annealed(lr2: float, totals: list[int]):
    """Every optimizer whose first step runs at `lr2` follows the schedule over `totals[i]` steps,
    i its creation order among them (the last total holds for any later one)."""
    if not totals or min(totals) < 1:
        raise ValueError("annealed needs a positive step total per scheduled optimizer")
    state: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
    order = [0]

    def hook(opt, args, kwargs):
        if opt not in state:
            if _is_lr2(opt, lr2):
                state[opt] = {"n": 0, "total": totals[min(order[0], len(totals) - 1)],
                              "base": [float(g["lr"]) for g in opt.param_groups]}
                order[0] += 1
            else:
                state[opt] = None
        s = state[opt]
        if s is None:
            return
        f = factor(s["n"] / s["total"])
        for g, base in zip(opt.param_groups, s["base"]):
            g["lr"] = base * f
        s["n"] += 1

    handle = register_optimizer_step_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()
    if order[0] == 0:
        raise NotScheduled(f"no optimizer stepped at lr2 = {lr2!r}; the schedule never applied")
