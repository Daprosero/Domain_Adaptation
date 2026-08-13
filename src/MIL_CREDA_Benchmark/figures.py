"""What the trajectories look like, drawn from the record phase one already wrote.

The tables in the notebook report the minimum, the maximum and the width of each
adaptation term. That answers the question. It is not what a reader looks at, and
a claim about scale — that a quantity stays inside its bounds, that it behaves the
same whichever pair of domains it measures — is seen before it is checked.

Nothing here re-runs anything. Every point comes from `runs.jsonl`, which the
campaign wrote step by step, so the figures describe exactly the runs the tables do.

Every curve is the **median across seeds** with an interquartile band, never one
run's trajectory and never the seeds concatenated. A single trajectory cannot show
whether the shape is the method's or the draw's, and gluing the repetitions end to
end would draw thirty runs as one long one.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MIL_CREDA_Benchmark import config


def load_curves(path: Path | None = None) -> dict:
    """{transfer: {arm: [curve_of_seed_0, curve_of_seed_1, ...]}}.

    Grouped by repetition and not flattened. Concatenating them would make a
    figure of thirty seeds indistinguishable from a figure of one run that took
    thirty times as long.
    """
    path = path or (config.RESULTS / "runs.jsonl")
    curves: dict[str, dict[str, list[list[dict]]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            run = json.loads(line)
            curves.setdefault(run["transfer"], {}).setdefault(run["arm"], []).append(
                run["curve"])
    return curves


def _quantiles(values: list[float]) -> tuple[float, float, float]:
    """Median and the two quartiles, by plain interpolation on the sorted values."""
    ordered = sorted(values)
    n = len(ordered)

    def at(fraction: float) -> float:
        if n == 1:
            return ordered[0]
        position = fraction * (n - 1)
        low = int(position)
        high = min(low + 1, n - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return at(0.25), at(0.5), at(0.75)


def band(repetitions: list[list[dict]], key: str) -> tuple[list[float], list[float], list[float]]:
    """The median trajectory of `key` across repetitions, and its interquartile band.

    Repetitions of unequal length are truncated to the shortest, which only happens
    when a run stopped early; extending the short ones would invent steps.
    """
    if not repetitions:
        return [], [], []
    length = min(len(curve) for curve in repetitions)
    low, mid, high = [], [], []
    for step in range(length):
        q1, q2, q3 = _quantiles([curve[step][key] for curve in repetitions])
        low.append(q1)
        mid.append(q2)
        high.append(q3)
    return low, mid, high


def _caption() -> str:
    seeds = len(config.SEEDS)
    stamp = "" if seeds >= len(config.FULL_SEEDS) else "  ·  PILOT, not a result"
    return (f"{config.BAGS_PER_DOMAIN}x{config.INSTANCES_PER_BAG} bags · "
            f"{config.EPOCHS} epochs · {seeds} seed(s), median with interquartile band · "
            f"lambda = ramp(delta {config.RAMP_DELTA}) x {config.LAMBDA_CONST:g} · "
            f"{config.REVISION}{stamp}")


def _panelled(path: Path, arms: tuple[str, ...], key: str, ylabel: str, title: str,
              shade_unit: bool = False, ylim: tuple[float, float] | None = None) -> Path:
    """One panel per transfer, one median-with-band per arm. The shape all three share."""
    curves = load_curves()
    transfers = list(curves)
    columns = min(3, len(transfers)) or 1
    rows = -(-len(transfers) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.2 * rows),
                                squeeze=False, sharey=True)

    for index, transfer in enumerate(transfers):
        axis = axes[index // columns][index % columns]
        if shade_unit:
            axis.axhspan(0.0, 1.0, color="0.88", zorder=0,
                         label="[0, 1]" if index == 0 else None)
            axis.axhline(0.0, color="0.6", linewidth=0.8, zorder=1)
        for arm in arms:
            repetitions = curves[transfer].get(arm)
            if not repetitions:
                continue
            low, mid, high = band(repetitions, key)
            steps = range(len(mid))
            line, = axis.plot(steps, mid, linewidth=1.3, zorder=3,
                              label=config.NAME_OF[arm] if index == 0 else None)
            if len(repetitions) > 1:
                axis.fill_between(steps, low, high, alpha=0.18, zorder=2,
                                  color=line.get_color(), linewidth=0)
        axis.set_title(transfer, fontsize=10)
        axis.set_xlabel("optimizer step", fontsize=8)
        axis.tick_params(labelsize=8)
        if ylim:
            axis.set_ylim(*ylim)
    axes[0][0].set_ylabel(ylabel, fontsize=9)

    for spare in range(len(transfers), rows * columns):
        axes[spare // columns][spare % columns].axis("off")

    figure.legend(loc="lower center", ncol=6, fontsize=8, frameon=False,
                  bbox_to_anchor=(0.5, 0.005))
    figure.suptitle(title, fontsize=11)
    figure.text(0.5, 0.055, _caption(), ha="center", fontsize=7, color="0.35")
    figure.tight_layout(rect=(0, 0.10, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def adaptation_curves(path: Path,
                      arms: tuple[str, ...] = ("C", "D", "E", "F", "G", "SK")) -> Path:
    """Each adaptation term across training, one panel per transfer.

    The shaded band is [0, 1]. Section 5 normalizes MIL-CREDA's terms onto exactly
    that interval, and the prior work's score has no such bound — so whether a curve
    stays inside the band, and whether it occupies the same part of it from one
    transfer to the next, is the claim itself rather than an illustration of it.
    """
    return _panelled(path, arms, "adaptation", "adaptation term",
                     "Where each adaptation term lives, transfer by transfer",
                     shade_unit=True)


def supervised_curves(path: Path,
                      arms: tuple[str, ...] = ("A", "D", "B", "G")) -> Path:
    """The supervised term beside the adaptation one.

    This is where an adaptation term that destabilizes the fit shows up. Reading the
    adaptation curve alone would call a term well-behaved while the classification it
    shares an objective with comes apart underneath it.
    """
    return _panelled(path, arms, "supervised", "supervised term",
                     "Did the adaptation term destabilize the fit?")


def contribution_curves(path: Path,
                        arms: tuple[str, ...] = ("C", "D", "E", "F", "G", "SK")) -> Path:
    """What share of the objective each declared term actually commands.

    Without this panel, "the term had no effect" and "the term had no weight" are
    the same picture. The coefficient is fixed at `LAMBDA_CONST` for every arm, and
    fixing the coefficient does not fix the share: a term whose magnitude differs
    by an order of magnitude between arms is a difference nobody declared, and a
    rung that ignores it credits the mechanism with what the scale did.
    """
    return _panelled(path, arms, "contribution", "lambda x adaptation",
                     "How much of the objective each adaptation term commands")
