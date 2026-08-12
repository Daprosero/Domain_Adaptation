"""What the trajectories look like, drawn from the record phase one already wrote.

The tables in the notebook report the minimum, the maximum and the width of each
adaptation term. That answers the question. It is not what a reader looks at, and
a claim about scale — that a quantity stays inside its bounds, that it behaves the
same whichever pair of domains it measures — is seen before it is checked.

Nothing here re-runs anything. Every point comes from `runs.jsonl`, which the
campaign wrote step by step, so the figures describe exactly the runs the tables do.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MIL_CREDA_Benchmark import config


def load_curves(path: Path | None = None) -> dict:
    """Every logged step, grouped by transfer and arm."""
    path = path or (config.RESULTS / "runs.jsonl")
    curves: dict[str, dict[str, list[dict]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            run = json.loads(line)
            curves.setdefault(run["transfer"], {}).setdefault(run["arm"], []).extend(
                run["curve"])
    return curves


def _caption() -> str:
    return (f"{config.BAGS_PER_DOMAIN}x{config.INSTANCES_PER_BAG} bags · "
            f"{config.EPOCHS} epochs · {len(config.SEEDS)} seed(s) · "
            f"lambda = ramp(delta {config.RAMP_DELTA}) x {config.LAMBDA_CONST:g} · "
            f"{config.REVISION}")


def adaptation_curves(path: Path, arms: tuple[str, ...] = ("C", "D", "E", "F", "G")) -> Path:
    """Each adaptation term across training, one panel per transfer.

    The shaded band is [0, 1]. Section 5 normalizes MIL-CREDA's terms onto exactly
    that interval, and the prior work's score has no such bound — so whether a curve
    stays inside the band, and whether it occupies the same part of it from one
    transfer to the next, is the claim itself rather than an illustration of it.
    """
    curves = load_curves()
    transfers = list(curves)
    columns = min(3, len(transfers)) or 1
    rows = -(-len(transfers) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.2 * rows),
                                squeeze=False, sharey=True)

    for index, transfer in enumerate(transfers):
        axis = axes[index // columns][index % columns]
        axis.axhspan(0.0, 1.0, color="0.88", zorder=0,
                     label="[0, 1]" if index == 0 else None)
        axis.axhline(0.0, color="0.6", linewidth=0.8, zorder=1)
        for arm in arms:
            points = curves[transfer].get(arm)
            if not points:
                continue
            axis.plot([p["adaptation"] for p in points], linewidth=1.2,
                      label=config.ARMS_BY_ID[arm]["label"] if index == 0 else None,
                      zorder=2)
        axis.set_title(transfer, fontsize=10)
        axis.set_xlabel("optimizer step", fontsize=8)
        axis.tick_params(labelsize=8)
    axes[0][0].set_ylabel("adaptation term", fontsize=9)

    for spare in range(len(transfers), rows * columns):
        axes[spare // columns][spare % columns].axis("off")

    figure.legend(loc="lower center", ncol=6, fontsize=8, frameon=False,
                  bbox_to_anchor=(0.5, 0.005))
    figure.suptitle("Where each adaptation term lives, transfer by transfer", fontsize=11)
    figure.text(0.5, 0.055, _caption(), ha="center", fontsize=7, color="0.35")
    figure.tight_layout(rect=(0, 0.10, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def supervised_curves(path: Path, arms: tuple[str, ...] = ("A", "B", "D", "G")) -> Path:
    """The supervised term beside the adaptation one.

    This is where an adaptation term that destabilizes the fit shows up. Reading the
    adaptation curve alone would call a term well-behaved while the classification it
    shares an objective with comes apart underneath it.
    """
    curves = load_curves()
    transfers = list(curves)
    columns = min(3, len(transfers)) or 1
    rows = -(-len(transfers) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.2 * rows),
                                squeeze=False, sharey=True)

    for index, transfer in enumerate(transfers):
        axis = axes[index // columns][index % columns]
        for arm in arms:
            points = curves[transfer].get(arm)
            if not points:
                continue
            axis.plot([p["supervised"] for p in points], linewidth=1.2,
                      label=config.ARMS_BY_ID[arm]["label"] if index == 0 else None)
        axis.set_title(transfer, fontsize=10)
        axis.set_xlabel("optimizer step", fontsize=8)
        axis.tick_params(labelsize=8)
    axes[0][0].set_ylabel("supervised term", fontsize=9)

    for spare in range(len(transfers), rows * columns):
        axes[spare // columns][spare % columns].axis("off")

    figure.legend(loc="lower center", ncol=4, fontsize=8, frameon=False,
                  bbox_to_anchor=(0.5, 0.005))
    figure.suptitle("Did the adaptation term destabilize the fit?", fontsize=11)
    figure.text(0.5, 0.055, _caption(), ha="center", fontsize=7, color="0.35")
    figure.tight_layout(rect=(0, 0.10, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
