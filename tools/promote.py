"""Which trained checkpoints the centre keeps, put where phase two reads them.

`shards.promote_candidates()` already holds the rule and holds it correctly: the
median is taken over *every* measurement the campaign made, and only the choice
is then restricted to the checkpoints that actually came back. Nothing called it.
The rule sat in `src/` with one test beside it while the checkpoints sat inside
ten returned artifacts and `MIL-CREDA/Models/Benchmark/` still held whatever an
earlier, smaller run had left there. Phase two reads that directory and would
have measured the earlier run's representations while the summary beside it
described the campaign — every check green, the two describing different runs.

This module is the wiring, and nothing more: it enumerates what came back, hands
that to `promote_candidates()`, and copies what it chose. It re-implements no part
of the selection, because a second copy of a rule is a rule that drifts.

**Enumerated from the disk, and identified from each checkpoint's own manifest.**
A `.pt` says which cell and seed it belongs to only through the manifest
`harness.keep_median()` wrote beside it. Reconstructing that from the filename
would be this module inventing a second naming contract, and a file whose name
and manifest disagreed would be resolved silently in favour of the guess. A
checkpoint with no manifest is reported as unidentifiable rather than parsed.

**What is already in place is moved, never removed.** The checkpoints being
replaced are the output of a real run; they are not this tool's to delete. They
move aside once, and a second promotion refuses rather than moving a second set
on top of the first.

**Nothing arriving is a refusal, not an empty promotion.** Promoting nothing
would move the existing checkpoints aside and copy none in, leaving phase two to
report that there are no checkpoints at all — a wiping dressed as a no-op.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from MIL_CREDA_Benchmark import config, latent, shards  # noqa: E402

#: Where the returned artifacts are unpacked, one directory per shard. Searched
#: recursively rather than by a fixed depth: how a backend lays out what it
#: returns is that backend's business, and a path shape hardcoded here would
#: quietly find nothing the day it changes.
ARTIFACTS = config.PRODUCT / ".remote-execution" / "campaign"

#: Where checkpoints already in place are moved before new ones land.
SUPERSEDED = config.MODELS.with_name(config.MODELS.name + ".superseded")

#: What this tool leaves beside the checkpoints it promoted: which seeds each
#: cell wanted, which of those existed, and what it fell short by. Without it
#: the directory says what was chosen and never why, and "the median of the
#: campaign" becomes a claim resting on nobody's record.
RECORD = "PROMOTION.json"

WEIGHTS_SUFFIX = ".pt"
MANIFEST_SUFFIX = ".manifest.json"


def read_runs(path: Path | None = None) -> list[dict]:
    """Every measurement the campaign made, which is what the median is over."""
    source = path if path is not None else config.RESULTS / "runs.jsonl"
    if not source.exists():
        raise SystemExit(f"no runs to take a median over: {source} does not exist")
    return [json.loads(line) for line in
            source.read_text(encoding="utf-8").splitlines() if line.strip()]


def arrived(root: Path | None = None) -> dict:
    """What came back, keyed `(arm, transfer, seed)`, read off each manifest.

    `duplicates` is reported rather than resolved: one seed of one cell is run
    on exactly one shard, so the same key arriving twice means two shards
    disagree about who ran it, and picking one of them would hide that.
    """
    home = root if root is not None else ARTIFACTS
    if not home.is_dir():
        raise SystemExit(f"no returned artifacts to promote from: {home}")

    found: dict = {}
    unidentifiable: list[str] = []
    duplicates: list[str] = []
    for weights in sorted(home.rglob("*" + WEIGHTS_SUFFIX)):
        manifest = weights.with_name(
            weights.name[: -len(WEIGHTS_SUFFIX)] + MANIFEST_SUFFIX)
        if not manifest.exists():
            unidentifiable.append(str(weights))
            continue
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        try:
            key = (entries["arm"], entries["transfer"], entries["seed"])
        except KeyError:
            unidentifiable.append(str(weights))
            continue
        if key in found:
            duplicates.append(f"{key[0]}|{key[1]} seed {key[2]}")
            continue
        found[key] = (weights, manifest)
    return {"found": found, "unidentifiable": unidentifiable,
            "duplicates": sorted(set(duplicates))}


def paired(chosen: dict, found: dict, floor_of: dict | None = None) -> dict:
    """Extend each floor's promotion to cover the seeds its dependants chose.

    `latent.against_floor()` reads a **paired** difference: an arm at one seed
    against its own floor at that same seed, because a distance in an embedding
    has no absolute meaning and only its movement carries anything. The pairing
    is a precondition of the comparison, and nothing in the selection produced it
    — `promote_candidates()` takes each cell's median on its own, so an arm and
    its floor share a promoted seed only by accident. Measured on the first
    campaign: of 48 `(arm, floor, transfer)` pairs, 27 shared no seed at all and
    only 4 shared two. The comparison did not fail; it quietly had almost nothing
    left to compare.

    **Every arm keeps its own median, and this only ever adds.** The alternative
    — choosing an arm's seeds to match its floor's — would promote a checkpoint
    that is not that arm's median, which is the one property the selection exists
    to guarantee. So the adapted arms are untouched and the floors carry the
    union of what their dependants need, in `paired`, kept apart from `chosen`
    rather than merged into it: a seed promoted to make somebody else's
    comparison possible is not part of this cell's median and must not be read as
    its typical artefact.

    A floor checkpoint that never came back is named in `unpairable` instead of
    dropped, because the comparison it blocks is one the record would otherwise
    show as simply absent.
    """
    relations = floor_of if floor_of is not None else dict(config.FLOOR_OF)
    extra: dict[str, set] = {}
    unpairable: list[str] = []
    for cell, entry in chosen.items():
        arm, transfer = cell.split("|", 1)
        floor = relations.get(arm)
        if not floor:
            continue
        for seed in entry["chosen"]:
            if (floor, transfer, seed) not in found:
                unpairable.append(
                    f"{arm}|{transfer} seed {seed}: its floor {floor} never arrived")
                continue
            extra.setdefault(f"{floor}|{transfer}", set()).add(seed)
    out = {}
    for cell, entry in chosen.items():
        added = sorted(extra.get(cell, set()) - set(entry["chosen"]))
        out[cell] = {**entry, "paired": added}
    return {"chosen": out, "unpairable": sorted(unpairable)}


def displayed(chosen: dict, found: dict, runs: list[dict],
              seed: int | None = None) -> dict:
    """Add the one seed every comparative panel is drawn from, to every cell.

    `latent.display_seed()` picks a single seed for the whole grid, and its own
    docstring says why it must be single: panels taken from different seeds would
    differ in the method *and* in the draw, and would not even share their bags,
    so "this subject sits beside the wrong one here and the right one there"
    would stop being a sentence anyone could say. Drawing each panel at its own
    median is therefore not an option — it destroys the comparison the figure
    exists to make.

    Nothing arranged that seed to exist. It is chosen over the measurements, and
    the checkpoints are chosen per cell by median, so the two agree only by
    accident: on the first paired campaign the display seed was present in **1 of
    21** grid cells, and the figure came out with one panel drawn and the rest
    blank canvas. A PNG was emitted, so nothing downstream noticed. No seed at all
    covered every cell — the best reached 15 of 60.

    Held apart from `chosen` and `paired` for the same reason they are held apart
    from each other: a checkpoint promoted so a figure can be drawn is not this
    cell's median and must never enter a marginal average.

    A cell whose display seed never came back is named in `undisplayable`. Its
    panel will be blank, and that is worth saying before the figure is read
    rather than after somebody notices the gap.
    """
    wanted = seed if seed is not None else latent.display_seed(runs)
    out, undisplayable = {}, []
    for cell, entry in chosen.items():
        arm, transfer = cell.split("|", 1)
        already = set(entry["chosen"]) | set(entry.get("paired") or [])
        if wanted in already:
            out[cell] = {**entry, "display": []}
            continue
        if (arm, transfer, wanted) not in found:
            undisplayable.append(f"{cell} seed {wanted}")
            out[cell] = {**entry, "display": []}
            continue
        out[cell] = {**entry, "display": [wanted]}
    return {"chosen": out, "displaySeed": wanted,
            "undisplayable": sorted(undisplayable)}


def promote(runs: list[dict] | None = None, root: Path | None = None,
            destination: Path | None = None,
            superseded: Path | None = None,
            floor_of: dict | None = None) -> dict:
    """Choose over every measurement, copy what exists, and say what is short."""
    measurements = runs if runs is not None else read_runs()
    if not measurements:
        raise SystemExit("no measurements: the median has nothing to be taken over")

    here = arrived(root)
    found = here["found"]
    if not found:
        raise SystemExit(
            f"no checkpoints came back under {root if root is not None else ARTIFACTS}; "
            "promoting nothing would empty the destination rather than fill it")

    chosen = shards.promote_candidates(measurements, set(found))
    pairing = paired(chosen, found, floor_of)
    display = displayed(pairing["chosen"], found, measurements)
    chosen = display["chosen"]

    into = destination if destination is not None else config.MODELS
    aside = superseded if superseded is not None else SUPERSEDED
    moved = None
    if into.is_dir() and any(into.iterdir()):
        if aside.exists():
            raise SystemExit(
                f"{aside} already holds a superseded set; move or remove it before "
                "promoting again, so one promotion never buries another")
        shutil.move(str(into), str(aside))
        moved = str(aside)
    into.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for cell, entry in sorted(chosen.items()):
        arm, transfer = cell.split("|", 1)
        for seed in (entry["chosen"] + entry.get("paired", [])
                     + entry.get("display", [])):
            weights, manifest = found[(arm, transfer, seed)]
            for source in (weights, manifest):
                shutil.copy2(source, into / source.name)
                copied.append(source.name)

    short = {cell: entry for cell, entry in sorted(chosen.items())
             if entry["shortfall"]}
    record = {
        "measurements": len(measurements),
        "candidates": len(found),
        "cells": len(chosen),
        "chosen": chosen,
        "shortfall": short,
        "unpairable": pairing["unpairable"],
        "displaySeed": display["displaySeed"],
        "undisplayable": display["undisplayable"],
        "unidentifiable": here["unidentifiable"],
        "duplicates": here["duplicates"],
        "superseded": moved,
    }
    (into / RECORD).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {**record, "copied": sorted(copied), "destination": str(into)}


def main() -> int:
    report = promote()
    print(f"promoted into {report['destination']}")
    print(f"  measurements the median was taken over: {report['measurements']}")
    print(f"  checkpoints that came back:             {report['candidates']}")
    print(f"  cells:                                  {report['cells']}")
    print(f"  files copied:                           {len(report['copied'])}")
    if report["superseded"]:
        print(f"  what was already there moved to:        {report['superseded']}")
    for name, entries in (("short of what the median wanted", report["shortfall"]),
                          ("could not be paired with a floor", report["unpairable"]),
                          ("no checkpoint at the display seed", report["undisplayable"]),
                          ("unidentifiable (no manifest)", report["unidentifiable"]),
                          ("claimed by more than one shard", report["duplicates"])):
        if entries:
            print(f"  {name}: {len(entries)}")
            for entry in (entries if isinstance(entries, list) else entries.items()):
                print(f"    {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
