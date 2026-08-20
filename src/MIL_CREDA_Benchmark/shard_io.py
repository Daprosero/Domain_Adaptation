#!/usr/bin/env python3
"""Reads shard results from disk — generic across any paper or benchmark repository.

Assumes only the ambient contract every shard directory keeps: a `shard.json`
stamp file and an optional `runs.jsonl` log of newline-delimited JSON records,
one per completed measurement. Nothing here knows what a "run" measures, or
how those records group into a result cell for reporting — that grouping is
this repository's own vocabulary, keyed on two experiment dimensions the
originating module still owns, and generalizing what a cell is stayed there
deliberately rather than being attempted here.

Only the file-reading half — the half that assumes nothing beyond the
`shard.json` + `runs.jsonl` contract — belongs to a forge meant to serve more
than one paper. A repository that wants the grouped-cell behavior imports
this module and layers its own grouping on top, the way the originating
repository's own re-export shim does.

Run with any Python 3.10+ (stdlib-only):
    python3 -m unittest tests.test_remote_execution
"""
from __future__ import annotations

import json
from pathlib import Path


def read_shards(root: Path) -> list[dict]:
    """Every shard found under `root`, with its stamp and its raw runs.

    Enumerated from disk, never assumed. A directory holding no `shard.json`
    is skipped rather than reported, because the caller needs to be able to
    tell "never came back" apart from "came back and reported nothing" — and
    only the first of those is silence at this layer. A `shard.json` with no
    matching `runs.jsonl` yet still reports, with an empty run list, since a
    stamp can legitimately land before its runs finish streaming in.
    """
    if not root.is_dir():
        return []
    found = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        stamp_path = folder / "shard.json"
        runs_path = folder / "runs.jsonl"
        if not stamp_path.exists():
            continue
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        runs = [json.loads(line) for line in
                runs_path.read_text(encoding="utf-8").splitlines()
                if line.strip()] if runs_path.exists() else []
        found.append({"shard": folder.name, "stamp": stamp, "runs": runs})
    return found


def disagreements(shards: list[dict], fields: list[str]) -> list[dict]:
    """Where the shards disagree on fields they were declared to match on.

    Reported per field with every value seen and which shards produced it,
    because "they disagree" is not actionable on its own — a caller needs to
    know which values, and where each one came from, to decide what to do
    about it.
    """
    found = []
    for field in fields:
        seen: dict[str, list[str]] = {}
        for entry in shards:
            seen.setdefault(json.dumps(entry["stamp"].get(field), sort_keys=True),
                            []).append(entry["shard"])
        if len(seen) > 1:
            found.append({"field": field,
                          "values": {value: names for value, names in seen.items()}})
    return found


def completeness(stamp: dict, required: list[str]) -> dict:
    """Which of the caller's declared fields this one stamp actually has.

    Same argument shape as `disagreements(shards, fields)` above — a data
    structure and a caller-supplied list of field designators — and, like
    that function, this one names no field of its own. Each entry in
    `required` is a dot-separated path (`"evidence.commit"`,
    `"environment.device.kind"`), walked one segment at a time starting at
    `stamp`, so a caller can ask about a value nested arbitrarily deep
    without this module ever learning what any segment means.

    A path counts as present only if every segment resolves to an existing
    mapping key, all the way to the last one. Two distinct ways a path can
    fail to resolve are both reported the same way — as missing, never as
    an error: a segment simply absent at that level, or an intermediate
    segment present but holding something other than a mapping (a string,
    a number, `None`), so the walk cannot continue into it. Reporting both
    alike, rather than raising on the second, is what lets a stamp with no
    `evidence` key at all — the shape every stamp had before this schema
    grew one — report its missing paths and nothing else: incomplete,
    never invalid, so a rollback strands nothing.
    """
    missing = []
    for path in required:
        node = stamp
        present = True
        for segment in path.split("."):
            if not isinstance(node, dict) or segment not in node:
                present = False
                break
            node = node[segment]
        if not present:
            missing.append(path)
    return {"complete": not missing, "missing": missing}
