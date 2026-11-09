"""Per-arm timing rows of one probed v63 unit, for remote-parallelism's `estimate`/`plan`.

Reads the Optuna studies a `v63_campaign` P unit left (`<product>/Results/v63/hpo/
<slug>__phase<p>__<arm>__outer<r>.db`) and turns each arm's trials into seconds per
training epoch of one inner split: trial wall time / (3 inner splits x trained epochs).
Appends one schema-2 row per arm to `<out>` (a `.../Results/timing.json`), with `scales`
copied from plan v63 (Secs. 4.6, 7.7) for a new record:

- full: 12 arms (3 phase-1 groups, 9 phase-2 strategies), 64 outer folds (16 transfers x 4
  rotations, the upper bound before the Sec. 0 exclusions), 1 seed (the 10 report seeds are
  in the epochs), 7000 epochs per arm and fold (20 trials x 3 inner splits x the 100-epoch
  cap, plus 10 refits x at most 100 epochs; patience stops most trials earlier);
- poc: the probe itself, 12 arms, 1 fold, 1 seed, `--probe-epochs` epochs per arm.

Usage: v63_unit_timing.py <product> <slug> <rotation> <host> <candidate> <out> [--probe-epochs N]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

import optuna

FULL = {"arms": 12, "folds": 64, "seeds": 1, "epochs": 20 * 3 * 100 + 10 * 100}


def rows(product: Path, slug: str, rotation: int, host: str, candidate: str) -> list[dict]:
    out = []
    for db in sorted((product / "Results" / "v63" / "hpo").glob(f"{slug}__phase*__outer{rotation}.db")):
        study = optuna.load_study(study_name=db.stem, storage=f"sqlite:///{db}")
        phase, arm = db.stem.split("__")[1], db.stem.split("__")[2]
        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not done:
            continue
        seconds = sum((t.datetime_complete - t.datetime_start).total_seconds() for t in done)
        epochs = sum(int(t.user_attrs["trainedEpochs"]) for t in done)
        out.append({"measuredAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "host": host, "candidate": candidate, "device": "cpu",
                    "arm": f"{phase}:{arm}", "unit": "epoch",
                    "secondsPerUnit": max(round(seconds / (3 * epochs), 4), 0.0001),
                    "unitsMeasured": 3 * epochs, "processes": 1,
                    "threads": int(os.environ.get("V63_THREADS", "4"))})
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    for name in ("product", "slug", "rotation", "host", "candidate", "out"):
        p.add_argument(name)
    p.add_argument("--probe-epochs", type=int, default=400)
    a = p.parse_args()
    out = Path(a.out)
    if out.name != "timing.json" or out.parent.name != "Results":
        raise SystemExit("the record is <Name>/Results/timing.json")
    new = rows(Path(a.product), a.slug, int(a.rotation), a.host, a.candidate)
    record = json.loads(out.read_text()) if out.is_file() else {
        "schema": 2, "hosts": {}, "measurements": [],
        "scales": {"poc": {"arms": 12, "folds": 1, "seeds": 1, "epochs": a.probe_epochs},
                   "full": FULL}}
    record["hosts"].setdefault(a.host, {"cpuCount": os.cpu_count(), "ramGB": None,
                                        "mps": False, "cuda": False})
    record["measurements"] += new
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{k: r[k] for k in ("arm", "secondsPerUnit", "unitsMeasured")} for r in new]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
