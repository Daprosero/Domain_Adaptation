"""Plan the v59 campaign's launch: chains of units, queues, the `--unit` order. Submits nothing.

    queues  --slots 30 [--stat median] [--scale 1.0] [--transfer NAME]... --out queues.json
    order   --queues queues.json --probe <distribute output on u0000..> --out units.json
    check   --queues queues.json --distribute <distribute output on the ordered units>

A phase-2 unit needs its phase-1 unit in the same session (`remote_units._check_ready`),
so the search wave is placed as CHAINS: one phase-1 study followed by the phase-2 studies
of its group, for one transfer and outer fold (common: 1 + 7 units, AdaptMIL and
MIL-CREDA: 1 + 1). `remote_cli distribute` deals unit i to place i mod places and every
place holds the same number of units, so `queues` balances the chains by minutes under
equal unit counts, and `order` inverts the deal: given the deal of placeholder ids, it
writes the unit list that lands each queue on one place, in order. `check` proves it.

Minutes come from the patience exercise and the T4 timing probe: each study's trial epochs
are the measured stop under plan v59's patience (30 phase 1, 10 phase 2; `--stat`), replayed
on the exercise's full 100-epoch series, per (name, arm). `--scale` applies a measured
ratio (a smoke's real seconds over this model). The final step's upper bound trains its
whole cap (no patience, L10). The search units are `remote_units.enumerate_units`' ids.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import patience_exercise_v58 as tool  # noqa: E402
from MIL_CREDA_EEG import patience_exercise as px  # noqa: E402

rec = tool.rec
RAW = tool.PRODUCT / "Results" / "patience_exercise_v58" / "raw"
PATIENCE = {1: 30, 2: 10}
CAP = 100


def slug(name: str) -> str:
    from MIL_CREDA_EEG import model_store
    return model_store.name_slug(name)


def transfer_slug(name: str) -> str:
    from MIL_CREDA_EEG import assignment, model_store
    return model_store.transfer_slug({t.name: t for t in assignment.TRANSFERS}[name])


def stops(rows: list[dict]) -> list[dict]:
    """Every exercise trial's (stop, best) epochs under plan v59's patience."""
    out = []
    for r in rows:
        if r.get("error") or not r.get("series"):
            continue
        o = px.patience_stop(r["series"], PATIENCE[r["phase"]])
        out.append({**{k: r[k] for k in ("phase", "name", "arm")},
                    "stop": o["stopEpoch"], "best": o["bestEpoch"]})
    return out


def epochs_of(rows: list[dict], stat: str):
    def at(xs):
        if stat == "mean":
            return statistics.fmean(xs)
        return px._quantile(xs, {"p25": .25, "median": .5, "p90": .9}[stat])

    def epochs(phase: int, name: str, arm: str) -> tuple[float, float]:
        sub = [r for r in rows if r["phase"] == phase and r["name"] == name]
        on_arm = [r for r in sub if r["arm"] == arm] or sub
        return at([r["stop"] for r in on_arm]), at([r["best"] for r in on_arm])
    return epochs


def unit_minutes(model: dict, epochs) -> dict[str, float]:
    """Unit id -> minutes, the terms of `patience_exercise_v58.unit_seconds` per unit."""
    from MIL_CREDA_EEG import assignment
    arm_of = {t.name: t.arm for t in assignment.TRANSFERS}
    table, setup = model["table"], model["table"]["setup"]
    out: dict[str, float] = {}
    for name in model["transfers"]:
        r, arm, t = tool.ratios(model, name), arm_of[name], transfer_slug(name)
        final = setup
        for fold in range(rec.N_FOLDS):
            for g in rec.GROUPS:
                e = table["phase1"][g]
                c, b = epochs(1, g, arm)
                out[f"search.phase1.{t}.{slug(g)}.outer{fold}"] = (
                    rec.N_TRIALS * rec.N_INNER * e * r["r1"] * c + e * r["refit1"] * b) / 60
            for s in rec.STRATEGIES:
                e = table["phase2"][s]
                c, _ = epochs(2, s, arm)
                out[f"search.phase2.{t}.{slug(s)}.outer{fold}"] = (
                    rec.N_TRIALS * rec.N_INNER * e * r["r2"] * c
                    + table["sourcePass"][rec.GROUP_OF[s]] * r["r1"]) / 60
            e = table["phase1"]["common"]
            final += rec.N_INNER * e * r["r1"] * CAP + e * r["refit1"] * CAP
            final += sum(2 * table["phase2"][s] * r["refit2"] * epochs(2, s, arm)[1]
                         for s in rec.STRATEGIES)
        out[f"final.{t}"] = final / 60
        if name in model["core"] or name in model["k2Only"]:
            _, b1 = epochs(1, "MIL-CREDA", arm)
            _, b2 = epochs(2, "MIL-CREDA", arm)
            per = table["phase2"]["MIL-CREDA"] * r["refit2"] * b2
            k3 = (per if name not in model["core"] else
                  rec.K3_AGGREGATION * table["phase1"]["MIL-CREDA"] * r["refit1"] * b1
                  + (rec.K3_AGGREGATION + rec.K3_ABLATION) * per)
            out[f"k3.{t}"] = (rec.N_FOLDS * k3 + setup) / 60
    return out


def chains(minutes: dict[str, float], model: dict) -> list[dict]:
    """One chain per (transfer, outer fold, group): its phase-1 unit, then its phase 2."""
    out = []
    for name in model["transfers"]:
        t = transfer_slug(name)
        for fold in range(rec.N_FOLDS):
            for g in rec.GROUPS:
                ids = [f"search.phase1.{t}.{slug(g)}.outer{fold}"]
                ids += [f"search.phase2.{t}.{slug(s)}.outer{fold}" for s in rec.STRATEGIES
                        if rec.GROUP_OF[s] == g]
                out.append({"units": ids, "minutes": sum(minutes[i] for i in ids)})
    return out


def balance(items: list[dict], slots: int) -> list[list[dict]]:
    """Chains into `slots` queues of equal unit count, longest first, then swaps of equal
    size between the heaviest queue and any other while the heaviest gets lighter."""
    total = sum(len(c["units"]) for c in items)
    if total % slots:
        raise SystemExit(f"{total} units do not deal evenly over {slots} places")
    size = total // slots
    queues: list[list[dict]] = [[] for _ in range(slots)]
    room = [size] * slots
    load = [0.0] * slots
    for c in sorted(items, key=lambda c: (-len(c["units"]), -c["minutes"])):
        fits = [k for k in range(slots) if room[k] >= len(c["units"])]
        if not fits:
            raise SystemExit("the chains do not pack into queues of equal size")
        k = min(fits, key=lambda k: load[k])
        queues[k].append(c)
        room[k] -= len(c["units"])
        load[k] += c["minutes"]
    if any(room):
        raise SystemExit(f"queues left unfilled: {room}")
    improved = True
    while improved:
        improved = False
        worst = max(range(slots), key=lambda k: load[k])
        for a in sorted(queues[worst], key=lambda c: -c["minutes"]):
            best = None
            for k in range(slots):
                for b in queues[k]:
                    if k == worst or len(b["units"]) != len(a["units"]):
                        continue
                    delta = a["minutes"] - b["minutes"]
                    if delta > 0 and load[k] + delta < load[worst] - 1e-9 and \
                            (best is None or max(load[worst] - delta, load[k] + delta) < best[0]):
                        best = (max(load[worst] - delta, load[k] + delta), k, b, delta)
            if best:
                _, k, b, delta = best
                queues[worst].remove(a)
                queues[k].remove(b)
                queues[worst].append(b)
                queues[k].append(a)
                load[worst] -= delta
                load[k] += delta
                improved = True
                break
    return queues


def queues_cmd(args) -> int:
    rows = stops(tool.load_rows(sorted(glob.glob(str(RAW / "*.trials.jsonl")))))
    model = tool.cost_model()
    if args.transfer:
        model = {**model, "transfers": {k: model["transfers"][k] for k in args.transfer}}
    minutes = {k: v * args.scale for k, v in unit_minutes(model, epochs_of(rows, args.stat)).items()}
    groups = chains(minutes, model)
    queues = balance(groups, args.slots)
    loads = sorted(sum(c["minutes"] for c in q) for q in queues)
    search = sum(v for k, v in minutes.items() if k.startswith("search."))
    wave_two = {k: v for k, v in minutes.items() if not k.startswith("search.")}
    doc = {"stat": args.stat, "scale": args.scale, "patience": PATIENCE,
           "units": len([k for k in minutes if k.startswith("search.")]),
           "searchGpuHours": search / 60,
           "queueHours": {"min": loads[0] / 60, "median": statistics.median(loads) / 60,
                          "max": loads[-1] / 60},
           "waveTwoMinutes": wave_two, "waveTwoGpuHours": sum(wave_two.values()) / 60,
           "queues": [{"minutes": sum(c["minutes"] for c in q),
                       "units": [u for c in sorted(q, key=lambda c: -c["minutes"])
                                 for u in c["units"]]} for q in queues]}
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k not in ("queues", "waveTwoMinutes")},
                     indent=1))
    return 0


def order_cmd(args) -> int:
    """Invert the deal: placeholder `u<i>` of probe queue k, position p, becomes unit p of
    queue k."""
    plan = json.loads(Path(args.queues).read_text())["queues"]
    probe = json.loads(Path(args.probe).read_text())["queues"]
    if len(probe) != len(plan) or any(len(p["units"]) != len(q["units"])
                                      for p, q in zip(probe, plan)):
        raise SystemExit("the deal does not match the plan: every place must hold one queue")
    ordered: dict[int, str] = {}
    for p, q in zip(probe, plan):
        for placeholder, unit in zip(p["units"], q["units"]):
            ordered[int(placeholder[1:])] = unit
    Path(args.out).write_text(json.dumps([ordered[i] for i in sorted(ordered)], indent=0) + "\n")
    print(f"{len(ordered)} units ordered")
    return 0


def check_cmd(args) -> int:
    plan = [q["units"] for q in json.loads(Path(args.queues).read_text())["queues"]]
    dealt = [q["units"] for q in json.loads(Path(args.distribute).read_text())["queues"]]
    same = sorted(map(tuple, plan)) == sorted(map(tuple, dealt))
    print(json.dumps({"queues": len(dealt), "everyQueueIsItsPlannedChainOrder": same}))
    return 0 if same else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    verbs = parser.add_subparsers(dest="verb", required=True)
    q = verbs.add_parser("queues")
    q.add_argument("--slots", type=int, default=30)
    q.add_argument("--stat", default="median", choices=("mean", "p25", "median", "p90"))
    q.add_argument("--scale", type=float, default=1.0)
    q.add_argument("--transfer", action="append", default=[],
                   help="repeatable: plan only these transfers (default every transfer)")
    q.add_argument("--out", required=True)
    o = verbs.add_parser("order")
    o.add_argument("--queues", required=True)
    o.add_argument("--probe", required=True)
    o.add_argument("--out", required=True)
    c = verbs.add_parser("check")
    c.add_argument("--queues", required=True)
    c.add_argument("--distribute", required=True)
    args = parser.parse_args(argv)
    return {"queues": queues_cmd, "order": order_cmd, "check": check_cmd}[args.verb](args)


if __name__ == "__main__":
    raise SystemExit(main())
