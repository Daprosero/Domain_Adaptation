"""Plan, select and report the v58 patience exercise (`MIL_CREDA_EEG.patience_exercise`).

    plan   --wave 1 --bins 15 --out wave1.json
    select --trials <trials.jsonl>... --out selections.json   (phase-1 choice, patience 10)
    plan   --wave 2 --bins 30 --selections selections.json --out wave2.json
    report --trials <trials.jsonl>... --accounts accounts.json --parallelism-cli <cli>
           --observed observed.json --out summary.json

`plan` draws each group's / strategy's trials 0..9 from the campaign's own sampler (the
GPSampler seeded `search.SAMPLER_SEED`, whose first 10 trials come from its seeded
independent RandomSampler), spreads them over one transfer per arm (4/3/3), estimates
each trial's T4 seconds from the timing probe and balances them over `--bins` notebooks
(longest processing time first). `report` aggregates the trials and plugs the measured
stop epochs into the campaign's unit accounting (`tools/timing_v58_record.py`) and the
remote-parallelism `schedule` verb, for patience 10 and the fixed caps C=20 and C=100.
Nothing here submits anything.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from MIL_CREDA_EEG import patience_exercise as px  # noqa: E402
import timing_v58_record as rec  # noqa: E402

PRODUCT = ROOT / "MIL-CREDA-EEG"
T4_PROBE = (PRODUCT / "Results" / "timing_kaggle_v58" / "probes" / "probe_kaggle_gpu-e5.json")
BREAKDOWN = PRODUCT / "Results" / "timing_probe_v58" / "breakdown.json"
GROUP_OF = rec.GROUP_OF


def draws(phase: int, name: str, n: int = px.TRIALS_PER_NAME) -> list[dict]:
    """Trials 0..n-1 of the campaign's study of `name` (identical for every transfer)."""
    import optuna
    from MIL_CREDA_EEG import search
    space = search.PHASE_ONE_SPACES[name] if phase == 1 else search.phase_two_space(name)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.GPSampler(seed=search.SAMPLER_SEED))
    return [search.suggest(study.ask(), space) for _ in range(n)]


def cost_model(probe_path: Path = T4_PROBE, breakdown_path: Path = BREAKDOWN) -> dict:
    probe = json.loads(Path(probe_path).read_text())
    breakdown = json.loads(Path(breakdown_path).read_text())
    return {"table": rec.epoch_table(probe), "base": rec.phase_windows(probe["windows"]),
            "transfers": breakdown["transfers"], "core": set(breakdown["core"]),
            "k2Only": set(breakdown["k2Only"]), "probe": str(probe_path)}


def ratios(model: dict, transfer: str) -> dict:
    w = model["transfers"][transfer]
    inner, base = rec.phase_windows(w["inner"]), model["base"]
    return {"r1": inner["phase1"] / base["phase1"], "r2": inner["phase2"] / base["phase2"],
            "refit1": w["refit"]["source_train"] / base["phase1"],
            "refit2": 2 * w["refit"]["target_train"] / base["phase2"]}


def trial_seconds(model: dict, phase: int, name: str, transfer: str,
                  epochs: int = px.CAP) -> float:
    r = ratios(model, transfer)
    e = model["table"][f"phase{phase}"][name]
    return rec.N_INNER * epochs * e * (r["r1"] if phase == 1 else r["r2"])


def plan(wave: int, bins: int, selections: list | None, model: dict) -> dict:
    trials = []
    if wave == 1:
        names = [(1, g) for g in px.GROUPS]
    else:
        from MIL_CREDA_EEG import campaign
        campaign.register(list(px.STRATEGIES))
        names = [(2, s) for s in px.STRATEGIES]
    for index, (phase, name) in enumerate(names):
        for k, params in enumerate(draws(phase, name)):
            t = px.transfer_of_draw(k, index)
            trials.append({"phase": phase, "name": name, "transfer": t, "draw": k,
                           "params": params,
                           **({"group": GROUP_OF[name]} if phase == 2 else {}),
                           "seconds": trial_seconds(model, phase, name, t)})
    setup = model["table"]["setup"]
    group_setup = {}
    sel = {(s["group"], s["transfer"]): s for s in selections or []}
    for (g, t), s in sel.items():
        r = ratios(model, t)
        group_setup[(g, t)] = model["table"]["phase1"][g] * r["refit1"] * s["epochs"]
    notebooks = px.assign(trials, bins, transfer_setup=setup * 0.2, group_setup=group_setup)
    jobs = []
    for i, nb in enumerate(notebooks):
        need = sorted({(t["group"], t["transfer"]) for t in nb if t["phase"] == 2})
        est = (setup + sum(t["seconds"] for t in nb)
               + sum(group_setup.get(k, 0.0) for k in need))
        jobs.append({"job": f"v58-patience-w{wave}-{i:02d}", "estimatedSeconds": est,
                     "trials": [{k: v for k, v in t.items() if k not in ("seconds", "group")}
                                for t in nb],
                     "phase_one": [sel[k] for k in need]})
    return {"wave": wave, "bins": bins, "costProbe": model["probe"], "schedule": px.SCHEDULE,
            "jobs": jobs}


def load_rows(paths) -> list[dict]:
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select(rows: list[dict], patience: int = 10) -> list[dict]:
    """Per (group, transfer): the phase-1 trial with the best score under `patience`
    (earliest draw on ties) and E1 = its best epoch under that patience."""
    out = {}
    for r in rows:
        if r.get("error") or r["phase"] != 1:
            continue
        o = r["outcomes"][f"patience{patience}"]
        key = (r["name"], r["transfer"])
        cand = (o["bestScore"] if o["bestScore"] is not None else -1.0, -r["draw"])
        if key not in out or cand > out[key][0]:
            out[key] = (cand, {"group": r["name"], "transfer": r["transfer"],
                               "params": r["params"], "epochs": o["bestEpoch"],
                               "draw": r["draw"], "score": o["bestScore"]})
    return [v for _, v in sorted(out.values(), key=lambda x: (x[1]["group"], x[1]["transfer"]))]


# ------------------------------------------------------------------ report


def aggregate(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    out = {"trials": len(rows), "errors": [r for r in rows if r.get("error")],
           "byName": {}, "byNameArm": {}, "byArm": {}}
    for phase in (1, 2):
        for name in sorted({r["name"] for r in ok if r["phase"] == phase}):
            sub = [r for r in ok if r["phase"] == phase and r["name"] == name]
            key = f"p{phase}:{name}"
            out["byName"][key] = {f"patience{p}": px.distribution(sub, p) for p in px.PATIENCES}
            out["byName"][key]["fullBestEpoch"] = sorted(r["outcomes"]["full"]["bestEpoch"]
                                                         for r in sub)
            for arm in sorted({r["arm"] for r in sub}):
                s2 = [r for r in sub if r["arm"] == arm]
                out["byNameArm"][f"{key}|{arm}"] = {f"patience{p}": px.distribution(s2, p)
                                                    for p in px.PATIENCES}
        for arm in sorted({r["arm"] for r in ok if r["phase"] == phase}):
            sub = [r for r in ok if r["phase"] == phase and r["arm"] == arm]
            out["byArm"][f"p{phase}|{arm}"] = {f"patience{p}": px.distribution(sub, p)
                                               for p in px.PATIENCES}
    return out


def epochs_of(rows: list[dict], phase: int, name: str, arm: str, stat: str,
              patience: int = 10) -> tuple[float, float]:
    """(trial epochs, refit epochs) of a study under `patience` at `stat` (mean, p25,
    median, p90): the stop epoch every trial trains, and the best epoch its refit trains.
    Per (name, arm) when measured there, else per name."""
    key = f"patience{patience}"
    sub = [r for r in rows if not r.get("error") and r["phase"] == phase
           and r["name"] == name]
    on_arm = [r for r in sub if r["arm"] == arm] or sub
    stops = [r["outcomes"][key]["stopEpoch"] for r in on_arm]
    bests = [r["outcomes"][key]["bestEpoch"] for r in on_arm]

    def at(xs):
        import statistics
        if stat == "mean":
            return statistics.fmean(xs)
        return px._quantile(xs, {"p25": .25, "median": .5, "p90": .9}[stat])
    return at(stops), at(bests)


def unit_seconds(model: dict, epochs) -> dict:
    """Step -> unit seconds over every transfer and outer fold, the terms of
    `timing_v58_record.campaign_cost` with each study's own trial and refit epochs:
    `epochs(phase, name, arm) -> (trial epochs, refit epochs)`."""
    from MIL_CREDA_EEG import assignment
    arm_of = {t.name: t.arm for t in assignment.TRANSFERS}
    table, setup = model["table"], model["table"]["setup"]
    steps: dict[str, list[float]] = {}
    for name in model["transfers"]:
        r = ratios(model, name)
        arm = arm_of[name]
        for _ in range(rec.N_FOLDS):
            for g in rec.GROUPS:
                e = table["phase1"][g]
                c, b = epochs(1, g, arm)
                steps.setdefault(f"p1:{g}", []).append(
                    rec.N_TRIALS * rec.N_INNER * e * r["r1"] * c + e * r["refit1"] * b + setup)
            for s in rec.STRATEGIES:
                e = table["phase2"][s]
                c, _ = epochs(2, s, arm)
                steps.setdefault(f"p2:{s}", []).append(
                    rec.N_TRIALS * rec.N_INNER * e * r["r2"] * c + setup
                    + table["sourcePass"][GROUP_OF[s]] * r["r1"])
            e = table["phase1"]["common"]
            c, b = epochs(1, "common", arm)
            final = rec.N_INNER * e * r["r1"] * c + e * r["refit1"] * b + setup
            final += sum(2 * table["phase2"][s] * r["refit2"] * epochs(2, s, arm)[1]
                         for s in rec.STRATEGIES)
            steps.setdefault("final", []).append(final)
            if name in model["core"] or name in model["k2Only"]:
                _, b1 = epochs(1, "MIL-CREDA", arm)
                _, b2 = epochs(2, "MIL-CREDA", arm)
                if name in model["core"]:
                    k3 = (rec.K3_AGGREGATION * table["phase1"]["MIL-CREDA"] * r["refit1"] * b1
                          + (rec.K3_AGGREGATION + rec.K3_ABLATION)
                          * table["phase2"]["MIL-CREDA"] * r["refit2"] * b2)
                else:
                    k3 = table["phase2"]["MIL-CREDA"] * r["refit2"] * b2
                steps.setdefault("k3", []).append(k3 + setup)
    return steps


def schedule(cli: Path, steps: dict, accounts: list, setup_minutes: float,
             scratch: Path) -> dict:
    import timing_kaggle_v58 as tk
    units = {k: [(0.0, 0.0, s) for s in v] for k, v in steps.items()}
    doc = tk.plan_doc({"kaggle-gpu": units}, 0.0, 0.0, accounts, {"kaggle-gpu": setup_minutes})
    res = tk.run_schedule(cli, doc, scratch)
    quota = [q for q in res.get("quota", []) if q.get("gpuHours")]
    return {"makespanHours": res.get("makespanHours"), "serialHours": res.get("serialHours"),
            "speedup": res.get("speedup"), "quotaOk": res.get("quotaOk"),
            "quotaWeeks": res.get("quotaWeeks"), "reasons": res.get("reasons"),
            "error": res.get("error"),
            "gpuHoursTotal": sum(q["gpuHours"] for q in quota) if quota else None,
            "gpuHoursMaxAccount": max((q["gpuHours"] for q in quota), default=None),
            "quotaHoursPerAccount": quota[0].get("quotaHours") if quota else None,
            "longestUnitHours": max(max(v) for v in steps.values()) / 3600,
            "serialUnitHours": sum(sum(v) for v in steps.values()) / 3600}


def report(rows, model, accounts, cli, setup_minutes, scratch) -> dict:
    out = {"aggregate": aggregate(rows), "estimate": {}}
    variants = {f"patience10-{stat}": (lambda stat: lambda p, n, a: epochs_of(rows, p, n, a, stat))(stat)
                for stat in ("mean", "p25", "median", "p90")}
    variants["fixed-C20"] = lambda p, n, a: (20, 20)
    variants["fixed-C100"] = lambda p, n, a: (100, 100)
    for label, fn in variants.items():
        steps = unit_seconds(model, fn)
        out["estimate"][label] = schedule(cli, steps, accounts, setup_minutes, scratch)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--wave", type=int, choices=(1, 2), required=True)
    p.add_argument("--bins", type=int, required=True)
    p.add_argument("--selections")
    p.add_argument("--out", required=True)
    s = sub.add_parser("select")
    s.add_argument("--trials", nargs="+", required=True)
    s.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--trials", nargs="+", required=True)
    r.add_argument("--accounts", required=True)
    r.add_argument("--parallelism-cli", required=True)
    r.add_argument("--setup-minutes", type=float, required=True,
                   help="per-session overhead before the first unit: queue + boot + clone")
    r.add_argument("--scratch", required=True)
    r.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "plan":
        sel = json.loads(Path(args.selections).read_text()) if args.selections else None
        if args.wave == 2 and not sel:
            parser.error("wave 2 needs --selections")
        doc = plan(args.wave, args.bins, sel, cost_model())
        Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps([round(j["estimatedSeconds"] / 60, 1) for j in doc["jobs"]]))
    elif args.cmd == "select":
        doc = select(load_rows(args.trials))
        Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps([(d["group"], d["transfer"], d["draw"], d["epochs"]) for d in doc]))
    else:
        doc = report(load_rows(args.trials), cost_model(),
                     json.loads(Path(args.accounts).read_text()), Path(args.parallelism_cli),
                     args.setup_minutes, Path(args.scratch))
        Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps({k: {x: v[x] for x in ("makespanHours", "gpuHoursTotal",
                                                  "quotaWeeks", "quotaOk")}
                          for k, v in doc["estimate"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
