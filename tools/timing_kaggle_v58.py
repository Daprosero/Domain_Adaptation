"""Per-candidate timing of the v58 campaign (local CPU, Kaggle CPU, Kaggle GPU) and makespans.

Inputs: the probe records of `tools/timing_v58.py` (local) and of its remote entry point
`MIL_CREDA_EEG.remote_timing` (Kaggle rehearsals), several per candidate when the probe
was sharded (a group or strategy measured twice is averaged). Each candidate is turned
into the campaign cost with the SAME extrapolation as `tools/timing_v58_record.py`
(`campaign_cost`: per-epoch seconds scaled per transfer by windows; 120 phase-1 studies,
360 phase-2 studies, the final step, K3), so

    serial seconds(C1, C2) = coef1 * C1 + coef2 * C2 + fixed

Per step of the campaign (the units of `remote_units.enumerate_units`), the same terms
give each unit's minutes as a1 * C1 + a2 * C2 + b; those are what the remote-parallelism
skill's `schedule` verb places on the team's accounts (phase 1 -> phase 2 of its group ->
final -> K3, one service per step), for every cap pair asked. Nothing is launched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import timing_v58_record as rec  # noqa: E402

N_FOLDS = rec.N_FOLDS


def merge_probes(paths: list[Path]) -> dict:
    """One probe from shards: union of groups/strategies, repeated entries averaged."""
    probes = [json.loads(p.read_text()) for p in paths]
    out = {k: v for k, v in probes[0].items() if k not in ("phase1", "phase2", "sourcePass")}
    for key in ("phase1", "phase2", "sourcePass"):
        merged: dict = {}
        for probe in probes:
            for name, entry in probe[key].items():
                merged.setdefault(name, []).append(entry)
        out[key] = {}
        for name, entries in merged.items():
            first = dict(entries[0])
            if len(entries) > 1 and key != "sourcePass":
                first["steadyEpochSeconds"] = [s for e in entries
                                               for s in e.get("steadyEpochSeconds") or []]
            elif len(entries) > 1:
                first["seconds"] = sum(e["seconds"] for e in entries) / len(entries)
            out[key][name] = first
    out["setup"] = {k: sum(p["setup"][k] for p in probes) / len(probes)
                    for k in probes[0]["setup"]}
    out["shards"] = [str(p) for p in paths]
    return out


def step_units(probe: dict, transfers: dict, core: set, k2_only: set) -> dict:
    """Per step: the list of its units' (a1, a2, b) seconds, unit seconds = a1*C1 + a2*C2 + b.
    The same terms as `timing_v58_record.campaign_cost`, grouped by unit instead of summed."""
    table = rec.epoch_table(probe)
    base = rec.phase_windows(probe["windows"])
    steps: dict[str, list] = {}
    for name, w in transfers.items():
        inner = rec.phase_windows(w["inner"])
        r1, r2 = inner["phase1"] / base["phase1"], inner["phase2"] / base["phase2"]
        refit1 = w["refit"]["source_train"] / base["phase1"]
        refit2 = 2 * w["refit"]["target_train"] / base["phase2"]
        final = [0.0, 0.0, table["setup"]]
        k3 = [0.0, 0.0, table["setup"]]
        for _ in range(N_FOLDS):
            for g in rec.GROUPS:
                e = table["phase1"][g]
                steps.setdefault(f"p1:{g}", []).append(
                    (rec.N_TRIALS * rec.N_INNER * e * r1 + e * refit1, 0.0, table["setup"]))
            for s in rec.STRATEGIES:
                e = table["phase2"][s]
                steps.setdefault(f"p2:{s}", []).append(
                    (0.0, rec.N_TRIALS * rec.N_INNER * e * r2,
                     table["setup"] + table["sourcePass"][rec.GROUP_OF[s]] * r1))
            e = table["phase1"]["common"]
            final[0] += rec.N_INNER * e * r1 + e * refit1
            final[1] += sum(2 * table["phase2"][s] * refit2 for s in rec.STRATEGIES)
            if name in core:
                k3[0] += rec.K3_AGGREGATION * table["phase1"]["MIL-CREDA"] * refit1
                k3[1] += (rec.K3_AGGREGATION + rec.K3_ABLATION) * table["phase2"]["MIL-CREDA"] * refit2
            elif name in k2_only:
                k3[1] += table["phase2"]["MIL-CREDA"] * refit2
        steps.setdefault("final", []).append(tuple(final))
        if name in core or name in k2_only:
            steps.setdefault("k3", []).append(tuple(k3))
    return steps


def unit_minutes(units: list, c1: float, c2: float) -> float:
    """Mean unit minutes of a step at the caps (schedule takes one duration per step)."""
    return sum(a1 * c1 + a2 * c2 + b for a1, a2, b in units) / len(units) / 60


def plan_doc(steps_by_service: dict, c1: float, c2: float, accounts: list,
             setup_minutes: dict) -> dict:
    services = sorted(steps_by_service)
    any_steps = steps_by_service[services[0]]
    steps = []
    for sid, units in any_steps.items():
        if sid.startswith("p1:"):
            requires = []
        elif sid.startswith("p2:"):
            requires = [f"p1:{rec.GROUP_OF[sid[3:]]}"]
        elif sid == "final":
            requires = [k for k in any_steps if k.startswith("p2:")]
        else:
            requires = ["final"]
        steps.append({"id": sid.replace(":", "-").replace(" — ", "-").replace(" ", "-")
                      .replace("+", "plus"),
                      "units": len(units), "services": services,
                      "unitMinutes": {svc: round(unit_minutes(steps_by_service[svc][sid], c1, c2), 4)
                                      for svc in services},
                      "requires": requires, "axis": "fold"})
    ids = {s: d["id"] for s, d in zip(any_steps, steps)}
    for d in steps:
        d["requires"] = [ids[r] for r in d["requires"]]
    return {"schema": 1, "setupMinutes": {s: setup_minutes[s] for s in services},
            "steps": steps, "accounts": [a for a in accounts if a["service"] in services]}


def run_schedule(cli: Path, plan: dict, scratch: Path) -> dict:
    path = scratch / "plan.json"
    path.write_text(json.dumps(plan))
    done = subprocess.run([sys.executable, str(cli), "schedule", "--plan", str(path)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        return {"error": done.stderr.strip()[:2000]}
    return json.loads(done.stdout)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", action="append", required=True,
                        help="<name>=<probe.json>[,<probe.json>...]")
    parser.add_argument("--breakdown", required=True,
                        help="timing_probe_v58/breakdown.json (transfers, core, k2Only)")
    parser.add_argument("--accounts", required=True, help="accounts JSON (schedule shape)")
    parser.add_argument("--service-of", action="append", default=[],
                        help="<candidate>=<service slug> for the remote candidates")
    parser.add_argument("--setup-minutes", action="append", default=[],
                        help="<service>=<minutes> measured per-session setup")
    parser.add_argument("--parallelism-cli", required=True)
    parser.add_argument("--caps", default="10,20,30")
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    breakdown = json.loads(Path(args.breakdown).read_text())
    transfers, core, k2 = breakdown["transfers"], set(breakdown["core"]), set(breakdown["k2Only"])
    accounts = json.loads(Path(args.accounts).read_text())
    service_of = dict(item.split("=", 1) for item in args.service_of)
    setup = {k: float(v) for k, v in (item.split("=", 1) for item in args.setup_minutes)}
    cli, scratch = Path(args.parallelism_cli), Path(args.scratch)
    caps = [float(c) for c in args.caps.split(",")]

    out = {"formula": "serial seconds(C1, C2) = coef1 * C1 + coef2 * C2 + fixed "
                      "(tools/timing_v58_record.py campaign_cost)",
           "candidates": {}, "options": {}}
    steps_of = {}
    for item in args.candidate:
        name, paths = item.split("=", 1)
        probe = merge_probes([Path(p) for p in paths.split(",")])
        costs = rec.campaign_cost(probe, transfers, core, k2)
        steps = step_units(probe, transfers, core, k2)
        total = [sum(u[i] for units in steps.values() for u in units) for i in range(3)]
        assert abs(total[0] - costs["coef1"]) < 1e-6 * costs["coef1"], "coef1 mismatch"
        assert abs(total[1] - costs["coef2"]) < 1e-6 * costs["coef2"], "coef2 mismatch"
        steps_of[name] = steps
        out["candidates"][name] = {
            "device": probe["device"], "deviceEvidence": probe.get("deviceEvidence"),
            "commit": probe.get("commit"), "measuredAt": probe["measuredAt"],
            "shards": probe["shards"], "epochTable": costs["epochTable"],
            "coef1": costs["coef1"], "coef2": costs["coef2"], "fixed": costs["fixed"],
            "epochs1": costs["epochs1"], "epochs2": costs["epochs2"],
            "serialHoursPerEpochOfC1": costs["coef1"] / 3600,
            "serialHoursPerEpochOfC2": costs["coef2"] / 3600,
            "serialHours": {f"C{int(c)}": (costs["coef1"] * c + costs["coef2"] * c
                                          + costs["fixed"]) / 3600 for c in caps},
            "stepUnitSeconds": {k: {"units": len(v),
                                    "meanA1": sum(u[0] for u in v) / len(v),
                                    "meanA2": sum(u[1] for u in v) / len(v),
                                    "meanB": sum(u[2] for u in v) / len(v),
                                    "maxA1": max(u[0] for u in v), "maxA2": max(u[1] for u in v)}
                                for k, v in steps.items()}}

    remote = {n: s for n, s in service_of.items() if n in steps_of}
    options = {f"{n} only": {s: n} for n, s in remote.items()}
    if len(remote) > 1:
        options["mixed " + " + ".join(remote)] = {s: n for n, s in remote.items()}

    def schedule_at(mapping, c1, c2):
        plan = plan_doc({svc: steps_of[cand] for svc, cand in mapping.items()}, c1, c2,
                        accounts, setup)
        return run_schedule(cli, plan, scratch)

    for label, mapping in options.items():
        rows = {}
        for c in caps:
            res = schedule_at(mapping, c, c)
            rows[f"C{int(c)}"] = {k: res.get(k) for k in
                                  ("makespanHours", "serialHours", "speedup", "quotaOk",
                                   "quotaWeeks", "reasons", "error")}
            rows[f"C{int(c)}"]["stepServices"] = {s["id"]: s["services"] for s in
                                                  res.get("steps", [])}
            rows[f"C{int(c)}"]["quota"] = [{k: q.get(k) for k in
                                            ("account", "gpuHours", "quotaHours", "overQuota")}
                                           for q in res.get("quota", [])
                                           if q.get("gpuHours")][:3]
        fits = {}
        if len(mapping) > 1:
            # a mixed schedule searches 4096 step-to-service assignments (about 2 min per
            # call), so it is evaluated at the illustrative caps only; see `probes` below
            fits = {"1w": "not bisected (mixed search cost)", "2w": "not bisected"}
        for weeks in ((1, 2) if len(mapping) == 1 else ()):
            def ok(c):
                res = schedule_at(mapping, c, c)
                return res.get("makespanHours") is not None and res["makespanHours"] <= 168 * weeks
            lo_c, hi_c = 0, 1000  # largest whole C1 = C2 whose makespan fits (bisection)
            if not ok(1):
                fits[f"{weeks}w"] = None
                continue
            lo_c = 1
            while hi_c - lo_c > 1:
                mid = (lo_c + hi_c) // 2
                lo_c, hi_c = (mid, hi_c) if ok(mid) else (lo_c, mid)
            fits[f"{weeks}w"] = lo_c
        # makespan per epoch of cap, from a large-cap slope
        lo, hi = rows.get("C10", {}), rows.get("C30", {})
        out["options"][label] = {"services": mapping, "illustrative": rows,
                                 "maxEqualCapFitting": fits,
                                 "makespanHoursPerEpochOfCap": (
                                     (hi["makespanHours"] - lo["makespanHours"]) / 20
                                     if hi.get("makespanHours") and lo.get("makespanHours") else None)}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: {"maxEqualCapFitting": v["maxEqualCapFitting"],
                          "C10": v["illustrative"]["C10"]["makespanHours"],
                          "C20": v["illustrative"]["C20"]["makespanHours"],
                          "C30": v["illustrative"]["C30"]["makespanHours"]}
                      for k, v in out["options"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
