"""Extrapolate the local timing probes (`tools/timing_v58.py`) to the v58 campaign.

The campaign's units (`remote_units.enumerate_units`) and the training each one does
(experiments plan v58, Secs. 4.5, 6.3, 7.3, 7.7), with C1 and C2 the search's epoch caps
(still pending, Sec. 7.6, so everything here is linear in them and stated per epoch of
cap), per transfer t and outer fold r (4 folds, one report seed):

- `search.phase1.<t>.<group>.outer<r>` (3 groups): 20 trials x 3 inner splits x C1
  epochs, plus the refit on all of F for E1 <= C1 epochs;
- `search.phase2.<t>.<strategy>.outer<r>` (9 strategies): 20 trials x 3 inner splits x
  C2 epochs, plus the centroid rule's source pass once per study (inference);
- `final.<t>`, per fold: the upper bound (common group, 3 inner splits + refit, C1 each)
  and, per strategy, at most two phase-2 refits (primary, and EnsV where it differs) of
  E2 <= C2 epochs -- the bound `remote_units.timing_estimate` uses;
- `k3.<t>`, per fold, on the core transfers: four aggregation variants through both
  phases (C1 + C2) and two ablations through phase 2 (C2), refit-sized, at MIL-CREDA's
  epoch time; on the declared K2 transfer that is not core, one ablation (C2).

Epoch seconds per transfer: the probe's steady epoch (inner split 0 of outer fold 0 of
the probed transfer) scaled by windows, phase 1 by source train + source validation
windows, phase 2 by 2 x target train + target validation windows (each phase-2 step pairs
a target batch with a source batch of the same size), a refit by its own training windows
(no validation pass). The windows of inner split 0 of outer fold 0 of each transfer stand
for all of its folds and splits.

Writes the remote-parallelism schema-1 record: one measurement row whose
`secondsPerUnit` is the campaign-weighted mean epoch (total seconds of the C1 = C2 = 1
campaign over its epoch count) and whose `setupSeconds` is one fold's share of the
non-epoch work (bag loading per unit, source passes), with `scales.full` = 1 arm x 4
folds x 1 seed x the epoch count of one fold at C1 = C2 = 1; so `estimate` returns the
makespan of the campaign per epoch of cap. `scales.poc` is one outer fold of it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

N_TRIALS, N_INNER, N_FOLDS = 20, 3, 4
GROUPS = ("common", "AdaptMIL", "MIL-CREDA")
STRATEGIES = ("AdaptMIL", "UDA-DDA", "CCA-LSC", "CASUAL", "COT", "CS+CCS",
              "CLUST — variante source-free", "CLUDA", "MIL-CREDA")
GROUP_OF = {s: (s if s in ("AdaptMIL", "MIL-CREDA") else "common") for s in STRATEGIES}
K3_AGGREGATION, K3_ABLATION = 4, 2


def phase_windows(windows: dict) -> dict:
    """The windows one epoch of each kind processes (see the module docstring)."""
    return {"phase1": windows["source_train"] + windows["source_valid"],
            "phase2": 2 * windows["target_train"] + windows["target_valid"]}


def epoch_table(probe: dict) -> dict:
    """Steady epoch seconds per phase-1 group and phase-2 strategy, from one probe."""
    def steady(entry):
        if entry.get("error") or not entry.get("steadyEpochSeconds"):
            raise ValueError(f"no steady epoch measured: {entry.get('error')}")
        values = entry["steadyEpochSeconds"]
        return sum(values) / len(values)
    return {"phase1": {g: steady(probe["phase1"][g]) for g in GROUPS},
            "phase2": {s: steady(probe["phase2"][s]) for s in STRATEGIES},
            "sourcePass": {g: probe["sourcePass"][g]["seconds"] for g in GROUPS},
            "setup": probe["setup"]["loadBagsSeconds"] + probe["setup"]["searchDataSeconds"]}


def campaign_cost(probe: dict, transfers: dict, core: set, k2_only: set) -> dict:
    """Seconds of the whole campaign as coef1 * C1 + coef2 * C2 + fixed, and the epoch
    counts at C1 = C2 = 1. `transfers`: name -> {"inner": windows, "refit": windows}."""
    table = epoch_table(probe)
    base = phase_windows(probe["windows"])
    coef1 = coef2 = fixed = 0.0
    epochs1 = epochs2 = 0
    per_unit = {}
    for name, w in transfers.items():
        inner = phase_windows(w["inner"])
        r1, r2 = inner["phase1"] / base["phase1"], inner["phase2"] / base["phase2"]
        refit1 = w["refit"]["source_train"] / base["phase1"]
        refit2 = 2 * w["refit"]["target_train"] / base["phase2"]
        for r in range(N_FOLDS):
            for g in GROUPS:
                e = table["phase1"][g]
                c = N_TRIALS * N_INNER * e * r1 + e * refit1
                coef1 += c
                epochs1 += N_TRIALS * N_INNER + 1
                fixed += table["setup"]
                per_unit.setdefault(f"search.phase1.{g}", []).append(c)
            for s in STRATEGIES:
                e = table["phase2"][s]
                c = N_TRIALS * N_INNER * e * r2
                coef2 += c
                epochs2 += N_TRIALS * N_INNER
                fixed += table["setup"] + table["sourcePass"][GROUP_OF[s]] * r1
                per_unit.setdefault(f"search.phase2.{s}", []).append(c)
            # final: upper bound (3 inner + refit, C1) and two phase-2 refits per strategy
            e = table["phase1"]["common"]
            coef1 += N_INNER * e * r1 + e * refit1
            epochs1 += N_INNER + 1
            final2 = sum(2 * table["phase2"][s] * refit2 for s in STRATEGIES)
            coef2 += final2
            epochs2 += 2 * len(STRATEGIES)
            per_unit.setdefault("final.fold-C1", []).append(N_INNER * e * r1 + e * refit1)
            per_unit.setdefault("final.fold-C2", []).append(final2)
            if name in core:
                coef1 += K3_AGGREGATION * table["phase1"]["MIL-CREDA"] * refit1
                coef2 += (K3_AGGREGATION + K3_ABLATION) * table["phase2"]["MIL-CREDA"] * refit2
                epochs1 += K3_AGGREGATION
                epochs2 += K3_AGGREGATION + K3_ABLATION
            elif name in k2_only:
                coef2 += table["phase2"]["MIL-CREDA"] * refit2
                epochs2 += 1
        fixed += table["setup"]  # final step's load
    return {"coef1": coef1, "coef2": coef2, "fixed": fixed, "epochs1": epochs1,
            "epochs2": epochs2, "epochTable": table,
            "perUnitSecondsPerEpochOfCap": {k: {"mean": sum(v) / len(v), "max": max(v),
                                                "units": len(v)}
                                            for k, v in per_unit.items()}}


def timing_record(costs: dict, device: str, measured_at: str, arm: str = "campaign-v58",
                  units_measured: int = 1) -> dict:
    """Schema 1, one local row; `estimate` over it gives coef1 + coef2 + fixed."""
    epochs = costs["epochs1"] + costs["epochs2"]
    if epochs % N_FOLDS:
        raise ValueError(f"{epochs} epochs do not split over {N_FOLDS} folds")
    seconds = costs["coef1"] + costs["coef2"]
    return {"schema": 1,
            "measurements": [{"measuredAt": measured_at, "host": "local", "device": device,
                              "arm": arm, "unit": "epoch",
                              "secondsPerUnit": seconds / epochs,
                              "unitsMeasured": units_measured,
                              "setupSeconds": costs["fixed"] / N_FOLDS}],
            "scales": {"poc": {"arms": 1, "folds": 1, "seeds": 1, "epochs": epochs // N_FOLDS},
                       "full": {"arms": 1, "folds": N_FOLDS, "seeds": 1,
                                "epochs": epochs // N_FOLDS}}}


def transfer_windows() -> tuple[dict, set, set]:
    from MIL_CREDA_EEG import assignment, bag_data, mechanisms, search
    bags = bag_data.load_bags()
    out = {}
    for t in assignment.TRANSFERS:
        data = search.search_data(t, bags)
        inner = data.inner_view(0, 0).rotations[0]
        refit = data.rotations[0]
        count = lambda d: {k: int(sum(b.windows.shape[0] for b in v)) for k, v in d.items()}
        out[t.name] = {"inner": count(inner), "refit": count(refit)}
    summary = json.loads((ROOT / "MIL-CREDA-EEG" / "Results" / "separability" /
                          "summary.json").read_text())
    core = {n for names in summary["core"].values() for n in names}
    k2 = {t.name for t in mechanisms.K2_DECLARED.values()} - core
    return out, core, k2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", action="append", required=True,
                        help="probe JSON; the first one is written to --out")
    parser.add_argument("--out", required=True, nargs="+",
                        help="one timing record path per --probe")
    parser.add_argument("--breakdown", required=True)
    args = parser.parse_args(argv)
    if len(args.out) != len(args.probe):
        parser.error("one --out per --probe")
    windows, core, k2 = transfer_windows()
    breakdown = {"transfers": windows, "core": sorted(core), "k2Only": sorted(k2),
                 "formula": ("seconds(C1, C2) = coef1 * C1 + coef2 * C2 + fixed; see "
                             "tools/timing_v58_record.py"), "devices": {}}
    for probe_path, out in zip(args.probe, args.out):
        probe = json.loads(Path(probe_path).read_text())
        costs = campaign_cost(probe, windows, core, k2)
        steady = sum(len(v.get("steadyEpochSeconds") or [])
                     for k in ("phase1", "phase2") for v in probe[k].values())
        record = timing_record(costs, probe["device"], probe["measuredAt"],
                               units_measured=steady)
        Path(out).write_text(json.dumps(record, indent=1) + "\n")
        breakdown["devices"][probe["device"]] = {
            **costs, "probe": {k: probe[k] for k in probe if k not in ("phase1", "phase2",
                                                                      "sourcePass")},
            "probeRaw": {k: probe[k] for k in ("phase1", "phase2", "sourcePass")},
            "record": out}
    Path(args.breakdown).write_text(json.dumps(breakdown, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({d: {k: v[k] for k in ("coef1", "coef2", "fixed", "epochs1", "epochs2")}
                      for d, v in breakdown["devices"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
