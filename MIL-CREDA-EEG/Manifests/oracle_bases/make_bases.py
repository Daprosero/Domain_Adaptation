"""Phase-1 bases of the oracle screening, computed once so every method of every notebook starts phase 2
from the same file. One synthetic development unit, seed 1 only (phase 1 is frozen): xcompare.shared_base at the unit's recorded
phase-1 settings (cut from the inner split, base on all source training subjects, the bridged and
target-initialised copy the comparators start from). Usage: make_bases.py <unit d1p5-s0p5-r0> <outdir>"""
import glob, json, os, subprocess, sys, time
import torch
from MIL_CREDA_EEG import campaign, remote_units, spectral_trunk, xcompare as XC, xpilot as XP
unit_key, out = sys.argv[1], sys.argv[2]
th = int(os.environ.get("PROBE_THREADS", "4")); torch.set_num_threads(th); os.environ["MIL_CREDA_THREADS"] = str(th)
RH = "/Users/diego/Proyectos/MIL-CREDA/implementations/Domain_Adaptation/MIL-CREDA-EEG/.remote-execution/rehearsal"
d = glob.glob(f"{RH}/*/papersmith-cf-{unit_key}")[0]
kw = json.load(open(f"{d}/run-config.json"))["run"]["kwargs"]; x = json.load(open(glob.glob(f"{d}/xpilot-*.json")[0]))
p1 = kw.get("phase1_params") or x["runs"]["1"]["phase1Search"]["params"]
rot = int(unit_key.rsplit("-r", 1)[1])
unit = {k: kw[k] for k in ("setting", "d_target", "shift", "artifact_rate", "event_fraction", "paired", "n_subjects") if k in kw}
cfg = remote_units.resolve_inputs(campaign.check_config(kw["config"]), environ={})
built = XP.build_sides(cfg, {}, experiment="sim", rotation=rot, half=0, **unit)
sides, name, cfg = built["sides"], built["unit"], built["cfg"]
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                        cwd=os.path.dirname(XC.__file__)).stdout.strip()
with spectral_trunk.installed(), XP.window_samples(built["samples"]), XP.full_trace_mode(False):
    for sd in (1,):
        t0 = time.time(); b = XC.shared_base(sides, cfg, torch.device("cpu"), sd, rot, p1, name)
        rec = {"unitKey": unit_key, "unit": name, "seed": sd, "rotation": rot, "phase1": p1, "unitKwargs": unit,
               "config": kw["config"], "cut": float(b["cut"]), "arm": b["arm"], "context": b["context"],
               "sourceTest": b["sourceTest"], "predict": b["predict"], "commit": commit,
               "torch": torch.__version__, "base": b["base"], "prepared": b["prepared"]}
        torch.save(rec, f"{out}/{unit_key}-s{sd}.pt")
        print(unit_key, sd, "cut", round(b["cut"], 4), "predictBA", b["predict"]["BA"], round(time.time() - t0, 1), "s", flush=True)
print("BASES_DONE", unit_key, flush=True)
