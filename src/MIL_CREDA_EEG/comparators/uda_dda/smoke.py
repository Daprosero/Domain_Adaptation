"""CPU-only synthetic BALLADEER-shape execution check, never a benchmark."""
import torch

from MIL_CREDA_EEG import mil_wrapper, two_phase
from .training import DEFAULTS, adapt

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v60.md",
    "sections": ["7.2", "7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["uda_synthetic_smoke_executes"],
}


def fixture(channels=29, target_channels=14, seed=13):
    """Unit seed 13; disjoint subjects, two windows/bag, 512 samples/window."""
    torch.manual_seed(seed)
    def bags(prefix, c):
        return [two_phase.BagTensor(f"{prefix}{i}", prefix, i % 2,
                                    torch.randn(2, c, 512) + 0.2 * (i % 2)) for i in range(4)]
    return (mil_wrapper.CommonMILModel(channels, target_channels),
            bags("source", channels), bags("source-valid", channels),
            bags("target", target_channels), bags("target-valid", target_channels))


def run(epochs2=10):
    """Ten phase-2 epochs visit all four Eq. 7 stages after rescaling (paper epochs 0..90)."""
    torch.set_num_threads(1)
    model, source, source_valid, target, valid = fixture()
    schedule = {"epochs1": 3, "epochs2": epochs2, "bagsPerStep": 4}
    hyper = {**mil_wrapper.COMMON_PHASE_ONE_FIXED, **schedule, "lr1": 0.001,
             "wd1": 0., "lr2": 0.001, "wd2": 0., "tauLocal": 1.}
    phase1 = two_phase.train_source_phase(model, source, source_valid, hyper,
                                          torch.Generator().manual_seed(13), torch.device("cpu"))
    result = adapt(model, source, target, valid, {**DEFAULTS, "lr2": 0.001, "wd2": 0.},
                   schedule, seed=13)
    h = result["history"]
    print("synthetic CPU only; seed=13; source=(4 bags,2 windows,29 channels,512); "
          "target=(4 bags,2 windows,14 channels,512)")
    print("phase1_loss", [round(r["loss"], 6) for r in phase1["history"]])
    print("phase2_loss", [round(r["loss"], 6) for r in h])
    print("phase2_mmd", [round(r["mmd"], 6) for r in h])
    print("phase2_cmmd", [round(r["cmmd"], 6) for r in h])
    print("source_ce", [round(r["source_ce"], 6) for r in h])
    print("alpha", [round(r["alpha"], 4) for r in h])
    print("stage", [r["stage"] for r in h], "threshold", [r["threshold"] for r in h],
          "selected", [round(r["selected"], 3) for r in h])
    print("sigma", [round(r["sigma"], 4) for r in h])
    return result


if __name__ == "__main__":
    run()
