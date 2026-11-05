"""CPU-only synthetic BALLADEER-shape execution check, never a benchmark."""
import torch

from MIL_CREDA_EEG import mil_wrapper, two_phase
from .training import DEFAULTS, adapt

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v54.md",
    "sections": ["7.2", "7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["cca_synthetic_smoke_executes"],
}


def fixture(channels=29, target_channels=14, seed=11):
    """Unit seed 11; disjoint subjects, two windows/bag, 512 samples/window."""
    torch.manual_seed(seed)
    def bags(prefix, c):
        return [two_phase.BagTensor(f"{prefix}{i}", prefix, i % 2,
                                    torch.randn(2, c, 512) + 0.2 * (i % 2)) for i in range(4)]
    return (mil_wrapper.CommonMILModel(channels, target_channels),
            bags("source", channels), bags("source-valid", channels),
            bags("target", target_channels), bags("target-valid", target_channels))


def run():
    torch.set_num_threads(1)
    model, source, source_valid, target, valid = fixture()
    schedule = {"epochs1": 1, "epochs2": 5, "bagsPerStep": 4}
    hyper = {**mil_wrapper.COMMON_PHASE_ONE_FIXED, **schedule, "lr1": 0.001,
             "wd1": 0., "lr2": 0.001, "wd2": 0., "tauLocal": 1.}
    phase1 = two_phase.train_source_phase(model, source, source_valid, hyper,
                                          torch.Generator().manual_seed(11), torch.device("cpu"))
    result = adapt(model, source, target, valid, {**DEFAULTS, "lr2": 0.001, "wd2": 0.},
                   schedule, seed=11)
    print("synthetic CPU only; seed=11; source=(4 bags,2 windows,29 channels,512); "
          "target=(4 bags,2 windows,14 channels,512)")
    print("phase1_loss", [round(h["loss"], 6) for h in phase1["history"]])
    print("phase2_loss", [round(h["loss"], 6) for h in result["history"]])
    print("calibrated", [h["calibrated"] for h in result["history"]],
          "target_prior", result["target_prior"].tolist())
    return result


if __name__ == "__main__":
    run()
