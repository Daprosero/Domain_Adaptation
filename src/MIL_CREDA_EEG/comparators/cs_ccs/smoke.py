"""CPU-only synthetic BALLADEER-shape execution check, never a benchmark."""
import hashlib

import torch

from MIL_CREDA_EEG import mil_wrapper, two_phase
from .training import DEFAULTS, adapt

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v57.md",
    "sections": ["7.2", "7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["cs_ccs_synthetic_smoke_executes"],
}

UNIT = "cs_ccs-synthetic-device-rotation0"
#: Seed derived from the unit id: int(sha256(UNIT)[:8], 16) mod 2**31.
SEED = int(hashlib.sha256(UNIT.encode()).hexdigest()[:8], 16) % 2 ** 31


def fixture(channels=29, target_channels=14, seed=SEED, bags_per_role=4):
    """Disjoint subjects, two windows per bag, 512 samples per window."""
    torch.manual_seed(seed)

    def bags(prefix, c):
        return [two_phase.BagTensor(f"{prefix}{i}", prefix, i % 2,
                                    torch.randn(2, c, 512) + 0.2 * (i % 2))
                for i in range(bags_per_role)]
    return (mil_wrapper.CommonMILModel(channels, target_channels),
            bags("source", channels), bags("source-valid", channels),
            bags("target", target_channels), bags("target-valid", target_channels))


def run(epochs1=3, epochs2=3):
    torch.set_num_threads(1)
    model, source, source_valid, target, valid = fixture()
    schedule = {"epochs1": epochs1, "epochs2": epochs2, "bagsPerStep": 4}
    hyper = {**mil_wrapper.COMMON_PHASE_ONE_FIXED, **schedule, "lr1": 0.001,
             "wd1": 0., "lr2": 0.001, "wd2": 0., "tauLocal": 1.}
    phase1 = two_phase.train_source_phase(model, source, source_valid, hyper,
                                          torch.Generator().manual_seed(SEED),
                                          torch.device("cpu"))
    result = adapt(model, source, target, valid, {**DEFAULTS, "lr2": 0.001, "wd2": 0.},
                   schedule, seed=SEED)
    print(f"synthetic CPU only; unit={UNIT}; seed={SEED}; "
          "source=(4 bags,2 windows,29 channels,512); target=(4 bags,2 windows,14 channels,512)")
    print("phase1_loss", [round(h["loss"], 6) for h in phase1["history"]])
    for key in ("loss", "source_ce", "cs", "ccs"):
        print(f"phase2_{key}", [round(h[key], 6) for h in result["history"]])
    print("bandwidths", result["bandwidths"], "best_epoch", result["best_epoch"])
    return phase1, result


if __name__ == "__main__":
    run()
