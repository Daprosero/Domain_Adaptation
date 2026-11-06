"""Seeded synthetic CPU smoke, not a reproduction of CASUAL's published results."""
import json

import torch

from MIL_CREDA_EEG import two_phase
from MIL_CREDA_EEG.mil_wrapper import CommonMILModel, COMMON_PHASE_ONE_FIXED
from .training import adapt_target_phase

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v56.md",
    "sections": ["7.2", "7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["casual_smoke_runs_common_phase_one_and_cpu_adaptation"],
}


def run(seed=20261102):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    device = torch.device("cpu")

    def bags(channels, prefix):
        return [two_phase.BagTensor(f"{prefix}{i}", "synthetic", i % 2,
                                    torch.randn(3, channels, 512)) for i in range(4)]

    source, source_valid = bags(29, "s"), bags(29, "sv")
    target, target_valid = bags(14, "t"), bags(14, "tv")
    model = CommonMILModel(29, 14)
    schedule = dict(epochs1=1, epochs2=3, bagsPerStep=2)
    phase1 = two_phase.train_source_phase(model, source, source_valid,
        {**COMMON_PHASE_ONE_FIXED, **schedule, "lr1": 0.001, "wd1": 1e-5,
         "lr2": 0.001, "wd2": 1e-5, "tauLocal": 1.}, generator, device, seed=seed)
    two_phase.bridge(model)
    result = adapt_target_phase(model, source, target, target_valid,
        dict(lr2=0.001, wd2=1e-5, lambda_align=1., lambda_entropy=0.1,
             lambda_vat=0.1, vat_radius_fraction=0.5), schedule, seed=seed, device=device)
    return {"data": "seeded synthetic; no BALLADEER data present in worktree",
            "device": "cpu", "seed": seed, "source_shape": [4, 3, 29, 512],
            "target_shape": [4, 3, 14, 512], "phase1_epochs": phase1["epochs"],
            "phase1_loss": [r["loss"] for r in phase1["history"]],
            "phase2_epochs": result["epochs"],
            "phase2_loss": [r["loss"] for r in result["history"]],
            "phase2_vat": [r["vat"] for r in result["history"]],
            "typical_window_norm": result["history"][0]["typical_window_norm"],
            "vat_radius": result["history"][0]["vat_radius"],
            "best_epoch": result["best_epoch"]}


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
