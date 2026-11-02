"""CPU-only synthetic execution evidence, not an accuracy replication.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m MIL_CREDA_EEG.comparators.hasd.smoke
"""
from __future__ import annotations

import json

import torch

from MIL_CREDA_EEG import search, two_phase
from .model import HASDModel
from .training import begin_adaptation, train_phase_two

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v40.md",
    "sections": ["7.3", "7.4"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["hasd_synthetic_smoke_executes_both_phases"],
}


def run(seed: int = 20261102) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    # 4 s x 128 Hz, native unequal 29/14-channel device inputs, ragged bags.
    def bags(channels, prefix, labels=True):
        return [two_phase.BagTensor(f"{prefix}{i}", "synthetic", i % 2 if labels else -999,
                                   torch.randn(3 + i % 2, channels, 512)) for i in range(4)]

    source, source_valid = bags(29, "s"), bags(29, "sv")
    target, target_valid = bags(14, "t", False), bags(14, "tv")
    model = HASDModel(29, 14)
    schedule = {"epochs1": 2, "patience1": 1, "epochs2": 3, "bagsPerStep": 2}
    hyper1 = search.phase_one_hyper("HASD", {"lr1": .001, "wd1": 1e-6}, schedule)
    phase1 = two_phase.train_source_phase(model, source, source_valid, hyper1,
                                         torch.Generator().manual_seed(seed), torch.device("cpu"))
    context = begin_adaptation(model)
    params = {"lr2": 1e-4, "wd2": 1e-6, "lambda_sgir": .001,
              "lambda_pacr": .1, "epsilon": .01, "tau": .1}
    phase2 = train_phase_two(model, context, source, target, target_valid, params, schedule,
                             seed=seed, fold=0, transfer="synthetic29-to14")
    prediction = two_phase.evaluate(model, target_valid, "target", torch.device("cpu"))
    return {"data": "seeded synthetic, not BALLADEER observations", "device": "cpu",
            "seed": seed, "source_channels": 29, "target_channels": 14, "samples": 512,
            "phase1_loss": [h["loss"] for h in phase1["history"]],
            "phase2_loss": [h["loss"] for h in phase2["history"]],
            "phase2_das": [h["das"] for h in phase2["history"]],
            "phase2_sgir": [h["sgir"] for h in phase2["history"]],
            "phase2_pacr": [h["pacr"] for h in phase2["history"]],
            "prediction_shape": list(prediction.shape),
            "prediction_finite": bool(torch.isfinite(prediction).all()),
            "kept_initialization": context.layer_copy["kept_initialization"]}


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
