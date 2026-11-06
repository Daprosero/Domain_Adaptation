"""CPU-only seeded synthetic BALLADEER smoke, not reproduced paper results."""
import torch
from MIL_CREDA_EEG import search, two_phase
from MIL_CREDA_EEG.mil_wrapper import CommonMILModel
from .training import adapt_target_phase

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v55.md",
    "sections": ["7.3", "7.7"], "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["clust_smoke_uses_seeded_balladeer_shapes_on_cpu"],
}


def run(seed=17):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = CommonMILModel(29, 14)
    schedule = {"epochs1": 1, "epochs2": 3, "bagsPerStep": 4}
    def bags(prefix, channels):
        return [two_phase.BagTensor(f"{prefix}{i}", prefix, i % 2,
                                    torch.randn(2 + i % 2, channels, 512, generator=generator))
                for i in range(4)]
    # Exercise the actual shared supervised phase before adaptation.
    full = search.phase_one_hyper("common", {"lr1": .001, "wd1": .0001}, schedule)
    phase1 = two_phase.train_source_phase(model, bags("source-train", 29),
                                          bags("source-valid", 29), full, generator,
                                          torch.device("cpu"), seed=seed)
    joint = two_phase.bridge(model)
    params = {"lr2": .001, "wd2": .0001, "clusteringWeight": 1,
              "consistencyWeight": 1, "informationWeight": 1,
              "confidence": .6, "emaMomentumComplement": .001}
    result = adapt_target_phase(model, bags("target-train", 14), bags("target-valid", 14),
                                params, schedule, generator, seed=seed)
    output = {"material": "seeded synthetic; no worktree toy data", "seed": seed,
              "device": "cpu", "shape": "ragged (2/3,29/14,512)",
              "phase1_loss": [h["loss"] for h in phase1["history"]],
              "phase2_loss": [h["loss"] for h in result["history"]],
              "selected_epoch": result["best_epoch"], "layer_copy": joint["layer_copy"]}
    print(output)
    return output


if __name__ == "__main__":
    run()
