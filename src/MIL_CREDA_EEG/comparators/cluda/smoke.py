"""Seeded synthetic EEG-shape CPU execution check, never published-score evidence."""
import hashlib

import torch

from MIL_CREDA_EEG import mil_wrapper, two_phase
from .training import DEFAULT_PARAMS, train_phase_two
from MIL_CREDA_EEG import assignment, search

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v54.md",
    "sections": ["7.2", "7.3", "7.4"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["cluda_synthetic_two_phase_smoke"],
}


def run():
    """Synthetic fixture because existing lunas/rollo/toroide toys are not EEG tensors."""
    seed = int(hashlib.sha256(b"CLUDA:synthetic:rotation0").hexdigest()[:8], 16) % 2**31
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    def bags(prefix, channels):
        return [two_phase.BagTensor(f"{prefix}{i}", "synthetic", i % 2,
                                    torch.randn(2, channels, 512)+.2*(i % 2)) for i in range(8)]
    roles = {"source_train": bags("s", 29), "source_valid": bags("sv", 29),
             "target_train": bags("t", 14), "target_valid": bags("tv", 14)}
    data = search.SearchData(assignment.Transfer("device", "L6", "Robots"), "synthetic", {0: roles})
    model = mil_wrapper.CommonMILModel(29, 14)
    schedule = {"epochs1": 3, "epochs2": 5, "bagsPerStep": 8}
    hyper = search.phase_one_hyper("common", {"lr1": .001, "wd1": 0.}, schedule)
    phase1 = two_phase.train_source_phase(model, roles["source_train"], roles["source_valid"],
                                          hyper, torch.Generator().manual_seed(seed), torch.device("cpu"))
    record = (phase1["checkpoint"]["state"], {
        "phase": 1, "name": "common", "transfer": data.transfer.name,
        "rotation": 0, "seed": search.PHASE_ONE_SEED, "partition_sha256": "synthetic",
        "channels": data.channels(), "hyper": hyper,
    })
    result = train_phase_two(data, 0, seed, record, DEFAULT_PARAMS, schedule)["result"]
    print("seeded synthetic EEG CPU only; NOT paper numbers; seed="+str(seed))
    print("source=(8 bags,2 windows,29 channels,512); target=(8 bags,2 windows,14 channels,512)")
    print("phase1_loss", [round(r["loss"], 6) for r in phase1["history"]])
    print("phase2_loss", [round(r["loss"], 6) for r in result["history"]])
    print("contrastive", [round(r["contrastive"], 6) for r in result["history"]])
    print("source_source", [round(r["source_source"], 6) for r in result["history"]])
    print("step_anchors_without_positive", [r["step_anchors_without_positive"] for r in result["history"]])
    print("step_source_anchors_without_positive", [r["step_source_anchors_without_positive"] for r in result["history"]])
    print("source_zero_loss_pair_fraction", [r["source_zero_loss_pair_fraction"] for r in result["history"]])
    print("skipped_singleton_target_steps", [r["skipped_singleton_target_steps"] for r in result["history"]])
    print("step_target_pseudo_classes", [r["step_target_pseudo_classes"] for r in result["history"]])
    return result


if __name__ == "__main__":
    run()
