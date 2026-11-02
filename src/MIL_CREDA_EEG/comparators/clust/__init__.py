"""Explicit integration of CLUST's source-free controlled variant.

Import has no registration side effects; parent owns comparator discovery.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import torch

from MIL_CREDA_EEG import model_store, search, two_phase
from MIL_CREDA_EEG.mil_wrapper import CommonMILModel
from .training import NAME, PARAMETERS, adapt_target_phase, restore_model

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v40.md",
    "sections": ["4.5", "7.3", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["clust_search_reuses_common_checkpoint_and_never_reads_source_data",
                   "clust_registration_is_explicit_and_uses_generic_spaces",
                   "clust_store_checks_comparator_code_version"],
}


def train_phase_two(data, rotation, seed, record, params, schedule, device="cpu"):
    """Shared search trainer protocol; only target train/validation accessed."""
    state, meta = record
    expected = {"phase": 1, "name": "common", "rotation": rotation, "seed": seed,
                "transfer": data.transfer.name, "partition_sha256": data.partition_sha256}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise search.SearchError("CLUST requires this rotation/seed/partition common checkpoint")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    channels = meta["channels"]
    model = CommonMILModel(channels["source"], channels["target"]).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("target_trunk.") for k in missing):
        raise search.SearchError("the phase-one state does not fit the common architecture")
    joint = two_phase.bridge(model)
    d = data.rotations[rotation]
    result = adapt_target_phase(model, d["target_train"], d["target_valid"], params,
                                search.check_schedule(schedule), generator, device,
                                seed=seed, fold=rotation, transfer=data.transfer.name)
    return {"strategy": NAME, "model": model, "result": result, "hyper": dict(params),
            "channels": channels, "layer_copy": joint["layer_copy"]}


def register():
    """Explicit registration. EMA uses allowed bounded [.5,.95] generic axis.

    The shared hook permits no arbitrary EMA distribution; this range excludes
    the paper's .999. REPLICATION.md discloses the controlled-search restriction.
    """
    search.register_phase_two(NAME, {
        "clusteringWeight": search.LOSS_WEIGHT, "consistencyWeight": search.LOSS_WEIGHT,
        "informationWeight": search.LOSS_WEIGHT, "confidence": search.CONFIDENCE_THRESHOLD,
        "emaMomentum": search.CONFIDENCE_THRESHOLD}, train_phase_two)


def code_version():
    """Content hash of this adapter, supplementing the common store code hash."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def save(root, *, trained, transfer, rotation, seed, partition_sha256, selection="primary"):
    """Shared store plus adapter provenance; use separate roots for the two rules."""
    if selection not in ("primary", "secondary"):
        raise ValueError("selection must be primary or secondary")
    checkpoint = trained["result"]["checkpoint" if selection == "primary" else "secondary_checkpoint"]
    return model_store.save(root, transfer=transfer, rotation=rotation, seed=seed, phase=2,
                            name=NAME, state=checkpoint["state"], hyper=checkpoint["hyper"],
                            partition_sha256=partition_sha256, channels=trained["channels"],
                            extra={"clust_code_version": code_version(), "selection": selection,
                                   "epoch": checkpoint["epoch"],
                                   "valid_balanced_accuracy": checkpoint["valid_balanced_accuracy"],
                                   "label_free_score": checkpoint["label_free_score"]})


def load(path, **kwargs):
    """Shared partition/state checks plus adapter code check and reconstruction."""
    state, meta = model_store.load(path, **kwargs)
    if meta.get("name") != NAME or meta.get("phase") != 2:
        raise model_store.StoreError("not a CLUST phase-two model")
    if meta.get("clust_code_version") != code_version():
        raise model_store.CodeVersionMismatch("CLUST adapter code differs")
    return restore_model(state, meta["channels"]["source"], meta["channels"]["target"]), meta
