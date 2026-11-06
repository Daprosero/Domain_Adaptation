"""AdaptMIL's explicitly registered BALLADEER controlled variant (plan §7.4).

Call register() once in the parent runner. It wires the native phase-one CE
(train_source_phase) into search.train_phase_one through search.register_phase_one.
"""
from .model import AdaptMILModel, NativeABMIL
from .training import (adapt_target_phase, bridge, load_model, save_model,
                       train_phase_one, train_phase_two, train_source_phase)

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v55.md",
    "sections": ["7.1", "7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["adaptmil_registration_is_explicit_and_uses_shared_search_spaces"],
}


def register():
    """Register only through existing extension seams; no shared source file edits."""
    from MIL_CREDA_EEG import model_store, search
    model_store.BUILDERS["AdaptMIL"] = lambda source_channels, target_channels, hyper: (
        AdaptMILModel(source_channels, target_channels))
    search.register_phase_one("AdaptMIL", train_source_phase)
    search.register_phase_two("AdaptMIL", {"mmdWeight": search.LOSS_WEIGHT,
                                          "kernelMultiplier": search.KERNEL_MULTIPLIER},
                               train_phase_two)


__all__ = ["AdaptMILModel", "NativeABMIL", "register", "train_source_phase", "bridge",
           "adapt_target_phase", "train_phase_one", "train_phase_two", "save_model",
           "load_model"]
