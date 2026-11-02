"""HASD controlled BALLADEER variant; explicit, reversible registry integration."""
from .model import HASDModel

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v40.md",
    "sections": ["7.1", "7.3", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["hasd_registration_uses_existing_store_and_search_hooks"],
}


def register():
    """Opt in before shared search/model-store use; never mutate registries on import."""
    from MIL_CREDA_EEG import model_store, search
    from .training import EXTRA_KEYS, train_search_phase_two

    model_store.BUILDERS["HASD"] = lambda source, target, hyper: HASDModel(source, target)
    search.register_phase_two("HASD", {k: search.LOSS_WEIGHT for k in EXTRA_KEYS},
                              train_search_phase_two)


__all__ = ["HASDModel", "register"]
