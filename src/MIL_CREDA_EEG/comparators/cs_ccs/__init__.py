"""CS+CCS (Yin et al., UAI 2024) controlled BALLADEER variant, not a reproduction.

Registration is explicit (`training.register()`); importing changes no registry.
"""
from .training import DEFAULTS, adapt, load_phase_two, register, train_phase_two

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v57.md",
    "sections": ["7.1", "7.4"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["cs_ccs_package_is_a_controlled_variant"],
}

__all__ = ["DEFAULTS", "adapt", "load_phase_two", "register", "train_phase_two"]
