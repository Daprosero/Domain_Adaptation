"""CCA-LSC controlled BALLADEER variant, not a reproduction of published results."""
from .training import DEFAULTS, adapt, register, train_phase_two

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v59.md",
    "sections": ["7.1", "7.4"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["cca_package_is_a_controlled_variant"],
}

__all__ = ["DEFAULTS", "adapt", "register", "train_phase_two"]
