"""UDA-DDA controlled BALLADEER variant, not a reproduction of published results."""
from .training import DEFAULTS, adapt, register, train_phase_two

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v60.md",
    "sections": ["7.1", "7.4"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["uda_package_is_a_controlled_variant_and_import_registers_nothing"],
}

__all__ = ["DEFAULTS", "adapt", "register", "train_phase_two"]
