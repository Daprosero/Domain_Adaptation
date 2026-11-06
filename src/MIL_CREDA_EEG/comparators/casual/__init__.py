"""CASUAL controlled variant; explicit registration, no import-time side effects."""
from .training import (DomainDiscriminator, adapt_target_phase, alternating_step,
                       load_final_model, register, save_final_model, train_phase_two)

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v56.md",
    "sections": ["7.1", "7.4"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["casual_import_does_not_register_or_fork_common_model"],
}
