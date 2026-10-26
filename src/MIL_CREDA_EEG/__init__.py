"""MIL_CREDA_EEG: the BALLADEER EEG data audit.

Experiments plan v02 (`experiments-plan-experimental-mil-creda-sobre-balladeer-v02.md`),
Secs. 2.1-2.6 and 2.10. One module per audit step, each declaring the section
it implements in its own `__provenance__`.

Kept apart from `MIL_CREDA`, which holds the mathematical objects of the
proposal: this package reads and measures raw recordings and implements no
equation. It imports nothing from `MIL_CREDA`, so the audit runs without torch.
"""
