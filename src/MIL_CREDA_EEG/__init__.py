"""MIL_CREDA_EEG: the BALLADEER EEG data audit, preprocessing and data units.

Experiments plan v56 (`experiments-plan-experimental-mil-creda-sobre-balladeer-v56.md`;
the audit was first written against v02), Secs. 2.1-2.6 and 2.10, for the audit. One
module per audit step, each declaring the revision and section it implements in its
own `__provenance__`. The data units -- the four-fold partition (`partition`), bags
(`bag_data`), domain assignment per transfer (`assignment`) and the re-measured
separability (`separability`) -- declare theirs too; `partition` stays bound to v50,
the revision its frozen document was regenerated under, because that document's
identity hashes the module's own bytes.

Kept apart from `MIL_CREDA`, which holds the mathematical objects of the
proposal: the audit, preprocessing and data-unit modules read and measure raw
recordings, implement no equation and import nothing from `MIL_CREDA`, so they
run without torch.

Three modules (Sec. 7.3) are the exception, and only they import
torch and `MIL_CREDA`: `eegnet` (EEGNet-8,2 as trunk and common head, and the
phase-2 layer copy), `mil_creda_model` (two trunks, one shared attention, one
head, wired to the proposal's equations without re-implementing them) and
`two_phase` (phase 1, bridge and phase 2 for one transfer, rotation and seed).
Four more build on them (Secs. 4.5, 7.2, 7.3, 7.5, 7.7): `mil_wrapper` (the
common MIL wrapper of the non-MIL comparators, Eq. 15 at gamma = 0),
`controls` (source-only references and the target-supervised upper bound),
`search` (the Optuna studies of both phases) and `model_store` (every final
model with its hyperparameters, partition hash and code version). Nothing in
this file imports them, so importing the package still needs no torch.
"""
