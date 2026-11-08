# Phase-1 and phase-2 diagnostics (toward plan v60)

Local diagnostic scripts behind the failure ledger `odd/tasks/v60-phase1-source-generalization.md`.
They sit outside the plan's code files and choose nothing: every number is read on unseen subjects.
All of them use the spectral encoder recipe in `src/MIL_CREDA_EEG/spectral_check.py` (log absolute and
relative band power, 1-45 Hz, common average reference, at most 60 windows per bag, per-domain
standardization, explicit L2 .01, lr1 3e-3, 200 epochs).

Run from the repository root, with the data in `MIL-CREDA-EEG/Data/processed`:

    zsh tools/diagnostics/run.sh <script.py> 'CogniFit->Robots' 'Robots->CogniFit'

Each run prints one `RESULT seed <s> <transfer> {...}` line with pooled AUC over the 4 outer rotations.

| Script | Question | Ledger |
|---|---|---|
| `specopt.py` | Phase-1 unseen-source AUC per source (`LR1`, `EP1` environment variables; args: L2, seed, transfers) | L10, L12 |
| `p2spec.py` | Phase 2 variants: lr2, epochs, global or local term only, blend .5 with the start | L13 |
| `p2src.py` | The same, with the target kept at the source standardization | L13 |
| `p2cls.py` | Class anchors in phase 2: source CE through the target trunk, distillation to the start | L14 |
| `p2orc.py` | Phase-2 headroom: the target trunk trained with target labels (diagnostic only) | L15 |
| `p2dev.py` | Device pair (different channels): random target trunk start, phase 2, its terms, labelled headroom | L16 |
| `ceiling.py` | True ceiling: predict vs a model trained with target labels vs source+target labels, same test subjects | L16 |

`memory_guard.sh <pid> <GB>` stops a process whose physical footprint passes the limit (macOS `vmmap`).
