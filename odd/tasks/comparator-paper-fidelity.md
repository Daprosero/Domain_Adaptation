# Comparator paper fidelity: DALN + four fixes

Objective: add DALN as an official comparator and fix the four defects found when each vendored comparator was checked against its ingested paper (2026-10-15).
Branch: `two-encoders` (shared working tree; stage by name). Delivery: no commits until the user asks. Test runner: `.venv/bin/python -m pytest -q` from `implementations/Domain_Adaptation`.

## Specs

- **S1 — DALN.** "la primera implmentar DALN de acuerdo al paper siguiendo las isntrucciones de paper y de su repo si está." New `src/CREDA/comparators/daln/` following the vendored pattern (official repo pinned by commit, `upstream_*.py` with citation header, `adapter.py` exposing the `method(f_s, f_t, g_s, g_t, y_s, **extra) -> 0-d tensor` contract, alignment term only, no own discriminator: NWD reuses the shared head h). Paper: `guidance/_tmp_comparator_validation/daln/` (arXiv 2204.03838).
- **S2 — SHOT.** "Que el adaptador reciba los índices del lote para las pseudo-etiquetas." Whole-dataset labels from `obtain_label` must not be silently applied as the first B labels.
- **S3 — CDAN+E.** "Usar el mapa aleatorio que pide la Ec. 8 del paper y corregir su bug en GPU/MPS." Randomized multilinear map when d_f·d_g > 4096 (here 256·31 = 7936); `RandomLayer.random_matrix` must follow `.to(device)` and live in `state_dict`.
- **S4 — JUMBOT.** "Poner los hiperparámetros por dataset del paper y aceptar `**extra`." Paper values: Office-Home η1=0.01 η2=0.5 τ=0.5 ε=0.01; VisDA η1=0.005 η2=1 τ=0.3 ε=0.01; digits η1=0.1 η2=0.1 τ=1 ε=0.1. Assumption: default preset is Office-Home (the paper has no Office-31 setting); explicit kwargs override.
- **S5 — PFAN.** "Darle al discriminador su propio optimizador y corregir la documentación." PFAN uses G = −D, not a GRL: its discriminator must not go into the main optimiser; the discriminator step needs its own backward. Document that T=1.8 has no effect under the protocol.
- Out of scope (user, 2026-10-16): "en la tabla queda como está" — no "parámetros propios" column in the funnel/artifact.

## Tasks

| ID | Specs | Route | Status | Evidence |
|---|---|---|---|---|
| T1 | S1 | delegated (parallel unit) | done (registry in T6) | xiaoachen98/DALN@e40ca55 nwd.py+grl.py vendored; caller passes `head=h`; RED ImportError → GREEN 21 passed; mutation (sign) caught; parent re-ran daln file |
| T2 | S2 | delegated (parallel unit) | done | RED 3 failed/1 passed → GREEN 19 passed (shot_indices + official); parent re-ran shot_indices; adapter takes `tar_idx` | 
| T3 | S3 | delegated (parallel unit) | done | use_random auto by Eq. 8 (2,100,225 params at 256x31); random matrices as buffers (LOCAL EDIT), MPS forward ran; RED 9 failed → GREEN 28 passed; parent re-ran cdan_random file |
| T4 | S4 | delegated (parallel unit) | done | presets verified vs PDF p.21 + train_jumbot.py L419-425; RED 10 failed/1 passed → GREEN 26 passed; parent re-ran presets file |
| T5 | S5 | delegated (parallel unit) | done | generator_parameters/discriminator_param_groups/discriminator_step on detached features; forward value unchanged; RED 7 failed → GREEN 23 passed; 2 mutations caught |
| T6 | S1–S5 | inline (parent: shared files `comparators/__init__.py`, `CITATIONS.md`, registry test; full suite seam check; delete temp ingestion of validated papers) | done | registry test RED (1 failed) → GREEN 16 passed; 58/58 comparator tests; full suite 1559 passed, 3 failed — none caused here (see L4); temp ingestion deleted |

Route evidence: five independent units with disjoint edit surfaces (one comparator folder + one new test file each), launched together; shared files owned by the parent.

## Log

- L1 (2026-10-16, user): "abre tu bloque de tareas, la primera implmentar  DALN de acuerdo al paper siguiendo las isntrucciones de paper y de su repo si está. una tarea por cada arrreglo que se menciona y que yo pueda ver cada tarea como va"
- L2: validation reports (2026-10-15) — DSAN/LMMD and CS-DA clean (temp deleted); SHOT, CDAN+E, JUMBOT, PFAN have the defects in S2–S5.
- L3 (2026-10-16): T1–T5 delegated in parallel, each RED→GREEN, parent re-ran each new test file. T6 inline: DALN registered, registry docstring rewritten (PFAN separate optimiser, DALN head=, SHOT tar_idx), `build` docstring notes paper-over-code defaults, CITATIONS.md gains DALN row/BibTeX and a local-deviations section.
- L4 (2026-10-16): full suite 1559 passed / 3 failed. `test_every_module_under_the_package_declares_a_provenance` fails on all 17 comparator modules (14 pre-existing since the package was vendored on 2026-10-12, plus 3 new DALN files): the comparators never carried `__provenance__`. Open decision for the user. The other two failures (`toy_plots` invariants without tests; hand-typed numbers in notebook prose) are in files this feature did not touch.
- L5: no commits made; user decides delivery.
- L6 (2026-10-16, user): "si" — to: add `__provenance__` to our comparator modules and exempt the verbatim `upstream_*.py` files from the rule.
- L7: `tests/test_creda_provenance.py` gains `_is_vendored` (only `comparators/*/upstream_*.py`) and `test_only_verbatim_comparator_files_are_exempt_from_the_rule` (exempt files must carry the citation header; every adapter stays enumerated; mutation widening the exemption to `adapter.py` made it fail, restored). 8 modules (7 adapters + `pfan_torch.py`) declare `claims_no_equation` (third-party baseline, paper equations named) with invariants pointing to existing tests. Full suite: 1562 passed, 1 failed — `test_every_declared_invariant_has_a_test_somewhere_under_tests`, only on `src/CREDA/toy_plots.py` invariants (not touched here).
