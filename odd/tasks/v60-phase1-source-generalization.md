# Phase-1 source generalization (toward plan v60)

Objective: EEGNet phase 1 that generalizes to unseen source subjects in every source, as it already does on Robots, so the adaptation strategies have a sound starting point. Branch: `main` (diagnostics only, outside plan v59, `code_version` unchanged). Runner: `.venv/bin/python -m pytest tests/test_phase_one_sweep.py`.

## Specs

- S1. Phase 1 is judged only by unseen-source generalization: "en datos de source generalice ese es la misión, la adaptación la hace las estrategias, si buscas que el .predict en objetivo se suba estás matando las estrategias de adaptación".
- S2. One backbone, EEGNet: "no quiero tener dos backbone"; ShallowConvNet discarded after an equal search.
- S3. Short tests only, inside the scenario: "la idea es que hagas pruebas cortas sin salirte de nuestro escenario para lograr esas generalización en la fase 1".
- S4. A structured, located account of each failure: "tener algo bien estructurado con ubicación acérrima de los fallos que no dejan generalizar al modelo, entiéndase eso como que en test de source tenga buenos resultados como en el que ya funciona".
- S5. Accounts: "usa las que tien menos de 2 horas de uso, te puedo dar libertad para que no bajen las cuentas de 20 horas, porque comprometemos la prueba grande".
- S6. No backbone comparisons outside the goal: "porque vas a comparar !??? no te salgas del objetivo generalizar en fuente".

## Tasks

| ID | Spec | Task | Route | Status |
|---|---|---|---|---|
| T1 | S1, S4 | Failure ledger per source with evidence | inline | open |
| T2 | S3 | CogniFit: bag size (sub-bags 72), input norm | inline, Kaggle | running |
| T3 | S3 | L11, L6: spatial capacity (D=1), input norm | inline, Kaggle | running |
| T4 | S3, S4 | Next located tests from the ledger | inline | open |

## Failure ledger

Unseen-source AUC, pooled over the 4 outer rotations. Permutation null for these fold sizes is about 0.50 +- 0.17 (5-95%).

| Source | Signal exists? (band-power LR, permutation p) | EEGNet + fixes | Ruled out | Open |
|---|---|---|---|---|
| Robots | yes (.71) | .71-.78, works | - | - |
| CogniFit | yes (.75, p .03) | .52-.54 | input scale (std, BN, LN), aggregation (6 kinds), Euclidean alignment, bag cap with old recipe, search (12 trials) | bag size with fixes, summary of windows |
| L11 | yes (.72, p .005) | .55-.58 | input scale, aggregation, D=1, search | why the linear model wins |
| L6 (device) | weak (.61, p .13) | .36-.40 | D=1 | signal strength |
| L1 | no (.50, p .47) | .35-.40 | partition (folds balanced, no shared subject) | excluded: no detectable signal |

Fixes in every EEGNet run: inputs standardized by the source training statistics, head max-norm lifted, explicit L2, BatchNorm recalibrated, sigma recomputed after phase 1.

## Log

- L1 (2026-11-08). User: "si buscas que el .predict en objetivo se suba estás matando las estrategias de adaptación"; phase 1 is judged by unseen-source generalization only.
- L2. Signal quality marks (lost, interpolated, flat, saturated) are near zero in every source and do not differ by class: ruled out.
- L3. D=1: L11 .573, L6 .43; no change.
