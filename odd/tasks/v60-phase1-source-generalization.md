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
| T4 | S3, S4 | Locate the failure inside the linear model | inline, local | done |
| T5 | S3 | Fixed band-pass temporal filters with log power (FBCNet structure) | inline, Kaggle | done, fails |
| T6 | S1, S3 | Spectral encoder with CAR and L2 inside MIL-CREDA (spectral_check.py) | inline, local + Kaggle | phase 1 solved in every source with signal |
| T7 | S1 | Phase 2 on top of the solved phase 1 | inline, local | done: ties predict at best |

## Failure ledger

Unseen-source AUC, pooled over the 4 outer rotations. Permutation null for these fold sizes is about 0.50 +- 0.17 (5-95%).

| Source | Signal exists? (band-power LR, permutation p) | EEGNet + fixes | Ruled out | Open |
|---|---|---|---|---|
| Robots | yes (.71) | .71-.78, works | - | - |
| CogniFit | yes (.75, p .03) | .52-.54 | input scale (std, BN, LN .64), aggregation (6 kinds), Euclidean alignment, bag size (sub-bags 72: .50, +LN .50), search (12 trials), log-power block 1 (.59) | learned temporal filters |
| L11 | yes (.72, p .005) | .55-.58 | input scale, aggregation, D=1 (.57, +LN .59), search, log-power block 1 (.53) | learned temporal filters |
| L6 (device) | weak (.61, p .13) | .36-.40 | D=1 (.43, +LN .47), log-power block 1 (.40) | signal strength |
| L1 | no (.50, p .47) | .35-.40 | partition (folds balanced, no shared subject) | excluded: no detectable signal |

Fixes in every EEGNet run: inputs standardized by the source training statistics, head max-norm lifted, explicit L2, BatchNorm recalibrated, sigma recomputed after phase 1.

## Log

- L1 (2026-11-08). User: "si buscas que el .predict en objetivo se suba estás matando las estrategias de adaptación"; phase 1 is judged by unseen-source generalization only.
- L2. Signal quality marks (lost, interpolated, flat, saturated) are near zero in every source and do not differ by class: ruled out.
- L3. D=1: L11 .573, L6 .43; no change.
- L4. Ablation of the logistic model, unseen-source AUC (CogniFit / L11 / Robots): subject log .750/.687/.713; window log .768/.665/.713; subject linear .676/.616/.739; window linear .559/.536/.493. Learning from windows needs log compression.
- L5. Log-power block 1 in EEGNet: Robots .636, CogniFit .588, L11 .526, L6 .399. Log compression alone does not fix it; the remaining difference with the linear model is learned versus fixed band filters.
- L6. Fixed band-pass EEGNet (FBCNet structure): Robots .665, CogniFit .471, L11 .498, L6 .489. Fixing the bands does not close the gap.
- L7. MIL-CREDA reading the logistic model's own features matches that model on L11 (.59-.60 vs .585): MIL is not the gap.
- L8. Preprocessing ablation of the logistic model (CogniFit / L11 / Robots / L6): full .761/.713/.724/.626; no common average .827/.597/.618/.643; no filter .717/.701/.699/.649; mean of log .713/.681/.732/.600; FFT .743/.705/.721/.642; none .787/.632/.662/.644. Common average reference matters on L11 and Robots; the plan uses no re-reference.
- L9. EEGNet with common average reference: Robots .699, CogniFit .445, L11 .512, L6 .468. EEGNet stays at chance outside Robots with every fix tried (standardization, free head, explicit L2, BatchNorm recalibration, input BN/LN, D=1, sub-bags, log-power block, fixed bands, CAR, a 12-trial search).
- L10. Phase-1 solution: MIL-CREDA with the spectral encoder (log absolute and relative band power per channel), windows filtered 1-45 Hz with common average reference (at most 60 per bag), features standardized by each domain's training windows, near-uniform attention, explicit L2 .01, lr1 1e-2, 100 epochs. Unseen-source AUC, seeds 1/2/3 (local): Robots .735/.743/.710, CogniFit .673/.658/.684, L11 .704/.436/.606, L6 .596/.568/.580, L1 -/.467/.441 (no signal). Kaggle seed 1 agrees: Robots .735 and .748, CogniFit .673, L11 .697, L6 .620, L1 .467. L11 is seed-sensitive (train AUC .68-.79): optimization under test.
- L11. Full flow, Kaggle seed 1, target AUC predict / MIL-CREDA: R->C .794/.721, C->R .827/.621, L11->L1 .436/.417, L1->L11 .551/.520, L6->Robots -/.637, Robots->L6 -/.444. Phase 2 lowers a strong predict again: next problem, outside phase 1.
- L12. Phase-1 recipe changed to lr1 3e-3, 200 epochs (L11 was seed-sensitive at 1e-2). Unseen-source AUC, seeds 1/2/3 (local): L11 .668/.689/.670, Robots .732/.721/.706, CogniFit .706/.691/.665, L6 .591/.607/.575. Stable on every source with signal; set in spectral_check HYPER.
- L13. Phase 2 on the solved phase 1 (local, seeds 1/2/3, target AUC). C->R predict .798/.794/.805; R->C .798/.765/.772. Standardizing the target by its own windows alone costs C->R .07 (.728/.735/.761). Every adaptation that moves the target trunk lowers AUC in proportion to how far it moves: lr2 1e-3 x20 C->R .621/.533/.507, local term only worst (.548/.515/.566). Best setting: target keeps the source standardization, global term only, lr2 1e-4, blended .5 with the start: C->R .809/.783/.801, R->C .798/.757/.776, a tie with predict, never above it. Located cause: the alignment objective does not preserve the class direction; gentle settings only shrink the damage.

