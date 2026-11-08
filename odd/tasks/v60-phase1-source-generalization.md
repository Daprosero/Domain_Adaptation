# Phase-1 source generalization (experiments plan v60)

Objective: EEGNet phase 1 that generalizes to unseen source subjects in every source, as it already does on Robots, so the adaptation strategies have a sound starting point. Branch: `main` (diagnostics only, outside the plan's protocol; code bound to experiments plan v60 since L18). Runner: `.venv/bin/python -m pytest tests/test_phase_one_sweep.py`.

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
- L14. Class anchors in phase 2 (local, seeds 1/2/3, target keeps source standardization). Source CE through the target trunk: C->R .713/.625/.699, no help. Distillation to the start, mu 10, blend .5: C->R .801/.790/.816, R->C .790/.761/.772: a tie with predict.
- L15. Headroom (diagnostic only, chooses nothing): phase 2 trained with the target training labels, evaluated on unseen target subjects. Best oracle, lr2 1e-3, blend .5: R->C .827/.812/.812 vs predict .798/.765/.772 (+.04); C->R .776/.783/.798 vs .798/.794/.805, best oracle setting .805/.798/.820 (+.01). With a strong phase 1, even labelled adaptation of the target trunk gains at most about .04 AUC on the task pair; a label-free phase 2 that ties predict is close to that ceiling.
- L16. Device pair (different channels, the target trunk starts random; local, seeds 1/2/3, target AUC). L6->Robots: start .696/.460/.400, MIL-CREDA lr2 1e-3 .547/.531/.501, lr2 1e-2 .418/.717/.591, global only lr2 1e-2 .298/.580/.519, local only .427/.646/.606, labelled headroom (target trunk trained with target labels) .777/.772/.772. Robots->L6: start .497/.563/.647, MIL-CREDA lr2 1e-3 .493/.422/.509, lr2 1e-2 .443/.482/.575, headroom .627/.620/.636. There is headroom (.77 on L6->Robots); MIL-CREDA does not reach it and swings with the seed (.42-.72). The start's seed-1 .696 was luck, not signal. EEGNet v59 stored task-pair predictions (AUC, one seed, 33 subjects): C->R predict .761, MIL-CREDA .548, target-supervised upper bound .662; R->C .445/.485/.393.
- L17. True ceiling, spectral encoder, task pair (local, seeds 1/2/3, AUC on the same unseen target test subjects; full models trained from scratch). C->R: predict (source labels) .798/.794/.805, model trained with target labels .732/.721/.706, model trained with source+target labels .761/.757/.757. R->C: predict .798/.765/.772, target labels .706/.691/.665, source+target .805/.798/.812. On C->R no model beats predict, not even with every label; on R->C the labelled ideal gains about .03. The tie of the label-free phase 2 with predict on the task pair is at that ceiling. The same holds with EEGNet (L16): its target-supervised bound .662 sits below its predict .761 on C->R.
- L18 (2026-11-08). User: "hay una nueva versión de experimentos ... enlaza todo a esa versión me refiero al código y artefactos y comit"; "esta versión no cambia nada de código se puede migrar a esa versión sin tener que implementar nada. de igual esos modelos ... de la campaña completa no tienen validez." Experiments v60 adds only K4-conectividad (Sec. 6.4, optional, not implemented). Code and tests rebound from v59 to v60; historical notes (\"changed in plan v59\", the v59 launch tools and kernels) kept. code_version changes, so the stored v59 models raise CodeVersionMismatch; the user judged them without validity.
- L19 (2026-11-08). User: "con EEGNet quiero identificar si le falta épocas para salir de ese problema". EEGNet phase 1 (subbag_check recipe, BatchNorm recalibrated before each reading) for 500 epochs on Kaggle T4, seed 1, unseen source test AUC per epoch (pooled over 4 rotations), with 1 inner source split read per epoch (`eegnet_epochs.py`). Robots (control) ep10 .801, ep50 .761, ep100 .846, ep200 .790, ep500 .757: plateau by epoch ~10. L11 .513/.583/.617/.636/.616 (ep10/50/100/200/500), BA .48-.56, isolated peak .723 at ep189. L6 .380/.431/.491/.437/.420, best .541. CogniFit .496/.540/.460/.522/.544, isolated peak .665 at ep437. No source improves with more epochs; the failing sources stay near chance throughout. The 1-split inner validation (7-22 subjects) is too noisy to pick an epoch (it would pick L11 ep30 .568, CogniFit ep328 .574). Next: a learning curve over subjects (user: "necesitamos es sujetos no ventanas").
- L20. Reference ceilings, spectral encoder, seeds 1/2/3 (`references.py`), AUC / BA on unseen subjects. C->R: CogniFit model on CogniFit test .706/.691/.665 / .60-.63; Robots model on Robots test .732/.721/.706 / .665; predict .798/.794/.805 / .704/.675/.704; phase 2 with target labels .750/.750/.765 / .759; blended .5 .776/.783/.798 / .792/.761/.792. R->C: predict .798/.765/.772 / .669/.610/.669; labelled blended .5 .827/.812/.812 / .732/.669/.730. L6->Robots: L6 model .59 / .52; Robots model .742/.738/.745 / .69-.73; labelled phase 2 .777/.772/.772 / .71. Predict ranks the target subjects well but its 0.5 threshold is off: the BA headroom on the task pair is about .06-.09 while the AUC headroom is near zero.

