# Agreed for the toyset — bound to research-concept-r34.md

Every item here was settled with the operator, not derived from a plan of work.
An item that never reaches the code shows up as an unticked box rather than as a
silence. Nothing here is re-decided while implementing: a collision with any of
these is raised, not resolved in passing.

**This is not `MIL-CREDA/AGREED.md`.** That document is bound to r21 and governs
the DIGIT benchmark report, which is being retired. Its content stays in git at
`7b9f4a0` and is not repaired. This file governs the toyset and nothing else.

## Scope — two notebooks, and the rest is gone

- [ ] The toyset is `Toyset_Generator.ipynb` and `Toyset_Plot.ipynb`, and no other
      notebook exists under `MIL-CREDA/Notebooks/`. The generator trains and writes
      the artifact; the plot reads it and draws. **No number is recomputed in the
      plot notebook.**
- [ ] The code the retired notebooks reach and the toyset does not goes with them.
      The surface is measured before anything is deleted, never estimated.

## The unit of time is the EPOCH

- [x] Everything a reader sees is in epochs: the axes, the table columns, the row
      labels and the frame titles. Settled 2026-10-19 and applied 2026-10-20 over
      thirteen sites in both packages.
- [x] The step survives only as the index of the history array, and never reaches
      a reader. Where a guard must compare steps and a label must print epochs,
      the two stay separate rather than being collapsed.
- [x] `fronteras_w` says **`barrido`**, not step and not epoch: its points are a
      CONSTRUCTED walk over the weight frontier of Eq. (24), not training.
- [x] A phase-1 epoch is 5 steps and a phase-2 epoch is 16, each one bag per class.
      The two phases count epochs differently and the figures say which they mean.

## Phase 1

- [x] Trains the source encoder and the head. Nothing of the target.
- [x] Its whole objective is Eq. (21), normalized by `B_src` so every bag lands in
      `[0, 1)` and the three terms of Eq. (39) are readable on one scale.
- [x] The criterion is **source VALID accuracy** and never the role the table
      publishes. **The patience stays**: the signal is supervised, so stopping on
      it risks nothing.
- [x] Eq. (15)'s `||v_R||_1 <= 1` is re-imposed AFTER each optimizer step, never
      before: the rule counts optimizer steps.
- [x] The search runs 25 trials over six dimensions, and each trial's score is the
      MEAN over 3 normalizations x 3 search seeds. Ties within `TIE_WIDTH = 0.01`
      are broken by the valid loss, which still discriminates once the accuracy is
      against its ceiling.
- [x] Search seeds (100, 101, 102) and report seeds (0..9) are DISJOINT. The
      hyperparameters never see the material their number is measured on.
- [x] The budget is reported rather than defended: 25 trials over 6 dimensions is
      4.17 per dimension, and a gap smaller than the variance between two searches
      of this budget is not a reading of the method.

## The bridge — four acts, in this order

- [x] The WEIGHTS of the best epoch are restored, not its number, and a second
      reading proves the load landed.
- [x] The frozen bandwidth is measured here, over the source training role ENTIRE,
      and a truncated reading is refused rather than tolerated.
- [x] Everything but the target encoder is frozen. **Every guarantee phase 2 offers
      rests on this act.**
- [x] `theta_s -> theta_t` is copied exactly once, WITH the buffers, or a BatchNorm
      `theta_t` would not be bit-identical to the encoder it claims to start as.

## Phase 2

- [x] Moves `theta_t` alone.
- [x] Minimizes `lambda_glob * L_glob (Eqs. 32-37) + lambda_loc * L_loc (Eqs. 30-31, 38)`
      and NOTHING else. Both coefficients are exactly 1, fixed, with no ramp.
- [x] Eq. (21) is computed, recorded as `supervised` and drawn, and **summed into
      nothing**. This is the objective, not a flag: there is no pole where it comes
      back. Measured parameter by parameter: the optimizer's step is bit-identical
      with and without it.
- [x] The confidence weighting is ON. `sopesado` OFF in Eq. (39) means the two
      COEFFICIENTS at 1, which is a different thing and was confused once.
- [x] The criterion is `max(cef)`, which reads no target label. **No early stop
      since 2026-10-19**: it runs its ceiling whole and chooses looking back.
- [x] The balance of `K_mix` is by TRACE and the bounds are the entropy-cosh pair,
      because that is what r34 states -- not because CREDA does it. Eq. (36)
      divides by the literal `ln 2` that r30 proved.
- [x] Eq. (37) aggregates over EVERY class with `n_c^s / N_s`, weights the source
      LABELS observed, and charges `1` to a class with no active pair. Excluding it
      rewarded collapse.
- [x] Eq. (38) is the uniform mean over the target bags, `(1/n_t) sum_j d_j^2 / 2`.
      The confidence left this aggregation in r32: `w_j^t` is differentiable, so a
      weighted mean could be lowered on the worst-aligned subjects instead of
      aligning them.

## The bandwidth

- [x] ONE bandwidth serves two jobs -- within-bag (Eqs. 14-15) and between-bags
      (Eqs. 32-37) -- and no value is in band for both. That is Decision 1's
      consequence: the range is widened to span both and the SEARCH chooses the
      trade-off. Which of the two is larger is not fixed across families.
- [x] The frozen bandwidth is source-only, so nothing injected into the target can
      move it. Damage entering through the class block stops being confused with
      damage entering through the unit of measurement.
- [ ] **Phase 1's bandwidth is measured on an UNTRAINED encoder and held fixed for
      the whole phase.** Measured over the 90 cells of the full run, the space then
      moves by a median of 2.46x, up to 31.21x, and in 27 of 90 cells it CONTRACTS.
      The operator proposes letting it move during phase 1 and ending with the
      frozen one. r34 declares `sigma in R+` and says nothing about its estimation
      or constancy -- it says "fijo durante el entrenamiento" of `gamma` and not of
      `sigma` -- so this contradicts nothing. **Open:** the cadence, and the
      consequence below.

## The sections of the toyset

- [x] Section 1 is the method against its rungs, one figure, one table and the
      declared tests per family block -- CREDA's shape, so the two documents can be
      read side by side.
- [ ] **Section 2 perturbs a BAG, not a point.** `w_j` is defined per bag (Eq. 24),
      so a bag-level perturbation enters exactly where the weighting can see it;
      perturbing single points dilutes the damage across a weight that is not
      defined at that scale. The table structure mirrors CREDA's.
- [ ] **Section 3 varies the ACTIVATION on the best architecture**, with the same
      tables and the same tests. The best architecture is read from the existing
      record BEFORE the activations run, never chosen after seeing their result. It
      is not the same in the three families (`lunas` none, `toroide` and `rollo`
      BatchNorm), so either one is fixed and said, or one runs per family.
- [ ] **Section 4 is the ablation**: local only, global only, and both. `config.ARMS`
      already declares the floor (`B`), global unweighted (`E`), global weighted
      (`F`) and the full method (`G`); **local-only is the one that does not exist**
      and has to be declared.
- [ ] **Section 5 validates the bag-to-bag correspondence.** Eq. (29) IS the
      bag-to-bag relation and `latent.correspondence_grid` already draws it,
      choosing its columns by MECHANISM so the figure can come out wrong. What is
      missing is a quantitative reading, and it must be scored against something
      the method's own kernel never evaluates -- the true class, and proximity in
      the INPUT space -- or a collapse would read as clean correspondence.

## Open -- nobody has decided these

- [ ] `MIN_EPOCAS_2 = 5` (CREDA) against `MIN_EPOCHS_2 = 3` (MIL): both declared,
      deliberately not equalized, still open.
- [ ] A moving `sigma` during phase 1 moves the scale of Eq. (15)'s consensus, and
      `gamma` -- which r34 declares FIXED during training -- would stop meaning the
      same thing across steps. **This one goes to the deliberator**: not "is it
      allowed", but "does fixed `gamma` with moving `sigma` still say what Eq. (15)
      says".
- [ ] The two-arm contamination design cannot tell "the weighting does not help"
      from "there was nothing to gain here", because there is no upper reference.
      Making the oracle arm survive changes what it computes, which is a claim.

## Ladder

- [x] `.predict` -- the same run read at its step-0 checkpoint. No adaptation.
- [x] `AdaBN` -- that checkpoint with the target's normalization statistics
      re-estimated. No optimizer, no label.
- [x] `MIL-CREDA` -- the full method, global and local, weighting on.

The first two cost no second sweep: `run_cell` measures them itself. A rung that
reads `y_t` is a CEILING and never a competing method, and a table that prints
one says so.
