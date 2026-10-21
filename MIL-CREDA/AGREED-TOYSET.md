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

**Only section 1 searches.** From section 2 on, every section reads the
hyperparameters section 1 chose and runs its own tests. Nothing re-searches, so
no later section can win by being tuned harder than the one before it.

**Every section from 2 on runs on the BEST ARCHITECTURE only**, read from the
record before it runs and never picked after seeing a result. Measured on the
full run: `lunas` none (0.4056), `toroide` BatchNorm (0.7944), `rollo` BatchNorm
(0.3944) -- it is not the same in the three families, so either one is fixed and
said out loud, or one runs per family.

- [x] **Section 1 -- the method against its rungs.** Three families, three
      architectures, ten seeds. One figure, one table and the declared tests per
      family block, which is CREDA's shape so the two documents can be read side
      by side. The only section that searches.

- [ ] **Section 2 -- the weighting.** APPROVED 2026-10-20, after reading what
      CREDA's own section 2 emits rather than inferring it. THREE emissions, in
      CREDA's own order, and the section is these and nothing else. (This line
      said "Four" from the day it was written and the list under it has always
      had three; the count was never the agreement, the list was.)

      1. **The gif.** Three BAGS walk from a decision frontier of the SOURCE head
         to the centre of a class, and `w_j` is read along the walk. **It runs
         nothing**: `w = 1 - H2(g)/ln C` is a function of the posterior and
         nothing else, so the curve is a property of Eq. (24) and the already
         trained head -- exact, with no seeds and no variance. Two panels: the
         source latent with the three partial paths, and `w` against the
         distance walked. The curve is PARTIAL per frame, because what this
         shows is that `w` rises WHILE the bag walks, not that it ends at one.
         The three start in DIFFERENT places on purpose -- double frontier,
         TRIPLE frontier, class centre -- because the triple is the only place
         where `w` truly vanishes, and from a three-way tie the bag can go to
         ANY class and `w` rises all the same: nobody asked it its label.
         Its unit is a **barrido**, never a step and never an epoch (operator,
         2026-10-19): these are the points of a CONSTRUCTED walk, not optimizer
         updates and not passes over the data.
      2. **The accuracy table** under contamination at a single level, with the
         clean base left out -- "sin la base limpia, eso ya es la seccion 1".
         CREDA's five columns: method, the epoch it chose, source accuracy,
         target accuracy, final watchman.
      3. **The test table**: ONE declared pair, `con sopesado - sin sopesado`,
         with CREDA's ten columns. One pair, so **no Holm**: there is no family
         to spread over.

      **The contamination mechanism, measured on 2026-10-20 and NOT what was
      first proposed.** Three findings, in the order they bite:

      1. **MIL's own declared label noise is structurally unreachable here.**
         `NOISE_PAIRING` must be an involution with no fixed point -- five
         mutual pairs -- asserted by its own test, and `bags.py` refuses
         otherwise. The toyset has THREE classes, and on an odd number a
         fixed-point-free involution does not exist. So `NOISE_ROLES`,
         the reserve and the donor class never enter this section.
      2. **CREDA's contamination is not label noise either.** `contaminar`
         RELOCATES a fraction of the target train geometrically, keeping the
         label: `x_j <- x_j - mu_{y_j} + mu_{c'}`. The point becomes an
         OUTLIER, not a label error. This section calls that function rather
         than copying its arithmetic, so the geometry, the determinism and its
         own exit assertion are CREDA's.
      3. **It runs ONE mode, and the other is unreachable.** CREDA declares two
         perturbations at one rho and REFUSES to pick one by default
         (`two_perturbations_at_one_rho_refuse_to_pick_one_by_default`),
         because they predict opposite things. `frontera` needs a `pesar(X)`
         returning `(w, routing)` over POINTS, and here `w_j` is defined on
         Eq. (20), the head's distribution over a BAG representation --
         `lectura_de_bolsas` refuses a non-bag arm for that written reason. A
         per-point `w` does not exist. Its bag-level counterpart is moving a
         bag to the frontier, which IS `desplazar_bolsas`, the ladder retired
         the same day.

      **So this section runs `reubicado` at rho = 0.30, the mode in which the
      weighting is NOT predicted to help**, and that has to be read in the
      table: the relocated point is left CONFIDENTLY WRONG, so its `w` does not
      drop and the weighting cannot protect it. **A null result here is
      `reubicado`'s own prediction, not a failure of the measurement.**

      Only the target TRAIN is contaminated, which is the only reason the
      accuracy is legible -- the metric stays on data nobody moved. The draw is
      not assumed to be unchanged: `barrido_sopesado` compares the three role
      splits and `members` before and after, because `bags.build` seeds from
      `(seed, code)` and sorts on LABELS, so changing coordinates must not move
      it. Without that comparison "only the train" is an intention.

      **AdaBN has NO row here, and that is the operator's ruling (2026-10-20):**
      AdaBN exists only where there are buffers to re-estimate, and this section
      runs on the BEST architecture, which is not necessarily BatchNorm. A row
      that sometimes does not exist is not a row.

      **What this supersedes.** The displacement ladder built earlier that day --
      `ESCALERA_DESPLAZAMIENTO`, `desplazar_bolsas`, `barrido_desplazamiento` --
      is NOT this figure. It retrains a whole cell per rung; CREDA's figure
      retrains nothing. It is superseded here rather than deleted, because
      whether it is retired or reused somewhere is not this document's call.

- [ ] **Section 3 -- is the aggregation the best one?** Two axes:
      - mechanism: `ours`, `abmil-published`, `abmil-gated`, `max`, `mean`.
        `wiring.MECHANISMS` already implements all five.
      - selection: all instances, `top-k`, `arbitrary-k`, `regular-k`.
      **`top-k` against `arbitrary-k` at the SAME `k` is the clean test**: if the
      attention knows which instances matter, keeping its best `k` has to beat
      keeping `k` at random. If they tie, the attention is scoring nothing even
      with the mechanism switched on, and `regular-k` is the third foot that
      separates "top-k wins" from "fewer instances already helps".
      **`k` is declared BEFORE the run: 5, 10 and 15 of 30, and all three are
      reported, never the best.** A `k` chosen after seeing the result chooses
      the number that gives the answer we wanted.
      Not the full cross: the five mechanisms over all instances, then the four
      selections over the winning mechanism.

      **APPROVED 2026-10-20, as FOUR tables and no figure yet.** `cef` is out of
      all of them -- it is not what this section asks.

      1. accuracy over the five mechanisms, all instances. CREDA's five columns.
      2. its test: `ours` against each of the other four, Holm over that family
         of 4. The order is `MECANISMOS`' own declaration order and NEVER the
         measured effect -- the first draft of this table sorted by effect size,
         which is exactly what CREDA's "the pair order does not follow the
         winner" forbids: a sign that always comes out positive says nothing.
         The ten pairs among the five exist and are NOT the family.
      3. accuracy over the four selection rules at ONE budget.
      4. its test: the three pairs between rules, Holm over that family of 3.

      **`k = 10` of 30 and nothing else** (operator, 2026-10-20), taken before
      any full run -- which is the whole of what keeps it honest. Cutting from
      three budgets to one removes the family over budgets; what it costs is
      that the section can no longer say "the effect holds across budgets".
      And the three pairs between rules bring a family of 3 back in the other
      axis: at m=3 the smallest threshold is 0.0167, so an effect at p=0.0195
      does NOT survive, and the table has to be able to say that.

      **What each of those three pairs buys, and the third one was measured:**
      `topk` vs `arbitrary` is the section's question; `topk` vs `regular` is
      the harder control, because `regular` carries no draw noise, so a `topk`
      that only beat `arbitrary` through noisy draws falls here; `regular` vs
      `arbitrary` is the only pair with NO attention in it. `regular` keeps `k`
      instances evenly spaced in the BAG'S OWN ORDER, and that order carries no
      geometry -- measured over 6 bags x 3 coordinates of `lunas`, the absolute
      correlation between within-bag index and coordinate is median 0.156 and
      max 0.457, where chance alone at n=30 already gives E|r| ~ 0.146. So
      `regular` is a DETERMINISTIC arbitrary sample, not a spatially spread one,
      and that third pair measures draw variance rather than coverage.

      - [ ] **The figure is still open.** CREDA has one per section and this one
            declares none. The standing proposal is the attention weights inside
            one bag, one panel per mechanism, which runs nothing -- or the
            section goes without a figure.
      It runs the FULL two phases, not source only. The aggregation enters both
      -- phase 1 scores bags with it and phase 2 builds its representations from
      it -- so source-only would answer "which classifies source bags best" while
      the claim is about transfer. The two can disagree, and the interesting one
      is the second.

- [ ] **Section 4 -- the correspondence, on one group of bags.** APPROVED
      2026-10-20. The question is narrower than the title used to promise: for a
      declared group of target bags, **which source bags did each one attach
      itself to, with and without the local term, and what class were those
      source bags**. Four emissions, and the section is these and nothing else.

      1. **The gif -- synthetic, and it runs nothing.** The same shape as
         section 2's and for the same reason: replicating the real conditions of
         this term is hard, demonstrating what the term DOES is not. One target
         bag's embedding is walked while the source bags stay fixed, and the
         frame shows `pi_ji` over them and the mixture `mu_j = sum_i pi_ji
         Psi_i^s` the term pulls toward. `pi` is a function of the embeddings,
         the head, `sigma` and `tau_loc` -- so the curve is a property of
         Eqs. (18), (24) and (29) and the already trained model, exact, with no
         seeds and no variance. Its unit is a **barrido**, never a step and
         never an epoch, by the same ruling section 2's gif carries.

         **The operator's refusal, and how it is answered (2026-10-20):** "si no
         hay termino local, no puedes buscar esa relacion". Correct, and the gif
         does not try: it is drawn once, from the arm that HAS the term, and
         there is no `sin local` panel in it. What has two arms is the table,
         and that is sound for a different reason -- see below.

      2. **The table -- the group of bags each one attached itself to.** Per
         target bag in the declared group, in both arms:
         - the **5 source bags of most mass**, each with its REAL class;
         - **how many of those 5 share the target bag's class** (an integer in
           `[0, 5]`);
         - **`pi_{ji|c}` on the exact partner** -- the same `(class, subject)`,
           resolved by `socio_exacto`.

         **There is no "mass to its own class" column, and that is a
         correction of mine, not a cut.** `sum_{i in c} pi_ji = g_{j,c}`
         exactly, because `class_correspondence` normalizes strictly within the
         class. So that column is the classifier's POSTERIOR under another
         name -- the number the accuracy table already reports -- and it carries
         no information about WHICH subject. What carries it is `pi_{ji|c}`, the
         split inside the class among its 16 subjects.

      3. **The accuracy table** of the same two arms, CREDA's five columns.

      4. **Its test**: ONE declared pair, `TGL - TG`, paired over the 30 cells.

      **How the two arms are obtained, which is the thing that makes the table
      legitimate.** ONE `run_cell(familia, norma, semilla, ..., arms=("TG",
      "TGL"))` per (family, seed): phase 1 has no local term -- it is Eq. (21)
      over the source -- so the two arms share it bit for bit, and only phase 2
      differs. Then `masa_de_correspondencia` is called on each arm's final
      weights. **The reader composes `pi` itself**, from the embeddings, the
      head, `sigma` and `tau_loc`; it never asks the arm's spec whether the term
      was on. That is why the arm without the local term has a `pi` to read at
      all. `tau_loc` is the SAME for both, measured: `PHASE_TWO_DIMENSIONS =
      ("tauLocal",)` is searched once on the full arm and `run_cell` takes one
      `hyper` for its whole arm list, so no part of this comparison is confounded
      by a temperature nothing selected. 3 families x 10 seeds = 30 cells.

      **The conditions of every quantity, before any number is read.**

      | quantity | what it is | range | floor of indifference | direction | what bounds it |
      | --- | --- | --- | --- | --- | --- |
      | `pi_{ji\|c}` on the partner | the split WITHIN the class among its 16 subjects | `(0.029, 0.131)` at the searched `tau_loc` -- NOT `(0, 1)` | **0.0625** = 1/16, uniform | up is better | Eq. (29)'s inner softmax AND `tau_loc`; see the ceiling below |
      | of its class, among the 5 of most mass | how many of those 5 share the real class | integer `[0, 5]` | **1.67** = 5*(16/48) | up is better | `g_{j,c}` of Eq. (24), which IS the sum of `pi` over the class |
      | the arm's accuracy | on the target | `[0, 1]` | the `TG` arm's own | up is better | the section's own table |

      **The ceiling is structural, and I gave it wrong the first time.** This
      table first said `pi_{ji|c}` lives in `(0, 1)` open. That is formally
      true and useless, because `tau_loc` tightens it enormously and the
      reading of the whole table depends on it. The inner softmax of Eq. (29)
      scores `kappa^B(B_i^s, B_j^t) / tau_loc`, and `kappa^B` is in `(0, 1]`,
      so every logit within a class lives in `(0, 1/tau_loc]`. With `n_c`
      subjects the partner's share therefore cannot exceed

          ceiling = e^(1/tau_loc) / (e^(1/tau_loc) + n_c - 1)

      no matter how perfectly the kernel resolves that subject, and cannot fall
      below `1 / (1 + (n_c - 1) e^(1/tau_loc))`. Measured on the pilot, with
      `tau_loc` chosen by the phase-2 search and `n_c = 16`:

      | family | `tau_loc` | floor | chance | CEILING | measured at the partner | % of the reachable range |
      | --- | --- | --- | --- | --- | --- | --- |
      | lunas | 1.2296 | 0.0287 | 0.0625 | **0.1307** | 0.0803 | 26.1% |
      | toroide | 1.2521 | 0.0291 | 0.0625 | **0.1291** | 0.0795 | 25.6% |
      | rollo | 1.2521 | 0.0291 | 0.0625 | **0.1291** | 0.0746 | 18.2% |

      So `0.08` against a chance of `0.0625` reads one way against a ceiling of
      `1.0` and a completely different way against a ceiling of `0.13`. The
      table must print the ceiling beside the number, and a run that reports
      this column without it is reporting a fraction of an unstated whole.

      **And it means a null here has TWO candidate causes, not one.** Either
      the embedding does not separate subjects, or `tau_loc` is too large for
      the split to express it. They are different findings and the ceiling is
      what tells them apart: a measured value AT the ceiling is the second.

      **Why up is better, and why it is not circular.** The term minimizes
      `l_loc,j = d2_j/2` with `d2_j = ||Psi_j^t - mu_j||^2` and `mu_j = sum_i
      pi_ji Psi_i^s`. **Nothing in that loss sees the subject id.** No gradient
      rewards concentrating mass on the partner; all the term asks is that the
      target bag approach the mixture `pi` assembles. So partner mass above 1/16
      is a hit on a label the term NEVER HAD, which is what makes it evidence.

      **And what bad looks like, with a precise shape.** At `pi_{ji|c} ~ 0.0625`
      -- uniform -- `mu_j` is the source's CLASS CENTROID, so the term aligned
      the bag to a class average and not to a subject. That is what the global
      term already does per class: the local one became a second, more expensive
      global. There is a THIRD state the table has to be able to show -- mass
      concentrated on the WRONG subject of the right class, high and misplaced --
      which is why the 5 of most mass sit BESIDE the number: five of the class
      with the partner at the floor means the term resolved the class and not
      the subject.

      **The rule -- when the local term can be said to buy something.** Two
      conditions, BOTH required:
      1. **the mass moves**: `TGL - TG` on the partner mass, paired over the 30
         cells, Holm over the three families;
      2. **the accuracy does not lose**: `TGL >= TG` on those same cells.

      And the four crosses are declared HERE, before the run, so no reading is
      improvised after seeing the numbers:

      | mass moves | accuracy moves | what may be said |
      | --- | --- | --- |
      | yes | yes | **the term buys, and buys THROUGH correspondence** -- the only cell that sustains the section |
      | yes | no | the mechanism works and **buys nothing**: correspondence is recovered and does not become accuracy |
      | no | yes | it won, but **not through correspondence**; the section cannot attribute it to the local term |
      | no | no | the term does nothing here |

      **What this supersedes.** The earlier design on this line -- read `pi`,
      DISPLACE the bag toward another class's region, read `pi` again, and ask
      whether the mass followed -- is not this section. It is superseded for the
      reason the operator gave for section 2's ladder: reproducing the real
      conditions of that displacement is very hard, and what the section needs
      is to show the term responding and what it looks for. `ESCALERA_DESPLAZAMIENTO`,
      `desplazar_bolsas` and `barrido_desplazamiento` are superseded here, not
      deleted.

- [ ] **Sections 3 and 4 run CONSECUTIVELY, in that order, and that is not
      cosmetic.** They are MIL's claim split in two halves: "we know which
      INSTANCES matter inside the bag" and "we know which BAGS correspond across
      domains". Neither half holds alone -- a good attention over arbitrary
      correspondences aligns noise carefully.

- [ ] **Section 5 -- the ablation.** APPROVED 2026-10-20. **TWO emissions and
      NO FIGURE**, by the operator's ruling: "este solo dejaria las tablas de
      ablacion que sera de cada termino por separado vs los individuales y un
      test que apoye los resultados, nada mas". The synthetic gif proposed for
      this section -- a bag walked from its class centroid to one source
      subject, reading `||mu_j - centroid_c||` as what the local term asks for
      beyond the global -- is NOT part of it. It goes LAST because it opens the
      fewest new questions and because read after 3 and 4 it interprets better:
      if the local term buys nothing, section 4 has already said whether that is
      because the correspondence fails or IN SPITE OF it working, and those two
      conclusions are different ones no section can separate alone.

      The three arms exist and are declared in `ABLACION` (`toy_run.py:895-920`):
      `TG` global only (Eqs. 32-37), `TL` local only (Eqs. 30-31 and 38), `TGL`
      both coefficients at 1, which is Eq. (39) as declared. Local-only is the
      arm that did not exist before 2026-10-20: `spec["local"]` already gated the
      local term and the symmetric `spec["global"]` gate was added.

      1. **The accuracy table**, CREDA's five columns, FOUR rows: `.predict`,
         `TL`, `TG`, `TGL`. `.predict` costs no second sweep -- it is the same
         run read at its step-0 checkpoint and `run_cell` measures it itself.

      2. **Its test**, in CREDA's N-arm shape and not the paired-t one -- measured,
         not assumed: `tabla_tres_brazos` calls `tabla_n_brazos`
         (`src/CREDA/toy_plots.py:1490-1550`), which is Friedman over treatments x
         seed blocks, post-hoc Wilcoxon on a DECLARED set of pairs, and Holm inside
         that family. Seven columns: `gana si D>0`, `contra`, `D = 1a - 2a`, `IC95
         de D`, `p (Wilcoxon)`, `p Holm (familia de 3)`, `gana`, with the Friedman
         ranks in the footer.

         **The family is the THREE pairs among the terms**, declared before the
         run: `TGL - TG` (what the local adds), `TGL - TL` (what the global adds),
         `TG - TL` (which term alone is stronger). At m=3 the smallest threshold
         is 0.0167, so an effect at p=0.0195 does NOT survive and the table has to
         be able to say that.

         **`.predict` is NOT a treatment and NOT in the Holm family**, and this is
         not an invention: it is the exact mirror of how CREDA treats its oracle
         ceiling -- own marked row carrying its LEVEL, plus a DECLARED PAIR
         REPORTED WITHOUT TEST, with delta, CI95 and wins. The reason is the same
         one `toy_plots` writes for the ceiling: the ladder `.predict -> method`
         **is already section 1**, so testing it here re-tests a hypothesis
         another section owns.

      **Direction and the reading, declared before the run.** Target accuracy in
      `[0, 1]`, floor is `.predict`'s own, up is better; a Friedman rank in
      `[1, 3]` where **lower is better** and 2.0 is the tie. Five results and
      what each one permits:

      | result | what may be said |
      | --- | --- |
      | `TGL > TG` and `TGL > TL` | the two terms are COMPLEMENTARY: each adds something the other does not |
      | `TGL ~ TG > TL` | **the global carries the transfer**; the local adds nothing on top of it, even if it is worth something alone |
      | `TGL ~ TL > TG` | **the local carries the transfer** |
      | `TGL < max(TG, TL)` | the two terms INTERFERE: the method as Eq. (39) declares it is worse than one of its own halves |
      | all four ~ `.predict` | the adaptation buys nothing over doing nothing |

      **One measured lock, reported once and not as a table.** Both gates bite on
      real runs: `TG` gives `contributionLocal = 0.0` EXACT and `TL` gives
      `contributionGlobal = 0.0` EXACT (`wiring.py:1026-1027`). That is the proof
      the arms are what their labels say.

## Approved, section by section -- 2026-10-20

- [x] **Section 1 is approved as it stands.** Measured against CREDA's own
      notebook cell by cell: per dataset ONE animated frame and THREE tables --
      accuracy in the intervention ladder, the paired `con`/`sin` test, and the
      declared pairs with Holm -- and MIL's notebook already emits all four.
      A reading of mine that it was missing the Holm table was FALSE: the dump
      that produced it printed only each code cell's first line, so the second
      `display` was never shown. `git diff` on the notebook is empty.

- [x] **Sections 2, 3 and 4 are approved** -- each one's emissions are written
      into its own bullet above, in CREDA's own order, with the figure described
      and every table rendered with example numbers before it was agreed. What
      each supersedes is named there rather than deleted.

- [x] **Section 5 is approved too, and it is the only section with NO
      FIGURE.** Deferred on 2026-10-20 ("no, nada seccion 5, sigamos") until 4
      was closed, then cut to two emissions the same day: the ablation table and
      one test. The synthetic gif proposed for it was refused -- so the
      expectation that it would read like 4 did NOT hold, and the figure that
      ran nothing is written into its bullet as refused rather than left as an
      open proposal somebody might revive.

**All five sections are approved in shape.** What exists as code is the sweep
machinery below, built and piloted BEFORE these approvals; the figures and
tables each section's bullet now declares are what `### Still owed` lists.

## Built and piloted -- 2026-10-20

The machinery of sections 2 to 5 exists and ran a pilot. The section checkboxes
above stay unticked because they mark a MEASUREMENT, and the pilot is not one:
one seed, eighteen eval bags, so the accuracy grid is 1/18 = 0.0556 wide and
several cells sit at 1.0000. It shows the shape of the data and nothing about
which arm wins.

Pilot cost, three families, one seed, CPU: section 2 **17.8 s**, section 3
**45.0 s**, section 4 **7.9 s**, section 5 **9.5 s**.

### Two decisions taken, each in one constant

- **`ARQUITECTURA_POR_FAMILIA = True`.** The paragraph above leaves "one fixed
  and said out loud, or one per family" open; this runs one per family. Fixing
  one would run two of the three families in an architecture their own section 1
  already measured worse, and then a weak result in those two is unreadable: the
  architecture or the section. The constant is the whole of the decision, so the
  other branch is one line away -- and it REFUSES rather than picking, because
  nobody has named which fixed one it would be.
- **Section 3 runs ONE arm per cell, each with its own phase 1.** The ablation
  shares a phase 1 and that is correct -- phase 1 is Eq. (21) over the source and
  touches neither term the ablation gates. The aggregation is the opposite: it
  enters BOTH phases (phase 1 scores bags through Eq. 19/20, phase 2 builds its
  representations from it), so a shared phase 1 would measure four mechanisms on
  the head a fifth one trained.

### Four defects the wiring exposed, all measured

- **`toy.true_partners` cannot serve this toyset.** It resolves the domain code
  in `toy._FAMILY`, which knows only the digit benchmark's codes (`r/t/n`); this
  toyset's are `FAMILY_CODES`' (`l/L/o/O/w/W`), so the call dies with
  `KeyError: 'l'`. The function is not broken -- it is written for other families.
- **And the ground truth here is stronger than what it offers.**
  `toy_bags.construir` draws ONE band permutation and applies it to BOTH domains,
  so subject `s` of class `c` is the SAME band of the same parameter in source and
  target. The partner is an exact identity, not a ranking: `toy_run.socio_exacto`.
- **The two domains split all three roles IDENTICALLY** -- measured on
  `toroide`/seed 0, `train_idx`, `valid_idx` and `eval_idx` agree element by
  element. So `source-train` and `target-eval` are disjoint BY CONSTRUCTION, which
  is why section 4's first run reported zero judgeable bags, and `source-train` and
  `target-train` are all 48. Section 4 reads `pi` over train/train, which is also
  the role phase 2 actually adapts on. This refutes the premise `true_partners`
  states in its own docstring to justify ranking.
- **The displacement barely displaced.** The destination was the mean of the
  runner-up class's bag centroids, and a mean over a whole class lands near the
  cloud's global centre -- which is where a frontier bag already sits, because that
  is what puts it on the frontier. Measured at `fraccion = 1.0`, distance travelled
  against the target cloud's dispersion: `lunas` 0.61-0.97 of 1.405, `toroide`
  0.51-0.72 of 3.205, `rollo` 1.11-1.50 of 10.189 -- a seventh of the dispersion,
  with `pi` moving 7e-04. A number that reads as "the correspondence does not
  follow the bag" when what happened is that the bag did not move.
  `DESTINO_DEL_DESPLAZAMIENTO` is now `"sujeto-mas-cercano"` and the old rule stays
  declared and runnable. The agreement's own wording decides it: "DISPLACED toward
  another class's region" -- the nearest subject IS in the region; the class mean is
  in the middle of everything. Under the new rule `w` on the moved bags rises
  monotonically (`toroide` 0.4192 -> 0.4438, `rollo` 0.1698 -> 0.1927) while the
  untouched bags stay flat, which is the right direction: a frontier bag is
  ambiguous and moving it into one class's band makes the head MORE decided.

### What the pilot proves about the harness, not about the method

- Section 2's rung `0.0` reproduces its own reference to **0.00e+00 exact** in all
  three families, so the displacement harness contributes nothing on its own.
- Section 4 carries `piMovimientoMaximo`. If it were zero, "the mass did not move"
  -- a result -- and "the displaced material never reached the reading" -- a defect
  -- would be the same two numbers written twice.
- Section 5's two gates bite: `TG` reports `contributionLocal = 0.0` exact and `TL`
  reports `contributionGlobal = 0.0` exact.

### Still owed

- [x] **The two gifs exist and run nothing.** Each section's sweep computes its
      walk -- it needs the trained model to ask it about positions that did not
      exist yet -- and the drawing reads the artefact, which is the split
      `animar_familia` quotes CREDA for: retouching a colour costs a redraw and
      not the sweep. `_pasos_del_barrido` is shared by both, for the defect
      below.
- [x] **Nine table renderers, as TWO parameterized ones plus declarations.**
      The four sections print the same pair of grids and what differs is WHICH
      arms and WHICH pairs, which is a declaration. `_holm`, `_par_medido` and
      `_rangos_medios` are section 1's own; nothing statistical was rewritten.
- [x] The notebook cells for the four sections, in both notebooks.
- [ ] A full-scale run. The pilot cannot separate anything and was never meant to.

### Four defects the build exposed, all measured -- 2026-10-20

- **A CPU 0-dim tensor multiplying an MPS one returns the ORIGINAL without
  failing.** `torch.linspace` with no `device=` is born on CPU, so
  `inicio + t * (final - inicio)` gave exactly `inicio` for all 24 barridos
  with `||final - inicio|| = 23.85`. BOTH constructed walks came out perfectly
  flat -- gap 0.0025, `d_j^2` 0.8131, max `pi` 0.0658 in all 24 -- and a flat
  walk reads as "the term does not respond", which is a RESULT, instead of as
  a defect. Closed by `_pasos_del_barrido`, which both walks go through.
- **`Psi` and `mu` are never materialized.** Eq. (31) expands entirely into
  kernel evaluations precisely because the explicit coordinates are never
  required. A first version of the section-4 walk averaged `(H, w)` to
  fabricate a `Psi` and subtracted vectors: it would have drawn a plausible
  curve of a quantity the formulation does not name. The gap to the class
  centre now comes from the same expansion, with `u_c` the uniform weight
  vector over the class's source bags.
- **`run_cell` had never run an arm with `weighting: False`.**
  `weighting_cost` returns `None` there -- correctly, there is no cost to
  measure -- and the row subscripted it unconditionally. `TypeError` on
  section 2's first cell. The three fields are now `nan` with a note saying
  why, never a zero: a zero would say "the weighting cost nothing", which is a
  measurement.
- **`wiring.select`'s `arbitrary` rule could not run on MPS.** Its deterministic
  seed casts the bag's own contents to float64, which MPS does not support.
  `.cpu()` before `.double()` gives the same value a CPU run produced, so no
  measured draw moves.
- **Section 3's two axes collided.** `_brazos_de_mecanismo` filtered on
  `selection is None`, and the second stage's reference arm `S-<winner>-todas`
  carries that too, so the mechanism table ate a row from the other stage and
  printed it with the selection label. Each row now writes its own `eje`.

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
- [ ] Which destination rule the displacement runs under. `"sujeto-mas-cercano"`
      is what runs and the measurement above is why, but BOTH rules are declared
      and either is one constant away. It is a decision about what sections 2 and
      4 perturb, so it is named here rather than left inside the code.

## Ladder

- [x] `.predict` -- the same run read at its step-0 checkpoint. No adaptation.
- [x] `AdaBN` -- that checkpoint with the target's normalization statistics
      re-estimated. No optimizer, no label.
- [x] `MIL-CREDA` -- the full method, global and local, weighting on.

The first two cost no second sweep: `run_cell` measures them itself. A rung that
reads `y_t` is a CEILING and never a competing method, and a table that prints
one says so.
