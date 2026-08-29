# Agreed for the benchmark report — bound to research-concept-r17.md

Every item here was settled in deliberation, not derived from a plan of work. An
item that never reaches the code shows up as an unticked box rather than as a
silence. Nothing here is re-decided while implementing: a collision with any of
these is raised, not resolved in passing.

## Dispersion and scale

- [x] The `±` in every phase-1 table is the dispersion **across seeds**, not the
      batch-wise dispersion the reference paper reports. The seed dispersion is
      what the verdict rule consumes; a batch-wise `±` would stay plausible while
      the run grants no verdicts.
- [x] `RAMP_CEILING` stays at **1.0** as the neutral each family's searched
      ceiling is read against. The realized share of the objective is reported
      too, per arm, so a term scaled to irrelevance is visible rather than
      inferred. Renamed from LAMBDA_CONST when the coefficient became one
      object with two knobs; superseded as *the* value by the per-family search,
      and the Reversed section below says how.
- [x] Accuracy granularity is **1/36 = 2.78 points** with 36 evaluation bags, so
      tables print one decimal and say so. Two decimals would be false precision.

## Nomenclature — asterisks count what is absent

| id | display name | what it lacks |
| --- | --- | --- |
| `A` | `Baseline` | everything |
| `C` | `CREDA*` | the confidence weighting |
| `D` | `CREDA` | nothing |
| `B` | `MIL-Baseline` | everything |
| `E` | `MIL-CREDA**` | the local term and the weighting |
| `F` | `MIL-CREDA*` | the local term |
| `G` | `MIL-CREDA` | nothing |
| `SU` | `MIL-CREDA-U` | 20 of its 30 instances, regular selection |
| `SA` | `MIL-CREDA-A` | 20 of its 30 instances, arbitrary fixed selection |
| `SK` | `MIL-CREDA-K` | 20 of its 30 instances, top-K by learned attention |

- [x] `SU`, `SA`, `SK` **select 10 instances** and then apply the same learned
      attention over the selected ones, so the trio differs in one thing: the
      selection rule.
- [x] `SA`'s draw uses a **dedicated generator**. Consuming the training
      generator would shift every later draw and the rung would credit the
      selection with what the offset did.

## Ladder

- [x] The three attention rungs are replaced by `SU->SK`, `SA->SK` and `SK->G`.
      The first two hold the budget at 10 instances; the third reads what
      dropping from 30 to the top 10 costs.

## Tables (phase 1) — Table 6 of the reference, our names

- [x] Rows are arms by display name, columns are the six transfers, plus `Avg`.
- [x] **Two tables**: target accuracy as the headline, source accuracy as the
      complement. A method that wins on target by wrecking source is the
      degenerate case, and one table cannot distinguish it from a success.
- [x] A `max` column beside `mean ± stdev`, so the peak is documented where a
      peak is a number rather than a picture.
- [x] A `share` column: the realized contribution of the adaptation term.
- [x] The ladder table stays, directly below. Levels say who is ahead; only the
      rungs say which piece did the work.
- [x] Below the declared repetition floor, no verdicts are granted, the reason is
      stamped in the header, and the table is printed anyway.
- [x] **The wall-time table comes first**, in the same shape, times instead of
      accuracy. It replaces the per-run listing, which repeated in prose what the
      tables say in a grid.
- [x] **Reading order**: time, source, source rungs, target, target rungs, and the
      transversal reading last.
- [x] **One table per cell**, each preceded by three short lines — what is
      measured, why, and which direction is better — and followed by the most
      relevant conclusion.
- [x] That conclusion is **computed from the table**, never written by hand: a
      hand-written conclusion is a second source of truth and goes stale in
      silence.
- [x] A rung is named `Baseline → MIL-Baseline`, never `A->B`. An identifier is
      not a name.
- [x] Notebook prose and table headings in Spanish; identifiers, JSON keys and
      method names stay English, because they are a data contract rather than
      prose.
- [x] Progress during the run prints one line per transfer, not one per run. It is
      the only sign of life in a run measured in hours, so it is kept — but it is
      progress, not a report.

## Figures — phase 2

- [x] **One display seed for every comparative grid**, chosen as the seed whose
      across-arm mean target accuracy is the median. Panels drawn from each arm's
      own median seed would differ in the method *and* in the draw, and — because
      the seed fixes the partition — would not even share their bags.
- [x] Artefacts are **median, never best**. The best of N grows with the arm's
      dispersion, so best-vs-best flatters the noisiest arm by the most.
- [x] **Three transfers in every figure, not six.** Six rows at a legible panel
      size do not fit on a page and the tables already carry all six. Fixed in
      `FIGURE_TRANSFER_COUNT` and `FIGURE_TRANSFER_RULE` by a rule that looks at
      no outcome: each domain appears once as source and once as target.
- [x] **Which three is computed, not fixed**: the transfers where the methods
      reach the highest mean target accuracy. It is a choice made by the outcome,
      so it is declared in every caption. What makes it defensible here is that
      the latent space of a transfer where everything sits near chance is a
      picture of a model that did not learn. It never touches *which draw* is
      shown — that stays the display seed.
- [x] Latent grid is the **shared original space** and then one column per method:
      both floors, both CREDA, all three MIL-CREDA. `Original` is the images
      themselves before any model — shared because preprocessing already brings
      both domains to one tensor shape — sampled representatively and stratified
      by class.
- [x] **Both floors stay, because the measurement says they are not redundant.**
      Drawn at the instance level on the pilot, `Baseline` and `MIL-Baseline`
      differ by up to 0.38 in distance ratio and 0.07 in domain separability.
      They train the same encoder through different objectives — per instance
      against per bag through the attention pooling — so their instance
      embeddings had no reason to agree. `latent.floors_agree` re-runs that check
      every campaign, and only it may retire one of the columns.
- [x] **Every panel of the grid is drawn at the instance level**, bag-unit arms
      included. Every arm encodes instances, so it is a space they all have and
      the only one where every panel carries the same number of points. One point
      per subject beside one point per instance made the instance-unit columns
      look like they covered the space and the bag-unit ones look sparse — the
      statistical unit drawn, not the alignment. The bag-level view stays in the
      phase-two tables, which measure each arm in its own unit.
- [x] Colour is the **class**; the marker is the domain — source circles, target
      triangles. Target markers larger with a dark edge, source smaller and
      semi-transparent, categorical 10-class palette.
- [x] Bag figure is **3 columns × 3 rows in one figure**, not one file per
      transfer: `MIL-Baseline` + `MIL-CREDA*` + `MIL-CREDA`, chosen by the
      mechanism rather than by the ranking — that is the rung the local
      correspondence lives on, so the figure can fail.
- [x] Bag figure highlights **the same bags in every panel**: the median bag of
      each class by correspondence mass, one colour each, everything else grey.
- [x] The nearest source bag is found with the **bag kernel in the representation
      space**, never Euclidean and never in the 2-D projection, and the pair is
      joined by a line so the correspondence survives UMAP's distortion.
- [x] The measured correspondence hit rate is printed with the figure, so it
      carries a number and not an impression.
- [x] **Every figure carries the same three lines as a table**: what is looked at,
      what is being sought, and the conclusion — and the conclusion is **computed
      from the results**. A sentence written by hand under a figure fixes itself:
      the figure is regenerated from other data and the sentence stays.
- [x] The bag figure keeps the projection rather than becoming a bipartite
      diagram. Three columns instead of six separate files already makes the
      panels large enough to read; if it is still a knot, the bipartite is next.

## Removed on purpose

- [x] The adaptation-term scale table (`min`/`max`/`width`). What it checked —
      that MIL-CREDA's terms stay in `[0, 1]` while the prior work's is bounded
      only by `ln n` — is guaranteed by the construction in Section 5 and already
      covered by the invariant suite. A table restating a proved bound is a third
      place for it to go stale.

## Figures — phase 1

- [x] Loss curves for `Baseline`, `CREDA`, `MIL-Baseline`, `MIL-CREDA`, as the
      median curve with an interquartile band across seeds.
- [x] A **contribution panel** beside them. The pilot already shows the realized
      share ranging from 0.03 to 0.99 across arms at a fixed coefficient, which
      is the thing the curves exist to make visible.

## Checkpoints

- [x] Three per arm per cell, for every arm, so the top 3 can be selected after
      the run instead of being guessed before it. Local, gitignored.

<!-- What follows was settled in the session of 16 August, against r17. It lived
in a separate `AGREEMENTS.md`, created by not having looked whether this file
already existed: they are two halves of one contract and they live in one. -->

## The objective and its coefficient

- [x] The supervised term of Eq. (18) is divided by its own supremum `B_src`, so the three terms of Eq. (39) live in [0, 1) and are read on a common scale.
- [x] The bag-unit arms call `source_loss` and `total_objective`. No supervised term written inline in the benchmark.
- [x] Each family searches **its own** ceiling and passes it to its derivations. A shared ceiling equalizes the coefficient and unequalizes the balance: the two objectives sit a factor of `B_src` apart, so one number puts adaptation at 85% of one and 10% of the other.
- [x] One search per family, on the complete method — D and G — and inherited. If every arm searched its own, B→E would differ in two things and no rung would be attributable. Declared consequence: E and F carry no local term, so the ceiling found on the complete method is not necessarily theirs.
- [x] The 1.0 stops being the value and becomes **the neutral the searched value is read against**. If MIL-CREDA's ceiling lands there, the normalization argument is confirmed by measurement rather than by reasoning.
- [x] `RAMP_CEILING`'s comment states that argument, not r16's scale one. The old 1e-4 measurement stays as historical record, marked as taken against the un-normalized objective.

## Prior work

- [x] CREDA is used as it is: per-instance cross-entropy, its own single-term objective, never edited to make the comparison work.
- [x] The 1e-4 leaves where it was and becomes the ceiling of CREDA's own ramp, with `creda_lambda_special` as the default. `train_creda` still reads it from the notebook's `cfg`, because it is not always 1e-4.
- [x] `get_lambda` is untouched. DANN, ADDA and CDAN+E still read it at full strength: lowering their coefficient would switch CDAN+E's domain adversary off rather than attenuate it.
- [x] The notebooks under `CREDA/Notebooks/` remain functional, verified statically without executing them.
- [x] The cost of moving the coefficient is measured and bounded, not denied: the product's reassociation stays within 2 ULP and the gradients come out bit-identical.

## The two ramps

- [x] Each method names its own ramp and carries its own default: `creda_ramp` at 1e-4, `milcreda_ramp` at 1.0. The defaults serve each method's own runs; the benchmark never uses them, it always passes explicit values.
- [x] A floor with no adaptation term gets a coefficient of zero, not the curve. It has no coefficient, and handing it one would suggest a term it does not carry.
- [x] The experiment hands both the same `delta` and the same `ceiling`, explicitly.
- [x] The curve is written once. `milcreda_ramp` binds to `creda_ramp` rather than copying it, and a test pins them to the same numbers.

## The record and the report

- [x] The benchmark declares revision r17 and which sections each arm exercises.
- [x] The record keeps the supervised term's magnitude and the ratio between terms, not only the contribution. Without a denominator, "the term commanded nothing" and "the term was scaled to nothing" print alike.
- [x] The benchmark declares `components` in its report contract.
- [ ] The report says which ceiling each family found, on which role, over how many repetitions, and whether the seeds agreed. That CREDA does not run at its published 1e-4 is a consequence of the search and has to be read there, not in a comment in `config.py`.
- [ ] The report marks the three cross-family rungs — C→E, D→F, D→G — with whatever remains of the confound **after** the search. With per-family ceilings the balance is partly equalized, so the magnitude has to be measured again rather than repeating the `B_src` factor that came from the shared ceiling. The within-family rungs stay attributable.
- [x] On the two transfers the search measured, the ceiling in force is the one that won **on that transfer**, by the same paired rule and the same tie-break. On the remaining four it is the one that won pooled across the two searched transfers. The report states that rule where the ceilings are shown, because the scalar of those four was not chosen by looking at them.
- [x] The record carries the per-transfer picks beside the pooled one. A record holding only the pooled winner cannot express the rule above, and a run reading it would silently apply the pooled value everywhere.
- [x] `identicalAcrossShards` names the ceilings. They are the parameter the search just changed, so two shards straddling the search would merge into one table with adaptation inert on one half and not on the other, and nothing would refuse.
- [x] MIL-CREDA no longer runs at one coefficient across transfers, so its row averaged over transfers mixes two scalars. Within a transfer every arm still shares the ceiling, which is what keeps each rung attributable; the report says which of the two readings it is giving.


## The ceiling search

- [x] Three material roles, disjoint: 64 training / 20 selection / 36 evaluation. The search reads selection and the verdict reads evaluation, and `run_one` measures one **or** the other — a role the search cannot see is a stronger guarantee than one it agrees not to use.
- [x] The selection role is funded with **new material**, not taken from training: 12 bags per class instead of 10. Evaluation keeps its 36 and the three-point resolution the campaign was sized for. USPS is what binds, at 18 bags per class.
- [x] The search runs at 20 epochs and 3 seeds, and **never at pilot scale**. The ramp climbs on the fraction of training elapsed, so at three epochs it is saturated by the second and a ceiling found there describes a landscape the campaign never trains in.
- [x] Its required scale is declared apart from the one it runs at, and `atRequiredScale` goes into the record. With only one of them, a cheaply-found ceiling and the configuration agree with each other.
- [x] The campaign **refuses** without ceilings, and with ceilings searched below scale. Funding your own coefficient out of the run you are about to report is choosing and judging in one pass.
- [x] Ceilings are compared **paired** by cell (seed, transfer): every ceiling is measured on the same material, so the cell's difficulty cancels instead of drowning the effect.
- [x] Whether each seed would have chosen the same on its own is recorded. Three seeds on three different ceilings and three on the same one produce the same winner and are not the same evidence.
- [x] The tie rule is written down: the **smallest** ceiling among the tied wins. Below some point a term is inert and everything ties, so there the tie-break is what actually chooses.
- [x] The grid runs between the two declared defaults, 1e-4 and 1.0, so nothing outside what was already defensible can come out.
- [x] The **growth rate is not searched**: it stays at `RAMP_DELTA = 20`, CREDA's own, shared by both sides. Searching it too would have turned the run into a 2D grid of three to five hours.
- [x] No adaptive sampling. With 20 points the exhaustive grid is affordable and gives strictly more evidence: pruning biases against slow deltas, and sparse sampling destroys the tie structure, which here is a finding.
- [x] The verdict is read over **all six** transfers. Withholding the two that funded the search bought nothing with the roles already disjoint by bag, and cost a third of the units the paired reading rests on.

## The full run

- [ ] The full grid — 30 seeds, 20 epochs — is not launched without an explicit authorization. Neither a clean verification nor a green pilot is permission.
- [ ] While the run stands at pilot scale, its numbers are not quoted as results: not in the report, not in the summary, not in conversation.


## The trials search

Agreed 2026-08-26/27, while replacing the grid engine. Every item here is
carried by code and by a test that dies when the code is mutated, except the one
marked open, which is open because the run has not happened.

- [x] The search measures **every one of the six transfers**. Nothing inherits: the pooled fallback in `ceiling_for` is no longer reachable. The grid measured two and four inherited out of sample, and both of `MIL-CREDA`'s significant losses fell on inherited transfers.
- [x] Searching transfers the verdict also judges leaks nothing. What keeps the material disjoint is the **role** — the search reads `valid`, the verdict reads `eval` — and that holds identically on all six. `search_ceilings`' docstring claimed the transfer split did that work; it was false as configured and is corrected.
- [x] Only the **full arms** are searched, `D` and `G`. The ceiling is per family and per transfer, **never per arm**: `ceiling_for(reduction, family, transfer)` takes no arm, so a per-arm ceiling is not expressible. If it were, the term and the coefficient could not be told apart.
- [x] The **growth rate is still not searched**, and the reason is new: the two families' objectives already sit an order of magnitude apart in `adaptationShare`, so a second free dimension amplifies that imbalance rather than resolving it. Ceiling and growth rate are also confounded — a high ceiling reached slowly and a low one reached fast give similar trajectories.
- [x] **One seed per trial, declared.** Two trials on different seeds would measure the ceiling and the draw at once. The material is drawn once per family and every trial of a transfer runs on it.
- [x] **A sampler seed per study**, derived from `(family, transfer)` with CRC32. Sharing one seed made all twelve studies visit the same four ceilings, and with a wide plateau the winner is the smallest *visited* point — so the record showed an agreement across transfers that was an artifact of the seed. Found by the local pilot, not by reading.
- [x] **The plateau is the instrument's resolution**, `1/VALID_BAGS`, and not the GP's own noise estimate. Two ceilings that differ by less than one bag are not distinguishable by the measurement, whatever the model thinks; using a fitted quantity would make the plateau's width depend on how well the model fitted.
- [x] **Within the plateau the smallest ceiling wins** — the same rule the tie-break had, and the same reason: the same outcome for less adaptation is the weaker claim.
- [x] The **grid engine stays reachable** by configuration. It wrote the record that governs the campaign of 1800 runs, and an engine that can no longer be run is a record that can no longer be reproduced.
- [x] The **pilot search writes its own record** and the full record always outranks it. With one file, a rehearsal's answer would have been consumed by a full campaign without a word.
- [ ] **The trials search has not run at full scale.** The ceilings governing the campaign come from the grid search, so `probe` reports `search-first` and it is right: the current declaration has no record yet. No new campaign is launched from this declaration until it does.

## Reading the campaign

- [x] The gains table is **paired within each transfer**, against each method's own floor, and reports four things because no single one of them is honest alone: the mean in points with its **between-transfer** error, the mean of the percentages, the span, and the agreement in words. The two means can disagree in sign on the same data — `+0.56` and `-3.61` — because the floors run from 23% to 81%.
- [x] The error of that mean is **between transfers, not over the pooled pairs**. A transfer is a setting and not a repetition; pooling claimed a stability across settings nobody measured, and understated the uncertainty by a factor of three.
- [x] **Phase two rests on 3 repetitions, not 30.** `CHECKPOINTS` keeps three per cell, so every latent claim sits on the verdict floor by construction, however long phase one ran.
- [x] Promotion never moves an arm's own median. Seeds added to keep a floor comparison possible, or to let a figure draw, are kept in separate fields and **never enter a marginal average**: a floor's extras were chosen by the dependent arms' orderings, so they are a biased sample of that floor.

## Reversed

What was agreed and later changed, and what replaced it. Written rather than deleted: an agreement that was turned over is part of the record, and removing it would lose exactly what this file exists to keep. No bullets, because the parser counts items and this is not one.

**"The ceiling is fixed at 1.0 and not chosen by looking at outcomes."** Agreed when the ceiling was shared and its value came from the neutral argument. Reversed on deciding that each method searches its own: a shared ceiling equalized the coefficient and unequalized the balance. The neutral argument did not die, it changed role — it is now the reference the searched value is read against.

**"The sweep is a sensitivity reading and not a selection, because there is no validation role."** Both halves changed. It is a selection now, and there is a validation role: it was carved, funded with new material so evaluation lost nothing.

**"The report says CREDA runs at the shared ceiling and not at its published 1e-4."** There is no shared ceiling any more. What the report has to say now is which ceiling each family found, on which role, over how many repetitions, and whether the seeds agreed.

**"No adaptive sampling. With 20 points the exhaustive grid is affordable and gives strictly more evidence: pruning biases against slow deltas, and sparse sampling destroys the tie structure, which here is a finding."** Reversed on 2026-08-27 for Optuna with `GPSampler`, 30 trials per `(family, transfer)` over a continuous range. Two halves of the old reason, and they did not fare the same. *Pruning biases against slow deltas* does not apply: nothing is pruned, every trial runs to its full epoch count. *Sparse sampling destroys the tie structure* was right, and the structure is not recovered — over a continuous range two evaluations never land on the same ceiling, so `tied` and `seedsAgree` cannot exist. What replaces them is the **plateau**: how many visited ceilings the criterion cannot tell apart. It carries the same finding — a wide plateau means the rule chose and not the criterion, which is what exposed CREDA's ceiling — but it is a different measurement and not a stronger one. The claim that the grid gives *strictly more evidence* was not tested before reversing it, and this file should say so rather than imply the trade was proven.

This reversal was made **without reading this file**, which is the failure the file exists to prevent. It is recorded here rather than quietly folded into the section above.

<!-- position revision=research-concept-r17.md sha256=92519350fabdfedc134b6a683a8855fff4a65443a792502c2d851a1590219280 derivedAt=2026-08-28T15:25:50Z session=correccion target=pilot -->
- [ ] 1. The invariants hold against the tree as it stands: the suite is green and the verification notebook ran against this exact source. Two-state on purpose -- it runs here and nowhere else, and giving it a rung would be the position asserting a state it does not have. Nothing below is worth reading until this is ticked: every later step measures something, and a broken tree makes every measurement a description of the break. `@notebook Notebooks/verification.ipynb`
- [ ] 2. A ceiling record exists at the scale the search declares for itself. Below that scale the search is a rehearsal: the ramp saturates by the second epoch, every ceiling is reached almost at once, and no value it chose is quotable. This is two-state because the record either meets its own declared scale or it does not. `@record`
- [ ] 3. The search's own report is current against this source, so what it says about how each ceiling was chosen -- by a difference in the criterion, or by the rule inside a plateau -- describes the code that is here. Its rung is the rung of the record behind it: a report is only ever as trustworthy about scale as the record it read. `@notebook:level Notebooks/Benchmark_Search_v1.ipynb`
- [ ] 4. The campaign ran and left evidence, at whichever rung it reached. Read this one knowing two things it cannot see: a campaign run here at pilot scale leaves no shard at all, so this reads as the floor even when a pilot did run; and an arrived shard says nothing about which search governed it, so a shard from a retired engine still reads as the top rung. Both are gaps in the witness, not in the campaign. `@shard:level s00`
- [ ] 5. The report is current against this source and reads the merged record rather than a rehearsal's leftovers. This is the step the whole position exists for: a report executed against a pilot record once printed sixty runs beside an eighteen-hundred-run record with every other check green, and the rung is what makes that visible without anyone having to remember to doubt it. `@notebook:level Notebooks/Benchmark_Report_v1.ipynb`
- [ ] 6. The latent analysis is current and measured the checkpoints the campaign promoted, not an earlier run's. Its own claims rest on three repetitions per cell whatever the campaign ran, because only three are kept -- so its rung describes the record it read, never the power of what it says. `@notebook:level Notebooks/Benchmark_Latent_v1.ipynb`
<!-- /position -->





